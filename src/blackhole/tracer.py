"""Kernel de ray tracing (Numba, paralelo).

Cada pixel lança um raio da câmera *para trás no tempo* (integramos o fóton
real com dλ < 0). Ao longo do caminho:
  * se cruzar o plano equatorial entre r_ISCO e r_out, soma a emissão do disco
    (o disco é quase opaco; a fração que passa revela imagens de ordem maior);
  * se cair no horizonte, fica preto;
  * se escapar, amostra o céu na direção assintótica (θ, φ) -> lente gravitacional.
"""
import math

import numpy as np
from numba import njit, prange

from .geodesic import (rhs, rk4_step, choose_step, fix_pole, horizon_radius,
                       CAPTURED, ESCAPED, MAXSTEPS)
from .sky import lut_color

# Índices do vetor de parâmetros P
P_A, P_RH, P_RIN, P_ROUT, P_TIN, P_ALPHA, P_TCAM, P_STEP, P_MAXSTEPS, P_RESC, \
    P_DISKB, P_SKYB, P_TEXS, P_MAXCROSS, P_TREF, P_HMAX = range(16)
NPARAM = 16

# Índices do vetor da câmera C
C_R, C_TH, C_PH, C_TANX, C_TANY, C_ALPHA, C_OMEGA, C_VARPI, C_SIG, C_DLT = range(10)


def camera_frame(r, th, ph, a, hfov_deg, width, height):
    """Quantidades do referencial ZAMO (observador de momento angular zero)."""
    s, c = math.sin(th), math.cos(th)
    sig = r * r + a * a * c * c
    dlt = r * r - 2 * r + a * a
    A = (r * r + a * a) ** 2 - a * a * dlt * s * s
    omega = 2 * a * r / A
    alpha = math.sqrt(sig * dlt / A)
    varpi = math.sqrt(A / sig) * s
    tanx = math.tan(math.radians(hfov_deg) / 2)
    tany = tanx * height / width
    return np.array([r, th, ph, tanx, tany, alpha, omega, varpi, sig, dlt], dtype=np.float64)


@njit(cache=True, fastmath=True)
def emitter_ut(r, a):
    """u^t de uma órbita circular kepleriana prógrada no equador (Kerr)."""
    r12 = math.sqrt(r)
    r32 = r * r12
    den = r32 - 3.0 * r12 + 2.0 * a
    if den <= 0.0:
        return -1.0
    return (r32 + a) / (r ** 0.75 * math.sqrt(den))


@njit(cache=True, fastmath=True)
def disk_texture(r, phi0, tex_k, strength):
    """Textura procedural (faixas espirais/anéis) no referencial do gás."""
    lr = math.log(r)
    v = 0.0
    for k in range(tex_k.shape[0]):
        v += tex_k[k, 0] * math.cos(tex_k[k, 1] * phi0 + tex_k[k, 2] * lr + tex_k[k, 3])
    v = 1.0 + strength * v
    return v if v > 0.05 else 0.05


@njit(cache=True, fastmath=True)
def sample_sky(sky, th, ph):
    H = sky.shape[0]
    W = sky.shape[1]
    tw = 2.0 * math.pi
    ph = ph - tw * math.floor(ph / tw)
    y = th / math.pi * H - 0.5
    x = ph / tw * W - 0.5
    if y < 0.0:
        y = 0.0
    if y > H - 1.001:
        y = H - 1.001
    j = int(y)
    fy = y - j
    i0 = int(math.floor(x))
    fx = x - i0
    i1 = (i0 + 1) % W
    i0 = i0 % W
    r = (sky[j, i0, 0] * (1 - fx) + sky[j, i1, 0] * fx) * (1 - fy) + (sky[j + 1, i0, 0] * (1 - fx) + sky[j + 1, i1, 0] * fx) * fy
    g = (sky[j, i0, 1] * (1 - fx) + sky[j, i1, 1] * fx) * (1 - fy) + (sky[j + 1, i0, 1] * (1 - fx) + sky[j + 1, i1, 1] * fx) * fy
    b = (sky[j, i0, 2] * (1 - fx) + sky[j, i1, 2] * fx) * (1 - fy) + (sky[j + 1, i0, 2] * (1 - fx) + sky[j + 1, i1, 2] * fx) * fy
    return r, g, b


@njit(cache=True, fastmath=True)
def trace_pixel(sx, sy, C, P, sky, lut, tex_k, out):
    """Traça um raio para a coordenada de tela (sx, sy) ∈ [-1,1]². Retorna código."""
    a = P[P_A]
    rh = P[P_RH]
    r_in = P[P_RIN]
    r_out = P[P_ROUT]
    # direção na base local (r, θ, φ) do ZAMO: frente = -e_r, cima = -e_θ, direita = +e_φ
    nx = sx * C[C_TANX]
    ny = sy * C[C_TANY]
    inv = 1.0 / math.sqrt(1.0 + nx * nx + ny * ny)
    n_r = -inv
    n_th = -ny * inv
    n_ph = nx * inv
    # o fóton real viaja na direção -n
    m_r = -n_r; m_th = -n_th; m_ph = -n_ph
    alpha_c = C[C_ALPHA]; omega_c = C[C_OMEGA]; varpi = C[C_VARPI]
    L = varpi * m_ph / (alpha_c + varpi * omega_c * m_ph)
    E_loc = (1.0 - omega_c * L) / alpha_c
    y = np.empty(6)
    y[0] = 0.0
    y[1] = C[C_R]
    y[2] = C[C_TH]
    y[3] = C[C_PH]
    y[4] = math.sqrt(C[C_SIG] / C[C_DLT]) * E_loc * m_r
    y[5] = math.sqrt(C[C_SIG]) * E_loc * m_th
    k1 = np.empty(6); k2 = np.empty(6); k3 = np.empty(6); k4 = np.empty(6)
    tmp = np.empty(6); yn = np.empty(6)
    rgb = np.empty(3)
    cr = 0.0; cg = 0.0; cb = 0.0
    trans = 1.0
    ncross = 0
    max_steps = int(P[P_MAXSTEPS])
    max_cross = int(P[P_MAXCROSS])
    alpha_d = P[P_ALPHA]
    state = MAXSTEPS
    half_pi = 0.5 * math.pi
    for n in range(max_steps):
        rhs(y, L, a, k1)
        h = choose_step(y, k1, rh, P[P_STEP], P[P_HMAX])
        # passo afim negativo = integração para o passado
        rk4_step(y, L, a, -h, k1, k2, k3, k4, tmp, yn)
        d0 = y[2] - half_pi
        d1 = yn[2] - half_pi
        if d0 * d1 < 0.0 and ncross < max_cross:
            f = d0 / (d0 - d1)
            rc = y[1] + f * (yn[1] - y[1])
            if rc >= r_in and rc <= r_out:
                tc = y[0] + f * (yn[0] - y[0])
                phc = y[3] + f * (yn[3] - y[3])
                ut = emitter_ut(rc, a)
                if ut > 0.0:
                    Om = 1.0 / (rc ** 1.5 + a)
                    # g = ν_obs/ν_emit: gravitacional + Doppler + aberração, completo
                    g = E_loc / (ut * (1.0 - Om * L))
                    T = P[P_TIN] * (r_in / rc) ** 0.75
                    # bordas suaves (ISCO e borda externa)
                    edge = min(1.0, (rc - r_in) / 0.6) * min(1.0, (r_out - rc) / (0.25 * r_out))
                    phi0 = phc - Om * (P[P_TCAM] + tc)  # ângulo do gás no instante t=0
                    texv = disk_texture(rc, phi0, tex_k, P[P_TEXS])
                    Tobs = g * T
                    lut_color(lut, Tobs, rgb)
                    # I_obs ∝ g⁴ I_emit (bolométrico): (gT)^4 = g⁴ T⁴
                    I = P[P_DISKB] * (Tobs / P[P_TREF]) ** 4 * texv * edge
                    w = trans * alpha_d * edge
                    cr += w * I * rgb[0]
                    cg += w * I * rgb[1]
                    cb += w * I * rgb[2]
                    trans *= 1.0 - alpha_d * edge
                    ncross += 1
                    if trans < 1e-3:
                        state = CAPTURED
                        break
        for i in range(6):
            y[i] = yn[i]
        fix_pole(y)
        if y[1] < rh * 1.001 + 1e-3:
            state = CAPTURED
            break
        if y[1] > P[P_RESC]:
            sr, sg, sb = sample_sky(sky, y[2], y[3])
            cr += trans * P[P_SKYB] * sr
            cg += trans * P[P_SKYB] * sg
            cb += trans * P[P_SKYB] * sb
            state = ESCAPED
            break
    out[0] = cr; out[1] = cg; out[2] = cb
    return state + 4 * ncross


@njit(parallel=True, cache=True, fastmath=True)
def render_pixels(px, py, k, width, height, C, P, sky, lut, tex_k, out_rgb, out_code):
    """Renderiza os pixels (px[i], py[i]) com k×k subamostras estratificadas."""
    n = px.shape[0]
    for i in prange(n):
        acc = np.zeros(3)
        rgb = np.empty(3)
        code = 0
        for sj in range(k):
            for si in range(k):
                fx = (px[i] + (si + 0.5) / k) / width
                fy = (py[i] + (sj + 0.5) / k) / height
                c = trace_pixel(2.0 * fx - 1.0, 1.0 - 2.0 * fy, C, P, sky, lut, tex_k, rgb)
                if sj == k // 2 and si == k // 2:
                    code = c
                acc[0] += rgb[0]; acc[1] += rgb[1]; acc[2] += rgb[2]
        inv = 1.0 / (k * k)
        out_rgb[i, 0] = acc[0] * inv
        out_rgb[i, 1] = acc[1] * inv
        out_rgb[i, 2] = acc[2] * inv
        out_code[i] = code
