"""Geodésicas nulas em Kerr (Boyer-Lindquist), com Schwarzschild como caso a = 0.

Unidades geométricas: G = c = M = 1.

Escolha numérica
----------------
Usamos a formulação Hamiltoniana H = ½ g^{μν} p_μ p_ν = 0 com as constantes de
movimento E = -p_t = 1 (energia) e L = p_φ (momento angular). O estado do raio é
y = (t, r, θ, φ, p_r, p_θ). Em Kerr o Hamiltoniano separa:

    H = [Δ p_r² + p_θ² − R²/Δ + (L − a sin²θ)²/sin²θ] / (2Σ),
    R = r² + a² − aL,  Σ = r² + a² cos²θ,  Δ = r² − 2r + a².

Integramos as equações de Hamilton com RK4 de passo adaptativo: o passo afim
dλ é escolhido para que a variação relativa de r (medida em relação à distância
ao horizonte) e as variações angulares por passo fiquem ~`step_frac`. Isso dá
passos grandes longe do buraco e passos pequenos perto do horizonte e da esfera
de fótons. Há um número máximo de passos por raio e um passo máximo, então
nenhum raio entra em loop infinito.

Para a = 0 isso é exatamente Schwarzschild (validado contra a equação orbital
u'' + u = 3u² em tests/).
"""
import math

import numpy as np
from numba import njit

# Estados finais de um raio
CAPTURED = 0
ESCAPED = 1
MAXSTEPS = 2

SIN_MIN = 1e-9


@njit(cache=True, fastmath=True)
def horizon_radius(a):
    return 1.0 + math.sqrt(max(0.0, 1.0 - a * a))


@njit(cache=True, fastmath=True)
def isco_radius(a):
    """ISCO prógrada (Bardeen, Press & Teukolsky 1972). a=0 -> 6."""
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    s = 1.0 if a >= 0 else -1.0
    return 3.0 + z2 - s * math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


@njit(cache=True, fastmath=True)
def rhs(y, L, a, out):
    """dy/dλ para o fóton (E = 1). y = (t, r, θ, φ, p_r, p_θ)."""
    r = y[1]
    th = y[2]
    pr = y[4]
    pth = y[5]
    s = math.sin(th)
    c = math.cos(th)
    if abs(s) < SIN_MIN:
        s = SIN_MIN if s >= 0 else -SIN_MIN
    s2 = s * s
    a2 = a * a
    r2 = r * r
    sig = r2 + a2 * c * c
    dlt = r2 - 2.0 * r + a2
    R = r2 + a2 - a * L
    A = L - a * s2
    out[0] = (R * (r2 + a2) / dlt + a * A) / sig            # dt/dλ
    out[1] = dlt * pr / sig                                  # dr/dλ
    out[2] = pth / sig                                       # dθ/dλ
    out[3] = (a * R / dlt + A / s2) / sig                    # dφ/dλ
    dW = (4.0 * r * R * dlt - R * R * (2.0 * r - 2.0)) / (dlt * dlt)
    out[4] = -((2.0 * r - 2.0) * pr * pr - dW) / (2.0 * sig)  # dp_r/dλ
    out[5] = A * c / (sig * s) * (2.0 * a + A / s2)          # dp_θ/dλ


@njit(cache=True, fastmath=True)
def hamiltonian(y, L, a):
    r = y[1]
    th = y[2]
    s = max(abs(math.sin(th)), SIN_MIN)
    c = math.cos(th)
    sig = r * r + a * a * c * c
    dlt = r * r - 2.0 * r + a * a
    R = r * r + a * a - a * L
    A = L - a * s * s
    return (dlt * y[4] ** 2 + y[5] ** 2 - R * R / dlt + A * A / (s * s)) / (2.0 * sig)


@njit(cache=True, fastmath=True)
def rk4_step(y, L, a, h, k1, k2, k3, k4, tmp, ynew):
    rhs(y, L, a, k1)
    for i in range(6):
        tmp[i] = y[i] + 0.5 * h * k1[i]
    rhs(tmp, L, a, k2)
    for i in range(6):
        tmp[i] = y[i] + 0.5 * h * k2[i]
    rhs(tmp, L, a, k3)
    for i in range(6):
        tmp[i] = y[i] + h * k3[i]
    rhs(tmp, L, a, k4)
    for i in range(6):
        ynew[i] = y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


@njit(cache=True, fastmath=True)
def choose_step(y, k1, rh, step_frac, h_max):
    """Passo afim adaptativo (k1 = dy/dλ já calculado em y)."""
    r = y[1]
    # |dφ| sem o fator sinθ: resolve a passagem rápida de raios perto do eixo (polo)
    rate = abs(k1[1]) / max(r - rh, 1e-6) + abs(k1[2]) + abs(k1[3]) + 1e-12
    h = step_frac / rate
    hm = h_max * r  # passo máximo cresce com r (longe do buraco o espaço é ~plano)
    return h if h < hm else hm


@njit(cache=True, fastmath=True)
def fix_pole(y):
    """Raios que cruzam o eixo: reflete θ e gira φ em π."""
    if y[2] < 0.0:
        y[2] = -y[2]
        y[3] += math.pi
        y[5] = -y[5]
    elif y[2] > math.pi:
        y[2] = 2.0 * math.pi - y[2]
        y[3] += math.pi
        y[5] = -y[5]


@njit(cache=True, fastmath=True)
def null_pr(r, th, pth, L, a, sign):
    """p_r que satisfaz H = 0 (sign = ±1)."""
    s = max(abs(math.sin(th)), SIN_MIN)
    dlt = r * r - 2.0 * r + a * a
    R = r * r + a * a - a * L
    A = L - a * s * s
    val = (R * R / dlt - A * A / (s * s) - pth * pth) / dlt
    if val < 0.0:
        val = 0.0
    return sign * math.sqrt(val)


@njit(cache=True, fastmath=True)
def trace_simple(r0, th0, ph0, pr0, pth0, L, a, direction, r_stop, step_frac, max_steps):
    """Integra um raio sem disco (usado nos testes).

    direction = +1 integra para o futuro, -1 para o passado.
    Retorna (estado, r_final, θ_final, φ_final, r_min, passos).
    """
    rh = horizon_radius(a)
    y = np.empty(6)
    y[0] = 0.0
    y[1] = r0
    y[2] = th0
    y[3] = ph0
    y[4] = pr0
    y[5] = pth0
    k1 = np.empty(6); k2 = np.empty(6); k3 = np.empty(6); k4 = np.empty(6)
    tmp = np.empty(6); yn = np.empty(6)
    rmin = r0
    for n in range(max_steps):
        rhs(y, L, a, k1)
        h = choose_step(y, k1, rh, step_frac, 0.1) * direction
        rk4_step(y, L, a, h, k1, k2, k3, k4, tmp, yn)
        for i in range(6):
            y[i] = yn[i]
        fix_pole(y)
        if y[1] < rmin:
            rmin = y[1]
        if y[1] < rh * 1.001 + 1e-3:
            return CAPTURED, y[1], y[2], y[3], rmin, n
        if y[1] > r_stop:
            return ESCAPED, y[1], y[2], y[3], rmin, n
    return MAXSTEPS, y[1], y[2], y[3], rmin, max_steps


@njit(cache=True, fastmath=True)
def orbit_equation_deflection(b, dphi):
    """Deflexão por u'' + u = 3u² (Schwarzschild, M=1), RK4 em φ.

    Começa no infinito (u=0, u'=1/b) e integra até u voltar a 0.
    Retorna (deflexão, capturado?).
    """
    u = 0.0
    v = 1.0 / b
    phi = 0.0
    for _ in range(50_000_000):
        k1u = v; k1v = 3 * u * u - u
        u2 = u + 0.5 * dphi * k1u; v2 = v + 0.5 * dphi * k1v
        k2u = v2; k2v = 3 * u2 * u2 - u2
        u3 = u + 0.5 * dphi * k2u; v3 = v + 0.5 * dphi * k2v
        k3u = v3; k3v = 3 * u3 * u3 - u3
        u4 = u + dphi * k3u; v4 = v + dphi * k3v
        k4u = v4; k4v = 3 * u4 * u4 - u4
        un = u + dphi / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        vn = v + dphi / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        if un > 0.5:  # r < 2M: caiu no horizonte
            return phi, True
        if un < 0.0:
            # interpolação linear do ponto onde u = 0
            f = u / (u - un)
            return phi + f * dphi - math.pi, False
        u = un; v = vn; phi += dphi
    return phi - math.pi, False
