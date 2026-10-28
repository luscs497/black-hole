"""Valida o shader em tempo real contra o render offline (src/blackhole).

Abre realtime/index.html?validate no Chrome headless (GPU real via ANGLE),
lê quais pixels de uma faixa de 2048 px caem no buraco negro e compara com
o mesmo cálculo feito pelo kernel offline (Numba, float64), pixel a pixel.

Uso:  .venv/bin/python realtime/validate.py
"""
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from numba import njit  # noqa: E402

from blackhole.geodesic import horizon_radius, rhs, rk4_step, fix_pole  # noqa: E402
from blackhole.tracer import (camera_frame, trace_pixel, NPARAM, P_A, P_RH, P_RIN,  # noqa: E402
                              P_ROUT, P_STEP, P_MAXSTEPS, P_RESC, P_HMAX, P_SKYB, P_MAXCROSS,
                              C_TANX, C_TANY)

B_CRIT = 3 * math.sqrt(3)


def kerr_equatorial_bc(a, prograde):
    r = 2 * (1 + math.cos(2 / 3 * math.acos(-a if prograde else a)))
    return abs((r * r * (3 - r) - a * a * (r + 1)) / (a * (r - 1)))


def find_chrome():
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        p = shutil.which(name)
        if p:
            return p
    return None


def run_shader(chrome, page):
    flags = ["--headless=new", "--no-sandbox", "--use-gl=angle", "--use-angle=gl"]
    out = subprocess.run([chrome, *flags, "--dump-dom", page.as_uri() + "?validate"],
                         capture_output=True, text=True, timeout=180).stdout
    m = re.search(r'<pre id="validation">(.*?)</pre>', out, re.S)
    if not m:
        raise RuntimeError("o shader não produziu resultado de validação")
    return json.loads(m.group(1).replace("&quot;", '"'))


def offline_strip(case):
    a, n, half, rc = case["a"], case["N"], case["half_fov_deg"], case["r_cam"]
    P = np.zeros(NPARAM)
    P[P_A] = a; P[P_RH] = horizon_radius(a); P[P_RIN] = 1e9; P[P_ROUT] = 0.0
    P[P_STEP] = 0.01; P[P_MAXSTEPS] = 200_000; P[P_RESC] = 3 * rc; P[P_HMAX] = 0.1
    P[P_SKYB] = 1.0; P[P_MAXCROSS] = 0
    C = camera_frame(rc, math.pi / 2, 0.0, a, 2 * half, n, n)
    tan_h = math.tan(math.radians(half))
    C[C_TANX] = tan_h; C[C_TANY] = tan_h
    sky = np.ones((4, 8, 3), dtype=np.float32); lut = np.ones((2048, 3))
    tex = np.zeros((1, 4)); out = np.empty(3)
    cap = []
    for i in range(n):
        t = 2 * (i + 0.5) / n - 1
        sx, sy = (t, 0.0) if case["dir"] == "h" else (0.0, t)
        cap.append(1 if trace_pixel(sx, sy, C, P, sky, lut, tex, out) % 4 == 0 else 0)
    return "".join(map(str, cap))


def edge_b(i_boundary, case):
    """Parâmetro de impacto do raio na fronteira entre os pixels i-1 e i."""
    t = 2 * i_boundary / case["N"] - 1
    psi = math.atan(abs(t) * math.tan(math.radians(case["half_fov_deg"])))
    rc = case["r_cam"]
    return rc * math.sin(psi) / math.sqrt(1 - 2 / rc)


@njit(cache=True)
def offline_direction(r0, th0, a, nx, ny, r_esc):
    """Direção final (cartesiana) de um raio da câmera ZAMO, integrada em float64
    com as mesmas equações do render offline e passo bem fino."""
    s, c = math.sin(th0), math.cos(th0)
    sig = r0 * r0 + a * a * c * c
    dlt = r0 * r0 - 2 * r0 + a * a
    A = (r0 * r0 + a * a) ** 2 - a * a * dlt * s * s
    al = math.sqrt(sig * dlt / A); vp = math.sqrt(A / sig) * s; om = 2 * a * r0 / A
    inv = 1 / math.sqrt(1 + nx * nx + ny * ny)
    mr, mth, mph = inv, ny * inv, -nx * inv
    L = vp * mph / (al + vp * om * mph)
    E = (1 - om * L) / al
    y = np.array([0.0, r0, th0, 0.0, math.sqrt(sig / dlt) * E * mr, math.sqrt(sig) * E * mth])
    k1 = np.empty(6); k2 = np.empty(6); k3 = np.empty(6); k4 = np.empty(6); tmp = np.empty(6); yn = np.empty(6)
    rh = horizon_radius(a)
    out = np.zeros(4)
    for _ in range(2_000_000):
        rhs(y, L, a, k1)
        rate = abs(k1[1]) / (y[1] - rh) + abs(k1[2]) + abs(k1[3])
        h = -min(0.002 / rate, 0.02 * y[1])
        rk4_step(y, L, a, h, k1, k2, k3, k4, tmp, yn)
        yp1, yp2, yp3 = y[1], y[2], y[3]
        for i in range(6):
            y[i] = yn[i]
        fix_pole(y)
        if y[1] < rh * 1.01:
            return out
        if y[1] > r_esc:
            rr0 = math.sqrt(yp1 * yp1 + a * a); rr1 = math.sqrt(y[1] * y[1] + a * a)
            dx = rr1 * math.sin(y[2]) * math.cos(y[3]) - rr0 * math.sin(yp2) * math.cos(yp3)
            dy = rr1 * math.sin(y[2]) * math.sin(y[3]) - rr0 * math.sin(yp2) * math.sin(yp3)
            dz = y[1] * math.cos(y[2]) - yp1 * math.cos(yp2)
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            out[0] = dx / n; out[1] = dy / n; out[2] = dz / n; out[3] = 1.0
            return out
    return out


def compare_grid(g):
    """Erro angular entre a direção final do shader e a do offline, pixel a pixel."""
    w, h, raw = g["w"], g["h"], g["raw"]
    tx = math.tan(math.radians(g["fov"]) / 2); ty = tx * h / w
    errs, mism = [], 0
    for j in range(h):
        for i in range(w):
            sx = 2 * (i + 0.5) / w - 1; sy = 2 * (j + 0.5) / h - 1
            ref = offline_direction(g["r"], g["th"], g["a"], sx * tx, sy * ty, g["resc"])
            gp = raw[(j * w + i) * 4:(j * w + i) * 4 + 4]
            esc_gpu, esc_cpu = gp[3] == 1, ref[3] == 1
            if esc_gpu != esc_cpu:
                mism += 1
            elif esc_cpu:
                d = max(-1.0, min(1.0, sum(gp[k] * ref[k] for k in range(3))))
                errs.append(math.degrees(math.acos(d)))
    errs.sort()
    return mism, errs[len(errs) // 2], errs[int(len(errs) * 0.99)], errs[-1]


def main():
    chrome = find_chrome()
    if chrome is None:
        print("Chrome/Chromium não encontrado: não dá pra rodar o shader headless.")
        return 2
    res = run_shader(chrome, ROOT / "realtime" / "index.html")
    print(f"GPU: {res['renderer']}")
    warn = {k: v for k, v in res["shaderLogs"].items() if v}
    print("logs de compilação/link:", "nenhum warning" if not warn else warn)
    ok = not warn
    for case in res["cases"]:
        gpu = case["captured"]
        cpu = offline_strip(case)
        diff = sum(x != y for x, y in zip(gpu, cpu))
        g0, g1 = gpu.index("1"), gpu.rindex("1") + 1
        c0, c1 = cpu.index("1"), cpu.rindex("1") + 1
        b_lo, b_hi = edge_b(g0, case), edge_b(g1, case)
        if case["a"] == 0:
            exp_lo = exp_hi = B_CRIT
        elif case["dir"] == "h":   # prógrado à esquerda, retrógrado à direita
            exp_lo, exp_hi = kerr_equatorial_bc(case["a"], True), kerr_equatorial_bc(case["a"], False)
        else:
            exp_lo = exp_hi = None
        line = (f"a={case['a']:.1f} {case['dir']}: GPU borda [{g0},{g1}) offline [{c0},{c1}) "
                f"px diferentes={diff}  b = {b_lo:.4f} / {b_hi:.4f}")
        if exp_lo:
            line += f"  (esperado {exp_lo:.4f} / {exp_hi:.4f})"
            ok &= abs(b_lo - exp_lo) / exp_lo < 3e-3 and abs(b_hi - exp_hi) / exp_hi < 3e-3
        ok &= diff <= 2 and abs(g0 - c0) <= 1 and abs(g1 - c1) <= 1
        print(line)
    for g in res.get("grids", []):
        mism, med, p99, mx = compare_grid(g)
        n = g["w"] * g["h"]
        print(f"a={g['a']:.1f} grade {g['w']}x{g['h']} da câmera padrão: estado diferente em {mism}/{n} px, "
              f"erro de direção mediana {med:.3f}°, p99 {p99:.3f}°, máx {mx:.3f}°")
        ok &= mism <= 0.01 * n and med < 0.3 and p99 < 1.5
    print("OK" if ok else "FALHOU")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
