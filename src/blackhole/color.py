"""Corpo negro (Planck) -> RGB linear (sRGB/Rec.709).

Integramos a lei de Planck contra as funções de cor CIE 1931 (aproximação
analítica multi-lobo de Wyman, Sloan & Shirley 2013), convertemos XYZ -> RGB
linear e normalizamos para luminância Y = 1. O brilho absoluto é aplicado à
parte via σT⁴ (I ∝ T⁴), assim a cor (cromaticidade) e a intensidade ficam
separadas e o shift g entra nos dois.
"""
import numpy as np

T_MIN = 500.0
T_MAX = 100_000.0
LUT_SIZE = 2048

_XYZ_TO_RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def _g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cie_cmf(lam_nm):
    x = 1.056 * _g(lam_nm, 599.8, 37.9, 31.0) + 0.362 * _g(lam_nm, 442.0, 16.0, 26.7) \
        - 0.065 * _g(lam_nm, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam_nm, 568.8, 46.9, 40.5) + 0.286 * _g(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam_nm, 437.0, 11.8, 36.0) + 0.681 * _g(lam_nm, 459.0, 26.0, 13.8)
    return x, y, z


def planck(lam_nm, T):
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    lam = lam_nm * 1e-9
    return (2 * h * c * c / lam ** 5) / np.expm1(h * c / (lam * k * T))


def blackbody_rgb(T):
    """Cromaticidade RGB linear de um corpo negro a T kelvin, com Y = 1."""
    lam = np.linspace(380.0, 780.0, 401)
    xb, yb, zb = cie_cmf(lam)
    B = planck(lam, T)
    xyz = np.array([np.trapezoid(B * xb, lam), np.trapezoid(B * yb, lam), np.trapezoid(B * zb, lam)])
    xyz /= xyz[1]
    rgb = _XYZ_TO_RGB @ xyz
    return np.clip(rgb, 0.0, None)


def build_lut():
    """LUT em log(T): linhas = [R, G, B] com luminância 1."""
    Ts = np.exp(np.linspace(np.log(T_MIN), np.log(T_MAX), LUT_SIZE))
    lut = np.empty((LUT_SIZE, 3), dtype=np.float64)
    for i, T in enumerate(Ts):
        lut[i] = blackbody_rgb(T)
    return lut
