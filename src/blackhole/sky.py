"""Fundo do céu: mapa equiretangular (./assets) ou céu procedural.

O céu procedural tem estrelas com distribuição de magnitudes tipo lei de
potência (muitas fracas, poucas brilhantes), temperaturas de 2800 K a 30000 K
(cor por corpo negro) e uma faixa difusa de Via Láctea com poeira escura.
A textura é gerada uma vez e guardada em cache (outputs/cache).
"""
import math
from pathlib import Path

import numpy as np
from numba import njit, prange

from .color import build_lut, T_MIN, T_MAX, LUT_SIZE

ASSET_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".hdr")


@njit(cache=True, fastmath=True)
def lut_color(lut, T, out):
    x = (math.log(T) - math.log(T_MIN)) / (math.log(T_MAX) - math.log(T_MIN)) * (LUT_SIZE - 1)
    if x < 0.0:
        x = 0.0
    if x > LUT_SIZE - 1.001:
        x = LUT_SIZE - 1.001
    i = int(x)
    f = x - i
    for c in range(3):
        out[c] = lut[i, c] * (1 - f) + lut[i + 1, c] * f


@njit(cache=True)
def _hash(i, j, seed):
    h = (i * 374761393 + j * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return (h ^ (h >> 16)) / 4294967296.0


@njit(cache=True, fastmath=True)
def _vnoise(x, y, seed, period):
    xi = int(math.floor(x)); yi = int(math.floor(y))
    fx = x - xi; fy = y - yi
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy)
    a = _hash(xi % period, yi, seed); b = _hash((xi + 1) % period, yi, seed)
    c = _hash(xi % period, yi + 1, seed); d = _hash((xi + 1) % period, yi + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


@njit(cache=True, fastmath=True)
def _fbm(x, y, seed, period, octaves):
    v = 0.0; amp = 0.5; norm = 0.0
    for o in range(octaves):
        v += amp * _vnoise(x, y, seed + o * 17, period)
        norm += amp; amp *= 0.5; x *= 2; y *= 2; period *= 2
    return v / norm


@njit(parallel=True, cache=True, fastmath=True)
def _milky_way(tex, nx, ny, nz, width, seed, strength):
    H, W, _ = tex.shape
    for j in prange(H):
        th = (j + 0.5) / H * math.pi
        for i in range(W):
            ph = (i + 0.5) / W * 2 * math.pi
            x = math.sin(th) * math.cos(ph); y = math.sin(th) * math.sin(ph); z = math.cos(th)
            d = x * nx + y * ny + z * nz  # distância (seno) ao plano galáctico
            band = math.exp(-(d / width) ** 2)
            core = math.exp(-(d / (0.35 * width)) ** 2)
            u = i / W * 16.0; v = j / H * 8.0
            n = _fbm(u, v, seed, 16, 6)
            dust = _fbm(u * 2.0, v * 2.0, seed + 99, 32, 5)
            dust_lane = 1.0 - 0.85 * math.exp(-(d / (0.12 * width)) ** 2) * (0.4 + 0.6 * dust)
            glow = strength * (band * (0.35 + 0.9 * n * n) + 0.6 * core * n) * dust_lane
            tex[j, i, 0] += glow * 1.00
            tex[j, i, 1] += glow * 0.88
            tex[j, i, 2] += glow * 0.78


@njit(cache=True, fastmath=True)
def _splat_stars(tex, th, ph, flux, col):
    H, W, _ = tex.shape
    sig = 0.6
    for k in range(th.shape[0]):
        y = th[k] / math.pi * H - 0.5
        x = ph[k] / (2 * math.pi) * W - 0.5
        st = max(math.sin(th[k]), 0.02)
        sx = sig / st  # equiretangular estica horizontalmente perto dos polos
        rx = int(3 * sx) + 1
        ry = 2
        norm = flux[k] / (2 * math.pi * sig * sx)
        for dj in range(-ry, ry + 1):
            jj = int(y) + dj
            if jj < 0 or jj >= H:
                continue
            for di in range(-rx, rx + 1):
                ii = int(x) + di
                w = math.exp(-0.5 * (((ii - x) / sx) ** 2 + ((jj - y) / sig) ** 2)) * norm
                ii %= W
                tex[jj, ii, 0] += w * col[k, 0]
                tex[jj, ii, 1] += w * col[k, 1]
                tex[jj, ii, 2] += w * col[k, 2]


def procedural_sky(width, n_stars=60000, seed=7):
    H, W = width // 2, width
    rng = np.random.default_rng(seed)
    tex = np.zeros((H, W, 3), dtype=np.float32)
    # Plano galáctico inclinado em relação ao eixo de spin do buraco
    nrm = np.array([0.35, -0.55, 0.76]); nrm /= np.linalg.norm(nrm)
    _milky_way(tex, nrm[0], nrm[1], nrm[2], 0.22, seed, 0.012)
    # Estrelas: uniformes na esfera + excesso perto do plano galáctico
    n_bg = int(n_stars * 0.6)
    z = rng.uniform(-1, 1, n_stars)
    ph = rng.uniform(0, 2 * np.pi, n_stars)
    v = np.stack([np.sqrt(1 - z * z) * np.cos(ph), np.sqrt(1 - z * z) * np.sin(ph), z], 1)
    # empurra 40% das estrelas para perto do plano galáctico
    gal = v[n_bg:]
    d = gal @ nrm
    gal -= np.outer(d * (1 - rng.uniform(0.0, 0.3, len(d))), nrm)
    gal /= np.linalg.norm(gal, axis=1, keepdims=True)
    v[n_bg:] = gal
    th = np.arccos(np.clip(v[:, 2], -1, 1))
    ph = np.mod(np.arctan2(v[:, 1], v[:, 0]), 2 * np.pi)
    # Magnitudes: N(<m) ∝ 10^(0.45 m) -> fluxo ∝ 10^(-0.4 m)
    m = 6.5 + np.log10(rng.uniform(1e-4, 1.0, n_stars)) / 0.45
    flux = (10 ** (-0.4 * (m - 1.0))).astype(np.float64) * 0.6
    # Temperaturas: maioria fria (K/M), algumas quentes (O/B)
    T = np.exp(rng.normal(np.log(5200), 0.45, n_stars))
    T = np.clip(T, 2800, 30000)
    lut = build_lut()
    col = np.empty((n_stars, 3))
    tmp = np.empty(3)
    for k in range(n_stars):
        lut_color(lut, T[k], tmp)
        col[k] = tmp
    _splat_stars(tex, th, ph, flux, col)
    return tex


def load_asset_sky(assets_dir):
    assets_dir = Path(assets_dir)
    if not assets_dir.is_dir():
        return None
    files = sorted(p for p in assets_dir.iterdir() if p.suffix.lower() in ASSET_EXT)
    if not files:
        return None
    import imageio.v3 as iio
    img = np.asarray(iio.imread(files[0])).astype(np.float32)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3]
    if files[0].suffix.lower() not in (".exr", ".hdr"):
        img = (img / 255.0) ** 2.2  # sRGB -> linear (aprox.)
        img *= 0.25
    return img, files[0].name


def get_sky(width, cache_dir, assets_dir="assets", seed=7):
    """Retorna (textura HxWx3 float32 linear, descrição)."""
    asset = load_asset_sky(assets_dir)
    if asset is not None:
        return asset[0], f"asset:{asset[1]}"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"sky_{width}_{seed}.npy"
    if f.exists():
        return np.load(f), "procedural (cache)"
    tex = procedural_sky(width, seed=seed)
    np.save(f, tex)
    return tex, "procedural"
