"""Pós-processamento: bloom, tone mapping filmic (ACES), grão, sRGB."""
import numpy as np
from scipy.ndimage import gaussian_filter


def luminance(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def auto_exposure(hdr, target=1.0, pct=99.5):
    lum = luminance(hdr)
    ref = np.percentile(lum, pct)
    return float(target / ref) if ref > 0 else 1.0


def bloom(hdr, strength=0.08, radii=(0.004, 0.012, 0.035)):
    """Soma de gaussianas em várias escalas (relativas à largura da imagem)."""
    H, W, _ = hdr.shape
    out = hdr.copy()
    bright = np.maximum(hdr - 0.6 * np.percentile(luminance(hdr), 99.0), 0.0)
    for i, rr in enumerate(radii):
        sig = max(rr * max(H, W), 0.5)
        blur = np.stack([gaussian_filter(bright[..., c], sig, mode="nearest") for c in range(3)], -1)
        out += strength * blur / (i + 1)
    return out


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def finish(hdr, exposure, bloom_strength=0.08, grain=0.012, seed=0):
    img = bloom(hdr * exposure, bloom_strength) if bloom_strength > 0 else hdr * exposure
    img = srgb(aces(img))
    if grain > 0:
        rng = np.random.default_rng(seed)
        H, W, _ = img.shape
        n = rng.normal(0.0, grain, (H, W, 1)).astype(np.float32)
        img = img + n * (0.35 + 0.65 * np.sqrt(np.clip(img, 0, 1)))
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
