"""Render de um quadro: passe base (1 amostra/pixel) + supersampling adaptativo.

O quadro é feito em faixas (tiles) de linhas. Com `tile_dir`, cada tile vai
para o disco assim que fica pronto, então um render interrompido retoma de
onde parou.
"""
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .color import build_lut
from .config import Preset, Camera
from .geodesic import horizon_radius, isco_radius
from .sky import get_sky
from .tracer import camera_frame, render_pixels, NPARAM, P_A, P_RH, P_RIN, P_ROUT, \
    P_TIN, P_ALPHA, P_TCAM, P_STEP, P_MAXSTEPS, P_RESC, P_DISKB, P_SKYB, P_TEXS, \
    P_MAXCROSS, P_TREF, P_HMAX

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"


def disk_texture_modes(seed, n=14):
    rng = np.random.default_rng(seed + 1000)
    m = rng.integers(1, 9, n).astype(np.float64)         # modos azimutais
    kr = rng.uniform(-30, 30, n)                          # frequência radial em log r
    amp = rng.uniform(0.3, 1.0, n) / np.sqrt(n) / (1 + 0.15 * m)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.stack([amp, m, kr, ph], 1)


class Renderer:
    def __init__(self, preset: Preset):
        self.p = preset
        s = preset.scene
        self.sky, self.sky_desc = get_sky(s.sky_width, OUTPUTS / "cache", ROOT / "assets", s.seed)
        self.lut = build_lut()
        self.tex_k = disk_texture_modes(s.seed)
        self.r_in = isco_radius(s.spin)
        self.r_h = horizon_radius(s.spin)

    def params(self, t_cam):
        s = self.p.scene
        P = np.zeros(NPARAM)
        P[P_A] = s.spin
        P[P_RH] = self.r_h
        P[P_RIN] = self.r_in
        P[P_ROUT] = s.r_out
        P[P_TIN] = s.T_in
        P[P_ALPHA] = s.disk_alpha
        P[P_TCAM] = t_cam
        P[P_STEP] = s.step_frac
        P[P_MAXSTEPS] = s.max_steps
        P[P_RESC] = s.r_escape
        P[P_DISKB] = s.disk_brightness
        P[P_SKYB] = s.sky_brightness
        P[P_TEXS] = s.texture_strength
        P[P_MAXCROSS] = s.max_crossings
        P[P_TREF] = s.T_in
        P[P_HMAX] = s.h_max
        return P

    def camera(self, cam: Camera, width, height):
        return camera_frame(cam.r, math.radians(cam.theta_deg), math.radians(cam.phi_deg),
                            self.p.scene.spin, cam.hfov_deg, width, height)

    def trace(self, px, py, k, width, height, C, P):
        rgb = np.empty((len(px), 3))
        code = np.empty(len(px), dtype=np.int64)
        render_pixels(px.astype(np.int64), py.astype(np.int64), k, width, height,
                      C, P, self.sky, self.lut, self.tex_k, rgb, code)
        return rgb, code

    def render_hdr(self, cam: Camera, t_cam=0.0, width=None, height=None, aa=None,
                   tile_dir=None, progress=True, desc="render"):
        r = self.p.render
        W = width or r.width
        H = height or r.height
        k = r.aa if aa is None else aa
        C = self.camera(cam, W, H)
        P = self.params(t_cam)
        if tile_dir is not None:
            tile_dir = Path(tile_dir)
            tile_dir.mkdir(parents=True, exist_ok=True)

        img = np.zeros((H, W, 3), dtype=np.float32)
        codes = np.zeros((H, W), dtype=np.int64)
        rows = list(range(0, H, r.tile_rows))
        bar = tqdm(total=H * W, desc=f"{desc} [base]", unit="px", disable=not progress,
                   smoothing=0.05, leave=False)
        for y0 in rows:
            y1 = min(H, y0 + r.tile_rows)
            f = tile_dir / f"base_{y0:05d}.npz" if tile_dir else None
            if f is not None and f.exists():
                d = np.load(f)
                img[y0:y1], codes[y0:y1] = d["rgb"], d["code"]
            else:
                yy, xx = np.mgrid[y0:y1, 0:W]
                rgb, code = self.trace(xx.ravel(), yy.ravel(), 1, W, H, C, P)
                img[y0:y1] = rgb.reshape(y1 - y0, W, 3)
                codes[y0:y1] = code.reshape(y1 - y0, W)
                if f is not None:
                    np.savez(f, rgb=img[y0:y1], code=codes[y0:y1])
            bar.update((y1 - y0) * W)
        bar.close()

        if k > 1:
            mask = self.aa_mask(img, codes, r.aa_threshold)
            ys, xs = np.nonzero(mask)
            chunk = max(2048, (r.tile_rows * W) // (k * k))
            bar = tqdm(total=len(xs), desc=f"{desc} [AA {k}x{k}]", unit="px",
                       disable=not progress, smoothing=0.05, leave=False)
            for ci, i0 in enumerate(range(0, len(xs), chunk)):
                i1 = min(len(xs), i0 + chunk)
                f = tile_dir / f"aa_{ci:05d}.npy" if tile_dir else None
                if f is not None and f.exists():
                    rgb = np.load(f)
                else:
                    rgb, _ = self.trace(xs[i0:i1], ys[i0:i1], k, W, H, C, P)
                    if f is not None:
                        np.save(f, rgb.astype(np.float32))
                img[ys[i0:i1], xs[i0:i1]] = rgb
                bar.update(i1 - i0)
            bar.close()
            self.last_aa_fraction = mask.mean()
        else:
            self.last_aa_fraction = 0.0
        return img

    @staticmethod
    def aa_mask(img, codes, thr):
        """Pixels de alta frequência: borda da sombra, anel de fótons, bordas do disco, estrelas."""
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        ll = np.log(lum + 1e-3 * (np.median(lum) + 1e-8))
        mask = np.zeros(lum.shape, dtype=bool)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = ll[max(dy, 0):, max(dx, 0):ll.shape[1] + min(dx, 0)]
            b = ll[:ll.shape[0] - dy, max(-dx, 0):ll.shape[1] - max(dx, 0)]
            ca = codes[max(dy, 0):, max(dx, 0):codes.shape[1] + min(dx, 0)]
            cb = codes[:codes.shape[0] - dy, max(-dx, 0):codes.shape[1] - max(dx, 0)]
            m = (np.abs(a - b) > thr) | (ca != cb)
            mask[max(dy, 0):, max(dx, 0):mask.shape[1] + min(dx, 0)] |= m
            mask[:mask.shape[0] - dy, max(-dx, 0):mask.shape[1] - max(dx, 0)] |= m
        return mask
