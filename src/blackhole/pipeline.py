"""Imagem estática, vídeo (com retomada), benchmark e codificação."""
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import video_camera
from .post import auto_exposure, finish
from .render import Renderer, OUTPUTS


def save_png(path, rgb8):
    Image.fromarray(rgb8).save(path)


def run_name(preset):
    s, c, r = preset.scene, preset.camera, preset.render
    return f"{preset.name}_a{s.spin:+.2f}_i{c.theta_deg:.0f}_{r.width}x{r.height}"


def render_still(preset, out_dir=None, resume=True):
    out_dir = Path(out_dir or OUTPUTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = run_name(preset)
    R = Renderer(preset)
    print(f"céu: {R.sky_desc} | r_h = {R.r_h:.4f} M | r_ISCO = {R.r_in:.4f} M")
    tile_dir = out_dir / "tiles" / name if resume else None
    t0 = time.time()
    hdr = R.render_hdr(preset.camera, 0.0, tile_dir=tile_dir, desc=name)
    exp = preset.render.exposure or auto_exposure(hdr)
    img = finish(hdr, exp, preset.render.bloom, preset.render.grain, seed=preset.scene.seed)
    path = out_dir / f"{name}.png"
    save_png(path, img)
    np.save(out_dir / f"{name}_hdr.npy", hdr.astype(np.float16))
    if tile_dir is not None:
        shutil.rmtree(tile_dir, ignore_errors=True)
    print(f"ok: {path}  ({time.time() - t0:.1f} s, AA em {100 * R.last_aa_fraction:.1f}% dos pixels, "
          f"exposição {exp:.3g})")
    return path


def benchmark(preset, frames=None):
    """Renderiza em ~1/6 da resolução (com AA) e extrapola o tempo total."""
    R = Renderer(preset)
    r = preset.render
    w, h = max(96, r.width // 6), max(96, r.height // 6)
    # aquece o JIT
    R.render_hdr(preset.camera, 0.0, width=16, height=16, aa=1, progress=False)
    cams = [(preset.camera, 0.0)]
    n = 1
    if preset.is_video:
        n = frames or int(round(preset.video.seconds * preset.video.fps))
        cams = [video_camera(preset, 0, n), video_camera(preset, n - 1, n)]
    t_px, fracs = [], []
    for cam, tc in cams:
        t0 = time.time()
        R.render_hdr(cam, tc, width=w, height=h, aa=r.aa, progress=False)
        t_px.append((time.time() - t0) / (w * h))
        fracs.append(R.last_aa_fraction)
    # em resolução maior a fração de pixels com AA cai, então isso é conservador
    t_px = float(np.mean(t_px))
    per_frame = t_px * r.width * r.height
    return {"sec_per_frame": per_frame, "total_sec": per_frame * n,
            "aa_fraction": float(np.mean(fracs)), "frames": n, "bench_res": f"{w}x{h}"}


def fmt_time(s):
    s = int(s)
    h, m = divmod(s, 3600)
    m, s = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def render_video(preset, out_dir=None, frames=None):
    """Renderiza quadro a quadro em PNG (pula os que já existem) e codifica o MP4."""
    out_dir = Path(out_dir or OUTPUTS)
    name = run_name(preset)
    fdir = out_dir / "frames" / name
    fdir.mkdir(parents=True, exist_ok=True)
    n = frames or int(round(preset.video.seconds * preset.video.fps))
    state_f = fdir / "state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    R = Renderer(preset)
    print(f"céu: {R.sky_desc} | {n} quadros em {fdir}")
    exposure = preset.render.exposure or state.get("exposure")
    todo = [i for i in range(n) if not (fdir / f"frame_{i:05d}.png").exists()]
    if len(todo) < n:
        print(f"retomando: {n - len(todo)} quadros já prontos")
    bar = tqdm(total=n, initial=n - len(todo), desc="vídeo", unit="quadro", smoothing=0.1)
    for i in todo:
        cam, t_cam = video_camera(preset, i, n)
        hdr = R.render_hdr(cam, t_cam, tile_dir=fdir / "tiles", desc=f"quadro {i}")
        if exposure is None:
            # exposição travada no 1º quadro (evita "piscar" no vídeo)
            exposure = auto_exposure(hdr)
            state["exposure"] = exposure
            state_f.write_text(json.dumps(state))
        img = finish(hdr, exposure, preset.render.bloom, preset.render.grain, seed=i)
        tmp = fdir / f"frame_{i:05d}.tmp.png"
        save_png(tmp, img)
        tmp.rename(fdir / f"frame_{i:05d}.png")
        shutil.rmtree(fdir / "tiles", ignore_errors=True)
        bar.update(1)
    bar.close()
    mp4 = out_dir / f"{name}.mp4"
    encode(fdir, mp4, preset.video.fps)
    return mp4


def encode(frame_dir, mp4, fps):
    pattern = str(Path(frame_dir) / "frame_%05d.png")
    ff = shutil.which("ffmpeg")
    if ff:
        cmd = [ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
               "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(mp4)]
        subprocess.run(cmd, check=True)
    else:
        print("aviso: ffmpeg não encontrado no sistema, usando imageio-ffmpeg")
        import imageio.v2 as iio
        files = sorted(Path(frame_dir).glob("frame_*.png"))
        with iio.get_writer(mp4, fps=fps, codec="libx264", quality=9,
                            pixelformat="yuv420p") as w:
            for f in files:
                w.append_data(iio.imread(f))
    print(f"ok: {mp4}")
