"""CLI: python -m blackhole render --preset preview|hd|video --spin 0.0"""
import argparse
import os
import sys
from dataclasses import replace


def default_threads():
    return max(1, (os.cpu_count() or 2) - 2)


def build_parser():
    ap = argparse.ArgumentParser(prog="blackhole", description="Ray tracer relativístico de buraco negro")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="renderiza imagem ou vídeo")
    r.add_argument("--preset", choices=["preview", "hd", "video"], default="preview")
    r.add_argument("--spin", type=float, default=0.0, help="spin a de Kerr (0 = Schwarzschild, <1)")
    r.add_argument("--threads", type=int, default=default_threads(),
                   help="núcleos de CPU (padrão: núcleos - 2)")
    r.add_argument("--landscape", action="store_true", help="vídeo 1920x1080 (padrão: 1080x1920)")
    r.add_argument("--width", type=int)
    r.add_argument("--height", type=int)
    r.add_argument("--scale", type=float, default=1.0, help="multiplica a resolução (ex.: 0.25 p/ teste)")
    r.add_argument("--aa", type=int, help="subamostras k×k nas regiões de alta frequência")
    r.add_argument("--inclination", type=float, help="θ da câmera em graus (90 = disco de lado)")
    r.add_argument("--r-cam", type=float, help="distância da câmera (M)")
    r.add_argument("--phi", type=float, help="azimute da câmera em graus")
    r.add_argument("--fov", type=float, help="FOV horizontal em graus")
    r.add_argument("--r-out", type=float, help="raio externo do disco (M)")
    r.add_argument("--temp", type=float, help="temperatura na ISCO (K)")
    r.add_argument("--exposure", type=float, help="exposição fixa (padrão: automática)")
    r.add_argument("--seconds", type=float, help="duração do vídeo")
    r.add_argument("--fps", type=int)
    r.add_argument("--frames", type=int, help="renderiza só os N primeiros quadros (teste)")
    r.add_argument("--out", help="pasta de saída (padrão: outputs/)")
    r.add_argument("--no-resume", action="store_true", help="não usar cache de tiles")
    r.add_argument("-y", "--yes", action="store_true", help="não pedir confirmação após o benchmark")
    b = sub.add_parser("bench", help="só o benchmark")
    b.add_argument("--preset", choices=["preview", "hd", "video"], default="hd")
    b.add_argument("--spin", type=float, default=0.0)
    b.add_argument("--threads", type=int, default=default_threads())
    b.add_argument("--landscape", action="store_true")
    return ap


def make_preset(args):
    from .config import get_preset
    p = get_preset(args.preset, vertical=not getattr(args, "landscape", False))
    if not -1 < args.spin < 1:
        sys.exit("--spin precisa estar em (-1, 1)")
    p.scene = replace(p.scene, spin=args.spin)
    g = lambda k: getattr(args, k, None)
    rend, cam, scene, vid = p.render, p.camera, p.scene, p.video
    W, H = g("width") or rend.width, g("height") or rend.height
    sc = g("scale") or 1.0
    W, H = max(16, int(W * sc) // 2 * 2), max(16, int(H * sc) // 2 * 2)
    rend = replace(rend, width=W, height=H)
    if g("aa"): rend = replace(rend, aa=args.aa)
    if g("exposure"): rend = replace(rend, exposure=args.exposure)
    if g("inclination") is not None: cam = replace(cam, theta_deg=args.inclination)
    if g("r_cam"): cam = replace(cam, r=args.r_cam)
    if g("phi") is not None: cam = replace(cam, phi_deg=args.phi)
    if g("fov"): cam = replace(cam, hfov_deg=args.fov)
    if g("r_out"): scene = replace(scene, r_out=args.r_out)
    if g("temp"): scene = replace(scene, T_in=args.temp)
    if g("seconds"): vid = replace(vid, seconds=args.seconds)
    if g("fps"): vid = replace(vid, fps=args.fps)
    if g("inclination") is not None:
        vid = replace(vid, theta_start=args.inclination + 4, theta_end=args.inclination - 4)
    if g("r_cam"):
        vid = replace(vid, r_start=args.r_cam, r_end=args.r_cam * 0.55)
    p.render, p.camera, p.scene, p.video = rend, cam, scene, vid
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    import numba
    numba.set_num_threads(max(1, min(args.threads, numba.config.NUMBA_NUM_THREADS)))
    from .pipeline import benchmark, fmt_time, render_still, render_video
    p = make_preset(args)
    print(f"preset={p.name} {p.render.width}x{p.render.height} spin={p.scene.spin} "
          f"threads={numba.get_num_threads()}")

    if args.cmd == "bench" or p.name != "preview":
        print("benchmark curto...")
        b = benchmark(p, frames=g_frames(args))
        print(f"  ~{b['sec_per_frame']:.1f} s/quadro, {b['frames']} quadro(s) "
              f"-> estimado {fmt_time(b['total_sec'])} (AA em ~{100 * b['aa_fraction']:.0f}% dos px)")
        if args.cmd == "bench":
            return
        if not args.yes:
            ans = input("Continuar? [s/N] ").strip().lower()
            if ans not in ("s", "sim", "y", "yes"):
                print("cancelado.")
                return

    if p.is_video:
        render_video(p, args.out, frames=args.frames)
    else:
        render_still(p, args.out, resume=not args.no_resume)


def g_frames(args):
    return getattr(args, "frames", None)


if __name__ == "__main__":
    main()
