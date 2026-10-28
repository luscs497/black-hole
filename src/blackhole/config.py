"""Presets e parâmetros da cena."""
import math
from dataclasses import dataclass, field, asdict, replace


@dataclass
class Scene:
    spin: float = 0.0            # parâmetro de spin a (0 = Schwarzschild)
    r_out: float = 20.0          # borda externa do disco (M)
    T_in: float = 6500.0         # temperatura na ISCO (K), referencial do gás
    disk_alpha: float = 0.93     # opacidade por cruzamento do disco
    disk_brightness: float = 1.0
    sky_brightness: float = 1.0
    texture_strength: float = 0.35
    max_crossings: int = 5
    step_frac: float = 0.03      # tamanho relativo do passo RK4
    max_steps: int = 30000       # teto de passos por raio
    h_max: float = 0.1           # passo máximo (× r)
    r_escape: float = 400.0      # raio a partir do qual o raio "escapou"
    sky_width: int = 4096        # largura da textura do céu
    seed: int = 7


@dataclass
class Camera:
    r: float = 30.0
    theta_deg: float = 80.0      # inclinação: 90° = de lado, 0° = de cima
    phi_deg: float = 0.0
    hfov_deg: float = 60.0


@dataclass
class Render:
    width: int = 320
    height: int = 180
    aa: int = 3                  # subamostras k×k nos pixels de alta frequência
    aa_threshold: float = 0.3    # diferença em log-luminância que dispara AA
    tile_rows: int = 32
    bloom: float = 0.08
    grain: float = 0.012
    exposure: float = 0.0        # 0 = automático


@dataclass
class Video:
    seconds: float = 15.0
    fps: int = 30
    r_start: float = 45.0
    r_end: float = 24.0
    theta_start: float = 84.0
    theta_end: float = 76.0
    phi_sweep_deg: float = 120.0
    time_scale: float = 25.0     # M de tempo coordenado por segundo de vídeo


@dataclass
class Preset:
    name: str
    scene: Scene = field(default_factory=Scene)
    camera: Camera = field(default_factory=Camera)
    render: Render = field(default_factory=Render)
    video: Video = field(default_factory=Video)
    is_video: bool = False

    def to_dict(self):
        return asdict(self)


def get_preset(name, vertical=True):
    if name == "preview":
        return Preset("preview", Scene(sky_width=2048), Camera(), Render(320, 180, aa=2))
    if name == "hd":
        return Preset("hd", Scene(sky_width=8192), Camera(), Render(1920, 1080, aa=4))
    if name == "video":
        if vertical:
            r = Render(1080, 1920, aa=3, tile_rows=64)
            cam = Camera(hfov_deg=55.0)
        else:
            r = Render(1920, 1080, aa=3, tile_rows=64)
            cam = Camera(hfov_deg=62.0)
        return Preset("video", Scene(sky_width=8192), cam, r, Video(), is_video=True)
    raise ValueError(f"preset desconhecido: {name}")


def ease(t):
    """Suavização (início e fim lentos)."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def video_camera(preset, frame, nframes):
    v = preset.video
    t = frame / max(nframes - 1, 1)
    e = ease(t)
    cam = replace(preset.camera,
                  r=v.r_start + (v.r_end - v.r_start) * e,
                  theta_deg=v.theta_start + (v.theta_end - v.theta_start) * e,
                  phi_deg=preset.camera.phi_deg + v.phi_sweep_deg * e)
    t_cam = frame / v.fps * v.time_scale
    return cam, t_cam
