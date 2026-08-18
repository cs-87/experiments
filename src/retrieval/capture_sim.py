"""
Synthetic camera-capture ("camcording") degradation, used to build a retrieval
benchmark with exact ground truth.

The real leak pair gives realism but no reliable frame-level ground truth; this
gives the opposite, so the two are complementary and the experiment runs both.

The geometry is sampled ONCE per capture session and then only jittered per
frame. A fresh random homography every frame would model a camera that teleports,
which makes retrieval look harder than it is for the wrong reason -- and would
also destroy the one property a real camcord has that a detector can exploit,
namely that consecutive leak frames share a viewpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class CaptureConfig:
    """
    One camera-capture condition. Every field is an amplitude; zero disables the
    corresponding effect, so single-axis ablations are just a preset with one
    field set.
    """
    name: str = "custom"

    # Geometry: fraction of frame size by which each corner may be pulled.
    perspective: float = 0.0
    # Per-frame handheld jitter, in pixels, on top of the session homography.
    shake_px: float = 0.0
    rotation_deg: float = 0.0
    # Camera frames the screen loosely: <1 zooms out (screen bezel visible).
    zoom: float = 1.0

    # Photometric.
    gamma: float = 1.0
    exposure: float = 1.0
    wb_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    contrast: float = 1.0
    vignette: float = 0.0

    # Optics / sensor.
    blur_sigma: float = 0.0
    motion_blur_px: int = 0
    noise_sigma: float = 0.0

    # Screen re-photography artefacts.
    moire: float = 0.0
    # Intermediate resample factor. Non-integer is what actually produces
    # aliasing against the screen's pixel grid.
    resample: float = 1.0

    # Codec.
    jpeg_quality: int = 100

    # Output resolution of the simulated camera, (w, h). None keeps source size.
    out_size: tuple[int, int] | None = None


def _session_homography(cfg: CaptureConfig, w: int, h: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Viewpoint of the camera relative to the screen, fixed for the session."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    d = cfg.perspective * min(w, h)
    dst = src + rng.uniform(-d, d, size=(4, 2)).astype(np.float32)

    if cfg.zoom != 1.0:
        c = np.float32([w / 2, h / 2])
        dst = c + (dst - c) * cfg.zoom

    if cfg.rotation_deg:
        theta = np.deg2rad(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
        c = np.float32([w / 2, h / 2])
        r = np.float32([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        dst = (dst - c) @ r.T + c

    return cv2.getPerspectiveTransform(src, dst.astype(np.float32))


def _apply_geometry(img: np.ndarray, H: np.ndarray, cfg: CaptureConfig,
                    rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]

    if cfg.shake_px:
        jitter = np.eye(3, dtype=np.float32)
        jitter[0, 2] = rng.normal(0, cfg.shake_px)
        jitter[1, 2] = rng.normal(0, cfg.shake_px)
        H = jitter @ H

    out_w, out_h = cfg.out_size if cfg.out_size else (w, h)
    if cfg.out_size:
        # Fit the warped frame into the camera's sensor, so a zoom-out shows
        # black around the screen rather than silently cropping the content.
        S = np.diag([out_w / w, out_h / h, 1.0]).astype(np.float32)
        H = S @ H

    return cv2.warpPerspective(
        img, H, (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def _apply_photometric(img: np.ndarray, cfg: CaptureConfig) -> np.ndarray:
    x = img.astype(np.float32) / 255.0

    if cfg.gamma != 1.0:
        x = np.power(np.clip(x, 0, 1), cfg.gamma)

    if cfg.exposure != 1.0:
        x = x * cfg.exposure

    if cfg.wb_gain != (1.0, 1.0, 1.0):
        # BGR, matching cv2's channel order.
        x = x * np.float32(cfg.wb_gain).reshape(1, 1, 3)

    if cfg.contrast != 1.0:
        x = (x - 0.5) * cfg.contrast + 0.5

    if cfg.vignette:
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        x = x * (1.0 - cfg.vignette * np.clip(r / np.sqrt(2), 0, 1) ** 2)[..., None]

    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _apply_moire(img: np.ndarray, cfg: CaptureConfig,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Beat pattern between the display's pixel grid and the sensor's.

    Modelled as two high-frequency gratings at a slight angle to the pixel axes,
    modulating the image multiplicatively. This is a caricature of real moire,
    but it reproduces the property that matters here: broadband high-frequency
    energy that is not present in the source and that varies smoothly in space.
    """
    if not cfg.moire:
        return img

    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    theta = rng.uniform(0, np.pi)
    f = rng.uniform(0.18, 0.42)  # cycles/px, near Nyquist where beats live
    u = xx * np.cos(theta) + yy * np.sin(theta)
    v = -xx * np.sin(theta) + yy * np.cos(theta)

    pattern = (np.sin(2 * np.pi * f * u) * np.sin(2 * np.pi * f * 0.97 * v))
    gain = 1.0 + cfg.moire * pattern

    x = img.astype(np.float32) * gain[..., None]
    return np.clip(x, 0, 255).astype(np.uint8)


def _apply_optics(img: np.ndarray, cfg: CaptureConfig,
                  rng: np.random.Generator) -> np.ndarray:
    if cfg.blur_sigma:
        k = int(2 * round(3 * cfg.blur_sigma) + 1)
        img = cv2.GaussianBlur(img, (k, k), cfg.blur_sigma)

    if cfg.motion_blur_px and cfg.motion_blur_px > 1:
        n = int(cfg.motion_blur_px)
        kern = np.zeros((n, n), np.float32)
        kern[n // 2, :] = 1.0 / n
        theta = rng.uniform(0, 180)
        rot = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), theta, 1.0)
        kern = cv2.warpAffine(kern, rot, (n, n))
        s = kern.sum()
        if s > 0:
            img = cv2.filter2D(img, -1, kern / s)

    if cfg.resample != 1.0:
        h, w = img.shape[:2]
        mid = (max(1, int(w * cfg.resample)), max(1, int(h * cfg.resample)))
        img = cv2.resize(img, mid, interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    if cfg.noise_sigma:
        noise = rng.normal(0, cfg.noise_sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def _apply_codec(img: np.ndarray, cfg: CaptureConfig) -> np.ndarray:
    if cfg.jpeg_quality >= 100:
        return img
    ok, buf = cv2.imencode(
        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class CaptureSim:
    """
    Stateful simulator for one capture session.

    Construct once per (video, condition) pair and call ``apply`` per frame; the
    session homography and RNG stream persist across the call so the simulated
    camera stays put.
    """

    def __init__(self, cfg: CaptureConfig, frame_size: tuple[int, int], seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        w, h = frame_size
        self.H = _session_homography(cfg, w, h, self.rng)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """frame: HxWx3 uint8 BGR -> degraded HxWx3 uint8 BGR."""
        cfg = self.cfg
        x = _apply_geometry(frame, self.H, cfg, self.rng)
        x = _apply_photometric(x, cfg)
        x = _apply_moire(x, cfg, self.rng)
        x = _apply_optics(x, cfg, self.rng)
        x = _apply_codec(x, cfg)
        return x


# Named conditions. The single-axis ones exist so a failure can be attributed to
# a specific distortion instead of to "severe" as a whole.
CONDITIONS: dict[str, CaptureConfig] = {
    "clean": CaptureConfig(name="clean"),

    "mild": CaptureConfig(
        name="mild", perspective=0.01, shake_px=0.5, rotation_deg=0.5, zoom=0.98,
        gamma=1.05, exposure=1.03, wb_gain=(1.02, 1.0, 0.98), contrast=0.98,
        blur_sigma=0.6, noise_sigma=2.0, resample=0.9, jpeg_quality=90,
    ),
    "moderate": CaptureConfig(
        name="moderate", perspective=0.035, shake_px=1.5, rotation_deg=1.5, zoom=0.92,
        gamma=1.25, exposure=0.9, wb_gain=(1.08, 1.0, 0.9), contrast=0.9,
        vignette=0.2, blur_sigma=1.2, motion_blur_px=3, noise_sigma=5.0,
        moire=0.06, resample=0.72, jpeg_quality=70,
    ),
    "severe": CaptureConfig(
        name="severe", perspective=0.075, shake_px=3.0, rotation_deg=4.0, zoom=0.82,
        gamma=1.6, exposure=0.72, wb_gain=(1.18, 1.0, 0.82), contrast=0.78,
        vignette=0.38, blur_sigma=2.2, motion_blur_px=7, noise_sigma=10.0,
        moire=0.14, resample=0.5, jpeg_quality=45,
    ),

    "perspective_only": CaptureConfig(name="perspective_only", perspective=0.075,
                                      rotation_deg=4.0, zoom=0.82),
    "blur_only": CaptureConfig(name="blur_only", blur_sigma=2.2, motion_blur_px=7),
    "exposure_only": CaptureConfig(name="exposure_only", gamma=1.6, exposure=0.72,
                                   wb_gain=(1.18, 1.0, 0.82), contrast=0.78,
                                   vignette=0.38),
    "compression_only": CaptureConfig(name="compression_only", jpeg_quality=30),
    "moire_only": CaptureConfig(name="moire_only", moire=0.14, resample=0.5),
    "noise_only": CaptureConfig(name="noise_only", noise_sigma=12.0),

    # Worst realistic case: a phone filmed from an angle in a dark room.
    "extreme": CaptureConfig(
        name="extreme", perspective=0.11, shake_px=5.0, rotation_deg=7.0, zoom=0.7,
        gamma=2.0, exposure=0.6, wb_gain=(1.3, 1.0, 0.72), contrast=0.65,
        vignette=0.5, blur_sigma=3.0, motion_blur_px=11, noise_sigma=16.0,
        moire=0.2, resample=0.4, jpeg_quality=30,
    ),
}


def get_condition(name: str, out_size: tuple[int, int] | None = None) -> CaptureConfig:
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; have {sorted(CONDITIONS)}")
    cfg = CONDITIONS[name]
    return replace(cfg, out_size=out_size) if out_size else cfg
