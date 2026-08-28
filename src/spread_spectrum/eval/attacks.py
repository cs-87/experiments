"""
The attack model for prompt.md section 21.

src/retrieval/capture_sim.py already models a camera capture well -- session-constant
homography with per-frame jitter, moire, optics, photometrics -- and is reused here
rather than rewritten. What it does not cover, and what this module adds, is the
digital distribution path: H.265, real bitrate and preset sweeps, genuine resolution
changes, crop, sharpening, denoising, saturation, grayscale, and every temporal
attack.

Attacks operate on BGR frames, not luma. The mark lives in luma and the detector only
reads luma, but chroma subsampling and colour-space handling are part of what a real
codec does to a frame, and dropping chroma before the encoder would quietly remove
some of the damage being measured.

Three stages, applied in the order a real pipeline applies them:

    frame ops  ->  encode  ->  stream ops

Frame ops are spatial and photometric and happen before compression, because that is
where they happen in reality -- an editor resizes and then exports. Encode is a real
ffmpeg pass, so an attack that names a codec pays the real cost of that codec. Stream
ops are temporal and are applied at read-back, because dropping a frame from a file
and dropping it from a stream are the same thing to this detector.

An attack with no encode stage never materialises a file at all, which is most of
them and is the difference between a sweep that takes minutes and one that takes
hours.
"""

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


def iter_bgr(video_path, limit=None, stride=1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video_path}")
    i = kept = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield i, bgr
                kept += 1
                if limit is not None and kept >= limit:
                    break
            i += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------------
# frame operations (BGR uint8 in, BGR uint8 out)
# ---------------------------------------------------------------------------------

def _u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def op_scale(factor=None, height=None):
    """
    A genuine resolution change: down and back is NOT what this does. The frame comes
    out at the new size and stays there, which is what a 1080p asset re-published at
    720p looks like, and it is the case the detector's scale search has to handle.
    """
    def f(bgr):
        h, w = bgr.shape[:2]
        nh = int(height) if height else int(round(h * factor))
        nw = int(round(w * nh / h))
        return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return f


def op_resample(factor):
    """Down then back to the original size -- an aliasing attack, not a resize."""
    def f(bgr):
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (max(1, int(w * factor)), max(1, int(h * factor))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return f


def op_crop(fraction):
    """Centre crop keeping `fraction` of each axis. Output is smaller, not padded."""
    def f(bgr):
        h, w = bgr.shape[:2]
        ch, cw = int(h * fraction), int(w * fraction)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return bgr[y0:y0 + ch, x0:x0 + cw]
    return f


def op_crop_rescale(fraction):
    """Crop then scale back up -- the common "zoom in to remove a logo" edit."""
    crop = op_crop(fraction)
    def f(bgr):
        h, w = bgr.shape[:2]
        return cv2.resize(crop(bgr), (w, h), interpolation=cv2.INTER_CUBIC)
    return f


def op_translate(dy, dx):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return lambda bgr: cv2.warpAffine(bgr, m, (bgr.shape[1], bgr.shape[0]),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REFLECT)


def op_rotate(degrees):
    def f(bgr):
        h, w = bgr.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
        return cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT)
    return f


def op_blur(sigma):
    return lambda bgr: _u8(cv2.GaussianBlur(bgr.astype(np.float32), (0, 0), sigma))


def op_sharpen(amount=1.0, sigma=1.0):
    """Unsharp mask. The interesting one: sharpening AMPLIFIES a high-frequency mark."""
    def f(bgr):
        x = bgr.astype(np.float32)
        return _u8(x + amount * (x - cv2.GaussianBlur(x, (0, 0), sigma)))
    return f


def op_denoise(h=7):
    return lambda bgr: cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)


def op_noise(sigma, seed=0):
    rng = np.random.default_rng(seed)
    return lambda bgr: _u8(bgr.astype(np.float32)
                           + rng.normal(0, sigma, bgr.shape).astype(np.float32))


def op_brightness(delta):
    return lambda bgr: _u8(bgr.astype(np.float32) + delta)


def op_contrast(factor):
    return lambda bgr: _u8((bgr.astype(np.float32) - 128.0) * factor + 128.0)


def op_gamma(g):
    lut = _u8(((np.arange(256) / 255.0) ** g) * 255.0)
    return lambda bgr: cv2.LUT(bgr, lut)


def op_saturation(factor):
    def f(bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return f


def op_grayscale():
    return lambda bgr: cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                                    cv2.COLOR_GRAY2BGR)


def op_capture(condition, frame_size, seed=0):
    """One of src/retrieval/capture_sim.py's camera-capture presets."""
    from src.retrieval.capture_sim import CaptureSim, get_condition
    sim = CaptureSim(get_condition(condition), frame_size, seed=seed)
    return sim.apply


def chain(*ops):
    ops = [o for o in ops if o is not None]
    def f(bgr):
        for o in ops:
            bgr = o(bgr)
        return bgr
    return f


# ---------------------------------------------------------------------------------
# stream operations (temporal)
# ---------------------------------------------------------------------------------

def stream_drop(fraction, seed=0):
    rng = np.random.default_rng(seed)
    def f(frames):
        for idx, bgr in frames:
            if rng.random() >= fraction:
                yield idx, bgr
    return f


def stream_duplicate(fraction, seed=0):
    rng = np.random.default_rng(seed)
    def f(frames):
        for idx, bgr in frames:
            yield idx, bgr
            if rng.random() < fraction:
                yield idx, bgr
    return f


def stream_decimate(keep_every):
    """Frame-rate reduction: 30 -> 15 fps at keep_every = 2."""
    def f(frames):
        for n, (idx, bgr) in enumerate(frames):
            if n % keep_every == 0:
                yield idx, bgr
    return f


def stream_shift(offset):
    """Temporal offset: the first `offset` frames are simply never seen."""
    def f(frames):
        for n, (idx, bgr) in enumerate(frames):
            if n >= offset:
                yield idx, bgr
    return f


def stream_interpolate():
    """Average adjacent frames -- what naive frame-rate conversion produces."""
    def f(frames):
        prev = None
        for idx, bgr in frames:
            if prev is not None:
                yield idx, _u8((prev.astype(np.float32) + bgr.astype(np.float32)) / 2)
            prev = bgr
    return f


def stream_reverse():
    """Reordering. Costs this detector nothing, which is the point of measuring it."""
    def f(frames):
        yield from reversed(list(frames))
    return f


# ---------------------------------------------------------------------------------
# the attack
# ---------------------------------------------------------------------------------

@dataclass
class Encode:
    codec: str = "libx264"
    crf: int = None
    bitrate: str = None
    preset: str = "medium"
    extra: tuple = ()

    def args(self):
        a = ["-c:v", self.codec, "-preset", self.preset]
        if self.bitrate:
            a += ["-b:v", self.bitrate, "-maxrate", self.bitrate,
                  "-bufsize", self.bitrate]
        else:
            a += ["-crf", str(self.crf if self.crf is not None else 23)]
        return a + list(self.extra) + ["-pix_fmt", "yuv420p"]

    def label(self):
        rate = self.bitrate if self.bitrate else f"crf{self.crf}"
        short = {"libx264": "h264", "libx265": "h265"}.get(self.codec, self.codec)
        return f"{short}_{rate}" + (f"_{self.preset}" if self.preset != "medium" else "")


@dataclass
class Attack:
    name: str
    frame: object = None
    encode: Encode = None
    stream: object = None
    meta: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Attack({self.name})"


def _encode_frames(frames, path, fps, encode):
    """Pipe BGR through ffmpeg. Size is taken from the first frame, after frame ops."""
    proc = first = None
    try:
        for idx, bgr in frames:
            if proc is None:
                first = bgr.shape
                cmd = (["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "rawvideo", "-pix_fmt", "bgr24",
                        "-s", f"{bgr.shape[1]}x{bgr.shape[0]}", "-r", f"{fps:.6f}",
                        "-i", "-", "-an"] + encode.args() + [str(path)])
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            if bgr.shape != first:
                raise ValueError("frame size changed mid-stream; ffmpeg cannot take that")
            proc.stdin.write(np.ascontiguousarray(bgr, dtype=np.uint8).tobytes())
    finally:
        if proc is not None:
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {path}")
    return first


def apply_attack(attack, video_path, limit=None, stride=1, workdir=None, fps=30.0):
    """
    Yields (frame_index, luma float64) after the attack.

    The temporary encode, when there is one, lives in `workdir` and is deleted with it.
    """
    frames = iter_bgr(video_path, limit, stride)
    if attack.frame is not None:
        frames = ((i, attack.frame(b)) for i, b in frames)

    if attack.encode is not None:
        tmp = Path(workdir or tempfile.mkdtemp(prefix="ss_attack_"))
        tmp.mkdir(parents=True, exist_ok=True)
        out = tmp / f"{attack.name}.mp4"
        _encode_frames(frames, out, fps, attack.encode)
        frames = iter_bgr(out)

    if attack.stream is not None:
        frames = attack.stream(frames)

    for i, bgr in frames:
        yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float64)


# ---------------------------------------------------------------------------------
# the suite
# ---------------------------------------------------------------------------------

def build_suite(frame_size=(1920, 1080)):
    """Every axis prompt.md section 21 lists, plus the combined chains."""
    A = {}

    def add(name, **kw):
        A[name] = Attack(name=name, **kw)

    add("clean")

    for crf in (18, 23, 28, 32, 36, 40):
        add(f"h264_crf{crf}", encode=Encode("libx264", crf=crf))
    for crf in (23, 28, 32, 36):
        add(f"h265_crf{crf}", encode=Encode("libx265", crf=crf,
                                            extra=("-x265-params", "log-level=none")))
    for br in ("4M", "2M", "1M", "500k"):
        add(f"h264_{br}", encode=Encode("libx264", bitrate=br))
    for preset in ("ultrafast", "veryfast", "slow"):
        add(f"h264_crf28_{preset}", encode=Encode("libx264", crf=28, preset=preset))

    for h in (720, 540, 360):
        add(f"scale_{h}p", frame=op_scale(height=h), meta={"scale": h / frame_size[1]})
    add("upscale_1440p", frame=op_scale(height=1440), meta={"scale": 1440 / frame_size[1]})
    for f in (0.75, 0.5, 0.35):
        add(f"resample_{f}", frame=op_resample(f))

    for f in (0.9, 0.75, 0.5):
        add(f"crop_{f}", frame=op_crop(f))
        add(f"crop_rescale_{f}", frame=op_crop_rescale(f), meta={"scale": 1 / f})

    for s in (0.5, 1.0, 1.5, 2.0):
        add(f"blur_{s}", frame=op_blur(s))
    for a in (0.5, 1.0, 2.0):
        add(f"sharpen_{a}", frame=op_sharpen(a))
    add("denoise", frame=op_denoise(7))
    for s in (2, 5, 10, 20):
        add(f"noise_{s}", frame=op_noise(s))
    for d in (1, 2, 4):
        add(f"translate_{d}", frame=op_translate(d, d))
    for d in (0.25, 0.5, 1.0, 2.0):
        add(f"rotate_{d}", frame=op_rotate(d), meta={"rotation": d})

    for d in (-40, -20, 20, 40):
        add(f"brightness_{d}", frame=op_brightness(d))
    for f in (0.7, 0.85, 1.2, 1.5):
        add(f"contrast_{f}", frame=op_contrast(f))
    for g in (0.6, 0.8, 1.25, 1.6):
        add(f"gamma_{g}", frame=op_gamma(g))
    for f in (0.0, 0.5, 1.5):
        add(f"saturation_{f}", frame=op_saturation(f))
    add("grayscale", frame=op_grayscale())

    for f in (0.05, 0.2, 0.5):
        add(f"drop_{f}", stream=stream_drop(f))
        add(f"duplicate_{f}", stream=stream_duplicate(f))
    add("fps_half", stream=stream_decimate(2))
    add("fps_third", stream=stream_decimate(3))
    add("shift_7", stream=stream_shift(7))
    add("interpolate", stream=stream_interpolate())
    add("reverse", stream=stream_reverse())

    for cond in ("mild", "moderate", "severe"):
        add(f"capture_{cond}", frame=op_capture(cond, frame_size))

    # Combined chains, in the order a real pipeline applies them.
    add("combo_resize_h264_blur",
        frame=chain(op_scale(height=720), op_blur(0.8)),
        encode=Encode("libx264", crf=28), meta={"scale": 720 / frame_size[1]})
    add("combo_crop_h264_photometric",
        frame=chain(op_crop_rescale(0.8), op_gamma(1.2), op_contrast(1.15)),
        encode=Encode("libx264", crf=28), meta={"scale": 1 / 0.8})
    add("combo_resize_h265_drop",
        frame=op_scale(height=540), encode=Encode("libx265", crf=30,
                                                  extra=("-x265-params", "log-level=none")),
        stream=stream_drop(0.1), meta={"scale": 540 / frame_size[1]})
    add("combo_full",
        frame=chain(op_scale(height=720), op_noise(3), op_gamma(1.15)),
        encode=Encode("libx264", crf=32), stream=stream_drop(0.05),
        meta={"scale": 720 / frame_size[1]})
    return A


SUITE_ORDER = ("clean", "compression", "resolution", "spatial", "photometric",
               "temporal", "capture", "combined")
