"""
Find where the original starts inside a leak, as one global frame offset.

For the real-camera test the capture is a whole play-through of the clip at the same
frame rate, so the only unknown is *when the recording started*. That is a single
integer, and estimating a single integer does not need the streaming aligner in
src/retrieval/temporal_aligner.py -- that one carries an SSCD/LightGlue matcher and a
constant-velocity filter because it has to survive dropped and repeated frames and an
unknown, drifting mapping. None of that applies here, and all of it costs a GPU pass per
frame.

What this does instead:

  1. reduce every frame to an 8x8 grid of block means           (geometry-tolerant)
  2. normalise each frame's grid to zero mean, unit variance    (exposure/gamma/WB-proof)
  3. difference it in time                                      (kills static content)
  4. FFT cross-correlate the two sequences and take the peak    (the offset)

Steps 2 and 3 are what make it work on a camcord rather than on a transcode. A phone
camera changes exposure, white balance and gamma continuously, so the *absolute* level
of a frame is not comparable between the two videos -- but the way the level *changes*
from frame to frame is, because both videos are watching the same scene do the same
thing. Differencing also removes any constant photometric offset exactly, and reduces a
slow auto-exposure ramp to a small residual.

The 8x8 reduction is what makes it tolerant of the geometry. A handheld capture is
rotated, keystoned and cropped, so no pixel lands where it started; but a block mean over
an eighth of the frame moves very little under those transforms, and a cut or a big
motion still shows up in every block at once. It is also resolution-independent, so the
leak never has to be resized to match.

Sign convention, used everywhere in this module:

    original frame t  <->  leak frame t + offset

so `offset` is the leak frame index showing the original's first frame. It is positive
when the recording started before the clip did (the usual case: the camera was rolling,
then the video began), and negative when the capture missed the opening frames.

Typical use:

    from src.blur.frame_sync import find_offset, aligned_pairs
    from src.blur.blur_detect_mf import MFConfig, scan, decode

    sync = find_offset("inputs/120.mp4", "captures/phone.mp4")
    print(sync)                       # offset, confidence, drift
    cfg = MFConfig(radius=80, temp_redundancy=10)
    org, leak = aligned_pairs("inputs/120.mp4", "captures/phone.mp4", sync.offset)
    result = decode(scan(org, leak, cfg, aligner=GeometricAligner(lightglue=True)), cfg)

or, doing the whole thing at once:

    python -m src.blur.frame_sync inputs/120.mp4 captures/phone.mp4 --detect \
        --radius 80 --tr 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import fftconvolve

# Blocks per side in the per-frame signature. 8 gives 64 numbers per frame: enough that
# two different moments in a clip rarely produce the same vector, coarse enough that a
# few degrees of rotation or a small crop barely moves any of them.
GRID = 8

# Fraction trimmed from each edge before reducing. A camera capture usually carries some
# amount of surround -- bezel, wall, letterbox -- that belongs to no frame of the
# original and would otherwise sit in the outer blocks as a constant. Trimming the
# original by the same fraction keeps the two grids describing the same region.
BORDER_TRIM = 0.08

# A lag is only considered if this many frame-pairs actually overlap at it. Without a
# floor the extreme lags, where a handful of frames overlap, produce large correlations
# from almost no evidence and win on noise alone.
MIN_OVERLAP = 120

# Lags this close to the winner are treated as the same peak when measuring how far
# clear of the runner-up it finished. Adjacent lags always correlate -- consecutive
# frames are similar -- so the runner-up has to be looked for outside a small window or
# it is just the peak's own shoulder.
PEAK_EXCLUSION = 12


@dataclass(frozen=True)
class SyncResult:
    """
    Where the original sits inside the leak, and how much to believe it.

    `offset` is the answer. The rest is there so a bad answer is recognisable as one
    rather than being discovered later as a payload that will not decode.
    """

    offset: int
    peak: float             # correlation at the winning lag, roughly [-1, 1]
    margin: float           # winner / best lag outside the exclusion window
    overlap: int            # frame pairs the winning lag actually aligns
    org_frames: int
    leak_frames: int
    org_fps: float
    leak_fps: float
    thirds: tuple           # per-third offsets, for the drift check
    drift_per_1000: float   # frames of slip per 1000 frames, from the thirds

    @property
    def confident(self) -> bool:
        """
        Whether the offset is worth using without looking at it.

        Both halves matter. A high peak on its own can come from a clip whose every
        second looks like every other second; a high margin on its own can come from two
        equally poor lags. Requiring both is what separates "found it" from "picked the
        least bad option".
        """
        return self.peak >= 0.25 and self.margin >= 1.5

    @property
    def drifting(self) -> bool:
        """True when the three thirds disagree by more than rounding."""
        return abs(self.drift_per_1000) >= 1.0

    def __str__(self):
        verdict = "confident" if self.confident else "LOW CONFIDENCE"
        line = (f"offset={self.offset:+d} frames  ({verdict}: peak={self.peak:.3f}, "
                f"margin={self.margin:.2f}x, overlap={self.overlap})\n"
                f"  original {self.org_frames} frames @ {self.org_fps:.3f} fps"
                f"  |  leak {self.leak_frames} frames @ {self.leak_fps:.3f} fps\n"
                f"  thirds {self.thirds}  drift {self.drift_per_1000:+.2f} frames/1000")
        if self.drifting:
            line += ("\n  WARNING: the thirds disagree -- the two videos are not running at "
                     "the same rate.\n           A single offset cannot align them; see "
                     "the note in find_offset().")
        if abs(self.org_fps - self.leak_fps) > 0.01:
            line += (f"\n  WARNING: frame rates differ by "
                     f"{abs(self.org_fps - self.leak_fps):.3f} fps.")
        return line


def _signature(path, trim=BORDER_TRIM, grid=GRID, limit=None):
    """
    Per-frame signature of a video: (frames, grid*grid) of normalised block means.

    INTER_AREA is an exact block average when the target divides the source and a clean
    weighted one when it does not, so this is the block mean rather than a resampling of
    it. Frames whose blocks are all but identical -- a fade to black, a blank slate --
    have no usable shape, so they are left as zeros instead of being amplified into pure
    noise by the normalisation.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    rows = []
    try:
        while limit is None or len(rows) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            y = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]
            h, w = y.shape
            if trim > 0:
                dy, dx = int(h * trim), int(w * trim)
                y = y[dy:h - dy, dx:w - dx]
            cell = cv2.resize(y, (grid, grid),
                              interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            sd = cell.std()
            rows.append((cell - cell.mean()) / sd if sd > 1e-6
                        else np.zeros_like(cell))
    finally:
        cap.release()
    if not rows:
        raise ValueError(f"no frames read from {path}")
    return np.asarray(rows, dtype=np.float32), float(fps)


def _motion(sig):
    """
    Temporal derivative of a signature, z-scored per channel.

    The derivative is the part that survives a camera. An absolute block mean differs
    between the two videos by whatever the camera's exposure and gamma are doing; the
    frame-to-frame *change* in it is a property of the content. Per-channel z-scoring
    then stops a single high-variance block -- a window, a lamp -- from dominating the
    sum over channels.
    """
    d = np.diff(sig, axis=0)
    d -= d.mean(axis=0, keepdims=True)
    sd = d.std(axis=0, keepdims=True)
    return d / np.maximum(sd, 1e-6)


def _correlate(a, b, min_overlap=MIN_OVERLAP):
    """
    Mean per-channel correlation of `a` against `b` at every lag, and the lags.

    r[k] = mean over channels and over overlapping t of a[t] * b[t+k]. Computed by FFT
    because the direct form is O(T^2) per channel and these are thousands of frames long.

    Dividing by the true overlap at each lag, rather than by a constant, is what makes
    lags comparable: without it a lag that aligns 3000 frames always outscores one that
    aligns 300 regardless of how well either matches.
    """
    ta, tb = len(a), len(b)
    # fftconvolve with a reversed kernel is correlation; axes=0 keeps the channels
    # independent so they can be summed afterwards rather than mixed.
    full = fftconvolve(b, a[::-1], mode="full", axes=0).sum(axis=1)
    lags = np.arange(full.size) - (ta - 1)

    # Frames actually aligned at each lag: t in [0,ta) with t+k in [0,tb).
    overlap = np.minimum(ta, tb - lags) - np.maximum(0, -lags)
    valid = overlap >= min(min_overlap, ta, tb)
    if not valid.any():
        raise ValueError(
            f"no lag aligns at least {min_overlap} frames "
            f"(original {ta}, leak {tb}); the clips may not overlap at all")

    r = np.full(full.shape, -np.inf, dtype=np.float64)
    r[valid] = full[valid] / (overlap[valid] * a.shape[1])
    return r, lags, overlap


def _peak(r, lags, exclusion=PEAK_EXCLUSION):
    """Winning lag, its correlation, and how far clear of the next distinct peak it is."""
    i = int(np.argmax(r))
    peak = float(r[i])
    masked = r.copy()
    lo, hi = max(0, i - exclusion), min(len(r), i + exclusion + 1)
    masked[lo:hi] = -np.inf
    runner = float(masked.max()) if np.isfinite(masked).any() else 0.0
    # Both are correlations, so a negative or zero runner-up means the peak stands alone
    # and the ratio would be meaningless or misleadingly enormous.
    margin = peak / runner if runner > 1e-6 and peak > 0 else float("inf")
    return int(lags[i]), peak, min(margin, 999.0)


def find_offset(org_path, leak_path, trim=BORDER_TRIM, grid=GRID,
                min_overlap=MIN_OVERLAP, limit=None):
    """
    The leak frame index that shows the original's first frame.

    Returns a SyncResult; `offset` is the number, everything else says whether to trust
    it. Read it as `original frame t  <->  leak frame t + offset`.

    The thirds are estimated independently and reported as `drift_per_1000`. They exist
    because "same fps" is the assumption most likely to be wrong about a real capture:
    a phone recording nominal 30 fps often delivers 29.97, which slips a frame every 33
    seconds and silently destroys a temporally-coded payload halfway through a long clip.
    If the thirds disagree, the fix is not a better offset -- it is to resample one video
    onto the other's timebase, or to detect in windows short enough that the slip within
    a window stays under a frame.
    """
    a_sig, a_fps = _signature(org_path, trim, grid, limit)
    b_sig, b_fps = _signature(leak_path, trim, grid, limit)
    a, b = _motion(a_sig), _motion(b_sig)

    r, lags, overlap = _correlate(a, b, min_overlap)
    offset, peak, margin = _peak(r, lags)

    thirds = []
    n = len(a) // 3
    if n >= min(min_overlap, len(a)):
        for j in range(3):
            piece = a[j * n:(j + 1) * n]
            try:
                rj, lj, _ = _correlate(piece, b, min_overlap=min(min_overlap, n))
                # A third's own lag is relative to its start; shift it back to the
                # whole-clip frame of reference so the three are comparable.
                thirds.append(int(lj[int(np.argmax(rj))]) - j * n)
            except ValueError:
                thirds.append(None)
    got = [t for t in thirds if t is not None]
    # Slope across the thirds, expressed per 1000 frames of the original.
    drift = ((got[-1] - got[0]) / (2 * n) * 1000.0) if len(got) >= 2 and n else 0.0

    return SyncResult(
        offset=offset, peak=peak, margin=margin,
        overlap=int(overlap[int(np.argmax(r))]),
        org_frames=len(a_sig), leak_frames=len(b_sig),
        org_fps=a_fps, leak_fps=b_fps,
        thirds=tuple(thirds), drift_per_1000=drift,
    )


def aligned_pairs(org_path, leak_path, offset, limit=None, stride=1):
    """
    Two generators of luma planes, already lined up, ready for blur_detect_mf.scan().

    Whichever video starts late is skipped into, so both generators begin at the same
    moment of content and the caller can zip them without further bookkeeping. Returns
    (org_frames, leak_frames, first_org_index) -- the third value is the original frame
    number the pair sequence begins at, which scan() needs as the base of its
    `frame_indices` so the frame->bit map is read at the right place. It is nonzero only
    when the capture missed the opening frames.
    """
    from src.blur.blur_detect_mf import iter_y

    org_skip = max(0, -offset)
    leak_skip = max(0, offset)

    def skipped(path, n):
        gen = iter_y(path, limit=None, stride=1)
        for _ in range(n):
            next(gen, None)
        count = 0
        for frame in gen:
            if limit is not None and count >= limit:
                return
            if count % stride == 0:
                yield frame
            count += 1

    return skipped(org_path, org_skip), skipped(leak_path, leak_skip), org_skip


def sync_and_detect(org_path, leak_path, radius=80, temp_redundancy=10,
                    cluster_k=1, lightglue=True, homography_every=30,
                    limit=None, stride=1, offset=None, progress=True, **decode_kw):
    """
    Sync, then read the payload -- the whole real-capture path in one call.

    `offset=None` estimates it; pass a number to override when the estimate is known to
    be wrong. Alignment defaults on because a camera capture is geometrically distorted
    and the matched filter measured 19/32 without it against 31/32 with it.
    """
    from src.blur.blur_detect_mf import (GeometricAligner, MFConfig, decode, scan)

    sync = (find_offset(org_path, leak_path) if offset is None else
            SyncResult(offset, 1.0, float("inf"), 0, 0, 0, 0.0, 0.0, (), 0.0))

    cfg = MFConfig(radius=radius, temp_redundancy=temp_redundancy,
                   interleave=(temp_redundancy == 1), cluster_k=cluster_k)
    org, leak, base = aligned_pairs(org_path, leak_path, sync.offset, limit, stride)
    aligner = GeometricAligner(lightglue=lightglue, every=homography_every)

    result = scan(org, leak, cfg, aligner=aligner, progress=progress)
    # scan() numbers frames from zero; when the capture missed the opening frames the
    # first pair is original frame `base`, and the bit map has to be read from there.
    if base:
        result["frame"] = result["frame"] + base
    return sync, decode(result, cfg=cfg, **decode_kw)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("original")
    p.add_argument("leak")
    p.add_argument("--trim", type=float, default=BORDER_TRIM,
                   help="fraction cropped from each edge before reducing")
    p.add_argument("--grid", type=int, default=GRID)
    p.add_argument("--limit", type=int, default=None,
                   help="only read this many frames of each (faster on long clips)")
    p.add_argument("--detect", action="store_true",
                   help="also read the payload at the recovered offset")
    p.add_argument("--radius", type=int, default=80)
    p.add_argument("--tr", type=int, default=10)
    p.add_argument("--cluster-k", type=int, default=1)
    p.add_argument("--offset", type=int, default=None,
                   help="skip estimation and use this offset")
    p.add_argument("--no-lightglue", action="store_true")
    p.add_argument("--expect", default=None,
                   help="expected payload, e.g. 0xCAFECAFE, to report a BCR")
    args = p.parse_args(argv)

    if not args.detect:
        print(find_offset(args.original, args.leak, args.trim, args.grid,
                          limit=args.limit))
        return 0

    sync, got = sync_and_detect(
        args.original, args.leak, radius=args.radius, temp_redundancy=args.tr,
        cluster_k=args.cluster_k, lightglue=not args.no_lightglue,
        limit=args.limit, offset=args.offset)
    print(sync)
    print(f"\npayload   0x{got['watermark']:08X}   {got['bit_string']}")
    print(f"presence  {got['presence']:.2f}   frames used {got['frames_used']}")
    if args.expect:
        from src.blur.blur_detect_mf import bit_correct_rate, hard_bits
        want = int(args.expect, 0)
        print(f"expected  0x{want:08X}   BCR "
              f"{bit_correct_rate(want, hard_bits(got['L']))}/32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
