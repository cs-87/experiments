"""
DC-domain patch watermark: same architecture as src/pixelate_patch.py, different carrier.

pixelate_patch marks a patch by pixelating it, which puts the signal at the grid's
fundamental frequency -- period 2 lands on Nyquist, the first thing H.264 quantises to
zero. This module instead offsets the patch's mean luma. That puts the signal on DC,
which carries the finest quantiser step in the codec and cannot be discarded without
visible brightness pumping.

Measured on 4_sec_source.mp4 at CRF 28, both carriers held to the same distortion
(MSE 11.5, 37.5 dB patch PSNR), scoring 120 frames per generation:

    generation          1      5     20     50
    pixelate@2  d'    2.17   1.23   0.69   0.64     accuracy 0.98 -> 0.78
    dc_shift    d'   32.97  13.42   7.08   4.25     accuracy 1.00 -> 1.00

Everything the embedder and detector share -- patch selection, the pool rotation, the
per-run flush -- is unchanged from pixelate_patch, so the replay argument documented
there applies here verbatim. Only the mark and the per-frame statistic are different.
"""

from functools import lru_cache

import numpy as np
from tqdm import tqdm

from utils.video import Video_IO
from utils.bit import get_bit_string, BIT_LENGTH, get_integer
from utils.patch import SQUARE_SIZE, get_best_patch_in_two_halves, select_candidate
from utils.lightglue import warp_patch

INPUT = "videos/15_sec_source.mp4"
OUTPUT = "./out/wmk.mp4"

LEAKED = "rencode_2.mp4"

TEMP_REDUNDANCY = 5

# Luma offset applied at the centre of the patch, in grey levels. 3.5 measured at
# MSE 9.4 (38.4 dB patch PSNR) -- slightly under pixelate@2's own distortion, and it
# still decoded every frame through 50 generations at CRF 28. Raising it buys margin
# roughly linearly; there is no need to.
DC_DELTA = 3.5

# Width of the raised-cosine falloff at the patch border, in pixels. A hard-edged
# 256x256 rectangle offset by a few levels is visible as a faint box against a smooth
# gradient, and the step at its border is exactly the kind of edge a deblocking filter
# will chew on. Feathering removes the visible edge while leaving almost all of the DC
# mass intact, because the taper only touches the outer ring.
FEATHER = 32


@lru_cache(maxsize=8)
def _feather_mask(h, w, width=FEATHER):
    """
    Separable raised-cosine window: 1.0 across the interior, tapering to 0 at the edge.

    Cached because embed() and detect() rebuild it for every patch of every frame and
    it only ever takes a couple of distinct shapes.
    """
    def ramp(n):
        r = np.ones(n, dtype=np.float32)
        k = min(width, n // 2)
        if k > 0:
            t = np.arange(k, dtype=np.float32)
            r[:k] = 0.5 * (1.0 - np.cos(np.pi * (t + 0.5) / k))
            r[n - k:] = r[:k][::-1]
        return r

    return np.outer(ramp(h), ramp(w)).astype(np.float32)


def dc_mark(patch, delta=DC_DELTA):
    """Add a feathered constant to the patch. The mark IS the mean shift."""
    mask = _feather_mask(patch.shape[0], patch.shape[1])
    marked = patch.astype(np.float32) + delta * mask
    return np.clip(np.rint(marked), 0, 255).astype(patch.dtype)


def dc_response(leak_patch, orig_patch):
    """
    Matched-filter response: the mask-weighted mean of (leak - original).

    Weighting by the same window used to embed rather than taking a plain mean is what
    makes this a matched filter -- the taper region contributed less signal, so it
    should contribute less to the measurement. Referencing against the original is free
    here because detect() already has the original open to replay patch selection.
    """
    mask = _feather_mask(leak_patch.shape[0], leak_patch.shape[1])
    diff = leak_patch.astype(np.float32) - orig_patch.astype(np.float32)
    return float((mask * diff).sum() / mask.sum())


def embed(video_path, output_path, watermark: int):
    bit_string, length = get_bit_string(watermark)
    video_io = Video_IO(video_path)
    frame_count = video_io.frame_count
    count = 0

    # Centres already marked during the current bit run, so the mark rotates instead of
    # sitting in one place. Cleared at every run boundary: detect() replays this exact
    # history from the original video, and the flush is what makes the replay possible
    # without knowing the bits.
    pool = []

    for i in tqdm(range(frame_count), total=frame_count, unit="frame", desc="embedding"):
        frame = video_io.read_frame()
        if frame is None:
            break

        if i % TEMP_REDUNDANCY == 0:
            pool.clear()

        bit_index = (i // TEMP_REDUNDANCY) % length
        bit = int(bit_string[bit_index])

        first_half, second_half = get_best_patch_in_two_halves(frame.y)
        half = first_half if bit == 1 else second_half

        selection = select_candidate(half, pool)

        if selection is not None:
            patch, y, x, centre = selection
            wmk_frame_y = frame.y.copy()
            wmk_frame_y[y[0]:y[1], x[0]:x[1]] = dc_mark(patch)
            frame.set_y(wmk_frame_y)
            pool.append(centre)
            count += 1

        # Written unconditionally, even when this frame carries no mark: dropping a
        # frame would shift every later frame and desynchronise detect()'s replay.
        video_io.write_frame(frame, output_path)

    print(f"embedded at {count} locations")
    video_io.release()


def detect(watermark_path, org_video_path, bit_length=BIT_LENGTH):
    """
    Recover the embedded watermark from a (possibly re-encoded) leaked video.

    Two differences from pixelate_patch.detect(), both of which matter more than the
    carrier swap on their own:

    Each half is scored against its own patch in the original, so scene content cancels
    before the two halves are ever compared. pixelate_patch scores each half in
    isolation and leans on the left-vs-right comparison to cancel content; that works
    only when both halves are equally textured.

    Votes are combined softly. pixelate_patch collapses every frame to a hard 1/0 and
    majority-votes, which throws away the magnitude of the per-frame evidence and lets a
    frame that barely decided count as much as one that decided overwhelmingly. Summing
    the signed response instead recovers roughly sqrt(N) in effective SNR over a run,
    and reports a z-score so a marginal decode is visible as marginal rather than
    silently wrong.
    """
    wmk_io = Video_IO(watermark_path)
    org_io = Video_IO(org_video_path)

    temp_redundancy = TEMP_REDUNDANCY * (wmk_io.frame_count // org_io.frame_count)
    if temp_redundancy == 0:
        temp_redundancy = TEMP_REDUNDANCY

    responses = [[] for _ in range(bit_length)]
    frames_read = 0

    # One pool per half, mirroring embed()'s single pool. Both are advanced on every
    # frame regardless of how the vote goes, which is what makes the replay independent
    # of the bits -- see pixelate_patch.detect() for the full argument.
    pool_first, pool_second = [], []

    for i in tqdm(range(wmk_io.frame_count), desc=f"detecting:{watermark_path}"):
        frame_og = org_io.read_frame()
        frame = wmk_io.read_frame()
        if frame is None or frame_og is None:
            break
        frames_read += 1

        if i % temp_redundancy == 0:
            pool_first.clear()
            pool_second.clear()

        bit_index = (i // temp_redundancy) % bit_length

        homography = np.eye(3, dtype=np.float32)

        first_half, second_half = get_best_patch_in_two_halves(frame_og.y)

        sel_first = select_candidate(first_half, pool_first)
        sel_second = select_candidate(second_half, pool_second)

        # Advance each pool as soon as its own half yields a candidate, before any of
        # the bail-outs below, or this half's history falls a step behind embed()'s for
        # the rest of the run.
        if sel_first is not None:
            pool_first.append(sel_first[3])
        if sel_second is not None:
            pool_second.append(sel_second[3])

        if sel_first is None or sel_second is None:
            continue

        (x1, y1), (x2, y2) = sel_first[3], sel_second[3]

        wrapped_first = warp_patch(frame.y, x1, y1, homography, SQUARE_SIZE)
        wrapped_second = warp_patch(frame.y, x2, y2, homography, SQUARE_SIZE)

        if wrapped_first is None or wrapped_second is None:
            continue

        # embed() lifts the first (left) half for a 1 and the second for a 0, so a
        # positive difference is evidence for 1. Any luma drift the re-encode applied to
        # the whole frame is common to both terms and cancels here.
        left = dc_response(wrapped_first, sel_first[0])
        right = dc_response(wrapped_second, sel_second[0])
        responses[bit_index].append(left - right)

    wmk_io.release()
    org_io.release()

    bit_str = ""
    margins = []
    for slot in responses:
        if not slot:
            bit_str += "?"
            margins.append(0.0)
            continue
        arr = np.asarray(slot, dtype=np.float64)
        # z-score of the slot mean: how many standard errors the evidence sits from
        # "no mark". |z| below ~3 means this slot is a coin flip dressed up as a bit.
        z = float(arr.mean() * np.sqrt(len(arr)) / (arr.std() + 1e-12))
        margins.append(z)
        bit_str += "1" if z > 0 else "0"

    # A short clip cannot carry every bit: only frames_read // temp_redundancy bit slots
    # were ever embedded, so trust no more than that many.
    if frames_read < temp_redundancy * bit_length:
        expected_length = frames_read // temp_redundancy
    else:
        expected_length = bit_length

    decoded = bit_str[:expected_length]
    weakest = min((abs(m) for m in margins[:expected_length]), default=0.0)
    print(f"votes: {bit_str}  decoded: {decoded}  slots carried: {expected_length}")
    print(f"weakest slot |z|: {weakest:.1f}")

    if not decoded or "?" in decoded:
        print("undecided slots, watermark not recoverable")
        return None

    watermark = get_integer(decoded, expected_length)
    print(watermark)
    return watermark


if __name__ == "__main__":
    embed(video_path=INPUT, output_path=OUTPUT, watermark=87108710)
    import subprocess
    
    subprocess.run(["bash", "utils/compress.sh",OUTPUT,"out.mp4","6"], check=True)
    detect(watermark_path="out.mp4", org_video_path=INPUT)
