"""
Spread-spectrum embedder: one 32-bit payload into the DWT LL of every SIFT patch of
every frame.

The whole payload is carried by every patch. There is no temporal code, no bit-to-frame
map and no run structure, which is why the detector needs no temporal synchronisation
of any kind -- dropped, duplicated, reordered or resampled frames cost it observations
and nothing else.

What the mark actually is, in pixels: a level-1 Haar LL of a 128x128 patch is a 64x64
plane whose coefficients are twice the 2x2 block means, so adding ALPHA * W to LL is
identical to adding (ALPHA / 2) * W to every pixel with each W element replicated over
its 2x2 block. At ALPHA = 3 that is a mark of RMS 1.5 and peak ~6.4 grey levels, i.e.
PSNR 44.6 dB. Raising the DWT level widens the block without changing the number of
chips, which trades processing gain for a lower carrier frequency.
"""

import argparse

import numpy as np
from tqdm import tqdm

from utils.bit import get_bit_string
from utils.dwt import get_dwt_coeff, reconstruct_frame
from utils.patch import get_sift_patches
from utils.video import Video_IO

from src.spread_spectrum.prn import BalancedPRNGenerator


INPUT = "./4_sec_source.mp4"
OUTPUT = "./out/wmk.mp4"

ALPHA = 3
SQUARE_SIZE = 128
DWT_LEVEL = 1
SEED = 8787


def embed_patch(patch, prn, alpha=ALPHA, level=DWT_LEVEL):
    """
    Marked copy of `patch`, as float. Clipping is the caller's job because it needs to
    know the target dtype's range.
    """
    coeffs = get_dwt_coeff(patch.astype(np.float64), level=level)
    coeffs.LL = coeffs.LL + alpha * prn
    return reconstruct_frame(coeffs)


def embed(video_path, output_path, watermark: int, seed=SEED, alpha=ALPHA,
          square_size=SQUARE_SIZE, level=DWT_LEVEL, min_separation=None,
          max_frames=None, progress=True):
    bit_string, _ = get_bit_string(watermark)

    video_io = Video_IO(video_path)
    frame_count = video_io.frame_count
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)

    # The LL of a square_size patch at this level is what the PRN has to fill.
    ll_side = square_size >> level
    prn_gen = BalancedPRNGenerator((ll_side, ll_side), seed)
    prn = prn_gen.get_balanced_prn_for_bit_string(bit_string)

    frames = 0
    patches_marked = 0
    clipped = 0
    marked_pixels = 0

    it = range(frame_count)
    if progress:
        it = tqdm(it, total=frame_count, unit="frame", desc="embedding")

    for _ in it:
        frame = video_io.read_frame()
        if frame is None:
            break

        for patch, _, _ in get_sift_patches(frame=frame.y, square_size=square_size,
                                            min_separation=min_separation):
            if patch is None:
                continue

            marked = embed_patch(patch, prn, alpha=alpha, level=level)

            # patch is a view into frame.y, so writing through it is what makes the mark
            # visible in the frame that gets written out below -- reconstruct_frame
            # returns a fresh array, and rebinding `patch` would silently drop the mark.
            #
            # Clip and round before the store. frame.y is uint8, and assigning a float
            # array into a uint8 view casts by C rules: out-of-range wraps (258.9 -> 2,
            # -1.2 -> 255) rather than saturating, which puts salt-and-pepper speckle in
            # highlights and shadows. Measured 0.022% of marked pixels out of range at
            # ALPHA = 3. Rounding rather than truncating matters separately: truncation
            # darkens every marked patch by half a grey level on average, which is a
            # patch-shaped brightness step at the patch boundary. It costs the detector
            # nothing -- the PRNs have exactly zero DC, so a constant offset projects out
            # -- but it is visible.
            clipped += int(((marked > 255) | (marked < 0)).sum())
            marked_pixels += marked.size
            patch[:] = np.rint(np.clip(marked, 0, 255)).astype(np.uint8)
            patches_marked += 1

        # Written unconditionally, marked or not: the output has to stay frame-for-frame
        # with the source.
        video_io.write_frame(frame, output_path)
        frames += 1

    video_io.release()

    stats = {
        "frames": frames,
        "patches_marked": patches_marked,
        "patches_per_frame": patches_marked / frames if frames else 0.0,
        "clipped_pixels": clipped,
        "clipped_fraction": clipped / marked_pixels if marked_pixels else 0.0,
        "payload": bit_string,
        "alpha": alpha,
        "square_size": square_size,
        "level": level,
        "seed": seed,
        "min_separation": min_separation,
    }
    if progress:
        print(f"marked {patches_marked} patches over {frames} frames "
              f"({stats['patches_per_frame']:.1f}/frame); "
              f"{clipped} pixels clipped ({100 * stats['clipped_fraction']:.4f}%)")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--output", default=OUTPUT)
    ap.add_argument("--watermark", type=lambda s: int(s, 0), default=87108710,
                    help="32-bit payload; accepts 0x-prefixed hex")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--square-size", type=int, default=SQUARE_SIZE)
    ap.add_argument("--level", type=int, default=DWT_LEVEL)
    ap.add_argument("--min-separation", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    embed(video_path=args.input, output_path=args.output, watermark=args.watermark,
          seed=args.seed, alpha=args.alpha, square_size=args.square_size,
          level=args.level, min_separation=args.min_separation,
          max_frames=args.max_frames)


if __name__ == "__main__":
    main()
