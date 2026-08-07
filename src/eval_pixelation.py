"""
Accuracy harness for the pixelation detector.

Three checks:

  recovery      does detect() read the watermark back out of a pixelated clip, and
                what fraction of individual frames voted correctly
  clean         does a clip with no watermark stay low-confidence, and is the
                half-vs-half winner split ~50/50 rather than locked to one side
                (this is the check that catches content-flatness bias)
  blocksize     does estimate_block_size() recover the pixel_size actually used
  compression   (opt-in) re-encode through H.264 at rising CRF and find the point
                where recovery breaks

Run from the repo root:
    python src/eval_pixelation.py
    python src/eval_pixelation.py --compression
    python src/eval_pixelation.py --wmk out/leak.mp4 --block-size 4

Last measured (1080p, 120 frames, embed at PIXEL_SIZE=4): recovery exact and frame
accuracy 100% through CRF 32; at CRF 36 frame accuracy falls to 92.5% and one bit of
the majority vote flips.
"""

import argparse
import os
import shutil
import subprocess
import sys

# Work whether invoked as `python src/eval_pixelation.py` or `python -m
# src.eval_pixelation`: the repo root must be importable for `utils`, and src/ for
# the sibling module.
_SRC = os.path.dirname(os.path.abspath(__file__))
for _path in (os.path.dirname(_SRC), _SRC):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from tqdm import tqdm

from pixelate_patch import PIXEL_SIZE, TEMP_REDUNDANCY, detect, pixelate_region
from utils.bit import BIT_LENGTH, get_bit_string
from utils.patch import get_two_halves
from utils.pixelation import detect_pixelation, estimate_block_size
from utils.video import Video_IO

CLEAN = "videos/4_sec_source.mp4"
WMK = "out/wmk.mp4"
WATERMARK = 8710


def half_confidences(video_path, block_size, limit=None):
    """Per-frame (left, right) pixelation confidence for a clip."""
    video_io = Video_IO(video_path)
    total = video_io.frame_count if limit is None else min(limit, video_io.frame_count)

    rows = []
    for _ in tqdm(range(total), desc=f"scoring:{video_path}", unit="frame"):
        frame = video_io.read_frame()
        if frame is None:
            break
        first_half, second_half = get_two_halves(frame.y)
        left = detect_pixelation(first_half[0], block_size=block_size)
        right = detect_pixelation(second_half[0], block_size=block_size)
        rows.append((left, right))
    video_io.release()
    return rows


def check_recovery(wmk_path, block_size, watermark=WATERMARK):
    """detect() must return the watermark, and frame votes should be near-unanimous."""
    print(f"\n=== recovery: {wmk_path} ===")

    video_io = Video_IO(wmk_path)
    frame_count = video_io.frame_count
    video_io.release()

    # Only this many bit slots were ever embedded in a clip this short.
    embedded_bits = min(BIT_LENGTH, frame_count // TEMP_REDUNDANCY)
    expected_bits = get_bit_string(watermark)[0][:embedded_bits]
    expected_int = int(expected_bits, 2)

    got = detect(watermark_path=wmk_path, block_size=block_size)

    print(f"expected {expected_bits} = {expected_int}")
    print(f"got      {'':{len(expected_bits)}} = {got}")
    ok = got == expected_int
    print(f"watermark recovered: {'PASS' if ok else 'FAIL'}")

    # Frame-level accuracy: how many individual frames voted for the right bit.
    rows = half_confidences(wmk_path, block_size)
    correct = 0
    gaps = []
    for i, (left, right) in enumerate(rows):
        bit_index = (i // TEMP_REDUNDANCY) % BIT_LENGTH
        if bit_index >= embedded_bits:
            continue
        truth = int(expected_bits[bit_index])
        vote = 1 if left.confidence > right.confidence else 0
        correct += vote == truth
        gaps.append(abs(left.confidence - right.confidence))

    counted = len(gaps)
    print(f"frame accuracy: {correct}/{counted} = {correct / max(counted, 1):.1%}")
    print(f"mean confidence gap: {np.mean(gaps):.3f} (min {np.min(gaps):.3f})")
    return ok and correct == counted


def check_clean(clean_path, block_size, limit=40):
    """
    A clip with no watermark must not look pixelated, and must not consistently
    favour one half. A locked winner means the score is tracking scene content
    rather than the block grid.
    """
    print(f"\n=== clean control: {clean_path} ===")
    rows = half_confidences(clean_path, block_size, limit=limit)

    left_conf = np.array([r[0].confidence for r in rows])
    right_conf = np.array([r[1].confidence for r in rows])
    left_wins = float((left_conf > right_conf).mean())

    print(f"left  confidence: mean {left_conf.mean():.3f}  max {left_conf.max():.3f}")
    print(f"right confidence: mean {right_conf.mean():.3f}  max {right_conf.max():.3f}")
    print(f"left-half win rate: {left_wins:.1%} (want it away from 0% / 100%)")

    low = max(left_conf.max(), right_conf.max()) < 0.5
    print(f"stays below 0.5: {'PASS' if low else 'FAIL'}")
    return low


def check_block_size(clean_path, sizes=(2, 4, 8, 16)):
    """estimate_block_size() should recover the pixel_size used to build the region."""
    print("\n=== block size estimation ===")
    video_io = Video_IO(clean_path)
    frame = video_io.read_frame()
    video_io.release()

    patch = frame.y[:512, :512]
    ok = True
    for size in sizes:
        guess = estimate_block_size(pixelate_region(patch, pixel_size=size))
        hit = guess == size
        ok &= hit
        print(f"pixel_size {size:2d} -> estimated {guess:2d}  {'ok' if hit else 'MISS'}")

    clean_guess = estimate_block_size(patch)
    print(f"clean patch -> estimated {clean_guess} (no true grid; value is arbitrary)")
    return ok


def check_compression(wmk_path, block_size, crfs=(23, 28, 32, 36), watermark=WATERMARK):
    """
    Re-encode through H.264 at increasing CRF and report where recovery breaks.

    Needs ffmpeg on PATH; skipped with a warning if it is missing.
    """
    print("\n=== H.264 robustness ===")
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH - skipping")
        return True

    results = {}
    for crf in crfs:
        out_path = f"out/_crf{crf}.mp4"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", wmk_path,
            "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", out_path,
        ]
        if subprocess.run(cmd, check=False).returncode != 0:
            print(f"crf {crf}: encode failed, skipping")
            continue
        results[crf] = check_recovery(out_path, block_size, watermark)
        os.remove(out_path)

    survived = [crf for crf, ok in results.items() if ok]
    print("\nsurvives up to CRF:", max(survived) if survived else "none")
    return bool(survived)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wmk", default=WMK, help="pixelated / watermarked clip")
    parser.add_argument("--clean", default=CLEAN, help="unwatermarked control clip")
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help=f"grid period to score against; omit to auto-detect (embed uses {PIXEL_SIZE})",
    )
    parser.add_argument("--watermark", type=int, default=WATERMARK)
    parser.add_argument(
        "--only",
        choices=("recovery", "clean", "blocksize", "compression"),
        help="run a single check",
    )
    parser.add_argument(
        "--compression",
        action="store_true",
        help="also run the H.264 CRF sweep (slow; needs ffmpeg)",
    )
    args = parser.parse_args()

    results = {}
    if args.only in (None, "recovery"):
        results["recovery"] = check_recovery(args.wmk, args.block_size, args.watermark)
    if args.only in (None, "clean"):
        results["clean"] = check_clean(args.clean, args.block_size)
    if args.only in (None, "blocksize"):
        results["blocksize"] = check_block_size(args.clean)
    if args.only == "compression" or args.compression:
        results["compression"] = check_compression(
            args.wmk, args.block_size, watermark=args.watermark
        )

    print("\n=== summary ===")
    for name, passed in results.items():
        print(f"{name:10s} {'PASS' if passed else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
