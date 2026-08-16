"""
Ground truth for a real camera capture, by rectify-then-correlate.

Why not the motion profile (see build_gt.py): that needs the original to have a
distinctive temporal activity signature. On smooth animated content with no cuts
it is flat, and the capture's own handheld shake dominates the leak's signal, so
the two correlate at chance. Measured on 4_sec_source/leak: original cv=0.10,
leak cv=0.69, identity-map correlation -0.11.

This works on image content instead:

  1. Average all frames of each video into a "plate". The camera is
     approximately fixed, so the plate is a sharp picture of the static scene
     while moving foreground averages out. Critically, the plate does not depend
     on frame ORDER, so the homography it yields cannot encode the timing
     answer -- which is what keeps this oracle non-circular.
  2. Match plate_leak -> plate_orig with DISK+LightGlue to get one homography.
  3. Warp every leak frame into the original's coordinate frame with it.
  4. Score each rectified leak frame against every original frame with ZNCC on
     high-pass luma. High-pass because the capture's exposure and white balance
     are wrong by a slowly varying gain, which ZNCC only partly removes.

The result is a full n_leak x n_orig score matrix. Ground truth is its per-row
argmax, and the fit is trustworthy only if that argmax is close to monotonic --
which is checked and reported rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


def read_gray(path: str, max_frames: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    out = []
    while max_frames is None or len(out) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return out


def plate(frames: list[np.ndarray]) -> np.ndarray:
    """Order-independent static-scene estimate."""
    acc = np.zeros_like(frames[0], dtype=np.float64)
    for f in frames:
        acc += f
    acc /= len(frames)
    # Stretch to full range; the leak plate is typically much darker than the
    # original's and the feature detector responds better to matched contrast.
    acc = (acc - acc.min()) / max(acc.max() - acc.min(), 1e-6)
    return (acc * 255).astype(np.uint8)


def highpass(img: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    x = img.astype(np.float32)
    return x - cv2.GaussianBlur(x, (0, 0), sigma)


def zncc_matrix(leak_rect: np.ndarray, orig: np.ndarray,
                mask: np.ndarray) -> np.ndarray:
    """
    leak_rect: (T, H, W) float32, orig: (N, H, W) float32, mask: (H, W) bool.

    Reduces to a single matmul of unit-normalised, mean-removed vectors over the
    valid region, so the whole T x N matrix costs one GEMM.
    """
    def flatten_norm(stack):
        v = stack[:, mask]
        v = v - v.mean(axis=1, keepdims=True)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-6)

    return flatten_norm(leak_rect) @ flatten_norm(orig).T


def build(orig_path: str, leak_path: str, work_w: int = 640) -> dict:
    from utils.lightglue import LightGluePatchMatcher

    orig_frames = read_gray(orig_path)
    leak_frames = read_gray(leak_path)
    print(f"orig {len(orig_frames)} frames {orig_frames[0].shape}, "
          f"leak {len(leak_frames)} frames {leak_frames[0].shape}")

    p_orig, p_leak = plate(orig_frames), plate(leak_frames)
    os.makedirs("out/retrieval", exist_ok=True)
    cv2.imwrite("out/retrieval/plate_orig.png", p_orig)
    cv2.imwrite("out/retrieval/plate_leak.png", p_leak)

    matcher = LightGluePatchMatcher()
    H = matcher.compute_homography(p_leak, p_orig)
    if H is None:
        raise RuntimeError(
            "LightGlue could not match the two plates. The capture is probably "
            "too degraded, or the two videos do not show the same content.")
    print("plate homography:\n", np.array2string(H, precision=4))

    oh, ow = orig_frames[0].shape
    scale = work_w / ow
    wh, ww = int(oh * scale), work_w
    S = np.diag([scale, scale, 1.0]).astype(np.float32)
    Hs = S @ H

    # Where the warped leak actually has content; everything else is border fill
    # and would otherwise contribute a constant to every correlation.
    cover = cv2.warpPerspective(np.full(leak_frames[0].shape, 255, np.uint8),
                                Hs, (ww, wh))
    valid = cover > 250
    print(f"valid overlap: {valid.mean():.1%} of the frame")
    if valid.mean() < 0.15:
        raise RuntimeError("overlap too small to score reliably")

    rect = np.stack([
        highpass(cv2.warpPerspective(f, Hs, (ww, wh))) for f in leak_frames
    ]).astype(np.float32)
    orig = np.stack([
        highpass(cv2.resize(f, (ww, wh), interpolation=cv2.INTER_AREA))
        for f in orig_frames
    ]).astype(np.float32)

    M = zncc_matrix(rect, orig, valid)
    best = M.argmax(axis=1)
    peak = M.max(axis=1)

    # Trust check: a correct map is monotone non-decreasing in leak index.
    mono = float(np.mean(np.diff(best) >= 0))
    # And the peak should stand out from the row's typical score.
    margin = float(np.mean(peak - M.mean(axis=1)))

    return {
        "orig_path": orig_path, "leak_path": leak_path,
        "n_orig": len(orig_frames), "n_leak": len(leak_frames),
        "homography": H.tolist(),
        "valid_fraction": float(valid.mean()),
        "monotonicity": mono,
        "mean_peak": float(peak.mean()),
        "mean_peak_margin": margin,
        "pairs": [[int(i), int(k)] for i, k in enumerate(best)],
        "peak_scores": [float(p) for p in peak],
        "score_matrix_shape": list(M.shape),
    }, M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orig", default="4_sec_source.mp4")
    ap.add_argument("--leak", default="leak.mp4")
    ap.add_argument("--out", default="out/retrieval/gt_real.json")
    ap.add_argument("--save-matrix", default="out/retrieval/gt_zncc.npy")
    args = ap.parse_args()

    gt, M = build(args.orig, args.leak)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(gt, f, indent=2)
    np.save(args.save_matrix, M)

    print(f"\nmean peak ZNCC : {gt['mean_peak']:.4f}")
    print(f"peak margin    : {gt['mean_peak_margin']:+.4f}")
    print(f"monotonicity   : {gt['monotonicity']:.1%} of steps non-decreasing")
    print(f"first 20 pairs : {gt['pairs'][:20]}")
    print(f"wrote {args.out} and {args.save_matrix}")

    if gt["monotonicity"] < 0.85:
        print("\nWARNING: the recovered map is not close to monotone. Inspect "
              f"{args.save_matrix} before using this as ground truth.")


if __name__ == "__main__":
    main()
