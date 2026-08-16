"""
Recover the frame correspondence between an original video and a real camera
capture of it, without using any of the embeddings under test.

The signal is per-frame temporal activity -- mean |frame_t - frame_{t-1}| on a
heavily downsampled luma plane. That survives everything a camcord does to the
image (perspective, exposure, blur, moire) because it is a scalar per frame, and
it spikes on cuts and fast motion, which is exactly the structure needed to lock
two clocks together.

Deliberately model-free: scoring learned embeddings against ground truth that
was itself built from learned embeddings would make any result unfalsifiable.

The time map is affine, k_orig = round(a * t_leak + b), because both clocks are
constant-rate. It is fitted by brute force over (a, b) and reported with a
correlation score so a bad fit is visible rather than silent.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np


def activity_signal(path: str, small: int = 64) -> np.ndarray:
    """Per-frame temporal activity. Index i is the activity between frame i-1 and i."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")

    sig, prev = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (small, small), interpolation=cv2.INTER_AREA).astype(np.float32)
        # Per-frame mean removal makes the signal immune to the capture's
        # exposure drift and auto-gain, which otherwise dominate the difference.
        g -= g.mean()
        sig.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    cap.release()

    s = np.asarray(sig, dtype=np.float32)
    if len(s) > 1:
        s[0] = s[1]
    return s


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-8 else 1.0)


def fit_affine_map(sig_leak: np.ndarray, sig_orig: np.ndarray,
                   rate_range: tuple[float, float] = (0.8, 1.25),
                   rate_steps: int = 181) -> dict:
    """
    Find (a, b) maximising correlation between sig_leak[t] and sig_orig[a*t + b].

    Returns the fit plus the score, and the margin over the runner-up rate, which
    is the honest indicator of whether the alignment is trustworthy.
    """
    zl, zo = _z(sig_leak), _z(sig_orig)
    n_leak, n_orig = len(zl), len(zo)
    t = np.arange(n_leak, dtype=np.float64)

    results = []
    for a in np.linspace(*rate_range, rate_steps):
        # Offsets that keep a usable overlap between the two clips.
        lo, hi = -a * n_leak + 8, n_orig - 8
        for b in np.arange(lo, hi, 0.5):
            k = a * t + b
            valid = (k >= 0) & (k <= n_orig - 1)
            if valid.sum() < max(16, 0.3 * n_leak):
                continue
            # Linear interpolation: the true correspondence is generally
            # fractional, and rounding here would quantise away the rate signal.
            resampled = np.interp(k[valid], np.arange(n_orig), zo)
            v = zl[valid]
            if resampled.std() < 1e-8 or v.std() < 1e-8:
                continue
            score = float(np.corrcoef(v, resampled)[0, 1])
            results.append((score, float(a), float(b), int(valid.sum())))

    if not results:
        raise RuntimeError("no valid alignment found")

    results.sort(reverse=True)
    score, a, b, n = results[0]

    # Best score achieved at a materially different rate, as a confidence check.
    runner = next((s for s, aa, _, _ in results if abs(aa - a) > 0.02), float("nan"))

    return {"rate": a, "offset": b, "score": score, "overlap": n,
            "runner_up_score": runner, "margin": score - runner}


def build_ground_truth(orig_path: str, leak_path: str) -> dict:
    sig_o = activity_signal(orig_path)
    sig_l = activity_signal(leak_path)
    fit = fit_affine_map(sig_l, sig_o)

    t = np.arange(len(sig_l), dtype=np.float64)
    k = fit["rate"] * t + fit["offset"]
    valid = (k >= 0) & (k <= len(sig_o) - 1)

    return {
        "orig_path": orig_path,
        "leak_path": leak_path,
        "n_orig": len(sig_o),
        "n_leak": len(sig_l),
        **fit,
        # leak frame index -> original frame index, only where the map is in range.
        "pairs": [[int(i), int(round(k[i]))] for i in np.nonzero(valid)[0]],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orig", default="4_sec_source.mp4")
    ap.add_argument("--leak", default="leak.mp4")
    ap.add_argument("--out", default="out/retrieval/gt_real.json")
    args = ap.parse_args()

    gt = build_ground_truth(args.orig, args.leak)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"orig frames : {gt['n_orig']}")
    print(f"leak frames : {gt['n_leak']}")
    print(f"time map    : k = {gt['rate']:.5f} * t + {gt['offset']:.3f}")
    print(f"correlation : {gt['score']:.4f}  (runner-up rate {gt['runner_up_score']:.4f},"
          f" margin {gt['margin']:+.4f})")
    print(f"pairs       : {len(gt['pairs'])}")
    print(f"wrote       : {args.out}")

    if gt["score"] < 0.5:
        print("\nWARNING: weak correlation -- do not trust this map. Verify the pair "
              "manually (e.g. LightGlue inlier counts on a few frames) before using it.")


if __name__ == "__main__":
    main()
