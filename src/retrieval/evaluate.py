"""
Experiment 1: can a pretrained frame embedding alone retrieve the correct
original frame for a camera-captured query?

  original frames -> encoder -> index
  degraded query  -> encoder -> nearest neighbours -> compare against known k

No temporal tracking, no fps assumption, no sliding window, no DTW. Every query
is scored independently, which is the point: this measures the representation,
not a temporal prior that could mask a weak representation.

Ground truth is exact because queries are generated from known original frames.

Beyond the headline accuracy numbers this reports a CONFUSION RADIUS, which is
the diagnostic that actually explains a result. Define

  S_self(d)  = mean cosine between original frames k and k+d
  S_true     = mean cosine between a degraded query and its own original
  d*         = smallest d where S_self(d) <= S_true

d* is how far away an undistorted frame can sit and still look as similar as the
true match does after degradation. It is a property of encoder AND content
together, and it upper-bounds achievable precision: if d* is 20, no amount of
index tuning will get top-1 right, because 40 rival frames are genuinely closer
in the metric than the answer. Reporting only accuracy hides whether a failure
is the encoder's fault or the content's.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from retrieval.capture_sim import CaptureSim, get_condition
from retrieval.encoders import build_encoder
from retrieval.index import build_index


def load_frames(path: str, max_frames: int | None = None,
                stride: int = 1) -> tuple[list[np.ndarray], list[int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    frames, indices, i = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(f)
            indices.append(i)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames, indices


def metrics(pred_idx: np.ndarray, truth: np.ndarray,
            ref_positions: np.ndarray) -> dict:
    """
    pred_idx: (n_query, k) index positions into the reference set, best first.
    truth:    (n_query,) the correct reference POSITION for each query.
    ref_positions: (n_ref,) original frame numbers, for temporal error in frames.
    """
    top1 = pred_idx[:, 0]
    hit = pred_idx == truth[:, None]

    # Temporal error is measured in source frame numbers, so a strided index
    # does not silently rescale the error.
    err = np.abs(ref_positions[top1] - ref_positions[truth])

    # Rank of the true frame; inf when it never appears in the retrieved list.
    rank = np.where(hit.any(1), hit.argmax(1) + 1, np.inf)

    out = {
        "n_query": int(len(truth)),
        "top1": float(hit[:, 0].mean()),
        "top5": float(hit[:, :5].any(1).mean()),
        "top10": float(hit[:, :10].any(1).mean()),
        "mean_abs_err": float(err.mean()),
        "median_abs_err": float(np.median(err)),
        "p90_abs_err": float(np.percentile(err, 90)),
        "median_true_rank": float(np.median(rank)),
    }
    for tol in (1, 2, 5, 10):
        out[f"recall_pm{tol}"] = float((err <= tol).mean())
    return out


def confusion_radius(ref_emb: np.ndarray, query_emb: np.ndarray,
                     truth: np.ndarray, max_d: int = 60) -> dict:
    """Self-similarity decay of the reference set vs the degraded-to-true score."""
    n = len(ref_emb)
    s_self = {}
    for d in range(1, min(max_d, n - 1) + 1):
        s_self[d] = float(np.mean(np.sum(ref_emb[:n - d] * ref_emb[d:], axis=1)))

    s_true = float(np.mean(np.sum(query_emb * ref_emb[truth], axis=1)))

    d_star = next((d for d in sorted(s_self) if s_self[d] <= s_true), None)
    return {
        "s_true": s_true,
        "s_self": {str(d): s_self[d] for d in (1, 2, 5, 10, 20, 40) if d in s_self},
        "d_star": d_star if d_star is not None else -1,
    }


def run(video: str, encoder_name: str, conditions: list[str], tile: int = 1,
        stride: int = 1, max_frames: int | None = None, topk: int = 10,
        backend: str = "torch", device: str = "cuda", seed: int = 0) -> dict:
    frames, positions = load_frames(video, max_frames, stride)
    positions = np.asarray(positions)
    n = len(frames)
    h, w = frames[0].shape[:2]
    print(f"{video}: {n} reference frames at {w}x{h} (stride {stride})")

    enc = build_encoder(encoder_name, device)
    print(f"encoder {enc.name}: input {enc.input_size}px, dim {enc.dim}"
          f"{f' x {tile*tile} tiles' if tile > 1 else ''}")

    t0 = time.time()
    ref_emb = enc.encode(frames, tile=tile)
    t_index = time.time() - t0
    print(f"indexed {n} frames in {t_index:.1f}s ({t_index/n*1000:.1f} ms/frame)")

    index = build_index(ref_emb, backend, device)
    truth = np.arange(n)

    results = {}
    for cond in conditions:
        sim = CaptureSim(get_condition(cond), (w, h), seed=seed)
        queries = [sim.apply(f) for f in frames]

        t0 = time.time()
        q_emb = enc.encode(queries, tile=tile)
        t_enc = time.time() - t0

        t0 = time.time()
        _, idx = index.search(q_emb, topk)
        t_search = time.time() - t0

        m = metrics(idx, truth, positions)
        m.update(confusion_radius(ref_emb, q_emb, truth))
        m["ms_per_query_encode"] = t_enc / n * 1000
        m["ms_per_query_search"] = t_search / n * 1000
        results[cond] = m

        print(f"  {cond:17s} top1={m['top1']:.3f} top5={m['top5']:.3f} "
              f"top10={m['top10']:.3f} medErr={m['median_abs_err']:5.1f} "
              f"R+-1={m['recall_pm1']:.3f} R+-10={m['recall_pm10']:.3f} "
              f"d*={m['d_star']}")

    return {
        "video": video, "encoder": enc.name, "dim": int(ref_emb.shape[1]),
        "input_size": enc.input_size, "tile": tile, "stride": stride,
        "n_ref": n, "index_seconds": t_index, "backend": backend,
        "conditions": results,
    }


ALL_CONDITIONS = ["clean", "mild", "moderate", "severe", "extreme",
                  "perspective_only", "blur_only", "exposure_only",
                  "compression_only", "moire_only", "noise_only"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="4_sec_source.mp4")
    ap.add_argument("--encoders", nargs="+",
                    default=["pixel", "sscd", "dinov2_base", "siglip2"])
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS)
    ap.add_argument("--tile", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--backend", default="torch", choices=["torch", "faiss"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/retrieval/results.json")
    args = ap.parse_args()

    all_results = []
    for name in args.encoders:
        print(f"\n=== {name} (tile={args.tile}) ===")
        try:
            all_results.append(run(
                args.video, name, args.conditions, args.tile, args.stride,
                args.max_frames, args.topk, args.backend, args.device, args.seed))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n" + "=" * 100)
    print(f"{'encoder':16s} {'tile':>4s} {'condition':17s} {'top1':>6s} {'top5':>6s} "
          f"{'top10':>6s} {'medErr':>7s} {'R+-1':>6s} {'R+-5':>6s} {'R+-10':>6s} {'d*':>4s}")
    print("=" * 100)
    for r in all_results:
        for cond, m in r["conditions"].items():
            print(f"{r['encoder']:16s} {r['tile']:>4d} {cond:17s} {m['top1']:6.3f} "
                  f"{m['top5']:6.3f} {m['top10']:6.3f} {m['median_abs_err']:7.1f} "
                  f"{m['recall_pm1']:6.3f} {m['recall_pm5']:6.3f} "
                  f"{m['recall_pm10']:6.3f} {m['d_star']:>4d}")


if __name__ == "__main__":
    main()
