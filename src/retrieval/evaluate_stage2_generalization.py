"""
Experiment 1, stage 2c: does the Stage-2b `combined` score's advantage over raw
RANSAC inlier count generalize across content, or is it specific to the
near-static `4_sec_source.mp4` clip everything so far was measured on?

Architecture and scoring are UNCHANGED from evaluate_stage2_scores.py:

  SSCD -> top-K -> DISK (leak once, reference cached once) -> LightGlue -> RANSAC
  raw_inliers / geometric_residual / combined, exactly as defined there

This script only adds a loop over multiple reference videos of differing motion
character, plus a diagnostic (near-duplicate distance distribution) computed
strictly after Stage 2 has already picked a candidate -- frame index is
reporting metadata only and never touches retrieval or ranking.

Content categorisation is NOT done by filename. Each clip's motion was measured
(mean per-frame luma difference on 64x64 downsampled frames, and its
coefficient of variation, matching build_gt.py's existing activity-signal
convention) over several candidate windows before picking the one used here --
see STAGE2_GENERALIZATION_REPORT.md for the numbers that justified each choice.
`describe_motion()` below recomputes the same diagnostic on the actual loaded
frames so the report's numbers are reproducible from this script, not from
throwaway analysis.
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
from retrieval.evaluate import load_frames
from retrieval.evaluate_stage2_scores import SCORE_METHODS, pick, scores_from_raw
from retrieval.index import build_index
from retrieval.lightglue_verify import Stage2Verifier

# The three scores implement_3.md asks the report to focus on; all of
# SCORE_METHODS is still computed (cheap, shares the same raw match data) and
# written to the JSON, since retaining the rest is "essentially free."
REPORT_SCORES = ["raw_inliers", "geometric_residual", "combined"]

DEFAULT_VIDEOS = [
    ("4_sec_source.mp4", "near_static"),
    ("moderate_bigbuck.mkv", "moderate"),
    ("high_f1.mkv", "high_motion"),
    ("camera_tokyo.mkv", "camera_motion"),
]

NEAR_DUP_BUCKETS = ["exact", "pm1", "pm2_5", "pm6_10", "gt10"]


def describe_motion(frames: list[np.ndarray], small: int = 64) -> dict:
    """Mean per-frame luma difference and its coefficient of variation."""
    diffs = []
    prev = None
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (small, small), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(g - prev))))
        prev = g
    diffs = np.asarray(diffs)
    return {
        "mean_abs_diff": float(diffs.mean()),
        "cv": float(diffs.std() / (diffs.mean() + 1e-6)),
        "p90_diff": float(np.percentile(diffs, 90)),
    }


def _bucket(d: int) -> str:
    if d == 0:
        return "exact"
    if d == 1:
        return "pm1"
    if d <= 5:
        return "pm2_5"
    if d <= 10:
        return "pm6_10"
    return "gt10"


def near_duplicate_distribution(stage2_pred: np.ndarray, truth: np.ndarray,
                                retrieval_hit: np.ndarray) -> dict:
    """
    Among queries where the true frame WAS in the shortlist, how far (in
    reference-list position) is Stage 2's pick from the truth? Diagnostic only
    -- frame index is never used for ranking, only computed here after the
    fact from an already-decided prediction.
    """
    mask = retrieval_hit
    n = int(mask.sum())
    counts = {b: 0 for b in NEAR_DUP_BUCKETS}
    if n == 0:
        return {"n": 0, "counts": counts, "fractions": {b: 0.0 for b in NEAR_DUP_BUCKETS}}

    d = np.abs(stage2_pred[mask].astype(int) - truth[mask].astype(int))
    for dd in d:
        counts[_bucket(int(dd))] += 1
    return {"n": n, "counts": counts, "fractions": {b: counts[b] / n for b in NEAR_DUP_BUCKETS}}


def run_one_video(video: str, category: str, conditions: list[str], ks: list[int],
                  encoder_name: str, device: str, backend: str, max_keypoints: int,
                  min_matches: int, seed: int) -> dict:
    frames, positions = load_frames(video)
    positions = np.asarray(positions)
    n = len(frames)
    h, w = frames[0].shape[:2]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    motion = describe_motion(frames)

    print(f"\n=== {video} ({category}) ===")
    print(f"  {n} frames at {w}x{h}, {fps:.2f}fps, duration {n / fps:.2f}s "
          f"(fps/duration are metadata only, never used for retrieval)")
    print(f"  motion: mean_abs_diff={motion['mean_abs_diff']:.2f} cv={motion['cv']:.2f} "
          f"p90={motion['p90_diff']:.2f}")

    enc = build_encoder(encoder_name, device)
    ref_emb = enc.encode(frames)
    index = build_index(ref_emb, backend, device)
    truth = np.arange(n)

    t0 = time.time()
    verifier = Stage2Verifier(frames, device, max_keypoints=max_keypoints)
    print(f"  DISK cache for {n} reference frames built in {time.time() - t0:.1f}s")

    max_k = max(ks)
    method_names = list(SCORE_METHODS)
    cond_results = {}

    for cond in conditions:
        sim = CaptureSim(get_condition(cond), (w, h), seed=seed)
        queries = [sim.apply(f) for f in frames]

        q_emb = enc.encode(queries)
        _, idx_topk = index.search(q_emb, max_k)

        t0 = time.time()
        scores = {name: np.zeros((n, max_k), dtype=np.float64) for name in method_names}
        n_failed_pairs = 0
        for i in range(n):
            raw_list = verifier.verify_frame_raw(queries[i], idx_topk[i])
            method_scores, failed = scores_from_raw(raw_list, min_matches)
            n_failed_pairs += int(failed.sum())
            for name in method_names:
                scores[name][i] = method_scores[name]
        t_verify = time.time() - t0

        cr = {
            "stage1_top1": float((idx_topk[:, 0] == truth).mean()),
            "ms_per_query_verify": t_verify / n * 1000,
            "zero_match_or_failed_homography": {
                "count": n_failed_pairs, "of_pairs": int(n * max_k),
                "fraction": n_failed_pairs / (n * max_k),
            },
            "by_k": {}, "near_duplicate": {}, "per_query_max_k": {},
        }

        for K in ks:
            sub_idx = idx_topk[:, :K]
            retrieval_hit = (sub_idx == truth[:, None]).any(1)

            by_score = {}
            preds_at_k = {}
            for name in method_names:
                direction, _ = SCORE_METHODS[name]
                best_pos = pick(scores[name][:, :K], direction)
                stage2_pred = sub_idx[np.arange(n), best_pos]
                stage2_correct = stage2_pred == truth
                preds_at_k[name] = stage2_pred

                by_score[name] = {
                    "stage1_recall_at_k": float(retrieval_hit.mean()),
                    "stage2_top1": float(stage2_correct.mean()),
                    "retrieval_failure_rate": float((~retrieval_hit).mean()),
                    "verification_failure_rate": float((retrieval_hit & ~stage2_correct).mean()),
                }

            baseline_top1 = by_score["raw_inliers"]["stage2_top1"]
            for name in method_names:
                by_score[name]["improvement_over_baseline"] = (
                    by_score[name]["stage2_top1"] - baseline_top1)

            cr["by_k"][str(K)] = by_score
            cr["near_duplicate"][str(K)] = {
                name: near_duplicate_distribution(preds_at_k[name], truth, retrieval_hit)
                for name in method_names
            }
            if K == max_k:
                cr["per_query_max_k"] = {
                    "truth": truth.tolist(),
                    **{f"{name}_pred": preds_at_k[name].tolist() for name in REPORT_SCORES},
                }

        cond_results[cond] = cr
        zf = cr["zero_match_or_failed_homography"]
        print(f"  {cond:10s} stage1_top1={cr['stage1_top1']:.3f} "
              f"zero/failed={zf['count']}/{zf['of_pairs']} ({zf['fraction']:.3f}) "
              f"({t_verify/n*1000:.0f} ms/query)")
        for K in ks:
            for name in REPORT_SCORES:
                m = cr["by_k"][str(K)][name]
                print(f"    K={K:<3d} {name:18s} recall@K={m['stage1_recall_at_k']:.3f} "
                      f"stage2_top1={m['stage2_top1']:.3f} "
                      f"retr_fail={m['retrieval_failure_rate']:.3f} "
                      f"verify_fail={m['verification_failure_rate']:.3f} "
                      f"vs_raw={m['improvement_over_baseline']:+.3f}")

    return {
        "path": video, "category": category, "width": w, "height": h, "fps": fps,
        "n_frames": n, "duration_sec": n / fps, "motion": motion,
        "conditions": cond_results,
    }


def run(videos: list[tuple[str, str]], conditions: list[str], ks: list[int],
        encoder_name: str = "sscd", device: str = "cuda", backend: str = "torch",
        max_keypoints: int = 1024, min_matches: int = 10, seed: int = 0) -> dict:
    per_video = {}
    for video, category in videos:
        per_video[video] = run_one_video(video, category, conditions, ks, encoder_name,
                                         device, backend, max_keypoints, min_matches, seed)

    return {
        "ks": ks, "conditions": conditions, "min_matches": min_matches,
        "score_methods": {name: {"direction": d} for name, (d, _) in SCORE_METHODS.items()},
        "report_scores": REPORT_SCORES,
        "videos": per_video,
    }


def print_summary(result: dict):
    print("\n" + "=" * 130)
    print("PER-VIDEO SUMMARY")
    print("=" * 130)
    print(f"{'video':22s} {'category':14s} {'condition':10s} {'K':>4s} {'score':18s} "
          f"{'recall@K':>9s} {'stage2_top1':>12s} {'retr_fail':>10s} {'verify_fail':>11s} {'vs_raw':>8s}")
    print("-" * 130)
    for video, v in result["videos"].items():
        name = os.path.basename(video)
        for cond, cr in v["conditions"].items():
            for K in result["ks"]:
                for score in result["report_scores"]:
                    m = cr["by_k"][str(K)][score]
                    print(f"{name:22s} {v['category']:14s} {cond:10s} {K:>4d} {score:18s} "
                          f"{m['stage1_recall_at_k']:>9.3f} {m['stage2_top1']:>12.3f} "
                          f"{m['retrieval_failure_rate']:>10.3f} {m['verification_failure_rate']:>11.3f} "
                          f"{m['improvement_over_baseline']:>+8.3f}")

    print("\n" + "=" * 90)
    print("CATEGORY AGGREGATE (mean stage2_top1 over videos in that category --")
    print("one video per category in this run, so this equals the per-video number)")
    print("=" * 90)
    print(f"{'category':14s} {'condition':10s} {'K':>4s} {'score':18s} {'stage2_top1':>12s}")
    print("-" * 90)
    by_cat = {}
    for video, v in result["videos"].items():
        by_cat.setdefault(v["category"], []).append(v)
    for cat, vids in by_cat.items():
        for cond in result["conditions"]:
            for K in result["ks"]:
                for score in result["report_scores"]:
                    vals = [v["conditions"][cond]["by_k"][str(K)][score]["stage2_top1"] for v in vids]
                    print(f"{cat:14s} {cond:10s} {K:>4d} {score:18s} {np.mean(vals):>12.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", nargs="+", default=None,
                    help="path:category pairs, default: the four generalization clips")
    ap.add_argument("--conditions", nargs="+", default=["mild", "moderate"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    ap.add_argument("--encoder", default="sscd")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="torch", choices=["torch", "faiss"])
    ap.add_argument("--max-keypoints", type=int, default=1024)
    ap.add_argument("--min-matches", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/retrieval/stage2_generalization_results.json")
    args = ap.parse_args()

    if args.videos:
        videos = [tuple(v.split(":", 1)) for v in args.videos]
    else:
        videos = DEFAULT_VIDEOS

    result = run(videos, args.conditions, args.ks, args.encoder, args.device,
                 args.backend, args.max_keypoints, args.min_matches, args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}")

    print_summary(result)


if __name__ == "__main__":
    main()
