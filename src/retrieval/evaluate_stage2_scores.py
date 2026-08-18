"""
Experiment 1, stage 2b: does a more discriminative Stage-2 score beat raw RANSAC
inlier count at picking the exact frame out of an SSCD shortlist?

Architecture is UNCHANGED from evaluate_stage2.py:

  SSCD -> top-K -> DISK (leak once, reference cached once) -> LightGlue -> RANSAC

STAGE2_REPORT.md measured that raw inlier count is too coarse: near-duplicate
reference frames get similar inlier counts, so verification_failure_rate grows
faster than stage1_recall@K as K widens. This script tests four alternative
scores, all derived from the SAME per-(leak, candidate) LightGlue/RANSAC output
that produces the baseline score -- nothing is rerun per score method, and no
temporal prior, perspective correction, or rectification is introduced.

Scores (see SCORE_METHODS below for the exact formulas and worst-case values):
  raw_inliers         -- the existing baseline, unchanged
  inlier_ratio        -- inliers / matches
  match_density       -- inliers normalised by detected-keypoint evidence in
                         both frames (sqrt(n_kpts0 * n_kpts1)), not just matches
  geometric_residual  -- median inlier reprojection error (LOWER is better;
                         every other score is HIGHER is better)
  combined            -- inliers / (1 + median error); no fitted weights

All five are computed from one call to Stage2Verifier.verify_frame_raw() per
leak frame, at max(--ks) candidates; smaller K is a slice of the same ranked
list and the same raw match data, exactly as evaluate_stage2.py does for the
single baseline score.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from retrieval.capture_sim import CaptureSim, get_condition
from retrieval.encoders import build_encoder
from retrieval.evaluate import load_frames
from retrieval.index import build_index
from retrieval.lightglue_verify import Stage2Verifier

# name -> (direction, worst-case score used when the shared evidence gate fails)
SCORE_METHODS = {
    "raw_inliers": ("max", 0.0),
    "inlier_ratio": ("max", 0.0),
    "match_density": ("max", 0.0),
    "geometric_residual": ("min", float("inf")),
    "combined": ("max", 0.0),
}


def scores_from_raw(raw_list: list[dict], min_matches: int) -> tuple[dict, np.ndarray]:
    """
    Derive every SCORE_METHODS score from the same raw match/RANSAC list.

    A candidate is gated out (worst-case score for every method) exactly when
    the shared evidence is insufficient: fewer than `min_matches` LightGlue
    matches, or RANSAC found no homography. This is the one gate all five
    methods share, so they are compared on identical underlying evidence.
    """
    n = len(raw_list)
    out = {name: np.full(n, worst, dtype=np.float64)
           for name, (_, worst) in SCORE_METHODS.items()}
    failed = np.zeros(n, dtype=bool)

    for i, raw in enumerate(raw_list):
        ok = raw["homography_ok"] and raw["num_matches"] >= min_matches
        if not ok:
            failed[i] = True
            continue

        num_inliers, num_matches = raw["num_inliers"], raw["num_matches"]
        errs = raw["inlier_errs"]
        median_err = float(np.median(errs)) if errs.size else float("inf")

        out["raw_inliers"][i] = num_inliers
        out["inlier_ratio"][i] = num_inliers / num_matches if num_matches else 0.0
        out["match_density"][i] = num_inliers / np.sqrt(max(raw["n_kpts0"] * raw["n_kpts1"], 1))
        out["geometric_residual"][i] = median_err
        out["combined"][i] = num_inliers / (1.0 + median_err) if np.isfinite(median_err) else 0.0

    return out, failed


def pick(scores_2d: np.ndarray, direction: str) -> np.ndarray:
    """argmax/argmin over axis 1, per the method's direction."""
    return np.argmax(scores_2d, axis=1) if direction == "max" else np.argmin(scores_2d, axis=1)


def run(video: str, conditions: list[str], ks: list[int], encoder_name: str = "sscd",
        device: str = "cuda", backend: str = "torch", max_keypoints: int = 1024,
        min_matches: int = 10, seed: int = 0) -> dict:
    frames, positions = load_frames(video)
    positions = np.asarray(positions)
    n = len(frames)
    h, w = frames[0].shape[:2]
    print(f"{video}: {n} reference frames at {w}x{h}")

    enc = build_encoder(encoder_name, device)
    ref_emb = enc.encode(frames)
    index = build_index(ref_emb, backend, device)
    truth = np.arange(n)

    print(f"building DISK cache for {n} reference frames (once, reused across conditions)...")
    t0 = time.time()
    verifier = Stage2Verifier(frames, device, max_keypoints=max_keypoints)
    print(f"  done in {time.time() - t0:.1f}s")

    max_k = max(ks)
    method_names = list(SCORE_METHODS)
    results = {}

    for cond in conditions:
        sim = CaptureSim(get_condition(cond), (w, h), seed=seed)
        queries = [sim.apply(f) for f in frames]

        q_emb = enc.encode(queries)
        _, idx_topk = index.search(q_emb, max_k)  # (n, max_k), best-first

        t0 = time.time()
        # scores[method] has shape (n, max_k); one raw pass per leak frame feeds all methods.
        scores = {name: np.zeros((n, max_k), dtype=np.float64) for name in method_names}
        n_failed_pairs = 0
        for i in range(n):
            raw_list = verifier.verify_frame_raw(queries[i], idx_topk[i])
            method_scores, failed = scores_from_raw(raw_list, min_matches)
            n_failed_pairs += int(failed.sum())
            for name in method_names:
                scores[name][i] = method_scores[name]
        t_verify = time.time() - t0

        cond_result = {
            "stage1_top1": float((idx_topk[:, 0] == truth).mean()),
            "ms_per_query_verify": t_verify / n * 1000,
            "zero_match_or_failed_homography": {
                "count": n_failed_pairs, "of_pairs": int(n * max_k),
                "fraction": n_failed_pairs / (n * max_k),
            },
            "by_k": {},
        }

        for K in ks:
            sub_idx = idx_topk[:, :K]
            retrieval_hit = (sub_idx == truth[:, None]).any(1)

            by_score = {}
            for name in method_names:
                direction, _ = SCORE_METHODS[name]
                sub_scores = scores[name][:, :K]
                best_pos = pick(sub_scores, direction)
                stage2_pred = sub_idx[np.arange(n), best_pos]
                stage2_correct = stage2_pred == truth

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

            cond_result["by_k"][str(K)] = by_score

        results[cond] = cond_result
        zf = cond_result["zero_match_or_failed_homography"]
        print(f"  {cond:10s} stage1_top1={cond_result['stage1_top1']:.3f} "
              f"zero/failed={zf['count']}/{zf['of_pairs']} ({zf['fraction']:.3f}) "
              f"({t_verify/n*1000:.0f} ms/query)")
        for K in ks:
            for name in method_names:
                m = cond_result["by_k"][str(K)][name]
                print(f"    K={K:<3d} {name:18s} recall@K={m['stage1_recall_at_k']:.3f} "
                      f"stage2_top1={m['stage2_top1']:.3f} "
                      f"retr_fail={m['retrieval_failure_rate']:.3f} "
                      f"verify_fail={m['verification_failure_rate']:.3f} "
                      f"vs_baseline={m['improvement_over_baseline']:+.3f}")

    return {"video": video, "encoder": enc.name, "n_ref": n, "ks": ks,
            "score_methods": {name: {"direction": d} for name, (d, _) in SCORE_METHODS.items()},
            "min_matches": min_matches, "conditions": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="4_sec_source.mp4")
    ap.add_argument("--conditions", nargs="+", default=["mild", "moderate"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    ap.add_argument("--encoder", default="sscd")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="torch", choices=["torch", "faiss"])
    ap.add_argument("--max-keypoints", type=int, default=1024)
    ap.add_argument("--min-matches", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/retrieval/stage2_score_results.json")
    args = ap.parse_args()

    result = run(args.video, args.conditions, args.ks, args.encoder, args.device,
                 args.backend, args.max_keypoints, args.min_matches, args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n" + "=" * 110)
    print(f"{'condition':10s} {'K':>4s} {'score':18s} {'recall@K':>9s} {'stage2_top1':>12s} "
          f"{'retr_fail':>10s} {'verify_fail':>11s} {'vs_baseline':>12s}")
    print("=" * 110)
    for cond, cr in result["conditions"].items():
        for K in args.ks:
            for name in SCORE_METHODS:
                m = cr["by_k"][str(K)][name]
                print(f"{cond:10s} {K:>4d} {name:18s} {m['stage1_recall_at_k']:>9.3f} "
                      f"{m['stage2_top1']:>12.3f} {m['retrieval_failure_rate']:>10.3f} "
                      f"{m['verification_failure_rate']:>11.3f} "
                      f"{m['improvement_over_baseline']:>+12.3f}")


if __name__ == "__main__":
    main()
