"""
Experiment 1, stage 2: does an SSCD shortlist plus LightGlue geometric
verification pick the exact frame that SSCD alone cannot?

  original frames -> SSCD -> index                 (stage 1, reused from evaluate.py)
  original frames -> DISK -> cached features once  (stage 2, new)

  leak frame -> SSCD -> top-K candidates            (no temporal restriction)
  leak frame -> DISK once -> LightGlue vs each cached candidate -> inlier score
  argmax(score) over the candidates -> predicted original frame

No temporal prior, no perspective-correction preprocessing, no cheap filtering
between the two stages: LightGlue runs against every one of the K SSCD
candidates. Homography is estimated by RANSAC only AFTER matching, purely as
the verification score -- nothing is rectified.

Per implement_1.md section 5, the evaluation explicitly separates:
  retrieval failure    -- the true frame was not in the SSCD top-K at all
  verification failure -- the true frame was in the top-K, but LightGlue chose
                           a different candidate

Scores are computed once at max(--ks) candidates per query and sliced for
smaller K, since a smaller K is just a prefix of the same ranked list -- this
avoids rerunning LightGlue per K without skipping any of the matching implement_1.md
asks for at the largest K.
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


def run(video: str, conditions: list[str], ks: list[int], encoder_name: str = "sscd",
        device: str = "cuda", backend: str = "torch", max_keypoints: int = 1024,
        seed: int = 0) -> dict:
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
    results = {}
    for cond in conditions:
        sim = CaptureSim(get_condition(cond), (w, h), seed=seed)
        queries = [sim.apply(f) for f in frames]

        q_emb = enc.encode(queries)
        _, idx_topk = index.search(q_emb, max_k)  # (n, max_k), best-first

        t0 = time.time()
        scores = np.zeros((n, max_k), dtype=np.float32)
        for i in range(n):
            scores[i] = verifier.verify_frame(queries[i], idx_topk[i])
        t_verify = time.time() - t0

        cond_result = {"stage1_top1": float((idx_topk[:, 0] == truth).mean()),
                       "ms_per_query_verify": t_verify / n * 1000, "by_k": {}}

        for K in ks:
            sub_idx = idx_topk[:, :K]
            sub_scores = scores[:, :K]

            retrieval_hit = (sub_idx == truth[:, None]).any(1)
            best_pos = np.argmax(sub_scores, axis=1)
            stage2_pred = sub_idx[np.arange(n), best_pos]
            stage2_correct = stage2_pred == truth

            retrieval_failure = ~retrieval_hit
            verification_failure = retrieval_hit & ~stage2_correct
            success = retrieval_hit & stage2_correct

            cond_result["by_k"][str(K)] = {
                "stage1_recall_at_k": float(retrieval_hit.mean()),
                "stage2_top1": float(stage2_correct.mean()),
                "retrieval_failure_rate": float(retrieval_failure.mean()),
                "verification_failure_rate": float(verification_failure.mean()),
            }

        results[cond] = cond_result
        print(f"  {cond:10s} stage1_top1={cond_result['stage1_top1']:.3f} "
              f"({t_verify/n*1000:.0f} ms/query verify)")
        for K in ks:
            m = cond_result["by_k"][str(K)]
            print(f"    K={K:<3d} stage1_recall@K={m['stage1_recall_at_k']:.3f} "
                  f"stage2_top1={m['stage2_top1']:.3f} "
                  f"retrieval_fail={m['retrieval_failure_rate']:.3f} "
                  f"verify_fail={m['verification_failure_rate']:.3f}")

    return {"video": video, "encoder": enc.name, "n_ref": n, "ks": ks,
            "conditions": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="4_sec_source.mp4")
    ap.add_argument("--conditions", nargs="+", default=["mild", "moderate", "severe"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    ap.add_argument("--encoder", default="sscd")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="torch", choices=["torch", "faiss"])
    ap.add_argument("--max-keypoints", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/retrieval/stage2_results.json")
    args = ap.parse_args()

    result = run(args.video, args.conditions, args.ks, args.encoder, args.device,
                 args.backend, args.max_keypoints, args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n" + "=" * 100)
    print(f"{'condition':10s} {'K':>4s} {'recall@K':>9s} {'stage2_top1':>12s} "
          f"{'retr_fail':>10s} {'verify_fail':>11s} {'stage1_top1':>12s}")
    print("=" * 100)
    for cond, cr in result["conditions"].items():
        for K in args.ks:
            m = cr["by_k"][str(K)]
            print(f"{cond:10s} {K:>4d} {m['stage1_recall_at_k']:>9.3f} "
                  f"{m['stage2_top1']:>12.3f} {m['retrieval_failure_rate']:>10.3f} "
                  f"{m['verification_failure_rate']:>11.3f} {cr['stage1_top1']:>12.3f}")


if __name__ == "__main__":
    main()
