# Stage 2b: Comparing Stage-2 Verification Scores

**Experiment 1, continued — same architecture as `STAGE2_REPORT.md`, different Stage-2 scoring function.**

| | |
|---|---|
| Corpus | `4_sec_source.mp4` · 120 frames · 1920×1080 · 30fps |
| Hardware | Tesla T4 |
| Architecture | `SSCD → top-K → DISK → LightGlue → RANSAC` — **unchanged** from `STAGE2_REPORT.md` |
| What changed | Only the scalar score used to rank candidates within the shortlist |
| Conditions tested | mild, moderate |
| K values | 1, 5, 10, 20, 50 |
| Temporal prior | None at any stage |

---

## The question

`STAGE2_REPORT.md` showed the two-stage architecture beats SSCD alone, but that its score — raw
RANSAC inlier count — is too coarse: near-duplicate reference frames get similar inlier counts, so
`verification_failure_rate` grows faster than `stage1_recall@K` as K widens (at K=50/mild, the true
frame was in the shortlist 92.5% of the time but the wrong candidate won 60% of the time).

`implement_2.md` asked for a measurement, not a redesign: keep the architecture exactly as-is, and
test whether a more discriminative *scoring function*, computed from the same LightGlue/RANSAC output,
resolves near-duplicates better than raw inlier count.

## The finding

**Yes — a combined inlier-count / reprojection-error score roughly doubles Stage-2 accuracy over the
raw-inlier baseline, and the gain grows with K rather than shrinking.**

### Stage-2 top-1 accuracy by score, K=50 (largest shortlist tested)

| Condition | raw_inliers (baseline) | inlier_ratio | match_density | geometric_residual | **combined** |
|---|---|---|---|---|---|
| mild | 0.325 | 0.208 (−0.117) | 0.325 (+0.000) | 0.542 (+0.217) | **0.650 (+0.325)** |
| moderate | 0.225 | 0.142 (−0.083) | 0.225 (+0.000) | 0.392 (+0.167) | **0.458 (+0.233)** |

### Full sweep — mild

| K | score | recall@K | stage2_top1 | retr_fail | verify_fail | vs baseline |
|---|---|---|---|---|---|---|
| 1 | raw_inliers | 0.050 | 0.050 | 0.950 | 0.000 | +0.000 |
| 1 | inlier_ratio | 0.050 | 0.050 | 0.950 | 0.000 | +0.000 |
| 1 | match_density | 0.050 | 0.050 | 0.950 | 0.000 | +0.000 |
| 1 | geometric_residual | 0.050 | 0.050 | 0.950 | 0.000 | +0.000 |
| 1 | combined | 0.050 | 0.050 | 0.950 | 0.000 | +0.000 |
| 5 | raw_inliers | 0.200 | 0.108 | 0.800 | 0.092 | +0.000 |
| 5 | inlier_ratio | 0.200 | 0.100 | 0.800 | 0.100 | −0.008 |
| 5 | match_density | 0.200 | 0.108 | 0.800 | 0.092 | +0.000 |
| 5 | geometric_residual | 0.200 | 0.117 | 0.800 | 0.083 | +0.008 |
| 5 | combined | 0.200 | 0.142 | 0.800 | 0.058 | +0.033 |
| 10 | raw_inliers | 0.342 | 0.150 | 0.658 | 0.192 | +0.000 |
| 10 | inlier_ratio | 0.342 | 0.108 | 0.658 | 0.233 | −0.042 |
| 10 | match_density | 0.342 | 0.150 | 0.658 | 0.192 | +0.000 |
| 10 | geometric_residual | 0.342 | 0.200 | 0.658 | 0.142 | +0.050 |
| 10 | combined | 0.342 | 0.242 | 0.658 | 0.100 | +0.092 |
| 20 | raw_inliers | 0.608 | 0.233 | 0.392 | 0.375 | +0.000 |
| 20 | inlier_ratio | 0.608 | 0.133 | 0.392 | 0.475 | −0.100 |
| 20 | match_density | 0.608 | 0.233 | 0.392 | 0.375 | +0.000 |
| 20 | geometric_residual | 0.608 | 0.358 | 0.392 | 0.250 | +0.125 |
| 20 | combined | 0.608 | **0.425** | 0.392 | 0.183 | +0.192 |
| 50 | raw_inliers | 0.925 | 0.325 | 0.075 | 0.600 | +0.000 |
| 50 | inlier_ratio | 0.925 | 0.208 | 0.075 | 0.717 | −0.117 |
| 50 | match_density | 0.925 | 0.325 | 0.075 | 0.600 | +0.000 |
| 50 | geometric_residual | 0.925 | 0.542 | 0.075 | 0.383 | +0.217 |
| 50 | combined | 0.925 | **0.650** | 0.075 | 0.275 | +0.325 |

### Full sweep — moderate

| K | score | recall@K | stage2_top1 | retr_fail | verify_fail | vs baseline |
|---|---|---|---|---|---|---|
| 1 | raw_inliers | 0.017 | 0.017 | 0.983 | 0.000 | +0.000 |
| 1 | inlier_ratio | 0.017 | 0.017 | 0.983 | 0.000 | +0.000 |
| 1 | match_density | 0.017 | 0.017 | 0.983 | 0.000 | +0.000 |
| 1 | geometric_residual | 0.017 | 0.017 | 0.983 | 0.000 | +0.000 |
| 1 | combined | 0.017 | 0.017 | 0.983 | 0.000 | +0.000 |
| 5 | raw_inliers | 0.075 | 0.050 | 0.925 | 0.025 | +0.000 |
| 5 | inlier_ratio | 0.075 | 0.025 | 0.925 | 0.050 | −0.025 |
| 5 | match_density | 0.075 | 0.050 | 0.925 | 0.025 | +0.000 |
| 5 | geometric_residual | 0.075 | 0.050 | 0.925 | 0.025 | +0.000 |
| 5 | combined | 0.075 | 0.067 | 0.925 | 0.008 | +0.017 |
| 10 | raw_inliers | 0.158 | 0.067 | 0.842 | 0.092 | +0.000 |
| 10 | inlier_ratio | 0.158 | 0.058 | 0.842 | 0.100 | −0.008 |
| 10 | match_density | 0.158 | 0.067 | 0.842 | 0.092 | +0.000 |
| 10 | geometric_residual | 0.158 | 0.092 | 0.842 | 0.067 | +0.025 |
| 10 | combined | 0.158 | 0.125 | 0.842 | 0.033 | +0.058 |
| 20 | raw_inliers | 0.367 | 0.108 | 0.633 | 0.258 | +0.000 |
| 20 | inlier_ratio | 0.367 | 0.050 | 0.633 | 0.317 | −0.058 |
| 20 | match_density | 0.367 | 0.108 | 0.633 | 0.258 | +0.000 |
| 20 | geometric_residual | 0.367 | 0.225 | 0.633 | 0.142 | +0.117 |
| 20 | combined | 0.367 | **0.250** | 0.633 | 0.117 | +0.142 |
| 50 | raw_inliers | 0.792 | 0.225 | 0.208 | 0.567 | +0.000 |
| 50 | inlier_ratio | 0.792 | 0.142 | 0.208 | 0.650 | −0.083 |
| 50 | match_density | 0.792 | 0.225 | 0.208 | 0.567 | +0.000 |
| 50 | geometric_residual | 0.792 | 0.392 | 0.208 | 0.400 | +0.167 |
| 50 | combined | 0.792 | **0.458** | 0.208 | 0.333 | +0.233 |

`recall@K` and `retrieval_failure_rate` are identical across scores at a given K by construction —
they depend only on SSCD's shortlist, not on how Stage 2 ranks it. Zero-match / failed-homography
pairs were 0/6000 in both conditions, so every candidate had real evidence to score; none of the
observed differences are an artifact of the shared evidence gate discarding candidates.

### Reading the four scores

- **`combined` wins at every K≥5, in both conditions**, and its lead over baseline *grows* with K
  (mild: +0.033 at K=5 → +0.325 at K=50). This is the opposite of what you'd expect if a better score
  just moved the needle a little — it means the raw-inlier score's failure mode gets worse precisely
  where a bigger shortlist should help most, and reprojection error is what recovers that.
- **`geometric_residual` (median reprojection error, lower-is-better) alone is the second-best score**
  and the main driver behind `combined` — it captures most of the available gain by itself.
- **`match_density` never differs from `raw_inliers`** in this run. Within one query the leak frame's
  keypoint count is fixed, and reference-frame keypoint counts turned out similar enough across
  candidates that dividing by `sqrt(n_kpts_leak · n_kpts_ref)` never changed the argmax ranking. This
  is a property of this footage (uniform texture density across near-static frames), not necessarily
  of the score in general.
- **`inlier_ratio` is worse than baseline at every K>1.** It rewards a candidate with very few matches
  that all happen to be inliers over one with more matches and proportionally fewer (but more
  numerous) inliers — backwards when more matches means more geometric evidence, not less.

## What this means for the baseline

Raw inlier count was discarding a signal it already computes (reprojection error) in favor of one
that turned out uninformative here (keypoint-count normalization) and one that actively hurts
(match-normalized ratio). The fix costs nothing architecturally — `combined` is derived from the exact
same LightGlue/RANSAC call already being made per candidate.

Per `implement_2.md`, this is a measurement result, not a tuned optimum: `combined`'s formula
(`num_inliers / (1 + median_reproj_err)`) has no fitted weights and was not selected by search over
this data — it is the natural, parameter-free combination of the two individually-useful signals. A
follow-up could still explore whether a different combination does better, but that was explicitly out
of scope here.

---

## Implementation

Two files touched, one new.

| File | Change |
|---|---|
| `src/retrieval/lightglue_verify.py` | Additive refactor. `_match_and_ransac()` now holds the raw LightGlue+RANSAC computation (matches, inliers, per-inlier reprojection errors, detected keypoint counts in both frames); `score_pair()` is now a thin wrapper reproducing its exact previous output, so `evaluate_stage2.py`'s baseline behavior is unchanged (confirmed: its `raw_inliers` numbers are bit-for-bit identical here). Added `Stage2Verifier.verify_frame_raw()`, which runs this raw computation once per (leak, candidate) pair — every score below is derived from that same call, not recomputed per score. |
| `src/retrieval/evaluate_stage2_scores.py` | **New.** Reuses `evaluate.load_frames`, `encoders.build_encoder`, `index.build_index`, `capture_sim.CaptureSim`, and `lightglue_verify.Stage2Verifier` unchanged. Defines the five scores in `SCORE_METHODS`, slices one `max(K)` computation per leak frame down to every smaller K (as `evaluate_stage2.py` already did for the single baseline score), and reports `stage1_recall@K`, `stage2_top1`, `retrieval_failure_rate`, `verification_failure_rate`, and `improvement_over_baseline` per condition × K × score. |

### Score definitions

| Score | Formula | Direction | Worst-case value |
|---|---|---|---|
| `raw_inliers` | `num_inliers` | higher better | 0 |
| `inlier_ratio` | `num_inliers / num_matches` | higher better | 0 |
| `match_density` | `num_inliers / sqrt(n_kpts_leak · n_kpts_ref)` | higher better | 0 |
| `geometric_residual` | `median(inlier reprojection errors)` | **lower** better | +∞ |
| `combined` | `num_inliers / (1 + median(inlier reprojection errors))` | higher better | 0 |

All five share one evidence gate — fewer than `min_matches` (default 10) LightGlue matches, or RANSAC
finding no homography at all — so every score is compared on identical underlying evidence; only the
formula differs.

### Reproduce

```bash
PYTHONPATH=.:src /home/ubuntu/cs_exp/cam_cap_dwt_l2/.venv/bin/python3 src/retrieval/evaluate_stage2_scores.py \
  --conditions mild moderate --ks 1 5 10 20 50
```

Full results: `out/retrieval/stage2_score_results.json`.

Constraints from `implement_1.md`/`implement_2.md` honored throughout: no temporal/fps assumptions, no
sliding windows, no DTW, no cheap filtering between SSCD and LightGlue, no perspective-correction
preprocessing, and the estimated homography is used only to produce a scalar score — never to warp or
rectify a frame. `REPORT.md` and `STAGE2_REPORT.md` are left unmodified, per `implement_2.md`.
