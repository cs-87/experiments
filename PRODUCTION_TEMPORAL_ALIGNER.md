# Production Temporal Aligner

**The validated Stage 4A/4B design, hardened, cleanly decoupled from visual matching, and frozen as the production temporal preprocessing stage.**

| | |
|---|---|
| Visual matcher | `SSCD → top-K → DISK → LightGlue → RANSAC → combined` — **unchanged** |
| Temporal model | Scalar constant-velocity filter (position/velocity/uncertainty) — **unchanged from Stage 4A/4B**, one hardening addition |
| New this stage | `LARGE_MISMATCH_MULTIPLIER` immediate-reset hardening; `VisualMatcher`/`TemporalAligner` decoupling; production integration test |
| Backward compatibility | `evaluate_stage4_temporal.py` and `evaluate_stage4b_temporal_robustness.py` — **unmodified**, rerun as-is against the hardened aligner |
| Freeze decision | **Yes** — see §13 |

---

## 1. Final architecture

```
reference video --(once)--> VisualMatcher: SSCD index + DISK cache
                                                    |
leak video --(per frame, streaming)--> VisualMatcher.match() --> (candidate_idx, combined_scores)
                                                    |
                                     TemporalAligner.process(visual_candidates=...)
                                                    |
                                estimated_original_frame + ranked ±radius candidate region
```

`VisualMatcher` owns everything expensive and one-time (SSCD index, DISK feature cache) and everything
per-frame-expensive (SSCD retrieval, LightGlue/RANSAC verification, the frozen `combined` score).
`TemporalAligner` owns only the cheap streaming state update and candidate ranking. Neither imports,
calls, or knows anything about a downstream watermark detector — confirmed by the same
`no_watermark_import` check every prior stage ran (scans actual `import`/`from` lines, not just any
mention of the word), which still passes.

## 2. `TemporalAligner` API

Two supported call patterns, both exercised in this stage's validation:

```python
# Legacy / convenience mode -- unchanged since Stage 4A, what Stage 4A/4B's own
# (unmodified) evaluation scripts still call:
aligner = TemporalAligner(reference_frames, k=50, radius=5)
result = aligner.process(leak_frame_bgr)

# Decoupled production mode -- visual matching and temporal alignment as
# separate, independently reusable components:
matcher = VisualMatcher(reference_frames, k=50)
aligner = TemporalAligner(radius=5)                       # no reference frames needed at all
result = aligner.process(visual_candidates=matcher.match(leak_frame_bgr)[:2])
```

Both paths run through the identical temporal-update-and-ranking code (verified: legacy and decoupled
mode produce byte-identical `pos_hat`/`vel_hat`/`estimated_original_frame` sequences on the same input),
so the hardening below, and every prior Stage 4A/4B behavior, applies to both identically.
`.reset()` clears `pos_hat`/`vel_hat`/`sigma`/the mismatch counter and preserves the reference-side
caches; confirmed directly (`pos_hat=None, vel_hat=0.0, sigma=100.0` immediately after `reset()`).
`.process()` never receives a ground-truth or frame-index parameter — its signature is
`(leak_frame_bgr=None, visual_candidates=None)`, nothing else.

## 3. Candidate output contract

```json
{
  "estimated_original_frame": 503,
  "candidate_frames": [503, 502, 504, 501, 505, 500, 506, 499, 507, 498, 508],
  "candidate_temporal_scores": [...],
  "confidence": 0.94,
  "raw_visual_prediction": 502,
  "raw_visual_score": 187.3,
  "temporal_adjusted_prediction": 503
}
```

`estimated_original_frame` is the filtered center; `candidate_frames` is the ranked ±`radius` region
(default 5, i.e. 11 frames), clipped at reference-video boundaries (confirmed: requesting a region
around frame 0 or the last reference frame returns only in-range indices, never invented ones).
Ranking is primary by proximity to the center, tie-broken by the direction the current velocity implies
and then by real LightGlue-verified evidence for any candidate that happens to be in that frame's own
top-K — never by rerunning LightGlue for the region, never by any watermark/detector signal.

## 4. Temporal model

Unchanged from Stage 4A/4B: `pos_hat` (position), `vel_hat` (velocity, **initialized to 0.0, never
1.0**), `sigma` (uncertainty). Every frame: predict `pos_hat + vel_hat`, gate the raw visual pick against
that prediction, and if in the gate, blend the best visually-supported in-gate candidate into the state
via a Kalman-style gain; a fixed 3-consecutive-frame mismatch streak re-bootstraps for persistent (not
one-off) disagreement. This is a soft prior throughout — a single bad visual observation is smoothed by
the gate, and a genuinely different trajectory can (and does — see §5) redirect the state.

## 5. Discontinuity hardening

Stage 4B found that at a real discontinuity, the filter needed 2 frames to reach the existing 3-frame
mismatch-streak reset, and in the meantime partially blended toward the new evidence, causing `vel_hat`
to transiently explode (chasing a residual computed against an already-stale `predicted_pos`) and
producing an error of 365 frames on a single frame even though the raw visual matcher found the correct
region immediately.

**Fix**: a magnitude-scaled immediate-reset check, run *before* the existing gate/streak logic:

```python
if abs(raw_visual_prediction - predicted_pos) > LARGE_MISMATCH_MULTIPLIER * gate_radius:
    pos_hat, vel_hat, sigma, mismatch_streak = raw_visual_prediction, 0.0, SIGMA_INIT, 0
```

`LARGE_MISMATCH_MULTIPLIER = 50` was derived from two data points **already on disk before this stage
ran**, not tuned by rerunning the discontinuity test until it passed:

| Case | mismatch magnitude | gate_radius (from recorded confidence) | ratio |
|---|---|---|---|
| Stage 4A leak-106 (must **not** trigger — already handled correctly by the 3-frame path) | 79 | 4.81 | **16.4** |
| Stage 4B discontinuity (**must** trigger) | 602 | 4.81 | **125.2** |

`50` sits at roughly the midpoint of the valid window (>2× the 16.4 case, <½× the 125.2 case) — verified
directly (`50 × 4.81 = 240.4`, comfortably between 79 and 602) before any experiment was rerun.

## 6. Validation: Stage 4A → Stage 4B → final production version

| Dataset | Stage 4A/4B (original) | Final (hardened, rerun via the same unmodified scripts) |
|---|---|---|
| short4s/mild recall@±5 | 1.000 | **1.000** (identical) |
| short4s/moderate recall@±5 | 0.958 | **0.958** (identical) |
| long1min600/mild recall@±5 | 0.887 | **0.887** (identical) |
| 13 Stage 4B distortion experiments | recall@±5 = 1.000 on all 13 | **1.000 on all 13, every metric numerically identical** |
| discontinuity recovery time | 2 frames | **0 frames** |
| discontinuity max error during recovery | 365 | **2** |
| discontinuity recall@±5 | 0.898 | **0.902** |
| discontinuity `permanently_locked` | False | **False** |

Every non-discontinuity result is bit-for-bit unchanged, exactly as predicted from the derivation in §5
(none of those cases ever approached a mismatch ratio near 50) — the hardening is provably surgical, not
a general behavior change. Note: `max_err=42` still appears in the post-hardening discontinuity run, but
tracing the per-frame data shows every one of those `|error|>5` frames occurs in the *first* 40 frames of
the sequence (ordinary bootstrap-period noise, unrelated to the discontinuity, which now recovers with a
peak error of only 2) plus one isolated frame at leak 417 — not a discontinuity-handling issue.

## 7. Moderate + temporal validation (new — closes Stage 4B's stated limitation)

| Combined test | recall@±5 | exact | mean err | max err |
|---|---|---|---|---|
| moderate + drop_5pct | 0.956 | 0.342 | 1.19 | 9 |
| moderate + duplicate_5pct | 0.952 | 0.444 | 1.13 | 14 |
| moderate + speed_1.05 | 0.956 | 0.316 | 1.22 | 8 |
| moderate + mixed | 0.941 | 0.424 | 1.30 | 17 |

All four land within a few points of Stage 4A's moderate-alone baseline (0.958) — combining photometric
severity with temporal distortion does not compound into a materially worse result.

## 8. Candidate recall

Representative primary-config (K=50, radius=5) results, mild condition unless noted:

| | ±1 | ±3 | ±5 | ±10 |
|---|---|---|---|---|
| short4s/mild (identity) | 0.950 | 1.000 | 1.000 | 1.000 |
| long1min600/mild (identity) | 0.602 | 0.815 | 0.887 | 0.967 |
| drop/duplicate/speed/mixed (13 cases) | 0.907–0.983 | 0.984–1.000 | **1.000** (all 13) | 1.000 |
| discontinuity (hardened) | 0.696 | 0.844 | 0.902 | ~0.94 |
| moderate + temporal (4 cases) | — | — | 0.941–0.956 | — |

## 9. Ranked recall

Top-11 ranked recall equals ±5 candidate recall exactly on every dataset (both describe "is the true
frame anywhere in the returned 11-position region") — the expected cross-check for `radius=5`. Top-1
ranked recall equals exact-center accuracy for the same reason (rank 0 = the center itself).

## 10. Recovery

| | recovery time | max error during recovery | post-recovery recall@±5 | permanently locked? |
|---|---|---|---|---|
| discontinuity (hardened) | **0 frames** | **2** | 0.976 | **No** |
| local speed change (×2, unaffected by hardening) | 0 | 1 | 1.000 | No |

The discontinuity case is now the cleanest possible recovery outcome: the frame immediately after the
cut is already correct.

## 11. Runtime

| | ms/frame SSCD | ms/frame LightGlue | ms/frame temporal | Reference preprocessing |
|---|---|---|---|---|
| short4s (120 frames) | ~19–20 | ~584–596 | **~0.10** | ~2.9s index + ~15.3s DISK cache |
| long1min600 (600 frames) | ~18 | ~778–802 | **~0.10** | ~10.2s index + ~75.3s DISK cache |
| discontinuity (1200-frame ref, 500 leak frames) | ~18 | ~779 | **~0.10** | ~150s DISK cache |

Temporal overhead is unchanged at ~0.10 ms/frame — the hardening added one comparison and a branch, not
a measurable cost, exactly as required (roughly 6,000–8,000× cheaper than LightGlue in every
configuration tested). Total wall-clock scales with leak-frame count as before (no distortion type or
the hardening changes this).

## 12. Limitations

- **All distortions are synthetic and constructed**, not observed from a real capture device; the real
  `leak.mp4`/`IMG_2195.mov` pair still has no trustworthy ground truth (`REPORT.md` §03), so this
  component has never been run against genuine capture-hardware timing artifacts.
- **`LARGE_MISMATCH_MULTIPLIER=50` was derived from exactly two data points.** It is well-justified by
  those two (a comfortable margin on both sides), but a systematic sensitivity sweep against additional
  independent discontinuity magnitudes has not been done.
- **Only one discontinuity configuration and one moderate+temporal combination set were tested.**
  Multiple discontinuities in one sequence, or moderate/severe conditions combined with a discontinuity
  specifically, are untested.
- **The K/radius robustness grid from Stage 4B was not rerun here** (deliberately — it is orthogonal to
  this hardening, and its qualitative finding, K=50 materially beating K=20, already stands).
- **Peak memory scales with reference-frame count cached** (confirmed ~9GB host RSS for 600 reference
  frames in Stage 4A) — a full-length feature film's reference cache size has not been measured.

## 13. Production readiness

**Is the temporal aligner now ready to be used as the standalone temporal preprocessing stage before the
downstream detector? Yes.** Against the freeze criteria: the hardening (a) preserves every previously
successful distortion result exactly, (b) strictly improves discontinuity recovery (365→2 max error,
2→0 frame recovery time) without introducing any new permanent-lock risk, (c) preserves ~0.1ms/frame
temporal overhead, and (d) the new moderate+temporal combination lands within a few points of the
already-accepted moderate-alone baseline. All four freeze criteria hold. Per the brief's own decision
rule, **temporal-algorithm development stops here** — no DTW, global optimization, neural temporal
model, or additional heuristic layer is added. `VisualMatcher` and `TemporalAligner` are the frozen
production interface; the next project stage is integration with the downstream watermark detector, not
further temporal-alignment experimentation.

---

*`REPORT.md`, `STAGE2_REPORT.md`, `STAGE2_SCORES_REPORT.md`, `STAGE2_GENERALIZATION_REPORT.md`,
`STAGE3A_REPORT.md`, `STAGE3B_REPORT.md`, `STAGE4A_REPORT.md`, and `STAGE4B_REPORT.md` are unmodified.
`evaluate_stage4_temporal.py` and `evaluate_stage4b_temporal_robustness.py` are unmodified and remain
reproducible; they were rerun as-is (new `--out-dir` values only) to produce this stage's "after"
numbers.*
