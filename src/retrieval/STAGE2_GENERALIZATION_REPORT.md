# Stage 2c: Does the Combined Score Generalize Across Content?

**Experiment 1, continued — same architecture and same three scores as `STAGE2_SCORES_REPORT.md`, tested on four videos with different motion character instead of one.**

| | |
|---|---|
| Architecture | `SSCD → top-K → DISK → LightGlue → RANSAC` — **unchanged** |
| Scores | `raw_inliers`, `geometric_residual`, `combined` — **unchanged formulas** from Stage 2b |
| Conditions | mild, moderate |
| K values | 1, 5, 10, 20, 50 |
| Videos | 4, each a different source, each trimmed to 120 frames at native resolution/fps |
| Frame-index use | Diagnostic only (near-duplicate distance bucketing), never in retrieval or ranking |

---

## Videos used

Selected by measuring actual motion (mean 64×64-downsampled luma difference between adjacent frames,
and its coefficient of variation, over several candidate windows per source), not by filename — see
the "Videos found and how they were chosen" section below for the full search.

| Video | Category | Resolution | FPS | Frames | Duration | mean\|Δ\| | cv |
|---|---|---|---|---|---|---|---|
| `4_sec_source.mp4` | near_static | 1920×1080 | 30.00 | 120 | 4.00s | 0.50 | 0.08 |
| `moderate_bigbuck.mkv` | moderate | 1920×1080 | 30.00 | 120 | 4.00s | 3.26 | 0.28 |
| `high_f1.mkv` | high_motion | 1920×1080 | 50.00 | 120 | 2.40s | 9.60 | 0.42 |
| `camera_tokyo.mkv` | camera_motion | 1920×1080 | 30.00 | 120 | 4.00s | 9.41 | 0.32 |

FPS and duration are reporting metadata only — never used by SSCD, DISK, LightGlue, or the scoring
functions.

---

## The finding — and it is not the same story as Stage 2b

**`combined` generalizes essentially perfectly: it never loses to `raw_inliers` on any video, and on**
**three of four videos it reaches stage2_top1 = 1.000 by K=5–20. But `geometric_residual` alone — the**
**score that did most of the work in Stage 2b — is NOT safe in general: on higher-motion content it can**
**catastrophically underperform `raw_inliers` as K grows, sometimes collapsing to near-zero accuracy.**
**This is the opposite of what Stage 2b's near-static-only result would have predicted for that score**
**in isolation, and implement_3.md asked to report exactly this kind of divergence rather than reconcile it.**

### stage2_top1 at K=50, all three scores, both conditions

| Video (category) | cond | raw_inliers | geometric_residual | combined |
|---|---|---|---|---|
| near_static | mild | 0.325 | 0.542 | **0.650** |
| near_static | moderate | 0.225 | 0.392 | **0.458** |
| moderate | mild | 1.000 | 0.817 | **1.000** |
| moderate | moderate | 0.992 | 0.775 | **1.000** |
| high_motion | mild | 1.000 | **0.058** | **1.000** |
| high_motion | moderate | 1.000 | **0.200** | **1.000** |
| camera_motion | mild | 1.000 | 0.992 | **1.000** |
| camera_motion | moderate | 1.000 | 1.000 | 1.000 |

`geometric_residual` is the best score on near-static content and the worst — by a wide margin — on
every other content type tested. `combined` matches or beats `raw_inliers` in all eight rows.

---

## Answering implement_3.md's seven questions

### 1. Does the combined score consistently beat raw inlier count across videos?

Yes, with no exceptions in this run: `combined`'s `stage2_top1` is `≥ raw_inliers` at every video ×
condition × K in the sweep (40 combinations). Where `raw_inliers` already reaches 1.000, `combined`
ties it; on `moderate`/moderate condition, `combined` is measurably better (1.000 vs 0.992). The
near-static clip remains the only case where the gap is large (e.g. +0.325 at K=50/mild).

### 2. How much does the improvement vary with content type?

From "roughly double the baseline" (near-static) to "no measurable difference" (moderate, high-motion,
camera-motion, where `raw_inliers` alone already solves the problem once K is large enough that
`stage1_recall@K` reaches ~1.0). The improvement is concentrated entirely in the regime where Stage 1
struggles — i.e. it is a property of *how hard the shortlist is to disambiguate*, not a universal
multiplier.

### 3. Does geometric residual remain the main useful signal?

**No — this is the headline divergence from Stage 2b.** In isolation, `geometric_residual` is the best
score on near-static content but the *worst* score everywhere else, catastrophically so on `high_f1`
(stage2_top1 collapses from 1.000 at K=5 to 0.058 at K=50 under `mild` capture — worse than doing
nothing). `combined` is not simply "riding on `geometric_residual`'s coattails"; the `num_inliers`
numerator is what protects it (see the mechanism below).

### 4. Does increasing K continue helping the combined score?

On near-static content, yes, monotonically (0.050→0.650 as K goes 1→50 under mild capture) — this is
the Stage 2b finding, confirmed again here on the same clip. On the other three videos, `combined`
reaches 1.000 by K=5–20 and *stays there* — K stops mattering because there's no more Stage-1 headroom
to convert, not because `combined` degrades. Critically, `combined` never goes *down* as K grows on any
video here. `geometric_residual` alone does not share that property: on `high_f1` it falls from 1.000
(K=5) to 0.058 (K=50) — a bigger shortlist actively hurts it. **Larger K is safe for `combined` in this
data; it is not safe for `geometric_residual` alone.**

### 5. Are verification failures predominantly temporally adjacent frames?

**It depends entirely on content, and the two videos where it matters most give opposite answers.**
Near-duplicate distance distribution among retrieval-successful queries, K=50:

| Video | Score | exact | ±1 | ±2–5 | ±6–10 | >10 |
|---|---|---|---|---|---|---|
| near_static / mild | `raw_inliers` | 35% | 29% | 35% | 1% | 0% |
| near_static / mild | `combined` | 70% | 24% | 5% | 0% | 0% |
| moderate / mild | `geometric_residual` | 82% | 6% | 7% | 3% | 3% |
| high_motion / mild | `geometric_residual` | 6% | 0% | 0% | 0% | **94%** |

On near-static content, essentially every wrong pick (from any score) is within ±5 frames — the
near-duplicate-collapse hypothesis from `REPORT.md`/`STAGE2_REPORT.md` holds exactly as described. On
`moderate_bigbuck`, `geometric_residual`'s failures are mostly near-adjacent too (82% exact, 95% within
±5), a graceful degradation. But on `high_f1`, `geometric_residual`'s failures are **94% more than 10
frames away** — essentially unrelated frames, not near-duplicates. Wrong selections there are not
"confusable neighbours"; they look like arbitrary wins by a spurious signal.

### 6. Does the dominant bottleneck shift between near-static and high-motion content?

Yes, cleanly. On `4_sec_source.mp4`, `stage1_recall@50` tops out at 0.925 (mild) / 0.792 (moderate) —
Stage 1 itself never fully solves the shortlist problem, and on top of that `raw_inliers` verification
fails 60%/57% of the time even when the answer is present — Stage 2 is clearly the larger bottleneck,
though Stage 1 is not perfect either. On the other three videos, `stage1_recall@K` reaches 1.000 by
K=10–20 in every case, and once it does, `combined`'s `stage2_top1` reaches 1.000 too —
**neither stage is a meaningful bottleneck for this footage once K is moderately generous.** At K=1
specifically (no verification headroom by construction), 100% of the shortfall on every video is
retrieval failure, trivially resolved by widening K a little.

### 7. Is the Stage 2b result general, or specific to `4_sec_source.mp4`?

**Both, in different halves.** The *architecture* result (two-stage retrieval-then-verification beats
either stage alone, and a smarter score can improve verification without touching Stage 1) generalizes
completely. But the specific claim "`geometric_residual` is the main useful signal" is **not** general —
it is a property of near-static, low-blur content where wrong candidates still produce plentiful, honest
matches. On content with real motion blur or repetitive texture (this run's clearest example is
`high_f1`, which also had the highest zero-match/failed-homography rate: 5.1% of mild pairs vs 0% on
every other video), a low-evidence, low-error degenerate fit against the *wrong* frame can look better
than a high-evidence fit against the *right* one under a score that ignores how much evidence supports
it. `combined`'s `num_inliers` numerator is precisely what prevents that failure mode, which is
presumably why it was defined as a combination in the first place rather than being replaced outright
by `geometric_residual`.

---

## Why `geometric_residual` fails specifically on `high_f1`

`median(inlier reprojection error)` says nothing about *how many* points that median is computed over.
A wrong candidate that happens to produce only 4–6 spuriously well-aligned points can post a near-zero
median error and beat a right candidate with 40 honest inliers and a slightly larger error. This is
rare on near-static, high-evidence content (most candidates are genuine near-duplicates with real,
plentiful matches, so the honest candidate's error is competitive even against occasional flukes) but
becomes common on content where evidence is thinner — `high_f1`'s onboard racing footage has motion
blur and repetitive background texture (barriers, grass, curbs), both of which increase the rate of
sparse, coincidentally-low-error matches against unrelated frames. `combined`'s `num_inliers` term
downweights exactly these low-evidence fits, which is why it does not share the failure.

---

## Videos found and how they were chosen

Per implement_3.md, clips were not classified from their filenames. Motion was measured directly (mean
64×64-downsampled luma difference between adjacent frames, and its coefficient of variation) at several
offsets in each candidate before picking a window:

- **`4_sec_source.mp4`** (existing baseline) — reused as-is for continuity with Stage 2/2b.
- **`dwt_dct/bigbuck1080_standard.mp4`** (Big Buck Bunny, animated) — highly inconsistent across
  offsets (near-static stretches interrupted by scene-cut spikes up to cv≈6). Frames 2400–2520
  (t=80s) gave sustained animated-character motion (mean 3.26, cv 0.28) with no cut spike — used as
  the `moderate` category.
- **`dwt_dct/F1_10min.mp4`** (onboard racing footage, 50fps) — frames 10000–10120 (t=200s, mean 9.60,
  cv 0.42) gave sustained fast motion including an overtake-scale event — used as `high_motion`.
- **`dwt_dct/tokyo_night.mp4`** (driving/walking POV) — sustained motion across every window sampled;
  frames 2700–2820 (t=90s, mean 9.41, cv 0.32) used as `camera_motion`.
- **`feature_based_embedding/inputs/afterdeath_1min.mp4`** was evaluated and **rejected**: every
  window sampled was cut-driven and inconsistent (mean swung 1.5→13.4 between adjacent 4-second
  windows with cv 2.5–5.0), not a clean "sustained high motion" signal, so it was not forced into a
  category.

All four source videos are existing 1920×1080 assets already present in this machine's other
watermarking/retrieval repos (`dwt_dct/`), not new or external content. Clips were trimmed to exactly
120 frames with `trim_video.py --start-frame N` (a new, backward-compatible flag — default 0 leaves
existing behavior, e.g. `4_sec_source.mp4`'s own provenance, unchanged), using the same lossless ffv1
pipeline already in the repo.

---

## Implementation

| File | Change |
|---|---|
| `trim_video.py` | Added `--start-frame` (default 0, backward compatible) so a 120-frame window can be cut from any offset, not just the start of a video. |
| `src/retrieval/evaluate_stage2_generalization.py` | **New.** Reuses `evaluate.load_frames`, `encoders.build_encoder`, `index.build_index`, `capture_sim.CaptureSim`, `lightglue_verify.Stage2Verifier`, and imports `SCORE_METHODS`/`scores_from_raw`/`pick` directly from `evaluate_stage2_scores.py` rather than reimplementing them. Adds: `describe_motion()` (the objective motion diagnostic above), a loop over videos, and `near_duplicate_distribution()` — a diagnostic computed strictly from already-decided Stage-2 predictions; frame index never enters `Stage2Verifier` or any argmax/argmin call. |

Confirmed no regression: the `4_sec_source.mp4` row of this run reproduces `STAGE2_SCORES_REPORT.md`'s
numbers exactly (same video, same conditions, same three scores).

### Reproduce

```bash
python3 trim_video.py --start-frame 2400  /home/ubuntu/cs_exp/dwt_dct/bigbuck1080_standard.mp4 120 moderate_bigbuck.mkv
python3 trim_video.py --start-frame 10000 /home/ubuntu/cs_exp/dwt_dct/F1_10min.mp4            120 high_f1.mkv
python3 trim_video.py --start-frame 2700  /home/ubuntu/cs_exp/dwt_dct/tokyo_night.mp4          120 camera_tokyo.mkv

PYTHONPATH=.:src /home/ubuntu/cs_exp/cam_cap_dwt_l2/.venv/bin/python3 \
  src/retrieval/evaluate_stage2_generalization.py --conditions mild moderate --ks 1 5 10 20 50
```

Full per-video, per-K, per-score metrics, near-duplicate distance distributions, and per-query
predictions at the largest K: `out/retrieval/stage2_generalization_results.json`.

---

## Caveats — do not overclaim from this

- **One video per category.** The "category aggregate" table in the raw output is literally identical
  to the single video in each category — this run establishes that the result is *not universal
  across four specific clips*, not a statistically robust per-category estimate. A real generalization
  claim needs multiple clips per category.
- **`high_f1` had the highest zero-match/failed-homography rate** (5.1% mild, 3.9% moderate, vs 0% on
  every other video) — its native motion blur measurably thins the evidence LightGlue has to work
  with, which is plausibly connected to why `geometric_residual` fails so badly there specifically.
  That connection is a plausible mechanism, not independently verified here.
- **This is still a measurement of one combined formula against one alternative formula on four
  clips.** It answers the question implement_3.md asked — does Stage 2b's result generalize — and the
  honest answer is "the architecture and the `combined` score do; the standalone claim about
  `geometric_residual` does not." It is not a claim that `combined` is optimal, and per implement_3.md
  no further tuning was done here.

`REPORT.md`, `STAGE2_REPORT.md`, and `STAGE2_SCORES_REPORT.md` are left unmodified.
