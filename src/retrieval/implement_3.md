We completed Stage 2b of the SSCD → DISK/LightGlue retrieval experiment.

## Current architecture

Keep the architecture exactly unchanged:

SSCD → Top-K → DISK → LightGlue → RANSAC → candidate score

There is:

* no temporal prior
* no FPS assumption
* no sliding window
* no DTW
* no cheap filtering between SSCD and LightGlue
* no perspective-correction preprocessing
* no image warping/rectification for scoring

Reference DISK features are cached once. LightGlue/RANSAC is computed once per leak/candidate pair.

## Current result

The original Stage-2 baseline ranked candidates using raw RANSAC inlier count.

Stage 2b compared:

* `raw_inliers = num_inliers`
* `inlier_ratio = num_inliers / num_matches`
* `match_density = num_inliers / sqrt(n_kpts_leak * n_kpts_ref)`
* `geometric_residual = median(inlier reprojection error)`, lower is better
* `combined = num_inliers / (1 + median_reprojection_error)`

The combined score was substantially better.

At K=50:

mild:

* raw_inliers: 0.325
* geometric_residual: 0.542
* combined: 0.650
* stage1_recall@50: 0.925

moderate:

* raw_inliers: 0.225
* geometric_residual: 0.392
* combined: 0.458
* stage1_recall@50: 0.792

The important result is that the combined score greatly reduced verification failures without changing Stage 1 or the underlying LightGlue/RANSAC evidence.

However, this result is currently based on only one short source video:

`4_sec_source.mp4`
120 frames
1920×1080
30fps

The footage is relatively near-static.

## NEXT EXPERIMENT: GENERALIZATION ACROSS CONTENT

Do NOT optimize the scoring formula further yet.

Do NOT redesign the architecture.

The next goal is to determine whether the Stage-2b improvement generalizes to different types of video content.

The primary research question is:

> Does the combined inlier-count/reprojection-error score consistently outperform raw RANSAC inlier count across videos with different amounts and types of motion?

Before implementing anything, inspect:

* `REPORT.md`
* `STAGE2_REPORT.md`
* the Stage 2b report
* `implement_1.md`
* `implement_2.md`
* `src/retrieval/evaluate_stage2_scores.py`
* `src/retrieval/lightglue_verify.py`

Understand and reuse the existing evaluation pipeline.

## Content categories

Find appropriate existing source videos in the repository/data available on this machine if possible.

We want several clips representing meaningfully different temporal characteristics.

At minimum try to cover:

1. Near-static / low motion

   * little camera motion
   * little object motion
   * many adjacent frames visually almost identical

2. Moderate motion

   * normal object/person movement
   * adjacent frames still related but visibly changing

3. High motion

   * substantial object or camera motion
   * adjacent frames differ considerably

4. Camera motion, if available

   * pan / tilt / zoom / handheld movement

Do not artificially classify clips purely from filename unless their content is known.

If suitable existing videos are not available, report that clearly rather than silently generating synthetic content or changing the experiment.

Keep clips reasonably short so the experiment remains practical, but each clip should contain enough frames for retrieval ambiguity to exist.

Record for each clip:

* filename
* resolution
* FPS
* frame count
* duration
* assigned content category

Do NOT use FPS or temporal position as retrieval information. FPS is metadata for reporting only.

## Scoring methods

For this experiment, the important comparison is:

1. `raw_inliers`
2. `geometric_residual`
3. `combined`

Use exactly the Stage-2b definitions:

`raw_inliers`:
num_inliers
higher is better

`geometric_residual`:
median(inlier reprojection errors)
lower is better

`combined`:
num_inliers / (1 + median(inlier reprojection errors))
higher is better

Do NOT tune the combined formula per video.

Do NOT introduce fitted weights.

The exact same formula must be used for every clip and every capture condition.

You may retain the other Stage-2b scores in the implementation if doing so is essentially free, but the report should focus on the three scores above.

## Evaluation

Use the same capture conditions as the existing experiment:

* mild
* moderate

Use the same K sweep:

* 1
* 5
* 10
* 20
* 50

For each:

video × condition × K × score

report:

* `stage1_recall@K`
* `stage2_top1`
* `retrieval_failure_rate`
* `verification_failure_rate`
* improvement over `raw_inliers`

The exact same SSCD shortlist and exact same LightGlue/RANSAC evidence must be used when comparing scoring methods.

Do not rerun feature matching separately for each scoring method.

## Important additional analysis

The main purpose is not just to generate one aggregate accuracy number.

We specifically want to understand how content motion affects the system.

For each video, compare:

### 1. Stage-1 behavior

Does SSCD recall@K improve faster on videos with more temporal variation?

For example, does high-motion content naturally make exact frames easier to retrieve than near-static content?

### 2. Stage-2 behavior

Compare:

`raw_inliers` vs `geometric_residual` vs `combined`

Does reprojection error provide its largest benefit on near-static content where many candidates share similar geometry?

Or does it remain useful on high-motion content?

### 3. K behavior

Determine whether the Stage-2b finding:

> combined benefits increasingly from larger K

holds across content types.

Do not assume that K=50 is optimal.

We are measuring behavior, not choosing a production K yet.

### 4. Failure-mode decomposition

Continue separating:

retrieval failure:
correct frame is not present in SSCD top-K

verification failure:
correct frame is present but Stage 2 selects another frame

This is critical.

For each content category, identify whether the dominant limitation is:

* SSCD retrieval
* Stage-2 verification
* or neither clearly dominates

## Near-duplicate analysis

If practical, add one diagnostic specifically aimed at understanding the near-duplicate hypothesis.

For Stage-2 verification failures, record:

* ground-truth frame index
* selected frame index
* absolute frame-index difference

This is for analysis only.

Do NOT use frame-index difference during retrieval or ranking.

Report distributions such as:

* exact
* ±1 frame
* ±2–5 frames
* ±6–10 frames
* > 10 frames

This will help answer whether "wrong" LightGlue selections are mostly adjacent near-duplicates or completely unrelated frames.

Again: frame index is evaluation metadata only and must NEVER influence the ranking.

## Implementation constraints

Prefer a new evaluation driver, for example:

`src/retrieval/evaluate_stage2_generalization.py`

Reuse the existing Stage-2b implementation wherever possible.

Do not modify the baseline scoring behavior.

Do not duplicate DISK/LightGlue computation unnecessarily.

Do not introduce:

* temporal priors
* frame-index priors
* FPS-based assumptions
* temporal smoothing
* sliding windows
* DTW
* optical-flow temporal tracking
* candidate filtering between SSCD and LightGlue
* perspective-correction preprocessing

This experiment is specifically testing whether the existing visual retrieval architecture and fixed Stage-2b score generalize across content.

## Output

Write detailed machine-readable results to something like:

`out/retrieval/stage2_generalization_results.json`

Include per-video metadata and per-query results where practical so failures can be inspected later.

Also print a concise summary table:

video | category | condition | K | score | recall@K | stage2_top1 | retrieval_failure | verification_failure | vs_raw

Then produce an aggregate summary grouped by content category:

category | condition | K | score | stage2_top1

Do NOT hide per-video results behind aggregate averages.

## Interpretation

After running the experiment, write a report, but do not modify the existing reports.

The report should answer:

1. Does the Stage-2b combined score consistently beat raw inlier count across videos?
2. How much does the improvement vary with content type?
3. Does geometric residual remain the main useful signal?
4. Does increasing K continue helping the combined score?
5. Are verification failures predominantly temporally adjacent frames?
6. Does the dominant bottleneck shift between near-static and high-motion content?
7. Is the Stage-2b result likely a general property of the verifier, or does it appear specific to `4_sec_source.mp4`?

Do not overclaim from a small dataset.

If the results differ from the previous experiment, report that directly rather than trying to reconcile them.

The objective of this iteration is **generalization testing**, not further optimization.

Before coding, briefly explain:

* which videos you found,
* how you categorized them,
* what existing code will be reused,
* what new code is actually necessary.

Then implement and run the experiment.
