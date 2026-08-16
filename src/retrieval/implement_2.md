We just completed Experiment 1, Stage 2: SSCD shortlist + DISK/LightGlue geometric verification.

Current baseline:

- Corpus: 4_sec_source.mp4, 120 frames, 1920x1080, 30fps
- Hardware: Tesla T4
- Stage 1: SSCD (`sscd_disc_mixup`), unchanged
- Stage 2: DISK (1024 keypoints) + LightGlue
- Verification currently uses RANSAC homography only to compute a raw inlier-count score
- No temporal prior
- No sliding windows
- No DTW
- No cheap filtering between SSCD and LightGlue
- No perspective-correction preprocessing
- Homography is NOT used to warp/rectify anything
- Reference DISK features are cached once and reused
- Leak-frame DISK features are extracted once
- Stage 1 is run once at max(K), then the ranked list is sliced for smaller K
- Current K values: 1, 5, 10, 20, 50
- Conditions: mild, moderate

Current results show that the two-stage architecture works, but raw RANSAC inlier count is too coarse:

mild:
K=1   stage1_recall=0.050  stage2_top1=0.050
K=5   stage1_recall=0.200  stage2_top1=0.108
K=10  stage1_recall=0.342  stage2_top1=0.150
K=20  stage1_recall=0.608  stage2_top1=0.233
K=50  stage1_recall=0.925  stage2_top1=0.325

moderate:
K=1   stage1_recall=0.017  stage2_top1=0.017
K=5   stage1_recall=0.075  stage2_top1=0.050
K=10  stage1_recall=0.158  stage2_top1=0.067
K=20  stage1_recall=0.367  stage2_top1=0.108
K=50  stage1_recall=0.792  stage2_top1=0.225

The main failure mode is near-duplicate frames: multiple adjacent reference frames can produce similarly strong geometric matches, so raw inlier count does not reliably identify the exact source frame.

NEXT EXPERIMENT:

Do NOT redesign the architecture.

Keep:

SSCD → Top-K → DISK → LightGlue → RANSAC

exactly as the baseline.

Instead, investigate whether a better Stage-2 ranking score can distinguish the correct frame from geometrically plausible near-duplicates.

First inspect the existing implementation and REPORT.md / implement_1.md so you understand the current code and conventions. Do not modify existing behavior unnecessarily.

Implement the following candidate verification scores:

1. Raw RANSAC inlier count
   score = number of RANSAC inliers
   This must remain as the baseline.

2. Inlier ratio
   score = inliers / total LightGlue matches
   Handle zero matches safely.

3. Match-density normalization
   Implement a reasonable normalized inlier score that accounts for the amount of available matching evidence. Keep the definition explicit and reproducible.

4. Geometric residual
   After estimating the homography from the LightGlue matches, compute a geometric reprojection-error statistic for the RANSAC inliers.
   IMPORTANT:
   - Homography may be estimated for this calculation.
   - Do NOT warp/rectify the actual image.
   - Do NOT introduce perspective-correction preprocessing.
   - Report a robust statistic such as median or mean inlier reprojection error.
   - Make the score direction explicit: higher-is-better or lower-is-better.

5. If practical, implement one combined score using the existing quantities, but do not tune arbitrary weights on the test set.
   Prefer either a clearly justified normalized combination or keep the individual metrics separate first.

The goal of this experiment is NOT to find the best possible score through aggressive tuning.

The goal is to answer:

"Does a more discriminative geometric verification score improve exact-frame retrieval over raw RANSAC inlier count?"

EXPERIMENT DESIGN:

For every condition and every K in:

1, 5, 10, 20, 50

evaluate every scoring method on exactly the same SSCD shortlist and exactly the same LightGlue matches.

Do NOT rerun SSCD independently for each score.

Ideally compute LightGlue/RANSAC information once per leak-frame/candidate pair and then derive all scores from that same result. This makes the comparison fair and avoids unnecessary computation.

Report for every condition × K × score:

- stage1_recall@K
- stage2_top1
- retrieval_failure_rate
- verification_failure_rate

Also report:

- improvement over raw-inlier baseline
- whether the correct candidate was present in the shortlist but lost under each score
- number of zero-match / failed-homography cases, if any

Most importantly, preserve the existing failure-mode distinction:

retrieval failure:
    correct frame not in top-K

verification failure:
    correct frame is in top-K but Stage 2 selects another candidate

OUTPUT:

Create a new evaluation script rather than modifying the existing baseline evaluation behavior, for example:

src/retrieval/evaluate_stage2_scores.py

Reuse the existing:
- frame loading
- SSCD encoder/index
- CaptureSim
- DISK cache
- LightGlue matching

where possible.

Do not duplicate large amounts of existing code if the current implementation can be cleanly reused.

Write results to something like:

out/retrieval/stage2_score_results.json

The JSON should contain enough information to reproduce and analyze the comparison later.

Also produce a concise human-readable summary showing something like:

condition | K | score | stage1_recall | stage2_top1 | retrieval_failure | verification_failure

Do not modify REPORT.md yet.

Do not make any claims about which score is better until the experiment has actually been run.

Before coding, inspect the current Stage-2 implementation and explain briefly:
1. where matches are produced,
2. where RANSAC is currently performed,
3. where the current score is calculated,
4. what can be reused for the new scoring methods.

Then implement and run the experiment.

Keep the implementation minimal and experimental. The purpose of this iteration is measurement, not production cleanup.