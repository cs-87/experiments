We have completed Stage 2c.

Before doing anything, inspect the existing reports and implementation, especially:

* `REPORT.md`
* `STAGE2_REPORT.md`
* `STAGE2_SCORES_REPORT.md`
* `STAGE2C_REPORT.md` (or the latest Stage 2c report)
* `src/retrieval/lightglue_verify.py`
* `src/retrieval/evaluate_stage2.py`
* `src/retrieval/evaluate_stage2_scores.py`
* `src/retrieval/evaluate_stage2_generalization.py`

Do not redesign the retrieval architecture yet.

# Goal

The ultimate goal is a production-grade temporal alignment system:

> Given a leaked video, process it frame by frame and produce a mapping from every leaked frame to the most likely original/reference frame.

For example:

```text
leak_frame    original_frame
0             143
1             144
2             145
3             146
4             147
...
```

Ideally the mapping should evolve approximately monotonically/linearly through time, but we are NOT going to enforce temporal constraints yet.

This experiment is the first step toward that system.

# Stage 3a — Independent full-video alignment

For this stage, treat every leak frame independently.

The pipeline must be:

```text
leak frame
    ↓
SSCD retrieval
    ↓
Top-K shortlist
    ↓
DISK + LightGlue
    ↓
RANSAC
    ↓
combined Stage-2 score
    ↓
best original frame
```

Use the frozen Stage 2c combined score:

```text
combined =
    num_inliers / (1 + median_inlier_reprojection_error)
```

Do NOT modify this formula.

Do NOT add temporal information yet.

Specifically, do NOT add:

* temporal priors
* previous-frame constraints
* sliding windows
* DTW
* dynamic programming
* monotonicity constraints
* linear regression for correction
* optical flow
* temporal smoothing

The purpose of this experiment is to observe the raw temporal behavior before deciding what temporal model is appropriate.

# Important distinction

Stage 2 evaluated isolated leak frames.

Stage 3a must process an entire leak video sequentially and produce one prediction for every leak frame.

Conceptually:

```text
Leak video

L0 ──→ matcher ──→ O143
L1 ──→ matcher ──→ O144
L2 ──→ matcher ──→ O145
L3 ──→ matcher ──→ O200
L4 ──→ matcher ──→ O147
...
```

The `O200`-type jump is exactly the kind of thing we want to observe.

Do NOT correct it yet.

# Reference-side preprocessing

Design the implementation so the reference/original video is preprocessed once.

Ideally:

```text
Original video
     ↓
reference frames
     ↓
SSCD index/cache
     +
DISK feature cache
```

Then leak processing should reuse these caches.

Do not repeatedly extract DISK features from reference frames.

Do not rebuild the SSCD index for every leak frame.

Use the existing caching/index mechanisms wherever possible.

# Input design

Use an original/reference video and a corresponding captured/leaked video for which the ground-truth relationship is known.

First inspect the repository for existing capture-simulation and evaluation infrastructure.

Reuse the existing `CaptureSim` and related code rather than inventing a new capture pipeline.

Start with the existing 4-second source/capture setup because it gives us continuity with Stages 2–2c.

If the existing infrastructure naturally supports longer/full videos, also test a longer sequence after the short sanity run.

Do not silently invent ground truth.

# K values

Use:

```text
K = 5, 10, 20, 50
```

K=1 is optional because it provides no temporal headroom and Stage 2 already established its behavior.

Run the same K for every leak frame.

Do not select K separately for different frames.

# Output

Create a new evaluation script, preferably:

```text
src/retrieval/evaluate_stage3a.py
```

Do not break existing evaluation scripts.

For every leak frame, save at least:

```text
leak_frame_index
predicted_original_frame
combined_score
raw_inliers
median_reprojection_error
stage1_rank_of_predicted_frame
```

Also save the top-K candidate information if practical:

```text
candidate_original_frame
candidate_score
candidate_rank
```

This will be useful later when designing temporal alignment.

Write detailed results to:

```text
out/retrieval/stage3a_results.json
```

A compact CSV is also useful:

```text
out/retrieval/stage3a_mapping.csv
```

with at least:

```text
leak_frame, predicted_original_frame, score
```

Do not overwrite previous experiment results.

# Ground truth

For every leak frame where ground truth is known, compare:

```text
predicted_original_frame
vs
ground_truth_original_frame
```

Report:

### Exact accuracy

```text
predicted == ground_truth
```

### ±1 accuracy

```text
abs(predicted - ground_truth) <= 1
```

### ±5 accuracy

```text
abs(predicted - ground_truth) <= 5
```

### Mean absolute frame error

```text
mean(abs(predicted - ground_truth))
```

### Median absolute frame error

Also report the maximum error.

# Temporal trajectory analysis

This is the most important new part.

Create the sequence:

```text
M[i] = predicted_original_frame for leak frame i
```

Analyze:

```text
M[0], M[1], M[2], ...
```

Do NOT alter this sequence.

We want to understand its natural behavior.

Compute:

## 1. Monotonicity

Measure:

```text
M[i+1] >= M[i]
```

and report the percentage of adjacent transitions satisfying this.

Also separately report:

* number of backward jumps
* number of zero jumps
* number of positive jumps

## 2. Jump distribution

Compute:

```text
Δ[i] = M[i+1] - M[i]
```

Report:

* mean
* median
* standard deviation
* min
* max
* percentiles if useful

Especially count:

```text
Δ < 0
Δ = 0
Δ = 1
Δ > 1
Δ > 5
Δ > 10
```

## 3. Ground-truth trajectory

Do the same for:

```text
G[i] = ground_truth_original_frame
```

This tells us what temporal behavior is actually expected.

## 4. Prediction vs ground truth trajectory

Compare:

```text
M[i]
G[i]
```

and calculate residual:

```text
E[i] = M[i] - G[i]
```

Analyze whether errors are:

* isolated spikes
* persistent offsets
* gradually drifting
* systematic speed differences
* local frame duplication
* backward jumps

Do NOT correct any of them yet.

# Linear behavior

The final system is expected to have approximately linear growth of original-frame index with leak-frame index in many cases.

Measure this, but DO NOT use it to improve predictions yet.

Fit:

```text
G[i] ≈ a*i + b
```

and separately:

```text
M[i] ≈ a_hat*i + b_hat
```

Report:

* slope
* intercept
* R²
* residual statistics

This is diagnostic only.

Do not replace predictions with the fitted line.

The purpose is to discover whether the mapping is:

1. approximately linear,
2. piecewise linear,
3. nonlinear,
4. or dominated by isolated matching errors.

# Plotting

Generate useful diagnostic plots.

At minimum:

### Plot 1

```text
x = leak frame index
y = original frame index
```

Plot:

* ground-truth trajectory
* raw Stage-3a predicted trajectory

This is the most important plot.

### Plot 2

Prediction error:

```text
x = leak frame
y = predicted_original - ground_truth_original
```

### Plot 3

Frame-to-frame jump:

```text
x = leak frame
y = M[i+1] - M[i]
```

Save them under something like:

```text
out/retrieval/stage3a/
```

Do not modify predictions merely to make the plots cleaner.

# Performance

This experiment is also the first step toward the eventual production pipeline.

Measure:

* total processing time
* time per leak frame
* SSCD time
* LightGlue/RANSAC time
* number of candidates verified per frame
* reference preprocessing time
* reference DISK cache size if practical
* memory usage if practical

The eventual requirement is approximately linear scaling with the number of leak frames.

So explicitly report:

```text
N leak frames
→ total runtime
→ average runtime/frame
```

If practical, run two different leak-video lengths and compare runtime scaling.

Do not prematurely optimize unless something is obviously being recomputed unnecessarily.

# Important architectural requirement

The eventual production system should be capable of:

```text
reference video
      ↓
preprocess once
      ↓
cache/index

leak video
      ↓
frame 0 → original frame
frame 1 → original frame
frame 2 → original frame
...
frame N → original frame
      ↓
mapping file
```

Avoid an implementation that repeatedly processes the entire reference video for every leak frame.

The current Stage-2 design already caches reference DISK features, so reuse that design.

# What NOT to do

Do NOT implement the final temporal alignment system yet.

Do NOT add:

* DTW
* Viterbi
* dynamic programming
* Kalman filtering
* temporal smoothing
* monotonic constraints
* linear regression correction
* optical flow
* frame interpolation
* previous-frame candidate restriction
* temporal windows

Those will be considered only after we understand the raw Stage-3a trajectory.

# Final analysis

The report should answer:

1. How accurate is independent frame matching over an entire leak video?
2. Is the predicted original-frame sequence mostly monotonic?
3. How often are there catastrophic jumps?
4. Are errors mostly ±1/±5 frame errors or large jumps?
5. Is the leak→original mapping approximately linear?
6. Is there a systematic speed/offset difference?
7. Does K=20/50 materially improve the trajectory compared with K=5/10?
8. Does the combined score behave consistently across the entire sequence?
9. What type of temporal errors remain after the frame-level matcher?
10. What temporal alignment mechanism appears justified by the observed errors?

Do not propose or implement the temporal mechanism until these observations are available.

The main deliverable is NOT a better accuracy number.

The main deliverable is:

> **A complete raw mapping from every leak frame to an original frame, plus enough trajectory/error analysis to tell us exactly what the temporal layer needs to solve.**

Before coding, inspect the repository and existing Stage 2 implementation, then briefly state:

* which existing components will be reused
* how reference-side preprocessing will be cached
* what input leak/reference pair will be used
* what files will be added
* how runtime scaling will be measured

Then implement and run the Stage 3a experiment.
