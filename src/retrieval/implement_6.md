We are now moving from the experimental temporal stages toward the first production-oriented temporal alignment implementation.

Read the latest reports and existing implementation first, especially:

- `STAGE2C_REPORT.md`
- `STAGE3A_REPORT.md`
- `STAGE3B_REPORT.md`
- `src/retrieval/lightglue_verify.py`
- `src/retrieval/evaluate_stage3a.py`
- `src/retrieval/stage3b_temporal.py`
- any existing Stage 2/3 evaluation code and output formats

Do not modify the previous experiments. They must remain reproducible.

==================================================
GOAL
==================================================

The final system should take:

    original/reference video
    +
    leak/captured video

and produce, for every leak frame:

    estimated original-frame position
    +
    ranked nearby original-frame candidates

The temporal alignment component must be completely independent of the downstream watermark detector.

The temporal aligner must NOT know:

- what watermark is being detected
- what patches are being examined
- what the detector score means
- how the detector works

Its only responsibility is:

> "Given this leak frame, where in the original video should the downstream system look?"

The downstream detector will consume the output later.

==================================================
CURRENT FINDINGS
==================================================

Stage 2 established the visual matcher:

    SSCD
      ↓
    Top-K
      ↓
    DISK
      ↓
    LightGlue
      ↓
    RANSAC
      ↓
    combined score

The frozen Stage-2 score is:

    combined =
        num_inliers /
        (1 + median_inlier_reprojection_error)

Stage 2c showed that this combined score generalizes better than raw RANSAC inlier count.

Stage 3a then ran the matcher independently across an entire leak video.

The important result is that the resulting leak→original trajectory is already approximately linear/monotonic, with mostly local errors and occasional large outliers.

On the 600-frame long/mild experiment:

- exact accuracy @ K=50 = 34.3%
- ±1 = 59.0%
- ±5 = 87.2%
- median absolute error = 1 frame
- mean absolute error = 2.60 frames

Stage 3b added a cheap local temporal correction.

At K=50 on the same long/mild sequence:

- Variant B exact = 35.7%
- Variant B ±1 = 62.8%
- Variant B ±5 = 91.5%
- mean absolute error = 1.86 frames

The important conclusion is:

> We do NOT need to obsess over exact-frame prediction.

For the eventual application, the temporal aligner's much more important job is to place the correct original frame inside a small candidate neighborhood.

The key target is therefore:

    high recall within ±5 frames

rather than:

    perfect exact-frame accuracy

==================================================
NEXT SYSTEM
==================================================

Implement:

    Stage 4A — Production-oriented streaming temporal aligner

Do NOT build the watermark detector into this stage.

Do NOT call the watermark detector.

Do NOT import any watermark/detection code.

==================================================
CORE ARCHITECTURE
==================================================

The architecture should be:

    ORIGINAL VIDEO
          │
          ├── reference frame preprocessing
          ├── SSCD index/cache
          └── DISK feature cache
                    │
                    ▼
               TemporalAligner
                    ▲
                    │
              LEAK VIDEO
                    │
                    ▼
              frame by frame
                    │
                    ▼
          SSCD → Top-K → LightGlue
                    │
                    ▼
             combined scores
                    │
                    ▼
             temporal state
                    │
                    ▼
          estimated original position
                    │
                    ▼
          ranked candidate region
                    │
                    ▼
               output file

The downstream detector is completely outside this system.

==================================================
IMPORTANT DESIGN DECISION
==================================================

Do NOT implement full-video DTW or another large global alignment algorithm yet.

The observed data does not justify it.

The mapping is already approximately linear and the dominant errors are local.

Instead, implement a lightweight streaming/stateful temporal model.

The aligner should maintain state such as:

- current estimated original position
- estimated local step/speed
- confidence
- optionally recent accepted positions

The state must be based only on observations already seen.

Do not use future frames in the streaming mode unless explicitly implementing a separate offline mode.

==================================================
VISUAL MATCHING
==================================================

Keep the visual matcher frozen.

For every leak frame:

1. compute SSCD embedding
2. retrieve Top-K reference candidates
3. run DISK/LightGlue/RANSAC
4. calculate the existing combined score

Use:

    K = 50

as the default production candidate-generation value.

Keep K configurable.

Do not introduce another visual score.

Do not modify:

- SSCD
- DISK
- LightGlue
- RANSAC
- combined score

Reference DISK features must be precomputed/cached once.

Reference SSCD embeddings/index must be built once.

Do not process the entire reference video repeatedly for every leak frame.

==================================================
TEMPORAL MODEL
==================================================

The temporal model should produce an estimated original position for each leak frame.

Start with a simple robust state update.

At the beginning, there may be no reliable temporal state.

Therefore:

- bootstrap from the visual match
- establish an initial position and local step
- update the state as new leak frames arrive

The model should prefer candidates that:

1. have strong visual evidence
2. are reasonably consistent with the current temporal trajectory

But temporal consistency must be a SOFT prior.

Do NOT hardcode:

    original_frame = leak_frame

Do NOT hardcode:

    next_original = previous_original + 1

The real capture may eventually contain:

- dropped frames
- duplicated frames
- FPS mismatch
- variable playback speed
- local timing irregularities

The system must be capable of handling those.

==================================================
CANDIDATE OUTPUT
==================================================

For each leak frame i, produce:

    estimated_original_frame

and a ranked candidate neighborhood.

Default:

    radius = 5

Therefore if:

    estimated_original_frame = 503

the candidate region is approximately:

    498 ... 508

Handle beginning/end-of-video boundaries correctly.

The candidate list should be ranked according to temporal plausibility, not watermark evidence.

The output must make it possible for a downstream consumer to simply ask:

    "Which original frames should I inspect for leak frame i?"

==================================================
IMPORTANT: DO NOT CONFUSE "MAPPING" WITH "CANDIDATES"
==================================================

The aligner should distinguish:

1. estimated center

    leak frame 500
    → estimated original 503

2. candidate region

    [498,499,500,501,502,503,504,505,506,507,508]

The center does NOT have to be the exact ground-truth frame.

The candidate region is the important output.

==================================================
CANDIDATE RANKING
==================================================

Use the visual candidate information already produced by Stage 2/3.

Do not rerun LightGlue just to rank candidates.

If the estimated center is 503, candidates near 503 should receive higher temporal priority, while still allowing strong visual candidates to remain available.

Do not simply output:

    center-5, ..., center+5

in arbitrary numerical order.

Produce an explicit ranking.

For example:

    503
    502
    504
    501
    505
    ...

The ranking should come from the temporal model and existing visual evidence.

However, do NOT incorporate any watermark/detector score.

==================================================
STREAMING API
==================================================

Design the implementation around a stateful API similar to:

    aligner = TemporalAligner(reference_data)

    for leak_frame in leak_video:
        result = aligner.process(leak_frame)
        write_result(result)

The exact API can differ based on the existing repository architecture, but the important property is:

> The aligner processes leak frames incrementally and does not need the entire leak video loaded into memory.

Provide a way to reset/reinitialize the temporal state.

Also provide an offline evaluation driver that uses the exact same alignment logic.

==================================================
OUTPUT FORMAT
==================================================

Create a clean machine-readable output.

Prefer JSONL or SQLite for production-scale streaming output rather than keeping the entire result in memory.

Each leak frame should contain at least:

    leak_frame
    estimated_original_frame
    candidate_frames
    candidate_temporal_scores
    confidence

Also preserve useful debugging information:

    raw_visual_prediction
    raw_visual_score
    temporal_adjusted_prediction

if available.

Do NOT include watermark detector information.

The output should be incrementally writable so that a long video does not need to wait until the entire video finishes before results are persisted.

==================================================
PRODUCTION REQUIREMENT: LINEAR SCALING
==================================================

The system should scale approximately linearly with the number of leak frames.

The reference side should be processed once.

The leak side should be processed approximately once per frame.

Avoid:

    O(N_leak × N_reference)

full rescans.

The intended expensive operation is:

    SSCD retrieval
    +
    LightGlue on Top-K candidates

Everything after that should be cheap.

Measure:

- reference preprocessing time
- SSCD time/frame
- LightGlue time/frame
- temporal alignment time/frame
- total time/frame
- total time
- peak memory
- reference cache size

The temporal layer should remain negligible compared with LightGlue.

==================================================
EVALUATION
==================================================

Use the existing synthetic datasets first so the new implementation can be compared directly against Stage 3a/3b.

At minimum:

- short4s/mild
- short4s/moderate
- long1min600/mild

Use:

    K = 50
    radius = 5

as the primary configuration.

Also make K and radius configurable.

Evaluate:

### Exact center accuracy

    estimated_original == ground_truth

### Candidate recall @ ±1

    true_original inside candidate region radius 1

### Candidate recall @ ±3

### Candidate recall @ ±5

This is now the PRIMARY metric.

### Candidate recall @ ±10

Useful as a secondary diagnostic.

Also report:

- mean center error
- median center error
- maximum center error
- monotonicity
- backward jumps
- large temporal jumps

But do NOT optimize the system around exact accuracy.

==================================================
CRITICAL METRIC
==================================================

The most important number is:

    candidate_recall@±5

Interpretation:

> For what fraction of leak frames does the output candidate set contain the true original frame?

This is the metric that matters for the eventual downstream consumer.

If the center is off by 3 frames but the correct frame is ranked #2 inside the ±5 candidate region, that should count as a successful alignment for this stage.

==================================================
RANKING METRICS
==================================================

Because the candidate list is ranked, also report:

    recall@1
    recall@3
    recall@5
    recall@11

where these refer to the first N temporal candidates returned to the downstream consumer.

Be explicit about the distinction between:

- radius-based candidate recall
- top-N ranked candidate recall

For example:

    ±5 recall = true frame exists somewhere in the 11-frame neighborhood

while:

    top-3 recall = true frame appears in the first 3 ranked candidates

This will be useful later.

==================================================
REAL-WORLD ROBUSTNESS
==================================================

Do not implement special handling for real camera artifacts yet.

But the architecture must not assume:

    leak frame i == original frame i

or:

    FPS is identical

The temporal state should represent a general mapping:

    original_position ≈ f(leak_position)

rather than assuming identity.

If the current synthetic tests use identity mapping, that is only the evaluation setup.

==================================================
SEPARATE OFFLINE AND STREAMING MODES
==================================================

If useful, implement two modes:

### Streaming mode

Only past/current information is available.

Suitable for production processing.

### Offline mode

The complete leak-video candidate sequence may be available.

This can later support stronger temporal smoothing.

For this stage, the primary implementation should be streaming.

Do not add a separate offline algorithm unless it is essentially free.

==================================================
FAILURE / RECOVERY BEHAVIOR
==================================================

The aligner must not become permanently locked onto a bad prediction.

For example:

    correct
    correct
    catastrophic visual mismatch
    correct
    correct

should recover automatically.

Likewise, if the visual matcher provides strong evidence that the trajectory has genuinely changed, the temporal model must be able to follow it.

Do not create an irreversible temporal lock.

==================================================
NO WATERMARK DETECTOR
==================================================

This is a hard architectural boundary.

The temporal aligner must not:

- import detector code
- call detector code
- inspect watermark patches
- inspect detector scores
- know which frame contains a watermark
- use watermark evidence for ranking

Its output is purely:

    leak frame
        →
    original-frame estimate
        →
    ranked candidate original frames

==================================================
FILES
==================================================

Prefer adding something like:

    src/retrieval/temporal_aligner.py

for the reusable production component.

Then add:

    src/retrieval/evaluate_stage4_temporal.py

for evaluation.

Do not modify previous experiment scripts unless absolutely necessary.

Keep the temporal aligner independent from the evaluation driver.

==================================================
SANITY CHECKS
==================================================

Before running the full evaluation:

1. Verify reference preprocessing happens only once.
2. Verify no LightGlue call occurs inside the temporal-ranking logic.
3. Verify no ground truth enters the production aligner.
4. Verify no watermark/detector code is imported.
5. Verify results can be written incrementally.
6. Verify restarting the aligner resets temporal state cleanly.
7. Verify beginning/end frame boundaries work.
8. Verify a bad visual prediction does not permanently corrupt the state.
9. Verify runtime grows approximately linearly with leak-frame count.

==================================================
FINAL REPORT
==================================================

Create:

    STAGE4A_REPORT.md

Explain:

1. Production architecture
2. Temporal state representation
3. Candidate-generation process
4. Candidate-ranking process
5. Streaming behavior
6. Runtime scaling
7. Memory usage
8. Exact center accuracy
9. ±1 / ±3 / ±5 / ±10 candidate recall
10. Top-1 / top-3 / top-5 / top-11 recall
11. Failure/recovery behavior
12. Comparison against Stage 3a and Stage 3b
13. Limitations
14. What remains before real-camera validation

The conclusion should focus on:

> Does this component reliably produce a small candidate region containing the correct original frame while operating independently of the downstream detector?

Do not add watermark detection.

Do not implement DTW.

Do not add a complicated global temporal optimizer unless the current data demonstrates that the streaming model fundamentally fails.

The primary objective is now to turn the experimentally validated pieces into a clean, reusable, streaming temporal-alignment component.