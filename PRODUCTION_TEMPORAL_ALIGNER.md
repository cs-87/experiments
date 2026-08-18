# Temporal Aligner — Usage Guide

Given a reference (original) video and a stream of leak frames, tells you which small region of the
original video each leak frame most likely corresponds to. It never inspects a watermark or detector —
its only job is: *"given this leak frame, where should the downstream system look?"*

```
reference video --(once)--> VisualMatcher: SSCD index + DISK cache
                                                    |
leak video --(per frame, streaming)--> VisualMatcher.match() --> (candidate_idx, combined_scores)
                                                    |
                                     TemporalAligner.process(visual_candidates=...)
                                                    |
                                estimated_original_frame + ranked ±radius candidate region
```

## Files

| File | Purpose |
|---|---|
| `src/retrieval/temporal_aligner.py` | The component: `VisualMatcher` (SSCD + DISK/LightGlue/RANSAC visual matching, built once) and `TemporalAligner` (streaming position/velocity/uncertainty filter + candidate ranking). |
| `src/retrieval/evaluate_production_integration.py` | Minimal runnable example — reference video in, leak frames out, incrementally-written JSONL mapping file. Run this first to confirm your environment works. |
| `src/retrieval/encoders.py`, `index.py`, `lightglue_verify.py`, `capture_sim.py`, `evaluate.py`, `evaluate_stage2_scores.py` | Supporting modules `temporal_aligner.py` imports (SSCD encoder, exact-search index, DISK/LightGlue verification, frame loading, the frozen `combined` visual score). Not called directly. |

## Requirements

Python with `torch`, `opencv-python`, `numpy`, and [`lightglue`](https://github.com/cvg/LightGlue)
(`pip install git+https://github.com/cvg/LightGlue.git`) installed and CUDA available for real-time
use (CPU works via `device="cpu"`, just much slower). The SSCD encoder checkpoint is downloaded
automatically on first use.

## Quickstart

```python
from retrieval.evaluate import load_frames
from retrieval.temporal_aligner import VisualMatcher, TemporalAligner

reference_frames, _ = load_frames("reference_video.mp4")

matcher = VisualMatcher(reference_frames, k=50)   # builds the SSCD index + DISK cache once
aligner = TemporalAligner(radius=5)                # no reference frames needed here

for leak_frame in leak_video_frames:               # e.g. read one at a time with cv2.VideoCapture
    candidate_idx, combined_scores, _timing = matcher.match(leak_frame)
    result = aligner.process(visual_candidates=(candidate_idx, combined_scores))

    print(result["estimated_original_frame"])   # single best-guess center
    print(result["candidate_frames"])           # ranked ±5 region, e.g. [503,502,504,501,505,...]
    print(result["confidence"])                 # 0..1

aligner.reset()   # start a new leak video against the same reference (caches untouched)
```

A convenience all-in-one mode also exists if you don't need the two components separately:

```python
aligner = TemporalAligner(reference_frames, k=50, radius=5)   # builds its own internal VisualMatcher
result = aligner.process(leak_frame)                          # pass the raw frame directly
```

Run `python3 src/retrieval/evaluate_production_integration.py` for a working end-to-end demo
(uses `4_sec_source.mp4` as a stand-in reference/leak pair) that writes one JSON line per frame.

## Output per frame

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

`candidate_frames` — the ranked region a downstream consumer should actually inspect — is the important
output, not `estimated_original_frame` alone; it does not need to be the exact ground-truth frame.

## Configuration

- `k` (default 50): size of the SSCD shortlist verified by LightGlue per frame. Larger = better
  shortlist recall, more LightGlue cost per frame.
- `radius` (default 5): half-width of the returned candidate region (11 frames total at the default).
- Everything else (`T_GATE`, `Q_PROCESS`, `R0_MEASUREMENT`, `SIGMA_INIT`, `MISMATCH_RESET_STREAK`,
  `LARGE_MISMATCH_MULTIPLIER`) is a fixed constant at the top of `temporal_aligner.py`, derived from
  measurement rather than fit to any single test sequence — see the comments beside each for the
  reasoning. Do not tune these against a specific video; they are meant to generalize.

## Notes

- `.process()` never receives ground truth, a frame index, or anything related to a downstream
  detector — only the leak frame (or its precomputed visual candidates).
- `.reset()` clears position/velocity/uncertainty and leaves the reference-side caches untouched — use
  it between separate leak videos against the same reference.
- The filter tolerates dropped/duplicated frames and speed mismatch (it learns velocity, it does not
  assume `original_frame == leak_frame` or a fixed step of 1) and recovers from both isolated visual
  mismatches and genuine discontinuities without becoming permanently stuck.
