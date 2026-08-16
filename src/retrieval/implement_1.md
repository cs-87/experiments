For the implementation, make the following adjustments to the architecture described in the report:

### 1. Keep the first experiment deliberately simple

For now, do NOT introduce:

* temporal synchronization assumptions
* FPS/frame-number correspondence
* sliding temporal windows
* DTW
* temporal tracking
* cheap filtering between retrieval and geometric verification
* perspective correction as preprocessing
* any modification or embedding into the original video

I want to test the basic two-stage architecture directly.

### 2. Stage 1 — neural retrieval

Use the global retrieval descriptor (SSCD initially, CopyNCE optionally).

For each leak frame:

```text
leak frame
    ↓
SSCD
    ↓
compare against ALL original-frame embeddings
    ↓
top-K candidates
```

Do not restrict the search using temporal information.

Evaluate several K values, e.g.:

```text
K = 1, 5, 10, 20, 50
```

The important Stage-1 question is:

> Does the correct original frame appear somewhere in the top-K candidates?

### 3. Stage 2 — LightGlue verification

For every leak frame:

1. Extract SuperPoint/DISK features ONCE.
2. Use the already-cached local features of each retrieved original-frame candidate.
3. Run LightGlue against every candidate.

There should be NO additional cheap filtering stage between SSCD retrieval and LightGlue.

For example:

```text
Leak L_t
   │
   ├── SSCD ──► top 20 candidates
   │
   └── SuperPoint ──► leak features ONCE
                         │
                         ├── LightGlue ↔ cached O_k features
                         ├── LightGlue ↔ cached O_k+1 features
                         ├── LightGlue ↔ cached O_k+2 features
                         └── ...
```

Do NOT recompute the original-frame SuperPoint/DISK features at runtime.

### 4. Perspective correction

Do NOT perform perspective correction as a preprocessing stage.

LightGlue should first establish feature correspondences between the leak frame and each candidate original frame.

If homography estimation with RANSAC is useful, calculate it AFTER LightGlue matching and use it as part of the candidate verification/scoring.

The initial output is simply:

```text
leak frame L_t → predicted original frame O_k
```

We can decide later whether to actually warp/rectify the frame for the downstream detector.

### 5. Separate retrieval failure from verification failure

The evaluation should explicitly distinguish:

**Retrieval failure:**

The true `O_k` was not present in the SSCD top-K.

versus

**Verification failure:**

The true `O_k` was present in the top-K, but LightGlue selected another candidate.

This distinction is important because it tells us whether we need a better global descriptor or a better geometric verifier.

### 6. Performance

Do not prematurely optimize the candidate count.

Initially, if SSCD gives top-20, simply run LightGlue against all 20 candidates.

The purpose of this experiment is to establish whether the architecture works before introducing optimizations.

However, cache all original-frame local features so that runtime cost is approximately:

```text
one local-feature extraction for the leak frame
+
K LightGlue matching operations
```

rather than recomputing feature extraction for every candidate.

### 7. Desired final pipeline

The first implementation should therefore be exactly:

```text
                 ORIGINAL VIDEO
                       │
              ┌────────┴────────┐
              ▼                 ▼
          SSCD embedding    SuperPoint
              │                 │
              ▼                 ▼
        save embeddings    save features
              │                 │
              └────────┬────────┘
                       │
                       │
                  LEAK FRAME
                       │
              ┌────────┴────────┐
              ▼                 ▼
           SSCD             SuperPoint
              │                 │
              ▼                 │
           top-K                │
              │                 │
              └────────┬────────┘
                       ▼
                  LightGlue
              against cached
              original features
                       │
                       ▼
             geometric scoring
                       │
                       ▼
              predicted O_k
```

Keep this as the baseline. Do not add temporal priors, cheap filtering, or perspective preprocessing until this experiment has been measured.
