# Design a State-of-the-Art Detector for My 32-Bit Spread-Spectrum Video Watermark

I am building a robust video watermark detector.

I currently have a working **1-bit / 1-PRN spread-spectrum watermarking system**, and I am now retrofitting it to support a **32-bit watermark**.

I want you to act as a senior researcher specializing in:

* digital watermarking
* spread-spectrum communications
* statistical signal detection
* error-correcting codes
* computer vision
* robust video processing
* neural network-based signal detection

Your task is to **design a genuinely strong detector specifically for my exact embedding system**.

Do not immediately propose a generic CNN or Transformer.

First inspect the project files I provide, understand the actual implementation, mathematically model the watermark signal, establish the strongest classical baseline, and then determine where machine learning provides a real advantage.

---

# 1. PROJECT FILES
relevant project files inside src/spread_spectrum/ and thier import files

**Do not recreate or guess the implementation from this prompt if the actual files are available.**

Inspect the supplied files and base the detector design on the actual implementation.

If something important is missing, explicitly identify the missing file/information rather than assuming its behavior.

---

# 2. CURRENT WATERMARKING SYSTEM

The original system successfully embeds:

```
1 bit
  ↓
1 PRN
  ↓
ALPHA × PRN
  ↓
DWT LL embedding
```

I am now extending this to:

```
32 bits
  ↓
32 PRNs
  ↓
signed combination of PRNs
  ↓
normalize combined watermark
  ↓
ALPHA × combined watermark
  ↓
DWT LL embedding
```

The exact implementation should be obtained from the supplied project files.

---

# 3. PRN STRUCTURE

There are at least 32 deterministic PRNs.

The detector knows:

* PRN generation algorithm
* seed
* complete PRN codebook
* patch size
* DWT configuration
* embedding algorithm
* approximate ALPHA

The individual PRNs are designed to have:

* values approximately/strictly in {-1, +1}
* zero DC
* RMS = 1
* identical dimensions
* approximate orthogonality

Let the PRNs be:

```
P1, P2, ..., P32
```

with approximately:

```
RMS(P_i) = 1

mean(P_i) = 0
```

and:

```
<P_i, P_j> ≈ 0
```

for i != j.

---

# 4. BIT REPRESENTATION

For each bit:

```
bit = 1 → +P_i

bit = 0 → -P_i
```

Therefore the watermark is:

```
W = s1 P1 + s2 P2 + ... + s32 P32
```

where:

```
s_i ∈ {-1,+1}
```

The actual implementation of this combination should be verified from the supplied PRN generator file.

---

# 5. IMPORTANT: COMBINED WATERMARK NORMALIZATION

This is a critical part of the system.

Each individual PRN has RMS approximately equal to 1.

If the 32 PRNs are approximately orthogonal, then:

```
RMS(W) ≈ sqrt(32) ≈ 5.657
```

The combined watermark is then normalized back to:

```
RMS(W_norm) = 1
```

before embedding.

Thus the final embedding is effectively:

```
LL_watermarked = LL_original + ALPHA × W_norm
```

The normalization is intentional because **perceptual quality is a hard constraint**.

I cannot simply leave the combined watermark at RMS ≈ sqrt(32), because that would substantially increase the overall embedding energy compared with my existing 1-bit system.

My existing ALPHA was calibrated with an RMS-1 watermark.

Therefore:

```
ALPHA
```

should be interpreted primarily as the **overall watermark perturbation strength**.

Do not casually multiply ALPHA by sqrt(32).

If you recommend changing ALPHA, quantify the robustness/perceptual tradeoff.

---

# 6. SPATIAL EMBEDDING

The watermark is embedded into DWT LL coefficients of selected image patches.

The current patch size is:

```
128 × 128
```

The current DWT is:

```
level = 1
```

The watermark is embedded into:

```
DWT.LL
```

The selected patches are obtained from SIFT-based patch extraction.

The same 32-bit watermark is embedded into multiple selected patches across frames.

Inspect the actual implementation to determine precisely:

* how SIFT patches are selected
* how patch coordinates are generated
* whether patches overlap
* how many patches typically exist per frame
* whether patch dimensions are always identical
* whether patch positions depend on image content
* whether there are additional transformations before embedding

---

# 7. VIDEO REDUNDANCY

The same watermark is embedded repeatedly:

* across many spatial patches
* across many video frames

This provides substantial redundancy.

The detector should exploit both dimensions.

Do not make a hard decision independently for every patch and then blindly majority vote.

Preserve soft evidence for as long as possible.

---

# 8. VERY IMPORTANT: FINITE WATERMARK UNIVERSE

There is a major production constraint.

I will **not use arbitrary 32-bit watermark strings**.

I have approximately:

```
200,000
```

valid 32-bit watermark strings.

At detection time, the detector can be given the **complete list of all approximately 200,000 valid watermark strings**.

Therefore this is fundamentally a:

```
CLOSED-SET CODEWORD IDENTIFICATION
```

problem.

The detector does not necessarily need to independently recover arbitrary 32-bit values.

Instead, it can determine:

> Which of the approximately 200,000 valid watermark codewords best explains the observed signal?

This constraint should be exploited heavily.

---

# 9. CLOSED-SET DETECTION

Compare these approaches:

### A. Independent bit recovery

Estimate:

```
bit_0 ... bit_31
```

then reconstruct the ID.

### B. Direct codeword identification

Score the observation against each of the 200,000 valid watermark strings.

### C. Hybrid detection

Extract 32 soft PRN measurements and compare those measurements against all 200,000 valid codewords.

### D. Learned codeword retrieval

Map the observed watermark evidence into an embedding space and retrieve the closest valid codeword.

### E. Hybrid signal processing + neural network + finite codebook

Do not assume any approach is optimal.

Analyze them mathematically and experimentally.

---

# 10. CLASSICAL MATCHED-FILTER ANALYSIS

Because the watermark is:

```
W = Σ s_i P_i
```

and the PRNs are approximately orthogonal, rigorously derive:

```
<W, P_k>
```

for the correct and incorrect PRNs.

Analyze:

* ideal orthogonality
* imperfect orthogonality
* finite PRN length
* normalization
* embedding strength
* attack noise
* interference between PRNs

Explicitly derive the expected per-bit detection SNR.

---

# 11. EFFECT OF NORMALIZATION

Analyze this carefully.

Before normalization:

```
RMS(W) ≈ sqrt(32)
```

After normalization:

```
RMS(W_norm) = 1
```

Therefore each individual PRN contributes substantially less energy than in the original 1-bit system.

Approximately, under ideal orthogonality:

```
per-PRN amplitude contribution scales like:

    1 / sqrt(32)
```

relative to an individual RMS-1 watermark.

Determine:

* how much per-bit SNR is lost
* how total watermark energy behaves
* whether temporal aggregation compensates for this
* whether spatial aggregation compensates for this
* whether the finite codeword constraint compensates for this
* whether another embedding allocation would be better

Perception must remain a hard constraint.

---

# 12. INDIVIDUAL PRN VS COMBINED PRN

Analyze the difference between:

```
correlation(observation, P_i)
```

and:

```
correlation(observation, W)
```

where W corresponds to a particular 32-bit codeword.

The combined PRN represents one particular watermark ID.

The individual PRN correlations provide a 32-dimensional soft evidence vector.

Determine which representation contains more useful information for the detector.

Also analyze how the approximately 200,000 valid codewords change the answer.

---

# 13. SOFT 32-D EVIDENCE

Suppose the detector obtains:

```
c = [c1,c2,...,c32]
```

where each:

```
c_i
```

is the correlation/evidence associated with PRN_i.

Each valid watermark codeword can be represented as:

```
q_j = [q1,q2,...,q32]
```

where:

```
q_i ∈ {-1,+1}
```

Investigate whether the correct codeword score is:

```
score(q_j) = c · q_j
```

or whether a better statistically principled score should be used.

Analyze:

* normalized dot product
* Euclidean distance
* weighted correlation
* covariance-aware scoring
* Mahalanobis distance
* reliability-weighted likelihood
* learned scoring

Do not prematurely convert continuous correlation values into hard bits.

---

# 14. 200,000-CODEWORD SEARCH

Approximately:

```
200,000 × 32
```

values are involved.

An exhaustive matrix operation such as:

```
CODEWORDS @ evidence
```

may be computationally trivial.

Analyze the actual computational complexity.

Do not introduce approximate nearest-neighbor methods, hashing, or large classifiers unless there is a demonstrated reason to do so.

Exact exhaustive search should be the baseline.

---

# 15. UNKNOWN / NO-WATERMARK DETECTION

This is critical.

The detector must NOT always return one of the 200,000 valid IDs.

Even an unwatermarked video will have some codeword that gives the highest score by chance.

The detector therefore needs:

```
WATERMARK PRESENT
```

versus:

```
NO VALID WATERMARK
```

as a separate hypothesis test.

Analyze:

* null distribution
* maximum score over 200,000 hypotheses
* best-vs-second-best margin
* absolute score
* false-positive probability
* false-negative probability
* multiple-hypothesis testing
* calibration using unwatermarked videos

The final system should be capable of returning:

```
NO_WATERMARK
```

rather than forcing the closest valid codeword.

---

# 16. CODEWORD DESIGN

Because I control the approximately 200,000 valid watermark IDs, investigate whether they should be deliberately designed as an error-correcting code.

Instead of arbitrary 32-bit values, I may be able to choose 200,000 codewords with a useful minimum Hamming distance.

Analyze:

* achievable minimum Hamming distance
* number of codewords possible at different distances
* correction capability
* relationship between Hamming distance and PRN correlation
* whether random codewords are sufficient
* BCH-like constructions
* custom codebook construction
* constant-weight constructions
* whether codeword design materially improves robustness

This could be extremely important.

---

# 17. PATCH-LEVEL DETECTION

The detector should produce soft evidence from individual candidate patches.

A potential pipeline is:

```
candidate patch
    ↓
DWT
    ↓
preprocessing
    ↓
32 PRN matched filters
    ↓
32-dimensional soft evidence
```

But investigate alternatives.

Consider:

* direct DWT-domain correlation
* pixel-domain preprocessing
* correlation maps
* local spatial search
* multi-scale search
* learned feature extraction
* hybrid signal-processing/neural models

---

# 18. PATCH LOCALIZATION

The watermark is embedded into SIFT-derived patches.

After attacks, exact patch locations may no longer be identical.

Analyze robust localization strategies:

* SIFT
* SuperPoint
* DISK
* LightGlue
* LoFTR
* dense sliding windows
* multi-scale search
* learned keypoint detectors
* watermark-correlation-based localization
* temporal patch tracking

Compare them in terms of:

* geometric robustness
* computational cost
* scaling robustness
* crop robustness
* rotation robustness
* implementation complexity

Determine whether the best approach is:

```
feature-based localization
```

or:

```
dense watermark search
```

or:

```
a hybrid.
```

---

# 19. TEMPORAL AGGREGATION

The same watermark exists throughout the video.

Suppose each patch produces:

```
c_patch ∈ R^32
```

A frame produces:

```
c_frame ∈ R^32
```

and the entire video produces:

```
c_video ∈ R^32
```

Determine the strongest aggregation method.

Compare:

* mean
* weighted mean
* median
* trimmed mean
* Huber aggregation
* robust M-estimators
* confidence-weighted aggregation
* RANSAC-style aggregation
* temporal filtering
* learned aggregation
* temporal neural networks
* transformers, if justified

The detector should retain soft evidence.

---

# 20. PATCH RELIABILITY

Not every SIFT patch will necessarily be equally useful.

Some patches may be:

* low texture
* heavily compressed
* badly localized
* partially cropped
* blurred
* geometrically distorted
* corrupted

Design a method to estimate:

```
reliability(patch)
```

and use it when aggregating watermark evidence.

Investigate whether this reliability should be:

* handcrafted
* correlation-based
* estimated from signal statistics
* learned using a small neural network

---

# 21. ATTACK MODEL

The detector should be robust to realistic degradation.

At minimum consider:

### Compression

* H.264
* H.265
* bitrate sweeps
* different encoder settings
* quantization

### Scaling

* 1080p → 720p
* 1080p → 540p
* 1080p → 360p
* arbitrary downscaling
* upscaling

### Spatial degradation

* blur
* sharpening
* denoising
* noise
* crop
* translation
* rotation
* perspective distortion

### Photometric degradation

* brightness
* contrast
* gamma
* saturation
* grayscale
* color-space changes

### Temporal degradation

* frame dropping
* frame duplication
* frame-rate changes
* temporal shifts
* interpolation

### Combined attacks

Examples:

```
resize + H264 + blur + crop
```

or:

```
resize + compression + photometric changes + frame dropping
```

Camera capture can be investigated later, but first optimize the detector for digital attacks.

---

# 22. TRAINING DATA

I can generate essentially unlimited synthetic training data.

The basic process is:

```
original video
    ↓
choose one of the valid watermark IDs
    ↓
embed watermark
    ↓
apply attack/degradation
    ↓
training sample
```

Randomize:

* source videos
* watermark IDs
* patch locations
* compression
* resolution
* attacks
* attack combinations
* temporal distortions

The model must not memorize source videos.

Investigate:

* curriculum learning
* hard-negative mining
* attack mixtures
* adversarial augmentation
* difficult codeword pairs
* out-of-distribution attacks

---

# 23. NEURAL NETWORK ROLE

I want a state-of-the-art detector, but I do NOT want deep learning used just because it sounds modern.

The known PRNs are valuable side information.

The neural network should exploit them.

Compare:

1. Pure matched filtering
2. Matched filtering + robust aggregation
3. CNN on image patches
4. CNN on DWT coefficients
5. CNN + PRN correlations
6. CNN + PRN correlation maps
7. Siamese networks
8. Metric learning
9. Transformer
10. Learned matched filtering
11. Signal-processing + neural evidence correction
12. Signal-processing + neural reliability estimation
13. Temporal neural aggregation
14. Hybrid signal-processing + neural + codeword decoder

Determine which architecture you would actually build.

---

# 24. IMPORTANT: DO NOT MAKE THE NETWORK REDISCOVER THE PRNs

The detector knows the exact PRNs.

Therefore a neural network should not be forced to learn the watermark pattern from raw pixels if classical signal processing can explicitly provide the PRN evidence.

Investigate architectures such as:

```
attacked patch
    ↓
DWT
    ↓
32 matched filters
    ↓
32-D soft evidence
    ↓
neural correction/calibration
    ↓
robust aggregation
    ↓
200k codeword search
```

Also investigate whether the network should additionally receive:

* DWT coefficients
* correlation maps
* local texture statistics
* patch quality metrics

---

# 25. LEARNED EVIDENCE CORRECTION

One particularly interesting architecture is:

```
attacked patch
    ↓
classical PRN correlations
    ↓
raw 32-D evidence
    ↓
small neural network
    ↓
corrected/calibrated 32-D evidence
    ↓
codeword decoder
```

Analyze whether this is superior to:

```
attacked patch → giant neural network → watermark ID
```

The goal is to use machine learning for what classical signal processing struggles with:

* compression distortion
* nonlinear distortions
* geometric misalignment
* local texture effects
* correlation bias
* patch reliability

---

# 26. CODEWORD-AWARE DETECTION

Because there are only approximately 200,000 valid IDs, investigate whether the detector should optimize directly for:

```
correct valid-codeword identification
```

rather than:

```
independent bit accuracy.
```

These are not necessarily equivalent objectives.

For example, a detector might have:

```
95% average bit accuracy
```

but poor exact 32-bit recovery.

Conversely, codeword constraints may allow correct ID recovery even when some individual bit estimates are uncertain.

Analyze this carefully.

---

# 27. TEMPORAL + CODEWORD DECODING

Do not necessarily identify a watermark independently from each frame.

Instead investigate:

```
patch evidence
    ↓
frame-level evidence
    ↓
temporal aggregation
    ↓
video-level 32-D evidence
    ↓
exhaustive search over 200,000 valid codewords
    ↓
statistical acceptance test
```

Compare this with:

* frame-level codeword voting
* patch-level codeword voting
* direct correlation accumulation
* confidence-weighted evidence accumulation
* learned temporal aggregation

---

# 28. ERROR-CORRECTING CODES

Currently the payload is a raw 32-bit ID.

I may later introduce ECC.

Analyze:

* hard bit decisions + ECC
* soft bit decisions + ECC
* codeword-constrained decoding
* incorporating ECC into the 200k codebook itself

Quantify the relationship between:

* BER
* Hamming distance
* exact ID recovery
* number of valid codewords
* correction capability

Do not assume ECC automatically solves weak detection.

---

# 29. PERCEPTUAL QUALITY

Perception is a hard constraint.

Evaluate:

* PSNR
* SSIM
* LPIPS if useful
* frame-level visual difference
* temporal visual artifacts

The central optimization problem is:

```
maximize robustness
```

subject to:

```
acceptable perceptual distortion
```

Do not recommend simply increasing ALPHA until detection works.

---

# 30. EVALUATION

Design a rigorous experiment.

Measure:

* bit accuracy
* BER
* exact 32-bit recovery
* valid-codeword recovery
* false-positive rate
* false-negative rate
* precision
* recall
* ROC
* AUC
* confidence calibration
* best-vs-second-best margin
* patch-level accuracy
* frame-level accuracy
* video-level accuracy

Produce attack-vs-performance curves.

Evaluate:

1. no attack
2. H264 bitrate sweep
3. H265 bitrate sweep
4. resolution sweep
5. blur sweep
6. noise sweep
7. resize sweep
8. crop sweep
9. rotation sweep
10. grayscale
11. brightness
12. contrast
13. saturation
14. gamma
15. combined attacks

Compare the 32-bit system against the existing 1-bit baseline.

---

# 31. MOST IMPORTANT MATHEMATICAL QUESTIONS

Answer these rigorously before finalizing the architecture:

1. Given 32 approximately orthogonal RMS-1 PRNs, what is the expected RMS of their sum?

2. After normalizing the sum back to RMS=1, what happens to the contribution of each PRN?

3. What is the expected matched-filter output for the correct PRN?

4. What is the expected matched-filter output for an incorrect PRN?

5. What is the per-bit SNR after normalization?

6. What is the total watermark SNR?

7. How does spatial averaging across patches affect SNR?

8. How does temporal averaging across frames affect SNR?

9. Does temporal aggregation recover the approximately 1/sqrt(32) per-bit energy reduction?

10. Does closed-set decoding over 200,000 codewords provide a meaningful robustness advantage?

11. Is dot-product codeword matching the maximum-likelihood detector under reasonable noise assumptions?

12. When should correlation values be weighted?

13. How should imperfect PRN orthogonality be handled?

14. Is individual PRN correlation fundamentally more useful than correlation against the combined PRN?

15. How should NO-WATERMARK detection be performed statistically?

16. What minimum Hamming distance is achievable for approximately 200,000 32-bit codewords?

17. Would deliberately designing the 200,000 IDs as an error-correcting code materially improve detection?

18. Can perceptual quality remain fixed while improving per-bit detectability?

19. Is equal energy per bit optimal?

20. What is the strongest detector you would actually build for this system?

---

# 32. RECOMMENDED DEVELOPMENT STRATEGY

I want a staged implementation.

## PHASE 1 — Classical baseline

Build the mathematically correct baseline:

```
patch localization
    ↓
DWT
    ↓
32 PRN matched filters
    ↓
32-D soft evidence
    ↓
spatial aggregation
    ↓
temporal aggregation
    ↓
200k exhaustive codeword search
    ↓
NO-WATERMARK statistical threshold
```

This must be established first.

---

## PHASE 2 — Signal preprocessing

Investigate:

* whitening
* local mean removal
* variance normalization
* LL normalization
* frequency filtering
* robust correlation
* covariance compensation

---

## PHASE 3 — Patch reliability

Introduce a reliability estimator.

The objective is:

```
identify which patches contain trustworthy watermark evidence
```

rather than immediately predicting the watermark.

---

## PHASE 4 — Neural evidence correction

Investigate:

```
raw PRN correlations
    ↓
neural network
    ↓
improved 32-D evidence
```

The network should learn to compensate for distortions that classical matched filtering cannot model.

---

## PHASE 5 — Temporal aggregation

Investigate learned and non-learned temporal aggregation.

---

## PHASE 6 — Robust localization

Improve SIFT-based localization or augment it with modern feature methods.

---

## PHASE 7 — Codeword-aware decoding

Fully exploit the 200,000 valid IDs.

---

## PHASE 8 — ECC / production hardening

Add error-correcting mechanisms and rigorous false-positive calibration.

---

# 33. COMPUTATIONAL REQUIREMENTS

The final system should be realistically implementable in Python using:

* NumPy
* OpenCV
* PyWavelets
* PyTorch where appropriate

The approximately 200,000 × 32 codeword search should be treated as a cheap operation unless profiling demonstrates otherwise.

Keep the architecture modular.

Suggested conceptual components:

```
PRNCorrelator
PatchLocaliser
PatchEvidenceExtractor
PatchReliabilityEstimator
SpatialAggregator
TemporalAggregator
CodewordDecoder
WatermarkHypothesisTester
```

Do not create a giant monolithic system.

---

# 34. UNIT TESTS

Design tests for:

* PRN RMS
* PRN zero DC
* PRN orthogonality
* combined watermark RMS
* correlation recovery
* correct/incorrect PRN response
* no-attack recovery
* noisy recovery
* codeword search
* codeword distance
* NO-WATERMARK threshold
* spatial aggregation
* temporal aggregation
* attack robustness

---

# 35. WHAT I WANT FROM YOU

Do NOT start by dumping implementation code.

First provide a detailed technical design document.

Structure the response as:

1. Executive summary
2. Understanding of the actual supplied implementation
3. Mathematical signal model
4. Analysis of current embedding
5. Effect of combined-watermark normalization
6. Optimal classical detector
7. Individual PRN vs combined PRN
8. 32-D soft evidence representation
9. 200k closed-set decoding
10. NO-WATERMARK hypothesis testing
11. Codeword/Hamming-distance analysis
12. Patch localization
13. Patch reliability
14. Spatial aggregation
15. Temporal aggregation
16. Neural network opportunities
17. Recommended hybrid architecture
18. Training strategy
19. Attack augmentation
20. Evaluation methodology
21. Perceptual-quality evaluation
22. ECC strategy
23. Computational complexity
24. Failure modes
25. Phase-by-phase implementation roadmap
26. Final recommended architecture

Only after the design is agreed upon should you propose implementation details.

---

# 36. CRITICAL CONSTRAINTS

Keep these constraints in mind throughout the design:

* The PRNs are known at detection time.
* There are 32 PRNs.
* Individual PRNs are approximately orthogonal.
* Individual PRNs have RMS=1.
* Individual PRNs have zero DC.
* The combined 32-PRN watermark is normalized to RMS=1.
* Perceptual quality is a hard constraint.
* ALPHA was calibrated using RMS=1.
* Approximately 200,000 32-bit watermark strings are valid.
* The complete valid watermark list is available to the detector.
* The detector must support NO-WATERMARK.
* Soft evidence should be preserved as long as possible.
* The same watermark appears across many patches and frames.
* The detector should exploit spatial and temporal redundancy.
* Deep learning should complement, not blindly replace, known signal-processing structure.
* The current embedding algorithm should be treated as fixed initially.
* Any proposed embedding changes should be clearly separated as future improvements.
* The detector should be designed around the actual supplied project files rather than assumptions.

The ultimate objective is:

```
MAXIMUM ROBUST WATERMARK RECOVERY
/
MINIMUM PERCEPTUAL DISTORTION
```

while maintaining a principled and measurable false-positive rate.

Start by inspecting the supplied project files and then analyze the mathematical signal model before proposing the final detector architecture.
