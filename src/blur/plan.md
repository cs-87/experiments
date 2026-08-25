# Blur watermark: root cause of the TR paradox + a matched-filter detector

## Context

`src/blur/blur_recv_two_patches.py` embeds 32 bits by DCT low-pass filtering one of two
240x240 patches straddling frame centre (left = 1, right = 0), holding each bit for
`TEMP_REDUNDANCY` consecutive frames. `src/blur/blur_detect.py` reads it back non-blind by
pooling raw high-frequency energy per bit run and comparing `loss = 1 - imp_hf/org_hf`
between the two candidates. Robustness was observed to *fall* as `TEMP_REDUNDANCY` rose.

I measured the existing `inputs/{TR}.mp4` + `outputs/MIDDLE/{TR}/{RADIUS}.mp4` pairs
read-only (payload `0xCAFECAFE`). The findings below are measurements, not hypotheses, and
they change what needs building.

Agreed scope: survive **camcorder / screen capture**; visibility budget **RADIUS ~110-140**;
**interleaved** frame->bit map allowed (no ECC); **cluster of cells** per side allowed.

---

## Findings

### F1 — The TR sweep is confounded; TR is not the variable being changed

[blur_recv_two_patches.py:34](src/blur/blur_recv_two_patches.py#L34) is
`INPUT = f"./inputs/{TEMP_REDUNDANCY}.mp4"`. Raising TR swaps the source clip: `inputs/` is
symlinked so `32 bits x TR frames` exactly fills each one (TR=3 -> 4 s @ 0.24 Mbps,
TR=15 -> 16 s, TR=30 -> 32 s, TR=60 -> 64 s, TR=120 -> 128 s of Big Buck Bunny).

BCR on the **clean, un-attacked** embedder output with the current detector:

| TR | 3 | 15 | 30 | 60 | 120 |
|----|---|----|----|----|-----|
| BCR | 32/32 | 28/32 | 28/32 | 28/32 | 29/32 |

There is no monotone TR effect. It drops once at 3 -> 15 (the 4-second clip is a near-static
0.24 Mbps intro) and is then flat. "TR hurts" is really "the longer clips are harder content".

### F2 — The detector is already broken with zero attack

28-30/32 on the embedder's own output means there was never any robustness headroom; every
attack measurement started from a broken baseline.

### F3 — Root cause: an energy (quadratic) statistic over a near-zero denominator

`pooled_loss = 1 - imp_hf/org_hf` ([blur_detect.py:71-80](src/blur/blur_detect.py#L71-L80))
compares *energies*, so codec noise **power** adds into `imp_hf` whether or not the mark is
there. At TR=120, RADIUS=200:

- left-patch `org_hf` percentiles p1=807, p25=6.6k, **p50=15.0k**, p99=1.53M — a 100x spread;
  the median frame has almost nothing above the cutoff to remove
- 38.6% of frames have left `org_hf < 1e4`
- **49.2% of frames have `imp_hf > org_hf` on the *untouched* patch** — the encoder puts more
  HF in than was there, so "loss" on the reference side is a coin flip

The failing bits are exactly the dead-content ones (TR=30 per-bit dump):

```
 b exp got    oneL   zeroL   margin      medLo      medRo
 8   1   0  -0.611  -0.181   -0.429      35605      87264  <== WRONG
 9   1   0  -0.798  -0.258   -0.541      29441      70277  <== WRONG
15   0   1   0.015  -0.184    0.199        812       3847  <== WRONG
16   1   0  -0.714  -0.618   -0.096        806        793  <== WRONG
17   1   1   0.894   0.032    0.862     738847     417900   (healthy, for contrast)
```

Bits 15/16 sit on ~800 units of HF energy against ~740,000 for bit 17. Negative losses are
the codec *adding* HF.

### F4 — Why temporal redundancy cannot rescue those bits

A run's frames are one scene measured repeatedly, not independent draws. At TR=120:

- per-frame error autocorrelation 0.69 @ lag 1, 0.49 @ lag 5, 0.15 @ lag 30, ~0 @ lag 120
- mean error-run length 4.3 frames (max 74) => ~212 independent error events in 3840 frames
- **per-bit frame-accuracy spans 0.28 to 1.00**

A bit whose contiguous run lands on a dark/soft scene has a *majority* of frames voting wrong.
More frames make it more confidently wrong. **That is the answer: contiguous runs make each
bit's errors perfectly correlated with one scene's content.**

### F5 — Fix A: matched filter instead of energy loss (detector-only, large)

Detection is non-blind, so the exact expected residual is known: `delta = org - blur_region(org)`,
supported precisely on the masked band. Score the **linear** projection of the observed
residual `org - imp` onto `delta` rather than comparing energies — noise uncorrelated with
`org` then contributes zero mean instead of a positive bias.

Per-frame accuracy / BCR (contiguous runs, TR=30, clean output):

| RADIUS | energy loss (current) | matched projection | normalised correlation |
|--------|----------------------|--------------------|------------------------|
| 80  | 95.1%  31/32 | 99.8%  32/32 | **100.0%  32/32** |
| 110 | 93.3%  31/32 | 98.1%  32/32 | 98.8%  32/32 |
| 140 | 94.7%  31/32 | 99.1%  32/32 | 99.3%  32/32 |
| 170 | 92.9%  31/32 | 98.0%  32/32 | 99.0%  32/32 |
| 200 | 85.0%  28/32 | 97.5%  32/32 | **98.3%  32/32** |

### F6 — Fix B: interleave the frame->bit map (one line each side)

`bit_index = i % bit_length` instead of `i // TR % bit_length`. Same embedding primitive,
same patch selection, same frames per bit — only *which* frames. Simulated at TR=120:

| map | per-bit frame-accuracy min/med/max | hard majority | soft weighted |
|-----|-----------------------------------|---------------|---------------|
| contiguous `i//TR%32` | **0.28** / 0.78 / 1.00 | 29/32 | 30/32 |
| interleaved `i%32`    | **0.68** / 0.76 / 0.82 | **32/32** | **32/32** |

Caveat: simulated by regrouping per-frame correctness measured on a *contiguously* embedded
video. Real interleaving flips the marked side every frame, which changes encoder behaviour
(a static blurred block is cheap to preserve via SKIP macroblocks; a flickering one is
re-coded every frame) and may flicker visibly. **Must be validated by re-embedding** — this
is the one finding below that is not yet measured end to end.

### F7 — RADIUS is the dominant robustness knob and 200 sits on the cliff

Matched-filter per-frame accuracy under in-memory attacks (TR=30, every 3rd frame):

| RADIUS | patch PSNR | none | scale->1280 | scale->854 | blur s=1.5 | jpeg q=40 |
|--------|-----------|------|-------------|------------|------------|-----------|
| 80  | 29.7 dB | 100.0% | 96.2% | **94.7%** | **88.1%** | 97.5% |
| 110 | 31.5 dB | 98.8%  | 93.8% | 91.2% | 83.4% | 96.2% |
| 140 | 33.2 dB | 99.1%  | 90.9% | 70.6% | 54.7% | 97.5% |
| 200 | 36.1 dB | 98.4%  | **60.3%** | **49.7%** | **52.2%** | 83.4% |

PSNR is over the 240x240 patch only (2.8% of a 1080p frame).

**The concern to state plainly:** this watermark *is* a low-pass operation and a camera
capture is also a low-pass operation, so the chosen attack model is close to a matched
attack. It can still work because the detector compares two cells **within the same frame** —
a spatially uniform capture blur hits both equally and cancels in the difference. The design
rule that follows is that RADIUS must sit *below* the capture chain's effective cutoff, or
both cells land on the noise floor and no detector can separate them. This is why RADIUS
matters more than any detector refinement, and why the sweep should decide it rather than
picking a value up front. I'd expect to land near 110.

---

## Plan

### 1. Fix the measurement before changing anything

- New `src/blur/eval_harness.py`: sweep `(RADIUS x TR x condition)` with **one fixed source
  video** (`inputs/120.mp4`) so TR is the only variable — this alone removes F1.
- Reuse `CaptureSim` / `get_condition` from
  [src/retrieval/capture_sim.py](src/retrieval/capture_sim.py) for the camcorder conditions
  (`mild`/`moderate`/`severe`/`blur_only`/`moire_only`/`extreme`) rather than writing new
  attack code. Report per-frame accuracy, BCR, per-bit confidence, and frames-needed-for-32-bits.
- Make the encode honest: `utils/video.py` writes `mp4v` (MPEG-4 Part 2) at OpenCV defaults,
  producing 3-20 Mbps 1080p that varies uncontrolled across the sweep. Write H.264 at a fixed
  CRF via ffmpeg (`utils/transcode_n.py` already shells out to ffmpeg — follow that pattern).
- Harness appends every result to `src/blur/FINDINGS.md` so a session can restart from it.

### 2. New detector — `src/blur/blur_detect_mf.py`

Keep `blur_detect.detect()` untouched as the baseline to compare against.

**Stage 0 — resync (camcorder).**
Temporal: `TemporalAligner` from [src/retrieval/temporal_aligner.py](src/retrieval/temporal_aligner.py)
maps each leak frame to an original frame index; use *that* index for patch selection and bit
assignment, never the loop counter. Geometric: `LightGluePatchMatcher.compute_homography` +
`warp_patch` from [utils/lightglue.py](utils/lightglue.py) to pull each candidate back into
the original's 240x240 sampling grid before any DCT. LightGlue is installed but **CUDA is not
available here**, so estimate the homography on a sparse schedule (every N frames, or on an
aligner-confidence drop) and reuse it between updates — `capture_sim`'s own docstring notes the
session geometry is sampled once and only jittered, so this is sound.

**Stage 1 — per-frame, per-cell matched filter.** For each candidate cell:

```
O = dct2(org_patch); I = dct2(warped_leak_patch)
mask  = RADIUS < r < RADIUS + W          # annulus, not the whole band
delta = O[mask]                          # residual blur_region() would create
resid = (O - I)[mask]                    # residual actually observed
nc    = <delta, resid> / (||delta|| * ||resid||)      # in [-1, 1]
```

Band-limit `W` because the far corner coefficients are pure noise after capture (the 200..224
annulus alone already edged out the full band at R=200). Add per-ring spectral whitening with
`sigma_k` estimated from unmarked control cells in the same frame — this turns the correlator
into a proper GLRT and absorbs the camera's non-flat MTF. `nc` is gain-invariant by
construction; add a per-patch mean/variance match before the DCT to absorb gamma.

**Stage 2 — spatial pooling.** With the cluster extension, K cells per side carry the same
bit; pool their `nc` with inverse-variance weights (more `||delta||`, less noise = more
weight). Different regions are content-decorrelated, so this is genuinely independent evidence
— unlike more frames of the same scene (F4). Per-frame statistic
`s_f = pooled_nc(left) - pooled_nc(right)`.

**Stage 3 — calibrated soft LLR.** Estimate the null `(mu_0, sigma_0)` of `nc` from the
unmarked side plus never-marked grid cells in the same frame, then
`LLR_f = (s_f - mu_0) / sigma_f^2`. A frame with no removable energy or a rejected homography
contributes ~0 automatically. **Do not hard-gate**: dropping the lowest-energy 75%/90% of
frames measurably collapsed BCR to 21/32 and 16/32, while soft weighting was the best
aggregator (30/32 vs 29/32).

**Stage 4 — robust temporal aggregation.** `L_b = sum psi(LLR_f)` over the bit's frames with
`psi` a Huber/Tukey redescending influence function, so a few catastrophic frames (a cut, a
lost homography, a blown highlight) cannot dominate. `L_b` is the per-bit confidence and the
undecided threshold.

**Stage 5 — phase search + false-positive control.** Interleaving makes a residual temporal
offset catastrophic (offset k cyclically permutes the whole payload, where contiguous runs
only lose k frames per boundary). Mitigate by computing `s_f` once and evaluating all 32
cyclic phases of the `i % 32` map, keeping the phase that maximises `sum_b |L_b|` — cheap,
and it makes the payload self-synchronising. The same `sum_b |L_b|` answers "is this marked
at all" separately from "what is the payload", which the current code cannot do.

### 3. Embedder changes (minimal, as agreed)

- Interleave: `bit_index = i % length` at
  [blur_recv_two_patches.py:67](src/blur/blur_recv_two_patches.py#L67) and its mirror at
  [blur_detect.py:281](src/blur/blur_detect.py#L281). Keep the contiguous map behind a flag so
  the harness can A/B it.
- `get_middle_cluster(frame, k)` in [src/blur/patch.py](src/blur/patch.py) alongside
  `get_middle_patches`, returning K grid cells per side around centre (K configurable, default
  2); embedder blurs all K on the marked side. Built on the existing `get_patch` /
  `get_middle_split_col` so the grid convention cannot drift.
- `RADIUS` chosen from the sweep in the agreed 110-140 band.

### 4. Bugs to fix along the way

- [blur_detect.py:99-105](src/blur/blur_detect.py#L99-L105): the LightGlue branch is
  unreachable (a bare `return None` precedes it) and calls `wrap_patch`, which does not exist
  — the name is `warp_patch`. `LIGHTGLUE=True` today silently falls through to a plain slice.
- [blur_recv_two_patches.py:52-56](src/blur/blur_recv_two_patches.py#L52-L56) silently
  truncates `length` below 32 when the video is short; make it explicit.
- [utils/bit.py:22](utils/bit.py#L22) `get_bcr` has `bcr = +1` where it means `bcr += 1`.
  `blur_detect.bit_correct_rate` is the correct implementation; delete the broken one.

### 5. Verification

1. `eval_harness.py` on the fixed source, baseline detector vs matched-filter detector, over
   `RADIUS in {80,110,140,200} x TR in {1,3,5,10,30} x conditions {clean, transcode-CRF23,
   mild, moderate, severe, blur_only, moire_only}`. Success = BCR 32/32 with per-bit
   confidence margin, not just 32/32.
2. Confirm the F6 simulation for real: re-embed with the interleaved map and re-measure. This
   is the one claim not yet validated end to end, and the encoder-behaviour and flicker
   concerns above are the specific things to check.
3. Visibility: patch and full-frame PSNR/SSIM, plus a temporal-flicker metric (std over time
   of the marked cell) to catch interleaving flicker; dump sample frames at each RADIUS for
   you to eyeball before fixing the value.
4. False positives: run the detector on unmarked video and on a video marked with a
   *different* payload; `sum_b |L_b|` must separate cleanly.
5. Report the frames-to-32-bits curve per condition — the actual answer to "as few frames as
   possible".
