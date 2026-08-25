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
**interleaved** frame->bit map allowed (ECC as the last resort); **cluster of cells** per side allowed.

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

---

## Corrections found while implementing

Two things in the plan above turned out to be wrong once the code existed. Both are
recorded here rather than silently fixed in the source, because both change what the
design can claim.

### C1 — Stage 5's phase search cannot work, and self-synchronisation is not free

Plan section 2 Stage 5 claims that evaluating all 32 cyclic phases of the `i % 32` map
and keeping the one that maximises `sum_b |L_b|` makes the payload self-synchronising.
It does not, and the reason is a property of the scheme rather than of the
implementation.

Whenever the clip is a whole number of payload cycles -- `N = bit_length x TR`, which is
exactly how these clips are cut -- a phase shift is a pure relabelling of the bit
groups. The same frames land in the same groups; only the group names rotate. So every
phase yields the same multiset of `|z_b|` and therefore the identical sum. Measured: all
32 phases scored within 1e-9 of each other, and the argmax was decided by
floating-point summation order, turning a 32/32 read into 16/32.

From the evidence alone the payload is recoverable only **up to cyclic rotation**.
Fixing it needs information the evidence does not contain: a sync word in the payload, a
deliberately partial cycle, or a known frame index. The detector is non-blind and has
already aligned the leak against the original frame by frame, so phase 0 is known rather
than guessed -- `search_phase` is therefore **off by default**, ties resolve to the
lowest phase, and `decode()` reports `phase_margin` so a phase that was not determined by
the data is visible as a margin near zero.

The presence statistic `sum_b |z_b|` is unaffected: it answers "is this marked at all"
independently of the rotation, which is still something the energy detector could not do.

### C2 — CUDA is available; the sparse homography schedule is not forced

Plan section 2 Stage 0 assumes LightGlue must run on a sparse schedule because CUDA is
unavailable. This box has a Tesla T4 and `torch.cuda.is_available()` is True. Measured on
a 120-frame `moderate` cell, enabling the homography cost ~7 s of an ~80 s cell -- the
CaptureSim attack dominates, not the matcher -- while raising the matched filter from
19/32 to 31/32 and per-frame accuracy from 0.567 to 0.867.

So alignment is not the optional refinement the plan treats it as; it is most of what
makes the geometric conditions readable at all. The sparse schedule is kept (default
`--homography-every 30`) because it is nearly free and the capture geometry really is
fixed per session, but it is now a cost choice rather than a constraint.

### C3 — F6 is refuted end to end: interleaving costs accuracy, and buys acquisition speed instead

F6 predicted from simulation that the interleaved map would beat contiguous runs
(32/32 against 29/32) and flagged one caveat: the simulation regrouped per-frame
correctness measured on a *contiguously* embedded video, so it could not see the encoder
reacting to a cell that flips sides every frame. It said this "must be validated by
re-embedding". It has been, on one fixed source at a fixed CRF with the same 960 frames
and therefore the same 30 frames per bit in every row, and the caveat was the real
effect.

Per-frame accuracy of the matched filter, clean / crf23:

| R | TR=1 (interleaved) | TR=3 | TR=5 | TR=10 | TR=30 |
|---|---|---|---|---|---|
| 80  | 0.974 / 0.949 | 0.992 / 0.987 | 0.981 / 0.969 | 0.992 / 0.980 | 0.995 / 0.984 |
| 110 | 0.936 / 0.881 | 0.974 / 0.962 | 0.965 / 0.939 | 0.978 / 0.963 | 0.993 / 0.978 |
| 140 | **0.815 / 0.771** | 0.917 / 0.887 | 0.940 / 0.908 | 0.958 / 0.918 | 0.992 / 0.967 |
| 200 | 0.708 / 0.679 | 0.798 / 0.763 | 0.844 / 0.797 | 0.857 / 0.841 | 0.865 / 0.829 |

Interleaving is the *worst* map at every radius, and the gap widens with radius: at
R=140 it drops the matched filter to 31/32 clean and 27/32 after one transcode, where
every contiguous map holds 32/32. A cell that is blurred in frame i and untouched in
frame i+1 is re-coded every frame; a static blurred block is carried cheaply by SKIP
macroblocks. The mark is fighting the encoder rather than riding it.

What interleaving does buy is **acquisition speed**, and by a wide margin. Frames needed
to land all 32 bits, R=110: **48** interleaved against 96 (TR=3), 192 (TR=5), 672
(TR=10), 912 (TR=30). That is structural rather than statistical -- a contiguous map at
TR=30 cannot decide bit 31 until frame 930, whatever the evidence quality. So the real
trade is time-to-payload against per-frame accuracy, and section 5's "as few frames as
possible" and "BCR 32/32 with margin" pull in opposite directions.

### C4 — the TR effect is real after all, for the energy detector, and F1 overcorrected

F1 concluded the TR sweep was entirely confounded by the source clip swapping underneath
it. Holding the source, the frame budget and the encode fixed, a TR effect survives, and
it is specific:

    energy baseline, every radius, both conditions:  TR=3,5,10 -> 32/32   TR=30 -> 29/32

Eight cells out of eight. The matched filter is unmoved (32/32 at TR=30 for R<=140), so
this is a property of the energy statistic under maximal clumping, exactly the mechanism
F4 describes -- at TR=30 each bit is one 30-frame run on one scene, and a bit whose run
lands on dead content has no independent evidence anywhere to rescue it. F1's "no
monotone TR effect" was right that the old sweep could not measure it and wrong that
there was nothing to measure.

The per-bit margin says the same thing more sharply. Smallest |z| over the 32 bits at
R=110 clean: 104.9 (TR=3), 116.7 (TR=5), 67.8 (TR=10), 49.2 (TR=1), **1.3 (TR=30)**.
TR=30 reads 32/32 with essentially no margin -- a pass that would not survive a nudge.

### C5 — the flicker metric measured nothing; fixed, and the section 5.3 concern is unfounded

Section 5.3 asks for "a temporal-flicker metric (std over time of the marked cell) to
catch interleaving flicker". Implemented literally, as the standard deviation over time
of the cell's mean luma difference from the original, it cannot detect this watermark at
all: `blur_region` nulls the DCT coefficients *beyond* the cutoff and DC is not one of
them, so a blurred cell carries exactly the original's mean luma. The statistic read
0.197-0.241 everywhere, marked or not, interleaved or contiguous -- that was codec noise,
and every flicker number taken from it is meaningless.

Replaced with `flicker_rms_std`: the standard deviation over time of the per-cell *RMS*
difference, which does rise when a cell is blurred and fall when it is not, reported
alongside `mark_rms` (the mark's mean strength) and their ratio.

Measured, and it is the opposite of what section 5.3 feared:

| R | | TR=1 (interleaved) | TR=3 | TR=30 |
|---|---|---|---|---|
| 80  | mark RMS / flicker / ratio | 6.77 / **4.40** / 0.650 | 6.70 / 4.53 / 0.677 | 6.49 / **4.93** / 0.760 |
| 110 | | 5.57 / **3.07** / 0.550 | 5.53 / 3.28 / 0.594 | 5.38 / **3.61** / 0.671 |
| 140 | | 4.60 / **2.01** / 0.437 | 4.57 / 2.23 / 0.487 | 4.46 / **2.46** / 0.552 |

Interleaving has the *lowest* flicker amplitude at every radius and contiguous TR=30 the
highest, consistently. There is no interleaving-flicker artefact to weigh against its
acquisition-speed advantage.

It also identifies the mechanism behind C3, and corrects the guess made there. C3
attributed interleaving's accuracy loss to the block being "re-coded every frame", but at
a fixed CRF more coding effort should preserve the mark better, not worse -- and the
encoder does spend more: extra bits over contiguous TR=30 are +4.61% interleaved against
+1.04% at TR=3 (R=80), +2.71% against +0.97% (R=110), and ~+0.2% at R=200 where the mark
barely exists.

The resolution is that mean mark strength is essentially flat across maps (6.77 / 6.70 /
6.49 at R=80) while the *swing* falls by 11%. Interleaving is not producing a weaker
mark; it is producing a less-contrasted one. That is motion-compensated prediction
carrying each frame's blur into the next -- reinforcing when neighbouring frames are
marked alike, destructive when interleaving makes them opposite. The same smearing that
flattens the swing flattens the left-versus-right difference the detector reads, which is
why per-frame accuracy drops while PSNR does not.

---

## Where this lands

Full grid measured: 4 radii x 5 TR x 7 conditions on one fixed source at CRF 23, plus
visibility and false positives. 164 rows in `FINDINGS.md` / `findings.jsonl`.

**Operating point.** RADIUS **80**, TR **3-10** contiguous, matched filter on the `a`
statistic with alignment enabled. That reads 32/32 on clean, transcode, moire, mild and
moderate capture, and holds 32/32 under a sigma=2.2 blur where every larger radius
collapses to chance. It costs 1.1 dB of patch PSNR against R=110 (33.7 vs 34.8) and
0.5 dB of frame PSNR -- far less than F7's 29.7-vs-31.5 estimate, which was measured
through the old uncontrolled `mp4v` encode.

If a sigma=2.2 capture is out of scope, R=110 is the better-looking choice (SSIM 0.933
against 0.886) and is otherwise identical on every condition. That is the one judgement
the data cannot make for you; sample frames are in `outputs/harness/dump/`.

Avoid both ends of the TR range: TR=1 loses per-frame accuracy to prediction smearing
(C3, C5) and TR=30 loses robustness and margin to run-clumping (C4). The middle is
32/32 wherever the radius is viable and lands the payload in 96-192 frames.

**Detector.** The matched filter is not a uniform win -- over 111 attacked cells it beat
the energy baseline in 31, lost in 27 and tied in 53. It wins where it matters: full
payload recovery at R=80 goes 18/28 to 25/28, and it holds 32/32 at TR=30 where energy
reads 29/32 in all eight clean cells. `a` beats `z` and `nc` consistently under attack.
Presence separates marked from unmarked by ~20x (71-201 against 0.8-3.8) and reads a
foreign payload correctly rather than reporting the one it sought.

**What is not done.**

1. **The cluster is implemented but never measured.** Every row above is `cluster_k=1`.
   Stage 2's spatial pooling -- K cells per side, inverse-variance weighted -- is the
   one form of redundancy F4 argues is genuinely independent, and the sweep did not
   exercise it. It is also the most promising remaining lever, because it adds evidence
   without touching RADIUS.
2. **`severe` fails at every radius and every TR** (13-20/32, i.e. chance). Since
   `blur_only` carries the identical blur (sigma=2.2, 7 px motion) and R=80 passes it
   32/32, the blur is not the cause -- the geometry is (4 deg rotation, 0.075
   perspective, 3 px shake, 0.82 zoom). That points at Stage 0 alignment rather than at
   the mark, and is the thing to attack next.
3. ECC, still the last resort and still untouched.
