# Blur watermark: findings and decision

One-page account of what was measured, what it means, and what to set. Source data is
`FINDINGS.md` (164 rows) and `findings.jsonl`; the working document with full derivations
is `plan.md`.

**Measured on:** one fixed source (`inputs/120.mp4`), 960 frames per configuration,
H.264 at CRF 23, payload `0xCAFECAFE`, non-blind detection. Grid is
4 radii x 5 TR x 7 conditions, plus visibility and false-positive runs.

---

## 1. The measurement was broken before the detector was

Nothing in the original sweep could answer the question it was asked, for three
independent reasons. All three are fixed; every number below postdates the fix.

| | was | now |
|---|---|---|
| source clip | `inputs/{TEMP_REDUNDANCY}.mp4` — raising TR silently swapped the video | one fixed source, TR moves only the frame→bit map |
| encode | `cv2.VideoWriter('mp4v')`, MPEG-4 Part 2, 3–20 Mbps depending on content | H.264 at fixed CRF through `FFmpegWriter` |
| frame budget | varied with TR | fixed, so every bit gets the same frame count at every TR |

The old encode is why several of the original estimates were wrong in the *optimistic*
direction: at 20 Mbps a lot of high-frequency detail survives that a realistic H.264
stream discards.

## 2. The detector

`blur_detect.py` scored `1 - imp_hf/org_hf` — a ratio of **energies**. Energy is
quadratic, so codec noise *power* adds into the numerator whether or not the mark is
present. On the measured clips 49% of frames read negative loss on the patch that was
never touched.

`blur_detect_mf.py` scores the **linear** projection of the observed residual
`org - imp` onto the residual `blur_region()` is known to leave. Noise uncorrelated with
the original contributes zero mean instead of a positive bias. Five stages: geometric
resync, whitened per-cell matched filter over an annulus just outside the cutoff, spatial
pooling, per-frame calibration against never-marked control cells, and robust (Huber)
temporal aggregation.

**Where it actually helps** — cells reaching full 32/32 recovery, R=80 and R=110 pooled:

| condition | matched filter | energy baseline |
|---|---|---|
| clean | 10/10 | 8/10 |
| crf23 | 10/10 | 8/10 |
| moire_only | 10/10 | 8/10 |
| mild | 10/10 | 8/10 |
| **moderate** | **8/10** | **2/10** |
| **blur_only** | **4/10** | **0/10** |
| severe | 0/10 | 0/10 |

The advantage is concentrated under attack, which is the point. Across all 111 attacked
cells including the dead radii it is 31 better / 27 worse / 53 tied — but those ties and
losses are overwhelmingly R=140 and R=200 cells where both detectors sit at chance and
noise decides the comparison. Of the three statistics carried, `a` (the least-squares
fraction-of-band-removed) beats `z` and `nc` consistently under attack.

## 3. The grid

Matched-filter BCR, statistic `a`, out of 32:

| condition | R=80 | R=110 | R=140 | R=200 |
|---|---|---|---|---|
| clean | 32 all TR | 32 all TR | 31–32 | 22–32 |
| crf23 | 32 all TR | 32 all TR | 27–32 | 22–32 |
| moire_only | 32 all TR | 32 all TR | 28–32 | 21–25 |
| mild | 32 all TR | 32 all TR | 31–32 | 24–32 |
| moderate | 29–32 | 30–32 | 13–20 | 12–20 |
| blur_only | 27–32 | 8–19 | 9–17 | 19–29 |
| severe | 17–20 | 13–19 | 15–20 | 13–20 |

**RADIUS is the dominant knob, as expected — but the cliff is lower than thought.**
R=140 fails `moderate`; R=200 fails even `clean`. The original estimate had R=200 at
32/32 on clean, measured through the 20 Mbps `mp4v` encode.

**The R=80 / R=110 split is exactly one condition:** `blur_only`. That condition is
`blur_sigma=2.2, motion_blur_px=7` — `severe`'s blur in isolation, not a mild diagnostic.
`moderate` is only σ=1.2 / 3 px, and both radii pass it.

**Nothing survives `severe`** (13–20/32 = chance) at any radius or TR. Since `blur_only`
carries the identical blur and R=80 passes it, the blur is not the cause — the geometry
is (4° rotation, 0.075 perspective, 3 px shake, 0.82 zoom).

## 4. Temporal redundancy: avoid both ends

TR is a block-interleaver size, not a redundancy level — the frame budget is fixed, so
every TR gives each bit the same 30 frames and only their *arrangement* changes.

- **TR=1 (fully interleaved) loses accuracy.** Per-frame accuracy at R=110 is 0.936 clean
  / 0.881 transcoded, against 0.974 / 0.962 at TR=3. At R=140 it drops to 27/32 after a
  single transcode where every contiguous map holds 32/32.
- **TR=30 loses robustness and margin.** The energy baseline reads 29/32 at TR=30 in all
  eight clean cells and 32/32 at TR=3–10. Under attack the matched filter degrades too
  (R=110 `blur_only`: 16–19/32 at TR=1–10, **8/32** at TR=30 — below chance, i.e.
  confidently wrong). Smallest per-bit margin at R=110 clean: 104.9 at TR=3 against
  **1.3** at TR=30.
- **TR=3–10 is the operating range**, 32/32 wherever the radius is viable.

What interleaving *does* buy is acquisition speed, structurally: 48 frames to land all 32
bits against 912 at TR=30, because a contiguous map cannot decide bit 31 until frame 930
however good the evidence.

## 5. Visibility

| R | patch PSNR | patch SSIM | frame PSNR | mark RMS | flicker RMS | flicker ratio |
|---|---|---|---|---|---|---|
| 80 | 33.7–34.4 dB | 0.886 | 38.4 dB | 6.5–6.8 | 4.40–4.93 | 0.65–0.76 |
| 110 | 34.8–35.4 dB | 0.933 | 38.9 dB | 5.4–5.6 | 3.07–3.61 | 0.55–0.67 |
| 140 | 36.0–36.3 dB | 0.959 | 39.3 dB | 4.5–4.6 | 2.01–2.46 | 0.44–0.55 |
| 200 | 37.9 dB | 0.980 | 39.7 dB | 3.4 | 0.81–0.99 | 0.24–0.29 |

**R=80 costs ~1.1 dB of patch PSNR and ~0.5 dB of frame PSNR against R=110** — far less
than the pre-fix estimate of 29.7 vs 31.5 dB, which came through the old encode. SSIM is
where R=110 wins meaningfully (0.933 vs 0.886). Sample frames for eyeballing are in
`outputs/harness/dump/`.

Interleaving flickers **less** than contiguous at every radius, not more.

## 6. False positives

| case | presence | recovered | BCR |
|---|---|---|---|
| marked | 71 – 199 | `0xCAFECAFE` | 32/32 |
| different payload | 78 – 201 | `0x0F3C5A69` | 32/32 |
| unmarked | 0.8 – 3.8 | noise | — |

~20× separation, and the foreign-payload case recovers *that* payload rather than the one
being sought. `sum|L_b|` answers "is this marked at all" independently of "what does it
say" — something the energy detector could not do.

---

## Decision

**RADIUS 80 · TR 3–10 contiguous · matched filter, statistic `a` · alignment on.**

That reads 32/32 on clean, transcode, moire, mild and moderate capture, lands the payload
in 96–336 frames, and costs ~1.1 dB of patch PSNR against R=110.

### The honest caveat on σ=2.2 blur

R=80 reads 32/32 under `blur_only`, but the per-bit margin collapses to **0.9–1.7**
against ~20 under `moderate` and ~140 on clean, and it needs 288–720 frames rather than
48–336. Plan §5 set the bar at "32/32 **with** per-bit confidence margin, not just
32/32". **By that standard R=80 does not pass σ=2.2 blur — it lands on the edge of it.**
Treat R=80 as robust through `moderate` and *marginal* at `severe`-strength blur.

If σ=2.2 capture is out of scope, **R=110 is the better choice**: visibly better
(SSIM 0.933 vs 0.886) and identical on every other condition. This is the one judgement
the data cannot make — it depends on how bad a capture must still be read.

### What is not done

1. **The cluster is implemented but never measured.** Every one of the 164 rows is
   `cluster_k=1`. Spatial pooling over K cells per side is the one form of redundancy
   that is genuinely independent — different cells sit on different content, unlike more
   frames of the same scene — and it is the most promising remaining lever because it
   adds evidence without touching RADIUS. `--cluster-k 2,3` with `--resume` measures only
   the new cells.
2. **`severe` fails everywhere, on geometry not blur.** That points at Stage 0 alignment
   (homography cadence and refinement), not at the mark.
3. **ECC** — still the last resort, still untouched.
4. **A sync word.** The payload is recoverable only up to cyclic rotation from the
   evidence alone; the detector relies on being non-blind for its frame origin.

### Corrections to the original plan

Five claims did not survive measurement; full derivations are in `plan.md` under C1–C5.

| | claim | outcome |
|---|---|---|
| C1 | phase search makes the payload self-synchronising | **false** — every phase scores identically on a whole number of cycles; needs a sync word |
| C2 | CUDA unavailable, so alignment must be sparse | **false** — alignment is load-bearing, 19/32 → 31/32 |
| C3 | interleaving beats contiguous (F6) | **refuted end to end** — worst map at every radius; buys acquisition speed instead |
| C4 | the TR effect was pure confounding (F1) | **overcorrected** — a real TR=30 effect survives a controlled experiment |
| C5 | flicker metric catches interleaving flicker | **measured nothing** — `blur_region` preserves DC exactly; fixed, and the concern is unfounded |
