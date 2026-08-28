# 32-bit spread-spectrum watermark: detector report

Answers `prompt.md`. Read this first; **[DESIGN.md](DESIGN.md)** is the full 26-section
technical document it asks for, **[FINDINGS.md](FINDINGS.md)** is the measured attack
table, and **[README.md](README.md)** is the map of the code.

Everything below was measured against the code in this directory on `inputs/30.mp4`
and `inputs/3.mp4` at 1920×1080. Nothing here is an estimate unless it says so.

---

## What was built

A **blind** detector — no original video, no SIFT replay, no temporal alignment —
plus an error-correcting ID set, an 84-attack evaluation harness, empirical
false-positive calibration, perceptual measurement, and 38 unit tests.

```
suspect video → luma → 4-phase 2× decimation into Haar-LL planes
   → whiten → dense FFT correlation against all 32 PRNs
   → localise on T = Σ c_k², which is codeword-independent
   → per-site 32-D evidence + reliability, with a selection-null correction
   → inverse-variance, Huber-clipped pooling over patches and frames
   → 200,000 × 32 GEMM → four-statistic acceptance test → ID | NO_WATERMARK
```

---

## Five findings that changed the design

**1. The PRNs are exactly orthogonal, not approximately.** `P_i = H_i ⊙ R` with
`R ∈ {−1,+1}`, and elementwise multiplication by a ±1 vector preserves Hadamard inner
products. Measured across five seeds: Gram exactly `4096·I` (max off-diagonal **0.0**),
DC exactly 0, RMS exactly 1, `RMS(W)` exactly `√32` for every codeword. So there is
zero inter-PRN interference, the normalisation constant is a known constant, and the 32
per-PRN correlations are a **sufficient statistic** for the payload — scoring against
200,000 combined patterns cannot beat them and costs 200,000× more. This removes
`prompt.md` §10's imperfect-orthogonality analysis and §13's covariance machinery.

**2. Host interference, not attack noise, is the first-order problem.** An unmarked
patch produces correlations of σ ≈ 0.88 against a signal of 0.53 — per-patch SNR
**−3.8 dB**. Two lines of whitening (local mean removal + local variance
normalisation) take per-patch BER from **0.220 to 0.047**.

**3. Designing the 200,000 IDs as an ECC is worth 7.8 dB and costs nothing.** 200,000
IDs drawn at random have minimum Hamming distance **1** — measured 136 colliding pairs.
Extended BCH [32,21,6], verified by enumerating all 2,097,152 codewords, gives
`d_min` = 6 with ten times the headroom needed, and a linear code gives every ID the
same misattribution rate rather than leaving 272 customers permanently worse off.

**4. Anything that selects sites biases the statistic it selects on — twice.** The
patch localiser and the scale search both maximise the evidence the decision is later
made on. Uncorrected, an unmarked clip reads `S₂` = 83 against a χ²₃₂ null of 32, and
the scale search produced an outright false positive (locked scale 1.1, `S₁` = 7.43 over
a 6.81 threshold). Both are corrected against order statistics. The localiser fix
improves *both* ends at once: clean `S₂` 81.8 → 34.3 while marked `S₂` rises
1,147 → 4,481.

**5. Absence was never the dangerous case. Partial presence is.** At CRF 28 over 200
frames the detector reported `S₁` = 17.7 with **24 of 32 bits correct** and accepted —
a confidently wrong ID. It cleared a threshold calibrated against H₀ because it is not
an H₀ event. This took three successive controls to close; §"False positives" below.

---

## Measured performance

Payload embedded in `inputs/30.mp4`, detected blind against a 200,000-ID codebook at
the calibrated threshold.

**Acquisition.** One frame through H.264 CRF 23 is enough: `S₁` = 18.9, 32/32 exact.
40 frames: `S₁` = 79.6, weakest bit at 10.3σ.

**Attack sweep**, 30 attacks × 3 cases, 20 frames — **18 of 30 recover the exact
32-bit payload, zero false positives, no false attributions**:

| survives exactly | `S₁` |
|---|---|
| clean · H.264 CRF 23 | 43.8 · 25.1 |
| 1080p→720p · 1080p→540p (scale recovered exactly) | 41.2 · 37.8 |
| crop 0.75 · crop+rescale · translate 2 px · resample 0.5 | 30.7 · 31.6 · 43.6 · 34.1 |
| blur σ1.0 · sharpen 1.0 | 29.9 · 44.0 |
| gamma 1.6 · contrast 1.5 · saturation 0 · grayscale | 46.3 · 40.3 · 38.4 · 43.8 |
| **frame drop 20 % · half frame rate · reverse · interpolate** | **40.0 · 30.5 · 43.8 · 31.4** |

The temporal row is worth pausing on. Dropping a fifth of the frames, halving the frame
rate, playing the clip backwards and averaging adjacent frames all cost observations and
nothing else, because the payload has no temporal code. No aligner, no sync word, no
frame-to-bit map — and none needed.

**Geometry.** Scale recovered *exactly* at every tested ratio (0.6667, 0.5, 0.3333,
1.3333), with 1080p→360p — a 3× downscale — still decoding exactly. Unmarked clips lock
nothing at any scale.

---

## The one real weakness, and the measured fix

**Everything that fails, fails for one reason: compression past H.264 CRF 23.** With
geometry disabled entirely, `S₁` goes 26.6 (CRF 23) → 5.9 (CRF 28) → 3.9 (CRF 32).
H.265, low bitrates, noise σ10 and the combined chains are all the same mechanism: the
carrier is **white** in the DWT LL domain, so half the mark's energy sits above the band
a codec keeps.

The fix is chip size — a one-line embedding change (patch 256, DWT level 2) that halves
the carrier frequency at the same number of chips. 20 frames, calibrated threshold:

| attack | 128/L1 `α`=3 (56.1 dB) | **256/L2 `α`=3.75 (55.1 dB)** | 256/L2 `α`=6 (50.6 dB) |
|---|---|---|---|
| clean | 43.8 · 32/32 ✓ | **74.8 · 32/32 ✓** | 117.7 · 32/32 ✓ |
| H.264 CRF 23 | 25.1 · 32/32 ✓ | 24.5 · 32/32 ✓ | 82.3 · 32/32 ✓ |
| **H.264 CRF 28** | 5.4 · **16/32, chance** | 7.5 · **32/32 bits, below threshold** | **42.1 · 32/32 ✓** |
| H.264 CRF 32 | 4.7 · 18/32 ✗ | 2.8 · 26/32 ✗ | 5.3 · 24/32 ✗ |
| **blur σ2.0** | 13.4 · 32/32, **below threshold** | **52.4 · 32/32 ✓** | 105.5 · 32/32 ✓ |
| 1080p→540p | 37.8 · 32/32 ✓ | **74.3 · 32/32 ✓** | 118.9 · 32/32 ✓ |

The middle column is the fair comparison — same frame PSNR, and *better* local
perceptual metrics (marked-region PSNR 46.1 vs 43.1 dB, peak excursion 3.0 vs 5.0 grey
levels, LPIPS 0.00007 vs 0.00011, SSIM identical at 0.9990). It is better or equal on
five of six attacks, converts blur σ2.0 from a miss into a comfortable pass, and gets
**every bit right at CRF 28 where the 2×2 chip returns chance**. Spending 5.5 dB of
frame PSNR (`α`=6) makes CRF 28 comfortable. CRF 32 defeats all three.

**One non-obvious caveat: amplitude cannot be traded down linearly.** `S₁` at CRF 23 is
82.3 at `α`=6 and 24.5 at `α`=3.75 — a factor of 3.4 for a factor of 1.6 in amplitude.
Compression is a quantiser, not additive noise: below the quantiser step the mark is
destroyed rather than attenuated. Any ALPHA change has to be measured, never scaled.

Per `prompt.md` §36 this is **reported, not applied**. `embed.py` is unchanged; picking
a point on the curve is a viewing decision on real content.

---

## False positives: what it took to get to zero

Four statistics, each added because the previous set demonstrably failed.

| control | catches | measured limitation |
|---|---|---|
| `S₁` parametric threshold | little | admitted 3/24 decoy runs and 6/30 unmarked cells — ~1 FP in 5 |
| `S₁` calibrated (17.76 vs 6.81) | H₀ on comparable clips | null grows ≈√M: H₀ max 6.96 → 7.35 → **19.07** at 20/60/200 frames |
| `S₂` evidence energy | absence of a mark | same M-dependence |
| `S₃` split-half agreement | noise-driven partial recovery — rejected all three CRF 28/32/36 false attributions, kept the CRF 23 true positive | **1 of 6 decoy keys slipped**: both halves see the same scene, so content-driven bias reproduces in both |
| **`S₄` per-video decoy null** | both, by construction | costs (1+N)× the correlator |

The decisive measurement: **at CRF 28 over 200 frames the *true* key scored 17.73
against its own null of mean 14.00, max 18.85.** It was inside its own null. No
threshold read off other footage could have known that.

`S₄` reads the same clip with N decoy keys — same content, same attack, same observation
count, same correlation structure — so whatever inflates the true score inflates theirs.
Measured at 60 frames: true key `S₁` = 125.2, **decoy z = 73.5**; a decoy key `S₁` = 7.35,
**decoy z = 1.0**. `DESIGN.md` §23 gives the banding that keeps the (1+N)× cost off the
easy cases.

The mechanism behind the √M growth is worth recording: observations are positively
correlated, so the true variance of the weighted mean plateaus near `σ²ρ` while the
sandwich estimator keeps reporting `σ²/M`. A correlation-aware standard error
(frame-clustered or block bootstrap) is the other route to the same fix.

---

## Bugs found and fixed

| where | what |
|---|---|
| `utils/video.py` | **two classes named `FFmpegWriter`**, the second shadowing the first with swapped argument order. Every written video came out **1080×30 @1920 fps** from a 1920×1080@30 source. Blocked every experiment that writes a video |
| `src/spread_spectrum/embed.py` | marked patches stored into a `uint8` view without clipping, so out-of-range pixels **wrapped** (258.9→2, −1.2→255). **0.87 %** of marked pixels on `inputs/3.mp4`. Also truncated rather than rounded, darkening every patch by half a grey level |
| `utils/patch.py` | separation enforced against the module constant, ignoring the `square_size` passed — 128 px patches came out 256 apart, **7.5/frame instead of 27.0** |
| `utils/video.py` | `read_frame()` raised at end of stream instead of returning `None`; the writer opened only at `frame_number == 0` |
| `detect/aggregate.py` | `lim` referenced before assignment when the robust residual scale came out zero — raised `NameError` on identical evidence |
| `codebook.py`, `prn.py` | dead code that computed and discarded a normalised sum; misspelled method; `int8` accumulator; normalisation by measured rather than exact RMS |

---

## What is left

| | |
|---|---|
| **Phase E** | Sweep the whitener *under attack*. §6's ranking is a no-attack ranking, and a filter that whitens the host also amplifies codec noise — which lives in the same band as the mark. This is where any remaining compression headroom is |
| **Phase F** | Learned host suppression, only if E leaves a gap. Given that the binding limit is compression rather than host noise, the gap may be smaller than the 7.5 dB whitening result suggests |
| **Decision** | Chip size (above). Measured both ways; needs a viewing call |
| **Rotation** | `--geometry scale+rotation` exists, costs ~5×, unmeasured |
| **Footage** | `inputs/{15,30,60,120}.mp4` are **byte-identical content** at different lengths — verified, not assumed. There are two distinct sources on this machine. No generalisation claim should be made from them |
