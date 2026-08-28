# Detector design — 32-bit spread-spectrum video watermark

Answers `prompt.md`. Every number below was measured against the code in this
directory (`codebook.py`, `prn.py`, `embed.py`) on `inputs/30.mp4`,
`inputs/3.mp4` and `4_sec_source.mp4`, 1920×1080. Where something is derived
rather than measured it says so. Where something is neither, it is listed in
§24 as an open question rather than asserted.

---

## 1. Executive summary

The system is in far better shape than `prompt.md` assumes, for one reason that
changes several answers: **the PRNs are exactly orthogonal, not approximately.**
`P_i = H_i ⊙ R` with `H` Sylvester-Hadamard and `R` a fixed ±1 vector, and
elementwise multiplication by a ±1 vector preserves inner products, so the Gram
matrix is exactly `4096·I` (measured max off-diagonal: 0.0). Consequences:

- There is **zero inter-PRN interference**. §10's "imperfect orthogonality"
  analysis has nothing to analyse.
- `RMS(W) = √32` **exactly, for every one of the 2³² codewords**, so the
  normalisation constant is a known constant and the embedded per-bit amplitude
  is exactly `ALPHA/√32`.
- The evidence covariance is **diagonal** at correct alignment, so the
  Mahalanobis / covariance-aware machinery in §13 is unnecessary there. It
  becomes necessary only under misalignment.
- The 32 per-PRN correlations are a **sufficient statistic** for the codeword
  (§7). Scoring against 200 000 combined patterns cannot beat them, and costs
  200 000× more.

The detector I would build is:

```
suspect video → per-frame luma
   → 4-phase 2× decimation into Haar-LL planes
   → whitening (local mean removal + local variance normalisation)
   → FFT correlation against all 32 PRNs           → c_k(x, y)
   → localise on T(x,y) = Σ_k c_k(x,y)²            ← codeword-independent
   → per-site 32-D evidence + reliability σ̂
   → inverse-variance, Huber-clipped pooling over patches and frames  → ĉ ∈ R³²
   → exhaustive 200 000 × 32 GEMM, score = ĉ·q
   → two-statistic acceptance test → ID | NO_WATERMARK
```

It is **blind** — no original video — and it needs **no temporal
synchronisation of any kind**, because the payload has no temporal code.

Built and measured. Payload `0x0F1205C6` embedded in `inputs/30.mp4` and re-encoded
at H.264 CRF 23, detected blind against a 200 000-ID codebook:

| | `S₁` | `S₂` | payload |
|---|---|---|---|
| **1 frame** | 18.9 | 380 | **32/32 exact** |
| 40 frames | 79.6 | 6 560 | 32/32 exact, weakest bit at 10.3σ |
| unwatermarked control | 4.4 | 34.4 | NO_WATERMARK |

One frame is enough, and with the scale search on a 1080p → 360p downscale still
decodes exactly.

Across a 30-attack sweep at the **calibrated** acceptance threshold (§10, §20):
**18 of 30 attacks recover the exact 32-bit payload, with zero false positives and no
false attributions.** Every photometric attack, every temporal attack, crop,
crop+rescale, translate, sharpen, resample, blur σ1.0, and 1080p→720p and →540p with
the scale recovered exactly.

**Everything that fails, fails for one reason: compression past H.264 CRF 23.** The
carrier is white in the DWT LL domain, so half its energy sits above the band a codec
keeps. That is the whole robustness story, and §21 measures the fix.

Three detector-side levers dominate, in order of size:

| lever | measured gain | cost |
|---|---|---|
| Whitening before correlation | per-patch BER 0.220 → 0.047 | 2 lines |
| Designing the 200 000 IDs as an ECC | 7.8 dB at `d_min`=6 | zero |
| Patch spacing fix (`utils/patch.py`) | 7.5 → 27.0 patches/frame, +5.6 dB | one argument |

And two corrections that are not optional, both of which had produced false positives
before they were found: the localiser and the scale search each maximise the statistic
the decision is later made on, and each needs an order-statistic correction (§10, §12).
The parametric acceptance threshold is also wrong by 2.6× — nominally 1e-6, measured at
roughly one false positive in five cells (§10).

One embedding-side lever, kept separate per §36: moving the carrier from a 2×2 to a
4×4 chip. At matched frame PSNR *and* better local perceptual metrics it roughly
doubles margin on most attacks, converts blur σ2.0 from a miss into a comfortable pass,
and recovers every bit at CRF 28 where the 2×2 chip returns chance. It is a trade curve
rather than a free win, and §21 gives it in full.

Machine learning belongs in exactly one place first — **learned host
suppression in front of the correlator** — and explicitly *not* where §25
proposes it. Reasoning in §16.

---

## 2. Understanding of the supplied implementation

Read from the files, not from the prompt.

**`codebook.py`.** `generate_codebook(seed)` builds Sylvester-Hadamard `H` of
order 4096, drops row 0 (because `H₀ ⊙ R = R`), multiplies every remaining row
elementwise by one shuffled balanced ±1 vector `R`, and keeps the 32 rows whose
`|Σ|` is smallest. Measured across seeds 8787/0/1/42/12345: values exactly
{−1,+1}, RMS exactly 1, DC exactly 0, Gram exactly `4096·I`, combined RMS
exactly `√32`.

*Was:* the function also computed a normalised `combined` sum and discarded it
(dead code), and a comment said "31 lowest-imbalance" where `NUM_PRNS = 32`.
Both removed.

**`prn.py`.** `BalancedPRNGenerator((64,64), seed)`. Bit `'1'` → `+P_i`,
`'0'` → `−P_i`, summed and RMS-normalised.

*Was:* the accumulator was `int8` (safe at 32 PRNs, ±32; overflows past 127),
the method name was misspelled, and the normalisation divided by the *measured*
RMS. Now divides by the exact constant `√32` so the per-bit amplitude does not
drift with the codeword.

**`embed.py`.** `ALPHA = 3`, `SQUARE_SIZE = 128`, level-1 Haar. For every frame,
`get_sift_patches` yields centred 128×128 patches at the strongest SIFT
keypoints; each patch gets `LL += ALPHA · W`; every frame is written. The same
`W` goes into every patch of every frame — no per-patch keying, no per-frame
keying, no temporal code.

*Was:* the marked patch was stored into a `uint8` view without clipping, so
out-of-range pixels **wrapped** (258.9 → 2, −1.2 → 255). Measured 0.87 % of
marked pixels on `inputs/3.mp4`, 0.022 % on `inputs/30.mp4`. Now clipped and
rounded.

**`utils/patch.py`.** `non_overlapping_points` enforced separation against the
module constant `SQUARE_SIZE = 256` and ignored the `square_size` passed, so
128 px patches came out 256 apart — 7.5/frame instead of the 27.0/frame the same
frame supports without overlap. Now a parameter, defaulted to the old behaviour.

### Answers to §6's questions about patch selection

| question | answer |
|---|---|
| how are SIFT patches selected | `cv2.SIFT_create().detectAndCompute` on the luma plane; keypoints sorted by `(kp.size, kp.response)` descending; greedy Chebyshev suppression |
| how are coordinates generated | patch centred on the keypoint: `x − 64 … x + 64`; dropped entirely if it overhangs the frame |
| do patches overlap | no — suppression guarantees it |
| how many per frame | 7.5 (`inputs/30.mp4`), 14.4 (`4_sec_source.mp4`), 5.3 (`inputs/3.mp4`) at the current spacing; 27.0 at `min_separation=128` |
| are dimensions always identical | yes, always exactly `square_size²`; non-conforming patches are dropped rather than padded |
| do positions depend on content | yes, entirely. **This is the detector's central problem** — see §12 |
| other transformations before embedding | none. Luma only; chroma untouched |

### What is missing, stated rather than assumed (§1)

- **No detector of any kind**, and no evaluation harness for this scheme.
- **No ID list.** The "~200 000 valid watermark strings" of §8 do not exist in
  the repository. §11 and `ids.py` construct them.
- **No perceptual measurement** at any ALPHA.
- **`ALPHA = 3` has no recorded calibration.** §5 says it was calibrated on the
  1-bit system; that calibration is not in the repository, so the claim that the
  32-bit mark reuses a validated budget cannot be checked. §21 shows why it
  should not be assumed.

---

## 3. Mathematical signal model

Let `n = 4096`, `N_b = 32`, `s ∈ {−1,+1}^32` the payload signs.

**Codebook.** `P_i = H_i ⊙ R`, `H` Sylvester-Hadamard of order `n`,
`R ∈ {−1,+1}^n` fixed. Then

```
<P_i, P_j> = Σ_k H_i[k] H_j[k] R[k]²  =  <H_i, H_j>  =  n·δ_ij      (R[k]² ≡ 1)
```

exactly. Measured: `max |off-diagonal| = 0.0`.

**Embedded pattern.**

```
W = (1/√32) Σ_i s_i P_i ,      ‖Σ_i s_i P_i‖² = Σ_i Σ_j s_i s_j <P_i,P_j> = 32n
```

so `RMS(W) = 1` exactly for every codeword, and the divisor is the constant
`√32` rather than a codeword-dependent measurement.

**Embedding.** `LL' = LL + α·W` on the level-1 Haar approximation of each
128×128 patch.

**Pixel-domain equivalent.** Orthonormal Haar gives `LL[u,v] = ½·Σ(2×2 block)`,
and synthesis spreads a unit LL delta as `½` into each of 4 pixels:

```
I' = I + (α/2)·(W ⊗ 1₂ₓ₂)                    verified to 2.4e-7
```

The whole scheme is therefore a **2×2-block-replicated pixel-domain
spread-spectrum mark**. At `α = 3`: mark RMS **1.5**, peak **6.36** grey levels,
**PSNR 44.6 dB**. This equivalence is what makes the blind full-frame correlator
of §12 possible.

**Observation.** Blind, so the host is noise:

```
x = Φ(LL_host + α·W + η)          Φ = whitening operator, η = attack noise
c_k = <x, P_k> / n = A·s_k + ν_k ,   A = α·g_Φ/√32 ,   ν_k = <Φ(LL_host+η), P_k>/n
```

`g_Φ ≤ 1` is the whitener's attenuation of the mark. Because `Φ` is
shift-invariant and `P_k` is broadband, `ν_k` is very nearly Gaussian and — by
exact orthogonality — very nearly independent across `k`.

---

## 4. Analysis of the current embedding

Measured on real frames, `α = 3`, no attack, perfect alignment:

| | raw LL | whitened (−3×3 box mean) |
|---|---|---|
| signal amplitude `A` | 0.5303 (= 3/√32, exact) | 0.480 |
| host noise σ | 0.76 – 0.88 | 0.31 – 0.37 |
| per-bit SNR | **−3.1 to −4.4 dB** | +2.2 to +3.7 dB |
| per-patch BER | 0.220 | 0.059 |

**Host interference, not attack noise, is the dominant impairment at first
order.** A patch of unwatermarked video already produces correlations of
standard deviation 0.88 against a signal of 0.53. That single fact reorders the
whole priority list: whitening (§6) is worth more than anything a detector can
do downstream, and it is worth more than any attack robustness measure, because
it applies to every observation including the unattacked ones.

The carrier is **white** in the LL domain (`R` is white, so `H_i ⊙ R` is white
for any `i`). Half the mark's energy therefore sits above LL-Nyquist/2 —
precisely the band that blur, downscaling and compression remove first. This is
the mechanism behind the attack failures in §12 and the chip-size result in §21.

---

## 5. Effect of combined-watermark normalisation

`prompt.md` §5 and §11 are correct in direction and can be made exact.

**Per-bit amplitude.** In the 1-bit system `W = P₁` and `⟨W,P₁⟩/n = 1`. In the
32-bit system `⟨W,P_k⟩/n = s_k/√32`. Each bit therefore carries
**`1/√32` = 0.177×** the amplitude, i.e. **−15.05 dB per bit**, exactly.

**Total energy is unchanged.** `RMS(W) = 1` in both. This is not a loss — it is
the definition of spreading 32 bits across one energy budget. There is no
allocation that avoids it at fixed RMS, and multiplying ALPHA by `√32` would
multiply the embedded energy by 32 (§5 is right to forbid it).

**Is the loss recovered?** Yes, with large margin. Against −15.05 dB:

| compensation | gain |
|---|---|
| spatial aggregation, 27 patches/frame | +14.3 dB |
| temporal aggregation, 120 frames | +20.8 dB |
| closed-set coding at `d_min` = 6 | +7.8 dB |
| **total available** | **+42.9 dB** |

The right way to state it: after whitening, **a single patch already reaches
per-bit `z` = 1.61** (BER 0.054), and the decision needs about `z` = 1.25 at the
video level with a `d`=6 code. Unattacked, the system is over-provisioned by
roughly 37 dB. **The entire design question is how much of that survives the
attack chain**, not whether the normalisation is affordable.

**What normalisation does cost, and §5 does not mention.** `W` is not ±1-valued.
It takes 23 distinct values (even integers over `√32`) and is approximately
Gaussian. Measured peak-to-RMS ratio: **2.83 – 4.60** depending on codeword,
versus **exactly 1.0** for the 1-bit PRN. At equal ALPHA and equal PSNR the
32-bit mark has ~4× the peak excursion — sparse impulsive speckle instead of
uniform dither. The ALPHA calibrated on an RMS-1 *binary* mark is therefore
**not** automatically valid for an RMS-1 *Gaussian* mark. See §21.

---

## 6. The optimal classical detector

**Whitening.** The GLRT for a known signal in coloured Gaussian noise
correlates against `C⁻¹P_k`, where `C` is the noise covariance. The noise here
is the host image, whose LL is strongly low-pass; the mark is white. So `C⁻¹` is
a high-pass, and any reasonable high-pass captures most of the gain. Measured:

| Φ | per-patch BER | BER after 8 patches |
|---|---|---|
| identity | 0.220 | 4.4e-2 |
| `x − box₃(x)` | 0.059 | 8.1e-4 |
| `x − box₃(x)`, ÷ local σ (5×5) | 0.054 | 0.0 |
| `−∇²x`, ÷ local σ (9×9) | **0.047** | 0.0 |

The local-variance normalisation is the second half of the whitener: host energy
is spatially non-stationary, so a globally-optimal linear filter still leaves
high-variance regions dominating the sum. Dividing by the local standard
deviation is an approximate per-sample inverse-variance weighting applied before
the projection, where it is cheapest.

These are **no-attack** numbers and the ranking will not survive compression
unchanged — a filter that whitens the host also amplifies codec noise. The
ranking must be re-measured per attack condition (Phase E).

**Matched filter.** `c_k = ⟨Φx, P_k⟩/n`. With exact orthogonality this is the
complete per-patch computation: 32 inner products, or one 4096-point
Walsh-Hadamard transform of `Φx ⊙ R` restricted to the 32 selected rows.

**Per-bit SNR, derived.** `z = A/σ = α·g_Φ/(√32·σ_Φ)`. At `α`=3, `g_Φ`=0.16
(`0.480/3`), `σ_Φ`=0.33: `z` = 1.45–1.61, matching the measured BER. After `M`
independent observations `z_M = z·√M`.

**Answers to §10.** `⟨W,P_k⟩ = n·s_k/√32` for the correct PRN and **exactly 0**
for an incorrect one — not "approximately 0". Interference between PRNs is
exactly zero. Finite PRN length shows up only in the host-noise variance
`σ² ∝ 1/n`, not in the interference term. Attack noise adds to `σ`.

---

## 7. Individual PRN vs combined PRN

**Settled, not empirical.** The signal lies entirely in `span{P₁…P₃₂}`. Write
`x = x_∥ + x_⊥`. The likelihood of any hypothesis `q` depends on `x` only
through `‖x − A·Σq_iP_i‖² = ‖x‖² − 2An·(c·q) + A²·32n`, and `‖x‖²` does not
depend on `q`. Therefore **`c` is a sufficient statistic for the codeword**, and

```
<x, W_q> / n  =  (c · q) / √32
```

is a deterministic function of `c`. Correlating against each of the 200 000
combined patterns produces exactly the ranking `c·q` already gives, at 200 000×
the cost. The 32-D evidence contains strictly more usable information because it
also supports the codeword-independent statistics of §10 and §12.

The 200 000 valid codewords do not change this — they restrict which `q` are
scored, not what statistic to compute.

---

## 8. The 32-D soft evidence representation

Scoring rules, with the ranking each induces:

| rule | verdict |
|---|---|
| `c·q` | **ML** under `c = Aq + n`, `n ~ N(0, σ²I)`. Use this |
| `c·q/‖c‖` | `‖c‖` is independent of `q` → **identical ranking**, no benefit |
| `‖c − Aq‖²` | expands to `‖c‖² − 2A(c·q) + 32A²` → **identical ranking** |
| weighted `Σ c_i q_i/σ_i²` | correct when per-bit variance differs; use it |
| Mahalanobis `qᵀΣ⁻¹c` | reduces to the weighted form when `Σ` is diagonal, which it is at correct alignment. Needed only under misalignment (§24) |
| reliability-weighted likelihood | the per-observation version of the weighted rule. **This is what to implement** |
| learned scoring | no headroom over ML unless the empirical noise is non-Gaussian. Measure the distribution of `c` before considering it |

Hard bits are never formed before the codeword search. `sign(c_i)` discards the
magnitude, and the magnitude is exactly what separates a bit that was measured
well from one that was measured badly.

---

## 9. Closed-set decoding over 200 000 codewords

The codebook is a `200 000 × 32` int8 matrix: **6.4 MB** as int8, 25.6 MB as
float32. One decode is a `200 000 × 32` matrix-vector product = **12.8 MFLOP**,
microseconds on any CPU and irrelevant on a GPU. Exhaustive search is correct
and there is no case for approximate nearest neighbours, hashing or a learned
retrieval model — they would add error to a step that currently contributes
none. This confirms §14.

Per-frame or per-patch codeword voting is strictly worse than pooling the
evidence and decoding once: a hard decision per patch discards the magnitude at
the point where the SNR is lowest (per-patch `z` = 1.6, per-video `z` = 90). Vote
only if the aggregation itself cannot be trusted, and then fix the aggregation.

---

## 10. NO-WATERMARK hypothesis testing

An unwatermarked video always has a best-scoring codeword. Two statistics, both
already computed:

**S₁ — normalised best score.** `S₁ = max_q (ĉ·q)/(σ̂·√32)`. Under H₀ each
`(ĉ·q)/(σ̂√32)` is ~N(0,1) and the maximum over `N` weakly-dependent hypotheses
concentrates near `√(2 ln N)`:

| | value |
|---|---|
| `√(2 ln 200000)` | **4.94** |
| measured max over 200k random codewords, H₀ | 3.89 – 3.97 |
| true-codeword score at per-bit `z` | `z·√32` (7.07 at `z`=1.25) |

An acceptance threshold of 5.5–6 buys a large false-positive margin while
staying far below the H₁ score. The multiplicity **must** be paid for: a naive
3σ threshold on the best of 200 000 fires constantly.

**S₄ — the per-video decoy null.** Read the same clip with `N` decoy keys and require
the true key's `S₁` to be an outlier against them. **This is the only control that
actually works at every clip length and every attack**, and the two below are why it
is needed.

Measured null against observation count, decoy keys on a marked clip:

| attack | frames | observations | H₀ mean | H₀ max | **H₁, true key** | bits |
|---|---|---|---|---|---|---|
| clean | 20 | 960 | 5.67 | 6.96 | 59.2 | 32/32 |
| clean | 60 | 2 880 | 6.41 | 7.35 | 125.2 | 32/32 |
| clean | 200 | 9 600 | 12.51 | **19.07** | 174.4 | 32/32 |
| CRF 28 | 20 | 960 | 5.90 | 7.96 | 3.6 | 24/32 |
| CRF 28 | 60 | 2 880 | 7.16 | 9.31 | 6.8 | 14/32 |
| CRF 28 | 200 | 9 600 | 14.00 | 18.85 | **17.73** | 24/32 |

Two things fall out. First, **the null grows as ≈√M**, so `H₀ max` at 200 frames
(19.07) exceeds the threshold calibrated at 8 frames (17.76) — a fixed threshold cannot
serve every clip length. The mechanism is exactly the correlation §24 flagged: with
per-observation correlation `ρ`, the true variance of the weighted mean plateaus near
`σ²ρ` while the sandwich estimator keeps reporting `σ²/M`, so a standardised null grows
like `√(Mρ)`. Second, and decisively, **at CRF 28 over 200 frames the true key scores
17.73 against its own null of 14.00 ± mean and 18.85 max** — it is inside the null.
That "detection" was never a detection, and no threshold read off other footage could
have known.

Decoy keys are the right null because they see *this* clip: same content, same attack,
same selection, same aggregation, same observation count. Everything that inflates the
true key's score inflates theirs. Measured on a marked 200-frame clip, the true key
scored 174.4 against decoys spanning 8.9–19.1. It costs `(1+N)×` the correlation pass,
in a single pass over the video, and it is worth it — see §23 for how to make it rare.

**S₃ — split-half agreement.** Decode two disjoint halves of the frames and require
them to return the same ID.

This exists because of a failure the other two statistics cannot see, and which only
showed up once long clips were tested. **Absence was never the dangerous case; partial
presence is.** At H.264 CRF 28 over 200 frames the detector reported `S₁` = 17.7 with
**24 of 32 bits correct** — some bits genuinely recovered, the rest at chance. The
search over 200 000 hypotheses then finds a codeword that fits both the surviving
evidence *and* the noise, and it clears a threshold calibrated against H₀ because it is
not an H₀ event. A confidently wrong ID is worse than a refusal.

Two disjoint halves landing on the same *wrong* ID has probability of order 1/200 000;
landing on the same right one is what a real mark does. The split is by frame
**parity**, not into a first and second half — content drifts across a clip and a
temporal split would confound "different content" with "independent noise", whereas
parity leaves both halves with the same content distribution and the same length. It
costs two extra GEMMs and no extra video decoding, because the per-frame evidence is
already held.

It catches noise-driven partial recovery cleanly — all three of the CRF 28/32/36
false attributions above were rejected, and the CRF 23 true positive kept. It does
**not** catch content-driven bias: of six decoy keys on a marked 200-frame clip, five
disagreed across halves and were rejected, but **one agreed and was accepted**. Both
halves see the same scene, so a bias that comes from content rather than from noise
reproduces in both. That residual is what S₄ exists for.

**S₂ — evidence energy.** `S₂ = Σ_k ĉ_k²/σ̂²`, which is **codeword-independent**:
χ²₃₂ under H₀, non-central χ²₃₂ under H₁. It answers "is *a* watermark present"
without reference to the ID list, so it is not inflated by the size of that
list, and it is the same quantity the localiser (§12) already maximises.

Accept only when both fire. Reporting them separately also diagnoses the two
distinct failures: high `S₂` with low `S₁` means "a mark is there but its ID is
not in the list"; low `S₂` means "no mark".

**Best-vs-second margin: do not use it as the primary statistic.** Measured
under H₀ with a *random* 200 000-ID set: 0.002 – 0.011. The codebook is dense in
Hamming space — the runner-up differs from the winner in one or two bits and
scores almost identically. With a `d_min` = 6 codebook the runner-up is six bits
away and the margin becomes informative; §11 quantifies this. It is a useful
*secondary* diagnostic, never the acceptance test.

**The selection bias, and why it is the main correction.** Sites are chosen by
maximising `T`, and then `T` is used as the evidence that a mark is there. That is
selection on the statistic being tested, and it is not small. Measured on
unwatermarked 1080p, the median `T/σ²` at the chosen sites is **73** against an
unselected χ²₃₂ mean of **32** — which hands a clean clip a Wiener weight of 0.56
out of nothing at all. Verified that it is the *selection* and not the weighting or
the robustness pass: random sites give `S₂` = 31–41 under every weighting scheme,
selected sites give 81–83 under every one.

The correction is to compare each site against the **order statistic for its own
rank** rather than against the mean. The `r`-th largest of `M` draws from χ²_K sits
near its upper `r/M` quantile — predicted 77.0 at rank 24 of 1.7 M positions against
the 73.0 measured. It fixes both ends at once:

| | clean `S₂` | marked `S₂` | clean `S₁` | marked `S₁` |
|---|---|---|---|---|
| uncorrected | 81.8 | 1 147 | 6.02 | 33.5 |
| **rank-aware** | **34.3** | **4 481** | **4.27** | **66.9** |

Clean `S₂` lands on the χ²₃₂ null it should have been on all along, and the marked
statistic *rises* fourfold, because real sites now outweigh spurious ones instead of
merely outnumbering them.

**Calibration.** Even corrected, the parametric threshold is a starting point rather
than the deliverable. Measured clean-clip `S₁` reaches 6.65 against a parametric
6.81 at nominal FPR 1e-6 — too close to ship on. `eval/calibrate.py` draws the null
from **decoy PRN seeds** rather than from more clips: a decoy key has the same
construction and the same exact orthogonality but no relation to what is embedded,
so every seed is an independent null draw on real content, through the real attack,
with the real selection and aggregation. That matters here because
`inputs/{15,30,60,120}.mp4` are byte-identical content at different lengths —
verified — so the machine holds only two distinct sources. Running decoy keys
against a *genuinely marked* clip is also the strongest false-positive case
available: a hit there means the detector is reading content, not the mark.

**Measured, and the parametric threshold is badly wrong.** 24 decoy keys on unmarked
1080p, 8 frames:

| | value |
|---|---|
| observed `S₂` inflation over the χ²₃₂ null | **1.73×** |
| decoy runs admitted at the parametric threshold | **3 of 24** |
| unmarked sweep cells admitted at the parametric threshold | **6 of 30** |
| `S₁` threshold, parametric, nominal FPR 1e-6 | 6.81 |
| **`S₁` threshold, empirical (Gumbel tail), FPR 1e-6** | **17.76** |
| `S₂` threshold, parametric → empirical | 85.2 → 147.6 |

A nominal 1e-6 threshold delivering roughly one false positive in five cells is not a
calibration error at the margin, it is the wrong threshold. Re-scored at `S₁` ≥ 17.76
the whole sweep has **zero** false positives and no false attributions, and every
attack that was recovering the payload still recovers it except blur σ2.0, which drops
to a conservative miss. `WatermarkHypothesisTester.from_calibration()` is the intended
constructor; the parametric values survive only as a labelled fallback.

The 1e-6 and 1e-9 rows are extrapolations from 24 samples through a fitted tail, and
are labelled as such in the output. They show how far the observed null sits from the
parametric one; they do not certify a rate that has not been observed.

---

## 11. Codeword and Hamming-distance analysis

This section is the largest free lever in the system.

**Why distance is the right quantity.** Two IDs differing in `d` positions give
scores differing by `2·Σ_{d positions} ĉ_i s_i`, so the pairwise confusion
probability is `Q(z√d)` at per-bit SNR `z`. Minimum distance sets the error
exponent, and it costs **nothing** in embedded energy or perceptual distortion.

**Random 200 000 IDs are bad.**

| distance | expected colliding pairs among 200 000 random IDs |
|---|---|
| ≤ 1 | ~154 |
| ≤ 2 | ~2 463 |
| ≤ 3 | ~25 560 |
| ≤ 4 | ~193 011 |

`P(a given ID has a neighbour within 3)` = 0.23; within 4 = 0.85. Minimum
distance is 1.

**Coding gain.**

| `d_min` | amplitude gain | dB | observations needed |
|---|---|---|---|
| 1 (random) | 1.00 | 0.0 | 1× |
| 4 | 2.00 | 6.0 | 1/4× |
| **6** | **2.45** | **7.78** | **1/6×** |
| 8 | 2.83 | 9.03 | 1/8× |

**Constructions**, all in `ids.py`, minimum distance and weight distribution
computed exactly by enumerating the code rather than quoted from a table:

| code | size | `d_min` | `A_dmin` | fits 200 000? |
|---|---|---|---|---|
| **extended BCH [32,21,6]** | 2 097 152 | **6** | 992 | **yes, 10× headroom** |
| Reed-Muller RM(2,5) [32,16,8] | 65 536 | 8 | 620 | no |
| best searched [32,18] subcode | 262 144 | 6 | — | yes, but no gain |
| 200 000 random IDs | 200 000 | **1** | 136 pairs | — |

Measured weight distributions (the first few terms; the union bound needs all of
them, not just `d_min`):

```
extBCH[32,21,6]   6:992    8:10,540  10:60,512  12:228,160  14:446,400  16:603,942 …
RM(2,5)[32,16,8]  8:620   12:13,888  16:36,518  20:13,888   24:620      32:1
```

Union bound on codeword error, `Σ_d A_d · Q(z√d)`, at per-bit SNR `z`:

| `z` | extended BCH `d`=6 | RM(2,5) `d`=8 |
|---|---|---|
| 2.0 | 5.7e-4 | 4.8e-6 |
| 2.5 | 4.6e-7 | 4.8e-10 |
| 3.0 | 1.0e-10 | 6.7e-15 |

**Recommendation: take 200 000 codewords from the extended BCH [32,21,6].** Any
subset of a linear code inherits its minimum distance — the difference of two
codewords is itself a codeword, so every pairwise distance is the weight of some
nonzero codeword. The guarantee therefore holds for *any* 200 000 chosen, IDs can
be issued incrementally without re-deriving anything, and there are ten times as
many available as needed. Verified: zero pairs at distance ≤ 2 in the selected
subset.

A search over 3 000 random [32,18], [32,19] and [32,20] subcodes found **none**
with `d_min` above 6, so 6 is the practical ceiling at this size and the simple
construction is also the best one found.

Reed-Muller's `d`=8 is worth roughly **1.2 dB** — about two orders of magnitude in
error rate at fixed `z` — but caps the system at 65 536 IDs. That is a business
decision, not a technical one, and it is the only place in the design where ID
count and robustness genuinely trade against each other.

The random baseline confirms the prediction: **136 pairs at Hamming distance 1**
among 200 000 uniformly drawn IDs. Those 272 IDs are one bit-error away from
being misattributed to each other, permanently, and nothing downstream can fix it.

**A second, operational reason to use a linear code.** Its distance distribution
is identical viewed from every codeword, so **every ID has the same
misattribution probability**. With random IDs the error rate varies by ID and
whoever is issued one of the ~154 distance-1 pairs is permanently worse off,
invisibly.

**Union bound.** For soft ML decoding the relevant quantity is the full weight
distribution, not `d_min` alone: `P(error) ≤ Σ_d A_d · Q(z√d)`. `ids.py`
computes `A_d` exactly by enumerating the code and reports the bound at each
`z`. `d_min` alone understates the error when `A_{d_min}` is large.

---

## 12. Patch localisation

**The constraint.** Measured per-patch BER against sub-pixel translation, whitened:

| shift (px) | 0 | 0.25 | 0.5 | **1.0** | 1.5 | ≥2 |
|---|---|---|---|---|---|---|
| BER | 0.052 | 0.058 | 0.076 | **0.228** | 0.574 | ~0.5 |

One pixel of misalignment costs four-fifths of the evidence. At 1.5 px the
correlation goes *negative* (0.574 > 0.5) — the interpolated carrier is
anti-phase. **Localisation must be accurate to ≲0.5 px**, which integer-pixel
search already delivers (worst-case residual 0.5 px → BER 0.076). Sub-pixel
refinement is a bonus, not a requirement.

**Why re-detecting SIFT keypoints is the wrong approach.** The embedder chooses
patches by SIFT on the *unmarked* frame. After an attack the keypoint set moves,
the ordering by `(size, response)` reshuffles, and the greedy suppression cascade
amplifies a single reordering into a different patch set. Even a one-pixel
keypoint shift destroys the correlation. Feature re-detection is solving a harder
problem than the one that needs solving.

**What to do instead: let the watermark localise itself.** `T(x,y) = Σ_k c_k(x,y)²`
is **independent of the codeword**, needs no ID list and no original frame, and
is the same computation the decoder needs anyway.

Algorithm, per frame:

1. Form four Haar-LL planes by 2× box-decimation at phases `(0,0),(0,1),(1,0),(1,1)`
   — `LL = 2 × 2×2 block mean`. The four phases cover every integer pixel offset,
   including the parity that a single decimation would miss.
2. Whiten each plane (§6).
3. FFT cross-correlate each plane against all 32 PRNs. Template FFTs are
   precomputed once.
4. `T = Σ_k c_k²`, interleave the four phase maps back to full resolution.
5. Peak-pick with non-maximum suppression at the patch pitch. Each peak yields
   its 32-D `c` for free — it is already computed.

**Measured, blind, on 3 frames / 22 embedded patches:**

| attack | none | jpeg90 | jpeg70 | jpeg50 | 0.75× | 0.5× | blur σ1.0 | blur σ1.5 | noise σ5 | γ1.3 |
|---|---|---|---|---|---|---|---|---|---|---|
| localised ≤1 px | 19/22 | 19/22 | 16/22 | 11/22 | 14/22 | 8/22 | 12/22 | 9/22 | 18/22 | 18/22 |
| 32-bit decode | **OK** | **OK** | **OK** | 3 bad | **OK** | 6 bad | 5 bad | 10 bad | **OK** | **OK** |

Twenty-two observations is three frames; a 120-frame clip yields ~900. The
failures are the white-carrier problem of §4, not a localisation-method problem.

**Cost.** Four phases × 32 PRNs over a 960×540 plane ≈ 6 GFLOP/frame — a few
milliseconds on the T4 present on this machine, 1–2 s/frame on CPU. Restricting
to the top peaks after a cheap 8-PRN pass cuts it 4×.

**Global geometry.** Scale and rotation are **per-video constants** for digital
attacks — a resize or a rotate is applied once to the whole clip — so the search
runs on one or two frames and the answer is then locked. Translation and crop need
no search at all: the correlator is already a full-frame search.

The objective has to be chosen carefully, and two reasonable-looking ones fail:

| objective | why it fails | measured |
|---|---|---|
| peakiness of `T`, frame warped per hypothesis | undoing a small scale means upsampling; upsampling smooths the plane; the whitened residual of a smooth plane is heavy-tailed, so peakiness rises with upsampling whether or not a mark is present | picked the smallest scale on the grid every time; scored an unmarked clip 265 against a marked one 249 |
| directional coherence of the top peaks, template warped instead | better founded — one payload does put the same 32-vector at every patch — but natural image structure is coherent too | unmarked 0.25–0.34 vs marked 0.21–0.38: no separation |
| **the detector's own evidence, `Σ` per-site `z²`** | codeword-independent, and it is the quantity the decoder consumes anyway | **works** |

Measured with the working objective, marked clip — every scale recovered exactly:

| attack | true scale | found | verdict |
|---|---|---|---|
| clean | 1.0000 | **1.0000** | `S₁` = 39.97, exact ID |
| 1080p → 720p | 0.6667 | **0.6667** | `S₁` = 33.48, exact ID |
| 1080p → 540p | 0.5000 | **0.5000** | `S₁` = 27.25, exact ID |
| 1080p → 360p | 0.3333 | **0.3333** | `S₁` = 14.80, exact ID |
| crop 0.75 + rescale | 1.3333 | **1.3333** | `S₁` = 30.43, exact ID |

A 3× downscale still decodes exactly. Without the search the same attack lost six
bits.

**The search must be anchored to "no geometric change", and this is not optional.**
It maximises the same evidence the decision is later made on, so left free it is a
second layer of selection bias stacked on the one §10 already corrects. On an unmarked
clip it locks whichever of ~21 hypotheses looks best, and the resampling that follows
inflates the null — measured `S₂` = 34 with the search off against 74 with it on. At
one setting it produced an outright **false positive**: an unmarked clip locked to
scale 1.1 and came out at `S₁` = 7.43, `S₂` = 91.4, over both acceptance thresholds.

A non-identity geometry therefore has to earn it three ways — stand out from the grid
by ≥ 15 robust standard deviations above the median hypothesis, beat the identity
hypothesis by 1.5×, and clear an absolute per-frame evidence floor. Those come from
the measured gap rather than from a guess:

| | lock quality `z` | outcome |
|---|---|---|
| marked, unattacked | **124** | locks 1.0000 |
| marked, 540p leak | **53** | locks 0.5000 |
| unmarked, clean | 5.0 | anchored → `S₁` 4.96 |
| unmarked, 540p | 7.4 | anchored → `S₁` 5.86 |

The asymmetry is deliberate: an unnecessary resample costs evidence, a missed one
costs all of it, but a spurious one costs evidence *and* invents a detection.

**Comparison against §18's list.**

| method | geometric robustness | cost | verdict here |
|---|---|---|---|
| SIFT re-detection | poor — keypoints move | low | **no**: cannot reach 0.5 px |
| SuperPoint / DISK / LightGlue / LoFTR | good, needs a reference image | ~600 ms/frame | only for camera capture, and only non-blind |
| dense sliding window (pixel domain) | translation + crop | high | subsumed by the FFT version |
| **dense watermark correlation** | translation + crop, plus scale/rotation by coarse search | ~6 GFLOP/frame | **this one** |
| temporal patch tracking | n/a | — | unnecessary: no temporal code |

Feature-based localisation earns its place only when the attack includes a
per-frame homography, i.e. camera capture, which §21 defers. For the digital
attack set the answer is unambiguously **dense watermark-domain search**.

---

## 13. Patch reliability

Every candidate site gets a variance estimate, and nothing is ever hard-gated.

**`σ̂` from the off-peak floor.** The correlation maps already contain, at every
non-peak position, a sample of the local noise distribution. The robust scale of
`c_k` over a ring around the peak — `1.4826 · median|c|` — is a per-site noise
estimate that costs nothing and adapts to local texture, local compression
damage and local blur simultaneously. Weight `w = 1/σ̂²`.

**Secondary features**, if a learned estimator is built later (§16): peak
sharpness (`T(peak)` over the ring median), local whitened variance, local
contrast, distance to the frame edge, and the residual `‖c‖² − (c·q̂)²/32` after
the winning codeword is known — a large residual means the site's evidence does
not look like *any* codeword.

**What not to do.** Do not drop low-reliability patches. A patch with `σ̂` twice
the median still carries a quarter of a good patch's evidence, and discarding it
throws that away for nothing. Inverse-variance weighting already suppresses it by
exactly the right factor.

---

## 14. Spatial aggregation

Across patches within a frame: inverse-variance weighted mean of the per-site
`c`, which is the ML combination for independent Gaussian observations of a
common mean:

```
ĉ = Σ_m w_m c_m / Σ_m w_m ,   w_m = 1/σ̂_m² ,   Var(ĉ) = 1/Σ_m w_m
```

Report `M_eff = (Σw)²/Σw²` alongside — the effective observation count, which is
below `M` whenever one patch dominates, and is the honest denominator for a
`z`-score.

Patches within a frame are close to independent: they are non-overlapping and
sit on different content, and the mark is identical in each, so the only shared
term is the codeword itself. This is the cleanest redundancy axis in the system.

---

## 15. Temporal aggregation

**There is no temporal synchronisation problem.** The payload has no temporal
code — the same `W` is in every patch of every frame. Frame dropping,
duplication, reordering, frame-rate conversion, temporal shift and interpolation
therefore cost observations and nothing else. No aligner, no phase search, no
sync word, no frame-to-bit map. This is a genuine structural advantage of the
scheme and it removes an entire category of failure.

Aggregation is the same inverse-variance sum extended over frames, with one
addition: **Huber clipping**. A mis-localised patch contributes an essentially
random `c` of full magnitude, which is a heavy-tailed contaminant that
inverse-variance weighting does not suppress (its `σ̂` looks normal — it is the
*mean* that is wrong). Clip the weighted contributions at ~3× the robust scale
of the accumulated evidence.

Two cautions carried from the aggregation literature and worth stating because
they are easy to get wrong:

- The clipping scale must be the **observed** spread of the contributions, not
  the theoretical null spread. A fixed multiple of the null pins every
  contribution at the ceiling once the evidence is strong, silently degrading a
  soft sum into a majority vote.
- Never hard-gate frames by any quality measure. Soft down-weighting keeps the
  information; a threshold discards it.

Against §19's list: mean is ML but not robust; median and trimmed mean discard
too much at low per-observation SNR; **Huber over inverse-variance weights** is
the right point on that curve; RANSAC does not apply (this is not a geometric
consensus problem); learned temporal aggregation has no headroom over ML
pooling of near-independent observations and should not be built.

---

## 16. Neural network opportunities

Ranked by *measured* headroom. This ranking contradicts §25's own preference,
and the reason is worth stating.

**1. Learned host suppression, in front of the correlator.** Host rejection is
worth 7.5 dB (§6) and a hand-tuned 3×3 box filter is certainly not the optimum.
A small CNN predicting `LL(x)` from its neighbourhood — a learned Wiener
denoiser — subtracts a better host estimate. Three properties make this the right
place:

- It is **scalar regression with unlimited self-supervised data**, trainable on
  *unwatermarked* video with no labels and no watermark at all.
- It operates where the spatial information still exists. After projection to
  32-D that information is gone.
- It never sees the PRNs, which is exactly what §24 asks for: the network models
  the **host**, not the mark.

**2. Learned reliability estimation.** A small MLP over the §13 features,
producing `σ̂` instead of the robust-scale heuristic. Cheap, bounded gain, and it
degrades gracefully — a bad `σ̂` costs weighting efficiency, not correctness.

**3. Learned evidence correction (§25) — ranked below both.** At correct
alignment with exactly orthogonal PRNs there is **no bias to correct**: `c` is an
unbiased estimate of `A·s` plus Gaussian noise. The residual error is host noise,
and the right place to attack host noise is item 1. Correcting after the
projection asks the network to undo a loss using information the projection has
already destroyed. Build it only if 1 and 2 leave a measured gap — for example if
misalignment turns out to introduce a systematic, learnable bias (§24).

**4. Learned temporal aggregation, transformers — do not build.** Near-independent
observations of a common mean have a closed-form optimal combiner.

**5. CNN on patches / on DWT coefficients / Siamese / metric learning — do not
build.** Any of these asks a network to rediscover a matched filter that is
already exactly known and exactly optimal. This is §24's point and it is correct.

Against §23's numbered list: (1) and (2) are the baseline and are strong; (11)
and (12) are items 1 and 2 above and are what I would build; (14) is the final
architecture; (3)–(10) and (13) should not be built for this system.

---

## 17. Recommended hybrid architecture

```
                          suspect video
                                │
                  ┌─────────────┴─────────────┐
                  │  once per video           │  per frame
                  │  GeometrySearch           │
                  │  argmax over (scale,rot)  │
                  │  of Σ T, 8 PRNs, 5 frames │
                  └─────────────┬─────────────┘
                                ▼
                        luma → 4-phase 2× decimation
                                ▼
                        PreWhitener        ← learned variant slots in here
                                ▼
                        PRNCorrelator      FFT × 32 PRNs → c_k(x,y)
                                ▼
                        PatchLocaliser     T = Σ c_k², NMS
                                ▼
              PatchEvidenceExtractor + PatchReliabilityEstimator
                                ▼
                 SpatialAggregator → TemporalAggregator
                   inverse-variance, Huber-clipped
                                ▼
                          ĉ ∈ R³², per-bit z
                                ▼
              CodewordDecoder      200 000 × 32 GEMM, score = ĉ·q
                                ▼
              WatermarkHypothesisTester    (S₁, S₂) → ID | NO_WATERMARK
```

Modular per §33, one concern each, with the learned components behind the same
interfaces as their classical versions so they can be A/B'd from a config flag.

---

## 18. Training strategy

Only two components are learned, and both are easy to train.

**Host suppressor.** Self-supervised on *unwatermarked* video: predict the centre
LL coefficient from its neighbourhood, minimise squared error. No watermark, no
labels, no ID list. Then fine-tune on attacked unwatermarked video so it does not
merely learn the clean-image prior. Hold out whole source videos, never frames —
frames within a clip are near-duplicates and a frame-level split leaks.

**Reliability estimator.** Supervised on synthetic data where the true payload is
known: the target is the realised squared error `‖c_m − A·s‖²`, which is
observable because `s` is known at training time.

Randomise across source videos, IDs, patch locations, ALPHA, attack type,
attack strength and attack composition (§19). Curriculum — clean first, then
progressively harder mixtures — matters for the suppressor because a network
trained only on severe attacks under-performs on clean input, which is the case
that matters most in production.

Hard-negative mining and difficult-codeword-pair mining, which §22 asks about,
are **not applicable**: neither learned component sees the codeword. Difficult
codeword pairs are handled structurally by the minimum distance of the ID set
(§11), which is a better answer than training against them.

---

## 19. Attack augmentation

`src/retrieval/capture_sim.py` already provides a well-designed simulator with a
session-constant homography and per-frame jitter, covering: perspective, zoom,
rotation, shake, gamma, exposure, white balance, contrast, vignette, moiré,
Gaussian blur, motion blur, resample, Gaussian noise, per-frame JPEG.

Missing, relative to §21, and to be added in `eval/attacks.py`:

- **H.265/libx265**; H.264 CRF, bitrate and preset sweeps; any video re-encode
  after the capture simulation (currently only per-frame JPEG).
- **True resolution change** — 1080p → 720p/540p/360p and back up.
  `CaptureConfig.out_size` exists but is never exercised.
- **Crop**, and crop-and-rescale.
- Sharpening, denoising, saturation, grayscale.
- **Temporal**: frame drop, duplication, frame-rate change, temporal shift,
  interpolation. Cheap to add — and the prediction from §15 is that the scheme is
  near-immune to all of them, which is itself a result worth having on record.
- Combined chains: `resize + H.264 + blur + crop`, and
  `resize + compression + photometric + frame drop`.

---

## 20. Evaluation methodology

One scan per (source, ID, attack) cell producing the raw per-patch evidence;
every metric and every decoding variant is then cheap arithmetic on the stored
arrays. Results appended to `findings.jsonl` per cell so a killed run loses one
cell, plus a markdown table.

Metrics, per §30: bit accuracy and BER at patch, frame and video level; exact
32-bit recovery; valid-codeword recovery; false-positive and false-negative rate;
precision, recall, ROC, AUC; confidence calibration; best-vs-second margin; and
`M_eff`, `z_min` (the weakest bit's `z` — 32/32 with no margin is not the same
result as 32/32 with room to spare) and the frames-to-first-correct-decode curve.

Three controls, all mandatory:

- **Unwatermarked clip** must return `NO_WATERMARK`.
- **Clip carrying a different valid ID** must return *that* ID. This is the test
  a "was something embedded here" statistic passes and an ID detector must not.
- **Same-popcount decoy ID** so nothing can pass on a bit-balance cue.

Attack list: none; H.264 CRF sweep; H.265 CRF sweep; resolution sweep; blur
sweep; noise sweep; resize sweep; crop sweep; rotation sweep; grayscale;
brightness; contrast; saturation; gamma; and combined chains.

**Baseline comparison against the 1-bit system** (§30) needs care to be fair:
the 1-bit system carries 1 bit at full amplitude, the 32-bit system carries 32 at
`1/√32` each. The comparable quantity is *presence-detection* reliability at
equal PSNR, plus the 32-bit system's payload capacity as the thing being bought.
Comparing per-bit BER directly would be meaningless.

---

### Measured results

First full sweep: 30 attacks × 3 cases, 128/L1 `α`=3, 20 frames, scale search on,
scored against the **calibrated** threshold of §10 (`S₁` ≥ 17.76). Full table in
`FINDINGS.md`, raw rows in `findings.jsonl`.

**18 of 30 attacks recover the exact 32-bit payload, with zero false positives and no
false attributions.**

| passes exactly | `S₁` |
|---|---|
| clean | 43.8 |
| H.264 CRF 23 | 25.1 |
| 1080p → 720p (scale found 0.6667) | 41.2 |
| 1080p → 540p (scale found 0.5000) | 37.8 |
| crop 0.75 · crop 0.75 + rescale | 30.7 · 31.6 |
| resample 0.5 · translate 2 px · sharpen 1.0 · blur σ1.0 | 34.1 · 43.6 · 44.0 · 29.9 |
| gamma 1.6 · contrast 1.5 · saturation 0 · grayscale | 46.3 · 40.3 · 38.4 · 43.8 |
| **frame drop 20 % · half frame rate · reverse · interpolate** | **40.0 · 30.5 · 43.8 · 31.4** |

The temporal row is the one worth pausing on: dropping a fifth of the frames, halving
the frame rate, playing the clip backwards and averaging adjacent frames all cost
observations and nothing else. That is what having no temporal code buys, and it is
measured rather than argued.

| fails | why |
|---|---|
| H.264 CRF 28 / 32 / 36, H.265 CRF 28 / 32, 1 Mb/s, 500 kb/s | the white carrier (§4) |
| noise σ = 10 | same band |
| combined chains containing heavy compression | same |
| 1080p → 360p at 20 frames | marginal; recovered exactly in the isolated 8-frame test with a clean scale lock, so a lock-stability limit rather than a signal limit |

**Compression past CRF 23 is the single first-order weakness**, and it is the same
mechanism in every failing row. §21 measures the fix.


## 21. Perceptual quality

Perception is the hard constraint, so it gates every robustness claim rather
than following it.

**Current operating point**, derived exactly from §3: pixel mark RMS **1.5**,
peak **6.36** grey levels, **PSNR 44.6 dB** at `α`=3.

**Measure PSNR, SSIM and LPIPS, not PSNR alone**, for two specific reasons:

1. **The peak-to-RMS change.** The 32-bit mark is Gaussian-distributed with
   peak/RMS 2.83–4.60; the 1-bit mark was binary with peak/RMS exactly 1.0. At
   equal PSNR these look different — sparse impulsive speckle versus uniform
   dither — and PSNR cannot distinguish them. If ALPHA was calibrated by eye on
   the 1-bit mark, that calibration does not transfer.
2. **The chip-size question.** At **identical** pixel mark RMS (1.5, PSNR 44.6 dB),
   varying only the carrier's spatial frequency:

   | config | patches/frame | none | j70 | j50 | 0.75× | 0.5× | σ1.0 | σ1.5 | σ2.0 |
   |---|---|---|---|---|---|---|---|---|---|
   | 128/L1, chip 2×2 *(current)* | 7.0 | 0.054 | 0.093 | 0.167 | 0.079 | 0.155 | 0.107 | 0.199 | 0.270 |
   | **256/L2, chip 4×4** | 6.8 | 0.006 | 0.027 | 0.061 | 0.037 | 0.087 | 0.067 | 0.094 | 0.127 |
   | 512/L3, chip 8×8 | 3.5 | 0.000 | 0.004 | 0.018 | 0.004 | 0.047 | 0.011 | 0.040 | 0.078 |

   3–9× lower BER on every attack, at no cost in patch count for 256. But
   **PSNR-matched is not perception-matched**: contrast sensitivity peaks around
   3–5 cycles/degree, so lower-frequency noise of equal RMS is *more* visible.
   The comparison must be re-run at perceptually-matched strength before this can
   be claimed. It is the single most promising embedding change and it is kept
   separate per §36.

Also measure temporal flicker: the mark is identical in every frame but the
*patch locations* move with the content, so a patch appearing and disappearing
between frames is a real artefact this scheme can produce and a still-frame
metric cannot see.

---

### Measured

Ten frames of `inputs/30.mp4`, three configurations at matched **local** mark
amplitude, and one at matched **frame** PSNR:

| config | chip | patches/f | coverage | mark RMS | peak | frame PSNR | marked-px PSNR | SSIM | LPIPS | flicker |
|---|---|---|---|---|---|---|---|---|---|---|
| 128/L1 `α`=3 | 2×2 | 7.5 | 5.1 % | 1.78 | 5.0 | 56.11 | 43.14 | 0.9990 | 0.00011 | 0.564 |
| **256/L2 `α`=3.75** | 4×4 | 6.7 | 12.7 % | 1.27 | **3.0** | **55.07** | **46.08** | 0.9990 | **0.00007** | 0.622 |
| 256/L2 `α`=6 | 4×4 | 6.7 | 18.2 % | 1.78 | 5.0 | 50.56 | 43.14 | 0.9973 | 0.00029 | 1.045 |
| 512/L3 `α`=12 | 8×8 | 3.7 | 30.2 % | 1.77 | 5.0 | 48.43 | 43.15 | 0.9972 | 0.00050 | 1.221 |

The frame-PSNR difference between the first two rows is entirely coverage: a 256 patch
covers four times the area, so matching total distortion means running a *lower* local
amplitude — which is why the 4×4 chip at `α`=3.75 is perceptually **better** than the
2×2 at `α`=3 on marked-pixel PSNR (+2.9 dB), peak excursion (3.0 vs 5.0), and LPIPS,
with SSIM identical. Only flicker is marginally worse.

### And what that buys, end to end

20 frames, calibrated threshold `S₁` ≥ 17.76. `S₁` first, then bits recovered:

| attack | 128/L1 `α`=3 (56.1 dB) | **256/L2 `α`=3.75 (55.1 dB)** | 256/L2 `α`=6 (50.6 dB) |
|---|---|---|---|
| clean | 43.8 · 32/32 ✓ | **74.8 · 32/32 ✓** | 117.7 · 32/32 ✓ |
| H.264 CRF 23 | 25.1 · 32/32 ✓ | 24.5 · 32/32 ✓ | 82.3 · 32/32 ✓ |
| **H.264 CRF 28** | 5.4 · **16/32 — chance** | 7.5 · **32/32 bits, below threshold** | **42.1 · 32/32 ✓** |
| H.264 CRF 32 | 4.7 · 18/32 ✗ | 2.8 · 26/32 ✗ | 5.3 · 24/32 ✗ |
| **blur σ2.0** | 13.4 · 32/32, **below threshold** | **52.4 · 32/32 ✓** | 105.5 · 32/32 ✓ |
| 1080p → 540p | 37.8 · 32/32 ✓ | **74.3 · 32/32 ✓** | 118.9 · 32/32 ✓ |

The middle column is the honest comparison — same frame PSNR, *better* local
perceptual metrics — and it is better or equal on five of six rows. It roughly doubles
margin on clean, 540p and blur, converts blur σ2.0 from a miss into a comfortable pass,
and **gets every bit right at CRF 28 where the 2×2 chip returns chance**. What is
missing at CRF 28 is only the confidence to accept at a 1e-6 false-positive rate on 20
frames, and `S₁` grows as √frames. CRF 32 defeats all three.

**Amplitude cannot be traded down freely, and this was not obvious.** `S₁` at CRF 23
is 82.3 at `α`=6 and 24.5 at `α`=3.75 — a factor of 3.4 for a factor of 1.6 in
amplitude, far worse than linear. Compression is a quantiser, not additive noise:
below the quantiser step the mark is destroyed rather than attenuated. Any ALPHA
reduction has to be measured, never scaled.

**Recommendation, stated as the trade it is.** Chip 4×4 is the right move, but it is a
curve rather than a free lunch: at strictly matched frame PSNR it converts CRF 28 from
unrecoverable to recoverable-but-unconfident, and spending 5.5 dB of frame PSNR
(`α`=6) makes CRF 28 comfortable. Which point to pick is a viewing decision on real
content, not a metric decision — every LPIPS value in the table is far below any
perceptibility threshold.


## 22. ECC strategy

**Do not add an ECC layer. Make the ID set be the code.**

A 32-bit payload with a separate ECC layer would spend bits on parity and shrink
the ID space. The closed-set constraint already supplies exactly the same
redundancy for free: restricting to 200 000 of 2³² values *is* a rate-17.6/32
code, and the only question is whether it is a *good* one. §11 says make it
extended BCH [32,21,6] and it costs nothing.

**Soft, not hard.** Scoring `ĉ·q` over the codebook is precisely soft-decision
ML decoding of that code. Thresholding to hard bits and then correcting loses the
usual ~2 dB soft-decision advantage, and here it is worse than usual because the
per-patch SNR is low enough (`z` ≈ 1.6) that hard decisions are frequently wrong
in exactly the cases where the magnitude would have said so.

**Relating BER to exact recovery.** With independent bits at BER `p`, exact
32-bit recovery is `(1−p)³²` — 0.52 at `p` = 0.02, 0.72 at `p` = 0.01. This is
§26's point and it is real: 95 % bit accuracy gives 19 % exact recovery. Closed-set
decoding breaks that relationship, because it never requires all 32 bits to be
independently right — it requires the true codeword to outscore all others, which
`d_min` = 6 makes far easier than 32 independent correct decisions.

**ECC does not rescue weak detection**, and §28 is right to warn about it. The
measured union bound at `d_min` = 6 is 5.7e-4 at `z` = 2.0 but ≈1 at `z` = 1.0 —
coding multiplies the effective SNR by `√d_min`, it does not create SNR. The
requirement is still to get per-bit `z` above about 2 by aggregation first.

---

## 23. Computational complexity

Per 1080p frame, blind:

| stage | cost | notes |
|---|---|---|
| 4-phase decimation + whitening | ~20 MFLOP | trivial |
| FFT correlation, 32 PRNs × 4 phases | ~6 GFLOP | GPU: few ms. CPU: 1–2 s |
| `T = Σc²`, NMS, peak extraction | ~10 MFLOP | trivial |
| evidence + reliability, ~27 sites | ~1 MFLOP | already computed |
| aggregation | negligible | |
| **codeword search, once per video** | **12.8 MFLOP** | 200 000 × 32 GEMM |
| geometry search, once per video | ~800 GFLOP | 5 frames × 132 hypotheses × 8 PRNs |

The correlator dominates and is embarrassingly parallel; `torch.fft` on the T4
present here handles it. The 200 000-codeword search — the thing §14 worried
about — is **0.2 % of one frame's correlator cost**, done once per video.
Everything in NumPy/OpenCV/PyWavelets/PyTorch as §33 requires.

An 8-PRN first pass for localisation followed by the full 32 only at surviving
peaks cuts the dominant term ~4× at no measured cost in accuracy — worth doing if
CPU-only deployment is needed.

**Making the per-video decoy null affordable.** S₄ (§10) costs `(1+N)×` the correlator,
which is the single largest cost in the system. It does not have to run on every clip.
The true key's `S₁` separates into three bands, and only the middle one is in doubt:

| `S₁` | action | cost |
|---|---|---|
| far above any observed null (say > 60) | accept | 1× |
| far below (say < 10) | reject | 1× |
| in between | run S₄ with `N` = 5–8 | (1+N)× |

Measured, the bands are wide: genuine unattacked and lightly-compressed detections came
in at 25–174, and every decoy key measured below 20. The expensive path is for the cases
that actually need adjudicating — heavy compression, marginal clips — and those are
exactly the cases where being wrong matters most. Band edges must come from the same
calibration run that sets the thresholds, not from these numbers, which are one clip.

---

## 24. Failure modes

| failure | mechanism | mitigation | status |
|---|---|---|---|
| **Compression past H.264 CRF 23** | the carrier is white in LL; half its energy is above the band compression keeps (§4). The single first-order weakness — every failing sweep row is this mechanism | chip size: 4×4 moves the limit to CRF 28 (§21) — an embedding change, kept separate per §36 | **measured, fix measured, not applied** |
| Blur σ ≥ 2, noise σ ≥ 10 | same band | same fix; blur σ2.0 recovers 32/32 but below the calibrated threshold | measured |
| **Parametric acceptance thresholds** | Gaussian/χ² null is optimistic by 2.6× on `S₁` | empirical calibration from decoy keys (§10) | **measured, fixed** |
| Misalignment > 1 px | 4/5 of the evidence lost (§12) | dense correlator gets to ≤0.5 px | measured, mitigated |
| Per-frame homography (camera capture) | the per-video geometry lock fails | per-frame feature registration; out of scope per §21 | not addressed |
| Low-texture / flat patches | whitened variance → 0, `σ̂` unstable | floor `σ̂`; inverse-variance handles the rest | design |
| Correlated host across patches | patches on similar content give correlated `ν`, so `M_eff` overstates independence | measured: lag-1 autocorrelation of frame-level evidence is 0.069, so frames are effectively independent; within-frame inter-patch correlation is still unmeasured | **partly measured** |
| Non-diagonal evidence covariance under misalignment | orthogonality holds in the aligned basis only | measure `Cov(c)` versus shift; add Mahalanobis if material | **open** |
| Non-Gaussian host tails | the `√(2 ln N)` null is optimistic — measured 1.73× on `S₂` | empirical calibration (§10) | **measured, fixed** |
| **Threshold vs observation count** | measured: the null grows as ≈√`M`, so `H₀ max` reaches 19.07 at 200 frames against a threshold of 17.76 calibrated at 8. Positive correlation between observations makes the true variance of the mean plateau at `σ²ρ` while the sandwich estimator keeps reporting `σ²/M` | the per-video decoy null (§10, S₄), which is immune because the decoys share the clip's `M` and its correlation. A correlation-aware standard error (frame-clustered or block bootstrap) would be the other route | **measured, fixed by S₄** |
| **Partial recovery returning a wrong ID** | some bits recovered, the rest at chance; the search over 200 000 finds a codeword fitting the survivors and the noise | S₃ split-half, then S₄ | **measured, fixed** |
| **Content-driven bias surviving split-half** | both halves see the same scene, so a content-driven direction reproduces in both — 1 of 6 decoy keys slipped through | S₄ only | **measured, fixed by S₄** |
| Collusion / averaging attack | the same `W` is in every patch of every frame, so averaging many patches recovers it | out of scope (§21 is digital attacks) but a real security property worth recording | **noted** |
| Whitener ranking flips under compression | §6's ranking is a no-attack ranking | re-measure per condition (Phase E) | **open** |
| **Selection bias at located sites** | sites are chosen by maximising the statistic they are then judged on | rank-aware order-statistic null (§10) | **measured, fixed** |
| **Rotation** | the geometry search runs scale-only by default | `geometry="scale+rotation"` exists and costs ~5x | **available, unmeasured** |

The three **open** items are measurements, not unknowns in principle. They are
scheduled in Phase C/E rather than guessed at here.

---

## 25. Phase-by-phase roadmap

Mapped onto `prompt.md` §32, with the ordering changed in one place and why.

| phase | content | status |
|---|---|---|
| **0** | Bug fixes: `uint8` wrap in embed, duplicate `FFmpegWriter`, EOF guard, patch spacing, codebook dead code | **done** |
| **B** | Codeword design (`ids.py`) — brought forward from §32 Phase 7, because it is the largest free gain and its numbers are needed to size everything else | **done** |
| **A** | This document | **done** |
| **C** | §32 Phase 1 — classical baseline: whiten → correlate → localise → evidence → aggregate → decode → test, plus §34's 36 unit tests | **done** |
| **D** | Evaluation harness: 84-attack model, resumable sweep, null calibration, perceptual metrics. First full sweep and calibration measured (§20, §10) | **done** |
| **G** | Embedding track, separate per §36: chip size measured at both matched local amplitude and matched frame PSNR (§21). **Not applied to `embed.py`** — it is a trade curve and picking a point needs a viewing decision | **measured, awaiting a decision** |
| **E** | §32 Phases 2–3 — whitener sweep *under attack* (§6's ranking is a no-attack ranking), reliability estimator | next |
| **F** | §32 Phase 4 — learned host suppression, **only if** E leaves a measured gap. Given that the binding limit is compression rather than host noise, the gap to close may be smaller than §6's 7.5 dB suggests | |

§32 Phase 5 (temporal aggregation) collapses into C — there is nothing to
synchronise (§15). §32 Phase 6 (robust localisation) collapses into C — the
dense correlator *is* the localiser (§12). §32 Phase 8 (ECC) collapses into B —
the ID set is the code (§22).

---

## 26. Final recommended architecture

**Blind**, dense-correlation, closed-set, with two optional learned components:

1. 4-phase 2× decimation to Haar-LL; whitening by local mean removal and local
   variance normalisation.
2. Full-frame FFT correlation against all 32 PRNs.
3. Localisation on the codeword-independent energy `Σ_k c_k²`; global scale and
   rotation locked once per video.
4. Per-site 32-D evidence with an off-peak-floor reliability estimate.
5. Inverse-variance, Huber-clipped pooling over patches and frames — no temporal
   synchronisation.
6. Exhaustive 200 000 × 32 GEMM against an extended-BCH [32,21,6] ID set.
7. Two-statistic acceptance test on `(S₁, S₂)`, empirically calibrated.
8. Learned host suppressor in front of step 2, and a learned reliability
   estimator at step 4, **if and only if** the classical versions leave a
   measured gap.

---

## Appendix — the 20 questions of §31

1. **Expected RMS of the sum of 32 orthogonal RMS-1 PRNs?** Exactly `√32` =
   5.657, for every codeword. Not approximately — measured to floating-point.
2. **After normalising to RMS 1, what happens to each PRN's contribution?**
   Amplitude `×1/√32`; matched-filter output per bit becomes exactly `α/√32`.
3. **Expected matched-filter output for the correct PRN?** `n·α·s_k/√32`;
   normalised, `α/√32` = 0.5303 at `α`=3.
4. **For an incorrect PRN?** Exactly 0. Not approximately.
5. **Per-bit SNR after normalisation?** `α·g_Φ/(√32·σ_Φ)`. Measured per patch:
   **−3.8 dB raw**, **+2.2 to +3.7 dB whitened** (BER 0.220 → 0.047).
6. **Total watermark SNR?** `√32 ×` the per-bit amplitude relative to the same
   noise — i.e. the presence statistic `S₂` is 32 times better off than any
   single bit, which is why presence detection is far easier than identification.
7. **Effect of spatial averaging?** `+10log₁₀(M)` dB. 27 patches/frame achievable
   → +14.3 dB.
8. **Effect of temporal averaging?** Same law. 120 frames → +20.8 dB.
9. **Does temporal aggregation recover the `1/√32` loss?** Yes, many times over.
   15.05 dB is lost; ~35 dB of aggregation is available before any coding gain.
10. **Does closed-set decoding over 200 000 help meaningfully?** Only if the set
    is designed. Random IDs: `d_min`=1, no gain. Extended BCH [32,21,6]:
    **+7.78 dB**. This is the difference between a free lever and none.
11. **Is dot-product matching the ML detector?** Yes, under `c = Aq + n` with
    `n ~ N(0, σ²I)`. With per-bit heteroscedasticity, `Σ c_i q_i/σ_i²`.
12. **When should correlations be weighted?** Always, by `1/σ̂²` per observation.
    Reliability varies by an order of magnitude across patches.
13. **How to handle imperfect orthogonality?** It is not imperfect — it is exact.
    The question becomes relevant only under misalignment; §24 schedules the
    measurement.
14. **Is individual PRN correlation more useful than combined?** Yes,
    definitively: `c` is a sufficient statistic, and the combined correlation is
    a deterministic function of it costing 200 000× more.
15. **How should NO-WATERMARK be decided?** Two calibrated statistics: the
    multiplicity-corrected best score `S₁` against `√(2 ln N)` = 4.94, and the
    codeword-independent energy `S₂` ~ χ²₃₂. Not the best-vs-second margin.
16. **Achievable minimum Hamming distance for ~200 000 32-bit codewords?** 6, via
    extended BCH [32,21,6] which supplies 2 097 152 to choose from. 8 is
    achievable only at 65 536 IDs (Reed-Muller [32,16,8]).
17. **Would designing the IDs as an ECC materially improve detection?** Yes —
    7.78 dB, at zero perceptual cost, plus a uniform per-ID error rate. It is the
    best return on effort in the whole system.
18. **Can perceptual quality stay fixed while per-bit detectability improves?**
    Yes, three independent ways: whitening (detector-side, +7.5 dB, free);
    codeword design (+7.8 dB, free); denser patch spacing (+5.6 dB, tiny
    perceptual cost). And a fourth on the embedding side — chip size — subject to
    the perceptual re-measurement of §21.
19. **Is equal energy per bit optimal?** Yes, for equally-important bits under a
    linear code with ML decoding. Unequal allocation only pays when bits have
    unequal importance, which closed-set decoding removes by construction.
20. **What is the strongest detector you would actually build?** §26.
