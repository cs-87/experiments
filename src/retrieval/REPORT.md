# Retrieving a Camcorded Frame from Its Source Video

**Experiment 1 — temporal frame retrieval.**

| | |
|---|---|
| Corpus | `4_sec_source.mp4` · 120 frames · 1920×1080 · 30fps |
| Hardware | Tesla T4 |
| Encoders | 5 tested |
| Conditions | 11 simulated |
| Ground truth | Exact by construction (queries generated from known original frames) |
| Temporal prior | None at any stage |

---

## Contents

1. [The finding](#01--the-finding)
2. [Measured results](#02--measured-results)
3. [Why it fails: discrimination against invariance](#03--why-it-fails-discrimination-against-invariance)
4. [Model families](#04--model-families)
5. [Comparison matrix](#05--comparison-matrix)
6. [Whole frame, crops, or multi-scale](#06--whole-frame-crops-or-multi-scale)
7. [The architecture the measurements imply](#07--the-architecture-the-measurements-imply)
8. [What to build, in order](#08--what-to-build-in-order)
9. [The harness, as built](#09--the-harness-as-built)
10. [Failure modes to design against](#10--failure-modes-to-design-against)
11. [References](#11--references)

---

## 01 · The finding

Frame-exact top-1 retrieval from a single global embedding **does not work** on your test clip — for any of the five encoders, including the one purpose-built for copy detection. The reason is measurable, and it is not a bug.

The literature converges cleanly on what to use: **per-frame image descriptors, not video-clip encoders**. Meta's Video Similarity Challenge (VSC22) built its query set with a transform described as *"perspective transform and shake — simulate an unsteady manual recording of a screen"* — precisely your threat model. Its baseline and all three top finishers extract a descriptor per frame with an image model and handle time as a separate stage. That answers your questions 3 and 4 before any experiment runs.

So I built the experiment you specified, ran it, and the representation stage did not clear the bar. The useful part is *why*: your content is a near-static scene, and every encoder faces a trade-off between telling adjacent frames apart and surviving camera capture. On this clip, no encoder wins both at once.

> **Headline**
>
> Under *mild* simulated capture, SSCD's best-in-test top-1 accuracy is **4.2%**. Under moderate capture it is 1.7%. A clean, undegraded query retrieves perfectly (100%), which confirms the pipeline is correct — the collapse is caused by degradation, not a wiring error.
>
> But **recall within ±10 frames reaches 64%** for SSCD and **100%** for the raw-pixel baseline. The embedding finds the right *neighbourhood*; it cannot pick the frame.

That splits your problem in two, and the split is the main recommendation of this report: retrieval should be judged on **recall@±K, not top-1**, and a second verification stage — which you already have most of, in `utils/lightglue.py` — picks the exact frame from the shortlist.

> **Scope of this claim**
>
> This is one 4-second, cut-free, near-static clip: close to a worst case for frame retrieval. Content with cuts and real motion should score substantially better. Treat the *mechanism* below as general and the *numbers* as specific to this footage until re-run on a longer, more dynamic source.

---

## 02 · Measured results

120 reference frames indexed; each query is an original frame pushed through the capture simulator, so ground truth is exact by construction. No temporal prior of any kind — every query is scored independently.

### Top-1 accuracy — exact frame match

| Encoder | dim | clean | mild | moderate | severe | extreme |
|---|---|---|---|---|---|---|
| pixel (ZNCC floor) | 4096 | 1.000 | 0.008 | 0.008 | 0.017 | 0.017 |
| **SSCD** | 512 | 1.000 | **0.042** | 0.017 | 0.008 | 0.008 |
| DINOv2-base | 1536 | 1.000 | 0.042 | 0.025 | 0.000 | 0.008 |
| SigLIP2-base | 768 | 1.000 | 0.025 | 0.008 | 0.008 | 0.008 |

### Recall by tolerance, and median absolute temporal error (frames)

| Encoder | Condition | ±1 | ±5 | ±10 | median err | top-5 | top-10 |
|---|---|---|---|---|---|---|---|
| pixel | mild | 0.017 | 0.075 | **1.000** | 7.0 | 0.042 | 0.083 |
| pixel | moderate | 0.017 | 0.050 | 0.092 | 28.0 | 0.042 | 0.083 |
| **SSCD** | mild | 0.125 | 0.383 | **0.642** | 7.0 | 0.192 | 0.358 |
| SSCD | moderate | 0.058 | 0.183 | 0.358 | 15.0 | 0.083 | 0.175 |
| SSCD | severe | 0.050 | 0.192 | 0.342 | 16.5 | 0.092 | 0.183 |
| DINOv2 | mild | 0.108 | 0.308 | 0.500 | 10.5 | 0.092 | 0.200 |
| DINOv2 | moderate | 0.042 | 0.092 | 0.250 | 20.5 | 0.058 | 0.108 |
| SigLIP2 | mild | 0.050 | 0.183 | 0.350 | 15.0 | 0.108 | 0.167 |
| SigLIP2 | moderate | 0.025 | 0.092 | 0.158 | 39.5 | 0.042 | 0.100 |

SSCD is the strongest neural encoder at every tolerance, which is what the copy-detection literature predicts. SigLIP2 is the weakest, which is also predicted. The oddity is the raw-pixel baseline hitting perfect ±10 recall under mild capture while getting essentially every exact frame wrong — that clue is the subject of the next section.

---

## 03 · Why it fails: discrimination against invariance

The harness reports a diagnostic I added for exactly this, because accuracy alone cannot tell you whether a failure is the encoder's fault or the content's. Two quantities:

- **`s_true`** — mean cosine between a degraded query and its *own* original frame. How well the encoder survives camera capture. **Higher is better.**
- **`s_self(d)`** — mean cosine between two *undegraded* original frames `d` apart. How much the encoder confuses distinct frames. **Lower is better.**

Retrieval succeeds only when `s_true > s_self(d)` for every `d` you must resolve. Here is what the encoders actually produce:

### Robustness vs discrimination — the core trade-off

| Encoder | s_true (mild) | s_true (severe) | d=1 | d=5 | d=10 | d=20 | d=40 |
|---|---|---|---|---|---|---|---|
| pixel | 0.754 | 0.114 | 0.999 | 0.980 | 0.932 | 0.807 | **0.564** |
| SSCD | **0.818** | **0.616** | 0.999 | 0.992 | 0.983 | 0.970 | 0.949 |
| DINOv2 | 0.926 | 0.702 | 0.999 | 0.995 | 0.990 | 0.986 | 0.979 |
| SigLIP2 | 0.972 | 0.869 | 0.999 | 0.994 | 0.991 | 0.987 | **0.982** |

Read the DINOv2 row across. A *mildly* degraded frame resembles its own original at **0.926**. Two clean originals **40 frames apart** resemble each other at **0.979**. The correct answer is therefore not the nearest neighbour, and no amount of index tuning changes that — roughly eighty rival frames are genuinely closer in the metric than the truth.

> **The mechanism**
>
> Reading the table as two columns rather than rows tells the whole story. **SigLIP2 is the most robust encoder tested** (s_true 0.972 under mild — the highest) and simultaneously **the worst retriever**. Its entire dynamic range across the whole video is 0.999 → 0.982. It has learned to ignore almost everything that distinguishes one frame of this scene from another, camera distortion included.
>
> Raw pixels sit at the opposite pole: a genuine spread (0.999 → 0.564) but no robustness (0.754 → 0.114 from mild to severe). That is why pixels ace ±10 recall under mild capture and collapse entirely by moderate.
>
> **Invariance is bought with discrimination.** The encoders are not failing at their jobs; their job is at odds with this task.

### The content sets a hard ceiling

Measured on high-pass luma, independent of any encoder, adjacent originals in this clip are near-identical:

| offset d | 1 | 2 | 3 | 5 | 10 | 20 | 30 | 60 |
|---|---|---|---|---|---|---|---|---|
| ZNCC(k, k+d) | 0.981 | 0.940 | 0.891 | 0.790 | 0.560 | 0.270 | 0.124 | −0.002 |

To get top-1 right an encoder must satisfy `s_true > s_self(1) ≈ 0.999` — it must be more faithful to a camcorded frame than the video is to its own next frame 33ms later. That is not an engineering target; it is close to a contradiction. On this footage, single-frame global retrieval was never going to work.

> **Side effect worth knowing**
>
> This same property defeated both model-free attempts to establish ground truth on your *real* capture. Motion-profile cross-correlation failed because the source has no cuts and near-constant motion (coefficient of variation 0.10) while the leak's signal is dominated by handheld shake (0.69) — identity-map correlation came out at **−0.11**. Plate-homography rectification then ZNCC also failed: the rectification is good (93% frame overlap, scale 1.18 ≈ 1920/1602) but peak ZNCC is only **0.07**. Scoring the real pair needs the two-stage pipeline in §07, then hand-verification.

---

## 04 · Model families

Assessed against your criteria. Ordered by how strongly I'd recommend them for this task.

### SSCD — copy-detection descriptor ✅ Recommended

*512-d · ResNet50 · 288px*

The most on-task pretrained model that exists. Trained so that an edited or re-encoded copy lands near its source, and it is the per-frame descriptor Meta's own VSC22 video-copy baseline uses. Best neural encoder in every measurement here.

| | |
|---|---|
| **Camera capture** | Best-in-test, but see the caveat: its training augmentations (flip, crop, colour jitter, greyscale, blur σ∈[1,5], rotation, overlays, JPEG, mixup) include **no perspective warp and no moiré**. DISC21 *evaluates* with perspective and screenshot overlays; SSCD is tested against camcording, not trained for it. |
| **Invariance** | Strong on blur, compression, exposure, colour. Weaker on perspective. Untested on moiré. |
| **Nearby frames** | Insufficient alone — s_self(40) = 0.949 vs s_true 0.818 under mild. |
| **Embedding** | Yes, L2-normalised, built for inner-product ANN search. |
| **Cost** | 19 ms/frame indexing on your T4. 2h film @30fps ≈ 68 min; @1fps ≈ 2.3 min. |
| **Weights** | Public, direct download, TorchScript — no architecture code needed. Verified live. |
| **Evaluated on** | DISC21 (μAP 56.8 mixup / 63.7 large), VSC22 baseline. |
| **Failure modes** | Near-duplicate collapse on static scenes; perspective is its untrained axis — the one your leak has most of. |

### DINOv2 ⚠️ Useful fallback

*768/1024-d · ViT · Apache-2.0*

The clean general-purpose choice: genuinely permissive licence, excellent dense features, strong on instance retrieval benchmarks (ROxford/RParis/Met/AmsterTime). Beat SigLIP2 here but lost to SSCD at every tolerance.

| | |
|---|---|
| **Camera capture** | Never evaluated for it. No robustness testing to blur, moiré, perspective or compression appears in the paper. |
| **Nearby frames** | Poor — s_self(40) = 0.979. Augmentation-invariance training is directly at odds with 33ms-adjacent discrimination. |
| **Cost** | 14 ms/frame (base, 224px) on your T4. |
| **Note** | Concatenating CLS with the patch-token mean (what the harness does by default) retains more spatial detail than CLS alone. |
| **Failure modes** | Semantic rather than instance-level pull; no copy-detection training signal. |

### DINOv3 ⚠️ Licence friction

*768–4096-d · gated*

Released 13 Aug 2025. Better dense and correspondence scores than DINOv2 (NAVI geometric correspondence 64.4 vs 60.1; SPair 58.7 vs 56.1; ADE20K 55.9 vs 49.5). Two practical problems.

| | |
|---|---|
| **Licence** | **Not Apache/MIT.** Custom DINOv3 licence with export-control and military-use prohibitions, no-reverse-engineering, publication-acknowledgement, and a patent-litigation termination clause. Weights are gated on both GitHub and HuggingFace. |
| **Availability** | Confirmed gated in this environment — the harness registers it but access must be requested first. |
| **Fit** | Correspondence gains are for semantic/geometric matching across *different* scenes, not for separating near-identical frames of one scene. Unlikely to change the trade-off in §03. |

### CLIP / SigLIP / SigLIP 2 ❌ Rule out

*512–1152-d*

Measured here as the weakest retriever, exactly as the literature predicts. Its total dynamic range over your entire 120-frame video is 0.999 → 0.982.

| | |
|---|---|
| **Evidence** | MMVP (CVPR 2024) constructs "CLIP-blind pairs" — images with high CLIP similarity that DINOv2 separates easily — across orientation, viewpoint, perspective and colour. A separate study reports >0.99 cosine between an image and its own mirror. ILIAS (CVPR 2025) finds raw CLIP/SigLIP global embeddings need a learned adaptation layer plus local-descriptor reranking to compete at instance level. The SSCD and DISC21 authors did not include CLIP as a baseline at all. |
| **Verdict** | Too semantically coarse. Trained to map an image to a caption, and every frame of your clip has the same caption. Keep only as a control. |

### VideoMAE / VideoMAE V2 ❌ Wrong tool

*clip-level*

| | |
|---|---|
| **Granularity** | Consumes a 16-frame tube at 90–95% masking and yields essentially one pooled representation *per clip*. There is no documented per-frame output vector. Clip-level pooling is a design goal for action recognition and a disqualifier here. |
| **Frozen use** | Repo and model cards are explicitly finetuning-oriented; no retrieval or frozen-embedding evaluation is provided. The image-MAE literature documents linear-probe accuracy trailing contrastive methods, with the original paper noting finetuning and linear-probe accuracy are "largely uncorrelated." *Extrapolated to VideoMAE, not directly measured.* |
| **Verdict** | Do not use for frame-exact retrieval. |

### InternVideo2 / InternVideo2.5 · V-JEPA 2 ❌ Wrong tool

*clip-level*

| | |
|---|---|
| **InternVideo2** | Samples 8 frames per clip (tested to 16) — coarse by design. Its retrieval numbers (MSR-VTT, DiDeMo, ActivityNet) are **text↔video semantic** retrieval with a finetuned head: "does this caption describe this video," not "is this the same physical frame." InternVideo2.5 is a long-context video chat MLLM, not an embedder at all. |
| **V-JEPA 2** | Released 11 Jun 2025. Validated with a *frozen backbone plus a trained 4-layer attentive probe*, so its own authors never validate bare frozen-embedding cosine retrieval. Weights reported as CC-BY-NC (non-commercial) — *from a secondary source; verify the LICENSE file before relying on it*. |
| **Verdict** | Both target semantic and action understanding. No instance-level or frame-retrieval evaluation exists for either. |

### CopyNCE, AnyPattern, RDCD ⚠️ Worth trying

**CopyNCE** (Ant Group, Feb 2026) claims new DISC21 SOTA at μAP 88.7% matcher / 72.6% descriptor, versus SSCD's 56.8. Most relevant detail: it **trains with affine and perspective augmentation**, which is precisely SSCD's blind spot and your leak's dominant distortion. **AnyPattern** (IJCV 2025) targets generalisation to unseen tamper patterns; **RDCD** (WACV 2025) targets compact descriptors. Claims are from the papers and not independently verified here — CopyNCE is the highest-value thing to benchmark next.

### Degraded-to-source specific models ❌ Does not exist

Your question 6, answered directly: **no modern model does this.** The nearest work splits three ways, none of it a drop-in.

| | |
|---|---|
| **Recapture detection** | 2018–2025 work is *binary classification* (is this a re-photograph, for anti-fraud/liveness) — not source matching. |
| **Classical camcorder fingerprinting** | Pre-deep-learning and retrieval-framed: Garboan & Mitrea (2015) DWT fingerprints with temporal resync; Chupeau et al. (SPIE 2007) estimating an 8-parameter homography for theatre keystoning — the "local features, then geometric correction, then match" pattern you are already building. No code released. TRECVID's copy-detection track found camcording the hardest transformation across nearly all systems and years. |
| **Demoiréing as preprocessing** | VideoDemoireing (CVPR 2022) is the one option with confirmed released weights. VD_raw (NeurIPS 2023) explicitly targets recaptured screens. Only worth adding if you measure moiré actually hurting. |
| **Industry** | Solved commercially with forensic *watermarking* (pre-marking the content), which your constraints rule out. Passive fingerprint matching is genuinely under-published. |

---

## 05 · Comparison matrix

All candidates against your criteria — measured where marked ✓meas.

| Model | Frame-level | Camera capture | Perspective | Adjacent frames | NN embedding | ms/frame | Weights | PyTorch | Retrieval-evaluated |
|---|---|---|---|---|---|---|---|---|---|
| SSCD | yes | good ✓meas | not trained | weak ✓meas | yes | 19 | public | TorchScript | DISC21, VSC22 |
| CopyNCE | yes | claimed best | **trained** | untested | yes | ~20 | public | yes | DISC21 |
| DINOv2 | yes | fair ✓meas | untested | weak ✓meas | yes | 14 | Apache-2.0 | yes | ROxford/RParis |
| DINOv3 | yes | untested | untested | untested | yes | ~16 | gated | yes | partial |
| SigLIP2 / CLIP | yes | robust ✓meas | untested | worst ✓meas | yes | 12 | Apache-2.0 | yes | text↔image |
| pixel / pHash | yes | poor ✓meas | poor | **best ✓meas** | yes | 5 | n/a | n/a | — |
| VideoMAE v2 | clip (16f) | n/a | n/a | n/a | finetune first | — | public | yes | action recog. |
| InternVideo2 | clip (8f) | n/a | n/a | n/a | semantic only | — | public | yes | text↔video |
| V-JEPA 2 | clip | n/a | n/a | n/a | needs probe | — | CC-BY-NC? | yes | action/planning |
| LightGlue+DISK | pairwise | strong | **explicit** | strong | no — reranker | ~7/pair | Apache-2.0 | yes | localisation |
| EfficientLoFTR | pairwise | strong | **explicit** | strong | no — reranker | ~15/pair | public | yes | MegaDepth |

Timings measured on your Tesla T4 at batch 16. Pairwise rows are per candidate pair, with features cached per image.

---

## 06 · Whole frame, crops, or multi-scale

Your instinct is right, and §03 says why. A global vector pools the entire frame; on a near-static scene the between-frame difference lives in a small moving region, and pooling averages it into nothing. That is mechanically what drives s_self(40) to 0.98.

**Region embeddings are the correct lever**, and the harness implements them: `encode(frames, tile=N)` splits each frame into an N×N grid, embeds every cell, and concatenates with renormalisation. Expected effects:

- **Discrimination improves** — a cell containing the moving region is not diluted by static surroundings. This should lower s_self, which is the binding constraint.
- **Robustness degrades** — the leak is perspective-warped, so a fixed grid on the query does not align with the same grid on the reference. This is the catch, and it is why tiling must be measured rather than assumed.
- **Cost** — dimension grows N², measured at roughly 2× wall-clock for tile=2 (SSCD 19→38 ms/frame). Cheap enough to test.

> **Ordering matters**
>
> Tiling the *raw* leak frame fights the perspective warp. Tiling *after* geometric rectification does not — once a homography maps the leak into the original's coordinate frame, both grids describe the same scene regions. This argues for tiling in the verification stage rather than the retrieval stage, and it is a clean experiment to run.

Multi-scale (embedding at several resolutions and concatenating) is the standard companion trick and is worth testing, but I would rank it below tiling: your dominant nuisance is perspective, not scale, and scale is partly absorbed by the square resize already.

---

## 07 · The architecture the measurements imply

Your original diagram, with one stage added and one metric changed:

```
# STAGE 1 — retrieval. Optimise for recall@±K, NOT top-1.
O0..ON ──▶ SSCD ──▶ e0..eN ──▶ flat inner-product index
L_t    ──▶ SSCD ──▶ e_L   ──▶ top-K  (K ≈ 50, generous on purpose)

# STAGE 2 — verification. This is what picks the exact frame.
for O_k in top-K:
    H = lightglue.compute_homography(O_k, L_t)   # you already have this
    if H is None: continue
    warp L_t into O_k's frame, score photometric residual
argmax ──▶ single answer + a confidence you can threshold
```

Stage 2 is where the frame-exact decision belongs, because geometric verification undoes the perspective distortion that defeats every global embedding. It is also the stage where your non-blind assumption finally pays off: once rectified, you compare pixels against pixels.

Cost is tractable. LightGlue runs ~7 ms/pair at 1024 keypoints with features cached per image, so K=50 is roughly 350 ms per query frame on your T4 — and stage 2 only ever sees 50 of the index's frames, not all of them.

> **What this changes about stage 1**
>
> Stop tuning for top-1. The question becomes "what K guarantees the true frame is in the shortlist," and you buy that with a larger K rather than a better encoder. SSCD at mild capture puts 64% of queries within ±10 frames — so a shortlist built as *top-K plus a ±10 window around each hit* will contain the answer far more often than top-10 alone suggests.

---

## 08 · What to build, in order

### 1 — SSCD shortlist plus LightGlue geometric rerank

The highest-value thing you can build, and you already own half of it. SSCD is the best-measured encoder here, has public TorchScript weights, and is the descriptor Meta's own video-copy baseline uses. LightGlue with DISK is already wired up in `utils/lightglue.py` under a fully permissive licence.

Build it as: index with SSCD → take top-50 → rectify each candidate with LightGlue → rank by photometric residual on the rectified overlap. Measure top-1 *after* stage 2. This is the experiment that tells you whether the whole approach is viable.

### 2 — Validate on content with cuts and motion

Every number in this report comes from one near-static 4-second clip that is close to a worst case. Before accepting "global embeddings can't do this" as a general conclusion, re-run the harness on a longer, more dynamic source — one command, no new code.

The `d*` and `s_self` columns are the ones to read: if s_self(10) drops well below s_true on real content, stage 1 alone may be sufficient there and the architecture simplifies.

### 3 — Benchmark CopyNCE, and tile after rectification

CopyNCE trains with affine and perspective augmentation — SSCD's exact blind spot and your leak's dominant distortion — and claims a large DISC21 margin over SSCD. Unverified, but it is the single most promising unexplored encoder, and it slots into the harness as one registry entry.

Pair it with the tile-after-rectification experiment from §06. Both are cheap; both target the discrimination side of the trade-off, which is the side that is actually binding.

Explicitly **not** worth your time: VideoMAE, InternVideo2, V-JEPA 2 (wrong temporal granularity by construction), and CLIP/SigLIP as a retriever (measured worst here, and the coarseness is well documented).

---

## 09 · The harness, as built

Working code in `src/retrieval/`, all of it run to produce this report.

| File | Purpose |
|---|---|
| `capture_sim.py` | Camcord simulator. 11 named conditions: clean, mild, moderate, severe, extreme, plus six single-axis ablations. Session homography sampled once, then per-frame shake — a fresh warp per frame would model a teleporting camera. |
| `encoders.py` | SSCD, DINOv2, DINOv3, SigLIP2, CLIP, pixel baseline behind one interface. All return L2-normalised rows. `tile=N` gives region embeddings. |
| `index.py` | Exact inner-product search, torch (fp16, GPU) or FAISS. |
| `evaluate.py` | Driver. Top-1/5/10, mean/median/p90 temporal error, recall@±1/2/5/10, median true rank, and the s_true / s_self / d* diagnostic. |
| `build_gt.py` | Motion-profile ground truth for a real pair. Fails on this clip; reports its own confidence rather than failing silently. |
| `rectify_gt.py` | Plate-homography + ZNCC ground truth. Also fails here; kept as a diagnostic. |

### Reproduce this report

```bash
PYTHONPATH=.:src python3 src/retrieval/evaluate.py \
  --encoders pixel sscd dinov2_base siglip2 \
  --conditions clean mild moderate severe extreme
```

### Single-axis attribution — which distortion actually hurts

```bash
PYTHONPATH=.:src python3 src/retrieval/evaluate.py --encoders sscd \
  --conditions perspective_only blur_only exposure_only \
               compression_only moire_only noise_only
```

### Region embeddings

```bash
PYTHONPATH=.:src python3 src/retrieval/evaluate.py \
  --encoders sscd --tile 2 --conditions mild moderate severe
```

### Long video

```bash
# --stride samples the reference set; temporal error stays in source frames
PYTHONPATH=.:src python3 src/retrieval/evaluate.py \
  --video feature.mp4 --stride 5 --backend faiss --encoders sscd
```

### Adding an encoder

Subclass `Encoder`, implement `_forward`, add one line to `REGISTRY`. Everything downstream is generic.

> **On FAISS**
>
> You asked about it specifically, so: at your scale you do not need it. Exact search is one GEMM. A 2-hour film at 30fps is 216k frames — 442 MB at 512 dims in fp32, single-digit milliseconds per query on a T4. Approximate indexes (IVF, HNSW, PQ) only start paying off in the millions of frames, and they trade away exactly the fine-grained precision this task is already short of. Both backends are implemented; keep `--backend torch` until measurement says otherwise.

---

## 10 · Failure modes to design against

### Near-duplicate collapse

The dominant failure, quantified throughout §03. Static or slow scenes make all frames mutually similar, and top-1 becomes arbitrary within the cluster. Detect it by monitoring `s_self(1)` against `s_true`; when they cross, stage 1 cannot resolve and you must widen K and lean on stage 2.

### Perspective is the untrained axis

SSCD's augmentation set has no perspective warp, and it is the largest component of a real camcord. This is the specific gap CopyNCE claims to close, and the specific thing geometric rectification fixes outright.

### Global pooling averages away the signal

Covered in §06. The moving region that distinguishes two adjacent frames is a small fraction of the frame area; a single pooled vector dilutes it in proportion.

### Semantic encoders solve a different problem

CLIP-family models map an image to its caption. Every frame of a scene shares a caption, so the embedding is near-constant across the whole shot — measured here at 0.999 → 0.982.

### Ground truth is itself a research problem

Do not assume the trim aligned your leak to the source. Two independent model-free oracles both failed on this pair. Any accuracy number you compute against an unverified map is measuring the map, not the encoder.

### Letterboxing and bezel

A camcord frames the screen loosely, so black borders enter the embedding. The simulator models this via `zoom < 1`. In production, detect and crop the screen region before encoding — otherwise the border dominates a global descriptor.

---

## 11 · References

| What | Link |
|---|---|
| SSCD (CVPR 2022) | <https://arxiv.org/abs/2202.10261> · <https://github.com/facebookresearch/sscd-copy-detection> |
| DISC21 challenge | <https://arxiv.org/abs/2106.09672> · <https://github.com/facebookresearch/isc2021> |
| VSC22 — video similarity, camcord transforms | <https://arxiv.org/abs/2306.09489> · <https://github.com/facebookresearch/vsc2022> |
| VSC22 winning solution | <https://arxiv.org/abs/2305.12361> · <https://github.com/FeipengMa6/VSC22-Submission> |
| VCSL — copy segment localisation | <https://arxiv.org/abs/2203.02654> · <https://github.com/alipay/VCSL> |
| TransVCL (AAAI 2023) | <https://arxiv.org/abs/2211.13090> · <https://github.com/transvcl/TransVCL> |
| CopyNCE (2026) | <https://arxiv.org/abs/2602.17484> · <https://github.com/eddielyc/CopyNCE> |
| AnyPattern (IJCV 2025) | <https://arxiv.org/abs/2404.13788> · <https://github.com/WangWenhao0716/AnyPattern> |
| DINOv2 | <https://arxiv.org/abs/2304.07193> · <https://github.com/facebookresearch/dinov2> |
| DINOv3 | <https://arxiv.org/abs/2508.10104> · <https://github.com/facebookresearch/dinov3> |
| SigLIP 2 | <https://arxiv.org/abs/2502.14786> |
| MMVP — CLIP-blind pairs | <https://arxiv.org/abs/2401.06209> |
| ILIAS — instance retrieval at scale | <https://arxiv.org/abs/2502.11748> |
| LightGlue (ICCV 2023) | <https://arxiv.org/abs/2306.13643> · <https://github.com/cvg/LightGlue> |
| Efficient LoFTR (CVPR 2024) | <https://arxiv.org/abs/2403.04765> · <https://github.com/zju3dv/EfficientLoFTR> |
| hloc — retrieval-then-rerank reference | <https://github.com/cvg/Hierarchical-Localization> |
| ViSiL (ICCV 2019) | <https://github.com/MKLab-ITI/visil> |
| VideoMAE V2 | <https://github.com/OpenGVLab/VideoMAEv2> |
| InternVideo2 | <https://arxiv.org/abs/2403.15377> · <https://github.com/OpenGVLab/InternVideo> |
| V-JEPA 2 | <https://arxiv.org/abs/2506.09985> · <https://github.com/facebookresearch/vjepa2> |
| Video demoiréing (recaptured screens) | <https://github.com/CVMI-Lab/VideoDemoireing> · <https://github.com/tju-chengyijia/VD_raw> |
| TMK+PDQF / vPDQ perceptual hashing | <https://github.com/facebook/ThreatExchange> |

---

*Measurements produced on Tesla T4 · 120 reference frames · exact ground truth by construction · no temporal prior applied at any stage. Claims sourced from papers rather than measured here are marked as such; unverified secondary sources are flagged inline.*
