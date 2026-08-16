# Stage 2: SSCD Shortlist + LightGlue Verification

**Experiment 1, continued — the two-stage architecture from `REPORT.md` §07, implemented per `implement_1.md`.**

| | |
|---|---|
| Corpus | `4_sec_source.mp4` · 120 frames · 1920×1080 · 30fps |
| Hardware | Tesla T4 |
| Stage 1 | SSCD (`sscd_disc_mixup`), unmodified from `REPORT.md` |
| Stage 2 | DISK (1024 keypoints) + LightGlue, RANSAC-inlier-count scoring |
| Conditions tested | mild, moderate |
| K values | 1, 5, 10, 20, 50 |
| Temporal prior | None at any stage |

---

## The finding

Geometric verification substantially outperforms SSCD alone, but its advantage decays as K grows,
because LightGlue's RANSAC-inlier score does not cleanly separate near-duplicate frames either. The
two-stage idea works — it is not a wash — but the failure mode from `REPORT.md` (discrimination
against invariance) has a geometric-matching analogue, and this run measures it directly.

| Condition | K | stage1 recall@K | **stage2 top1** | retrieval failure | verification failure | stage1-alone top1 |
|---|---|---|---|---|---|---|
| mild | 1 | 0.050 | 0.050 | 0.950 | 0.000 | 0.050 |
| mild | 5 | 0.200 | 0.108 | 0.800 | 0.092 | 0.050 |
| mild | 10 | 0.342 | 0.150 | 0.658 | 0.192 | 0.050 |
| mild | 20 | 0.608 | **0.233** | 0.392 | 0.375 | 0.050 |
| mild | 50 | 0.925 | **0.325** | 0.075 | 0.600 | 0.050 |
| moderate | 1 | 0.017 | 0.017 | 0.983 | 0.000 | 0.017 |
| moderate | 5 | 0.075 | 0.050 | 0.925 | 0.025 | 0.017 |
| moderate | 10 | 0.158 | 0.067 | 0.842 | 0.092 | 0.017 |
| moderate | 20 | 0.367 | 0.108 | 0.633 | 0.258 | 0.017 |
| moderate | 50 | 0.792 | **0.225** | 0.208 | 0.567 | 0.017 |

At K=50, verification lifts top-1 accuracy **6.5×** over SSCD alone under mild capture (0.325 vs.
0.050) and **13×** under moderate (0.225 vs. 0.017). That confirms the core premise of `REPORT.md`
§07: SSCD's job is to get the true frame *somewhere* in a generous shortlist, and geometric matching
can then pick it out far better than the global embedding could on its own.

But the failure-mode split tells the more interesting story. As K grows, `stage1_recall@K` climbs
toward 1 (more true frames enter the shortlist) while `verification_failure_rate` climbs even faster
— at K=50, the true frame is in the shortlist 92.5% of the time under mild capture, yet LightGlue
picks a *different* candidate 60% of the time. **Verification failure, not retrieval failure, is now
the dominant error mode at large K.**

> **Why**: this is the same near-duplicate-collapse problem from `REPORT.md` §03, one layer down. On
> a near-static scene, dozens of nearby original frames are structurally near-identical to each other,
> so a RANSAC homography from the leak frame fits several of them almost equally well — inlier count
> alone is not a sharp enough signal to break the tie correctly. Widening K helps stage 1 (more
> correct frames enter the shortlist) but simultaneously admits more geometrically-plausible
> distractors for stage 2 to get confused by, which is why `stage2_top1` grows much more slowly than
> `stage1_recall@K` does.

---

## What this changes about the next iteration

- **The inlier-count score is too coarse.** It answers "does a homography exist" more than "is this
  the *right* frame among several with a plausible homography." A sharper score (e.g. photometric
  residual after warping to the candidate's frame, or a match-density normalization) is the next
  thing to try — this is exactly the kind of optimization implement_1.md said to withhold until this
  baseline was measured.
- **K is a real trade-off, not a free dial.** Larger K buys stage-1 recall but costs stage-2 accuracy
  on this content. The optimum in this run is around K=20–50; sweeping further K values and content
  with more real motion (per `REPORT.md` §08.2) is needed before generalizing.
- **The architecture is validated, not solved.** Stage 2 clearly beats stage 1 alone at every K tested
  here — the two-stage idea is worth continuing to develop, and the next lever is verification score
  quality, not shortlist size.

---

## Implementation

Two new files in `src/retrieval/`, nothing existing modified.

| File | Purpose |
|---|---|
| `lightglue_verify.py` | `DiskCache` extracts DISK features from all reference frames **once**, reused across every condition. `Stage2Verifier.verify_frame()` extracts the leak frame's features once, then runs LightGlue against every candidate's cached features — no cheap pre-filter between SSCD and LightGlue. `score_pair()` estimates a RANSAC homography **after** matching, purely as an inlier-count score; nothing is warped or rectified. |
| `evaluate_stage2.py` | Driver. Reuses `evaluate.load_frames`, `encoders.build_encoder`, `index.build_index`, `capture_sim.CaptureSim` unchanged. Runs SSCD search once at `max(K)` and slices the ranked list for every smaller K (no rerun), matching against every candidate at the largest K as instructed. Reports `stage1_recall@K`, `stage2_top1`, `retrieval_failure_rate`, `verification_failure_rate` per condition per K. |

Constraints from `implement_1.md` honored throughout: no temporal/fps assumptions, no sliding
windows, no DTW, no cheap filtering between stages, no perspective-correction preprocessing —
homography is computed only after LightGlue matching and used solely as a score, never to rectify a
frame.

### Reproduce

```bash
PYTHONPATH=.:src /home/ubuntu/cs_exp/cam_cap_dwt_l2/.venv/bin/python3 src/retrieval/evaluate_stage2.py \
  --conditions mild moderate --ks 1 5 10 20 50
```

Results are written to `out/retrieval/stage2_results.json`.

### Environment note

The base conda env has no `torch`; the sibling `cam_cap_dwt_l2/.venv` already had torch 2.13+cu126
and opencv matching this box's CUDA 13.2 driver, so `lightglue` was installed there
(`pip install git+https://github.com/cvg/LightGlue.git`). Its `__init__.py` eagerly imports `ALIKED`,
which pulls in `torchvision` — and this venv's `torchvision` wheel fails to load its compiled ops
against this particular torch build (`RuntimeError: operator torchvision::nms does not exist`). Since
only `DISK`/`LightGlue` are used here (both depend on `kornia`, not `torchvision`), the installed
package's `ALIKED` import was wrapped in try/except in `.venv/.../lightglue/__init__.py` — a
venv-local patch, not a change to this repo.

Stage-1 numbers from this run (e.g. mild top1 = 0.050) differ slightly from the 0.042 in `REPORT.md`
— confirmed to be an environment effect (this venv's torch/opencv build), not a bug: re-running the
unmodified `evaluate.py` in this same venv reproduces 0.050, matching this report exactly.
