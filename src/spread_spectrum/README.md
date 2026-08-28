# Spread-spectrum watermark: embedder, blind detector, evaluation

32 bits carried by 32 exactly-orthogonal PRNs, summed with the payload's signs,
RMS-normalised, and added to the DWT LL of every SIFT patch of every frame.

Start with **[REPORT.md](REPORT.md)** — what was built, what was measured, what is
left. **[DESIGN.md](DESIGN.md)** is the full 26-section technical document `prompt.md`
asks for; **[FINDINGS.md](FINDINGS.md)** is the measured attack table. This file is just
the map of the code.

## Layout

```
codebook.py     the 32 PRNs: Hadamard rows x a fixed random +/-1 mask
prn.py          payload -> embedded pattern, and the bit convention both sides import
embed.py        the embedder (unchanged in design; see DESIGN.md section 2 for the bugs fixed)
ids.py          the valid-ID set, constructed as an error-correcting code

detect/
  whiten.py     host suppression -- the largest single lever
  correlate.py  4-phase LL decimation + dense FFT matched filtering
  localise.py   peaks of the codeword-independent energy; global scale recovery
  evidence.py   per-site 32-D evidence, reliability, and the selection-null correction
  aggregate.py  inverse-variance, Huber-clipped pooling over patches and frames
  decode.py     200,000 x 32 GEMM, and the NO-WATERMARK test
  pipeline.py   end-to-end; also the CLI

eval/
  attacks.py    84 attacks: compression, resolution, spatial, photometric, temporal
  sweep.py      resumable per-cell JSONL + FINDINGS.md
  calibrate.py  empirical NO-WATERMARK thresholds, from decoy PRN keys
  perceptual.py PSNR / SSIM / LPIPS / flicker, and the chip-size comparison

tests/          36 tests; `python -m src.spread_spectrum.tests.run`
FINDINGS.md     generated from findings.jsonl -- edit the JSONL, not the table
```

## Running it

Everything runs from the repository root.

```bash
# embed a valid ID
python -m src.spread_spectrum.embed --input inputs/30.mp4 --output out/wm.mp4 \
    --watermark 0x15771A93

# detect it, blind, with the acquisition curve
python -m src.spread_spectrum.detect.pipeline out/wm.mp4 --expect 0x15771A93 --curve

# ... and with global scale recovery, needed for any resolution change
python -m src.spread_spectrum.detect.pipeline out/wm.mp4 --geometry scale

# the valid-ID set: construction, distances, union bound
python -m src.spread_spectrum.ids

# attack sweep (resumable; --resume skips completed cells)
python -m src.spread_spectrum.eval.sweep --attacks clean,h264_crf23,scale_540p --resume

# NO-WATERMARK thresholds from the observed null
python -m src.spread_spectrum.eval.calibrate --unmarked inputs/30.mp4 --n-seeds 24

# perceptual cost, and the chip-size comparison
python -m src.spread_spectrum.eval.perceptual --configs 128:1:3,256:2:6

python -m src.spread_spectrum.tests.run
```

## Three things to know before changing anything

**The PRNs are exactly orthogonal.** `P_i = H_i ⊙ R` with `R ∈ {−1,+1}`, and
elementwise multiplication by a ±1 vector preserves Hadamard inner products, so the
Gram matrix is exactly `4096·I`. Zero inter-PRN interference, `RMS(W)` exactly `√32`
for every codeword, and the 32 per-PRN correlations are a *sufficient statistic* for
the payload. Several of the design's simplifications depend on this; the unit tests
assert it as equality, not tolerance.

**Anything that selects sites biases the statistic it selects on.** Both the patch
localiser and the scale search maximise the evidence the decision is later made on.
Uncorrected, an unmarked clip reads `S₂` = 83 against a χ²₃₂ null of 32, and at one
point the scale search manufactured an outright false positive. Both are corrected
against order statistics (`evidence.selection_null`, `GeometrySearch`'s anchoring) and
both corrections are load-bearing.

**There is no temporal code.** The same payload is in every patch of every frame, so
frame dropping, duplication, reordering and rate conversion cost observations and
nothing else. No aligner, no sync word, no frame-to-bit map — and no reason to add one.

## Known limits

- **H.264 above CRF ~23 is where it stops.** The carrier is white in the LL domain, so
  half its energy sits above the band compression preserves. This is the one
  first-order weakness and DESIGN.md section 21 quantifies the fix (chip size), which
  is an embedding change and deliberately kept separate.
- **Rotation is not searched by default.** `--geometry scale+rotation` exists and costs
  about 5x.
- **Two distinct source clips exist on this machine.** `inputs/{15,30,60,120}.mp4` are
  byte-identical content at different lengths — verified, not assumed. Any
  generalisation claim needs more footage.
