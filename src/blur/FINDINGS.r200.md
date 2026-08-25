# Blur watermark: measured results

Appended by `src/blur/eval_harness.py`. Every row is one
(radius, TR, cluster, condition) cell measured on one fixed source video
with a fixed CRF, both detectors read from the same frame pass.
The machine-readable copy is `findings.jsonl` beside this file.


## sweep 2026-08-25 17:56 UTC -- source=inputs/120.mp4 frames=960 crf=23 payload=0xCAFECAFE lightglue=True

| condition | R | TR | map | K | energy | mf a | mf z | mf nc | frame acc a | presence a | min |z| a | frames->32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 200 | 1 | interleaved | 1 | 22/32 | 22/32 | 22/32 | 22/32 | 0.708 | 28.96 | 8.50 | - |
| crf23 | 200 | 1 | interleaved | 1 | 22/32 | 22/32 | 22/32 | 22/32 | 0.679 | 21.84 | 8.80 | - |
| clean | 200 | 3 | contiguous | 1 | 30/32 | 31/32 | 31/32 | 31/32 | 0.798 | 23.95 | 1.82 | - |
| crf23 | 200 | 3 | contiguous | 1 | 30/32 | 30/32 | 30/32 | 31/32 | 0.763 | 17.43 | 0.59 | - |
| clean | 200 | 5 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.844 | 22.75 | 7.68 | 720 |
| crf23 | 200 | 5 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.797 | 15.50 | 5.34 | 864 |
| clean | 200 | 10 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.857 | 26.58 | 6.00 | 672 |
| crf23 | 200 | 10 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.841 | 19.55 | 3.65 | 720 |
| clean | 200 | 30 | contiguous | 1 | 29/32 | 27/32 | 29/32 | 28/32 | 0.865 | 36.35 | 0.02 | - |
| crf23 | 200 | 30 | contiguous | 1 | 29/32 | 26/32 | 28/32 | 27/32 | 0.829 | 28.91 | 0.02 | - |


## sweep 2026-08-25 17:56 UTC -- source=inputs/120.mp4 frames=960 crf=23 payload=0xCAFECAFE lightglue=True

| condition | R | TR | map | K | energy | mf a | mf z | mf nc | frame acc a | presence a | min |z| a | frames->32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blur_only | 200 | 1 | interleaved | 1 | 14/32 | 25/32 | 22/32 | 10/32 | 0.529 | 1.22 | 0.04 | - |
| moire_only | 200 | 1 | interleaved | 1 | 11/32 | 21/32 | 22/32 | 10/32 | 0.581 | 1.06 | 0.03 | - |
| mild | 200 | 1 | interleaved | 1 | 10/32 | 24/32 | 22/32 | 10/32 | 0.637 | 2.85 | 0.17 | - |
| moderate | 200 | 1 | interleaved | 1 | 13/32 | 20/32 | 16/32 | 10/32 | 0.529 | 0.73 | 0.00 | - |
| severe | 200 | 1 | interleaved | 1 | 20/32 | 20/32 | 18/32 | 10/32 | 0.511 | 0.82 | 0.10 | - |
| blur_only | 200 | 3 | contiguous | 1 | 15/32 | 29/32 | 21/32 | 9/32 | 0.554 | 1.30 | 0.16 | - |
| moire_only | 200 | 3 | contiguous | 1 | 11/32 | 24/32 | 22/32 | 9/32 | 0.627 | 1.35 | 0.04 | - |
| mild | 200 | 3 | contiguous | 1 | 11/32 | 30/32 | 22/32 | 9/32 | 0.672 | 2.54 | 0.05 | - |
| moderate | 200 | 3 | contiguous | 1 | 11/32 | 16/32 | 22/32 | 10/32 | 0.506 | 0.83 | 0.02 | - |
| severe | 200 | 3 | contiguous | 1 | 16/32 | 16/32 | 19/32 | 10/32 | 0.508 | 1.03 | 0.06 | - |
| blur_only | 200 | 5 | contiguous | 1 | 15/32 | 27/32 | 22/32 | 14/32 | 0.557 | 1.14 | 0.00 | - |
| moire_only | 200 | 5 | contiguous | 1 | 12/32 | 21/32 | 23/32 | 17/32 | 0.615 | 1.14 | 0.03 | - |
| mild | 200 | 5 | contiguous | 1 | 13/32 | 32/32 | 22/32 | 14/32 | 0.686 | 2.58 | 0.23 | 816 |
| moderate | 200 | 5 | contiguous | 1 | 15/32 | 16/32 | 21/32 | 11/32 | 0.498 | 0.63 | 0.02 | - |
| severe | 200 | 5 | contiguous | 1 | 21/32 | 18/32 | 21/32 | 11/32 | 0.500 | 0.85 | 0.01 | - |
| blur_only | 200 | 10 | contiguous | 1 | 18/32 | 21/32 | 22/32 | 15/32 | 0.523 | 1.34 | 0.00 | - |
| moire_only | 200 | 10 | contiguous | 1 | 16/32 | 21/32 | 19/32 | 16/32 | 0.648 | 1.39 | 0.10 | - |
| mild | 200 | 10 | contiguous | 1 | 17/32 | 31/32 | 17/32 | 17/32 | 0.705 | 2.75 | 0.28 | - |
| moderate | 200 | 10 | contiguous | 1 | 16/32 | 19/32 | 18/32 | 16/32 | 0.519 | 0.71 | 0.05 | - |
| severe | 200 | 10 | contiguous | 1 | 18/32 | 13/32 | 18/32 | 15/32 | 0.516 | 0.71 | 0.01 | - |
| blur_only | 200 | 30 | contiguous | 1 | 16/32 | 19/32 | 21/32 | 15/32 | 0.529 | 1.28 | 0.08 | - |
| moire_only | 200 | 30 | contiguous | 1 | 17/32 | 25/32 | 23/32 | 15/32 | 0.648 | 1.89 | 0.17 | - |
| mild | 200 | 30 | contiguous | 1 | 18/32 | 28/32 | 22/32 | 16/32 | 0.745 | 3.64 | 0.10 | - |
| moderate | 200 | 30 | contiguous | 1 | 19/32 | 12/32 | 22/32 | 15/32 | 0.494 | 0.80 | 0.02 | - |
| severe | 200 | 30 | contiguous | 1 | 20/32 | 20/32 | 14/32 | 13/32 | 0.527 | 0.92 | 0.07 | - |
