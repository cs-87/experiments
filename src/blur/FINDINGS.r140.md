# Blur watermark: measured results

Appended by `src/blur/eval_harness.py`. Every row is one
(radius, TR, cluster, condition) cell measured on one fixed source video
with a fixed CRF, both detectors read from the same frame pass.
The machine-readable copy is `findings.jsonl` beside this file.


## sweep 2026-08-25 17:56 UTC -- source=inputs/120.mp4 frames=960 crf=23 payload=0xCAFECAFE lightglue=True

| condition | R | TR | map | K | energy | mf a | mf z | mf nc | frame acc a | presence a | min |z| a | frames->32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 140 | 1 | interleaved | 1 | 32/32 | 31/32 | 29/32 | 32/32 | 0.815 | 71.53 | 3.97 | 96 |
| crf23 | 140 | 1 | interleaved | 1 | 29/32 | 27/32 | 27/32 | 30/32 | 0.771 | 48.39 | 0.54 | - |
| clean | 140 | 3 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.917 | 90.12 | 47.07 | 96 |
| crf23 | 140 | 3 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.887 | 66.02 | 22.46 | 96 |
| clean | 140 | 5 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.940 | 98.15 | 57.20 | 192 |
| crf23 | 140 | 5 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.908 | 66.09 | 38.85 | 192 |
| clean | 140 | 10 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.958 | 100.25 | 32.13 | 672 |
| crf23 | 140 | 10 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.918 | 69.54 | 18.33 | 672 |
| clean | 140 | 30 | contiguous | 1 | 29/32 | 32/32 | 32/32 | 31/32 | 0.992 | 116.21 | 1.68 | 912 |
| crf23 | 140 | 30 | contiguous | 1 | 29/32 | 32/32 | 32/32 | 30/32 | 0.967 | 84.79 | 0.44 | 912 |


## sweep 2026-08-25 17:56 UTC -- source=inputs/120.mp4 frames=960 crf=23 payload=0xCAFECAFE lightglue=True

| condition | R | TR | map | K | energy | mf a | mf z | mf nc | frame acc a | presence a | min |z| a | frames->32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blur_only | 140 | 1 | interleaved | 1 | 22/32 | 17/32 | 22/32 | 10/32 | 0.440 | 0.54 | 0.05 | - |
| moire_only | 140 | 1 | interleaved | 1 | 32/32 | 28/32 | 22/32 | 29/32 | 0.794 | 13.52 | 0.39 | - |
| mild | 140 | 1 | interleaved | 1 | 32/32 | 31/32 | 22/32 | 25/32 | 0.801 | 16.37 | 0.21 | - |
| moderate | 140 | 1 | interleaved | 1 | 22/32 | 13/32 | 22/32 | 16/32 | 0.494 | 0.64 | 0.00 | - |
| severe | 140 | 1 | interleaved | 1 | 22/32 | 15/32 | 22/32 | 11/32 | 0.521 | 0.86 | 0.00 | - |
| blur_only | 140 | 3 | contiguous | 1 | 22/32 | 16/32 | 19/32 | 10/32 | 0.411 | 0.91 | 0.00 | - |
| moire_only | 140 | 3 | contiguous | 1 | 32/32 | 32/32 | 28/32 | 26/32 | 0.875 | 19.11 | 5.49 | 144 |
| mild | 140 | 3 | contiguous | 1 | 32/32 | 32/32 | 27/32 | 21/32 | 0.873 | 19.87 | 9.06 | 96 |
| moderate | 140 | 3 | contiguous | 1 | 22/32 | 19/32 | 22/32 | 13/32 | 0.520 | 0.72 | 0.01 | - |
| severe | 140 | 3 | contiguous | 1 | 22/32 | 15/32 | 21/32 | 11/32 | 0.485 | 0.86 | 0.01 | - |
| blur_only | 140 | 5 | contiguous | 1 | 22/32 | 16/32 | 20/32 | 11/32 | 0.433 | 0.88 | 0.00 | - |
| moire_only | 140 | 5 | contiguous | 1 | 32/32 | 32/32 | 28/32 | 23/32 | 0.905 | 20.95 | 12.48 | 192 |
| mild | 140 | 5 | contiguous | 1 | 32/32 | 32/32 | 30/32 | 23/32 | 0.882 | 21.24 | 14.58 | 192 |
| moderate | 140 | 5 | contiguous | 1 | 23/32 | 15/32 | 21/32 | 13/32 | 0.538 | 0.70 | 0.00 | - |
| severe | 140 | 5 | contiguous | 1 | 22/32 | 20/32 | 21/32 | 13/32 | 0.520 | 0.82 | 0.04 | - |
| blur_only | 140 | 10 | contiguous | 1 | 17/32 | 12/32 | 19/32 | 16/32 | 0.410 | 0.98 | 0.07 | - |
| moire_only | 140 | 10 | contiguous | 1 | 31/32 | 32/32 | 29/32 | 22/32 | 0.927 | 22.71 | 12.84 | 672 |
| mild | 140 | 10 | contiguous | 1 | 31/32 | 32/32 | 29/32 | 20/32 | 0.911 | 21.30 | 11.11 | 672 |
| moderate | 140 | 10 | contiguous | 1 | 19/32 | 20/32 | 24/32 | 15/32 | 0.530 | 0.78 | 0.02 | - |
| severe | 140 | 10 | contiguous | 1 | 20/32 | 19/32 | 18/32 | 16/32 | 0.521 | 0.69 | 0.02 | - |
| blur_only | 140 | 30 | contiguous | 1 | 17/32 | 9/32 | 14/32 | 16/32 | 0.368 | 1.19 | 0.08 | - |
| moire_only | 140 | 30 | contiguous | 1 | 25/32 | 32/32 | 24/32 | 20/32 | 0.938 | 24.30 | 1.53 | 912 |
| mild | 140 | 30 | contiguous | 1 | 26/32 | 32/32 | 19/32 | 19/32 | 0.936 | 23.82 | 0.33 | 912 |
| moderate | 140 | 30 | contiguous | 1 | 19/32 | 17/32 | 13/32 | 16/32 | 0.522 | 0.85 | 0.00 | - |
| severe | 140 | 30 | contiguous | 1 | 20/32 | 18/32 | 12/32 | 15/32 | 0.523 | 1.04 | 0.06 | - |
