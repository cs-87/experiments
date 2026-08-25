# Blur watermark: measured results

Appended by `src/blur/eval_harness.py`. Every row is one
(radius, TR, cluster, condition) cell measured on one fixed source video
with a fixed CRF, both detectors read from the same frame pass.
The machine-readable copy is `findings.jsonl` beside this file.


## sweep 2026-08-25 17:55 UTC -- source=inputs/120.mp4 frames=960 crf=23 payload=0xCAFECAFE lightglue=True

| condition | R | TR | map | K | energy | mf a | mf z | mf nc | frame acc a | presence a | min |z| a | frames->32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 80 | 1 | interleaved | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.974 | 234.50 | 141.48 | 48 |
| crf23 | 80 | 1 | interleaved | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.949 | 156.15 | 84.71 | 48 |
| clean | 80 | 3 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.992 | 268.16 | 202.38 | 96 |
| crf23 | 80 | 3 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.987 | 191.88 | 143.80 | 96 |
| clean | 80 | 5 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.981 | 290.46 | 173.64 | 192 |
| crf23 | 80 | 5 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.969 | 195.12 | 128.62 | 192 |
| clean | 80 | 10 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.992 | 302.10 | 145.83 | 336 |
| crf23 | 80 | 10 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.980 | 203.55 | 92.52 | 336 |
| clean | 80 | 30 | contiguous | 1 | 29/32 | 32/32 | 32/32 | 32/32 | 0.995 | 311.49 | 3.24 | 912 |
| crf23 | 80 | 30 | contiguous | 1 | 29/32 | 32/32 | 32/32 | 32/32 | 0.984 | 201.08 | 1.36 | 912 |


## sweep 2026-08-25 17:55 UTC -- source=inputs/120.mp4 frames=960 crf=23 payload=0xCAFECAFE lightglue=True

| condition | R | TR | map | K | energy | mf a | mf z | mf nc | frame acc a | presence a | min |z| a | frames->32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blur_only | 80 | 1 | interleaved | 1 | 22/32 | 32/32 | 22/32 | 10/32 | 0.824 | 2.87 | 1.71 | 288 |
| moire_only | 80 | 1 | interleaved | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.967 | 75.15 | 54.64 | 48 |
| mild | 80 | 1 | interleaved | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.952 | 98.44 | 62.94 | 48 |
| moderate | 80 | 1 | interleaved | 1 | 32/32 | 32/32 | 32/32 | 26/32 | 0.910 | 26.85 | 21.03 | 48 |
| severe | 80 | 1 | interleaved | 1 | 22/32 | 17/32 | 22/32 | 22/32 | 0.510 | 0.94 | 0.05 | - |
| blur_only | 80 | 3 | contiguous | 1 | 22/32 | 32/32 | 16/32 | 12/32 | 0.818 | 3.11 | 1.17 | 288 |
| moire_only | 80 | 3 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.983 | 79.97 | 68.23 | 96 |
| mild | 80 | 3 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.985 | 95.36 | 73.89 | 96 |
| moderate | 80 | 3 | contiguous | 1 | 32/32 | 32/32 | 30/32 | 27/32 | 0.915 | 27.38 | 20.23 | 96 |
| severe | 80 | 3 | contiguous | 1 | 22/32 | 20/32 | 22/32 | 20/32 | 0.479 | 0.71 | 0.00 | - |
| blur_only | 80 | 5 | contiguous | 1 | 26/32 | 32/32 | 17/32 | 17/32 | 0.826 | 3.13 | 1.58 | 336 |
| moire_only | 80 | 5 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.982 | 81.84 | 46.23 | 192 |
| mild | 80 | 5 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.980 | 99.99 | 55.28 | 192 |
| moderate | 80 | 5 | contiguous | 1 | 31/32 | 32/32 | 28/32 | 23/32 | 0.906 | 27.40 | 19.60 | 192 |
| severe | 80 | 5 | contiguous | 1 | 22/32 | 17/32 | 22/32 | 19/32 | 0.522 | 0.80 | 0.00 | - |
| blur_only | 80 | 10 | contiguous | 1 | 24/32 | 32/32 | 17/32 | 17/32 | 0.839 | 3.68 | 0.86 | 720 |
| moire_only | 80 | 10 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.988 | 84.04 | 36.27 | 336 |
| mild | 80 | 10 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.988 | 99.51 | 38.89 | 336 |
| moderate | 80 | 10 | contiguous | 1 | 28/32 | 32/32 | 19/32 | 17/32 | 0.917 | 29.14 | 18.41 | 336 |
| severe | 80 | 10 | contiguous | 1 | 21/32 | 18/32 | 21/32 | 16/32 | 0.513 | 0.86 | 0.01 | - |
| blur_only | 80 | 30 | contiguous | 1 | 18/32 | 27/32 | 20/32 | 16/32 | 0.814 | 3.17 | 0.00 | - |
| moire_only | 80 | 30 | contiguous | 1 | 29/32 | 32/32 | 31/32 | 29/32 | 0.989 | 87.27 | 2.39 | 912 |
| mild | 80 | 30 | contiguous | 1 | 29/32 | 32/32 | 32/32 | 28/32 | 0.991 | 100.11 | 1.98 | 912 |
| moderate | 80 | 30 | contiguous | 1 | 23/32 | 29/32 | 24/32 | 20/32 | 0.923 | 28.19 | 0.38 | - |
| severe | 80 | 30 | contiguous | 1 | 19/32 | 17/32 | 19/32 | 13/32 | 0.497 | 1.29 | 0.01 | - |
