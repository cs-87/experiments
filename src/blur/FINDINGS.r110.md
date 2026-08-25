# Blur watermark: measured results

Appended by `src/blur/eval_harness.py`. Every row is one
(radius, TR, cluster, condition) cell measured on one fixed source video
with a fixed CRF, both detectors read from the same frame pass.
The machine-readable copy is `findings.jsonl` beside this file.


## sweep 2026-08-25 17:55 UTC -- source=inputs/120.mp4 frames=960 crf=23 payload=0xCAFECAFE lightglue=True

| condition | R | TR | map | K | energy | mf a | mf z | mf nc | frame acc a | presence a | min |z| a | frames->32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 110 | 1 | interleaved | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.936 | 135.99 | 49.20 | 48 |
| crf23 | 110 | 1 | interleaved | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.881 | 85.09 | 21.08 | 144 |
| clean | 110 | 3 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.974 | 157.28 | 104.90 | 96 |
| crf23 | 110 | 3 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.962 | 106.52 | 71.52 | 96 |
| clean | 110 | 5 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.965 | 174.58 | 116.71 | 192 |
| crf23 | 110 | 5 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.939 | 117.37 | 76.95 | 192 |
| clean | 110 | 10 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.978 | 192.58 | 67.82 | 672 |
| crf23 | 110 | 10 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.963 | 127.54 | 23.09 | 672 |
| clean | 110 | 30 | contiguous | 1 | 29/32 | 32/32 | 32/32 | 32/32 | 0.993 | 199.46 | 1.30 | 912 |
| crf23 | 110 | 30 | contiguous | 1 | 29/32 | 32/32 | 31/32 | 31/32 | 0.978 | 135.43 | 0.40 | 912 |


## sweep 2026-08-25 17:55 UTC -- source=inputs/120.mp4 frames=960 crf=23 payload=0xCAFECAFE lightglue=True

| condition | R | TR | map | K | energy | mf a | mf z | mf nc | frame acc a | presence a | min |z| a | frames->32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blur_only | 110 | 1 | interleaved | 1 | 22/32 | 16/32 | 22/32 | 10/32 | 0.432 | 0.85 | 0.03 | - |
| moire_only | 110 | 1 | interleaved | 1 | 32/32 | 32/32 | 25/32 | 32/32 | 0.931 | 32.82 | 9.29 | 48 |
| mild | 110 | 1 | interleaved | 1 | 32/32 | 32/32 | 31/32 | 32/32 | 0.918 | 41.48 | 15.89 | 48 |
| moderate | 110 | 1 | interleaved | 1 | 22/32 | 32/32 | 22/32 | 17/32 | 0.783 | 3.72 | 1.51 | 432 |
| severe | 110 | 1 | interleaved | 1 | 22/32 | 16/32 | 22/32 | 16/32 | 0.494 | 0.88 | 0.03 | - |
| blur_only | 110 | 3 | contiguous | 1 | 22/32 | 19/32 | 22/32 | 10/32 | 0.469 | 1.03 | 0.08 | - |
| moire_only | 110 | 3 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.956 | 43.13 | 28.89 | 96 |
| mild | 110 | 3 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.941 | 47.45 | 29.92 | 96 |
| moderate | 110 | 3 | contiguous | 1 | 22/32 | 32/32 | 22/32 | 19/32 | 0.793 | 3.87 | 0.91 | 240 |
| severe | 110 | 3 | contiguous | 1 | 22/32 | 16/32 | 22/32 | 18/32 | 0.504 | 1.12 | 0.04 | - |
| blur_only | 110 | 5 | contiguous | 1 | 22/32 | 17/32 | 22/32 | 12/32 | 0.467 | 1.44 | 0.21 | - |
| moire_only | 110 | 5 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.962 | 44.68 | 31.39 | 192 |
| mild | 110 | 5 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 30/32 | 0.952 | 46.59 | 23.35 | 192 |
| moderate | 110 | 5 | contiguous | 1 | 22/32 | 32/32 | 22/32 | 20/32 | 0.822 | 4.25 | 1.06 | 192 |
| severe | 110 | 5 | contiguous | 1 | 22/32 | 15/32 | 22/32 | 12/32 | 0.496 | 0.84 | 0.00 | - |
| blur_only | 110 | 10 | contiguous | 1 | 19/32 | 18/32 | 18/32 | 17/32 | 0.463 | 1.10 | 0.02 | - |
| moire_only | 110 | 10 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 32/32 | 0.965 | 45.68 | 21.01 | 672 |
| mild | 110 | 10 | contiguous | 1 | 32/32 | 32/32 | 32/32 | 30/32 | 0.962 | 50.61 | 20.22 | 672 |
| moderate | 110 | 10 | contiguous | 1 | 19/32 | 32/32 | 23/32 | 16/32 | 0.818 | 4.39 | 1.76 | 336 |
| severe | 110 | 10 | contiguous | 1 | 21/32 | 19/32 | 20/32 | 15/32 | 0.500 | 0.99 | 0.03 | - |
| blur_only | 110 | 30 | contiguous | 1 | 17/32 | 8/32 | 18/32 | 15/32 | 0.378 | 2.43 | 0.03 | - |
| moire_only | 110 | 30 | contiguous | 1 | 28/32 | 32/32 | 30/32 | 28/32 | 0.990 | 49.11 | 7.36 | 912 |
| mild | 110 | 30 | contiguous | 1 | 28/32 | 32/32 | 28/32 | 26/32 | 0.981 | 52.78 | 2.22 | 912 |
| moderate | 110 | 30 | contiguous | 1 | 20/32 | 30/32 | 18/32 | 16/32 | 0.792 | 4.36 | 0.19 | - |
| severe | 110 | 30 | contiguous | 1 | 20/32 | 13/32 | 16/32 | 12/32 | 0.496 | 0.77 | 0.00 | - |
