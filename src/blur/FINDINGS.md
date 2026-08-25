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


## visibility 2026-08-25 21:31 UTC -- source=inputs/120.mp4 frames=960 crf=23

| radius | tr | map | cluster_k | patch_psnr_db | patch_ssim | frame_psnr_db | mark_rms | flicker_rms_std | flicker_ratio | flicker_luma_std |
|---|---|---|---|---|---|---|---|---|---|---|
| 80 | 1 | interleaved | 1 | 33.7 | 0.8863 | 38.37 | 6.772 | 4.404 | 0.65 | 0.213 |
| 80 | 3 | contiguous | 1 | 33.79 | 0.8867 | 38.39 | 6.696 | 4.532 | 0.677 | 0.241 |
| 80 | 30 | contiguous | 1 | 34.39 | 0.8881 | 38.41 | 6.489 | 4.933 | 0.76 | 0.239 |
| 110 | 1 | interleaved | 1 | 34.84 | 0.9325 | 38.92 | 5.573 | 3.067 | 0.55 | 0.216 |
| 110 | 3 | contiguous | 1 | 34.91 | 0.9324 | 38.93 | 5.525 | 3.284 | 0.594 | 0.228 |
| 110 | 30 | contiguous | 1 | 35.35 | 0.9333 | 38.94 | 5.381 | 3.608 | 0.671 | 0.236 |
| 140 | 1 | interleaved | 1 | 35.96 | 0.9585 | 39.31 | 4.604 | 2.012 | 0.437 | 0.213 |
| 140 | 3 | contiguous | 1 | 36.01 | 0.9591 | 39.33 | 4.57 | 2.225 | 0.487 | 0.22 |
| 140 | 30 | contiguous | 1 | 36.34 | 0.9602 | 39.33 | 4.463 | 2.464 | 0.552 | 0.235 |
| 200 | 1 | interleaved | 1 | 37.91 | 0.9795 | 39.71 | 3.38 | 0.805 | 0.238 | 0.197 |
| 200 | 3 | contiguous | 1 | 37.89 | 0.9797 | 39.72 | 3.378 | 0.864 | 0.256 | 0.204 |
| 200 | 30 | contiguous | 1 | 37.95 | 0.9799 | 39.71 | 3.37 | 0.99 | 0.294 | 0.238 |
