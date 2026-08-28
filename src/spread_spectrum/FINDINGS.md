# Spread-spectrum detector: measured results

Regenerated from `findings.jsonl` by `eval/sweep.py`. Edit the JSONL, not this file.

Scored against a **calibrated** acceptance threshold of S1 >= 17.76 (see `calibration.json`), not the parametric one.

## chip2 — patch 128 / DWT L1 / alpha 3.0 (chip 2x2 px), 20 frames

| attack | scale | marked: bits | marked: S1 | marked | unmarked: S1 | unmarked | other ID: S1 | other ID |
|---|---|---|---|---|---|---|---|---|
| `blur_1.0` | 1.0 | 32/32 | 29.9 | ok | 8.4 | ok | 31.5 | ok |
| `blur_2.0` | 1.0 | 32/32 | 13.4 | missed | 4.2 | ok | 9.4 | missed |
| `clean` | 1.0 | 32/32 | 43.8 | ok | 5.1 | ok | 40.6 | ok |
| `combo_full` | 1.0 | 14/32 | 4.0 | missed | 3.4 | ok | 4.9 | missed |
| `combo_resize_h264_blur` | 1.0 | 16/32 | 5.8 | missed | 9.3 | ok | 4.2 | missed |
| `contrast_1.5` | 1.0 | 32/32 | 40.3 | ok | 6.6 | ok | 41.0 | ok |
| `crop_0.75` | 1.0 | 32/32 | 30.7 | ok | 4.0 | ok | 27.6 | ok |
| `crop_rescale_0.75` | 1.3333 | 32/32 | 31.6 | ok | 4.0 | ok | 28.4 | ok |
| `drop_0.2` | 1.0 | 32/32 | 40.0 | ok | 5.2 | ok | 35.7 | ok |
| `fps_half` | 1.0 | 32/32 | 30.5 | ok | 3.9 | ok | 33.0 | ok |
| `gamma_1.6` | 1.0 | 32/32 | 46.3 | ok | 3.1 | ok | 42.9 | ok |
| `grayscale` | 1.0 | 32/32 | 43.8 | ok | 5.1 | ok | 40.6 | ok |
| `h264_1M` | 1.0 | 14/32 | 6.8 | missed | 6.7 | ok | 6.9 | missed |
| `h264_500k` | 1.0 | 22/32 | 9.6 | missed | 11.0 | ok | 5.4 | missed |
| `h264_crf23` | 1.0 | 32/32 | 25.1 | ok | 3.8 | ok | 24.2 | ok |
| `h264_crf28` | 1.0 | 16/32 | 5.4 | missed | 4.5 | ok | 4.9 | missed |
| `h264_crf32` | 1.0 | 18/32 | 4.7 | missed | 6.8 | ok | 4.9 | missed |
| `h264_crf36` | 1.0 | 16/32 | 3.4 | missed | 7.9 | ok | 5.5 | missed |
| `h265_crf28` | 1.0 | 18/32 | 4.5 | missed | 4.0 | ok | 5.0 | missed |
| `h265_crf32` | 1.0 | 16/32 | 4.6 | missed | 9.5 | ok | 4.7 | missed |
| `interpolate` | 1.0 | 32/32 | 31.4 | ok | 4.5 | ok | 26.4 | ok |
| `noise_10` | 1.0 | 18/32 | 4.4 | missed | 4.4 | ok | 4.2 | missed |
| `resample_0.5` | 1.0 | 32/32 | 34.1 | ok | 6.6 | ok | 33.1 | ok |
| `reverse` | 1.0 | 32/32 | 43.8 | ok | 5.1 | ok | 40.6 | ok |
| `saturation_0.0` | 1.0 | 32/32 | 38.4 | ok | 4.3 | ok | 40.4 | ok |
| `scale_360p` | 1.0 | 20/32 | 4.5 | missed | 4.6 | ok | 23.2 | ok |
| `scale_540p` | 0.5 | 32/32 | 37.8 | ok | 5.2 | ok | 34.5 | ok |
| `scale_720p` | 0.6667 | 32/32 | 41.2 | ok | 3.7 | ok | 39.8 | ok |
| `sharpen_1.0` | 1.0 | 32/32 | 44.0 | ok | 5.0 | ok | 40.3 | ok |
| `translate_2` | 1.0 | 32/32 | 43.6 | ok | 5.0 | ok | 40.8 | ok |

**18/30 attacks recovered the exact payload; 0 false positive(s) on the unmarked control.**

## chip4 — patch 256 / DWT L2 / alpha 6.0 (chip 4x4 px), 20 frames

| attack | scale | marked: bits | marked: S1 | marked | unmarked: S1 | unmarked | other ID: S1 | other ID |
|---|---|---|---|---|---|---|---|---|
| `blur_2.0` | 1.0 | 32/32 | 105.5 | ok | 6.0 | ok | 117.8 | ok |
| `clean` | 1.0 | 32/32 | 117.7 | ok | 4.3 | ok | 110.3 | ok |
| `h264_500k` | 1.0 | 18/32 | 4.8 | missed | 5.5 | ok | 3.7 | missed |
| `h264_crf23` | 1.0 | 32/32 | 82.3 | ok | 3.4 | ok | 91.2 | ok |
| `h264_crf28` | 1.0 | 32/32 | 42.1 | ok | 3.3 | ok | 39.6 | ok |
| `h264_crf32` | 1.0 | 24/32 | 5.3 | missed | 6.4 | ok | 9.2 | missed |
| `h264_crf36` | 1.0 | 16/32 | 5.0 | missed | 6.2 | ok | 3.7 | missed |
| `h264_crf40` | 1.0 | 14/32 | 6.0 | missed | 4.2 | ok | 7.3 | missed |
| `h265_crf32` | 1.0 | 24/32 | 6.1 | missed | 4.6 | ok | 17.5 | missed |
| `scale_540p` | 0.5 | 32/32 | 118.9 | ok | 5.3 | ok | 118.6 | ok |

**5/10 attacks recovered the exact payload; 0 false positive(s) on the unmarked control.**

