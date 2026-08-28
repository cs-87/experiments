"""
Perceptual cost of the mark, and the chip-size question.

Perception is the hard constraint, so this gates every robustness claim rather than
following it. Two things here are not answered by PSNR and need saying.

First, the 32-bit mark is not the 1-bit mark at equal ALPHA. W is a sum of 32 signed
PRNs renormalised to RMS 1, so it is approximately Gaussian and takes 23 distinct
values; the 1-bit mark was a single PRN and was strictly +/-1. Measured peak-to-RMS
2.83-4.60 against exactly 1.0. Same RMS, same PSNR, roughly four times the peak
excursion -- sparse impulsive speckle instead of uniform dither, and PSNR cannot tell
them apart. An ALPHA calibrated by eye on a binary mark does not transfer to a
Gaussian one.

Second, chip size. At identical pixel-domain mark RMS, moving the carrier from a 2x2
to a 4x4 chip cut per-patch BER 3-9x on every attack tested. But PSNR-matched is not
perception-matched: contrast sensitivity peaks around 3-5 cycles per degree, so
lower-frequency noise of equal RMS is MORE visible. That is why this module reports
SSIM and LPIPS alongside PSNR and why the chip-size claim is not made on PSNR alone.

Also measured: temporal flicker. The mark is identical in every frame, but the patch
LOCATIONS move with the content, so a patch appearing and disappearing between frames
is an artefact this scheme can produce and a still-frame metric cannot see.
"""

import argparse
import json

import cv2
import numpy as np

from src.spread_spectrum.embed import embed_patch
from src.spread_spectrum.eval.attacks import iter_bgr
from src.spread_spectrum.prn import BalancedPRNGenerator
from utils.bit import get_bit_string
from utils.patch import get_sift_patches


def psnr(a, b, peak=255.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse <= 0 else 20 * np.log10(peak) - 10 * np.log10(mse)


def ssim(a, b, C1=(0.01 * 255) ** 2, C2=(0.03 * 255) ** 2):
    """Gaussian-window SSIM, 11x11 sigma 1.5 -- the standard formulation."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    k = (11, 11)
    mu_a, mu_b = cv2.GaussianBlur(a, k, 1.5), cv2.GaussianBlur(b, k, 1.5)
    saa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a ** 2
    sbb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2)
    return float(np.mean(num / den))


_LPIPS = {}


def lpips(a, b, device=None):
    """
    LPIPS if torchvision's AlexNet weights are reachable, else None.

    Not a hard dependency: it needs a download, and a metric that silently becomes
    zero when offline is worse than one that says it is unavailable.
    """
    try:
        import torch
        import torchvision
    except ImportError:
        return None
    if "net" not in _LPIPS:
        try:
            m = torchvision.models.alexnet(weights="IMAGENET1K_V1").features.eval()
        except Exception:
            _LPIPS["net"] = None
        else:
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            _LPIPS["net"], _LPIPS["dev"] = m.to(dev), dev
    if _LPIPS["net"] is None:
        return None
    import torch
    dev = _LPIPS["dev"]

    def feats(x):
        t = torch.as_tensor(np.stack([x] * 3)[None] / 255.0, dtype=torch.float32,
                            device=dev)
        t = (t - 0.45) / 0.25
        out, h = [], t
        for i, layer in enumerate(_LPIPS["net"]):
            h = layer(h)
            if i in (1, 4, 7, 9, 11):
                out.append(h / (h.pow(2).sum(1, keepdim=True).sqrt() + 1e-8))
        return out

    with torch.no_grad():
        return float(sum(((p - q) ** 2).mean() for p, q in zip(feats(a), feats(b))))


def mark_stats(prn_2d, alpha, level):
    """Pixel-domain amplitude of the mark, derived exactly (see DESIGN.md section 3)."""
    b = 1 << level
    amp = alpha / b
    return {"chip_px": b, "mark_rms": amp, "mark_peak": amp * float(np.abs(prn_2d).max()),
            "peak_to_rms": float(np.abs(prn_2d).max()),
            "predicted_psnr_db": 20 * np.log10(255.0 / amp)}


def measure(source, watermark, alpha=3.0, square_size=128, level=1, seed=8787,
            frames=12, min_separation=None, want_lpips=True):
    bits, _ = get_bit_string(watermark)
    side = square_size >> level
    gen = BalancedPRNGenerator((side, side), seed)
    w = gen.get_balanced_prn_for_bit_string(bits)

    rows, prev_diff = [], None
    flicker, coverage = [], []
    for i, bgr in iter_bgr(source, limit=frames):
        y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float64)
        marked = y.copy()
        n_patch = 0
        for p, yy, xx in get_sift_patches(frame=y.astype(np.uint8),
                                          square_size=square_size,
                                          min_separation=min_separation):
            if p is None:
                continue
            marked[yy[0]:yy[1], xx[0]:xx[1]] = np.clip(
                embed_patch(p, w, alpha=alpha, level=level), 0, 255)
            n_patch += 1
        marked = np.rint(marked)
        diff = marked - y
        touched = np.abs(diff) > 0
        coverage.append(float(touched.mean()))
        rows.append({
            "frame_psnr_db": psnr(y, marked),
            "frame_ssim": ssim(y, marked),
            # Restricted to the marked pixels: a frame-wide PSNR is dominated by the
            # 90-odd percent of the frame that was never touched and flatters the mark.
            "marked_psnr_db": (psnr(y[touched], marked[touched])
                               if touched.any() else float("inf")),
            "mark_rms_measured": float(np.sqrt((diff[touched] ** 2).mean()))
                                 if touched.any() else 0.0,
            "mark_peak_measured": float(np.abs(diff).max()),
            "patches": n_patch,
            "lpips": lpips(y, marked) if want_lpips and i < 4 else None,
        })
        # Flicker: how much the mark itself changes frame to frame. The payload is
        # constant, so anything here comes from patch locations moving with content.
        if prev_diff is not None and prev_diff.shape == diff.shape:
            flicker.append(float(np.sqrt(((diff - prev_diff) ** 2).mean())))
        prev_diff = diff

    def avg(k):
        v = [r[k] for r in rows if r[k] is not None and np.isfinite(r[k])]
        return float(np.mean(v)) if v else None

    return {
        "source": source, "alpha": alpha, "square_size": square_size, "level": level,
        "frames": len(rows),
        "frame_psnr_db": avg("frame_psnr_db"), "frame_ssim": avg("frame_ssim"),
        "marked_psnr_db": avg("marked_psnr_db"),
        "mark_rms_measured": avg("mark_rms_measured"),
        "mark_peak_measured": max(r["mark_peak_measured"] for r in rows),
        "patches_per_frame": avg("patches"),
        "coverage": float(np.mean(coverage)),
        "lpips": avg("lpips"),
        "flicker_rms": float(np.mean(flicker)) if flicker else None,
        **mark_stats(w, alpha, level),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--source", default="inputs/30.mp4")
    ap.add_argument("--watermark", type=lambda s: int(s, 0), default=0x15771A93)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--configs", default="128:1:3,256:2:6,512:3:12",
                    help="square:level:alpha triples. The defaults are matched on "
                         "pixel-domain mark RMS, i.e. on PSNR")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--out", default="src/spread_spectrum/perceptual.json")
    args = ap.parse_args(argv)

    out = []
    hdr = (f"{'config':>14} {'chip':>5} {'patch/f':>8} {'cover':>6} {'markRMS':>8} "
           f"{'peak':>6} {'framePSNR':>10} {'markPSNR':>9} {'SSIM':>7} {'LPIPS':>8} "
           f"{'flicker':>8}")
    print(hdr)
    for spec in args.configs.split(","):
        sq, lv, al = spec.split(":")
        r = measure(args.source, args.watermark, alpha=float(al), square_size=int(sq),
                    level=int(lv), frames=args.frames, want_lpips=not args.no_lpips)
        out.append(r)
        print(f"{sq + '/L' + lv + ' a' + al:>14} {r['chip_px']}x{r['chip_px']:<3} "
              f"{r['patches_per_frame']:8.1f} {100 * r['coverage']:5.1f}% "
              f"{r['mark_rms_measured']:8.3f} {r['mark_peak_measured']:6.1f} "
              f"{r['frame_psnr_db']:10.2f} {r['marked_psnr_db']:9.2f} "
              f"{r['frame_ssim']:7.4f} "
              f"{('%8.5f' % r['lpips']) if r['lpips'] is not None else '     n/a'} "
              f"{r['flicker_rms']:8.3f}")
    print("\nPSNR-matched is not perception-matched: SSIM and LPIPS are the columns "
          "that decide the chip-size question, not frame PSNR.")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
