"""
Empirical calibration of the NO-WATERMARK thresholds.

The parametric thresholds in decode.py come from a Gaussian/chi-square null, and that
null is a starting point rather than the answer: real host noise has heavier tails,
the sites are chosen by maximising the very statistic being tested, and the natural
image structure that survives whitening is not independent between patches. All three
push the observed null above the theoretical one, and only measurement says by how
much.

Getting enough null samples is the practical problem. Running the detector over
unmarked clips gives one sample per clip, and there are two distinct source clips on
this machine (inputs/{15,30,60,120}.mp4 are byte-identical content at different
lengths -- verified, not assumed). So the null is drawn a different way: run the
detector with PRNs from a DECOY SEED. A decoy key produces a codebook with the same
construction, the same statistics and the same exact orthogonality, but no relationship
to whatever is embedded. Every decoy seed is therefore an independent draw from the
null, on real content, through the real attack, with the real selection procedure and
the real aggregation -- everything that inflates the null is still present. One clip
yields as many null samples as you are willing to pay for.

It also gives the strongest false-positive case there is: a decoy key run against a
genuinely marked clip. If the detector reports a watermark there, it is responding to
content rather than to the mark.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm

from src.spread_spectrum.detect.pipeline import Detector, DetectorConfig
from src.spread_spectrum.eval.attacks import apply_attack, build_suite
from src.spread_spectrum.ids import CodewordSet

OUT = Path("src/spread_spectrum/calibration.json")


def null_samples(video, ids, cfg, seeds, frames=8, attack=None, stride=1,
                 workdir="/tmp/ss_cal", verbose=True):
    """One (S1, S2) draw per decoy seed."""
    suite = build_suite()
    atk = suite[attack] if attack else suite["clean"]
    out = []
    for seed in seeds:
        d = Detector(ids, DetectorConfig(**{**cfg.__dict__, "seed": int(seed)}))
        try:
            r, _ = d.detect_frames(apply_attack(atk, video, limit=frames,
                                                stride=stride, workdir=workdir))
        except ValueError:
            continue
        out.append({"seed": int(seed), "s1": r.s1, "s2": r.s2,
                    "margin": r.margin, "total_weight": r.total_weight,
                    "accepted": r.accepted})
        if verbose:
            print(f"  seed {seed:6d}: S1={r.s1:7.2f}  S2={r.s2:9.1f}  "
                  f"margin={r.margin:6.2f}  w={r.total_weight:9.3g}"
                  + ("   <-- ACCEPTED (false positive)" if r.accepted else ""))
    return out


def recommend(samples, n_codewords, n_bits=32, fprs=(1e-3, 1e-6, 1e-9)):
    """
    Thresholds from the observed null.

    S1 is already a maximum over the codebook, so its upper tail is Gumbel-shaped; a
    Gumbel fit by moments extrapolates far more honestly than a Gaussian would. S2 is a
    sum of squares, fitted as a scaled chi-square with the scale taken from the observed
    mean -- which is exactly the inflation factor over the theoretical null.

    Extrapolating a 1e-9 threshold from a few dozen samples is an extrapolation and is
    labelled as one. Its purpose is to show how far the observed null sits from the
    parametric one, not to certify a rate that has not been observed.
    """
    s1 = np.array([s["s1"] for s in samples], float)
    s2 = np.array([s["s2"] for s in samples], float)
    out = {"n_samples": len(samples),
           "s1_observed": {"mean": float(s1.mean()), "sd": float(s1.std(ddof=1)),
                           "max": float(s1.max())},
           "s2_observed": {"mean": float(s2.mean()), "sd": float(s2.std(ddof=1)),
                           "max": float(s2.max())},
           "s2_inflation_over_chi2": float(s2.mean() / n_bits),
           "thresholds": {}}
    # Gumbel by moments: sd = beta*pi/sqrt(6), mean = mu + beta*euler_gamma
    beta = s1.std(ddof=1) * np.sqrt(6) / np.pi
    mu = s1.mean() - beta * np.euler_gamma
    scale2 = s2.mean() / n_bits
    for a in fprs:
        out["thresholds"][f"{a:g}"] = {
            "s1_empirical_gumbel": float(mu - beta * np.log(-np.log1p(-a))),
            "s1_parametric": float(norm.isf(a / n_codewords)),
            "s2_empirical_scaled_chi2": float(scale2 * chi2.isf(a, n_bits)),
            "s2_parametric": float(chi2.isf(a, n_bits)),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--unmarked", default="inputs/30.mp4")
    ap.add_argument("--marked", default=None,
                    help="a genuinely marked clip; decoy keys run against it are the "
                         "strongest false-positive case available")
    ap.add_argument("--n-seeds", type=int, default=24)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--attack", default=None)
    ap.add_argument("--id-count", type=int, default=200_000)
    ap.add_argument("--max-sites", type=int, default=48)
    ap.add_argument("--whiten", default="lm_lvn")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    ids = CodewordSet.recommended(args.id_count)
    cfg = DetectorConfig(whiten=args.whiten, max_sites=args.max_sites,
                         device=args.device)
    # Decoy seeds, deliberately far from the production seed 8787.
    seeds = list(range(100_000, 100_000 + args.n_seeds))

    report = {"attack": args.attack or "clean", "frames": args.frames,
              "id_count": args.id_count, "whiten": args.whiten}

    print(f"decoy keys on UNMARKED {args.unmarked}")
    unmarked = null_samples(args.unmarked, ids, cfg, seeds, args.frames, args.attack)
    report["unmarked"] = {"samples": unmarked,
                          **recommend(unmarked, args.id_count)}

    if args.marked:
        print(f"\ndecoy keys on MARKED {args.marked} "
              f"(a hit here means the detector is reading content, not the mark)")
        marked = null_samples(args.marked, ids, cfg, seeds, args.frames, args.attack)
        report["marked_decoy_key"] = {"samples": marked,
                                      **recommend(marked, args.id_count)}

    print(f"\n{'':22s} {'S1 mean':>9} {'S1 max':>8} {'S2 mean':>9} {'S2 max':>9} {'FP':>4}")
    for label in ("unmarked", "marked_decoy_key"):
        if label not in report:
            continue
        r = report[label]
        fp = sum(s["accepted"] for s in r["samples"])
        print(f"{label:22s} {r['s1_observed']['mean']:9.2f} {r['s1_observed']['max']:8.2f} "
              f"{r['s2_observed']['mean']:9.1f} {r['s2_observed']['max']:9.1f} "
              f"{fp:2d}/{len(r['samples'])}")

    print(f"\nS2 inflation over the chi2_32 null: "
          f"{report['unmarked']['s2_inflation_over_chi2']:.2f}x")
    print(f"\n{'target FPR':>11} {'S1 param':>9} {'S1 emp':>8} {'S2 param':>9} {'S2 emp':>8}")
    for a, t in report["unmarked"]["thresholds"].items():
        print(f"{a:>11} {t['s1_parametric']:9.2f} {t['s1_empirical_gumbel']:8.2f} "
              f"{t['s2_parametric']:9.1f} {t['s2_empirical_scaled_chi2']:8.1f}")
    print("\nthe 1e-6 and 1e-9 rows are extrapolations from "
          f"{report['unmarked']['n_samples']} samples, not observed rates")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
