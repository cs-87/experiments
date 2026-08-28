"""
Closed-set decoding and the NO-WATERMARK test.

Decoding is one matrix-vector product. The evidence c is a sufficient statistic for
the codeword -- the signal lies entirely in span{P_1..P_32}, so the likelihood of any
hypothesis q depends on the observation only through c -- and under c = A*q + n with
n ~ N(0, diag(se^2)) the log-likelihood is proportional to sum_k c_k q_k / se_k^2.
That makes ML decoding a 200,000 x 32 GEMM: 12.8 MFLOP, microseconds. There is no
case for approximate nearest neighbours or a learned retrieval model; they would add
error to the one step that currently contributes none.

Normalised dot product and Euclidean distance give the identical ranking here, since
||c|| does not depend on q and every q has the same norm. They are not alternatives,
they are the same rule written differently.

The acceptance test is the part that needs care. An unwatermarked video always has a
best-scoring codeword, and with 200,000 hypotheses the best of them is far from
typical: the maximum of N standard normals concentrates near sqrt(2 ln N), which is
4.94 at N = 200,000. A threshold set as though one hypothesis were being tested fires
constantly. Two statistics, both already computed:

  S1 = max_q (sum_k c_k q_k / se_k^2) / sqrt(sum_k 1/se_k^2)
       standard normal per hypothesis under H0; the multiplicity is paid for by
       setting the threshold from alpha / N rather than alpha.

  S2 = sum_k (c_k / se_k)^2
       chi2_32 under H0, noncentral under H1. CODEWORD-INDEPENDENT, so it is not
       inflated by the size of the ID list and it answers "is a mark present" apart
       from "which one". It is the same quantity the localiser maximises.

Requiring both is conservative -- they are positively dependent, so the joint false
positive rate is below the nominal -- and it separates the two distinct failures: high
S2 with low S1 means a mark is present whose ID is not in the list.

The best-vs-second margin is reported but is NOT the acceptance test. Measured under
H0 with a random 200,000-ID set it is 0.002 to 0.011, because the runner-up differs
from the winner in one bit and scores almost identically. With the recommended
d_min = 6 codebook the runner-up is six bits away and the margin becomes informative,
which is a second reason to design the ID set rather than draw it.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2, norm

from utils.bit import uint32_to_hex


@dataclass
class DecodeResult:
    watermark_id: int
    hex: str
    bits: str
    s1: float                  # normalised best score
    s1_second: float
    margin: float
    s2: float                  # evidence energy
    z: np.ndarray              # per-bit z
    min_abs_z: float
    n_obs: int
    m_eff: float
    accepted: bool = False
    reason: str = ""
    thresholds: dict = field(default_factory=dict)

    def __str__(self):
        verdict = self.hex if self.accepted else "NO_WATERMARK"
        return (f"{verdict}  S1={self.s1:.2f} (2nd {self.s1_second:.2f}, "
                f"margin {self.margin:.2f})  S2={self.s2:.1f}  "
                f"min|z|={self.min_abs_z:.2f}  obs={self.n_obs} "
                f"(M_eff={self.m_eff:.1f}){'' if self.accepted else '  -- ' + self.reason}")


class CodewordDecoder:
    """Exhaustive ML search over a CodewordSet."""

    def __init__(self, codeword_set):
        self.codewords = codeword_set
        self._signs = codeword_set.signs.astype(np.float32)      # (N, 32)

    def __repr__(self):
        return (f"CodewordDecoder({len(self.codewords):,} codewords, "
                f"d_min={self.codewords.min_distance})")

    def scores(self, c, se):
        """Normalised score of every codeword. Standard normal per codeword under H0."""
        v = (np.asarray(c) / np.asarray(se) ** 2).astype(np.float32)
        norm_factor = np.sqrt(float((1.0 / np.asarray(se) ** 2).sum()))
        return (self._signs @ v) / norm_factor

    def decode(self, aggregate):
        s = self.scores(aggregate.c, aggregate.se)
        best = int(np.argmax(s))
        # partition is O(N) and we only need the runner-up, not a sort of 200,000.
        second = float(np.partition(s, -2)[-2])
        wid = int(self.codewords.ids[best])
        z = aggregate.z
        return DecodeResult(
            watermark_id=wid,
            hex=uint32_to_hex(wid),
            bits=format(wid, "032b"),
            s1=float(s[best]),
            s1_second=second,
            margin=float(s[best]) - second,
            s2=float((z ** 2).sum()),
            z=z,
            min_abs_z=float(np.abs(z).min()),
            n_obs=aggregate.n_obs,
            m_eff=aggregate.m_eff,
        )


class WatermarkHypothesisTester:
    """
    Turns a DecodeResult into ID or NO_WATERMARK at a stated per-video false
    positive rate.

    The thresholds below are the Gaussian/chi2 null, which is a starting point and
    not the deliverable: real host noise has heavier tails than Gaussian, so the
    operating thresholds must come from running the whole detector over unwatermarked
    clips spanning the content range (eval/calibrate.py). Passing them in explicitly
    is how that calibration gets used.
    """

    def __init__(self, n_codewords, n_bits=32, fpr=1e-6, s1=None, s2=None):
        self.n_codewords = int(n_codewords)
        self.fpr = float(fpr)
        # Bonferroni over the codebook: the best of N hypotheses is being tested, so
        # each one gets alpha / N.
        self.s1_threshold = float(s1) if s1 is not None else float(
            norm.isf(self.fpr / self.n_codewords))
        self.s2_threshold = float(s2) if s2 is not None else float(
            chi2.isf(self.fpr, n_bits))

    def __repr__(self):
        return (f"WatermarkHypothesisTester(N={self.n_codewords:,}, fpr={self.fpr:g}, "
                f"S1>{self.s1_threshold:.2f}, S2>{self.s2_threshold:.1f})")

    def test(self, result):
        result.thresholds = {"s1": self.s1_threshold, "s2": self.s2_threshold,
                             "fpr": self.fpr}
        ok1 = result.s1 >= self.s1_threshold
        ok2 = result.s2 >= self.s2_threshold
        result.accepted = bool(ok1 and ok2)
        if result.accepted:
            result.reason = ""
        elif not ok2 and not ok1:
            result.reason = "no mark detected"
        elif not ok2:
            result.reason = f"energy below threshold (S2={result.s2:.1f})"
        else:
            result.reason = (f"mark present but no listed ID explains it "
                             f"(S1={result.s1:.2f})")
        return result
