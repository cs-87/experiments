"""
Per-site 32-D evidence, and how much to trust it.

Two different things degrade a site and they need different handling.

Noise level. How much host and codec noise is in this patch's correlations. The
correlation maps already contain the answer: at any position away from the peak,
T = sum_k c_k^2 is a sample of the null distribution, which is sigma^2 * chi2_K. The
robust centre of T over an annulus around the peak therefore gives sigma without any
extra computation, and it adapts to local texture, local compression damage and
local blur at once.

Presence. Whether there is a mark here at all. This is NOT the same thing, and
inverse-variance weighting alone cannot see it: an unmarked site has exactly the same
sigma as a marked one, so weighting by 1/sigma^2 lets pure noise into the sum at full
strength. The energy is what separates them. Under H1,

    T / sigma^2  ~  noncentral chi2_K with lambda = K * z^2,   z = A / sigma

so lambda_hat = max(0, T/sigma^2 - K) estimates K z^2 and the per-site per-bit SNR
falls straight out as z^2 = lambda_hat / K.

The weight combines both:

    w = (1 / sigma^2) * z^2 / (z^2 + 1)

The first factor is the ML weight for a Gaussian observation of a common mean; the
second is the Wiener shrinkage for the fraction of that observation which is signal.
It goes smoothly to zero on an unmarked site and to 1/sigma^2 on a strong one, with
no threshold anywhere -- which matters, because a hard gate on a quantity this noisy
throws away real evidence in exactly the conditions where evidence is scarce.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.stats import chi2


@lru_cache(maxsize=4096)
def selection_null(rank, n_positions, k):
    """
    Expected energy at a site that was picked as the `rank`-th strongest of
    `n_positions` candidates, under H0.

    Sites are chosen by maximising T = sum_k c_k^2, and then T is used as the evidence
    that a mark is there. That is selection on the statistic being tested, and it is
    not a small effect: measured on unwatermarked 1080p, the selected sites have
    median T/sigma^2 = 73 against an unselected chi2_32 mean of 32, which hands a
    clean video a Wiener weight of 0.56 out of nothing at all.

    The right reference is the order statistic rather than the mean: the r-th largest
    of M draws from chi2_k sits near its upper r/M quantile. Predicted 77.0 at rank 24
    of 1.7 M positions against the 73.0 measured -- close enough that the correction
    very nearly cancels the bias, and conservative in the direction that matters,
    since NMS makes the positions less independent than the count suggests.
    """
    return float(chi2.isf(min((rank + 1) / max(n_positions, 1), 1.0 - 1e-12), k))


@dataclass
class PatchEvidence:
    c: np.ndarray            # (K,) correlations
    sigma: float             # per-component noise standard deviation
    snr2: float              # z^2, the estimated per-bit SNR of this site
    weight: float            # w, as derived above
    energy: float            # T at the peak
    site: object


class PatchReliabilityEstimator:
    """
    sigma from the annulus of the energy map around a peak.

    inner  first radius outside the peak, in plane samples. The peak is one sample
           wide (shifted white PRNs are uncorrelated), so 2 is already clear of it;
           the default leaves margin for interpolation smearing after a resample.
    outer  last radius included. Large enough for a stable median, small enough that
           the estimate is still local.
    """

    # median of chi2_K, Wilson-Hilferty. Exact enough here and keeps scipy out of the
    # inner loop.
    @staticmethod
    def _chi2_median(k):
        return k * (1.0 - 2.0 / (9.0 * k)) ** 3

    def __init__(self, inner=3, outer=14, min_samples=32):
        self.inner = int(inner)
        self.outer = int(outer)
        self.min_samples = int(min_samples)

    def __call__(self, energy, i, j, n_prns):
        H, W = energy.shape
        y0, y1 = max(0, i - self.outer), min(H, i + self.outer + 1)
        x0, x1 = max(0, j - self.outer), min(W, j + self.outer + 1)
        block = energy[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        ring = (np.abs(yy - i) > self.inner) | (np.abs(xx - j) > self.inner)
        vals = block[ring & np.isfinite(block)]
        if vals.size < self.min_samples:
            vals = block[np.isfinite(block)]
        if vals.size == 0:
            return np.nan
        med = float(np.median(vals))
        # A constant region -- a letterbox bar, a blown highlight, a frame of black --
        # has no noise to measure. Returning a vanishing sigma would hand the site a
        # weight of 1/sigma^2 and let quantisation noise dominate the whole video, so
        # the site is dropped instead. The caller treats NaN as "no estimate".
        if not np.isfinite(med) or med <= 0:
            return np.nan
        return float(np.sqrt(med / self._chi2_median(n_prns)))


class PatchEvidenceExtractor:
    """Turns located sites into weighted 32-D evidence."""

    def __init__(self, correlator, reliability=None, weighting="wiener",
                 rank_aware=True):
        if weighting not in ("wiener", "inverse_variance", "unit"):
            raise ValueError(f"unknown weighting {weighting!r}")
        self.correlator = correlator
        self.reliability = reliability or PatchReliabilityEstimator()
        self.weighting = weighting
        self.rank_aware = bool(rank_aware)

    def extract(self, planes, energies, sites):
        K = self.correlator.n_prns
        by_phase = {}
        for s in sites:
            by_phase.setdefault(s.phase, []).append(s)

        out = []
        for phase, group in by_phase.items():
            plane = planes[phase]
            energy = energies[phase]
            cs, kept = self.correlator.at(plane, [(s.i, s.j) for s in group])
            lookup = {(s.i, s.j): s for s in group}
            for row, (i, j) in zip(cs, kept):
                site = lookup[(i, j)]
                sigma = self.reliability(energy, i, j, K)
                if not np.isfinite(sigma) or sigma <= 0:
                    continue
                ref = (selection_null(site.rank, site.n_positions, K)
                       if self.rank_aware else float(K))
                lam = max(0.0, float((row ** 2).sum()) / (sigma ** 2) - ref)
                snr2 = lam / K
                if self.weighting == "unit":
                    w = 1.0
                elif self.weighting == "inverse_variance":
                    w = 1.0 / sigma ** 2
                else:
                    w = (1.0 / sigma ** 2) * snr2 / (snr2 + 1.0)
                out.append(PatchEvidence(c=row, sigma=sigma, snr2=snr2, weight=w,
                                         energy=float((row ** 2).sum()), site=site))
        return out
