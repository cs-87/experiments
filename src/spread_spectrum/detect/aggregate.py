"""
Pooling evidence across patches and across frames.

These are the same operation, and saying so is more useful than pretending
otherwise. Patches within a frame and patches across frames are both independent
observations of one common 32-vector A*s: the payload is identical in every patch of
every frame, with no temporal code, so there is no frame-to-bit map to respect and
no synchronisation to get right. SpatialAggregator and TemporalAggregator are
therefore documented aliases of one accumulator, kept as separate names only because
the pipeline reads better with them.

That absence of a temporal code is worth stating plainly: frame dropping,
duplication, reordering, frame-rate conversion, temporal shift and interpolation
cost this detector observations and nothing else. No aligner is needed.

The estimator is the inverse-variance weighted mean, which is ML for independent
Gaussian observations of a common mean, followed by one Huber reweighting pass. The
Huber pass is belt-and-braces: the main contaminant, a site with no mark, is already
suppressed to near-zero weight by evidence.py's Wiener factor. What it catches is the
other kind -- a site with strong energy pointing the wrong way, from a local artefact
that happens to correlate.

Two things that are easy to get wrong and are deliberate here:

  The clipping scale is the OBSERVED spread of the residuals, not the theoretical
  null spread. Clipping at a fixed multiple of the null pins every contribution at
  the ceiling once the evidence is strong, which silently degrades a soft weighted
  sum into a majority vote.

  Nothing is ever hard-gated. A site with twice the median sigma still carries a
  quarter of a good site's evidence; discarding it gains nothing that
  inverse-variance weighting has not already done by exactly the right factor.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Aggregate:
    c: np.ndarray             # (K,) pooled evidence
    se: np.ndarray            # (K,) standard error of each component
    z: np.ndarray             # (K,) c / se
    n_obs: int                # sites contributing
    m_eff: float              # (sum w)^2 / sum w^2 -- effective observation count
    clipped: int              # entries touched by the Huber pass


class EvidenceAggregator:
    """
    Accumulate PatchEvidence, then pool.

    huber_c   clip threshold in units of the robust residual scale. None disables.
    passes    IRLS iterations of the Huber reweighting.
    """

    def __init__(self, huber_c=3.0, passes=2):
        self.huber_c = huber_c
        self.passes = int(passes)
        self._c = []
        self._w = []
        self._s = []

    def __len__(self):
        return len(self._c)

    def add(self, evidence):
        for e in evidence if isinstance(evidence, (list, tuple)) else [evidence]:
            if not np.isfinite(e.c).all() or not np.isfinite(e.weight):
                continue
            self._c.append(np.asarray(e.c, dtype=np.float64))
            self._w.append(float(e.weight))
            self._s.append(float(e.sigma))

    def result(self):
        if not self._c:
            raise ValueError("no evidence accumulated")
        C = np.stack(self._c)                       # (M, K)
        w = np.asarray(self._w)                     # (M,)
        sig = np.asarray(self._s)                   # (M,)
        if w.sum() <= 0:
            w = np.ones_like(w)

        chat = (w[:, None] * C).sum(0) / w.sum()
        clipped = 0

        if self.huber_c is not None and len(C) > 4:
            for _ in range(self.passes):
                # Residuals in units of each site's own noise, so a noisy site is not
                # penalised for being noisy -- only for disagreeing beyond its noise.
                r = (C - chat[None, :]) / sig[:, None]
                scale = 1.4826 * np.median(np.abs(r))
                if not np.isfinite(scale) or scale <= 0:
                    break
                lim = self.huber_c * scale
                rc = np.clip(r, -lim, lim)
                clipped = int((np.abs(r) > lim).sum())
                Cc = chat[None, :] + rc * sig[:, None]
                chat = (w[:, None] * Cc).sum(0) / w.sum()
            C_eff = chat[None, :] + np.clip((C - chat[None, :]) / sig[:, None],
                                            -lim, lim) * sig[:, None]
        else:
            C_eff = C

        # Sandwich standard error: the weighted scatter of what actually arrived,
        # rather than the model's sum w^2 sigma^2. It absorbs residual correlation
        # between nearby patches and non-Gaussian tails, both of which the model
        # form would understate.
        resid = C_eff - chat[None, :]
        se = np.sqrt((w[:, None] ** 2 * resid ** 2).sum(0)) / w.sum()
        se = np.maximum(se, 1e-12)

        return Aggregate(c=chat, se=se, z=chat / se, n_obs=len(C),
                         m_eff=float(w.sum() ** 2 / (w ** 2).sum()), clipped=clipped)


# The pipeline reads better with two names; they are one operation. See the module
# docstring for why pooling over patches and pooling over frames are not different
# problems for this scheme.
SpatialAggregator = EvidenceAggregator
TemporalAggregator = EvidenceAggregator
