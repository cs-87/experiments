"""
Host suppression: the largest single lever in the detector.

The mark is white in the DWT LL domain -- P_i = H_i * R with R a white +/-1 vector,
so the carrier is broadband whatever Hadamard row it came from. The host image is
emphatically not white: LL is a 2x downsample of natural luma and is dominated by
low spatial frequencies. Correlating the raw LL against a PRN therefore measures
mostly host, and measurement says so -- host-only correlations have standard
deviation 0.76-0.88 against a mark amplitude of 0.53, i.e. a per-bit SNR of about
-4 dB before anything is done about it.

The GLRT for a known signal in coloured Gaussian noise correlates against C^-1 P.
With white signal and low-pass noise, C^-1 is a high-pass, and most of the available
gain is captured by any reasonable one. Two stages:

  local mean removal      x - box_k(x)        kills the low-frequency host
  local variance norm     / local sigma       equalises spatially varying host power

The second stage matters because host energy is non-stationary: without it a few
high-contrast regions of a patch dominate the inner product and the quiet regions,
where the mark is relatively strongest, contribute almost nothing. Dividing by the
local standard deviation is an approximate per-sample inverse-variance weighting
applied before the projection, which is where it is cheapest.

Measured, no attack, per-patch BER over 245 patch-LLs of inputs/30.mp4:

    identity                        0.220
    lm3   (mean 3x3)                0.059
    lm3+lvn5                        0.054
    lap+lvn9                        0.047

These are NO-ATTACK numbers and the ranking is not expected to survive compression
unchanged -- a filter that whitens the host also amplifies codec noise, which lives
in the same band as the mark. Sweeping this per attack condition is Phase E; the
mode is a constructor argument for exactly that reason.
"""

import cv2
import numpy as np

MODES = ("identity", "lm", "lm_lvn", "lap", "lap_lvn")


class PreWhitener:
    """
    Callable: 2D float plane in, whitened plane out, same shape.

    mean_kernel  box size for local mean removal
    var_kernel   box size for the local variance estimate
    var_floor_pct  the local sigma is floored at this percentile of itself, so a flat
                   region does not divide by ~0 and manufacture enormous evidence out
                   of quantisation noise. This is the difference between a whitener
                   and a noise amplifier.
    """

    def __init__(self, mode="lm_lvn", mean_kernel=3, var_kernel=5, var_floor_pct=10.0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.mean_kernel = int(mean_kernel)
        self.var_kernel = int(var_kernel)
        self.var_floor_pct = float(var_floor_pct)

    def __repr__(self):
        return (f"PreWhitener(mode={self.mode!r}, mean_kernel={self.mean_kernel}, "
                f"var_kernel={self.var_kernel})")

    def _highpass(self, x):
        if self.mode in ("lm", "lm_lvn"):
            k = self.mean_kernel
            return x - cv2.blur(x, (k, k))
        if self.mode in ("lap", "lap_lvn"):
            # Negated so the mark keeps its sign: the 4-neighbour Laplacian kernel is
            # centred on -4, which inverts every correlation and would silently turn a
            # good detector into a good detector of the complement.
            return -cv2.Laplacian(x, cv2.CV_64F, ksize=1)
        return x

    def _normalise(self, d):
        k = self.var_kernel
        sd = np.sqrt(np.maximum(cv2.blur(d * d, (k, k)), 0.0))
        floor = np.percentile(sd, self.var_floor_pct)
        return d / np.maximum(sd, max(floor, 1e-6))

    def __call__(self, plane):
        x = np.ascontiguousarray(plane, dtype=np.float64)
        d = self._highpass(x)
        if self.mode in ("lm_lvn", "lap_lvn"):
            d = self._normalise(d)
        return d
