"""Shared fixtures. Synthetic hosts keep the unit tests off the video files."""

import numpy as np

from src.spread_spectrum.embed import embed_patch
from src.spread_spectrum.prn import BalancedPRNGenerator

SEED = 8787


def synthetic_host(shape=(512, 512), seed=0, smooth=3.0):
    """
    A textured host with roughly natural 1/f-ish statistics.

    Not a real frame, but it has the property the detector actually depends on: most
    of its energy is low-frequency, so it is a genuine adversary for a white carrier.
    Flat noise would make the detector look much better than it is.
    """
    import cv2
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    x = cv2.GaussianBlur(x, (0, 0), smooth)
    x = (x - x.min()) / max(float(np.ptp(x)), 1e-9)
    return 40.0 + 175.0 * x


def mark_positions(shape=(512, 512), square=128, margin=16):
    """Fixed, non-overlapping patch origins -- ground truth for localisation."""
    H, W = shape
    return [(y, x)
            for y in range(margin, H - square - margin + 1, square + margin)
            for x in range(margin, W - square - margin + 1, square + margin)]


def embed_into(host, bits, positions, alpha=3.0, square=128, level=1, seed=SEED):
    gen = BalancedPRNGenerator((square >> level, square >> level), seed)
    w = gen.get_balanced_prn_for_bit_string(bits)
    out = host.copy()
    for (y, x) in positions:
        patch = out[y:y + square, x:x + square]
        out[y:y + square, x:x + square] = np.clip(
            embed_patch(patch, w, alpha=alpha, level=level), 0, 255)
    return np.rint(out)
