"""
Finding the marked patches without the original frame and without the ID list.

The embedder puts patches wherever SIFT fired on the *unmarked* frame. After an
attack those keypoints move, the (size, response) ordering reshuffles, and the
greedy suppression cascade turns one reordering into a different patch set. Since
one pixel of misalignment costs four fifths of the evidence, re-detecting features
is solving a harder problem than the one that needs solving.

The watermark localises itself instead. The per-PRN correlations at position (y, x)
give

    T(y, x) = sum_k c_k(y, x)^2

which is INDEPENDENT OF THE CODEWORD -- it is the squared length of the projection
onto the span of the PRNs, and every valid payload is a unit vector in that span
scaled the same way. So T can be computed before knowing which ID is present, needs
no candidate list, and is the same quantity the decoder's evidence is built from.

The peak is genuinely one sample wide. At offset d the window overlaps the mark in
all but d rows, but what it overlaps there is the mark shifted by d, and shifted
white PRNs are uncorrelated with unshifted ones. So there is no coarse-to-fine
shortcut on this surface: the search has to be exhaustive, which is exactly what the
FFT gives for free.
"""

from dataclasses import dataclass

import numpy as np

from src.spread_spectrum.detect.correlate import ll_block_size, plane_to_frame


@dataclass
class Site:
    """A candidate marked patch, in both coordinate systems."""
    frame_y: int
    frame_x: int
    phase: tuple
    i: int
    j: int
    energy: float
    frame_index: int = -1
    rank: int = 0                # 0 = strongest peak in the frame
    n_positions: int = 1         # candidate positions this rank was selected from


class PatchLocaliser:
    """
    Peaks of the codeword-independent energy surface, across all B^2 phase planes.

    nms_radius_px  minimum separation between reported sites, in frame pixels.
                   Defaults to half the patch size: the embedder's own suppression
                   keeps patches at least one patch apart, so half of that can never
                   merge two real patches while still collapsing a peak's immediate
                   neighbourhood.
    max_sites      cap per frame. Extra sites are not harmful -- an unmarked site
                   contributes near-zero weight (see evidence.py) rather than noise --
                   but they cost correlator time.
    """

    def __init__(self, correlator, level=1, nms_radius_px=None, max_sites=64):
        self.correlator = correlator
        self.level = int(level)
        self.block = ll_block_size(level)
        self.patch_px = correlator.side * self.block
        self.nms_radius_px = int(nms_radius_px if nms_radius_px is not None
                                 else self.patch_px // 2)
        self.max_sites = int(max_sites)

    def __repr__(self):
        return (f"PatchLocaliser(level={self.level}, patch_px={self.patch_px}, "
                f"nms_radius_px={self.nms_radius_px}, max_sites={self.max_sites})")

    def energies(self, planes):
        """{phase: energy map}, one dense correlation pass per phase plane."""
        return {ph: self.correlator.dense(pl)[0] for ph, pl in planes.items()}

    def heatmap(self, planes, energies=None, frame_shape=None):
        """
        The B^2 phase energy maps interleaved back onto the full-resolution grid.

        Positions whose correlation window would run off the end of a plane hold
        circular wrap-around rather than a real overlap, so they are masked to -inf
        instead of being left to win a maximum they did not earn.
        """
        energies = energies if energies is not None else self.energies(planes)
        if frame_shape is None:
            any_ph, any_pl = next(iter(planes.items()))
            frame_shape = (any_pl.shape[0] * self.block + any_ph[0],
                           any_pl.shape[1] * self.block + any_ph[1])
        heat = np.full(frame_shape, -np.inf)
        B = self.block
        for (py, px), e in energies.items():
            hv, wv = self.correlator.valid_shape(e.shape)
            if hv <= 0 or wv <= 0:
                continue
            ys = py + np.arange(hv) * B
            xs = px + np.arange(wv) * B
            ys = ys[ys < frame_shape[0]]
            xs = xs[xs < frame_shape[1]]
            heat[np.ix_(ys, xs)] = e[:len(ys), :len(xs)]
        return heat

    def sites(self, planes, energies=None, frame_shape=None, frame_index=-1,
              candidate_factor=24):
        """
        Non-maximum-suppressed peaks of the heatmap, strongest first.

        Shortlist first, suppress second. Repeatedly taking argmax over a 1080p
        heatmap and blanking a box costs one full pass per site -- 48 passes over
        2 M elements, which measured as the single slowest step in the frame. One
        argpartition to the top `max_sites * candidate_factor` positions, then greedy
        suppression over that short list, gives the identical answer: suppression can
        only ever remove candidates, so a position outside the top-C cannot be
        promoted into the output unless C sites were suppressed ahead of it, and the
        factor is set well above the suppression ratio the geometry allows.
        """
        energies = energies if energies is not None else self.energies(planes)
        heat = self.heatmap(planes, energies=energies, frame_shape=frame_shape)

        finite = np.isfinite(heat) & (heat > 0)
        if not finite.any():
            return []                       # constant frame: nothing to localise
        n_positions = int(finite.sum())

        flat = np.where(finite.ravel(), heat.ravel(), -np.inf)
        n_cand = min(int(self.max_sites * candidate_factor), int(finite.sum()))
        cand = np.argpartition(flat, -n_cand)[-n_cand:]
        cand = cand[np.argsort(flat[cand])[::-1]]
        ys, xs = np.unravel_index(cand, heat.shape)

        r = self.nms_radius_px
        out, taken_y, taken_x = [], [], []
        for y, x in zip(ys, xs):
            if any(abs(y - ty) <= r and abs(x - tx) <= r
                   for ty, tx in zip(taken_y, taken_x)):
                continue
            out.append(Site(frame_y=int(y), frame_x=int(x),
                            phase=(int(y) % self.block, int(x) % self.block),
                            i=int(y) // self.block, j=int(x) // self.block,
                            energy=float(heat[y, x]), frame_index=frame_index,
                            rank=len(out), n_positions=n_positions))
            taken_y.append(y)
            taken_x.append(x)
            if len(out) >= self.max_sites:
                break
        return out


def refine_subpixel(heat, y, x):
    """
    Parabolic peak interpolation, per axis, on the full-resolution heatmap.

    Reported for diagnostics only. Integer localisation already lands within half a
    pixel and half a pixel costs almost nothing (BER 0.076 against 0.052 aligned),
    so nothing downstream resamples on this -- but a systematic sub-pixel offset
    across many sites is the signature of a scale error, which is worth being able
    to see.
    """
    def axis(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-12 else 0.5 * (a - c) / d
    H, W = heat.shape
    dy = axis(heat[y - 1, x], heat[y, x], heat[y + 1, x]) if 0 < y < H - 1 else 0.0
    dx = axis(heat[y, x - 1], heat[y, x], heat[y, x + 1]) if 0 < x < W - 1 else 0.0
    return float(np.clip(dy, -1, 1)), float(np.clip(dx, -1, 1))
