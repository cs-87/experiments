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

import cv2

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


class GeometrySearch:
    """
    Recover a global scale (and optionally rotation) before decoding.

    For digital attacks the geometry is a per-VIDEO constant -- a resize or a rotate is
    applied once to the whole clip, not per frame -- so this runs on one or two frames
    and the answer is then locked, which is what makes the search affordable.

    The objective is the detector's own evidence: total per-site z^2, summed over the
    sites the real pipeline finds after un-warping by each hypothesis. That is
    codeword-independent, so no ID list is needed, and it is the quantity the decoder
    actually consumes.

    Two cheaper surrogates were tried first and both failed, which is why the expensive
    one is here:

      Peakiness of the energy surface, with the frame warped per hypothesis. Undoing a
      small scale means upsampling, upsampling smooths the plane, and the whitened
      residual of a smooth plane is heavy-tailed -- so peakiness rose with the amount
      of upsampling whether or not a mark was present. It picked the smallest scale on
      the grid every time and scored an unmarked clip (265) above a marked one (249).

      Directional coherence of the top peaks, with the template warped instead of the
      frame. Better founded -- one payload really does put the same 32-vector at every
      patch, while content-driven peaks point anywhere -- but natural image structure
      is coherent too: unmarked clips measured 0.25-0.34 against marked clips at
      0.21-0.38, no separation at all.

    Using the real evidence, measured on one 1080p frame: an unattacked marked clip
    scores 14.8 at scale 1.0 against 0.7 for the runner-up, a 540p leak scores 6.2 at
    0.5 against 1.0, and every unmarked clip stays below 1.2 at every scale with no
    hypothesis standing out. Where it does miss -- a 720p leak -- the top score is 0.6,
    i.e. it correctly reports that there is nothing to lock onto, because at that
    downscale the mark really is gone.
    """

    #: common re-publication ratios, exact where they are exact
    DEFAULT_SCALES = (1 / 3, 0.36, 0.4, 0.45, 0.5, 0.5625, 2 / 3, 0.7, 0.75, 0.8, 0.9,
                      1.0, 1.1, 1.25, 4 / 3, 1.5, 2.0)

    def __init__(self, evidence_fn, scales=None, rotations=(0.0,), window=1280,
                 refine=True, min_z=15.0, anchor_ratio=1.5, min_score=2.0):
        self.evidence_fn = evidence_fn
        self.scales = tuple(scales if scales is not None else self.DEFAULT_SCALES)
        self.rotations = tuple(rotations)
        self.window = int(window)
        self.refine = bool(refine)
        self.min_z = float(min_z)
        self.anchor_ratio = float(anchor_ratio)
        self.min_score = float(min_score)

    def _window(self, luma):
        """
        Central crop, applied AFTER warping.

        Undoing a scale of 1/3 means upsampling the frame nine-fold in area, so
        cropping the input would leave that hypothesis ten times more expensive than
        the rest and make the whole search cost whatever the smallest scale on the grid
        happens to be. Cropping the output instead costs every hypothesis the same and
        simply gives the heavily-upsampled ones a smaller field of view, which is the
        honest trade: there genuinely is less independent information there.
        """
        h, w = luma.shape
        wh, ww = min(self.window, h), min(self.window, w)
        return luma[(h - wh) // 2:(h - wh) // 2 + wh, (w - ww) // 2:(w - ww) // 2 + ww]

    def _score(self, frames, scale, rotation):
        total = 0.0
        for f in frames:
            warped = self._window(self.warp(f, scale, rotation))
            if min(warped.shape) < 192:
                return 0.0
            total += sum(e.snr2 for e in self.evidence_fn(warped))
        return float(total)

    def estimate(self, frames, verbose=False):
        """
        Returns (scale, rotation, table), with every hypothesis scored so a flat
        surface reads as flat rather than as a confident answer.

        ANCHORED to (1.0, 0.0). The search maximises the same evidence the decision is
        later made on, so left unanchored it is a second layer of selection bias
        stacked on the one the evidence extractor already corrects for: on an unmarked
        clip it locks whichever of ~21 hypotheses happens to look best, and the
        resampling that follows then inflates the null. Measured, an unmarked clip went
        from S2 = 34 with the geometry search off to S2 = 74 with it on.

        So a non-identity geometry has to earn it three ways: stand out from the rest
        of the grid (min_z robust standard deviations above the median hypothesis),
        beat the identity hypothesis by anchor_ratio, and clear min_score of absolute
        per-frame evidence. The thresholds are set from the gap that was measured, not
        guessed: genuine locks came in at z = 53 (a 540p leak) and z = 124 (an
        unattacked marked clip), while spurious ones on unmarked clips sat at z = 5.0
        to 7.4. An earlier min_z of 5 let one of those through and it produced an
        outright false positive -- an unmarked clip locked to scale 1.1, and the
        resampling that followed pushed it to S1 = 7.43 and S2 = 91.4, over both
        acceptance thresholds. Anchored to 1.0 the same clip reads S1 = 4.3, S2 = 34.3.

        The asymmetry is deliberate. An unnecessary resample costs evidence, a missed
        one costs all of it -- but a spurious one costs evidence AND manufactures a
        false positive out of nothing, so the default has to be "no geometric change".
        """
        table = [(s, r, self._score(frames, s, r))
                 for s in self.scales for r in self.rotations]
        table.sort(key=lambda t: -t[2])

        if self.refine and table[0][2] > 0:
            # One local pass around the winner: the grid is coarse and a few percent of
            # scale error costs real evidence.
            s0, r0, _ = table[0]
            table += [(s0 * f, r0, self._score(frames, s0 * f, r0))
                      for f in (0.94, 0.97, 1.03, 1.06)]
            table.sort(key=lambda t: -t[2])

        best_s, best_r, best = table[0]
        identity = next((v for s, r, v in table
                         if abs(s - 1.0) < 1e-9 and abs(r) < 1e-9), 0.0)
        others = np.array([v for s, r, v in table[1:]], float)
        mad = 1.4826 * np.median(np.abs(others - np.median(others))) if others.size else 0.0
        z = (best - np.median(others)) / max(mad, 1e-9) if others.size else 0.0

        n = max(len(frames), 1)
        locked = (abs(best_s - 1.0) < 1e-9 and abs(best_r) < 1e-9) or (
            z >= self.min_z
            and best >= self.anchor_ratio * max(identity, 1e-9)
            and best / n >= self.min_score)
        if verbose:
            for s, r, sc in table[:6]:
                print(f"    scale {s:6.4f} rot {r:+5.2f}: {sc:9.2f}")
            print(f"    best z = {z:.1f} vs min {self.min_z}; identity = {identity:.2f}"
                  f" -> {'lock ' + format(best_s, '.4f') if locked else 'anchor to 1.0'}")
        if not locked:
            return 1.0, 0.0, table
        return best_s, best_r, table

    @staticmethod
    def lock_quality(table):
        """(z of the winner over the rest of the grid, winner score, identity score)."""
        if not table:
            return 0.0, 0.0, 0.0
        best = table[0][2]
        identity = next((v for s, r, v in table
                         if abs(s - 1.0) < 1e-9 and abs(r) < 1e-9), 0.0)
        others = np.array([v for _, _, v in table[1:]], float)
        if others.size == 0:
            return 0.0, best, identity
        mad = 1.4826 * np.median(np.abs(others - np.median(others)))
        return float((best - np.median(others)) / max(mad, 1e-9)), float(best), float(identity)

    @staticmethod
    def warp(luma, scale, rotation=0.0):
        """Bring a leak back to the original geometry, once the geometry is known."""
        out = luma
        if abs(scale - 1.0) > 1e-6:
            h, w = out.shape
            out = cv2.resize(out, (max(8, int(round(w / scale))),
                                   max(8, int(round(h / scale)))),
                             interpolation=cv2.INTER_CUBIC)
        if abs(rotation) > 1e-6:
            h, w = out.shape
            m = cv2.getRotationMatrix2D((w / 2, h / 2), -rotation, 1.0)
            out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REFLECT)
        return out
