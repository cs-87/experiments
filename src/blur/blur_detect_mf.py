"""
Matched-filter read-out of the blur watermark.

blur_detect.py compares *energies*: `loss = 1 - imp_hf/org_hf`, pooled over a bit's
frames. Energy is quadratic, so noise power adds into the numerator whether or not the
mark is there -- and on the measured clips that bias is not small: 49.2% of frames read
`imp_hf > org_hf` on the patch that was never touched, and 38.6% of frames carry less
than 1e4 units of HF energy in the left cell at all, so the denominator is near zero
exactly where the bias is worst. Bits landing on dark or soft scenes came out wrong with
high confidence.

Detection is non-blind, so the residual the embedder would create is known exactly:
blur_region() nulls every DCT coefficient beyond RADIUS, so the residual it leaves is
the original's own spectrum in that stop band. This module scores the *linear*
projection of the observed residual `org - imp` onto that template. Noise uncorrelated
with the original then contributes zero mean instead of a positive bias, which is the
whole difference.

The pipeline, stage by stage:

  0. resync   -- optional homography (LightGlue, sparse schedule) plus per-frame
                 sub-pixel translation, so a camcorded frame is measured in the
                 original's own sampling grid before any DCT.
  1. per-cell -- whitened matched filter over an annulus just outside the cutoff.
  2. pool     -- K cells per side, inverse-variance weighted. Cells sit on different
                 content, so unlike more frames of one scene their errors decorrelate.
  3. calibrate-- per-frame null mean and spread taken from never-marked control cells in
                 the same frame, turning the raw score into a standardised one.
  4. aggregate-- robust (Huber/Tukey) sum over each bit's frames, so one blown frame
                 cannot own a bit. Never a hard gate: dropping the lowest-energy 75% of
                 frames measurably collapsed BCR to 21/32 where soft weighting held.
  5. phase    -- a presence statistic separate from the payload, plus an optional
                 search over cyclic phases of the frame->bit map. The search is off by
                 default and does not do what it was meant to: on a clip that is a whole
                 number of payload cycles a phase shift only relabels the bit groups, so
                 every phase scores identically and the payload is recoverable just up
                 to rotation. See decode() -- this needs a sync word to fix, not a
                 better search.

The frame pass produces one standardised number per frame and nothing else; every
decision after that is arithmetic on those arrays. That is deliberate -- it means the
phase search, the choice of robust psi, and the frames-to-32-bits curve all come out of
a single decode of the video.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.fft import dctn

from src.blur.blur import RADIUS, TEMP_REDUNDANCY
from src.blur.mapping import bit_index_for_frame
from src.blur.patch import CLUSTER_K, SQUARE_SIZE, middle_pair_cells
from utils.bit import BIT_LENGTH, get_integer

EPS = 1e-9

# Threads scipy may use for the batched DCT. Set to 1 when several sweep configurations
# are being run as separate processes: eight of these each grabbing eight threads
# oversubscribes the box and measured 4.8x slower per frame than running alone.
DCT_WORKERS = int(os.environ.get("BLUR_DCT_WORKERS", "4"))


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MFConfig:
    """
    Everything the read-out depends on, in one value that can be printed next to a
    result. `radius`, `cluster_k`, `temp_redundancy` and `interleave` must match the
    embedder; the rest are detector-side choices the sweep is free to vary.
    """

    radius: int = RADIUS

    # Width of the annulus just outside the cutoff that carries the template. None
    # measures the whole stop band out to the corner. Band-limiting matters because the
    # far corner coefficients are pure noise once a camera has been in the path -- at
    # RADIUS=200 the 200..224 annulus alone already outscored the full band.
    band_width: int | None = 48

    # Width of the pass-band annulus just *inside* the cutoff, used as this cell's own
    # noise reference. The mark never touches it by construction, so its residual is
    # pure codec/capture noise measured on this cell's own content -- which is what a
    # control cell somewhere else in the frame cannot give.
    guard_width: int = 48

    # Radial bin width for the whitening profile. The camera's MTF and the codec's
    # quantiser both roll off smoothly with frequency, so noise is estimated per ring
    # rather than as one number for the band.
    ring_width: int = 8

    cluster_k: int = CLUSTER_K
    # Never-marked pairs, taken from the same ordering as the cluster so the two cannot
    # drift apart. They carry the frame's null distribution.
    control_pairs: int = 8

    square_size: int = SQUARE_SIZE
    whiten: bool = True
    match_moments: bool = True

    bit_length: int = BIT_LENGTH
    temp_redundancy: int = TEMP_REDUNDANCY
    interleave: bool = False

    def band_plan(self) -> "BandPlan":
        return BandPlan.get(self.square_size, self.radius, self.band_width,
                            self.guard_width, self.ring_width)


class BandPlan:
    """
    Which DCT coefficients the template lives on, cached per geometry.

    Rebuilt per frame this would dominate the frame cost, and the geometry only depends
    on (patch size, radius, widths) -- a handful of distinct values across a whole sweep.
    """

    _cache: dict = {}

    @classmethod
    def get(cls, size, radius, band_width, guard_width, ring_width):
        key = (size, radius, band_width, guard_width, ring_width)
        plan = cls._cache.get(key)
        if plan is None:
            plan = cls(*key)
            cls._cache[key] = plan
        return plan

    def __init__(self, size, radius, band_width, guard_width, ring_width):
        yy, xx = np.ogrid[:size, :size]
        # Distance from the DC corner, matching blur.get_blur_mask exactly -- the
        # template is only the residual of that mask if both use the same metric.
        dist = np.sqrt(xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2)

        outer = math.inf if band_width is None else radius + band_width
        band = (dist > radius) & (dist <= outer)
        guard = (dist > max(0.0, radius - guard_width)) & (dist <= radius)

        self.size = size
        self.radius = radius
        self.band_idx = np.flatnonzero(band.ravel())
        self.guard_idx = np.flatnonzero(guard.ravel())
        # The whole stop band, out to the corner: what blur.get_blur_mask(lower=False)
        # selects and therefore what blur_detect's energy statistic sums over. Carried
        # here so the baseline can be reproduced on exactly the frames, alignment and
        # patches the matched filter saw -- an A/B where the two detectors disagree
        # about which frames they read is not an A/B.
        self.stop_idx = np.flatnonzero((dist > radius).ravel())

        rings = ((dist[band] - radius) / max(1, ring_width)).astype(np.int64)
        self.ring = np.clip(rings, 0, None)
        self.n_rings = int(self.ring.max()) + 1 if self.ring.size else 1
        self.m = int(self.band_idx.size)
        self.ring_sel = [np.flatnonzero(self.ring == k) for k in range(self.n_rings)]

        if self.m == 0:
            raise ValueError(
                f"radius {radius} leaves no coefficients in a {size}x{size} patch")
        if self.guard_idx.size == 0:
            raise ValueError(f"guard band is empty at radius {radius}")


# ---------------------------------------------------------------------------
# stage 0 -- resync
# ---------------------------------------------------------------------------


class GeometricAligner:
    """
    Brings a leak frame back into the original's sampling grid.

    Two levels, because they move at different rates. A camcord's viewpoint is fixed for
    the session and only jittered per frame -- capture_sim's own docstring says so, and a
    real handheld capture behaves the same way -- so the homography is estimated on a
    sparse schedule and reused, while the residual handheld shake is chased every frame
    by phase correlation.

    The shake is the part that cannot be skipped. At RADIUS=110 on a 240-wide patch the
    template's finest components have a period of about 4 pixels, so the 1.5-3 px jitter
    of the "moderate" and "severe" conditions is most of a cycle: left unfixed it
    decorrelates the matched filter completely, and no amount of averaging recovers it.

    align() returns the leak frame resampled into the original's coordinates, so every
    caller downstream reads plain slices at the original's own patch coordinates.
    """

    def __init__(self, lightglue=False, every=30, refine=True, device=None,
                 max_shift_px=12.0, refine_window=768):
        self.lightglue = lightglue
        self.every = max(1, int(every))
        self.refine = refine
        self.max_shift_px = max_shift_px
        self.refine_window = refine_window

        self._matcher = None
        self._device = device
        self._hann_cache = {}
        self._H = None            # original -> leak, or None for "no warp needed"
        self._last_update = None
        self.n_homographies = 0
        self.n_rejected = 0

    def _get_matcher(self):
        if self._matcher is None:
            from utils.lightglue import LightGluePatchMatcher
            self._matcher = LightGluePatchMatcher(device=self._device)
        return self._matcher

    def _base_transform(self, org_y, imp_y):
        """Scale-only map when the leak was merely resized, identity when it was not."""
        if imp_y.shape == org_y.shape:
            return None
        sx = imp_y.shape[1] / org_y.shape[1]
        sy = imp_y.shape[0] / org_y.shape[0]
        return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float64)

    def _hann(self, shape):
        h, w = shape
        key = (h, w)
        win = self._hann_cache.get(key)
        if win is None:
            # Without a window the window's own hard edges dominate the cross-power
            # spectrum and the peak lands on zero shift whatever the content did.
            win = cv2.createHanningWindow((w, h), cv2.CV_32F)
            self._hann_cache[key] = win
        return win

    def _refine_shift(self, org_y, warped):
        """
        Residual translation between the original and the warped leak, sub-pixel.

        Measured on one window around frame centre rather than per cell: the shake is a
        whole-frame motion, and one phaseCorrelate is a great deal cheaper than twenty.
        """
        h, w = org_y.shape
        side = min(self.refine_window, h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        a = org_y[y0:y0 + side, x0:x0 + side].astype(np.float32)
        b = warped[y0:y0 + side, x0:x0 + side].astype(np.float32)

        # A frame that is flat, or a warp that blacked the window out, gives phase
        # correlation nothing to lock onto and it returns noise.
        if a.std() < 1e-3 or b.std() < 1e-3:
            return None

        win = self._hann(a.shape)
        (dx, dy), response = cv2.phaseCorrelate(a, b, win)
        if not np.isfinite(dx) or not np.isfinite(dy):
            return None
        if math.hypot(dx, dy) > self.max_shift_px:
            # Bigger than handheld jitter: this is a failed homography, not shake, and
            # trusting it would drag the patches somewhere arbitrary.
            return None
        return float(dx), float(dy), float(response)

    def align(self, org_y, imp_y, frame_index):
        """Leak frame resampled into the original's coordinate frame."""
        h, w = org_y.shape

        if self.lightglue and (self._last_update is None
                               or frame_index - self._last_update >= self.every):
            H = self._get_matcher().compute_homography(org_y, imp_y)
            self._last_update = frame_index
            if H is not None:
                self._H = H.astype(np.float64)
                self.n_homographies += 1
            else:
                self.n_rejected += 1
                # Keep the previous session homography rather than falling back to
                # identity: a stale viewpoint is much closer than no viewpoint.
                if self._H is None:
                    self._H = self._base_transform(org_y, imp_y)
        elif self._H is None:
            self._H = self._base_transform(org_y, imp_y)

        if self._H is None:
            warped = imp_y
        else:
            warped = cv2.warpPerspective(
                imp_y, np.linalg.inv(self._H), (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        if not self.refine:
            return warped

        shift = self._refine_shift(org_y, warped)
        if shift is None:
            return warped
        dx, dy, _ = shift
        if math.hypot(dx, dy) < 0.02:
            return warped

        # phaseCorrelate(a, b) reports how far b's content already sits from a's, so the
        # correction is its negative. Verified against synthetic shifts: applying +dx
        # doubled the misalignment instead of removing it.
        M = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float64)
        return cv2.warpAffine(warped, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# stages 1-3 -- per-frame statistic
# ---------------------------------------------------------------------------


def _abs_sd(x):
    """Robust scale of a zero-mean vector. Median-based, so a few outliers cannot move it."""
    if x.size == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(x)))


def _mad_sd(x):
    """Robust scale of a vector whose centre is unknown."""
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return 1.4826 * float(np.median(np.abs(x - med)))


def _batch_dct2(stack):
    """
    2D DCT-II of every patch in an (n, size, size) stack, flattened to (n, size*size).

    scipy rather than the cv2.dct in utils.dct because this is the frame pass's whole
    cost: forty separate cv2.dct calls per frame measured 25.7 ms where one batched
    scipy call over the same forty measured 3.5 ms. The two agree to float32 precision
    (max coefficient difference 0.006 on a 30641-magnitude spectrum), and both are the
    orthonormal DCT-II, so the template still lands on the same coefficient grid that
    blur.get_blur_mask carves the cutoff out of.
    """
    return dctn(stack, type=2, norm="ortho", axes=(1, 2),
                workers=DCT_WORKERS).reshape(stack.shape[0], -1)


def _cell_scores(template, residual, sigma_vec):
    """
    (nc, z, projection, template norm) for one whitened cell -- one inner product read
    three ways, because which reading is right is not obvious a priori and the sweep
    should decide it.

      proj/dn**2 -> `a`, the least-squares estimate of what fraction of the stop band
                    went missing. Directly interpretable, and measured to be so: on
                    inputs/30.mp4 at RADIUS=110 the marked cell reads a median 0.96
                    -- blur_region removes essentially all of it -- against 0.31 on the
                    untouched cell, which is the codec's own attenuation of the same
                    band. Its null variance is 1/dn**2, so a cell with little removable
                    energy is an imprecise measurement rather than a loud wrong one, and
                    it is the only one of the three with no content-dependent bias: the
                    two sides are compared on a quantity that does not scale with how
                    much high-frequency energy each side happened to hold.
      proj/dn    -> `z`, the same thing divided by its own standard error instead:
                    unit-variance under the null, expectation dn when marked. Simpler,
                    but z_left - z_right is a*dn differenced across two cells with
                    different dn, so it carries a content term that the per-frame
                    centring can only partly remove.
      proj/dn/rn -> `nc`, additionally divided by the residual's own norm. Gain
                    invariant and the best of the three before whitening, but a codec
                    attenuates the whole high band roughly proportionally, so the
                    residual on an *untouched* cell already points along the template
                    (that 0.31 above) and nc runs towards saturation on both sides at
                    once.

    All three are carried through the frame pass because separating them costs one
    inner product and choosing between them from a single clip would be guessing.
    """
    dw = template / sigma_vec
    rw = residual / sigma_vec
    dn = float(np.linalg.norm(dw))
    rn = float(np.linalg.norm(rw))
    proj = float(np.dot(dw, rw))
    nc = proj / (dn * rn + EPS)
    z = proj / (dn + EPS)
    return nc, z, proj, dn


def _cell_arrays(org_y, imp_y, pairs, plan, cfg, energy_cells):
    """
    Template, residual, noise level and stop-band energies for every cell of the frame,
    as (n_cells, ...) arrays laid out left, right, left, right ... pair by pair.
    Energies are returned only for `energy_cells`, in that order.

    Everything is done across all cells at once. Per cell it was the frame pass's entire
    cost; batched it is a fifth of it, and none of the arithmetic changes.

    Moment matching -- which absorbs the gamma, exposure and white balance of a camera
    capture, all affine on the patch and all otherwise producing a residual proportional
    to the original, i.e. correlated with the template itself, the one kind of noise a
    matched filter cannot ignore -- is applied in the DCT domain. An affine map of the
    pixels is a scalar gain on every coefficient plus an offset that lands only on DC,
    and DC is in neither the band nor the guard, so one gain per cell does it exactly.
    That is also what leaves the *raw* impaired spectrum in hand for the energy
    baseline, which does not moment match and must not be quietly improved.
    """
    size = cfg.square_size
    n = 2 * len(pairs)
    org_stack = np.empty((n, size, size), dtype=np.float32)
    imp_stack = np.empty((n, size, size), dtype=np.float32)

    for p, pair in enumerate(pairs):
        for side, (patch, y, x) in enumerate(pair):
            org_stack[2 * p + side] = patch
            imp_stack[2 * p + side] = imp_y[y[0]:y[1], x[0]:x[1]]

    O = _batch_dct2(org_stack)
    I = _batch_dct2(imp_stack)

    if cfg.match_moments:
        so = org_stack.reshape(n, -1).std(axis=1)
        sp = imp_stack.reshape(n, -1).std(axis=1)
        gain = np.where(sp > 1e-6, so / np.maximum(sp, 1e-6), 1.0)[:, None]
    else:
        gain = np.ones((n, 1), dtype=np.float32)

    template = O[:, plan.band_idx]
    residual = template - gain * I[:, plan.band_idx]
    guard = O[:, plan.guard_idx] - gain * I[:, plan.guard_idx]

    # Robust per-cell noise level, read off the pass band just inside the cutoff where
    # the mark cannot reach. Its median is zero by construction, so the absolute median
    # is the scale directly.
    noise = np.maximum(1.4826 * np.median(np.abs(guard), axis=1), 1e-6)

    # Only the cluster cells: the energy baseline never looks at controls, and the stop
    # band runs out to the corner, so it is five times as many coefficients as the band.
    stop = np.ix_(energy_cells, plan.stop_idx)
    e_org = np.sum(O[stop].astype(np.float64) ** 2, axis=1)
    e_imp = np.sum(I[stop].astype(np.float64) ** 2, axis=1)

    return template, residual, noise, e_org, e_imp


def frame_record(org_y, imp_y, cfg, plan=None):
    """
    One standardised number per statistic for this frame, or None if unmeasurable.

    Returns a dict carrying, for each statistic, the raw left-minus-right score `s`, the
    null centre `mu` and spread `sigma` estimated from this frame's own control cells,
    and the standardised `u = (s - mu) / sigma`. Positive u means the left cell lost the
    energy, i.e. this frame is voting 1.

    The null being re-estimated per frame is what makes a dead frame cost nothing: with
    no removable energy the score is noise, the control cells say so, and u lands near
    zero on its own. Nothing is gated out -- gating measurably collapsed BCR where soft
    weighting held it.
    """
    plan = plan or cfg.band_plan()

    if imp_y.shape != org_y.shape:
        return None

    pairs = middle_pair_cells(
        org_y, count=cfg.cluster_k + cfg.control_pairs,
        square_size=cfg.square_size)
    if len(pairs) < cfg.cluster_k + 2:
        # Fewer than two control pairs leaves no spread to calibrate against.
        return None

    k = cfg.cluster_k
    left = np.arange(0, 2 * len(pairs), 2)
    right = left + 1
    marked_l, marked_r = left[:k], right[:k]
    ctrl_l, ctrl_r = left[k:], right[k:]

    template, residual, noise, e_org, e_imp = _cell_arrays(
        org_y, imp_y, pairs, plan, cfg, np.arange(2 * k))

    # Whitening profile: the *shape* of the noise across frequency, pooled over the
    # control cells with each one first divided by its own level so a busy control does
    # not dominate a flat one. The level is then put back per cell, from that cell's own
    # guard band -- shape from the frame, scale from the cell.
    if cfg.whiten:
        ctrl = np.concatenate([ctrl_l, ctrl_r])
        absn = np.abs(residual[ctrl] / noise[ctrl, None])
        shape = np.ones(plan.n_rings, dtype=np.float64)
        for ring, sel in enumerate(plan.ring_sel):
            if sel.size * ctrl.size >= 8:
                shape[ring] = max(1.4826 * float(np.median(absn[:, sel])), 1e-6)
        profile = shape[plan.ring]
    else:
        profile = np.ones(plan.m, dtype=np.float64)

    sigma = profile[None, :] * noise[:, None]
    dw = template / sigma
    rw = residual / sigma
    dn = np.linalg.norm(dw, axis=1)
    rn = np.linalg.norm(rw, axis=1)
    proj = np.einsum("ij,ij->i", dw, rw)

    dn2 = dn * dn
    nc = proj / (dn * rn + EPS)
    z = proj / (dn + EPS)
    a = proj / np.maximum(dn2, EPS)

    # -- stage 2: pool the cluster ------------------------------------------------
    def pool(idx):
        w = dn2[idx]
        if w.sum() <= EPS:
            w = np.ones_like(w)
        nc_pooled = float(np.dot(w, nc[idx]) / w.sum())
        # Effective count under the inverse-variance weights: K when the cells carry
        # equal evidence, and correctly less when one cell dominates.
        k_eff = float(w.sum() ** 2 / np.dot(w, w))
        # Unit-variance under the null by construction, so the difference of two sides
        # has null variance 2 whatever K is.
        z_pooled = float(np.sum(z[idx]) / math.sqrt(idx.size))
        # Inverse-variance combination of the per-cell attenuation estimates is just
        # the ratio of the sums, and its variance is the reciprocal of the total.
        total_dn2 = float(np.sum(dn2[idx]))
        a_pooled = float(np.sum(proj[idx])) / max(total_dn2, EPS)
        return nc_pooled, z_pooled, a_pooled, total_dn2, k_eff

    nc_l, z_l, a_l, dn2_l, keff_l = pool(marked_l)
    nc_r, z_r, a_r, dn2_r, keff_r = pool(marked_r)

    # -- stage 3: calibrate against this frame's own controls ---------------------
    ctrl_nc = nc[ctrl_l] - nc[ctrl_r]
    ctrl_z = z[ctrl_l] - z[ctrl_r]
    # Control cells sit on their own content, so their attenuation estimates have their
    # own precisions. Standardising each control pair by its own theoretical spread
    # before pooling is what lets a handful of them calibrate a cluster whose precision
    # is different again -- what comes out is a unitless inflation factor, "how much
    # noisier than the whitened model says", and that is what transfers.
    ctrl_var = 1.0 / np.maximum(dn2[ctrl_l], EPS) + 1.0 / np.maximum(dn2[ctrl_r], EPS)
    ctrl_a_std = (a[ctrl_l] - a[ctrl_r]) / np.sqrt(ctrl_var)

    # A control pair is one cell against one cell; the cluster is k_eff against k_eff.
    # sd of a single-cell score is the pair spread over sqrt(2).
    sd_nc1 = max(_mad_sd(ctrl_nc), 1e-4) / math.sqrt(2.0)
    sigma_nc = sd_nc1 * math.sqrt(1.0 / max(keff_l, 1e-6) + 1.0 / max(keff_r, 1e-6))
    mu_nc = float(np.median(ctrl_nc))

    # z is already standardised per cell and the pooling preserves that, so the pair
    # spread applies to the cluster difference unchanged.
    sigma_z = max(_mad_sd(ctrl_z), 1e-3)
    mu_z = float(np.median(ctrl_z))

    s_a = a_l - a_r
    sigma_a_model = math.sqrt(1.0 / max(dn2_l, EPS) + 1.0 / max(dn2_r, EPS))
    # Floored, not free: with only a handful of control pairs the MAD can come back
    # implausibly small on a quiet frame, and dividing by it would hand that frame
    # unbounded confidence.
    inflation = max(_mad_sd(ctrl_a_std), 0.5)
    sigma_a = sigma_a_model * inflation
    mu_a = float(np.median(ctrl_a_std)) * sigma_a_model

    s_nc = nc_l - nc_r
    s_z = z_l - z_r

    return {
        "s_nc": s_nc, "mu_nc": mu_nc, "sigma_nc": sigma_nc,
        "u_nc": (s_nc - mu_nc) / max(sigma_nc, EPS),
        "s_z": s_z, "mu_z": mu_z, "sigma_z": sigma_z,
        "u_z": (s_z - mu_z) / max(sigma_z, EPS),
        "s_a": s_a, "mu_a": mu_a, "sigma_a": sigma_a,
        "u_a": (s_a - mu_a) / max(sigma_a, EPS),
        "a_left": a_l, "a_right": a_r,
        "dn_left": math.sqrt(dn2_l), "dn_right": math.sqrt(dn2_r),
        "e_org_left": float(e_org[marked_l].sum()),
        "e_imp_left": float(e_imp[marked_l].sum()),
        "e_org_right": float(e_org[marked_r].sum()),
        "e_imp_right": float(e_imp[marked_r].sum()),
        "n_controls": int(ctrl_l.size),
    }


# ---------------------------------------------------------------------------
# the frame pass
# ---------------------------------------------------------------------------


def iter_y(video_path, limit=None, stride=1):
    """Luma planes of a video, in order."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        i = 0
        while limit is None or i < limit:
            ok, frame = cap.read()
            if not ok:
                return
            if i % stride == 0:
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]
            i += 1
    finally:
        cap.release()


def scan(org_frames, imp_frames, cfg, aligner=None, frame_indices=None,
         progress=False, total=None):
    """
    Walk the two videos once and return the per-frame statistics as numpy arrays.

    Everything after this point -- phase search, robust aggregation, the
    frames-to-32-bits curve, the presence test -- is arithmetic on these arrays, so a
    whole sweep of decoding choices costs one decode of the video.

    `frame_indices` supplies the original-frame index of each pair when the caller has
    resynchronised temporally (a real camcord drops and repeats frames, so the loop
    counter is not the frame number). Left None the loop counter is used, which is
    correct for anything frame-synchronous.
    """
    plan = cfg.band_plan()

    fields = ("frame",
              "s_nc", "mu_nc", "sigma_nc", "u_nc",
              "s_z", "mu_z", "sigma_z", "u_z",
              "s_a", "mu_a", "sigma_a", "u_a", "a_left", "a_right",
              "dn_left", "dn_right",
              "e_org_left", "e_imp_left", "e_org_right", "e_imp_right")
    rows = {k: [] for k in fields}
    skipped = 0

    pairs = zip(org_frames, imp_frames)
    if progress:
        from tqdm import tqdm
        pairs = tqdm(pairs, total=total, unit="frame", desc="matched filter")

    for i, (org_y, imp_y) in enumerate(pairs):
        index = i if frame_indices is None else frame_indices[i]

        if aligner is not None:
            imp_y = aligner.align(org_y, imp_y, index)
        elif imp_y.shape != org_y.shape:
            imp_y = cv2.resize(imp_y, (org_y.shape[1], org_y.shape[0]),
                               interpolation=cv2.INTER_CUBIC)

        rec = frame_record(org_y, imp_y, cfg, plan)
        if rec is None:
            skipped += 1
            continue

        rows["frame"].append(index)
        for k in fields[1:]:
            rows[k].append(rec[k])

    out = {k: np.asarray(v, dtype=(np.int64 if k == "frame" else np.float64))
           for k, v in rows.items()}
    out["skipped"] = skipped
    out["config"] = cfg
    return out


# ---------------------------------------------------------------------------
# stages 4-5 -- aggregation, phase search, decision
# ---------------------------------------------------------------------------


def robust_scale(u):
    """
    Typical magnitude of the per-frame evidence, as the yardstick psi() clips against.

    Not the null spread. `u` is already standardised so that the *null* has unit
    variance, and on a marked clip almost every frame is far outside that -- clipping at
    a fixed multiple of 1.0 therefore flattened every frame to the same value and turned
    the soft aggregation into a hard majority vote. Measured: with the fixed threshold,
    per-bit evidence pinned at the clip ceiling for 30 of 32 bits and the phase search
    lost its grip because every phase scored alike.

    So the yardstick has to be the observed spread of the statistic on this clip.
    Floored at 1.0 so that on an unmarked video -- where median|u| is about 0.67 -- the
    scale does not shrink and inflate pure noise into apparent evidence.
    """
    if u.size == 0:
        return 1.0
    return max(float(np.median(np.abs(u))), 1.0)


def psi(u, kind="huber", c=3.0, scale=None):
    """
    Influence function bounding what one frame can contribute, in units of `scale`.

    A cut, a lost homography or a blown highlight produces a single enormous |u|; left
    unbounded it decides the bit on its own. Huber keeps the linear -- and therefore
    still soft -- weighting through the bulk and only clips the tail; Tukey additionally
    pushes the far tail back to zero. "none" is the plain sum, kept so the cost of
    robustifying is visible rather than assumed, and "sign" is the hard majority vote
    the pooling exists to beat.
    """
    if kind == "none":
        return u
    if kind == "sign":
        return np.sign(u)

    s = robust_scale(u) if scale is None else scale
    v = u / s
    if kind == "huber":
        return s * np.clip(v, -c, c)
    if kind == "tukey":
        w = np.clip(1.0 - (v / c) ** 2, 0.0, None) ** 2
        return s * v * w
    raise ValueError(f"unknown psi {kind!r}")


def _bit_indices(frames, cfg, phase):
    return np.array(
        [bit_index_for_frame(int(f), cfg.temp_redundancy, cfg.bit_length,
                             interleave=cfg.interleave, phase=phase)
         for f in frames], dtype=np.int64)


def aggregate(scan_result, phase=0, stat="a", kind="huber", c=3.0,
              weight="unit", cfg=None, upto=None):
    """
    Per-bit evidence L_b, and the per-bit z-score that makes it comparable across runs.

    `weight="llr"` divides each frame's contribution by its own null spread, which is
    the inverse-variance combination the calibration in stage 3 licenses. `weight="unit"`
    sums the already-standardised scores. They differ only when the frame-to-frame noise
    level varies a lot, which is exactly when it is worth knowing which one is right.

    `upto` truncates to frames before that original-frame index, which is how the
    frames-to-32-bits curve is read off without decoding anything again.
    """
    cfg = cfg or scan_result["config"]
    frames = scan_result["frame"]
    u = scan_result[f"u_{stat}"]
    sigma = scan_result[f"sigma_{stat}"]

    if upto is not None:
        keep = frames < upto
        frames, u, sigma = frames[keep], u[keep], sigma[keep]

    if frames.size == 0:
        return {"L": np.zeros(cfg.bit_length), "z": np.zeros(cfg.bit_length),
                "counts": np.zeros(cfg.bit_length, dtype=np.int64)}

    contrib = psi(u, kind, c)
    if weight == "llr":
        contrib = contrib / np.maximum(sigma, EPS)
    elif weight != "unit":
        raise ValueError(f"unknown weight {weight!r}")

    b = _bit_indices(frames, cfg, phase)
    L = np.bincount(b, weights=contrib, minlength=cfg.bit_length)
    counts = np.bincount(b, minlength=cfg.bit_length)
    # Under the null each standardised frame score is unit-variance, so dividing by
    # sqrt(n) turns the sum into a z-score -- the form that means the same thing whether
    # a bit got 30 frames or 300, and the only form a presence threshold can be set on.
    z = L / np.sqrt(np.maximum(counts, 1))
    return {"L": L, "z": z, "counts": counts}


def phase_candidates(cfg):
    """
    The temporal offsets worth testing: every frame for the interleaved map, every run
    start for the contiguous one.

    Interleaving makes a residual offset catastrophic in a way contiguous runs are not:
    offset k cyclically permutes the entire payload, where a contiguous map only loses k
    frames at each run boundary.
    """
    if cfg.interleave:
        return list(range(cfg.bit_length))
    return list(range(0, cfg.bit_length * cfg.temp_redundancy, cfg.temp_redundancy))


def decode(scan_result, cfg=None, stat="a", kind="huber", c=3.0, weight="unit",
           search_phase=False, min_z=0.0, upto=None):
    """
    Turn the per-frame scores into a payload plus the confidence to argue about it.

    `sum_b |z_b|` is the presence statistic, and it answers "is this marked at all"
    separately from "what does it say" -- which the energy detector could not do at all.

    The phase search is off by default, and that default is load-bearing rather than
    cautious. This detector is non-blind: it holds the original and has already lined
    the leak up against it frame by frame, so the frame index is known and phase 0 is
    not a guess.

    Searching it anyway does not work here, for a reason worth stating plainly because
    it is a property of the scheme and not of the implementation. Whenever the clip is a
    whole number of payload cycles -- N = bit_length x TR, which is exactly how these
    clips are cut -- a phase shift is a pure relabelling of the bit groups. The same
    frames land in the same groups, only the group names rotate, so *every* phase
    produces the same multiset of |z_b| and the same sum. Measured: all 32 phases scored
    within 1e-9 of each other, and the winner was decided by floating-point summation
    order, turning a 32/32 read into 16/32. The payload is recoverable only up to cyclic
    rotation from the evidence alone; resolving it needs something the evidence does not
    contain -- a sync word in the payload, a partial cycle, or a known frame index.

    So ties resolve to the lowest phase, and `phase_margin` in the result reports how
    much the winner actually won by. A margin near zero means the phase was not
    determined by the data.
    """
    cfg = cfg or scan_result["config"]

    phases = phase_candidates(cfg) if search_phase else [0]
    scored = []
    for ph in phases:
        agg = aggregate(scan_result, phase=ph, stat=stat, kind=kind, c=c,
                        weight=weight, cfg=cfg, upto=upto)
        scored.append((float(np.sum(np.abs(agg["z"]))), ph, agg))

    best_score = max(s for s, _, _ in scored)
    # Strictly better by more than float noise, else the earliest phase wins. Without
    # this the argmax over a degenerate tie is whichever summation order rounded up.
    presence, phase, agg = next(
        item for item in scored if item[0] >= best_score * (1.0 - 1e-9))
    runner_up = max((s for s, ph, _ in scored if ph != phase), default=0.0)
    phase_margin = (presence - runner_up) / presence if presence > 0 else 0.0
    L, z, counts = agg["L"], agg["z"], agg["counts"]

    bits = []
    undecided = []
    for i in range(cfg.bit_length):
        if counts[i] == 0 or abs(z[i]) < min_z:
            bits.append("?")
            undecided.append(i)
        else:
            bits.append("1" if L[i] > 0 else "0")
    bit_string = "".join(bits)

    watermark = (get_integer(bit_string)
                 if not undecided and cfg.bit_length == BIT_LENGTH else 0)

    return {
        "watermark": watermark,
        "bit_string": bit_string,
        "phase": phase,
        "phase_margin": phase_margin,
        # Mean |z| per bit: ~0.8 when nothing is there (half-normal), several units when
        # it is. Scale-free, so one threshold covers clips of any length.
        "presence": presence / max(cfg.bit_length, 1),
        "presence_total": presence,
        "L": L,
        "bit_z": z,
        "counts": counts,
        "undecided": undecided,
        "frames_used": int(counts.sum()),
    }


def energy_decode(scan_result, cfg=None, phase=0, upto=None):
    """
    blur_detect's pooled-energy read-out, computed from the same frame pass.

    `1 - imp_hf/org_hf` per side, energies summed over a bit's frames and the ratio
    taken once, exactly as blur_detect.detect() does it -- reproduced here rather than
    called there so that the baseline and the matched filter are measured on identical
    frames, identical patches and identical alignment. A comparison where the two
    detectors read different pixels is a comparison of two experiments, not two
    detectors.

    Verified against blur_detect.detect() on a file-based run; see FINDINGS.md.
    """
    cfg = cfg or scan_result["config"]
    frames = scan_result["frame"]
    keep = slice(None) if upto is None else (frames < upto)
    frames = frames[keep]
    if frames.size == 0:
        return {"bit_string": "?" * cfg.bit_length,
                "margins": np.zeros(cfg.bit_length), "counts": np.zeros(cfg.bit_length)}

    b = _bit_indices(frames, cfg, phase)
    pools = {}
    for name in ("e_org_left", "e_imp_left", "e_org_right", "e_imp_right"):
        pools[name] = np.bincount(b, weights=scan_result[name][keep],
                                  minlength=cfg.bit_length)
    counts = np.bincount(b, minlength=cfg.bit_length)

    one_loss = 1.0 - pools["e_imp_left"] / (pools["e_org_left"] + 1e-9)
    zero_loss = 1.0 - pools["e_imp_right"] / (pools["e_org_right"] + 1e-9)
    margins = one_loss - zero_loss

    bits = "".join("?" if counts[i] == 0 else ("1" if margins[i] > 0 else "0")
                   for i in range(cfg.bit_length))
    return {"bit_string": bits, "margins": margins, "counts": counts,
            "one_loss": one_loss, "zero_loss": zero_loss}


def hard_bits(L):
    """Best guess for every position, with no '?' -- for scoring BCR against a payload."""
    return "".join("1" if value > 0 else "0" for value in L)


def bit_correct_rate(expected, recovered, bit_length=BIT_LENGTH):
    """Number of bit positions that came back correct, out of bit_length."""
    expected_bits = format(expected, f"0{bit_length}b")
    return sum(1 for a, b in zip(expected_bits, recovered) if a == b)


def frame_accuracy(scan_result, expected, cfg=None, stat="a", phase=0):
    """
    Fraction of frames whose own sign already agrees with the bit it carries.

    Reported separately from BCR because the two answer different questions: BCR says
    whether the payload survived, per-frame accuracy says how much margin there was, and
    a run can hold 32/32 on 60% frame accuracy or lose bits at 95% depending entirely on
    how the errors clump (finding F4).
    """
    cfg = cfg or scan_result["config"]
    frames = scan_result["frame"]
    if frames.size == 0:
        return float("nan")
    u = scan_result[f"u_{stat}"]
    expected_bits = format(expected, f"0{cfg.bit_length}b")
    b = _bit_indices(frames, cfg, phase)
    want = np.array([1 if expected_bits[i] == "1" else -1 for i in b])
    return float(np.mean(np.sign(u) == want))


def frames_to_payload(scan_result, expected, cfg=None, targets=None, **decode_kw):
    """
    Smallest frame prefix that recovers all 32 bits, and the BCR curve behind it.

    The answer to "as few frames as possible": every point comes from the same single
    decode, so the curve costs nothing beyond the arithmetic.
    """
    cfg = cfg or scan_result["config"]
    frames = scan_result["frame"]
    if frames.size == 0:
        return {"first_full": None, "curve": []}

    last = int(frames.max()) + 1
    if targets is None:
        step = max(cfg.bit_length, last // 20)
        targets = list(range(step, last + step, step))
        if targets[-1] < last:
            targets.append(last)

    curve = []
    first_full = None
    for n in targets:
        res = decode(scan_result, cfg=cfg, upto=n, **decode_kw)
        bcr = bit_correct_rate(expected, hard_bits(res["L"]), cfg.bit_length)
        curve.append((int(n), bcr, res["presence"]))
        if first_full is None and bcr == cfg.bit_length:
            first_full = int(n)
    return {"first_full": first_full, "curve": curve}


# ---------------------------------------------------------------------------
# convenience
# ---------------------------------------------------------------------------


def detect_mf(org_video_path, imp_video_path, cfg=None, aligner=None,
              max_frames=None, stride=1, progress=True, verbose=True, **decode_kw):
    """Read the payload straight from two files, for use outside the sweep harness."""
    cfg = cfg or MFConfig()
    org = iter_y(org_video_path, limit=max_frames, stride=stride)
    imp = iter_y(imp_video_path, limit=max_frames, stride=stride)

    frame_indices = None
    if stride > 1:
        n = (max_frames or int(cv2.VideoCapture(str(org_video_path))
                               .get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_indices = list(range(0, n, stride))

    result = scan(org, imp, cfg, aligner=aligner, frame_indices=frame_indices,
                  progress=progress, total=max_frames)
    decoded = decode(result, cfg=cfg, **decode_kw)
    decoded["scan"] = result

    if verbose:
        print(f"recovered 0x{decoded['watermark']:08X}  ({decoded['bit_string']})")
        print(f"phase {decoded['phase']}, presence {decoded['presence']:.2f}, "
              f"{decoded['frames_used']} frames used, {result['skipped']} skipped")
    return decoded
