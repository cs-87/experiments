"""
Detect whether an image region has been intentionally pixelated.

Pixelation here means: downsample a region to 1/N resolution, then upsample back
with cv2.INTER_NEAREST (see src/pixelate_patch.py:pixelate_region). The result is a
grid of NxN blocks of constant colour. Every metric below is a different way of
measuring "is there a constant-colour grid of period N in this region".

The one idea running through all five metrics is **phase referencing**: we never ask
"is this region flat?" (real footage has flat regions too), we ask "is this region
flat *in a pattern that repeats every N pixels*?". Natural content has no reason to
align its detail to a fixed grid, so the boundary-vs-interior contrast is what
separates pixelation from calm scenery.

numpy + OpenCV only, no ML. Visualisation lives in utils/pixelation_viz.py so this
module stays importable without matplotlib.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

# Block sizes tried by estimate_block_size() when the caller doesn't know it.
CANDIDATE_BLOCK_SIZES = (2, 4, 8, 16, 32)

# Weight per metric in the final confidence. Gradient concentration and block variance
# carry most of the decision because they measured with by far the widest margin on
# real encoded footage. The other three are corroborating: they point the right way,
# but a lossy codec's own macroblocking pushes all of them partway toward "pixelated",
# so weighting them heavily would turn ordinary compression into a false positive.
DEFAULT_WEIGHTS = {
    "gradient": 0.35,
    "variance": 0.35,
    "laplacian": 0.10,
    "fft": 0.08,
    "neighbor": 0.12,
}

# Every raw->score mapping in one place, so the detector is tunable without reading
# the code. Each entry is the (lo, hi) of a linear ramp: raw at lo scores 0, raw at hi
# scores 1; hi < lo means lower-is-more-pixelated.
#
# Calibrated on 1080p footage, scoring at a fixed block size, against three references:
# a genuinely pixelated half, a clean half of the SAME mp4v-encoded file (codec
# blocking only), and a clean half of the unencoded source. The middle case is the one
# that sets the floor -- an 8x8 DCT codec produces real block structure, and the
# thresholds below are placed so that codec blocking alone stays near zero.
THRESHOLDS = {
    # boundary/interior gradient ratio: ~1.1 clean, ~1.8 codec blocking, ~9.4 pixelated
    "gradient_ratio": (2.00, 6.00),
    # intra-N / intra-2N variance: ~0.39 clean, ~0.38 codec, ~0.02 pixelated (inverted)
    "variance_ratio": (0.35, 0.05),
    # boundary/interior |Laplacian| ratio: ~1.06 clean, ~1.26 codec, ~1.53 pixelated
    "laplacian_ratio": (1.30, 2.20),
    # grid harmonic prominence: ~3-6 clean, but ~22 for codec blocking -- weak cue
    "fft_peak": (10.0, 30.0),
    # intra-block minus cross-boundary flatness: ~0.02 clean, ~0.17 codec, ~0.19 pixelated
    "neighbor_excess": (0.10, 0.35),
    # a neighbouring pair within this many grey levels counts as "identical"
    "flat_tolerance": 1.5,
}


@dataclass
class PixelationResult:
    """Final confidence plus every intermediate, as required for debugging/tuning."""

    confidence: float
    block_size: int
    scores: dict = field(default_factory=dict)   # per-metric, each in [0, 1]
    raw: dict = field(default_factory=dict)      # pre-normalisation measurements

    def __repr__(self):
        parts = " ".join(f"{k}={v:.3f}" for k, v in self.scores.items())
        return (
            f"<PixelationResult confidence={self.confidence:.3f} "
            f"block={self.block_size} {parts}>"
        )


# --------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------

def _to_gray(image):
    """Accept a BGR frame, a single-channel image, or a raw Y plane -> float32 2D."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            arr = arr[:, :, 0]
    return arr.astype(np.float32)


def _squash(value, lo, hi):
    """
    Linear ramp from lo->0 to hi->1, clipped to [0, 1].

    hi < lo is allowed and means "lower raw value is more pixelated" (used by the
    variance metric, where pixelation *reduces* the measurement).
    """
    if hi == lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _crop_to_blocks(gray, block_size):
    """Trim to a whole number of blocks so reshape-based block ops are exact."""
    h, w = gray.shape
    h -= h % block_size
    w -= w % block_size
    if h < block_size or w < block_size:
        return None
    return gray[:h, :w]


def _gradient_profiles(gray):
    """
    Mean absolute first difference along each axis, as 1D profiles.

    Collapsing a 2D gradient field to two 1D profiles is what makes the small-block
    case tractable: averaging over a whole 1080-pixel column integrates the tiny
    per-pixel step of a 2px grid into a clearly measurable peak, while averaging away
    content that isn't grid-aligned.

    Returns (column_profile, row_profile). Index i of the column profile is the mean
    gradient between column i and i+1, i.e. the strength of the vertical edge sitting
    at offset i.
    """
    col_profile = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    row_profile = np.abs(np.diff(gray, axis=0)).mean(axis=1)
    return col_profile, row_profile


def _fold(profile, block_size):
    """
    Fold a 1D profile onto one block period: result[k] = mean energy at offset k.

    For a pixelated region every block boundary lands on the same offset, so one entry
    of the folded profile towers over the rest. For natural content the profile is
    roughly uniform. This is the phase-referencing step the whole module rests on.
    """
    n = (len(profile) // block_size) * block_size
    if n < block_size:
        return None
    return profile[:n].reshape(-1, block_size).mean(axis=0)


def _boundary_interior_ratio(profile, block_size):
    """
    Ratio of the folded profile's boundary offset to the mean of its interior offsets.

    ~1.0 means no grid. Large means energy is concentrated on a regular lattice.
    block_size == 1 has no interior, so it can never be pixelated -> ratio 1.
    """
    folded = _fold(profile, block_size)
    if folded is None or block_size < 2:
        return 1.0
    # diff index b-1 is the step between the last pixel of one block and the first of
    # the next, i.e. the block boundary.
    boundary = folded[block_size - 1]
    interior = np.delete(folded, block_size - 1).mean()
    return float(boundary / (interior + 1e-6))


# --------------------------------------------------------------------------------
# block size estimation
# --------------------------------------------------------------------------------

def estimate_block_size(image, candidates=CANDIDATE_BLOCK_SIZES):
    """
    Guess the pixelation block size by picking the period with the strongest
    boundary/interior gradient contrast.

    A true grid of period N also produces (weaker) contrast at multiples of N, so ties
    are broken toward the smallest candidate: scanning ascending and requiring a clear
    improvement keeps 2 from being reported as 4.
    """
    gray = _to_gray(image)
    col_profile, row_profile = _gradient_profiles(gray)

    ratios = {}
    for size in sorted(candidates):
        ratios[size] = 0.5 * (
            _boundary_interior_ratio(col_profile, size)
            + _boundary_interior_ratio(row_profile, size)
        )

    # Pick the smallest period that gets within 10% of the best score. Testing a
    # period-N grid at period 2N still shows contrast (half the folded offsets are
    # boundaries), so a plain argmax can report a harmonic instead of the fundamental.
    best = max(ratios.values())
    return min(size for size, ratio in ratios.items() if ratio >= 0.9 * best)


# --------------------------------------------------------------------------------
# metric 1: block variance
# --------------------------------------------------------------------------------

def _mean_intra_block_variance(gray, block_size):
    cropped = _crop_to_blocks(gray, block_size)
    if cropped is None:
        return None
    h, w = cropped.shape
    blocks = cropped.reshape(h // block_size, block_size, w // block_size, block_size)
    return float(blocks.var(axis=(1, 3)).mean())


def block_variance_score(image, block_size):
    """
    Variance inside blocks of size N, normalised by variance inside blocks of size 2N.

    Pixelation makes each NxN block constant, so intra-block variance collapses toward
    zero -- but the normaliser matters enormously. Dividing by the variance of the
    *whole region* does not work: at N=2 the sub-2px detail of any natural 1080p frame
    is a tiny fraction of full-frame variance, so clean footage scores 0.004 and looks
    maximally pixelated. The comparison has to be against an adjacent spatial scale.

    Using 2N as the reference makes it a question about how detail grows with scale.
    Natural images accumulate variance smoothly, so the ratio sits well above zero.
    A pixelated region has ~0 variance at N but real variance at 2N (a 2Nx2N window
    spans four different blocks), so the ratio collapses.

    Score is inverted (low ratio -> high confidence) via a descending ramp.
    """
    gray = _to_gray(image)
    if block_size < 2:
        return 0.0, {"variance_ratio": 1.0, "intra_block_variance": 0.0}

    intra = _mean_intra_block_variance(gray, block_size)
    coarse = _mean_intra_block_variance(gray, block_size * 2)
    if intra is None or coarse is None:
        return 0.0, {"variance_ratio": 1.0, "intra_block_variance": 0.0}

    ratio = intra / (coarse + 1e-6)
    lo, hi = THRESHOLDS["variance_ratio"]
    return _squash(ratio, lo, hi), {
        "variance_ratio": ratio,
        "intra_block_variance": intra,
        "coarse_block_variance": coarse,
    }


def block_variance_map(image, block_size):
    """Per-block variance as a 2D array, for the heatmap visualisation."""
    gray = _to_gray(image)
    cropped = _crop_to_blocks(gray, block_size)
    if cropped is None:
        return np.zeros((1, 1), dtype=np.float32)
    h, w = cropped.shape
    blocks = cropped.reshape(h // block_size, block_size, w // block_size, block_size)
    return blocks.var(axis=(1, 3))


# --------------------------------------------------------------------------------
# metric 2: gradient concentration
# --------------------------------------------------------------------------------

def gradient_concentration_score(image, block_size):
    """
    Is gradient energy concentrated on regularly spaced vertical/horizontal lines?

    This is the strongest and most robust cue, because it is self-normalising: the
    boundary energy is measured against the interior energy of the *same* region, so
    scene brightness, contrast and compression level all cancel. Measured ~1.16 on
    clean 1080p footage and ~9.4 on the same footage pixelated at 4px.

    pixelated.md specifies Sobel here, but Sobel measured markedly worse and is not
    used: its 3-tap kernel smears a one-pixel block edge across three columns, which
    leaks boundary energy into the interior offsets and collapses exactly the contrast
    this metric depends on (9.4 -> 2.3 on the same region). A plain first difference
    is the sharper instrument at these block sizes. Sobel magnitude is still computed
    for the visualisation, where its thicker edges are easier to see.
    """
    gray = _to_gray(image)
    if block_size < 2:
        return 0.0, {"gradient_ratio": 1.0}

    col_profile, row_profile = _gradient_profiles(gray)

    col_ratio = _boundary_interior_ratio(col_profile, block_size)
    row_ratio = _boundary_interior_ratio(row_profile, block_size)
    ratio = 0.5 * (col_ratio + row_ratio)

    lo, hi = THRESHOLDS["gradient_ratio"]
    return _squash(ratio, lo, hi), {
        "gradient_ratio": ratio,
        "gradient_ratio_cols": col_ratio,
        "gradient_ratio_rows": row_ratio,
    }


# --------------------------------------------------------------------------------
# metric 3: laplacian energy
# --------------------------------------------------------------------------------

def laplacian_score(image, block_size):
    """
    Laplacian energy on block boundaries versus block interiors.

    Note this deliberately does NOT use total Laplacian energy as a "pixelated =
    smooth = low energy" cue. Measured on real footage, pixelating at 2px *raises*
    mean |Laplacian| (2.44 -> 3.25): the blocks are flat inside, but the grid adds a
    dense lattice of hard steps, and the steps win. Only the spatial distribution of
    the energy carries the signal, so we compare boundary rows/columns against
    interior rows/columns.
    """
    gray = _to_gray(image)
    if block_size < 2:
        return 0.0, {"laplacian_ratio": 1.0}

    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    col_ratio = _boundary_interior_ratio(lap.mean(axis=0), block_size)
    row_ratio = _boundary_interior_ratio(lap.mean(axis=1), block_size)
    ratio = 0.5 * (col_ratio + row_ratio)

    lo, hi = THRESHOLDS["laplacian_ratio"]
    return _squash(ratio, lo, hi), {
        "laplacian_ratio": ratio,
        "laplacian_mean": float(lap.mean()),
    }


# --------------------------------------------------------------------------------
# metric 4: frequency domain
# --------------------------------------------------------------------------------

def fft_score(image, block_size):
    """
    Look for the grid's periodic peak in the frequency domain.

    We take the 1D FFT of the gradient profiles rather than reading peaks off a raw 2D
    image FFT. At block_size 2 the grid's fundamental sits exactly at Nyquist, where
    H.264 has already discarded most energy and where a 2D spectrum is dominated by
    content; the profile FFT integrates one axis over the entire frame first, which
    lifts the surviving grid harmonic well clear of the noise floor.

    The peak is scored against the median of the surrounding spectrum, so a broadband
    (busy but non-periodic) region does not score.
    """
    gray = _to_gray(image)
    if block_size < 2:
        return 0.0, {"fft_peak": 1.0}

    col_profile, row_profile = _gradient_profiles(gray)

    peaks = []
    for profile in (col_profile, row_profile):
        n = len(profile)
        if n < 2 * block_size:
            continue
        spectrum = np.abs(np.fft.rfft(profile - profile.mean()))
        # Fundamental of a period-`block_size` grid, in rfft bin units.
        bin_index = int(round(n / block_size))
        if bin_index >= len(spectrum):
            bin_index = len(spectrum) - 1   # period 2 lands on the final (Nyquist) bin

        # Prominence against the LOCAL neighbourhood, not the global median. Gradient
        # profiles of real footage are strongly 1/f, so any individual bin sits an
        # order of magnitude above the global median and every clean region would
        # score as a grid. Comparing against nearby bins asks the right question:
        # does this exact frequency stand out from the ones surrounding it?
        window = max(8, bin_index // 8)
        lo_bin = max(1, bin_index - window)
        hi_bin = min(len(spectrum), bin_index + window + 1)
        neighbourhood = np.concatenate([
            spectrum[lo_bin:max(lo_bin, bin_index - 1)],
            spectrum[min(hi_bin, bin_index + 2):hi_bin],
        ])
        if neighbourhood.size == 0:
            continue
        floor = np.median(neighbourhood) + 1e-6
        peaks.append(float(spectrum[bin_index] / floor))

    ratio = float(np.mean(peaks)) if peaks else 1.0
    lo, hi = THRESHOLDS["fft_peak"]
    return _squash(ratio, lo, hi), {"fft_peak": ratio}


def fft_spectrum(image):
    """Centred log-magnitude 2D spectrum, for visualisation only."""
    gray = _to_gray(image)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(gray - gray.mean())))
    return np.log1p(spectrum)


# --------------------------------------------------------------------------------
# metric 5: neighbour similarity
# --------------------------------------------------------------------------------

def neighbor_similarity_score(image, block_size):
    """
    Excess flatness *inside* blocks over flatness *across* block boundaries.

    Counting near-identical neighbouring pixels on its own is not usable: on the clean
    source frame, 74% of horizontal neighbours in the right half are already identical
    versus 53% in the left half, purely because that side of the shot is calmer. Any
    absolute threshold inherits that content bias.

    Subtracting the cross-boundary flatness fixes it. Pixelation makes intra-block
    pairs identical while leaving boundary pairs as steps, so the difference is large.
    Natural content is equally flat at both offsets, so the difference is ~0 no matter
    how calm the scene.
    """
    gray = _to_gray(image)
    if block_size < 2:
        return 0.0, {"neighbor_excess": 0.0}

    tol = THRESHOLDS["flat_tolerance"]
    interior_flat, boundary_flat = [], []

    for axis in (0, 1):
        diff = np.abs(np.diff(gray, axis=axis))
        flat = diff < tol
        # Offset i of this diff is the pair (i, i+1); it straddles a block boundary
        # when i % block_size == block_size - 1.
        offsets = np.arange(flat.shape[axis]) % block_size
        is_boundary = offsets == (block_size - 1)

        if axis == 0:
            boundary_flat.append(flat[is_boundary].mean())
            interior_flat.append(flat[~is_boundary].mean())
        else:
            boundary_flat.append(flat[:, is_boundary].mean())
            interior_flat.append(flat[:, ~is_boundary].mean())

    interior = float(np.mean(interior_flat))
    boundary = float(np.mean(boundary_flat))
    excess = interior - boundary

    lo, hi = THRESHOLDS["neighbor_excess"]
    return _squash(excess, lo, hi), {
        "neighbor_excess": excess,
        "interior_flatness": interior,
        "boundary_flatness": boundary,
    }


# --------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------

def detect_pixelation(image, block_size=None, weights=None):
    """
    Score how likely `image` was pixelated by downsample + INTER_NEAREST upsample.

    image      : BGR frame, greyscale image, or raw Y plane.
    block_size : known grid period. Pass it when you know it (the embedder does) to
                 skip the search and gain accuracy; leave None to auto-detect.
    weights    : per-metric weights, defaults to DEFAULT_WEIGHTS.

    Returns a PixelationResult whose .confidence is in [0, 1].
    """
    gray = _to_gray(image)
    weights = DEFAULT_WEIGHTS if weights is None else weights

    if block_size is None:
        block_size = estimate_block_size(gray)

    variance, variance_raw = block_variance_score(gray, block_size)
    gradient, gradient_raw = gradient_concentration_score(gray, block_size)
    laplacian, laplacian_raw = laplacian_score(gray, block_size)
    fft, fft_raw = fft_score(gray, block_size)
    neighbor, neighbor_raw = neighbor_similarity_score(gray, block_size)

    scores = {
        "variance": variance,
        "gradient": gradient,
        "laplacian": laplacian,
        "fft": fft,
        "neighbor": neighbor,
    }

    # Weighted mean, divided by the weights actually used so a caller passing a
    # partial weight dict still gets a [0, 1] result.
    total_weight = sum(weights.get(name, 0.0) for name in scores)
    confidence = sum(weights.get(name, 0.0) * value for name, value in scores.items())
    confidence = confidence / total_weight if total_weight else 0.0

    raw = {}
    for part in (variance_raw, gradient_raw, laplacian_raw, fft_raw, neighbor_raw):
        raw.update(part)

    return PixelationResult(
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        block_size=block_size,
        scores=scores,
        raw=raw,
    )
