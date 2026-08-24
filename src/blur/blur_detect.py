"""
Non-blind read-out of the embedded bit.

Non-blind in the sense that it holds the original video, so unlike BLUR_IMPAIRMENT_VIEW
it never has to *search* for the mark. It replays the embedder's own patch selection on
the original frame to learn the only two places a mark could be -- the bit-1 candidate
in the first half, the bit-0 candidate in the second -- and then only has to decide
which of those two actually got blurred. Two patches scored per frame instead of a whole
grid, and no cliff heuristic: with the original in hand the question "how much of this
patch's high-frequency energy went missing" has an exact answer.

The measure is the same one the impairment view uses in its non-blind mode: the fraction
of a patch's own original HF energy that went missing. Being a fraction of *its own*
original, it stays directly comparable between the two candidates even when one sits on
sky and the other on foliage -- exactly what an absolute HF measure could not do.

Temporal redundancy is used as redundancy, not as an electorate. A bit is held still for
temp_redundancy frames, so those frames are repeated measurements of one quantity rather
than independent opinions: their energies are pooled and the ratio taken once, giving one
decision per bit. Deciding each frame and voting would discard the confidence of every
measurement before combining them.
"""

import cv2
import numpy as np
from tqdm import tqdm

import utils.dct as dct
from utils.bit import BIT_LENGTH, get_integer
from src.blur.blur import RADIUS, TEMP_REDUNDANCY, get_blur_mask, score_patch_for_watermark
from src.blur.patch import (
    get_best_patch_in_both_half,
    get_best_sift_patches_in_two_halves,
    get_half_patches_grid,
    get_middle_patches,
    select_candidate,
)

from utils.lightglue import LightGluePatchMatcher, warp_patch

# Default gate for the optional margin check. Both scores are fractions of removed HF
# energy, so their difference is on the same 0-1 scale: a blurred candidate against an
# untouched one separates by most of that range, while two untouched candidates sit on
# top of each other. Only used when detect() is asked to apply it -- passed as None the
# frame always votes, which is the right default when the temporal majority below is
# already doing the noise rejection.
MIN_MARGIN = 0

LIGHTGLUE = True  # use LightGlue to find the patch in the impaired frame and warp it back to the original shape


def hf_energy(org_patch, imp_patch, radius=RADIUS):
    """
    Raw high-frequency energy of the patch in both videos, as (org_hf, imp_hf).

    Returned unreduced rather than as a ratio because the caller pools these across
    every frame of a bit run before dividing once -- see pooled_loss().
    """
    org = org_patch.astype(np.float32)
    imp = imp_patch.astype(np.float32)

    mask = get_blur_mask(
        radius=radius, h=org.shape[0], w=org.shape[1], lower=False
    ).astype(bool)

    return float(np.sum(dct.dct2(org)[mask] ** 2)), float(np.sum(dct.dct2(imp)[mask] ** 2))


def pooled_loss(org_hf, imp_hf):
    """
    Fraction of original high-frequency energy that went missing, over pooled energies.

    1.0 means everything above the cutoff was removed -- what blur_region() does by
    construction. 0.0 means the band came through untouched. A plain re-encode nibbles at
    it and lands a little above zero, which is the noise floor the two candidates are
    compared against each other to reject.
    """
    return 1.0 - (imp_hf / (org_hf + 1e-9))  # 1e-9 to avoid zero division


def candidate_energy(candidate, imp_frame_y, radius=RADIUS, homography=None):
    """
    (org_hf, imp_hf) for one candidate, or None if it cannot be measured.

    The candidate carries the original patch and its coordinates, so the impaired side
    is a straight slice at the same place -- non-blind means never having to align.
    """
    if candidate is None:
        return None

    org_patch, y, x = candidate

    # get_patch signals "off frame" by handing back a (None, None, None) triple rather
    # than a bare None, so both shapes have to be caught here.
    if org_patch is None:
        return None

        if LIGHTGLUE and homography is not None:
            # The impaired frame may have been cropped or resized, so the original patch
            # coordinates may not land on the same content. Use LightGlue to find the
            # patch in the impaired frame and warp it back to the original shape.
            imp_patch = wrap_patch(
                imp_frame_y, x, y, homography, org_patch.shape[0])

    else:
        imp_patch = imp_frame_y[y[0]:y[1], x[0]:x[1]]

    # A frame boundary can clip the slice short even though the original patch was whole.
    # Measuring mismatched shapes would compare two different spectra and read as loss.
    if imp_patch.shape != org_patch.shape:
        return None

    return hf_energy(org_patch, imp_patch, radius=radius)


class RotatingCandidates:
    """
    Candidate pair for embedders that rotate the mark within a bit run.

    Mirrors blur_select_candidate.embed(): the same grid halves, the same scorer, the
    same per-run flush. One pool per half, because unlike the embedder this side does not
    know the bit -- so it walks both halves on every frame. Within a bit run the
    embedder marked one half on every frame, so that half's pool matches its history
    exactly while the other's is a harmless what-if; the per-run flush erases the
    divergence before it can reach the next run.
    """

    def __init__(self, temp_redundancy=TEMP_REDUNDANCY, key=score_patch_for_watermark):
        self.temp_redundancy = temp_redundancy
        self.key = key
        self.pool_one, self.pool_zero = [], []

    def __call__(self, frame_y, frame_index):
        if frame_index % self.temp_redundancy == 0:
            self.pool_one.clear()
            self.pool_zero.clear()

        first_halves, second_halves = get_half_patches_grid(frame_y)

        one = select_candidate(first_halves, self.pool_one, self.key)
        zero = select_candidate(second_halves, self.pool_zero, self.key)

        # Append the centres the embedder would have appended, so the next frame's walk
        # skips what this one used and the rotation stays in lockstep.
        if one is not None:
            self.pool_one.append(one[3])
        if zero is not None:
            self.pool_zero.append(zero[3])

        return (
            one[:3] if one is not None else None,
            zero[:3] if zero is not None else None,
        )


def middle_candidates(frame_y, frame_index):
    """Candidate pair for embedders that always mark the two cells at frame centre."""
    return get_middle_patches(frame_y)


def best_blur_candidates(frame_y, frame_index):
    """Candidate pair for embedders that pick each half's best-scoring grid cell."""
    return get_best_patch_in_both_half(frame_y, score_patch_for_watermark)


def sift_candidates(frame_y, frame_index):
    """Candidate pair for embedders that pick each half's strongest SIFT keypoint."""
    return get_best_sift_patches_in_two_halves(frame_y)


def _read_y(cap):
    """
    Luma plane of the next frame, or None at EOF.

    The cvtColor must not run before the ret check: at EOF read() hands back None and the
    conversion throws instead of ending the loop cleanly. Y rather than BGR2GRAY because
    Y is the plane the embedder actually blurred -- the two are close but not equal, and
    there is no reason to score in a domain the mark was not made in.
    """
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]


def detect(
    org_video_path,
    imp_video_path,
    temp_redundancy,
    candidate_fn,
    candidate_score=None,
    bit_length=BIT_LENGTH,
    min_margin=MIN_MARGIN,
    radius=RADIUS,
    verbose=True,
):
    """
    Recover the watermark from `imp_video_path`, given the original at `org_video_path`.

    candidate_fn(frame_y, frame_index) -> (bit_1_candidate, bit_0_candidate), each a
    (patch, y, x) triple or None. It must reproduce the embedder's selection exactly;
    the module ships one per embedder above. Defaults to RotatingCandidates, which is
    stateful, so a fresh instance is built per call rather than shared as a default arg.

    Temporal redundancy holds a bit still for `temp_redundancy` frames, so those frames
    are not independent opinions to be voted on -- they are repeated measurements of one
    quantity. They get pooled: the candidates' high-frequency energies are summed across
    the whole run and the loss ratio taken once at the end, yielding a single decision
    per bit. Pooling before dividing is what makes this stronger than a majority vote --
    see the accumulator comment below.

    min_margin is optional. Left None every bit is decided, however thin the evidence.
    Set it to MIN_MARGIN (or tighter) and bits whose pooled candidates sit too close
    together are listed in the result's "undecided" -- their best guess is still in the
    bit string, but you can see it was a guess. On an unmarked video that list fills up,
    which is the difference between "no watermark here" and a confident wrong answer.

    Returns a dict with the recovered integer, its bit string, and the per-bit pooled
    margins and frame counts -- the interesting part when a bit comes out wrong, since
    they separate "the two candidates tied" from "nothing was measurable here".
    """
    if candidate_fn is None:
        candidate_fn = RotatingCandidates(temp_redundancy=temp_redundancy)

    org_cap = cv2.VideoCapture(org_video_path)
    imp_cap = cv2.VideoCapture(imp_video_path)

    if not org_cap.isOpened():
        raise ValueError(f"could not open original video: {org_video_path}")
    if not imp_cap.isOpened():
        raise ValueError(f"could not open impaired video: {imp_video_path}")

    frame_count = int(org_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pooled HF energy per bit, one accumulator per candidate. Summing the raw energies
    # and dividing once at the end is deliberately not the same as averaging per-frame
    # ratios: it weights every frame by how much HF energy its patch actually carried,
    # so a flat patch -- whose ratio is mostly quantisation noise amplified by a tiny
    # denominator -- cannot shout down a textured one that measured the mark properly.
    # A hard per-frame vote is worse still, throwing that confidence away entirely
    # before combining.
    #
    # Indexed by bit position, not by run: a video longer than
    # bit_length * temp_redundancy wraps and gives some bits several runs, and since
    # every run of a given index carries the same bit they all pool together.
    one_org = [0.0] * bit_length
    one_imp = [0.0] * bit_length
    zero_org = [0.0] * bit_length
    zero_imp = [0.0] * bit_length
    pooled_frames = [0] * bit_length
    skipped = [0] * bit_length

    if LIGHTGLUE:
        matcher = LightGluePatchMatcher()
        homography = None

    try:
        for i in tqdm(range(frame_count), total=frame_count, unit="frame", desc="detecting"):
            org_y = _read_y(org_cap)
            if org_y is None:
                break

            imp_y = _read_y(imp_cap)
            if imp_y is None:
                break

            if LIGHTGLUE:
                homography = matcher.compute_homography(org_y, imp_y)

            # A re-encode at a different resolution still carries the mark, but the
            # candidate coordinates are in the original's frame of reference.
            else:
                if imp_y.shape != org_y.shape:
                    imp_y = cv2.resize(imp_y, (org_y.shape[1], org_y.shape[0]))

            bit_index = (i // temp_redundancy) % bit_length

            if candidate_score is None:
                one_candidate, zero_candidate = candidate_fn(org_y)
            else:
                one_candidate, zero_candidate = candidate_fn(
                    org_y, candidate_score)
            one_energy = candidate_energy(
                one_candidate, imp_y, radius=radius, homography=homography)
            zero_energy = candidate_energy(
                zero_candidate, imp_y, radius=radius, homography=homography)

            # Both sides or neither: pooling one candidate's frame without the other's
            # would bias the comparison the whole method rests on.
            if one_energy is None or zero_energy is None:
                skipped[bit_index] += 1
                continue

            one_org[bit_index] += one_energy[0]
            one_imp[bit_index] += one_energy[1]
            zero_org[bit_index] += zero_energy[0]
            zero_imp[bit_index] += zero_energy[1]
            pooled_frames[bit_index] += 1
    finally:
        org_cap.release()
        imp_cap.release()

    bits = []
    margins = []
    undecided = []

    for index in range(bit_length):
        if pooled_frames[index] == 0:
            # Nothing measurable landed on this bit at all. Emit a 0 to keep the string
            # the right length; "undecided" is where the caller sees it was not read.
            bits.append("?")
            margins.append(0.0)
            undecided.append(index)
            continue

        one_loss = pooled_loss(one_org[index], one_imp[index])
        zero_loss = pooled_loss(zero_org[index], zero_imp[index])
        margin = one_loss - zero_loss

        # reported as a percentage; min_margin gates the raw 0-1 value above
        margins.append(abs(margin) * 100)

        if abs(margin) < min_margin:
            undecided.append(index)
            bits.append("?")

        else:
            bits.append("1" if margin > 0 else "0")

    bit_string = "".join(bits)
    if len(undecided) == 0:
        watermark = get_integer(bit_string, bit_length)
    else:
        watermark = 0

    if verbose:
        print(f"recovered 0x{watermark:08X}  ({bit_string})")
        print(f"{sum(pooled_frames)} frames pooled, {sum(skipped)} skipped, "
              f"{len(undecided)}/{bit_length} bits undecided")

    return {
        "watermark": watermark,
        "bit_string": bit_string,
        "margins": margins,
        "pooled_frames": pooled_frames,
        "undecided": undecided,
        "skipped": skipped,
    }


def bit_correct_rate(expected, recovered, bit_length=BIT_LENGTH):
    """
    Percentage of bit positions that came back correct.

    Not utils.bit.get_bcr: that one writes `bcr=+1` where it means `bcr += 1`, so it
    reports 100/bit_length whenever any bit matches at all and 0 otherwise.
    """
    expected_bits = format(expected, f"0{bit_length}b")
    matched = sum(1 for a, b in zip(expected_bits, recovered) if a == b)
    return (matched * 100.0) / bit_length


def detect_multiple_patch(
    org_video_path,
    imp_video_path,
    temp_redundancy,
    bit_length=BIT_LENGTH,
    min_margin=MIN_MARGIN,
    radius=RADIUS,
    verbose=True,
):
    """
    Recover the watermark from `imp_video_path`, given the original at `org_video_path`.

    candidate_fn(frame_y, frame_index) -> (bit_1_candidate, bit_0_candidate), each a
    (patch, y, x) triple or None. It must reproduce the embedder's selection exactly;
    the module ships one per embedder above. Defaults to RotatingCandidates, which is
    stateful, so a fresh instance is built per call rather than shared as a default arg.

    Temporal redundancy holds a bit still for `temp_redundancy` frames, so those frames
    are not independent opinions to be voted on -- they are repeated measurements of one
    quantity. They get pooled: the candidates' high-frequency energies are summed across
    the whole run and the loss ratio taken once at the end, yielding a single decision
    per bit. Pooling before dividing is what makes this stronger than a majority vote --
    see the accumulator comment below.

    min_margin is optional. Left None every bit is decided, however thin the evidence.
    Set it to MIN_MARGIN (or tighter) and bits whose pooled candidates sit too close
    together are listed in the result's "undecided" -- their best guess is still in the
    bit string, but you can see it was a guess. On an unmarked video that list fills up,
    which is the difference between "no watermark here" and a confident wrong answer.

    Returns a dict with the recovered integer, its bit string, and the per-bit pooled
    margins and frame counts -- the interesting part when a bit comes out wrong, since
    they separate "the two candidates tied" from "nothing was measurable here".
    """

    org_cap = cv2.VideoCapture(org_video_path)
    imp_cap = cv2.VideoCapture(imp_video_path)

    if not org_cap.isOpened():
        raise ValueError(f"could not open original video: {org_video_path}")
    if not imp_cap.isOpened():
        raise ValueError(f"could not open impaired video: {imp_video_path}")

    frame_count = int(org_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pooled HF energy per bit, one accumulator per candidate. Summing the raw energies
    # and dividing once at the end is deliberately not the same as averaging per-frame
    # ratios: it weights every frame by how much HF energy its patch actually carried,
    # so a flat patch -- whose ratio is mostly quantisation noise amplified by a tiny
    # denominator -- cannot shout down a textured one that measured the mark properly.
    # A hard per-frame vote is worse still, throwing that confidence away entirely
    # before combining.
    #
    # Indexed by bit position, not by run: a video longer than
    # bit_length * temp_redundancy wraps and gives some bits several runs, and since
    # every run of a given index carries the same bit they all pool together.
    one_org = [0.0] * bit_length
    one_imp = [0.0] * bit_length
    zero_org = [0.0] * bit_length
    zero_imp = [0.0] * bit_length
    pooled_frames = [0] * bit_length
    skipped = [0] * bit_length

    if LIGHTGLUE:
        matcher = LightGluePatchMatcher()
        homography = None

    try:
        for i in tqdm(range(frame_count), total=frame_count, unit="frame", desc="detecting"):
            org_y = _read_y(org_cap)
            if org_y is None:
                break

            imp_y = _read_y(imp_cap)
            if imp_y is None:
                break

            if LIGHTGLUE:
                homography = matcher.compute_homography(org_y, imp_y)

            # A re-encode at a different resolution still carries the mark, but the
            # candidate coordinates are in the original's frame of reference.
            else:
                if imp_y.shape != org_y.shape:
                    imp_y = cv2.resize(imp_y, (org_y.shape[1], org_y.shape[0]))

            bit_index = (i // temp_redundancy) % bit_length

            # Patches must be located on the ORIGINAL frame. Taking them from imp_y
            # and then measuring imp_y at the same coordinates compares the impaired
            # frame with itself, so every loss is identically zero and the margin is
            # pure float rounding.
            first_halves, second_halves = get_half_patches_grid(org_y)

            def half_energy(halves):
                """(org_hf, imp_hf) summed over every measurable cell of one half."""
                org_hf = imp_hf = 0.0
                measured = 0
                for patch_details in halves:
                    energy = candidate_energy(
                        patch_details, imp_y, radius=radius)
                    if energy is None:
                        continue
                    org_hf += energy[0]
                    imp_hf += energy[1]
                    measured += 1
                return (org_hf, imp_hf) if measured else None

            one_energy = half_energy(first_halves)
            zero_energy = half_energy(second_halves)

            # Both sides or neither: pooling one half's frame without the other's
            # would bias the comparison the whole method rests on.
            if one_energy is None or zero_energy is None:
                skipped[bit_index] += 1
                continue

            # Accumulated, not overwritten: the whole point of temp_redundancy is that
            # every frame of a run adds into the same pool before the single divide.
            one_org[bit_index] += one_energy[0]
            one_imp[bit_index] += one_energy[1]
            zero_org[bit_index] += zero_energy[0]
            zero_imp[bit_index] += zero_energy[1]

            pooled_frames[bit_index] += 1
    finally:
        org_cap.release()
        imp_cap.release()

    bits = []
    margins = []
    undecided = []

    for index in range(bit_length):
        if pooled_frames[index] == 0:
            # Nothing measurable landed on this bit at all. Emit a 0 to keep the string
            # the right length; "undecided" is where the caller sees it was not read.
            bits.append("?")
            margins.append(0.0)
            undecided.append(index)
            continue

        one_loss = pooled_loss(one_org[index], one_imp[index])
        zero_loss = pooled_loss(zero_org[index], zero_imp[index])
        margin = one_loss - zero_loss

        # reported as a percentage; min_margin gates the raw 0-1 value above
        margins.append(abs(margin) * 100)

        if abs(margin) < min_margin:
            undecided.append(index)
            bits.append("?")

        else:
            bits.append("1" if margin > 0 else "0")

    bit_string = "".join(bits)
    if len(undecided) == 0:
        watermark = get_integer(bit_string, bit_length)
    else:
        watermark = 0

    if verbose:
        print(f"recovered 0x{watermark:08X}  ({bit_string})")
        print(f"{sum(pooled_frames)} frames pooled, {sum(skipped)} skipped, "
              f"{len(undecided)}/{bit_length} bits undecided")

    return {
        "watermark": watermark,
        "bit_string": bit_string,
        "margins": margins,
        "pooled_frames": pooled_frames,
        "undecided": undecided,
        "skipped": skipped,
    }


if __name__ == "__main__":
    INPUT = "4_sec_source.mp4"
    OUTPUT = "./out/wmk.mp4"

    # best_blur_candidates mirrors blur_recv_two_patches.py with BEST_BLUR. Swap in
    # RotatingCandidates(), middle_candidates or sift_candidates to match whichever
    # embedder wrote OUTPUT, and line temp_redundancy up with its own.
    result = detect(
        INPUT,
        OUTPUT,
        candidate_fn=best_blur_candidates,
        temp_redundancy=TEMP_REDUNDANCY,
    )

    # print(result)
