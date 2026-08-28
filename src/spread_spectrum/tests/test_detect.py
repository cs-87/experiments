"""
End-to-end detection on synthetic hosts, plus the two controls that matter.

Everything here runs against a textured synthetic host rather than a video file, so
the suite stays fast and does not depend on which clips happen to be on disk. The
real-footage numbers live in FINDINGS.md.
"""

import cv2
import numpy as np

from src.spread_spectrum.detect.aggregate import EvidenceAggregator
from src.spread_spectrum.detect.decode import WatermarkHypothesisTester
from src.spread_spectrum.detect.evidence import PatchEvidence, selection_null
from src.spread_spectrum.detect.pipeline import Detector, DetectorConfig
from src.spread_spectrum.ids import CodewordSet
from src.spread_spectrum.tests.helpers import embed_into, mark_positions, synthetic_host
from utils.bit import get_bit_string

SHAPE = (512, 512)
IDS = CodewordSet.recommended(20_000)
WID = int(IDS.ids[7_777])
BITS, _ = get_bit_string(WID)
CFG = DetectorConfig(device="cpu", max_sites=12)


def _detector():
    return Detector(IDS, CFG)


def _frames(n, attack=None, marked=True, seed0=100):
    for k in range(n):
        host = synthetic_host(SHAPE, seed=seed0 + k)
        y = embed_into(host, BITS, mark_positions(SHAPE)) if marked else host
        yield k, (attack(y) if attack else y)


def test_no_attack_recovery():
    det = _detector()
    r, _ = det.detect_frames(_frames(3))
    assert r.accepted and r.watermark_id == WID, str(r)
    assert r.bits == BITS


def test_clean_host_returns_no_watermark():
    det = _detector()
    r, _ = det.detect_frames(_frames(3, marked=False))
    assert not r.accepted, f"false positive on unmarked host: {r}"


def test_pure_noise_returns_no_watermark():
    rng = np.random.default_rng(0)
    det = _detector()
    r, _ = det.detect_frames(
        (k, rng.uniform(0, 255, SHAPE)) for k in range(3))
    assert not r.accepted, f"false positive on noise: {r}"


def test_noisy_recovery():
    rng = np.random.default_rng(1)
    det = _detector()
    r, _ = det.detect_frames(
        _frames(4, attack=lambda y: y + rng.normal(0, 4.0, y.shape)))
    assert r.accepted and r.watermark_id == WID, str(r)


def test_survives_jpeg_and_mild_blur():
    def jpeg(y, q=70):
        ok, buf = cv2.imencode(".jpg", np.clip(y, 0, 255).astype(np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    for name, atk in [("jpeg70", jpeg),
                      ("blur0.8", lambda y: cv2.GaussianBlur(y, (0, 0), 0.8))]:
        r, _ = _detector().detect_frames(_frames(4, attack=atk))
        assert r.accepted and r.watermark_id == WID, f"{name}: {r}"


def test_a_different_payload_returns_that_payload():
    """
    The control a presence statistic passes and an identifier must not: a clip
    carrying a different valid ID has to come back as THAT ID, not as the one being
    looked for and not as NO_WATERMARK.
    """
    other = int(IDS.ids[111])
    other_bits, _ = get_bit_string(other)
    assert other != WID
    det = _detector()
    frames = []
    for k in range(3):
        host = synthetic_host(SHAPE, seed=200 + k)
        frames.append((k, embed_into(host, other_bits, mark_positions(SHAPE))))
    r, _ = det.detect_frames(frames)
    assert r.accepted and r.watermark_id == other, str(r)


def test_rank_aware_selection_null_suppresses_unmarked_sites():
    """
    Sites are chosen by maximising the statistic they are then judged on, so an
    unmarked clip must be suppressed by the rank-aware null rather than by luck.
    Compared on total weight rather than on S2: once every site is suppressed the
    weight is zero and S2 is computed from a unit-weight fallback that is not
    evidence of anything, which is exactly why total_weight gates acceptance.
    """
    out = {}
    for ra in (False, True):
        det = Detector(IDS, DetectorConfig(device="cpu", max_sites=12, rank_aware=ra))
        r, per_frame = det.detect_frames(_frames(3, marked=False))
        out[ra] = (r, sum(e.weight for ev in per_frame for e in ev))
    assert out[True][1] < 0.05 * out[False][1], (out[False][1], out[True][1])
    assert not out[True][0].accepted and not out[False][0].accepted


def test_marked_clip_still_clears_the_rank_aware_null():
    """The suppression above must not also suppress a real mark."""
    det = Detector(IDS, DetectorConfig(device="cpu", max_sites=12, rank_aware=True))
    r, per_frame = det.detect_frames(_frames(3))
    assert r.accepted and r.watermark_id == WID
    assert sum(e.weight for ev in per_frame for e in ev) > 0


def test_selection_null_grows_with_the_candidate_count():
    k = 32
    assert selection_null(0, 10, k) < selection_null(0, 1_000_000, k)
    assert selection_null(0, 1_000_000, k) > selection_null(50, 1_000_000, k) > k


def test_aggregation_variance_falls_as_one_over_root_m():
    """The whole redundancy argument is this law; if it does not hold, nothing does."""
    rng = np.random.default_rng(2)
    truth = np.where(rng.random(32) > 0.5, 1.0, -1.0) * 0.05
    ses = {}
    for m in (16, 64, 256, 1024):
        agg = EvidenceAggregator(huber_c=None)
        for _ in range(m):
            agg.add(PatchEvidence(c=truth + rng.normal(0, 0.5, 32), sigma=0.5,
                                  snr2=1.0, weight=1.0, energy=0.0, site=None))
        ses[m] = agg.result().se.mean()
    for a, b in zip((16, 64, 256), (64, 256, 1024)):
        ratio = ses[a] / ses[b]
        assert 1.7 < ratio < 2.3, f"se({a})/se({b}) = {ratio:.2f}, expected ~2"


def test_spatial_and_temporal_pooling_are_the_same_operation():
    """
    There is no temporal code, so how observations are grouped into frames cannot
    change the answer. This is the property that makes the detector immune to frame
    drop, duplication and reordering.
    """
    rng = np.random.default_rng(3)
    ev = [PatchEvidence(c=rng.normal(0, 1, 32), sigma=1.0, snr2=1.0,
                        weight=float(rng.uniform(0.5, 2)), energy=0.0, site=None)
          for _ in range(120)]
    a = EvidenceAggregator(huber_c=None); a.add(ev)
    b = EvidenceAggregator(huber_c=None)
    order = rng.permutation(len(ev))
    for i in order:
        b.add(ev[i])
    assert np.abs(a.result().c - b.result().c).max() < 1e-12
    assert np.abs(a.result().se - b.result().se).max() < 1e-12


def test_hypothesis_thresholds_tighten_with_the_codebook_size():
    """Paying for the multiplicity is the whole point of the S1 threshold."""
    small = WatermarkHypothesisTester(1_000, fpr=1e-6).s1_threshold
    big = WatermarkHypothesisTester(200_000, fpr=1e-6).s1_threshold
    assert big > small
    assert WatermarkHypothesisTester(200_000, fpr=1e-9).s1_threshold > big


def test_calibrated_thresholds_are_stricter_than_parametric():
    """
    The parametric null is optimistic and the calibration file records by how much.
    Measured: a nominal 1e-6 parametric threshold admitted 3 of 24 decoy-key runs and
    6 of 30 unmarked sweep cells. If a future change makes the calibrated threshold
    LOOSER than the parametric one, either the calibration is broken or the detector
    changed enough that it has to be redone -- both worth failing on.
    """
    import json
    import pathlib
    path = pathlib.Path("src/spread_spectrum/calibration.json")
    if not path.exists():
        return                                    # calibration not run in this tree
    rep = json.loads(path.read_text())
    for fpr, t in rep["unmarked"]["thresholds"].items():
        assert t["s1_empirical_gumbel"] > t["s1_parametric"], (fpr, t)
        assert t["s2_empirical_scaled_chi2"] > t["s2_parametric"], (fpr, t)
    tester = WatermarkHypothesisTester.from_calibration(path, 200_000, fpr=1e-6)
    assert tester.s1_threshold > WatermarkHypothesisTester(200_000, fpr=1e-6).s1_threshold


def test_geometry_search_anchors_to_identity_without_evidence():
    """
    The scale search maximises the evidence the decision is later made on, so left
    unanchored it invents detections: measured, an unmarked clip locked scale 1.1 and
    came out at S1 = 7.43, S2 = 91.4, over both acceptance thresholds. Anchored it
    reads S1 = 4.3.

    Here the objective returns pure noise, so no hypothesis deserves to win.
    """
    from src.spread_spectrum.detect.localise import GeometrySearch
    rng = np.random.default_rng(0)

    class _Ev:
        def __init__(self, v):
            self.snr2 = v

    # noise: every hypothesis scores about the same, none stands out
    flat = GeometrySearch(lambda luma: [_Ev(rng.uniform(0, 0.05)) for _ in range(8)])
    s, r, table = flat.estimate([synthetic_host((512, 512))])
    assert (s, r) == (1.0, 0.0), f"locked {s} on noise: {table[:3]}"

    # a genuine lock: evidence peaks sharply at one scale and nowhere else.
    # The objective only sees the warped frame, so it recovers the hypothesis from the
    # frame height -- warp() resizes by 1/scale.
    def peaked(luma, truth=0.5):
        est = 512.0 / luma.shape[0]
        return [_Ev(4.0 * np.exp(-((est - truth) / 0.02) ** 2)) for _ in range(8)]

    sharp = GeometrySearch(peaked, scales=(0.4, 0.5, 0.75, 1.0, 1.25))
    s2, _, table2 = sharp.estimate([synthetic_host((512, 512))])
    assert abs(s2 - 0.5) < 0.03, f"failed to lock a clear peak: {table2[:3]}"
