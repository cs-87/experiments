"""
End-to-end blind detector.

    video -> luma -> phase-shifted LL planes -> whiten -> dense PRN correlation
          -> localise on the codeword-independent energy
          -> per-site 32-D evidence + reliability
          -> inverse-variance, Huber-clipped pooling over patches and frames
          -> exhaustive codeword search -> two-statistic acceptance test

No original video, no SIFT replay, no temporal alignment. What the detector needs to
know is the PRN seed, the patch size and the DWT level -- everything in
DetectorConfig -- plus the list of valid IDs.
"""

import argparse
import dataclasses
import itertools
import sys
import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from src.spread_spectrum.ids import CodewordSet
from src.spread_spectrum.prn import BalancedPRNGenerator
from src.spread_spectrum.detect.aggregate import EvidenceAggregator
from src.spread_spectrum.detect.correlate import PRNCorrelator, ll_planes
from src.spread_spectrum.detect.decode import CodewordDecoder, WatermarkHypothesisTester
from src.spread_spectrum.detect.evidence import (PatchEvidenceExtractor,
                                                 PatchReliabilityEstimator)
from src.spread_spectrum.detect.localise import GeometrySearch, PatchLocaliser
from src.spread_spectrum.detect.whiten import PreWhitener


@dataclass
class DetectorConfig:
    seed: int = 8787
    square_size: int = 128
    level: int = 1
    whiten: str = "lm_lvn"
    mean_kernel: int = 3
    var_kernel: int = 5
    max_sites: int = 48
    nms_radius_px: int = None
    weighting: str = "wiener"
    rank_aware: bool = True
    huber_c: float = 3.0
    ring_inner: int = 3
    ring_outer: int = 14
    fpr: float = 1e-6
    calibration: str = None   # eval/calibrate.py output; strongly preferred
    split_half_min_frames: int = 4
    decoys: int = 0            # per-video null from N decoy keys; 0 = off
    decoy_z: float = 8.0
    device: str = None
    geometry: str = "none"      # "none" | "scale" | "scale+rotation"
    geometry_frames: int = 1

    @property
    def prn_side(self):
        return self.square_size >> self.level


def iter_luma(video_path, limit=None, stride=1):
    """Luma planes as float64. Yields (frame_index, plane)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video_path}")
    i = kept = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float64)
                kept += 1
                if limit is not None and kept >= limit:
                    break
            i += 1
    finally:
        cap.release()


class Detector:
    def __init__(self, codeword_set, config=None):
        self.cfg = config or DetectorConfig()
        self.codewords = codeword_set

        gen = BalancedPRNGenerator((self.cfg.prn_side, self.cfg.prn_side), self.cfg.seed)
        self.correlator = PRNCorrelator(gen.prns, device=self.cfg.device)
        self.whitener = PreWhitener(mode=self.cfg.whiten,
                                    mean_kernel=self.cfg.mean_kernel,
                                    var_kernel=self.cfg.var_kernel)
        self.localiser = PatchLocaliser(self.correlator, level=self.cfg.level,
                                        nms_radius_px=self.cfg.nms_radius_px,
                                        max_sites=self.cfg.max_sites)
        self.extractor = PatchEvidenceExtractor(
            self.correlator,
            reliability=PatchReliabilityEstimator(inner=self.cfg.ring_inner,
                                                  outer=self.cfg.ring_outer),
            weighting=self.cfg.weighting, rank_aware=self.cfg.rank_aware)
        self.geometry = None                 # (scale, rotation), locked on first use
        self.geometry_table = None
        self.decoder = CodewordDecoder(codeword_set)
        self.tester = (
            WatermarkHypothesisTester.from_calibration(
                self.cfg.calibration, len(codeword_set), fpr=self.cfg.fpr)
            if self.cfg.calibration
            else WatermarkHypothesisTester(len(codeword_set), fpr=self.cfg.fpr))
        self.tester.decoy_z_threshold = self.cfg.decoy_z

        # Decoy keys for the per-video null. Seeds are derived from the real one so a
        # given deployment always draws the same null and results are reproducible.
        self.decoys = []
        for i in range(self.cfg.decoys):
            d = dataclasses.replace(self.cfg, decoys=0,
                                    seed=self.cfg.seed * 7919 + 104729 * (i + 1))
            self.decoys.append(Detector(codeword_set, d))

    def __repr__(self):
        return (f"Detector({self.cfg.square_size}px/L{self.cfg.level}, "
                f"whiten={self.cfg.whiten}, {self.decoder!r})")

    def _search(self):
        rot = ((-1.0, -0.5, 0.0, 0.5, 1.0) if self.cfg.geometry == "scale+rotation"
               else (0.0,))
        cheap = PatchLocaliser(self.correlator, level=self.cfg.level,
                               nms_radius_px=self.cfg.nms_radius_px, max_sites=16)

        def evidence_fn(luma):
            planes = {ph: self.whitener(pl)
                      for ph, pl in ll_planes(luma, self.cfg.level).items()}
            en = cheap.energies(planes)
            return self.extractor.extract(planes, en,
                                          cheap.sites(planes, en, luma.shape))

        return GeometrySearch(evidence_fn, rotations=rot)

    def lock_geometry(self, frames, verbose=False):
        """
        Estimate the global scale/rotation once and hold it for the whole video.
        `frames` is a short list of luma planes. Returns (scale, rotation).
        """
        s, r, table = self._search().estimate(frames, verbose=verbose)
        self.geometry, self.geometry_table = (s, r), table
        return s, r

    def frame_evidence(self, luma, frame_index=-1):
        """All located sites of one frame, as weighted 32-D evidence."""
        if self.geometry is not None:
            luma = GeometrySearch.warp(luma, *self.geometry)
        planes = {ph: self.whitener(pl)
                  for ph, pl in ll_planes(luma, self.cfg.level).items()}
        energies = self.localiser.energies(planes)
        sites = self.localiser.sites(planes, energies=energies,
                                     frame_shape=luma.shape, frame_index=frame_index)
        return self.extractor.extract(planes, energies, sites)

    def detect_frames(self, frames, progress=False):
        """
        frames: iterable of (index, luma). Returns (DecodeResult, per-frame evidence).

        The evidence is returned as well as the verdict because every decoding
        variant -- a different weighting, a prefix of the frames, a different
        codebook -- is then cheap arithmetic on it, with no second pass over the
        video. One decode of a sweep cell should cost one video decode, not one per
        question asked of it.
        """
        agg = EvidenceAggregator(huber_c=self.cfg.huber_c)
        decoy_aggs = [EvidenceAggregator(huber_c=self.cfg.huber_c)
                      for _ in self.decoys]
        per_frame, t0 = [], time.time()
        frames = iter(frames)
        if self.cfg.geometry != "none" and self.geometry is None:
            head = [next(frames) for _ in range(self.cfg.geometry_frames)]
            s, r = self.lock_geometry([f for _, f in head], verbose=progress)
            if progress:
                print(f"geometry locked: scale {s:.4f} rotation {r:+.2f}")
            frames = itertools.chain(head, frames)
        for idx, luma in frames:
            ev = self.frame_evidence(luma, idx)
            per_frame.append(ev)
            agg.add(ev)
            # Same frame, same geometry, same everything but the key -- so the decoys
            # see whatever this clip's content does to the statistic. One pass over the
            # video, not one per key.
            for d, dagg in zip(self.decoys, decoy_aggs):
                d.geometry = self.geometry
                dagg.add(d.frame_evidence(luma, idx))
            if progress and sys.stdout.isatty():
                print(f"\rframe {idx}: {len(ev)} sites, {len(agg)} total", end="")
        if progress:
            pad = " " * 24 if sys.stdout.isatty() else ""
            print(f"\r{len(per_frame)} frames, {len(agg)} sites, "
                  f"{time.time() - t0:.1f}s{pad}")
        if len(agg) == 0:
            raise ValueError("no evidence found in any frame")
        result = self.decoder.decode(agg.result())
        if len(per_frame) >= self.cfg.split_half_min_frames:
            result.split_half_agrees = self.split_half_agrees(per_frame)
        if decoy_aggs:
            s1 = [d.decoder.decode(a.result()).s1
                  for d, a in zip(self.decoys, decoy_aggs) if len(a)]
            if len(s1) >= 3:
                med = float(np.median(s1))
                mad = 1.4826 * float(np.median(np.abs(np.asarray(s1) - med)))
                result.decoy_s1 = [round(v, 2) for v in s1]
                result.decoy_z = float((result.s1 - med) / max(mad, 1e-9))
        return self.tester.test(result), per_frame

    def split_half_agrees(self, per_frame):
        """
        Do two disjoint halves of the frames decode to the same ID?

        Split by frame PARITY, not into a first and second half: content drifts across
        a clip, and a temporal split would confound "different content" with
        "independent noise". Parity leaves both halves with the same content
        distribution and the same length, so a disagreement means the evidence is not
        actually pinning down one codeword.

        Costs nothing extra -- the per-frame evidence is already in hand, and each half
        is one more 200,000 x 32 GEMM.
        """
        halves = [EvidenceAggregator(huber_c=self.cfg.huber_c) for _ in range(2)]
        for i, ev in enumerate(per_frame):
            halves[i % 2].add(ev)
        if any(len(h) == 0 for h in halves):
            return None
        ids = []
        for h in halves:
            try:
                ids.append(self.decoder.decode(h.result()).watermark_id)
            except ValueError:
                return None
        return ids[0] == ids[1]

    def detect(self, video_path, max_frames=None, stride=1, progress=False):
        return self.detect_frames(iter_luma(video_path, max_frames, stride),
                                  progress=progress)

    def decode_curve(self, per_frame, targets=None):
        """
        Verdict after the first n frames, for n over a schedule. Reuses the stored
        evidence, so the whole acquisition curve costs no extra video decoding.
        """
        n = len(per_frame)
        targets = targets or sorted({1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, n}
                                    & set(range(1, n + 1)))
        out = []
        for t in targets:
            agg = EvidenceAggregator(huber_c=self.cfg.huber_c)
            for ev in per_frame[:t]:
                agg.add(ev)
            if len(agg) == 0:
                continue
            r = self.decoder.decode(agg.result())
            # The curve has to apply the same acceptance rule as detect_frames, or
            # frames-to-identification reports a frame count at which the detector
            # would not in fact have accepted.
            if t >= self.cfg.split_half_min_frames:
                r.split_half_agrees = self.split_half_agrees(per_frame[:t])
            out.append((t, self.tester.test(r)))
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Blind spread-spectrum watermark detector")
    ap.add_argument("video")
    ap.add_argument("--expect", type=lambda s: int(s, 0), default=None,
                    help="known payload, for reporting bit accuracy")
    ap.add_argument("--ids", default=None, help=".npy of valid IDs; default is the "
                                                "recommended extended-BCH set")
    ap.add_argument("--id-count", type=int, default=200_000)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=8787)
    ap.add_argument("--square-size", type=int, default=128)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--whiten", default="lm_lvn")
    ap.add_argument("--weighting", default="wiener")
    ap.add_argument("--max-sites", type=int, default=48)
    ap.add_argument("--fpr", type=float, default=1e-6)
    ap.add_argument("--calibration", default=None,
                    help="eval/calibrate.py JSON; use it -- the parametric "
                         "thresholds measured ~1 false positive in 5 cells")
    ap.add_argument("--decoys", type=int, default=0,
                    help="calibrate the null on THIS clip using N decoy keys. "
                         "Costs (1+N)x the correlation pass and is the only "
                         "control that adapts to the clip's own content")
    ap.add_argument("--device", default=None)
    ap.add_argument("--geometry", default="none",
                    choices=["none", "scale", "scale+rotation"])
    ap.add_argument("--curve", action="store_true", help="print the acquisition curve")
    args = ap.parse_args(argv)

    ids = (CodewordSet.load(args.ids) if args.ids
           else CodewordSet.recommended(args.id_count))
    if args.expect is not None and not ids.contains(args.expect):
        print(f"note: {args.expect:#010x} is not in the ID set, so it can never be "
              f"returned. Use --ids or pick a valid ID.")

    cfg = DetectorConfig(seed=args.seed, square_size=args.square_size,
                         level=args.level, whiten=args.whiten,
                         weighting=args.weighting, max_sites=args.max_sites,
                         fpr=args.fpr, device=args.device, geometry=args.geometry,
                         calibration=args.calibration,
                         decoys=args.decoys)
    det = Detector(ids, cfg)
    print(det)
    result, per_frame = det.detect(args.video, args.max_frames, args.stride,
                                   progress=True)
    print(result)
    if args.expect is not None:
        exp = format(args.expect, "032b")
        bcr = sum(a == b for a, b in zip(result.bits, exp))
        print(f"expected {args.expect:#010x}  bit accuracy {bcr}/32"
              f"{'  EXACT' if bcr == 32 else ''}")
    if args.curve:
        print("\nacquisition curve:")
        for n, r in det.decode_curve(per_frame):
            print(f"  {n:4d} frames: {r}")


if __name__ == "__main__":
    main()
