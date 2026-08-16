"""
Stage-2 verifier: geometric re-ranking of an SSCD shortlist with DISK + LightGlue.

Per implement_1.md, this stage extracts local features from every REFERENCE frame
exactly once and caches them; only the leak (query) frame's features are computed
per query. Homography is estimated via RANSAC only AFTER LightGlue matching, and
only to produce an inlier-count score -- there is no perspective-correction
preprocessing and no rectification of anything.

This is deliberately not a reuse of utils/lightglue.py's LightGluePatchMatcher:
that class recomputes DISK features on both images on every call, which is
exactly the per-candidate recomputation implement_1.md says to avoid here.
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import torch

from lightglue import DISK, LightGlue
from lightglue.utils import numpy_image_to_torch, rbd


def _to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


class DiskCache:
    """DISK features for a fixed list of frames, extracted once and kept on device."""

    def __init__(self, frames: Iterable[np.ndarray], extractor: DISK, device: str):
        self.device = device
        self.feats = []
        with torch.no_grad():
            for f in frames:
                img = numpy_image_to_torch(_to_gray(f)).to(device)
                self.feats.append(extractor.extract(img))

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, i: int):
        return self.feats[i]


def _match_and_ransac(feats_leak, feats_ref, matcher: LightGlue,
                      ransac_thresh: float = 5.0) -> dict:
    """
    Run LightGlue once and estimate a homography by RANSAC purely to produce
    scoring ingredients -- nothing is warped or rectified. This is the single
    per-(leak, candidate) computation every Stage-2 score derives from, so
    comparing scores never means rerunning LightGlue/RANSAC per score.

    Returns raw, undecided evidence: num_matches, num_inliers, the per-inlier
    reprojection error array, and the number of DISK keypoints detected in each
    frame (n_kpts0/n_kpts1) -- the "amount of available matching evidence" a
    score like match-density normalizes against. `homography_ok` is False only
    when RANSAC itself could not run or found no model; it does not encode any
    particular score's notion of "good enough."
    """
    f0, f1 = rbd(feats_leak), rbd(feats_ref)
    n_kpts0, n_kpts1 = int(f0["keypoints"].shape[0]), int(f1["keypoints"].shape[0])

    with torch.no_grad():
        result = matcher({"image0": feats_leak, "image1": feats_ref})
    matches = rbd(result)["matches"]
    num_matches = int(len(matches))

    empty = {"num_matches": num_matches, "num_inliers": 0,
             "inlier_errs": np.empty(0, dtype=np.float64),
             "n_kpts0": n_kpts0, "n_kpts1": n_kpts1, "homography_ok": False}
    if num_matches < 4:  # cv2.findHomography's hard minimum
        return empty

    kpts0 = f0["keypoints"].cpu().numpy()
    kpts1 = f1["keypoints"].cpu().numpy()
    idx0 = matches[:, 0].cpu().numpy()
    idx1 = matches[:, 1].cpu().numpy()
    mkpts0, mkpts1 = kpts0[idx0], kpts1[idx1]

    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransac_thresh)
    if H is None:
        return empty

    inliers = mask.ravel().astype(bool)
    pts_h = cv2.convertPointsToHomogeneous(mkpts0)[:, 0, :]
    proj = (H @ pts_h.T).T
    proj = proj[:, :2] / proj[:, 2:]
    inlier_errs = np.linalg.norm(proj - mkpts1, axis=1)[inliers]

    return {"num_matches": num_matches, "num_inliers": int(inliers.sum()),
            "inlier_errs": inlier_errs, "n_kpts0": n_kpts0, "n_kpts1": n_kpts1,
            "homography_ok": True}


def score_pair(feats_leak, feats_ref, matcher: LightGlue,
               min_matches: int = 10, ransac_thresh: float = 5.0) -> dict:
    """
    Baseline Stage-2 score: raw RANSAC inlier count. Thin wrapper around
    `_match_and_ransac` that reproduces the original return shape exactly, so
    the existing evaluate_stage2.py baseline is unaffected by this refactor.
    """
    raw = _match_and_ransac(feats_leak, feats_ref, matcher, ransac_thresh)
    if raw["num_matches"] < min_matches or not raw["homography_ok"]:
        return {"num_matches": raw["num_matches"], "num_inliers": 0,
                "mean_reproj_err": float("inf"), "ok": False}

    mean_err = float(np.mean(raw["inlier_errs"])) if raw["inlier_errs"].size else float("inf")
    return {"num_matches": raw["num_matches"], "num_inliers": raw["num_inliers"],
            "mean_reproj_err": mean_err,
            "ok": bool(raw["num_inliers"] >= min_matches)}


class Stage2Verifier:
    """
    Owns one DISK extractor + one LightGlue matcher and the reference DISK cache.

    verify_frame() extracts the leak frame's features ONCE, then matches against
    every requested candidate's cached features -- no additional cheap filtering
    is applied between the SSCD shortlist and this loop.
    """

    def __init__(self, ref_frames: Iterable[np.ndarray], device: str = "cuda",
                 max_keypoints: int = 1024, min_matches: int = 10,
                 ransac_thresh: float = 5.0):
        self.device = device
        self.min_matches = min_matches
        self.ransac_thresh = ransac_thresh
        self.extractor = DISK(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher = LightGlue(features="disk").eval().to(device)
        self.ref_cache = DiskCache(ref_frames, self.extractor, device)

    def verify_frame(self, leak_frame_bgr: np.ndarray,
                      candidate_ref_indices: np.ndarray) -> np.ndarray:
        """Returns an inlier-count score per candidate, same order as the input."""
        with torch.no_grad():
            leak_feats = self.extractor.extract(
                numpy_image_to_torch(_to_gray(leak_frame_bgr)).to(self.device))

        scores = np.zeros(len(candidate_ref_indices), dtype=np.float32)
        for i, ref_idx in enumerate(candidate_ref_indices):
            s = score_pair(leak_feats, self.ref_cache[int(ref_idx)], self.matcher,
                           self.min_matches, self.ransac_thresh)
            scores[i] = s["num_inliers"]
        return scores

    def verify_frame_raw(self, leak_frame_bgr: np.ndarray,
                          candidate_ref_indices: np.ndarray) -> list[dict]:
        """
        Same evidence as verify_frame(), exposed undecided: one raw match/RANSAC
        dict per candidate (see `_match_and_ransac`), computed once. Every
        candidate scoring method compares on this shared list rather than
        rerunning LightGlue/RANSAC per method.
        """
        with torch.no_grad():
            leak_feats = self.extractor.extract(
                numpy_image_to_torch(_to_gray(leak_frame_bgr)).to(self.device))

        return [_match_and_ransac(leak_feats, self.ref_cache[int(ref_idx)],
                                  self.matcher, self.ransac_thresh)
                for ref_idx in candidate_ref_indices]
