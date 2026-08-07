import logging

import cv2
import numpy as np
import torch

from lightglue import DISK, LightGlue
from lightglue.utils import numpy_image_to_torch, rbd

logger = logging.getLogger(__name__)


def warp_patch(
    wmk_y: np.ndarray, x: float, y: float, homography: np.ndarray, patch_size: int
) -> np.ndarray | None:
    """
    Warp-extract the patch from ``wmk_y`` at position ``(x, y)`` in the
    original frame using ``homography``.

    Returns a ``(patch_size x patch_size)`` uint8 array, or ``None`` if the
    projected patch falls outside the watermarked frame.
    """
    h, w = wmk_y.shape[:2]
    half = patch_size // 2
    corners = np.array([
        [x - half, y - half],
        [x + half, y - half],
        [x + half, y + half],
        [x - half, y + half],
    ], dtype=np.float32)
    proj = cv2.perspectiveTransform(corners[None], homography)[0]
    if (
        proj[:, 0].min() < 0 or proj[:, 1].min() < 0
        or proj[:, 0].max() >= w or proj[:, 1].max() >= h
    ):
        return None
    dst = np.array(
        [[0, 0], [patch_size, 0], [patch_size, patch_size], [0, patch_size]],
        dtype=np.float32,
    )
    patch = cv2.warpPerspective(
        wmk_y,
        cv2.getPerspectiveTransform(proj.astype(np.float32), dst),
        (patch_size, patch_size),
    )
    return patch if patch.shape == (patch_size, patch_size) else None


class LightGluePatchMatcher:
    """
    DISK + LightGlue feature matcher used as an alternative to SIFT/BFMatcher
    for feature-based watermark detection.

    The models are loaded once per instance. Detection computes a single
    homography between the original and watermarked Y frame
    (``compute_homography``) and then warp-extracts the watermark patch around
    each original keypoint against that homography using the module-level
    ``warp_patch`` function.
    """

    def __init__(self, device: str | None = None):
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"LightGluePatchMatcher using device: {self._device}")
        self._extractor, self._matcher = self._load_matcher()

    def _load_matcher(self):
        extractor = DISK(max_num_keypoints=1024).eval().to(self._device)
        matcher = LightGlue(features='disk').eval().to(self._device)
        return extractor, matcher

    def _validate_homography(
        self,
        homography: np.ndarray,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        mask: np.ndarray,
        max_error: float = 5.0,
        min_inliers: int = 20,
    ) -> bool:
        if homography is None:
            return False
        pts_h = cv2.convertPointsToHomogeneous(mkpts0)[:, 0, :]
        proj = (homography @ pts_h.T).T
        proj = proj[:, :2] / proj[:, 2:]
        inliers = mask.ravel() == 1
        if inliers.sum() < min_inliers:
            return False
        return float(np.mean(np.linalg.norm(proj - mkpts1, axis=1)[inliers])) < max_error

    def compute_homography(
        self, orig_y: np.ndarray, wmk_y: np.ndarray
    ) -> np.ndarray | None:
        """
        Compute the homography mapping points in ``orig_y`` to ``wmk_y`` using
        DISK features matched by LightGlue. Returns the 3x3 homography matrix,
        or ``None`` if too few matches are found or the homography is rejected.
        """
        img0 = numpy_image_to_torch(orig_y).to(self._device)
        img1 = numpy_image_to_torch(wmk_y).to(self._device)

        with torch.no_grad():
            feats0 = self._extractor.extract(img0)
            feats1 = self._extractor.extract(img1)
            result = self._matcher({"image0": feats0, "image1": feats1})

        feats0 = rbd(feats0)
        feats1 = rbd(feats1)
        result = rbd(result)

        kpts0 = feats0["keypoints"].cpu().numpy()
        kpts1 = feats1["keypoints"].cpu().numpy()
        matches = result["matches"]

        if len(matches) < 10:
            logger.debug(f"LightGlue: too few matches ({len(matches)}), skipping frame")
            return None

        idx0 = matches[:, 0].cpu().numpy()
        idx1 = matches[:, 1].cpu().numpy()
        mkpts0 = kpts0[idx0]
        mkpts1 = kpts1[idx1]

        homography, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
        if not self._validate_homography(homography, mkpts0, mkpts1, mask):
            logger.debug("LightGlue: homography rejected by validation")
            return None
        return homography