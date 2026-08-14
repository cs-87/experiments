import cv2
import numpy as np


def dct2(block: np.ndarray) -> np.ndarray:
    """
    Computes the 2D DCT using OpenCV, equivalent to cv2.dct(block).
    """
    return cv2.dct(block.astype(np.float32))


def idct2(block: np.ndarray) -> np.ndarray:
    """
    Computes the inverse 2D DCT using OpenCV.
    """
    return cv2.idct(block.astype(np.float32))
