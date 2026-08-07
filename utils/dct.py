from scipy.fft import dct, idct
import numpy as np

def dct2(block: np.ndarray) -> np.ndarray:
    """
    Computes the 2-dimensional Discrete Cosine Transform (DCT-II) of an input
    image block.

    The transform is performed by applying a 1D DCT along the rows followed by
    a 1D DCT along the columns, which is mathematically equivalent to a 2D DCT.
    The input is first converted to float32 to ensure numerical precision during
    the transform.

    Orthogonal normalization ('ortho') is used so that the transform is
    energy-preserving and perfectly invertible using the corresponding inverse
    DCT (IDCT) with the same normalization.

    Args:
        block: A 2D NumPy array representing the input image block
            (e.g., an 8×8 DCT block or a DWT subband).

    Returns:
        A 2D NumPy array of the same shape containing the DCT coefficients.
        The coefficient at (0, 0) is the DC component, while the remaining
        coefficients represent increasing horizontal and vertical spatial
        frequencies (AC components).
    """
    return dct(
        dct(block.astype(np.float32), axis=0, norm="ortho"),
        axis=1,
        norm="ortho"
    )

def idct2(block: np.ndarray) -> np.ndarray:
    return idct(
        idct(block, axis=0, norm="ortho"),
        axis=1,
        norm="ortho"
    )