"""
Turning a 32-bit payload into the pattern that gets embedded, and back.

The bit convention lives here and nowhere else: bit '1' adds +P_i, bit '0' adds -P_i.
Embedder and detector both import it, because an inverted bit is not a degradation,
it is a different payload.
"""

import numpy as np

from src.spread_spectrum.codebook import generate_codebook


def signs_for_bit_string(bit_string):
    """'1' -> +1, '0' -> -1, as a float64 vector."""
    b = np.frombuffer(bit_string.encode("ascii"), dtype=np.uint8)
    if not np.isin(b, (ord("0"), ord("1"))).all():
        raise ValueError(f"bit string must be 0/1 only, got {bit_string!r}")
    return np.where(b == ord("1"), 1.0, -1.0)


def bit_string_for_signs(signs):
    """Inverse of signs_for_bit_string. Zero maps to '0', matching np.sign convention."""
    return "".join("1" if v > 0 else "0" for v in np.asarray(signs).ravel())


class BalancedPRNGenerator:

    def __init__(self, shape, seed):
        self.shape = tuple(shape)
        self.size = int(np.prod(shape))
        self.seed = seed

        self.codebook = generate_codebook(seed)

        if self.codebook.shape[1] != self.size:
            raise ValueError(
                f"codebook PRNs are length {self.codebook.shape[1]} but shape {self.shape} "
                f"needs {self.size}; the DWT LL of the patch must match PRN_LENGTH"
            )

        # <P_i, P_j> = 4096 * delta_ij exactly (see codebook.py), so the RMS of any
        # signed sum of all `n_prns` of them is exactly sqrt(n_prns), for every
        # codeword. Precomputed rather than measured per call: measuring it would give
        # the same number every time and hide the fact that it is a constant.
        self.n_prns = self.codebook.shape[0]
        self.combined_rms = np.sqrt(self.n_prns)

    @property
    def prns(self):
        """(n_prns, *shape) float64. The matched-filter bank."""
        return self.codebook.astype(np.float64).reshape((self.n_prns,) + self.shape)

    @property
    def matrix(self):
        """(n_prns, size) float64, flattened. c = matrix @ x / size is the correlation."""
        return self.codebook.astype(np.float64)

    def get_prn_for_index(self, index):
        if not 0 <= index < len(self.codebook):
            raise ValueError(
                f"PRN index must be between 0 and {len(self.codebook) - 1}"
            )
        return self.codebook[index].reshape(self.shape)

    def combine(self, signs):
        """
        sum_i s_i P_i, RMS-normalised to 1.

        The normalisation is by the exact constant sqrt(n_prns), not by the measured
        RMS of this particular sum. They are equal to floating-point precision, but
        using the constant makes the embedded amplitude per bit exactly
        ALPHA / sqrt(n_prns) rather than something that drifts with the codeword.
        """
        signs = np.asarray(signs, dtype=np.float64).ravel()
        if signs.size > self.n_prns:
            raise ValueError(
                f"payload needs {signs.size} PRNs, codebook has {self.n_prns}"
            )
        if not np.isin(signs, (-1.0, 1.0)).all():
            raise ValueError("signs must be -1 or +1")

        combined = signs @ self.codebook[:signs.size].astype(np.float64)
        combined /= np.sqrt(signs.size)
        return combined.reshape(self.shape)

    def get_balanced_prn_for_bit_string(self, bit_string):
        return self.combine(signs_for_bit_string(bit_string))
