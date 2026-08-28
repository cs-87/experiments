import numpy as np

from src.spread_spectrum.codebook import generate_codebook


class BalancedPRNGenerator:

    def __init__(self, shape, seed):
        self.shape = shape
        self.size = np.prod(shape)
        self.seed = seed

        self.codebook = generate_codebook(seed)

    def get_prn_for_index(self, index):
        if not 0 <= index < len(self.codebook):
            raise ValueError(
                f"PRN index must be between 0 and {len(self.codebook) - 1}"
            )

        prn = self.codebook[index]

        dc = prn.sum()

        if dc != 0:
            print(f"Warning: PRN {index} DC value is {dc}")

        return prn.reshape(self.shape)

    def get_blalanced_prn_for_bit_string(self, bit_string):
        if len(bit_string) > len(self.codebook):
            raise ValueError(
                f"Bit string length must be less than or equal to {len(self.codebook)}"
            )

        prn = np.zeros(self.size, dtype=np.int8)

        for i, bit in enumerate(bit_string):
            if bit == "1":
                prn += self.get_prn_for_index(i).flatten()
            elif bit == "0":
                prn -= self.get_prn_for_index(i).flatten()
            else:
                raise ValueError(f"Invalid bit: {bit}")

        prn = prn.astype(np.float32)

        prn /= np.sqrt(np.mean(prn ** 2))

        return prn.reshape(self.shape)
