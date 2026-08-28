"""
The 32-PRN codebook.

Construction: take the Sylvester-Hadamard matrix H of order 4096, drop row 0, and
multiply every remaining row elementwise by one fixed random balanced +/-1 vector R:

    P_i = H_i * R                       (elementwise)

The two properties the detector is built on both fall out of this exactly, not
approximately, and both are worth stating because prompt.md assumes only the
approximate versions:

  Orthogonality.  <P_i, P_j> = sum_k H_i[k] H_j[k] R[k]^2 = <H_i, H_j>, because
  R[k]^2 == 1 for every k. Hadamard rows are exactly orthogonal, so the Gram matrix
  of the codebook is exactly 4096 * I -- there is zero inter-PRN interference, and
  therefore the RMS of any signed combination of all 32 is exactly sqrt(32) for
  every one of the 2^32 codewords. That is what makes the normalisation constant in
  prn.py a known constant rather than a codeword-dependent one, and it is why the
  matched filter for one bit is unaffected by the other 31.

  Zero DC.  sum(P_i) = <H_i, R>, which is the i-th Walsh coefficient of R. Those are
  not zero in general, so the codebook is chosen by taking the 32 rows whose
  coefficient is smallest in magnitude. For a balanced R of length 4096 roughly 50 of
  the 4095 candidates come out exactly zero, so 32 is comfortable but not guaranteed
  -- verify_codebook() reports what a given seed actually achieved. A nonzero DC would
  make that PRN respond to a brightness change, which is a photometric attack.

Multiplying by R is what stops the PRNs being visually structured: a raw Hadamard row
is a regular block pattern, and R scrambles it into something noise-like at the same
inner products.
"""

from functools import lru_cache

import numpy as np


PRN_LENGTH = 4096
NUM_PRNS = 32


def generate_hadamard(n):
    H = np.array([[1]], dtype=np.int8)

    while H.shape[0] < n:
        H = np.block([
            [H,  H],
            [H, -H],
        ]).astype(np.int8)

    return H


@lru_cache(maxsize=8)
def generate_codebook(seed, length=PRN_LENGTH, count=NUM_PRNS):
    """
    (count, length) int8 array of +/-1 PRNs, deterministic in `seed`.

    Cached: the detector rebuilds this per patch batch and the Hadamard matrix is a
    16 MB intermediate. The return value is marked read-only so a cache hit cannot be
    mutated by one caller under another.
    """
    rng = np.random.default_rng(seed)

    # A balanced +/-1 mask. Balanced so that <H_0, R> = 0 exactly; H_0 is the all-ones
    # row and would otherwise give a PRN equal to R itself, with whatever DC R has.
    R = np.ones(length, dtype=np.int8)
    R[:length // 2] = -1
    rng.shuffle(R)

    H = generate_hadamard(length)

    # H[0] * R == R, and R is not orthogonal to the rest in any useful sense, so it is
    # dropped rather than competing for a slot.
    candidates = (H * R[np.newaxis, :])[1:]

    # |sum| is the DC imbalance; see the module docstring.
    imbalance = np.abs(candidates.sum(axis=1))
    selected = np.argsort(imbalance, kind="stable")[:count]

    codebook = candidates[selected].copy()
    codebook.setflags(write=False)
    return codebook


def verify_codebook(codebook):
    """
    The three invariants every downstream derivation depends on, as measured numbers.

    Returns a dict rather than asserting: the DC one is the only property the
    construction cannot guarantee, and a caller may want to see how close it got
    rather than crash.
    """
    cb = np.asarray(codebook, dtype=np.float64)
    n, length = cb.shape
    gram = cb @ cb.T
    off = gram - np.diag(np.diag(gram))
    return {
        "count": n,
        "length": length,
        "values": sorted(np.unique(cb).tolist()),
        "rms": np.sqrt((cb ** 2).mean(axis=1)),
        "dc": cb.sum(axis=1),
        "max_abs_dc": float(np.abs(cb.sum(axis=1)).max()),
        "gram_diag": float(np.unique(np.diag(gram))[0]) if len(np.unique(np.diag(gram))) == 1 else None,
        "max_abs_offdiag": float(np.abs(off).max()),
        # RMS of the signed sum of all `count` PRNs; exactly sqrt(count) iff the Gram
        # matrix is diagonal.
        "combined_rms": float(np.sqrt((cb.sum(axis=0) ** 2).mean())),
    }


if __name__ == "__main__":
    for seed in (8787, 0, 1, 42, 12345):
        v = verify_codebook(generate_codebook(seed))
        print(f"seed {seed:>6}: values={v['values']} rms={set(np.round(v['rms'], 12))} "
              f"max|DC|={v['max_abs_dc']:.0f} gram_diag={v['gram_diag']:.0f} "
              f"max|offdiag|={v['max_abs_offdiag']:.0f} "
              f"combined_rms={v['combined_rms']:.6f} (sqrt(32)={np.sqrt(32):.6f})")
