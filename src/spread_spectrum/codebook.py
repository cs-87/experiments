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


def generate_codebook(seed):
    rng = np.random.default_rng(seed)

    # ---------------------------------------------
    # 1. Generate random balanced PRN
    # ---------------------------------------------

    R = np.ones(PRN_LENGTH, dtype=np.int8)
    R[:PRN_LENGTH // 2] = -1

    rng.shuffle(R)

    # ---------------------------------------------
    # 2. Generate Hadamard matrix
    # ---------------------------------------------

    H = generate_hadamard(PRN_LENGTH)

    # ---------------------------------------------
    # 3. Generate candidates
    # ---------------------------------------------

    candidates = H * R[np.newaxis, :]

    # H[0] * R == R, so remove it
    candidates = candidates[1:]

    # ---------------------------------------------
    # 4. Calculate imbalance
    # ---------------------------------------------

    imbalance = np.abs(candidates.sum(axis=1))

    # ---------------------------------------------
    # 5. Pick 31 lowest-imbalance candidates
    # ---------------------------------------------

    selected = np.argsort(imbalance)[:NUM_PRNS]

    # ---------------------------------------------
    # 6. Construct final 32 PRNs
    # ---------------------------------------------

    codebook = np.vstack([
        candidates[selected]
    ])

    # we don't need the first one since it will be mother video id

    combined = np.sum(codebook, axis=0)

    # zero DC is already preserved
    combined = combined - combined.mean()

    # RMS normalize
    combined /= np.sqrt(np.mean(combined ** 2))

    return codebook
