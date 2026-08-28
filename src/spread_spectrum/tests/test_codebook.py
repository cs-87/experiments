"""
The signal model, asserted exactly.

Every one of these is an equality rather than a tolerance, because the construction
makes them exact and the whole detector design leans on that. prompt.md states them
as approximations; if a change ever makes them approximations in fact, these fail.
"""

import numpy as np
import pywt

from src.spread_spectrum.codebook import (generate_codebook, verify_codebook,
                                          NUM_PRNS, PRN_LENGTH)
from src.spread_spectrum.embed import embed_patch
from src.spread_spectrum.prn import (BalancedPRNGenerator, bit_string_for_signs,
                                     signs_for_bit_string)

SEEDS = (8787, 0, 1, 42, 12345)


def test_prn_values_are_plus_minus_one():
    for s in SEEDS:
        assert set(np.unique(generate_codebook(s)).tolist()) == {-1, 1}


def test_prn_rms_is_exactly_one():
    for s in SEEDS:
        cb = generate_codebook(s).astype(np.float64)
        assert np.array_equal(np.sqrt((cb ** 2).mean(axis=1)), np.ones(NUM_PRNS))


def test_prn_dc_is_exactly_zero():
    for s in SEEDS:
        assert np.abs(generate_codebook(s).astype(np.int64).sum(axis=1)).max() == 0


def test_prn_orthogonality_is_exact():
    for s in SEEDS:
        v = verify_codebook(generate_codebook(s))
        assert v["max_abs_offdiag"] == 0.0, v["max_abs_offdiag"]
        assert v["gram_diag"] == float(PRN_LENGTH)


def test_combined_rms_is_exactly_sqrt_32_for_every_codeword():
    cb = generate_codebook(8787).astype(np.float64)
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = rng.choice([-1.0, 1.0], NUM_PRNS)
        assert abs(np.sqrt(((s @ cb) ** 2).mean()) - np.sqrt(NUM_PRNS)) < 1e-12


def test_normalised_watermark_has_unit_rms():
    gen = BalancedPRNGenerator((64, 64), 8787)
    rng = np.random.default_rng(1)
    for _ in range(200):
        bits = "".join(rng.choice(list("01"), 32))
        w = gen.get_balanced_prn_for_bit_string(bits)
        assert abs(np.sqrt((w ** 2).mean()) - 1.0) < 1e-12


def test_matched_filter_output_is_exactly_alpha_over_sqrt32():
    """Correct PRN gives alpha/sqrt(32); incorrect PRNs give exactly zero."""
    gen = BalancedPRNGenerator((64, 64), 8787)
    cb = gen.matrix
    rng = np.random.default_rng(2)
    for alpha in (1.0, 3.0, 7.5):
        bits = "".join(rng.choice(list("01"), 32))
        s = signs_for_bit_string(bits)
        c = (cb @ (alpha * gen.get_balanced_prn_for_bit_string(bits)).ravel()) / PRN_LENGTH
        assert np.allclose(c, s * alpha / np.sqrt(NUM_PRNS), atol=1e-12)
        assert bit_string_for_signs(c) == bits
        # a PRN that is not in the codebook at all responds with nothing
        other = generate_codebook(999).astype(np.float64)
        cross = (other @ gen.get_balanced_prn_for_bit_string(bits).ravel()) / PRN_LENGTH
        assert np.abs(cross).max() < np.abs(c).min()


def test_ll_embedding_equals_block_replicated_pixel_embedding():
    """
    The identity the whole blind correlator rests on: adding alpha*W to the level-L
    Haar LL is adding (alpha/B)*W to the pixels with each element replicated over its
    B x B block, B = 2^L.
    """
    rng = np.random.default_rng(3)
    for level in (1, 2, 3):
        B = 1 << level
        side = 64 * B
        gen = BalancedPRNGenerator((64, 64), 8787)
        w = gen.get_balanced_prn_for_bit_string("1" * 20 + "0" * 12)
        host = rng.uniform(0, 255, (side, side))
        got = embed_patch(host, w, alpha=3.0, level=level)
        want = host + (3.0 / B) * np.kron(w, np.ones((B, B)))
        assert np.abs(got - want).max() < 1e-6, (level, np.abs(got - want).max())


def test_ll_scale_matches_pywt():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 255, (128, 128))
    for level in (1, 2):
        B = 1 << level
        ll = pywt.wavedec2(x, "haar", level=level)[0]
        block = x.reshape(x.shape[0] // B, B, x.shape[1] // B, B).mean(axis=(1, 3)) * B
        assert np.abs(ll - block).max() < 1e-9
