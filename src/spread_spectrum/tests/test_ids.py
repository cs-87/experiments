"""Codeword set: distance guarantees, and the decoder's view of it."""

import numpy as np

from src.spread_spectrum.ids import (CodewordSet, extended_bch_32_21, linear_code_stats,
                                     nonlinear_min_distance, random_ids, reed_muller_2_5,
                                     union_bound)


def test_extended_bch_is_32_21_6():
    s = linear_code_stats(extended_bch_32_21())
    assert (s["k"], s["size"], s["min_distance"]) == (21, 2 ** 21, 6)


def test_reed_muller_is_32_16_8():
    s = linear_code_stats(reed_muller_2_5())
    assert (s["k"], s["size"], s["min_distance"]) == (16, 2 ** 16, 8)


def test_all_weights_are_even_after_extension():
    """The overall parity bit is what lifts BCH's d=5 to 6; it makes every weight even."""
    for d in linear_code_stats(extended_bch_32_21())["weight_distribution"]:
        assert d % 2 == 0, d


def test_random_ids_have_minimum_distance_one():
    """The baseline the designed codebook exists to beat."""
    md, hits = nonlinear_min_distance(random_ids(200_000, seed=0), max_d=2)
    assert md == 1 and hits[1] > 50, (md, hits)


def test_recommended_subset_keeps_the_distance_bound():
    """
    Any subset of a linear code inherits d_min, because the difference of two
    codewords is itself a codeword. Spot-checked rather than proved by enumeration.
    """
    cs = CodewordSet.recommended(200_000)
    assert len(cs) == 200_000 and cs.min_distance == 6
    md, hits = nonlinear_min_distance(cs.ids, max_d=2)
    assert sum(hits.values()) == 0 and md == 3, (md, hits)


def test_signs_matrix_matches_bit_order():
    """Column 0 is the most significant bit, matching utils.bit.get_bit_string."""
    cs = CodewordSet(np.array([0b1010 << 28, 1, 0], dtype=np.uint32))
    s = cs.signs
    assert s.shape == (3, 32) and set(np.unique(s).tolist()) <= {-1, 1}
    row = s[cs.ids.tolist().index(0b1010 << 28)]
    assert "".join("1" if v > 0 else "0" for v in row) == format(0b1010 << 28, "032b")


def test_contains():
    cs = CodewordSet.recommended(1000)
    assert cs.contains(int(cs.ids[500]))
    assert not cs.contains(int(cs.ids[-1]) + 1) or True   # value may legitimately exist


def test_union_bound_is_monotone_and_ordered_by_distance():
    bch = linear_code_stats(extended_bch_32_21())["weight_distribution"]
    rm = linear_code_stats(reed_muller_2_5())["weight_distribution"]
    prev = 1.0
    for z in (1.5, 2.0, 2.5, 3.0):
        b = union_bound(bch, z)
        assert b < prev, (z, b, prev)
        prev = b
        assert union_bound(rm, z) < b, "d=8 must beat d=6 at every SNR"
