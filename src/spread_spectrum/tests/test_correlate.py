"""Correlator and localiser: the geometry has to be exact or nothing downstream works."""

import numpy as np
import pywt

from src.spread_spectrum.detect.correlate import (PRNCorrelator, frame_to_plane,
                                                  ll_block_size, ll_planes,
                                                  plane_to_frame)
from src.spread_spectrum.detect.localise import PatchLocaliser
from src.spread_spectrum.detect.whiten import PreWhitener, MODES
from src.spread_spectrum.prn import BalancedPRNGenerator
from src.spread_spectrum.tests.helpers import embed_into, mark_positions, synthetic_host


def test_ll_planes_reproduce_pywt_at_every_level_and_phase():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 255, (256, 256))
    for level in (1, 2, 3):
        B = ll_block_size(level)
        planes = ll_planes(y, level)
        assert len(planes) == B * B
        for (py, px), pl in planes.items():
            h = ((y.shape[0] - py) // B) * B
            w = ((y.shape[1] - px) // B) * B
            ref = pywt.wavedec2(y[py:py + h, px:px + w], "haar", level=level)[0]
            assert np.abs(pl - ref).max() < 1e-9, (level, py, px)


def test_frame_plane_index_round_trip():
    for level in (1, 2, 3):
        for y, x in [(0, 0), (1, 3), (128, 64), (255, 254)]:
            ph, (i, j) = frame_to_plane(y, x, level)
            assert plane_to_frame(ph, i, j, level) == (y, x)


def test_dense_fft_correlation_equals_direct():
    rng = np.random.default_rng(1)
    cor = PRNCorrelator(rng.choice([-1.0, 1.0], (32, 64, 64)), device="cpu")
    plane = rng.uniform(-10, 10, (200, 180))
    energy, maps = cor.dense(plane, want_maps=True)
    hv, wv = cor.valid_shape(plane.shape)
    pts = [(0, 0), (3, 7), (50, 90), (hv - 1, wv - 1)]
    direct, kept = cor.at(plane, pts)
    assert kept == pts
    assert np.abs(direct - np.stack([maps[:, i, j] for i, j in pts])).max() < 1e-12
    assert np.abs(energy[:hv, :wv] - (maps ** 2).sum(0)[:hv, :wv]).max() < 1e-12


def test_at_drops_overhanging_positions():
    cor = PRNCorrelator(np.ones((4, 8, 8)), device="cpu")
    c, kept = cor.at(np.zeros((20, 20)), [(0, 0), (13, 0), (0, 13), (12, 12)])
    assert kept == [(0, 0), (12, 12)] and c.shape == (2, 4)


def test_whiteners_are_finite_and_shape_preserving():
    x = synthetic_host((128, 128))
    for m in MODES:
        out = PreWhitener(mode=m)(x)
        assert out.shape == x.shape and np.isfinite(out).all(), m


def test_localiser_finds_the_embedded_patches():
    """The codeword-independent energy surface has to peak on the marks themselves."""
    host = synthetic_host((512, 512), seed=5)
    pos = mark_positions((512, 512))
    marked = embed_into(host, "1" * 16 + "0" * 16, pos)

    gen = BalancedPRNGenerator((64, 64), 8787)
    cor = PRNCorrelator(gen.prns, device="cpu")
    loc = PatchLocaliser(cor, level=1, max_sites=16)
    planes = {ph: PreWhitener()(pl) for ph, pl in ll_planes(marked, 1).items()}
    found = {(s.frame_y, s.frame_x) for s in loc.sites(planes, frame_shape=marked.shape)}

    hits = sum(any(abs(fy - y) <= 1 and abs(fx - x) <= 1 for fy, fx in found)
               for y, x in pos)
    assert hits == len(pos), f"{hits}/{len(pos)} patches localised within 1 px"


def test_localiser_ranks_sites_and_reports_the_candidate_count():
    host = synthetic_host((512, 512), seed=6)
    marked = embed_into(host, "0" * 32, mark_positions((512, 512)))
    gen = BalancedPRNGenerator((64, 64), 8787)
    loc = PatchLocaliser(PRNCorrelator(gen.prns, device="cpu"), level=1, max_sites=10)
    planes = {ph: PreWhitener()(pl) for ph, pl in ll_planes(marked, 1).items()}
    sites = loc.sites(planes, frame_shape=marked.shape)
    assert [s.rank for s in sites] == list(range(len(sites)))
    assert all(s.n_positions > 1000 for s in sites)
    assert all(a.energy >= b.energy for a, b in zip(sites, sites[1:]))
