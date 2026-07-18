"""Tests for the BPSK BER utilities (theoretical + Monte-Carlo)."""

import numpy as np

from simulator.utils import (
    ber_bpsk_montecarlo,
    ber_bpsk_theory,
    df_ber_theory,
    simulate_df_ber_montecarlo,
)


def test_theory_known_points():
    # BER = 0.5 at 0 dB-linear=0 SNR; monotonically decreasing with SNR.
    assert np.isclose(ber_bpsk_theory(0.0), 0.5, atol=1e-6)
    gammas = np.array([0.0, 1.0, 4.0, 10.0])
    bers = ber_bpsk_theory(gammas)
    assert np.all(np.diff(bers) < 0)          # strictly decreasing
    assert bers[-1] < 1e-2                     # high SNR -> low BER


def test_theory_accepts_scalar_and_array():
    assert np.isscalar(ber_bpsk_theory(2.0)) or ber_bpsk_theory(2.0).ndim == 0
    arr = ber_bpsk_theory([1.0, 2.0, 3.0])
    assert arr.shape == (3,)


def test_montecarlo_matches_theory():
    rng = np.random.default_rng(0)
    for gamma in [0.5, 1.0, 3.0]:
        mc = ber_bpsk_montecarlo(gamma, n_bits=200000, rng=rng)
        th = float(ber_bpsk_theory(gamma))
        # Monte-Carlo estimate should track theory within a few percent.
        assert abs(mc - th) < 0.02, f"gamma={gamma}: mc={mc}, theory={th}"


def test_montecarlo_negative_gamma_is_floored():
    # Negative SINR is treated as 0 SNR -> BER ~ 0.5, never crashes.
    mc = ber_bpsk_montecarlo(-5.0, n_bits=5000, rng=np.random.default_rng(1))
    assert 0.4 < mc <= 0.6


def test_df_theory_formula_and_ordering():
    g1, g2 = 10 ** (6 / 10), 10 ** (8 / 10)
    p1 = float(ber_bpsk_theory(g1))
    p2 = float(ber_bpsk_theory(g2))
    hop1, e2e = df_ber_theory(g1, g2)
    assert np.isclose(hop1, p1)
    assert np.isclose(e2e, p1 + p2 - 2 * p1 * p2)
    # DF end-to-end BER is never better than either individual hop.
    assert e2e >= p1 - 1e-12 and e2e >= p2 - 1e-12


def test_df_montecarlo_matches_df_theory():
    rng = np.random.default_rng(0)
    for g1_db, g2_db in [(3, 10), (6, 6), (10, 3)]:
        g1, g2 = 10 ** (g1_db / 10), 10 ** (g2_db / 10)
        th_hop1, th_e2e = df_ber_theory(g1, g2)
        mc_hop1, mc_e2e = simulate_df_ber_montecarlo(g1, g2, n_bits=300000, rng=rng)
        assert abs(mc_hop1 - th_hop1) < 0.01
        assert abs(mc_e2e - th_e2e) < 0.01


def test_df_e2e_exceeds_single_hop_mapping():
    # Two equal hops: end-to-end BER should be ~2x a single-hop mapping.
    g = 10 ** (7 / 10)
    single = float(ber_bpsk_theory(g))
    _, e2e = df_ber_theory(g, g)
    assert e2e > single
