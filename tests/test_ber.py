"""Tests for the BPSK BER utilities (theoretical + Monte-Carlo)."""

import numpy as np

from simulator.utils import ber_bpsk_montecarlo, ber_bpsk_theory


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
