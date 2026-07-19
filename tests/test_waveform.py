"""Tests for the waveform-level (imperfect-SIC) DF-NOMA BER simulator."""

import numpy as np

from simulator.ber_waveform import simulate_waveform_df_noma_ber
from simulator.utils import df_ber_theory


def _single_user(n0, n_bits, seed):
    """One SU, no NOMA co-interference and a weak PU-at-relay -> SIC is trivial,
    so the waveform BER should reproduce the perfect-SIC DF theory."""
    Ps, Pr, alpha, Ppt = 1.0, 3.0, 0.3, 1.0
    h_rd, h_pd = 1.0 + 0j, 0.02 + 0j
    g_sr = Ps / n0
    g_rd = (1 - alpha) * Pr * abs(h_rd) ** 2 / (Ppt * abs(h_pd) ** 2 + n0)
    res = simulate_waveform_df_noma_ber(
        h_sr=np.array([1.0 + 0j]), h_sp=np.array([0.01 + 0j]), h_pr=0.02 + 0j,
        h_pp=1.0 + 0j, h_pd=h_pd, h_rd=h_rd, h_rp=0.01 + 0j,
        p_su=np.array([Ps]), p_relay=Pr, alpha=alpha, p_pt=Ppt, n0=n0,
        sic_order=np.array([0]), pu_use_direct=True, n_bits=n_bits,
        rng=np.random.default_rng(seed),
    )
    return g_sr, g_rd, res


def test_single_user_matches_df_theory():
    g_sr, g_rd, res = _single_user(n0=0.4, n_bits=300000, seed=3)
    th_hop1, th_e2e = df_ber_theory(g_sr, g_rd)
    assert abs(res["ber_su_hop1"] - th_hop1) < 0.005
    assert abs(res["ber_su_e2e"] - th_e2e) < 0.005


def test_high_snr_is_near_zero():
    _, _, res = _single_user(n0=0.02, n_bits=200000, seed=4)
    assert res["ber_su_e2e"] < 1e-3


def test_output_shape_and_keys():
    N = 3
    res = simulate_waveform_df_noma_ber(
        h_sr=np.ones(N) + 0j, h_sp=0.1 * (np.ones(N) + 0j), h_pr=0.1 + 0j,
        h_pp=1.0 + 0j, h_pd=0.05 + 0j, h_rd=1.0 + 0j, h_rp=0.05 + 0j,
        p_su=np.array([1.0, 0.5, 0.25]), p_relay=3.0, alpha=0.3, p_pt=1.0, n0=0.1,
        sic_order=np.array([0, 1, 2]), pu_use_direct=True, n_bits=2000,
        rng=np.random.default_rng(5),
    )
    assert len(res["ber_su_hop1_per_user"]) == N
    assert len(res["ber_su_e2e_per_user"]) == N
    for k in ("ber_su_hop1", "ber_su_e2e", "ber_pu_e2e"):
        assert 0.0 <= res[k] <= 1.0


def test_imperfect_sic_penalty_vs_perfect():
    """Equal-power, similar-channel users are poorly separated: waveform
    end-to-end BER should exceed the perfect-SIC DF theory (an error floor)."""
    N = 3
    n0 = 1e-3  # very low noise: perfect-SIC theory would give ~0 BER
    h_sr = np.array([1.0 + 0j, 0.98 + 0j, 0.96 + 0j])
    res = simulate_waveform_df_noma_ber(
        h_sr=h_sr, h_sp=0.05 * (np.ones(N) + 0j), h_pr=0.05 + 0j,
        h_pp=1.0 + 0j, h_pd=0.02 + 0j, h_rd=1.0 + 0j, h_rp=0.02 + 0j,
        p_su=np.ones(N), p_relay=3.0, alpha=0.3, p_pt=1.0, n0=n0,
        sic_order=np.array([0, 1, 2]), pu_use_direct=True, n_bits=20000,
        rng=np.random.default_rng(6),
    )
    # Perfect-SIC theory at this noise would be ~0; the waveform shows a floor.
    assert res["ber_su_e2e"] > 1e-2
