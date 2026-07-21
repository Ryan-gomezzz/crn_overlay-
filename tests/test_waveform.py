"""Tests for the waveform-level (imperfect-SIC) cooperative DF-NOMA BER simulator.

Topology: M SU sources transmit to the relay-SU (one of the N=M+1 secondary
users), which then forwards their decoded data together with its OWN data and
the PU's data to the SU destination.
"""

import numpy as np

from simulator.ber_waveform import simulate_waveform_df_noma_ber
from simulator.utils import ber_bpsk_theory


def _run(*, M, p_src, h_sr, n0, p_relay=3.0, alpha=0.3, own_share=0.5,
         h_pr=0.01 + 0j, n_bits=200000, seed=0, mu=0.5):
    """Helper mirroring how the simulator splits the relay's slot-2 power."""
    p_relay_su = (1.0 - alpha) * p_relay
    p_own = own_share * p_relay_su
    p_fwd_total = (1.0 - own_share) * p_relay_su
    w = mu ** np.arange(M, dtype=float)
    w = w / w.sum()
    p_fwd = w * p_fwd_total          # sic_order is identity in these tests
    return simulate_waveform_df_noma_ber(
        h_sr=h_sr, h_sp=0.01 * (np.ones(M) + 0j), h_pr=h_pr,
        h_pp=1.0 + 0j, h_pd=0.02 + 0j, h_rd=1.0 + 0j, h_rp=0.02 + 0j,
        p_src=p_src, p_relay=p_relay, alpha=alpha, p_own=p_own, p_fwd=p_fwd,
        p_pt=1.0, n0=n0, sic_order=np.arange(M), pu_use_direct=True,
        n_bits=n_bits, rng=np.random.default_rng(seed),
    )


def test_per_user_arrays_include_the_relay_su():
    """Per-user arrays are [source_0..source_{M-1}, relay-SU] -> length M+1."""
    M = 2
    res = _run(M=M, p_src=np.array([1.0, 0.5]),
               h_sr=np.array([1.0 + 0j, 0.7 + 0j]), n0=0.05, n_bits=5000)
    assert len(res["ber_su_hop1_per_user"]) == M + 1
    assert len(res["ber_su_e2e_per_user"]) == M + 1
    # The relay-SU's own data is single-hop, so its hop-1 and e2e entries match.
    assert res["ber_su_hop1_per_user"][-1] == res["ber_su_e2e_per_user"][-1]
    assert 0.0 <= res["ber_relay_own"] <= 1.0


def test_hop1_single_source_matches_theory():
    """With one source there is no NOMA co-user interference at the relay, so
    the hop-1 BER must match the closed-form BPSK curve."""
    n0 = 0.4
    p = 1.0
    res = _run(M=1, p_src=np.array([p]), h_sr=np.array([1.0 + 0j]), n0=n0)
    expected = float(ber_bpsk_theory(p / n0))     # gamma_sr = P|h|^2 / N0
    assert abs(res["ber_su_hop1_per_user"][0] - expected) < 0.005


def test_high_snr_sources_decode_at_the_relay():
    """Well-separated source powers and low noise -> hop-1 SIC succeeds."""
    res = _run(M=2, p_src=np.array([1.0, 0.25]),
               h_sr=np.array([1.0 + 0j, 0.9 + 0j]), n0=1e-4, n_bits=50000)
    src_hop1 = res["ber_su_hop1_per_user"][:2]
    assert max(src_hop1) < 1e-3


def test_well_separated_hop2_decodes_cleanly():
    """Power-domain SIC at the destination only works when the stream being
    decoded dominates the ones not yet cancelled. With a small PU share and a
    dominant own-data share, every stream decodes."""
    res = _run(M=1, p_src=np.array([1.0]), h_sr=np.array([1.0 + 0j]),
               n0=1e-4, alpha=0.05, own_share=0.9, n_bits=50000)
    assert res["ber_relay_own"] < 1e-3
    assert res["ber_su_e2e_per_user"][0] < 1e-2


def test_gaussian_sinr_underestimates_ber_when_sic_is_marginal():
    """When the strongest stream does NOT dominate the rest, the discrete BPSK
    interference is far more damaging than a Gaussian-SINR approximation
    suggests -- the waveform BER greatly exceeds the closed-form value."""
    alpha, own_share, p_relay, n0 = 0.3, 0.5, 3.0, 1e-4
    res = _run(M=2, p_src=np.array([1.0, 0.25]),
               h_sr=np.array([1.0 + 0j, 0.9 + 0j]), n0=n0,
               alpha=alpha, own_share=own_share, p_relay=p_relay, n_bits=20000)
    # Gaussian-SINR view of the own stream (decoded first, others as noise):
    p_relay_su = (1 - alpha) * p_relay
    p_own = own_share * p_relay_su
    others = (alpha * p_relay) + (1 - own_share) * p_relay_su
    gaussian_ber = float(ber_bpsk_theory(p_own / (others + n0)))
    assert res["ber_relay_own"] > gaussian_ber


def test_imperfect_sic_floor_at_hop1():
    """Equal-power, near-identical channels are poorly separated: even at very
    low noise the relay's SIC fails, producing an error floor that the
    perfect-SIC (noise-only) theory would not predict."""
    n0 = 1e-4
    res = _run(M=2, p_src=np.ones(2),
               h_sr=np.array([1.0 + 0j, 0.99 + 0j]), n0=n0, n_bits=20000)
    # Noise-only theory would be ~0 here; imperfect SIC lifts it well above that.
    assert res["ber_su_hop1"] > 1e-2


def test_zero_relay_power_breaks_the_second_hop():
    """If the relay-SU transmits nothing, nothing reaches the destination, so
    end-to-end BER collapses to chance while hop-1 can still be clean."""
    res = _run(M=1, p_src=np.array([1.0]), h_sr=np.array([1.0 + 0j]),
               n0=1e-4, p_relay=0.0, n_bits=20000)
    assert res["ber_su_hop1_per_user"][0] < 1e-3
    assert res["ber_su_e2e_per_user"][0] > 0.2
