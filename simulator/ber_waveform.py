"""Waveform-level bit-error-rate simulation for the two-hop DF-NOMA overlay link.

This is the high-fidelity BER model: instead of mapping an effective SINR to a
BER, it transmits random BPSK symbols through the **complex** channels and runs
**actual Successive Interference Cancellation (SIC)** at the relay, subtracting
the *detected* symbols. Because a detection error is cancelled with the wrong
symbol, its residual leaks into the users decoded later — reproducing the
**imperfect-SIC error floor** that is the hallmark of NOMA BER analysis.
Decode-and-Forward error propagation is captured by comparing the destination's
decisions to the *original source bits*.

Where the imperfect SIC lives in this system:

  * **Hop 1 (SU sources + PT → relay)** is the genuine multi-user NOMA channel:
    N secondary users are superposed on the relay's antenna and separated by
    SIC. This hop is simulated at full waveform fidelity — the relay detects and
    cancels the PU, then the SUs in descending ``|h_sr,i|²`` order, subtracting
    detected symbols. This produces the NOMA/SIC error floor.

  * **Hop 2 (relay → SU destination)** is a per-user bottleneck: the rate model
    assigns every user the same relay→destination SINR ``γ_rd`` (there is no
    simultaneous N-user power-domain multiplexing on this hop). It is therefore
    simulated per user at full relay SU power ``(1-α)P_r``, with a two-layer SIC
    at the destination (cancel the relayed PU, then decode the SU) — consistent
    with the ``γ_e2e,i = min(γ_sr,i, γ_rd)`` rate model.

  * **Primary user** uses selection combining between its single-hop direct link
    (PT → PR, with SU interference) and its DF-relayed link (relay → PR on
    ``h_rp`` carrying the relay's decoded PU bits); the branch chosen by the
    physics (higher SINR) determines the reported PU BER.

All hops use the true complex channels ``h = g·√PL`` for the transmission whose
SINRs were reported by the simulator, so the BER is consistent with the rates.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _bpsk(bits: np.ndarray) -> np.ndarray:
    """Map bits {0,1} → BPSK symbols {+1,-1} (0→+1, 1→-1)."""
    return 1.0 - 2.0 * bits


def _detect(y: np.ndarray, h: complex) -> np.ndarray:
    """Coherent BPSK detection over channel ``h``: return decided bits {0,1}."""
    return (np.real(np.conj(h) * y) < 0.0).astype(np.int8)


def _cawgn(n: int, n0: float, rng: np.random.Generator) -> np.ndarray:
    """Circularly-symmetric complex AWGN with total variance ``n0``."""
    scale = np.sqrt(n0 / 2.0)
    return scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def simulate_waveform_df_noma_ber(
    *,
    h_sr: np.ndarray,          # complex, shape (M,)  source_i -> relay-SU
    h_sp: np.ndarray,          # complex, shape (M,)  source_i -> PR (interference)
    h_pr: complex,             # PT -> relay-SU
    h_pp: complex,             # PT -> PR (primary direct)
    h_pd: complex,             # PT -> SU destination (interference)
    h_rd: complex,             # relay-SU -> SU destination
    h_rp: complex,             # relay-SU -> PR (relayed PU / interference)
    p_src: np.ndarray,         # Watts, shape (M,)  source transmit powers
    p_relay: float,            # Watts, relay-SU total transmit power
    alpha: float,              # relay power fraction given to the PU
    p_own: float,              # Watts, relay-SU's OWN data power (slot 2)
    p_fwd: np.ndarray,         # Watts, shape (M,)  forwarded per-source powers
    p_pt: float,               # Watts
    n0: float,                 # noise power (Watts)
    sic_order: np.ndarray,     # relay SIC order over sources (strongest |h_sr|² first)
    pu_use_direct: bool,       # selection-combining decision from the physics
    n_bits: int = 4000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Run the waveform DF-NOMA BER simulation for one channel realization.

    Returns per-source hop-1 (relay decode) and end-to-end BER, the relay-SU's
    own-data BER, and the PU's relay/end-to-end BER. Index ``M`` of the per-user
    arrays is the relay-SU itself.
    """
    rng = rng or np.random.default_rng()
    M = len(p_src)
    K = int(n_bits)
    sic_order = np.asarray(sic_order, dtype=int)

    # -------------------- source bits --------------------
    src_bits = [rng.integers(0, 2, size=K).astype(np.int8) for _ in range(M)]
    own_bits = rng.integers(0, 2, size=K).astype(np.int8)   # relay-SU's own data
    pu_bits = rng.integers(0, 2, size=K).astype(np.int8)
    src_sym = [_bpsk(b) for b in src_bits]
    pu_sym = _bpsk(pu_bits)

    # ============ HOP 1: sources + PT -> relay-SU (multi-user NOMA) ============
    # The relay-SU is half-duplex: it only listens in slot 1.
    y = np.sqrt(p_pt) * h_pr * pu_sym
    for i in range(M):
        y = y + np.sqrt(p_src[i]) * h_sr[i] * src_sym[i]
    y = y + _cawgn(K, n0, rng)

    # Relay SIC: PU first (sources as interference), then sources in descending
    # |h_sr|², each time cancelling the *detected* symbol (imperfect-SIC residual).
    pu_bits_relay = _detect(y, h_pr)
    y = y - np.sqrt(p_pt) * h_pr * _bpsk(pu_bits_relay)

    relay_bits: List[np.ndarray] = [None] * M
    for i in sic_order:
        bi = _detect(y, h_sr[i])
        relay_bits[i] = bi
        y = y - np.sqrt(p_src[i]) * h_sr[i] * _bpsk(bi)

    ber_src_hop1 = np.array([float(np.mean(relay_bits[i] != src_bits[i])) for i in range(M)])
    ber_pu_hop1 = float(np.mean(pu_bits_relay != pu_bits))

    # ============ HOP 2: relay-SU -> destination (superposed NOMA) ============
    # The relay-SU transmits its OWN data + the decoded source data + the PU's
    # data, all superposed on h_rd. The destination SICs in descending power.
    x_r = (np.sqrt(alpha * p_relay) * _bpsk(pu_bits_relay)
           + np.sqrt(p_own) * _bpsk(own_bits))
    for i in range(M):
        x_r = x_r + np.sqrt(p_fwd[i]) * _bpsk(relay_bits[i])

    pu_new = rng.integers(0, 2, size=K).astype(np.int8)   # PT's fresh slot-2 data
    y_d = h_rd * x_r + np.sqrt(p_pt) * h_pd * _bpsk(pu_new) + _cawgn(K, n0, rng)

    # Stream powers/bits in one table so we can cancel in descending power order.
    stream_p = np.concatenate(([alpha * p_relay, p_own], p_fwd))
    stream_bits = [pu_bits_relay, own_bits] + [relay_bits[i] for i in range(M)]
    dest_order = np.argsort(-stream_p)

    decoded = [None] * len(stream_p)
    for idx in dest_order:
        bi = _detect(y_d, h_rd)
        decoded[idx] = bi
        y_d = y_d - np.sqrt(stream_p[idx]) * h_rd * _bpsk(bi)

    # End-to-end: compare to the ORIGINAL bits (DF errors propagate).
    ber_src_e2e = np.array(
        [float(np.mean(decoded[2 + i] != src_bits[i])) for i in range(M)]
    )
    ber_own = float(np.mean(decoded[1] != own_bits))   # single-hop (slot 2 only)

    # ==================== PU end-to-end (selection combining) ====================
    # Direct branch: PT -> PR with source interference (slot 1).
    y_pr = np.sqrt(p_pt) * h_pp * pu_sym
    for i in range(M):
        y_pr = y_pr + np.sqrt(p_src[i]) * h_sp[i] * src_sym[i]
    y_pr = y_pr + _cawgn(K, n0, rng)
    ber_pu_direct = float(np.mean(_detect(y_pr, h_pp) != pu_bits))

    # Relayed branch: the relay-SU's decoded PU forwarded to PR on h_rp, with
    # PT's new transmission and the relay's secondary payload as interference.
    pu_new_pr = rng.integers(0, 2, size=K).astype(np.int8)
    y_pr_rel = (h_rp * x_r + np.sqrt(p_pt) * h_pp * _bpsk(pu_new_pr)
                + _cawgn(K, n0, rng))
    ber_pu_relayed = float(np.mean(_detect(y_pr_rel, h_rp) != pu_bits))

    ber_pu_e2e = ber_pu_direct if pu_use_direct else ber_pu_relayed

    # Per-user arrays are ordered [source_0..source_{M-1}, relay-SU].
    hop1_all = np.concatenate((ber_src_hop1, [ber_own]))
    e2e_all = np.concatenate((ber_src_e2e, [ber_own]))

    return {
        "ber_su_hop1_per_user": hop1_all.tolist(),
        "ber_su_hop1": float(np.mean(hop1_all)),
        "ber_su_e2e_per_user": e2e_all.tolist(),
        "ber_su_e2e": float(np.mean(e2e_all)),
        "ber_relay_own": ber_own,
        "ber_pu_hop1": ber_pu_hop1,
        "ber_pu_direct": ber_pu_direct,
        "ber_pu_relayed": ber_pu_relayed,
        "ber_pu_e2e": ber_pu_e2e,
    }
