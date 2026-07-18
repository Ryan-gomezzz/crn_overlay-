"""
Utility functions for the CRN-RL simulator.
Author: Ryan

Provides unit-conversion helpers and array-manipulation utilities
used across the simulator and environment modules.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc


def ber_bpsk_theory(gamma_lin) -> np.ndarray:
    """Theoretical BPSK bit-error rate over an AWGN channel.

    For BPSK, ``BER = Q(sqrt(2*gamma)) = 0.5 * erfc(sqrt(gamma))`` where
    ``gamma`` is the (linear) received SNR / SINR. Accepts a scalar or array.

    Args:
        gamma_lin: Linear SNR/SINR value(s). Negative values are floored at 0.

    Returns:
        BER value(s) as a float numpy array (0.5 at gamma=0, → 0 as gamma→∞).
    """
    g = np.maximum(np.asarray(gamma_lin, dtype=float), 0.0)
    return 0.5 * erfc(np.sqrt(g))


def ber_bpsk_montecarlo(
    gamma_lin: float,
    n_bits: int = 20000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte-Carlo BPSK bit-error rate at a given linear SNR/SINR.

    Transmits ``n_bits`` random bits as BPSK symbols (±1) through an AWGN
    channel and counts errors. The symbol amplitude is ``sqrt(2*gamma)`` against
    unit-variance noise, so the error probability equals ``Q(sqrt(2*gamma)) =
    0.5*erfc(sqrt(gamma))`` — i.e. it converges to :func:`ber_bpsk_theory`.

    Args:
        gamma_lin: Linear SNR/SINR. Negative values are floored at 0.
        n_bits: Number of transmitted bits (more bits → lower measurable BER).
        rng: Optional numpy Generator for reproducibility.

    Returns:
        Empirical BER in ``[0, 1]``. Returns ``0.0`` when no errors are observed
        (i.e. below the ~``1/n_bits`` resolution floor).
    """
    rng = rng or np.random.default_rng()
    g = max(float(gamma_lin), 0.0)
    bits = rng.integers(0, 2, size=n_bits)
    symbols = 2 * bits - 1                       # {0,1} -> {-1,+1}
    received = math.sqrt(2.0 * g) * symbols + rng.standard_normal(n_bits)
    decoded = (received > 0).astype(int)
    return float(np.mean(decoded != bits))


def df_ber_theory(gamma_hop1, gamma_hop2):
    """Theoretical per-hop and end-to-end BER of a Decode-and-Forward link.

    The relay decodes hop 1 (error prob ``P1``) and forwards its decision;
    the destination decodes hop 2 (error prob ``P2``). An end-to-end bit is in
    error iff it was flipped an odd number of times, so::

        P_e2e = P1 + P2 - 2*P1*P2

    which is always ``>= max(P1, P2)`` — the DF error-propagation penalty that a
    single ``min(SINR)`` mapping cannot capture.

    Args:
        gamma_hop1: Linear SINR of hop 1 (source → relay).
        gamma_hop2: Linear SINR of hop 2 (relay → destination).

    Returns:
        Tuple ``(ber_hop1, ber_e2e)``.
    """
    p1 = float(ber_bpsk_theory(gamma_hop1))
    p2 = float(ber_bpsk_theory(gamma_hop2))
    return p1, p1 + p2 - 2.0 * p1 * p2


def simulate_df_ber_montecarlo(
    gamma_hop1,
    gamma_hop2,
    n_bits: int = 20000,
    rng: np.random.Generator | None = None,
):
    """Bit-level Monte-Carlo BER of a two-hop Decode-and-Forward link.

    Random source bits are BPSK-modulated and **decoded at the relay** (hop 1),
    the relay's (possibly erroneous) decisions are **re-encoded and forwarded**,
    then **decoded at the destination** (hop 2). Errors are counted per hop and
    end-to-end against the original source bits, so DF error propagation is
    captured exactly (converges to :func:`df_ber_theory`).

    Args:
        gamma_hop1: Linear SINR of hop 1 (source → relay).
        gamma_hop2: Linear SINR of hop 2 (relay → destination).
        n_bits: Number of transmitted bits.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        Tuple ``(ber_hop1, ber_e2e)`` of empirical BERs in ``[0, 1]``.
    """
    rng = rng or np.random.default_rng()
    g1 = max(float(gamma_hop1), 0.0)
    g2 = max(float(gamma_hop2), 0.0)

    source = rng.integers(0, 2, size=n_bits)

    # Hop 1: source -> relay (coherent BPSK detection at SINR g1)
    rx1 = math.sqrt(2.0 * g1) * (2 * source - 1) + rng.standard_normal(n_bits)
    relay = (rx1 > 0).astype(int)

    # Hop 2: relay re-encodes its decisions -> destination (SINR g2)
    rx2 = math.sqrt(2.0 * g2) * (2 * relay - 1) + rng.standard_normal(n_bits)
    dest = (rx2 > 0).astype(int)

    ber_hop1 = float(np.mean(relay != source))
    ber_e2e = float(np.mean(dest != source))
    return ber_hop1, ber_e2e


def dbm_to_watt(dbm: float) -> float:
    """Convert power from dBm to Watts.

    Args:
        dbm: Power in dBm.

    Returns:
        Power in Watts.
    """
    return 10.0 ** ((dbm - 30.0) / 10.0)


def watt_to_dbm(watt: float) -> float:
    """Convert power from Watts to dBm.

    Args:
        watt: Power in Watts (must be > 0).

    Returns:
        Power in dBm.

    Raises:
        ValueError: If *watt* is non-positive.
    """
    if watt <= 0:
        raise ValueError(f"Power must be positive, got {watt}")
    return 10.0 * math.log10(watt) + 30.0


def db_to_linear(db: float) -> float:
    """Convert a value from dB to linear scale.

    Args:
        db: Value in dB.

    Returns:
        Linear-scale value.
    """
    return 10.0 ** (db / 10.0)


def linear_to_db(linear: float) -> float:
    """Convert a value from linear scale to dB.

    Args:
        linear: Linear-scale value (must be > 0).

    Returns:
        Value in dB.

    Raises:
        ValueError: If *linear* is non-positive.
    """
    if linear <= 0:
        raise ValueError(f"Value must be positive, got {linear}")
    return 10.0 * math.log10(linear)


def validate_power(power: float, p_max: float) -> float:
    """Clip a power value to the valid range ``[0, p_max]``.

    Args:
        power: Proposed transmit power (Watts).
        p_max: Maximum allowable power (Watts).

    Returns:
        Clipped power value.
    """
    return float(np.clip(power, 0.0, p_max))


def normalize_observation(
    obs: np.ndarray,
    obs_min: np.ndarray,
    obs_max: np.ndarray,
) -> np.ndarray:
    """Min-max normalise an observation vector to [0, 1].

    Args:
        obs: Raw observation array.
        obs_min: Per-element minimum values.
        obs_max: Per-element maximum values.

    Returns:
        Normalised observation in [0, 1].
    """
    denom = obs_max - obs_min
    # Avoid division by zero
    denom = np.where(denom == 0, 1.0, denom)
    return np.clip((obs - obs_min) / denom, 0.0, 1.0).astype(np.float32)
