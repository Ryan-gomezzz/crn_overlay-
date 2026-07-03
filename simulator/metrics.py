"""
Performance metrics calculation.
Assignee: Shreya
"""

import numpy as np
from scipy.special import erfc


def calculate_sinr(
    signal=None,
    interference=None,
    noise=None,
    signal_power=None,
    interference_power=None,
    noise_power=None,
) -> float:
    """
    Calculate Signal-to-Interference-plus-Noise Ratio (SINR).

    Supports both:
    - signal, interference, noise
    - signal_power, interference_power, noise_power
    """

    if signal is None:
        signal = signal_power
    if interference is None:
        interference = interference_power
    if noise is None:
        noise = noise_power

    denominator = interference + noise

    if denominator == 0:
        return float("inf")

    return float(signal / denominator)


def calculate_capacity(sinr: float, bandwidth: float = 1.0) -> float:
    """
    Calculate Shannon capacity (bps/Hz if bandwidth=1.0).
    C = bandwidth * log2(1 + SINR)
    """
    safe_sinr = max(sinr, 0.0)
    return float(bandwidth * np.log2(1.0 + safe_sinr))


def calculate_throughput(
    sinr: float,
    time_fraction: float = 0.5,
    bandwidth: float = 1.0,
) -> float:
    """
    Calculate throughput accounting for time slot fraction.
    Throughput = time_fraction * capacity
    """
    return float(time_fraction * calculate_capacity(sinr, bandwidth))


def calculate_ber(sinr: float, modulation_order: int = 4) -> float:
    """
    Approximate Bit Error Rate (BER) for M-QAM.
    BER ≈ (4 / log2(M)) * (1 - 1 / sqrt(M)) * Q(sqrt(3 * log2(M) * SINR / (M - 1)))
    where Q(x) = 0.5 * erfc(x / sqrt(2)).
    """
    if sinr <= 0.0:
        return 0.5

    M = modulation_order
    k = np.log2(M)

    # Q(x) calculation using erfc
    # Q(x) = 0.5 * erfc(x / sqrt(2))
    # We want Q(sqrt(3 * k * sinr / (M - 1)))
    x = np.sqrt((3.0 * k * sinr) / (M - 1.0))
    q_val = 0.5 * erfc(x / np.sqrt(2.0))

    ber = (4.0 / k) * (1.0 - 1.0 / np.sqrt(M)) * q_val
    return float(np.clip(ber, 0.0, 0.5))
