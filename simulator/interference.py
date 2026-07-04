"""
Interference modeling.
Assignee: Shreya
"""

from typing import List


def compute_interference(transmit_powers, channel_gains):
    """
    transmit_powers: list of powers [P1, P2, ...]
    channel_gains: list of gains [h1, h2, ...]
    """
    if len(transmit_powers) != len(channel_gains):
        raise ValueError("Mismatch in powers and channel gains")

    total_interference = 0.0

    for p, h in zip(transmit_powers, channel_gains):
        total_interference += p * h

    return float(total_interference)


def calculate_received_power(transmit_power: float, channel_gain: float) -> float:
    """
    P_rx = P_tx * |h|^2
    """
    return float(transmit_power * channel_gain)


def calculate_interference(received_powers: List[float]) -> float:
    """
    Sum of interfering received powers.
    """
    return float(sum(received_powers))


def calculate_sinr(
    signal_power: float,
    interference_power: float,
    noise_power: float,
) -> float:
    """
    SINR = S / (I + N)
    Safe numerical version for stability.
    """
    denom = interference_power + noise_power

    if denom <= 1e-15:
        denom = 1e-15

    return float(signal_power / denom)