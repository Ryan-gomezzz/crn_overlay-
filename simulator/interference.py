"""
Interference modeling.
Assignee: Shreya
"""


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

    return total_interference

from typing import List


def calculate_received_power(transmit_power: float, channel_gain: float) -> float:
    """
    Compute received power in linear scale (Watts).
    P_rx = P_tx * |h|^2
    """
    return transmit_power * channel_gain


def calculate_interference(received_powers: List[float]) -> float:
    """
    Sum the received power from all interfering sources.
    """
    return float(sum(received_powers))
 main
