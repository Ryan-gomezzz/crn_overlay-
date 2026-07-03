"""
Interference modeling.
Assignee: Shreya
"""

from typing import List


def calculate_received_power(
    transmit_power: float, channel_gain: float
) -> float:
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
