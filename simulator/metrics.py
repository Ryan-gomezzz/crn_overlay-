"""
Performance metrics calculation.
Assignee: Shreya
"""

# TODO (Shreya): Implement SINR, capacity, and throughput equations.
# Expected Inputs: signal power, interference power, noise power.
# Expected Outputs: SINR, achievable rate.
# Reference: docs/team_guides/relay_module.md

import math


def calculate_sinr(signal: float, interference: float, noise: float) -> float:
    denominator = interference + noise
    if denominator == 0:
        return float("inf")
    return signal / denominator


def calculate_capacity(bandwidth: float, sinr: float) -> float:
    return bandwidth * math.log2(1 + sinr)


def calculate_throughput(capacity: float, efficiency: float = 1.0) -> float:
    return capacity * efficiency