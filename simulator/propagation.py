"""
Propagation and path loss models.
Assignee: Sneha
"""

import numpy as np


def calculate_path_loss(
    distance: float,
    path_loss_exponent: float = 3.5,
    frequency: float = 2.4e9,
) -> float:
    """
    Calculate path loss in linear scale.
    Formula: PL = distance^(-path_loss_exponent)
    To prevent division by zero, distance is clamped to a minimum value of 1e-3.
    """
    safe_distance = max(distance, 1e-3)
    return float(safe_distance ** (-path_loss_exponent))
