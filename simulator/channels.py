"""
Wireless channel models.
Assignee: Sneha
"""

from typing import Protocol
import numpy as np
from .propagation import calculate_path_loss


class WirelessChannel(Protocol):
    """
    Protocol for Wireless Channel models.
    """

    def generate_gain(self, distance: float, path_loss_exponent: float) -> float:
        """
        Generate the channel power gain |h|^2.
        """
        ...


class RayleighFading(WirelessChannel):
    """
    Rayleigh fading channel model.
    """

    def __init__(self, frequency: float = 2.4e9):
        self.frequency = frequency

    def generate_coefficient(self) -> complex:
        """
        Generate complex Rayleigh fading coefficient g.
        g ~ CN(0, 1)
        """
        real = np.random.normal(0, 1.0 / np.sqrt(2.0))
        imag = np.random.normal(0, 1.0 / np.sqrt(2.0))
        return complex(real, imag)

    def generate_gain(self, distance: float, path_loss_exponent: float) -> float:
        """
        Generate channel power gain |h|^2 = PL * |g|^2.
        """
        path_loss = calculate_path_loss(distance, path_loss_exponent, self.frequency)
        g = self.generate_coefficient()
        power_gain = path_loss * (abs(g) ** 2)
        return float(power_gain)
