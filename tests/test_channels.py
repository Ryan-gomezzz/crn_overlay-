"""
Tests for Wireless Channels.
Assignee: Sneha
"""

import numpy as np

from simulator.channels import RayleighFading
from simulator.propagation import calculate_path_loss


def test_path_loss_positive():
    """
    Path loss should always be positive.
    """
    pl = calculate_path_loss(10)

    assert pl > 0


def test_path_loss_decreases_with_distance():
    """
    Path loss should decrease as distance increases.
    """
    near = calculate_path_loss(1)

    far = calculate_path_loss(10)

    assert near > far


def test_path_loss_at_small_distance():
    """
    Path loss should remain finite for very small distances.
    """
    pl = calculate_path_loss(0)

    assert np.isfinite(pl)


def test_rayleigh_gain_is_non_negative():
    """
    Channel gain should never be negative.
    """
    channel = RayleighFading()

    gain = channel.generate_gain(
        distance=5,
        path_loss_exponent=3,
    )

    assert gain >= 0


def test_rayleigh_gain_is_float():
    """
    Generated channel gain should be a float.
    """
    channel = RayleighFading()

    gain = channel.generate_gain(
        distance=5,
        path_loss_exponent=3,
    )

    assert isinstance(gain, float)


def test_rayleigh_coefficient_is_complex():
    """
    Rayleigh coefficient should be complex.
    """
    channel = RayleighFading()

    g = channel.generate_coefficient()

    assert isinstance(g, complex)
