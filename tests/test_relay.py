"""
Tests for Relay Logic.
Assignee: Shreya
"""

# TODO (Shreya): Add tests for decode-and-forward and SINR.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.metrics import calculate_sinr
from simulator.relay import DecodeAndForward


def test_df_forward_success():
    """
    Relay should forward when SINR is above threshold.
    """
    relay = DecodeAndForward(snr_threshold=2)

    relay.receive(signal=10, interference=1, noise=1)

    assert relay.transmit() == 10


def test_df_forward_failure():
    """
    Relay should not forward when SINR is below threshold.
    """
    relay = DecodeAndForward(snr_threshold=10)

    relay.receive(signal=1, interference=5, noise=5)

    assert relay.transmit() == 0.0


def test_calculate_sinr():
    """
    Verify SINR calculation.
    """
    sinr = calculate_sinr(signal=10, interference=2, noise=3)

    assert sinr == 2.0


def test_calculate_sinr_zero_denominator():
    """
    SINR should be infinity when interference and noise are zero.
    """
    sinr = calculate_sinr(signal=10, interference=0, noise=0)

    assert sinr == float("inf")
