"""
Pytest configuration and fixtures.
Author: Ryan
"""
import os
import sys
import pytest

# Ensure the project root is in sys.path so tests can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.base_model import SimulatorConfig
from simulator.overlay_model import OverlaySimulator
from envs.crn_env import OverlayCRNEnv


@pytest.fixture
def default_config():
    """Return a SimulatorConfig with default values."""
    return SimulatorConfig()


@pytest.fixture
def simulator(default_config):
    """Return a fresh OverlaySimulator instance."""
    return OverlaySimulator(default_config)


@pytest.fixture
def env():
    """Return a fresh OverlayCRNEnv instance."""
    return OverlayCRNEnv()
