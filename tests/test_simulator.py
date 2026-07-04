"""
Tests for Simulator logic.
Author: Ryan
"""
import numpy as np

from simulator.base_model import SimulationState, SimulationResult, SimulatorConfig
from simulator.overlay_model import OverlaySimulator


def test_simulator_config_defaults(default_config):
    """Verify SimulatorConfig has sensible defaults."""
    assert default_config.p_max_su == 1.0
    assert default_config.p_max_relay == 1.0
    assert default_config.p_pt == 1.0
    assert default_config.num_channels == 7
    assert default_config.max_steps == 200


def test_simulator_initialization(simulator):
    """Create OverlaySimulator, verify it initializes."""
    assert isinstance(simulator, OverlaySimulator)
    assert simulator.config is not None


def test_simulator_reset(simulator):
    """Call reset(), verify it returns a SimulationState."""
    state = simulator.reset(seed=42)
    assert isinstance(state, SimulationState)
    assert len(state.channel_gains) == 7
    assert "pt_pr" in state.channel_gains
    assert state.step_count == 0


def test_simulator_step(simulator):
    """Call step() with a valid action, verify SimulationResult."""
    simulator.reset(seed=42)
    action = np.array([0.5, 0.5])
    result = simulator.step(action)
    
    assert isinstance(result, SimulationResult)
    assert isinstance(result.observation, np.ndarray)
    assert result.observation.shape == (7,)
    assert isinstance(result.reward, float)
    assert isinstance(result.info, dict)


def test_simulator_step_action_shape(simulator):
    """Verify action must be shape (2,)."""
    simulator.reset()
    action = np.array([0.5]) # Only 1 dimension
    result = simulator.step(action) # Should automatically append to make it size 2
    assert isinstance(result, SimulationResult)


def test_simulator_multiple_steps(simulator):
    """Run 10 steps, verify no errors."""
    simulator.reset()
    for _ in range(10):
        action = np.random.rand(2)
        result = simulator.step(action)
        assert not result.terminated


def test_simulator_reset_reproducibility(simulator):
    """Reset with same seed twice, verify same initial state."""
    state1 = simulator.reset(seed=123)
    gains1 = dict(state1.channel_gains)
    
    state2 = simulator.reset(seed=123)
    gains2 = dict(state2.channel_gains)
    
    assert gains1 == gains2


def test_simulation_state_fields():
    """Verify all fields exist on SimulationState."""
    state = SimulationState()
    assert hasattr(state, "channel_gains")
    assert hasattr(state, "power_allocation")
    assert hasattr(state, "su_throughput")
    assert hasattr(state, "interference_at_pr")


def test_simulation_result_fields():
    """Verify all fields exist on SimulationResult."""
    result = SimulationResult()
    assert hasattr(result, "observation")
    assert hasattr(result, "reward")
    assert hasattr(result, "terminated")
    assert hasattr(result, "truncated")
    assert hasattr(result, "info")
