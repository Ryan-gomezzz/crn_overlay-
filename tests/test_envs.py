"""
Tests for Gymnasium environment.
Author: Ryan
"""
import numpy as np
from gymnasium.spaces import Box
try:
    from gymnasium.utils.env_checker import check_env
    HAS_CHECK_ENV = True
except ImportError:
    HAS_CHECK_ENV = False

from envs.crn_env import OverlayCRNEnv, make_crn_env


def test_env_creation(env):
    """Create OverlayCRNEnv, verify no errors."""
    assert isinstance(env, OverlayCRNEnv)


def test_env_observation_space(env):
    """Verify observation space shape and bounds."""
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (7,)
    assert np.all(env.observation_space.low == 0.0)
    assert np.all(env.observation_space.high == np.inf)


def test_env_action_space(env):
    """Verify action space shape and bounds."""
    assert isinstance(env.action_space, Box)
    assert env.action_space.shape == (2,)
    assert np.all(env.action_space.low == 0.0)
    assert np.all(env.action_space.high == 1.0)


def test_env_reset(env):
    """Call reset(), verify obs shape matches observation_space."""
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)


def test_env_step(env):
    """Call step() with sampled action, verify 5-tuple return."""
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_step_reward_is_float(env):
    """Verify reward is a float."""
    env.reset()
    action = np.array([0.5, 0.5])
    _, reward, _, _, _ = env.step(action)
    assert isinstance(reward, float)


def test_env_multiple_episodes(env):
    """Run 3 episodes of 10 steps each."""
    for _ in range(3):
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break


def test_env_render(env):
    """Call render() without errors."""
    env.render_mode = "human"
    env.reset()
    env.step(env.action_space.sample())
    env.render() # Should print to console


def test_env_close(env):
    """Call close() without errors."""
    env.close()


def test_env_gymnasium_check(env):
    """Use gymnasium.utils.env_checker.check_env if available."""
    if HAS_CHECK_ENV:
        check_env(env)


def test_env_reset_seed(env):
    """Reset with seed, verify reproducibility."""
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)


def test_make_crn_env():
    """Test factory function."""
    env = make_crn_env()
    assert isinstance(env, OverlayCRNEnv)
