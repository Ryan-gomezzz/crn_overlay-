"""Environment and simulator tests aligned with docs/SYSTEM_MODEL.md.

Verifies the spaces, observation dimensionality, the N+2 action layout,
and that the reward equals the measured secondary sum-rate.
"""

import math

import numpy as np

from envs.multi_agent_crn_env import make_ma_crn_env
from tests.conftest import CONFIG_PATH


def test_action_and_observation_spaces(num_su):
    """Action space is N+2 (SU powers + relay power + alpha); obs is (N, 8)."""
    env = make_ma_crn_env(CONFIG_PATH)
    assert env.action_space.shape == (num_su + 2,)
    assert env.observation_space.shape == (num_su, 8)


def test_reset_returns_obs_and_history(num_su):
    env = make_ma_crn_env(CONFIG_PATH)
    obs, info = env.reset(seed=42)
    assert obs.shape == (num_su, 8)
    # CTDE recurrent belief needs the observation history in the info dict.
    for key in ("obs_history", "act_history", "dec_history", "out_history"):
        assert key in info


def test_step_reward_matches_sum_rate(num_su):
    """SYSTEM_MODEL: R_SU = sum_i 1/2 log2(1 + gamma_e2e_i).

    With no interference violation the reward equals the SU sum-rate.
    """
    env = make_ma_crn_env(CONFIG_PATH)
    env.reset(seed=7)
    action = np.zeros(num_su + 2, dtype=np.float32)  # zero power -> no violation
    _, reward, _, _, info = env.step(action)
    assert not info["constraint_violated"]
    assert math.isclose(reward, info["sum_rate"], rel_tol=1e-6)


def test_reward_is_finite_over_random_rollout(num_su):
    env = make_ma_crn_env(CONFIG_PATH)
    env.reset(seed=1)
    for _ in range(50):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        assert np.all(np.isfinite(obs))
        if term or trunc:
            env.reset()
