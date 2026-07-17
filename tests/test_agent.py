"""MATD3 agent tests: action shape, and a checkpoint save/load round-trip.

The save/load round-trip is a regression guard: these methods were once
no-op stubs, which silently invalidated evaluate/resume.
"""

import numpy as np
import torch

from agents.matd3 import MATD3Agent
from envs.multi_agent_crn_env import make_ma_crn_env
from tests.conftest import CONFIG_PATH


def _make_agent(config):
    return MATD3Agent(config, device="cpu")


def test_select_action_shape(config, num_su):
    env = make_ma_crn_env(CONFIG_PATH)
    obs, info = env.reset(seed=42)
    agent = _make_agent(config)
    action = agent.select_action(obs, info, explore=False)
    assert action.shape == (num_su + 2,)
    assert np.all(action >= 0.0) and np.all(action <= 1.0)


def test_save_load_roundtrip(config, tmp_path):
    """save() must write a file and load() must restore identical weights."""
    env = make_ma_crn_env(CONFIG_PATH)
    obs, info = env.reset(seed=42)

    agent = _make_agent(config)
    ckpt = tmp_path / "agent.pth"
    agent.save(str(ckpt))
    assert ckpt.exists(), "save() did not write a checkpoint file"

    # A fresh agent has different random weights; deterministic actions differ.
    other = _make_agent(config)
    a_before = other.select_action(obs, info, explore=False)

    other.load(str(ckpt))
    a_after = other.select_action(obs, info, explore=False)
    a_ref = agent.select_action(obs, info, explore=False)

    # After loading, the two agents produce the same deterministic action.
    assert np.allclose(a_after, a_ref, atol=1e-5)
    # And loading actually changed something relative to the fresh init.
    assert not np.allclose(a_before, a_after, atol=1e-6)


def test_load_restores_training_counter(config, tmp_path):
    agent = _make_agent(config)
    agent.total_it = 1234
    ckpt = tmp_path / "counter.pth"
    agent.save(str(ckpt))

    restored = _make_agent(config)
    restored.load(str(ckpt))
    assert restored.total_it == 1234
