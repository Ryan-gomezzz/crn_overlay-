"""
Overlay CRN Environment.
Assignee: Ryan
"""

from collections import deque
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulator.overlay_model import OverlayCRNModel
from simulator.utils import dbm_to_watt, watt_to_dbm


class OverlayCRNEnv(gym.Env):
    """Gymnasium environment for Overlay Cognitive Radio Networks."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict):
        """Initialize the CRN environment."""

        self.config = config
        sim_cfg = config.get("simulation", {})
        camo_cfg = config.get("camo_td3", {})

        # Action space: [power_secondary, bandwidth_ratio]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation space (4D channel gains)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(4,), dtype=np.float32
        )

        # Initialize model
        self.model = OverlayCRNModel(config)

        # Env configs
        self.pu_rate_threshold = camo_cfg.get("pu_rate_threshold", 1.5)

        self.energy_limit = camo_cfg.get("energy_limit_watts", 0.1)

        self.interference_limit = dbm_to_watt(
            camo_cfg.get("interference_limit_dbm", -50.0)
        )

        # History length for context
        self.history_len = camo_cfg.get("history_length", 10)
        self.obs_history = []
        self.act_history = []
        self.dec_history = []
        self.out_history = []

        self.current_step = 0
        self.max_steps = self.config.get("simulation", {}).get(
            "time_steps_per_episode", 100
        )

    def reset(self, seed: int = None, options: dict = None):
        """Reset the environment."""
        super().reset(seed=seed)

        self.model.reset()
        self.current_step = 0

        # Initialize history buffers
        self.obs_history.clear()
        self.act_history.clear()
        self.dec_history.clear()
        self.out_history.clear()

        for _ in range(self.history_len):
            self.obs_history.append(
                np.zeros(self.observation_space.shape, dtype=np.float32)
            )
            self.act_history.append(
                np.zeros(self.action_space.shape, dtype=np.float32)
            )
            self.dec_history.append(np.zeros((1,), dtype=np.float32))
            self.out_history.append(np.zeros((1,), dtype=np.float32))

        obs = np.array(
            [
                self.model.channel_gains["pt_pr"],
                self.model.channel_gains["sus_sur"],
                self.model.channel_gains["sur_sud"],
                self.model.channel_gains["sus_pr"],
            ],
            dtype=np.float32,
        )

        info = {
            "obs_history": np.array(self.obs_history, dtype=np.float32),
            "act_history": np.array(self.act_history, dtype=np.float32),
            "dec_history": np.array(self.dec_history, dtype=np.float32),
            "out_history": np.array(self.out_history, dtype=np.float32),
        }

        return obs, info

    def step(self, action: np.ndarray):
        """Execute one step in the environment."""

        self.current_step += 1

        # Clip action
        action = np.clip(action, 0.0, 1.0)

        power_secondary = action[0]
        bw_secondary = action[1]

        # Run model step
        rewards, metrics = self.model.step(power_secondary, bw_secondary)

        # Observation
        obs = np.array(
            [
                self.model.channel_gains["pt_pr"],
                self.model.channel_gains["sus_sur"],
                self.model.channel_gains["sur_sud"],
                self.model.channel_gains["sus_pr"],
            ],
            dtype=np.float32,
        )

        # Update histories
        self.obs_history.pop(0)
        self.obs_history.append(obs)

        self.act_history.pop(0)
        self.act_history.append(np.array(action, dtype=np.float32))

        self.dec_history.pop(0)
        self.dec_history.append(
            np.array([1.0 if metrics.get("relay_decoded", False) else 0.0])
        )

        self.out_history.pop(0)
        self.out_history.append(
            np.array([metrics.get("outage", 0.0)])
        )

        info = {
            "obs_history": np.array(self.obs_history, dtype=np.float32),
            "act_history": np.array(self.act_history, dtype=np.float32),
            "dec_history": np.array(self.dec_history, dtype=np.float32),
            "out_history": np.array(self.out_history, dtype=np.float32),
            **metrics,
        }

        done = self.current_step >= self.max_steps
        reward = rewards.get("total", 0.0)

        return obs, reward, done, False, info