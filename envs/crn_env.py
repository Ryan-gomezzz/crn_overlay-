"""
Overlay CRN Environment.
Assignee: Ryan
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from simulator.overlay_model import OverlayCRNModel
from simulator.utils import dbm_to_watt, watt_to_dbm


class OverlayCRNEnv(gym.Env):
    """Gymnasium environment for Overlay Cognitive Radio Networks."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict):
        """Initialize the CRN environment.

        Args:
            config (dict): Configuration dictionary with keys:
                - simulation: Simulation parameters
                - camo_td3: CAMO-specific parameters
        """
        self.config = config
        sim_cfg = config.get("simulation", {})
        camo_cfg = config.get("camo_td3", {})

        # Action space: [power_secondary, bandwidth_ratio]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # 4D Observation space:
        # [|h_pt_pr|^2, |h_sus_sur|^2, |h_sur_sud|^2, |h_sus_pr|^2]
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(4,), dtype=np.float32
        )

        # Initialize model
        self.model = OverlayCRNModel(config)

        # Extract CAMO parameters
        self.energy_limit = camo_cfg.get("energy_limit_watts", 0.1)
        # Interference limit: Default -50 dBm (1e-8 Watts)
        self.interference_limit = dbm_to_watt(
            camo_cfg.get("interference_limit_dbm", -50.0)
        )

        # History length for context
        self.history_len = camo_cfg.get("history_length", 10)
        self.obs_history = []
        self.act_history = []
        self.dec_history = []
        self.out_history = []

        # Track steps in current episode
        self.current_step = 0
        self.max_steps = self.config.get(
            "simulation", {}
        ).get("time_steps_per_episode", 100)

    def reset(
        self,
        seed: int = None,
        options: dict = None,
    ):
        """Reset the environment.

        Args:
            seed (int, optional): Random seed. Defaults to None.
            options (dict, optional): Options. Defaults to None.

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)

        # Reset model
        self.model.reset()
        self.current_step = 0

        # Initialize histories
        for _ in range(self.history_len):
            self.obs_history.append(
                np.zeros(self.observation_space.shape, dtype=np.float32)
            )
            self.act_history.append(
                np.zeros(self.action_space.shape, dtype=np.float32)
            )
            self.dec_history.append(np.zeros((1,), dtype=np.float32))
            self.out_history.append(np.zeros((1,), dtype=np.float32))

        # Get initial observation
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
        """Execute one step in the environment.

        Args:
            action (np.ndarray): Agent's action [power, bw_ratio]

        Returns:
            tuple: (observation, reward, done, info)
        """
        self.current_step += 1

        # Clip action to valid range
        action = np.clip(action, 0.0, 1.0)
        power_secondary = action[0]
        bw_secondary = action[1]

        # Run model step
        rewards, metrics = self.model.step(
            power_secondary, bw_secondary
        )

        # Extract observation
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
            np.array(
                [1.0 if metrics.get("relay_decoded", False) else 0.0],
                dtype=np.float32,
            )
        )
        self.out_history.pop(0)
        self.out_history.append(
            np.array(
                [metrics.get("outage", 0.0)],
                dtype=np.float32,
            )
        )

        # Compute info
        info = {
            "obs_history": np.array(self.obs_history, dtype=np.float32),
            "act_history": np.array(self.act_history, dtype=np.float32),
            "dec_history": np.array(self.dec_history, dtype=np.float32),
            "out_history": np.array(self.out_history, dtype=np.float32),
            **metrics,
        }

        # Determine termination
        done = self.current_step >= self.max_steps

        # Compute reward based on rewards dict
        reward = rewards.get("total", 0.0)

        return obs, reward, done, False, info
