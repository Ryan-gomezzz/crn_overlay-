"""
Gymnasium Environment for the Overlay CRN.
Author: Ryan
"""

from typing import Any, Dict, Optional, Tuple
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulator.overlay_model import OverlaySimulator
from simulator.utils import dbm_to_watt


class OverlayCRNEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    Integrates the physical OverlaySimulator.
    """

    metadata = {"render_modes": ["console"]}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config if config is not None else {}

        # 2D Action space: [SUs power fraction, SUR relay power fraction]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # 4D Observation space: [|h_pt_pr|^2, |h_sus_sur|^2, |h_sur_sud|^2, |h_sus_pr|^2]
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(4,), dtype=np.float32
        )

        # Initialize physical simulator
        self.simulator = OverlaySimulator(self.config)

        # Env configs
        camo_cfg = self.config.get("camo_td3", {})
        self.pu_rate_threshold = camo_cfg.get("pu_rate_threshold", 1.5)
        
        # We define limits for Interference (Watts) and Energy (Watts)
        # Default limits: Interference = -50 dBm (1e-8 Watts)
        # Default Energy Limit = 0.1 Watts
        self.interference_limit = dbm_to_watt(camo_cfg.get("interference_limit_dbm", -50.0))
        self.energy_limit = camo_cfg.get("energy_limit_watts", 0.1)

        # Penalty coefficients for standard TD3 scalar reward
        self.penalty_coef_inf = camo_cfg.get("penalty_coef_inf", 10.0)
        self.penalty_coef_nrg = camo_cfg.get("penalty_coef_nrg", 10.0)

        # History sequence variables for online tracking
        self.history_len = camo_cfg.get("history_length", 10)
        self.obs_history = deque(maxlen=self.history_len)
        self.act_history = deque(maxlen=self.history_len)
        self.dec_history = deque(maxlen=self.history_len)
        self.out_history = deque(maxlen=self.history_len)

        # Track steps in current episode
        self.current_step = 0
        self.max_steps = self.config.get("simulation", {}).get("time_steps_per_episode", 300)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment to an initial state.
        """
        super().reset(seed=seed)
        
        # Reset simulator
        sim_data = self.simulator.reset()
        obs = sim_data["state"]

        self.current_step = 0
        
        # Initialize history buffers with zeros
        self.obs_history.clear()
        self.act_history.clear()
        self.dec_history.clear()
        self.out_history.clear()
        
        for _ in range(self.history_len):
            self.obs_history.append(np.zeros(self.observation_space.shape, dtype=np.float32))
            self.act_history.append(np.zeros(self.action_space.shape, dtype=np.float32))
            self.dec_history.append(np.zeros((1,), dtype=np.float32))
            self.out_history.append(np.zeros((1,), dtype=np.float32))

        # The first observation is pushed to history
        self.obs_history.append(obs)

        info = {
            "obs_history": np.array(self.obs_history, dtype=np.float32),
            "act_history": np.array(self.act_history, dtype=np.float32),
            "dec_history": np.array(self.dec_history, dtype=np.float32),
            "out_history": np.array(self.out_history, dtype=np.float32),
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Run one timestep of the environment's dynamics.
        """
        self.current_step += 1

        # Clip action to ensure valid power fractions
        act = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Step the physical simulator
        sim_data = self.simulator.step(act)
        obs = sim_data["next_state"]
        metrics = sim_data["metrics"]

        # Extract/compute history states early for sequence deques
        relay_decoded = metrics.get("relay_decoded", 0.0)
        outage = 1.0 if metrics["throughput_p"] < self.pu_rate_threshold else 0.0

        # Push to history
        self.act_history.append(act)
        self.obs_history.append(obs)
        self.dec_history.append(np.array([relay_decoded], dtype=np.float32))
        self.out_history.append(np.array([outage], dtype=np.float32))

        # Compute separate components
        throughput_p = metrics["throughput_p"]
        throughput_s = metrics["throughput_s"]
        power_s1 = metrics["power_s1"]
        power_rel = metrics["power_rel"]
        ber = metrics["ber_s"]

        # Calculate actual physical interference caused to PR (Watts)
        # TS1: SUs to PR; TS2: SUR to PR
        h_sus_pr = self.simulator.channel_gains["sus_pr"]
        h_sur_pr = self.simulator.channel_gains["sur_pr"]
        interference_ts1 = power_s1 * h_sus_pr
        interference_ts2 = (1.0 - metrics["beta"]) * power_rel * h_sur_pr
        total_interference = interference_ts1 + interference_ts2

        # Calculate energy consumed (average across two slots of 0.5 fraction each)
        total_energy = 0.5 * power_s1 + 0.5 * power_rel

        # Multi-objective individual metrics/rewards
        # Throughput reward = secondary rate
        throughput_reward = throughput_s

        # Interference constraint: total_interference <= interference_limit
        # Violation is the positive difference
        interference_violation = max(0.0, total_interference - self.interference_limit)
        # Interference reward can be defined as negative violation (or negative value)
        interference_reward = -total_interference

        # Energy constraint: total_energy <= energy_limit
        energy_violation = max(0.0, total_energy - self.energy_limit)
        energy_reward = -total_energy

        # Calculate standard scalar reward (for standard TD3)
        scalar_reward = (
            throughput_reward
            - self.penalty_coef_inf * interference_violation
            - self.penalty_coef_nrg * energy_violation
        )

        # Check termination & truncation
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Outage event: primary rate drops below target
        outage = 1.0 if throughput_p < self.pu_rate_threshold else 0.0

        # Construct info dict
        info = {
            "throughput_reward": throughput_reward,
            "interference_reward": interference_reward,
            "energy_reward": energy_reward,
            "interference_violation": interference_violation,
            "energy_violation": energy_violation,
            "primary_throughput": throughput_p,
            "outage": outage,
            "ber": ber,
            "average_power": total_energy,
            "obs_history": np.array(self.obs_history, dtype=np.float32),
            "act_history": np.array(self.act_history, dtype=np.float32),
            "dec_history": np.array(self.dec_history, dtype=np.float32),
            "out_history": np.array(self.out_history, dtype=np.float32),
            "relay_decoded": relay_decoded,
        }

        return obs, scalar_reward, terminated, truncated, info

    def render(self):
        """
        Render the environment state.
        """
        print("Rendering CRN Environment state.")

    def close(self):
        """
        Cleanup resources.
        """
        pass
