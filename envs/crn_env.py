"""
Gymnasium Environment for the Overlay CRN.
Author: Ryan
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class OverlayCRNEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    """

    metadata = {"render_modes": ["console"]}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        # TODO (Ryan): Define actual action and observation spaces based on config.
        # Example: Action space could be power allocation (continuous).
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Example: Observation space could be channel gains.
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(4,), dtype=np.float32
        )

        # TODO (Ryan): Initialize the abstract simulator.
        self.simulator = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment to an initial state and returns an initial observation.
        """
        super().reset(seed=seed)
        # TODO (Ryan): Call self.simulator.reset()
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Run one timestep of the environment's dynamics.
        """
        # TODO (Ryan): Call self.simulator.step(action)
        # Compute reward based on simulator metrics.
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment.
        """
        print("Rendering CRN Environment state.")

    def close(self):
        """
        Cleanup resources.
        """
        pass
