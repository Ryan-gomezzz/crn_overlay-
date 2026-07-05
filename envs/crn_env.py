"""
Gymnasium environment for the Overlay Cognitive Radio Network.
Author: Ryan

Wraps the ``OverlaySimulator`` in a standard Gymnasium interface so
that any Stable-Baselines3 (or compatible) RL algorithm can train on
the CRN power-allocation problem without modification.

Usage:
    env = OverlayCRNEnv()          # uses default SimulatorConfig
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from simulator.base_model import SimulatorConfig
from simulator.overlay_model import OverlaySimulator


class OverlayCRNEnv(gym.Env):
    """Gymnasium environment for Overlay CRN power allocation.

    **Action space** — ``Box(0, 1, shape=(2,))``:
        ``[p_su_normalised, p_relay_normalised]``.

    **Observation space** — ``Box(0, +inf, shape=(num_channels,))``:
        Effective channel gains for all network links.

    Args:
        config: Optional dictionary of overrides (same keys as YAML).
        render_mode: ``"human"`` for console output, or ``None``.
    """

    metadata: Dict[str, List[str]] = {
        "render_modes": ["human"],
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode

        # Build SimulatorConfig from dict overrides
        sim_cfg = _build_simulator_config(config or {})
        self._config = sim_cfg
        self._max_steps = sim_cfg.max_steps
        self._num_channels = sim_cfg.num_channels

        # Create the simulator
        self.simulator = OverlaySimulator(config=sim_cfg)

        # -- Spaces -------------------------------------------------
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self._num_channels,),
            dtype=np.float32,
        )

        # Episode bookkeeping
        self._step_count: int = 0
        self._episode_reward: float = 0.0
        self._current_state = None

        # History Tracking for CAMO-TD3 / Overlay-TD3
        cfg = config or {}
        self.history_length = cfg.get("camo_td3", {}).get("history_length", 10)
        self._obs_history = np.zeros((self.history_length, self._num_channels), dtype=np.float32)
        self._act_history = np.zeros((self.history_length, 2), dtype=np.float32)
        self._dec_history = np.zeros((self.history_length, 1), dtype=np.float32)
        self._out_history = np.zeros((self.history_length, 1), dtype=np.float32)

    # ----------------------------------------------------------------
    # Gymnasium API
    # ----------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: RNG seed for reproducibility.
            options: Unused — reserved for future extensions.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)

        # Use the gymnasium RNG to derive a simulator seed
        sim_seed = (
            seed
            if seed is not None
            else int(self.np_random.integers(0, 2**31))
        )
        state = self.simulator.reset(seed=sim_seed)
        self._current_state = state

        self._step_count = 0
        self._episode_reward = 0.0

        # Reset history buffers
        self._obs_history.fill(0.0)
        self._act_history.fill(0.0)
        self._dec_history.fill(0.0)
        self._out_history.fill(0.0)

        obs = self.simulator.get_observation(state)
        
        # Add initial observation
        self._obs_history = np.roll(self._obs_history, -1, axis=0)
        self._obs_history[-1] = obs

        info: Dict[str, Any] = {
            "step": 0,
            "obs_history": self._obs_history.copy(),
            "act_history": self._act_history.copy(),
            "dec_history": self._dec_history.copy(),
            "out_history": self._out_history.copy(),
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Run one timestep of the environment.

        Args:
            action: Shape ``(2,)`` power allocations in [0, 1].

        Returns:
            ``(observation, reward, terminated, truncated, info)``
        """
        result = self.simulator.step(action)

        self._step_count += 1
        self._episode_reward += result.reward

        # Build the info dictionary with correct keys expected by logging and buffers
        info = dict(result.info)
        info["episode_reward"] = self._episode_reward
        info["episode_length"] = self._step_count
        
        info["throughput_reward"] = info.get("su_throughput", 0.0)
        info["primary_throughput"] = info.get("pu_throughput", 0.0)
        info["average_power"] = (info.get("p_su", 0.0) + info.get("p_relay", 0.0)) / 2.0
        info["outage"] = 1.0 if info.get("interference_at_pr", 0.0) > getattr(self._config, "interference_threshold", 0.1) else 0.0
        info["su_outage"] = 1.0 if info["throughput_reward"] <= 0.0 else 0.0

        # Update histories
        self._act_history = np.roll(self._act_history, -1, axis=0)
        self._act_history[-1] = action

        self._obs_history = np.roll(self._obs_history, -1, axis=0)
        self._obs_history[-1] = result.observation

        self._dec_history = np.roll(self._dec_history, -1, axis=0)
        self._dec_history[-1] = [float(info.get("relay_decoded", 0.0))]

        self._out_history = np.roll(self._out_history, -1, axis=0)
        self._out_history[-1] = [info["outage"]]

        info["obs_history"] = self._obs_history.copy()
        info["act_history"] = self._act_history.copy()
        info["dec_history"] = self._dec_history.copy()
        info["out_history"] = self._out_history.copy()

        return (
            result.observation,
            float(result.reward),
            result.terminated,
            result.truncated,
            info,
        )

    def render(self) -> None:
        """Print current state metrics to the console."""
        if self.render_mode != "human":
            return
        if self._current_state is None:
            print("[OverlayCRNEnv] No state to render.")
            return
        s = self._current_state
        print(
            f"[Step {self._step_count}]  "
            f"SU-tput={s.su_throughput:.4f}  "
            f"PU-tput={s.pu_throughput:.4f}  "
            f"SINR_SU={s.sinr_su:.4f}  "
            f"SINR_PU={s.sinr_pu:.4f}  "
            f"Int@PR={s.interference_at_pr:.6f}  "
            f"Relay={'Y' if s.relay_decoded else 'N'}"
        )

    def close(self) -> None:
        """Clean up resources (no-op for this environment)."""
        pass


# ----------------------------------------------------------------
# Factory helper
# ----------------------------------------------------------------


def make_crn_env(
    config_path: Optional[str] = None,
    render_mode: Optional[str] = None,
) -> OverlayCRNEnv:
    """Create an ``OverlayCRNEnv`` from a YAML config file.

    Args:
        config_path: Path to a YAML configuration file. If ``None``,
            uses default ``SimulatorConfig`` values.
        render_mode: Gymnasium render mode.

    Returns:
        A ready-to-use ``OverlayCRNEnv`` instance.
    """
    config: Dict[str, Any] = {}
    if config_path is not None and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if raw is not None:
            config = raw
    return OverlayCRNEnv(config=config, render_mode=render_mode)


# ----------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------

# Mapping from flat YAML keys to SimulatorConfig field names.
_YAML_KEY_MAP: Dict[str, str] = {
    "d_pt_pr": "d_pt_pr",
    "d_pt_relay": "d_pt_relay",
    "d_su_relay": "d_su_relay",
    "d_relay_pr": "d_relay_pr",
    "d_relay_sud": "d_relay_sud",
    "d_su_sud": "d_su_sud",
    "d_pt_sud": "d_pt_sud",
    "p_max_su": "p_max_su",
    "p_max_relay": "p_max_relay",
    "p_pt": "p_pt",
    "noise_power": "noise_power",
    "bandwidth": "bandwidth",
    "num_channels": "num_channels",
    "max_steps": "max_steps",
    "interference_threshold": "interference_threshold",
    "penalty_weight": "penalty_weight",
}


def _build_simulator_config(
    raw: Dict[str, Any],
) -> SimulatorConfig:
    """Merge a (possibly nested) YAML dict into a SimulatorConfig.

    Supports both flat and nested layouts:
    ``network.d_pt_pr`` or ``environment.interference_threshold``.
    """
    flat: Dict[str, Any] = {}

    # Flatten nested sections
    for section_key in ("network", "channel", "simulation", "environment"):
        section = raw.get(section_key)
        if isinstance(section, dict):
            flat.update(section)

    # Also accept top-level keys
    flat.update(
        {k: v for k, v in raw.items() if not isinstance(v, dict)}
    )

    # Build kwargs for the dataclass
    kwargs: Dict[str, Any] = {}
    for yaml_key, field_name in _YAML_KEY_MAP.items():
        if yaml_key in flat:
            kwargs[field_name] = flat[yaml_key]

    return SimulatorConfig(**kwargs)
