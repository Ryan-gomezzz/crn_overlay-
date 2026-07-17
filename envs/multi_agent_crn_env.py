"""
Multi-Agent Gymnasium-style environment for NOMA Overlay CRN.
Author: Ryan

Wraps NOMAOverlaySimulator to provide history tracking and a standard
interface for MATD3.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from simulator.noma_overlay_model import NOMAConfig, NOMAOverlaySimulator


class MultiAgentCRNEnv(gym.Env):
    """
    Multi-Agent Environment for NOMA Overlay CRN.
    
    Action Space: Box(0, 1, shape=(N+2,))
        - [p_su_1, ..., p_su_N, p_relay, alpha]

    Observation Space: Box(-inf, inf, shape=(N, 8))
        - N agents, each with 8 features
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._raw_config = config or {}
        
        # Build NOMAConfig
        cfg = NOMAConfig()
        
        if "network" in self._raw_config:
            net_cfg = self._raw_config["network"]
            if "p_primary" in net_cfg: cfg.p_primary_dbm = float(net_cfg["p_primary"])
            if "p_max_su" in net_cfg: cfg.p_max_su_dbm = float(net_cfg["p_max_su"])
            if "pt_coords" in net_cfg: cfg.pt_coords = net_cfg["pt_coords"]
            if "pr_coords" in net_cfg: cfg.pr_coords = net_cfg["pr_coords"]
            
        if "channel" in self._raw_config:
            chan_cfg = self._raw_config["channel"]
            if "noise_power_dbm" in chan_cfg: cfg.noise_power_dbm = float(chan_cfg["noise_power_dbm"])
            if "path_loss_exponent" in chan_cfg: cfg.path_loss_exponent = float(chan_cfg["path_loss_exponent"])
            if "csi_error_variance" in chan_cfg: cfg.csi_error_variance = float(chan_cfg["csi_error_variance"])

        if "multi_user" in self._raw_config:
            mu_cfg = self._raw_config["multi_user"]
            if "num_su" in mu_cfg: cfg.num_su = mu_cfg["num_su"]
            if "su_coords" in mu_cfg: cfg.su_coords = mu_cfg["su_coords"]
            if "sud_coords" in mu_cfg: cfg.sud_coords = mu_cfg["sud_coords"]
            if "sur_coords" in mu_cfg: cfg.sur_coords = mu_cfg["sur_coords"]
            if "interference_threshold_dbm" in mu_cfg: cfg.interference_threshold_dbm = float(mu_cfg["interference_threshold_dbm"])
            
        if "camo_td3" in self._raw_config:
            camo_cfg = self._raw_config["camo_td3"]
            if "penalty_coef_inf" in camo_cfg:
                cfg.penalty_weight = float(camo_cfg["penalty_coef_inf"])
            
        self.simulator = NOMAOverlaySimulator(cfg)
        self.num_agents = self.simulator.num_agents
        
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_agents + 2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_agents, 8), dtype=np.float32
        )
        
        # History Tracking
        hist_len = self._raw_config.get("camo_td3", {}).get("history_length", 10)
        self.history_length = hist_len
        
        # Buffer shapes: (N, hist_len, dim)
        self._obs_history = np.zeros((self.num_agents, hist_len, 8), dtype=np.float32)
        self._act_history = np.zeros((self.num_agents, hist_len, 1), dtype=np.float32)  # Each SU's own action
        self._dec_history = np.zeros((self.num_agents, hist_len, 1), dtype=np.float32)  # Was this SU decoded?
        self._out_history = np.zeros((self.num_agents, hist_len, 1), dtype=np.float32)  # Global outage
        
        self._step_count = 0
        self._episode_reward = 0.0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        sim_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        
        res = self.simulator.reset(seed=sim_seed)
        obs = res["observations"]  # (N, 8)
        
        self._step_count = 0
        self._episode_reward = 0.0
        
        self._obs_history.fill(0.0)
        self._act_history.fill(0.0)
        self._dec_history.fill(0.0)
        self._out_history.fill(0.0)
        
        self._obs_history[:, -1, :] = obs
        
        info = res["info"]
        info["obs_history"] = self._obs_history.copy()
        info["act_history"] = self._act_history.copy()
        info["dec_history"] = self._dec_history.copy()
        info["out_history"] = self._out_history.copy()
        
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        res = self.simulator.step(action)
        obs = res["observations"]
        reward = res["reward"]
        
        self._step_count += 1
        self._episode_reward += reward
        
        info = res["info"]
        info["episode_reward"] = self._episode_reward
        info["episode_length"] = self._step_count
        
        # Compatibility keys for runner logging
        info["throughput_reward"] = info["sum_rate"]
        info["pu_throughput"] = info["pu_rate"]
        info["primary_throughput"] = info["pu_rate"]
        info["sinr_su"] = float(np.mean(info["gamma_e2e"])) if "gamma_e2e" in info else 0.0
        info["average_power"] = (sum(info["p_su_watts"]) + info["p_relay_watts"]) / (self.num_agents + 1)
        info["outage"] = 1.0 if info["constraint_violated"] else 0.0
        info["su_outage"] = 1.0 if info["sum_rate"] <= 0.0 else 0.0
        
        # Update histories
        self._obs_history = np.roll(self._obs_history, -1, axis=1)
        self._obs_history[:, -1, :] = obs
        
        self._act_history = np.roll(self._act_history, -1, axis=1)
        self._act_history[:, -1, 0] = action[:self.num_agents]  # SU powers
        
        # Relay decodes user i if user i is in sic_order and gamma_sr is decent? 
        # Actually in NOMA simulator, everyone is decoded in sic_order, but we can just use 1 for simplicity 
        # or base it on gamma_sr > 0
        dec = (np.array(info["gamma_sr"]) > 1e-6).astype(np.float32)
        info["relay_decoded"] = float(np.mean(dec))
        self._dec_history = np.roll(self._dec_history, -1, axis=1)
        self._dec_history[:, -1, 0] = dec
        
        self._out_history = np.roll(self._out_history, -1, axis=1)
        self._out_history[:, -1, 0] = info["outage"]
        
        info["obs_history"] = self._obs_history.copy()
        info["act_history"] = self._act_history.copy()
        info["dec_history"] = self._dec_history.copy()
        info["out_history"] = self._out_history.copy()
        
        return obs, reward, res["terminated"], res["truncated"], info

def make_ma_crn_env(config_path: Optional[str] = None, render_mode: Optional[str] = None) -> MultiAgentCRNEnv:
    config = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
            if raw: config = raw
    return MultiAgentCRNEnv(config=config, render_mode=render_mode)
