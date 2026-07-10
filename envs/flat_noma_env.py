"""
Flattened observation wrapper for NOMA CRN Environment.
Useful for Centralized Single-Agent baselines.
"""
import gymnasium as gym
import numpy as np
from envs.multi_agent_crn_env import make_ma_crn_env

class FlatNOMAEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # obs is (N, 8)
        self.num_agents = env.num_agents
        flat_dim = self.num_agents * 8
        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.flatten(), info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # We need a single relay_decoded scalar for the standard SequenceReplayBuffer
        # We can just use the mean or minimum decode success
        dec_array = info.get("gamma_sr", np.zeros(self.num_agents))
        info["relay_decoded"] = float((np.array(dec_array) > 1e-6).mean())
        
        return obs.flatten(), reward, terminated, truncated, info

def make_flat_noma_env(config_path=None):
    base_env = make_ma_crn_env(config_path)
    return FlatNOMAEnv(base_env)
