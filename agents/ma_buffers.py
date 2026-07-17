"""
Multi-Agent Sequence Replay Buffer.
Author: Ryan
"""

import random
from typing import Dict

import numpy as np
import torch


class MAReplayBuffer:
    def __init__(self, capacity: int, num_agents: int, obs_dim: int, sequence_length: int = 10, device: str = "cpu"):
        self.capacity = capacity
        self.num_agents = num_agents
        self.sequence_length = sequence_length
        self.device = device
        self.obs_dim = obs_dim
        # Joint action = N SU powers + relay power + alpha power-split factor.
        self.action_dim = num_agents + 2
        
        # We will store episodes just like SequenceReplayBuffer
        self.episodes = []
        self.current_episode = []
        self.total_size = 0

    def add(
        self,
        obs: np.ndarray,      # (N, 8)
        action: np.ndarray,   # (N+2,)
        reward: float,
        next_obs: np.ndarray, # (N, 8)
        done: bool,
        info: Dict
    ):
        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "primary_throughput": info.get("primary_throughput", 0.0),
            "average_power": info.get("average_power", 0.0),
            "relay_decoded": info.get("dec_history")[:, -1, 0], # (N,)
            "outage": info.get("outage", 0.0)
        }
        self.current_episode.append(transition)
        self.total_size += 1
        
        if done:
            self.episodes.append(self.current_episode)
            self.current_episode = []
            
            # Evict old episodes if capacity exceeded
            while self.total_size > self.capacity and len(self.episodes) > 1:
                evicted = self.episodes.pop(0)
                self.total_size -= len(evicted)

    def sample_sequences(self, batch_size: int):
        hist_seqs = []
        next_hist_seqs = []
        actions = []
        rewards = []
        dones = []
        r_thr = []
        r_qos = []
        r_nrg = []
        
        valid_episodes = [ep for ep in self.episodes if len(ep) > 0]
        if not valid_episodes:
            valid_episodes = [self.current_episode] if len(self.current_episode) > 0 else []
            
        for _ in range(batch_size):
            ep = random.choice(valid_episodes)
            t = random.randint(0, len(ep) - 1)
            
            # Build window [t - seq_len + 1 : t]
            start_t = t - self.sequence_length + 1
            
            h_seq = []
            next_h_seq = []
            
            for step_idx in range(start_t, t + 1):
                if step_idx < 0:
                    # Pad with zeros
                    h_step = np.zeros((self.num_agents, 11), dtype=np.float32)
                    next_h_step = np.zeros((self.num_agents, 11), dtype=np.float32)
                else:
                    trans = ep[step_idx]
                    o = trans["obs"] # (N, 8)
                    a = trans["action"][:self.num_agents] # (N,)
                    d = trans["relay_decoded"] # (N,)
                    out = trans["outage"] # scalar
                    
                    # h_step = (N, 11)
                    h_step = np.zeros((self.num_agents, 11), dtype=np.float32)
                    h_step[:, 0:8] = o
                    h_step[:, 8] = a
                    h_step[:, 9] = d
                    h_step[:, 10] = out
                    
                    # Next step
                    if step_idx + 1 < len(ep):
                        ntrans = ep[step_idx + 1]
                        no = ntrans["obs"]
                        na = ntrans["action"][:self.num_agents]
                        nd = ntrans["relay_decoded"]
                        nout = ntrans["outage"]
                    else:
                        no = trans["next_obs"]
                        na = np.zeros(self.num_agents)
                        nd = np.zeros(self.num_agents)
                        nout = out
                        
                    next_h_step = np.zeros((self.num_agents, 11), dtype=np.float32)
                    next_h_step[:, 0:8] = no
                    next_h_step[:, 8] = na
                    next_h_step[:, 9] = nd
                    next_h_step[:, 10] = nout
                    
                h_seq.append(h_step)
                next_h_seq.append(next_h_step)
                
            hist_seqs.append(np.stack(h_seq, axis=1)) # (N, L, 11)
            next_hist_seqs.append(np.stack(next_h_seq, axis=1))
            
            trans_t = ep[t]
            actions.append(trans_t["action"]) # (N+1,)
            rewards.append([trans_t["reward"]])
            dones.append([1.0 if trans_t["done"] else 0.0])
            r_thr.append([trans_t["reward"]])
            r_qos.append([trans_t["primary_throughput"]])
            r_nrg.append([trans_t["average_power"]])
            
        hist_seqs = torch.FloatTensor(np.array(hist_seqs)).to(self.device) # (B, N, L, 11)
        next_hist_seqs = torch.FloatTensor(np.array(next_hist_seqs)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device) # (B, N+1)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device) # (B, 1)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        r_thr = torch.FloatTensor(np.array(r_thr)).to(self.device)
        r_qos = torch.FloatTensor(np.array(r_qos)).to(self.device)
        r_nrg = torch.FloatTensor(np.array(r_nrg)).to(self.device)
        
        return hist_seqs, next_hist_seqs, actions, rewards, dones, r_thr, r_qos, r_nrg
