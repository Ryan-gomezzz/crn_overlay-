"""
Multi-Agent Replay Buffer for MATD3.

Stores multi-agent transitions with global state, per-agent observations
and actions, shared rewards decomposed by objective, and done flags.
Supports efficient batch sampling for centralized critic training.
"""

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


class MultiAgentReplayBuffer:
    """Episode-aware replay buffer for multi-agent transitions.

    Stores transitions of the form:
        (global_state, per_agent_obs, per_agent_actions, relay_action,
         reward_thr, reward_qos, reward_nrg, next_global_state,
         next_per_agent_obs, done)

    All rewards are scalar (shared across agents).

    Args:
        capacity: Maximum number of transitions to store.
        num_agents: Number of SU agents.
        obs_dim_per_agent: Local observation dimensionality per agent.
        device: PyTorch device string.
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim_per_agent: int,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim_per_agent = obs_dim_per_agent
        self.device = device

        # Global state dim = obs_dim_per_agent * num_agents
        self.global_state_dim = obs_dim_per_agent * num_agents

        # Flat storage lists (ring buffer semantics via list + eviction)
        self.episodes: List[List[Dict[str, Any]]] = []
        self.current_episode: List[Dict[str, Any]] = []
        self.total_size = 0

    def add(
        self,
        global_state: np.ndarray,
        per_agent_obs: Dict[int, np.ndarray],
        per_agent_actions: List[float],
        relay_action: float,
        reward_throughput: float,
        reward_qos: float,
        reward_energy: float,
        reward_total: float,
        next_global_state: np.ndarray,
        next_per_agent_obs: Dict[int, np.ndarray],
        done: bool,
        info: Dict[str, Any],
    ):
        """Store a single multi-agent transition.

        Args:
            global_state: Concatenated observation of shape (obs_dim * N,).
            per_agent_obs: Dict mapping agent_id → local obs array (obs_dim,).
            per_agent_actions: List of N scalar power actions (one per agent).
            relay_action: Scalar relay power action.
            reward_throughput: Shared SU sum-rate reward.
            reward_qos: PU throughput (for QoS constraint critic).
            reward_energy: Average power (for energy constraint critic).
            reward_total: Overall scalar reward (env reward signal).
            next_global_state: Next global state.
            next_per_agent_obs: Next per-agent observations.
            done: Episode termination flag.
            info: Auxiliary info from the environment.
        """
        transition = {
            "global_state": global_state.copy(),
            "per_agent_obs": {
                i: per_agent_obs[i].copy() for i in range(self.num_agents)
            },
            "per_agent_actions": list(per_agent_actions),
            "relay_action": relay_action,
            "reward_throughput": reward_throughput,
            "reward_qos": reward_qos,
            "reward_energy": reward_energy,
            "reward_total": reward_total,
            "next_global_state": next_global_state.copy(),
            "next_per_agent_obs": {
                i: next_per_agent_obs[i].copy() for i in range(self.num_agents)
            },
            "done": done,
            # Preserve per-agent histories for GRU encoding
            "agent_histories": info.get("agent_histories", {}),
            "relay_decoded": info.get("relay_decoded", 0.0),
            "outage": info.get("outage", 0.0),
        }

        self.current_episode.append(transition)
        self.total_size += 1

        if done:
            self.episodes.append(self.current_episode)
            self.current_episode = []

        # Evict oldest episodes when capacity exceeded
        while self.total_size > self.capacity and len(self.episodes) > 0:
            removed_ep = self.episodes.pop(0)
            self.total_size -= len(removed_ep)

    def sample(
        self, batch_size: int
    ) -> Dict[str, Any]:
        """Sample a random batch of multi-agent transitions.

        Returns:
            Dictionary with the following tensor keys:
            - global_states:      (batch, global_state_dim)
            - next_global_states: (batch, global_state_dim)
            - all_actions:        (batch, num_agents + 1)  [agent powers + relay]
            - reward_thr:         (batch, 1)
            - reward_qos:         (batch, 1)
            - reward_nrg:         (batch, 1)
            - reward_total:       (batch, 1)
            - dones:              (batch, 1)
            - per_agent_obs:      list of N tensors, each (batch, obs_dim)
            - next_per_agent_obs: list of N tensors, each (batch, obs_dim)
        """
        # Flatten all complete episodes + current partial episode
        all_transitions: List[Dict[str, Any]] = []
        for ep in self.episodes:
            all_transitions.extend(ep)
        all_transitions.extend(self.current_episode)

        if len(all_transitions) == 0:
            raise ValueError("Multi-agent replay buffer is empty.")

        sampled = random.choices(all_transitions, k=batch_size)

        # Pre-allocate arrays
        global_states = np.zeros((batch_size, self.global_state_dim), dtype=np.float32)
        next_global_states = np.zeros((batch_size, self.global_state_dim), dtype=np.float32)
        all_actions = np.zeros((batch_size, self.num_agents + 1), dtype=np.float32)
        reward_thr = np.zeros((batch_size, 1), dtype=np.float32)
        reward_qos = np.zeros((batch_size, 1), dtype=np.float32)
        reward_nrg = np.zeros((batch_size, 1), dtype=np.float32)
        reward_total = np.zeros((batch_size, 1), dtype=np.float32)
        dones = np.zeros((batch_size, 1), dtype=np.float32)

        per_agent_obs = [
            np.zeros((batch_size, self.obs_dim_per_agent), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        next_per_agent_obs = [
            np.zeros((batch_size, self.obs_dim_per_agent), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        for idx, t in enumerate(sampled):
            global_states[idx] = t["global_state"]
            next_global_states[idx] = t["next_global_state"]

            # Actions: [agent_0, agent_1, ..., agent_N-1, relay]
            for i in range(self.num_agents):
                all_actions[idx, i] = t["per_agent_actions"][i]
            all_actions[idx, self.num_agents] = t["relay_action"]

            reward_thr[idx, 0] = t["reward_throughput"]
            reward_qos[idx, 0] = t["reward_qos"]
            reward_nrg[idx, 0] = t["reward_energy"]
            reward_total[idx, 0] = t["reward_total"]
            dones[idx, 0] = float(t["done"])

            for i in range(self.num_agents):
                per_agent_obs[i][idx] = t["per_agent_obs"][i]
                next_per_agent_obs[i][idx] = t["next_per_agent_obs"][i]

        # Convert to tensors
        return {
            "global_states": torch.as_tensor(global_states, device=self.device),
            "next_global_states": torch.as_tensor(next_global_states, device=self.device),
            "all_actions": torch.as_tensor(all_actions, device=self.device),
            "reward_thr": torch.as_tensor(reward_thr, device=self.device),
            "reward_qos": torch.as_tensor(reward_qos, device=self.device),
            "reward_nrg": torch.as_tensor(reward_nrg, device=self.device),
            "reward_total": torch.as_tensor(reward_total, device=self.device),
            "dones": torch.as_tensor(dones, device=self.device),
            "per_agent_obs": [
                torch.as_tensor(per_agent_obs[i], device=self.device)
                for i in range(self.num_agents)
            ],
            "next_per_agent_obs": [
                torch.as_tensor(next_per_agent_obs[i], device=self.device)
                for i in range(self.num_agents)
            ],
        }

    def sample_with_histories(
        self, batch_size: int, history_length: int = 10,
    ) -> Dict[str, Any]:
        """Sample transitions with per-agent GRU history sequences.

        Extends ``sample()`` output with:
            - agent_hist_seqs: list of N tensors (batch, history_length, input_dim)
              where input_dim = obs_dim + 1 (action) + 1 (decoded) + 1 (outage)

        This method samples individual steps *within episodes* so that
        the preceding ``history_length`` transitions are available for
        building GRU input sequences.
        """
        all_episodes = self.episodes.copy()
        if len(self.current_episode) > 0:
            all_episodes.append(self.current_episode)

        if len(all_episodes) == 0:
            raise ValueError("Buffer has no episodes for history sampling.")

        L = history_length
        obs_dim = self.obs_dim_per_agent
        # GRU input: obs + action + decoded + outage = obs_dim + 3
        hist_input_dim = obs_dim + 3

        # Pre-allocate
        global_states = np.zeros((batch_size, self.global_state_dim), dtype=np.float32)
        next_global_states = np.zeros((batch_size, self.global_state_dim), dtype=np.float32)
        all_actions = np.zeros((batch_size, self.num_agents + 1), dtype=np.float32)
        reward_thr = np.zeros((batch_size, 1), dtype=np.float32)
        reward_qos = np.zeros((batch_size, 1), dtype=np.float32)
        reward_nrg = np.zeros((batch_size, 1), dtype=np.float32)
        reward_total = np.zeros((batch_size, 1), dtype=np.float32)
        dones = np.zeros((batch_size, 1), dtype=np.float32)

        per_agent_obs = [
            np.zeros((batch_size, obs_dim), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        next_per_agent_obs = [
            np.zeros((batch_size, obs_dim), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        # Per-agent history sequences: (batch, L, hist_input_dim)
        agent_hist_seqs = [
            np.zeros((batch_size, L, hist_input_dim), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        next_agent_hist_seqs = [
            np.zeros((batch_size, L, hist_input_dim), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        for b in range(batch_size):
            ep = random.choice(all_episodes)
            ep_len = len(ep)
            t = random.randint(0, ep_len - 1)
            trans = ep[t]

            global_states[b] = trans["global_state"]
            next_global_states[b] = trans["next_global_state"]

            for i in range(self.num_agents):
                all_actions[b, i] = trans["per_agent_actions"][i]
                per_agent_obs[i][b] = trans["per_agent_obs"][i]
                next_per_agent_obs[i][b] = trans["next_per_agent_obs"][i]
            all_actions[b, self.num_agents] = trans["relay_action"]

            reward_thr[b, 0] = trans["reward_throughput"]
            reward_qos[b, 0] = trans["reward_qos"]
            reward_nrg[b, 0] = trans["reward_energy"]
            reward_total[b, 0] = trans["reward_total"]
            dones[b, 0] = float(trans["done"])

            # Build history sequences for each agent
            for agent_id in range(self.num_agents):
                # Current history: steps [t-L+1, ..., t]
                for seq_idx, step_idx in enumerate(range(t - L + 1, t + 1)):
                    if step_idx < 0:
                        # Zero-padded
                        agent_hist_seqs[agent_id][b, seq_idx] = 0.0
                    else:
                        step_t = ep[step_idx]
                        obs_i = step_t["per_agent_obs"][agent_id]
                        act_i = step_t["per_agent_actions"][agent_id]
                        dec_i = float(step_t.get("relay_decoded", 0.0))
                        out_i = float(step_t.get("outage", 0.0))
                        agent_hist_seqs[agent_id][b, seq_idx] = np.concatenate(
                            [obs_i, [act_i], [dec_i], [out_i]]
                        )

                # Next history: steps [t-L+2, ..., t+1]
                for seq_idx, step_idx in enumerate(range(t - L + 2, t + 2)):
                    if step_idx < 0:
                        next_agent_hist_seqs[agent_id][b, seq_idx] = 0.0
                    elif step_idx <= t:
                        step_t = ep[step_idx]
                        obs_i = step_t["per_agent_obs"][agent_id]
                        act_i = step_t["per_agent_actions"][agent_id]
                        dec_i = float(step_t.get("relay_decoded", 0.0))
                        out_i = float(step_t.get("outage", 0.0))
                        next_agent_hist_seqs[agent_id][b, seq_idx] = np.concatenate(
                            [obs_i, [act_i], [dec_i], [out_i]]
                        )
                    else:
                        # step_idx == t + 1 → use next_obs from transition t
                        next_obs_i = trans["next_per_agent_obs"][agent_id]
                        act_i = trans["per_agent_actions"][agent_id]
                        dec_i = float(trans.get("relay_decoded", 0.0))
                        out_i = float(trans.get("outage", 0.0))
                        next_agent_hist_seqs[agent_id][b, seq_idx] = np.concatenate(
                            [next_obs_i, [act_i], [dec_i], [out_i]]
                        )

        # Convert to tensors
        result = {
            "global_states": torch.as_tensor(global_states, device=self.device),
            "next_global_states": torch.as_tensor(next_global_states, device=self.device),
            "all_actions": torch.as_tensor(all_actions, device=self.device),
            "reward_thr": torch.as_tensor(reward_thr, device=self.device),
            "reward_qos": torch.as_tensor(reward_qos, device=self.device),
            "reward_nrg": torch.as_tensor(reward_nrg, device=self.device),
            "reward_total": torch.as_tensor(reward_total, device=self.device),
            "dones": torch.as_tensor(dones, device=self.device),
            "per_agent_obs": [
                torch.as_tensor(per_agent_obs[i], device=self.device)
                for i in range(self.num_agents)
            ],
            "next_per_agent_obs": [
                torch.as_tensor(next_per_agent_obs[i], device=self.device)
                for i in range(self.num_agents)
            ],
            "agent_hist_seqs": [
                torch.as_tensor(agent_hist_seqs[i], device=self.device)
                for i in range(self.num_agents)
            ],
            "next_agent_hist_seqs": [
                torch.as_tensor(next_agent_hist_seqs[i], device=self.device)
                for i in range(self.num_agents)
            ],
        }
        return result

    def __len__(self) -> int:
        return self.total_size
