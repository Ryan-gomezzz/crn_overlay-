"""
Sequence Replay Buffer for CAMO-TD3 and TD3.
"""

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


class SequenceReplayBuffer:
    """
    Episode-aware replay buffer supporting sequential sampling and standard 
    flat sampling.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        sequence_length: int = 10,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.device = device

        self.episodes: List[List[Dict[str, Any]]] = []
        self.current_episode: List[Dict[str, Any]] = []
        self.total_size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Dict[str, Any],
    ):
        """
        Store a single transition step.
        """
        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "throughput_reward": info.get("throughput_reward", reward),
            "interference_reward": info.get("interference_reward", 0.0),
            "energy_reward": info.get("energy_reward", 0.0),
            "primary_throughput": info.get("primary_throughput", 0.0),
            "average_power": info.get("average_power", 0.0),
            "relay_decoded": info.get("relay_decoded", 0.0),
            "outage": info.get("outage", 0.0),
        }

        self.current_episode.append(transition)
        self.total_size += 1

        if done:
            self.episodes.append(self.current_episode)
            self.current_episode = []

        # Evict old episodes if capacity is exceeded
        while self.total_size > self.capacity and len(self.episodes) > 0:
            removed_ep = self.episodes.pop(0)
            self.total_size -= len(removed_ep)

    def sample_standard(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standard random sampling of flat transitions (backward compatible with TD3).
        """
        # Flatten all complete episodes + current partial episode
        all_transitions = []
        for ep in self.episodes:
            all_transitions.extend(ep)
        all_transitions.extend(self.current_episode)

        if len(all_transitions) == 0:
            raise ValueError("Buffer is empty.")

        sampled = random.choices(all_transitions, k=batch_size)

        obs = np.array([t["obs"] for t in sampled], dtype=np.float32)
        actions = np.array([t["action"] for t in sampled], dtype=np.float32)
        rewards = np.array([t["reward"] for t in sampled], dtype=np.float32).reshape(
            -1, 1
        )
        next_obs = np.array([t["next_obs"] for t in sampled], dtype=np.float32)
        dones = np.array([t["done"] for t in sampled], dtype=np.float32).reshape(-1, 1)

        return (
            torch.as_tensor(obs, device=self.device),
            torch.as_tensor(actions, device=self.device),
            torch.as_tensor(rewards, device=self.device),
            torch.as_tensor(next_obs, device=self.device),
            torch.as_tensor(dones, device=self.device),
        )

    def sample_sequences(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Sample continuous sequences of length L (for CAMO-TD3).
        Returns padded historical inputs:
            obs_seqs:       (batch_size, L, obs_dim)
            act_seqs:       (batch_size, L, act_dim)
            rewards:        (batch_size, 1)
            next_obs_seqs:  (batch_size, L, obs_dim)
            next_act_seqs:  (batch_size, L, act_dim)
            dones:          (batch_size, 1)
            throughput_rew: (batch_size, 1)
            interference_rew:(batch_size, 1)
            energy_rew:     (batch_size, 1)
        """
        all_episodes = self.episodes.copy()
        if len(self.current_episode) > 0:
            all_episodes.append(self.current_episode)

        if len(all_episodes) == 0:
            raise ValueError("Buffer has no episodes.")

        obs_seq_batch = []
        act_seq_batch = []
        reward_batch = []
        next_obs_seq_batch = []
        next_act_seq_batch = []
        done_batch = []
        thr_reward_batch = []
        inf_reward_batch = []
        nrg_reward_batch = []

        L = self.sequence_length

        for _ in range(batch_size):
            # Sample a random episode
            ep = random.choice(all_episodes)
            ep_len = len(ep)

            # Sample a step index t in this episode
            t = random.randint(0, ep_len - 1)

            # Build current sequence: t - L + 1 to t
            obs_seq = []
            act_seq = []
            for i in range(t - L + 1, t + 1):
                if i < 0:
                    obs_seq.append(np.zeros(self.obs_dim, dtype=np.float32))
                    act_seq.append(np.zeros(self.action_dim, dtype=np.float32))
                else:
                    obs_seq.append(ep[i]["obs"])
                    # Previous action
                    if i - 1 < 0:
                        act_seq.append(np.zeros(self.action_dim, dtype=np.float32))
                    else:
                        act_seq.append(ep[i - 1]["action"])

            # Build next sequence: t - L + 2 to t + 1
            next_obs_seq = []
            next_act_seq = []
            for i in range(t - L + 2, t + 2):
                if i < 0:
                    next_obs_seq.append(np.zeros(self.obs_dim, dtype=np.float32))
                    next_act_seq.append(np.zeros(self.action_dim, dtype=np.float32))
                elif i <= t:
                    next_obs_seq.append(ep[i]["obs"])
                    if i - 1 < 0:
                        next_act_seq.append(np.zeros(self.action_dim, dtype=np.float32))
                    else:
                        next_act_seq.append(ep[i - 1]["action"])
                else:  # i == t + 1
                    next_obs_seq.append(ep[t]["next_obs"])
                    next_act_seq.append(ep[t]["action"])

            obs_seq_batch.append(obs_seq)
            act_seq_batch.append(act_seq)
            next_obs_seq_batch.append(next_obs_seq)
            next_act_seq_batch.append(next_act_seq)

            # Metrics at step t
            reward_batch.append(ep[t]["reward"])
            done_batch.append(ep[t]["done"])
            thr_reward_batch.append(ep[t]["throughput_reward"])
            inf_reward_batch.append(ep[t]["interference_reward"])
            nrg_reward_batch.append(ep[t]["energy_reward"])

        # Convert to arrays and tensors
        obs_seqs = torch.as_tensor(
            np.array(obs_seq_batch, dtype=np.float32), device=self.device
        )
        act_seqs = torch.as_tensor(
            np.array(act_seq_batch, dtype=np.float32), device=self.device
        )
        rewards = torch.as_tensor(
            np.array(reward_batch, dtype=np.float32).reshape(-1, 1), device=self.device
        )
        next_obs_seqs = torch.as_tensor(
            np.array(next_obs_seq_batch, dtype=np.float32), device=self.device
        )
        next_act_seqs = torch.as_tensor(
            np.array(next_act_seq_batch, dtype=np.float32), device=self.device
        )
        dones = torch.as_tensor(
            np.array(done_batch, dtype=np.float32).reshape(-1, 1), device=self.device
        )
        thr_rewards = torch.as_tensor(
            np.array(thr_reward_batch, dtype=np.float32).reshape(-1, 1),
            device=self.device,
        )
        inf_rewards = torch.as_tensor(
            np.array(inf_reward_batch, dtype=np.float32).reshape(-1, 1),
            device=self.device,
        )
        nrg_rewards = torch.as_tensor(
            np.array(nrg_reward_batch, dtype=np.float32).reshape(-1, 1),
            device=self.device,
        )

        return (
            obs_seqs,
            act_seqs,
            rewards,
            next_obs_seqs,
            next_act_seqs,
            dones,
            thr_rewards,
            inf_rewards,
            nrg_rewards,
        )

    def sample_sequences_overlay(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Sample continuous sequences for OVERLAY_CAMO_TD3.
        Returns concatenated historical inputs:
            history_seqs:      (batch_size, L, 8) -> [obs, act, dec, out]
            next_history_seqs: (batch_size, L, 8) -> [next_obs, next_act, next_dec, 
            next_out]
            actions:           (batch_size, act_dim) -> action at step t
            rewards:           (batch_size, 1)
            dones:             (batch_size, 1)
            thr_rewards:       (batch_size, 1) -> secondary throughput
            pu_qos_rewards:    (batch_size, 1) -> primary throughput (QoS constraint)
            energy_rewards:    (batch_size, 1) -> average power (Energy constraint)
        """
        all_episodes = self.episodes.copy()
        if len(self.current_episode) > 0:
            all_episodes.append(self.current_episode)

        if len(all_episodes) == 0:
            raise ValueError("Buffer has no episodes.")

        hist_seq_batch = []
        next_hist_seq_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        thr_reward_batch = []
        pu_qos_reward_batch = []
        nrg_reward_batch = []

        L = self.sequence_length

        for _ in range(batch_size):
            ep = random.choice(all_episodes)
            ep_len = len(ep)
            t = random.randint(0, ep_len - 1)

            # Build current 8D sequence: t - L + 1 to t
            hist_seq = []
            for i in range(t - L + 1, t + 1):
                if i < 0:
                    step_vec = np.zeros(
                        self.obs_dim + self.action_dim + 2, dtype=np.float32
                    )
                else:
                    obs = ep[i]["obs"]
                    prev_act = (
                        ep[i - 1]["action"]
                        if i - 1 >= 0
                        else np.zeros(self.action_dim, dtype=np.float32)
                    )
                    dec = ep[i]["relay_decoded"]
                    out = ep[i]["outage"]
                    step_vec = np.concatenate([obs, prev_act, [dec], [out]], axis=0)
                hist_seq.append(step_vec)

            # Build next 8D sequence: t - L + 2 to t + 1
            next_hist_seq = []
            for i in range(t - L + 2, t + 2):
                if i < 0:
                    step_vec = np.zeros(
                        self.obs_dim + self.action_dim + 2, dtype=np.float32
                    )
                elif i <= t:
                    obs = ep[i]["obs"]
                    prev_act = (
                        ep[i - 1]["action"]
                        if i - 1 >= 0
                        else np.zeros(self.action_dim, dtype=np.float32)
                    )
                    dec = ep[i]["relay_decoded"]
                    out = ep[i]["outage"]
                    step_vec = np.concatenate([obs, prev_act, [dec], [out]], axis=0)
                else:  # i == t + 1
                    obs = ep[t]["next_obs"]
                    prev_act = ep[t]["action"]
                    dec = 0.0
                    out = 0.0
                    step_vec = np.concatenate([obs, prev_act, [dec], [out]], axis=0)
                next_hist_seq.append(step_vec)

            hist_seq_batch.append(hist_seq)
            next_hist_seq_batch.append(next_hist_seq)

            action_batch.append(ep[t]["action"])
            reward_batch.append(ep[t]["reward"])
            done_batch.append(ep[t]["done"])
            thr_reward_batch.append(ep[t]["throughput_reward"])
            pu_qos_reward_batch.append(ep[t]["primary_throughput"])
            nrg_reward_batch.append(ep[t]["average_power"])

        # Convert to tensors
        hist_seqs = torch.as_tensor(
            np.array(hist_seq_batch, dtype=np.float32), device=self.device
        )
        next_hist_seqs = torch.as_tensor(
            np.array(next_hist_seq_batch, dtype=np.float32), device=self.device
        )
        actions = torch.as_tensor(
            np.array(action_batch, dtype=np.float32), device=self.device
        )
        rewards = torch.as_tensor(
            np.array(reward_batch, dtype=np.float32).reshape(-1, 1), device=self.device
        )
        dones = torch.as_tensor(
            np.array(done_batch, dtype=np.float32).reshape(-1, 1), device=self.device
        )
        thr_rewards = torch.as_tensor(
            np.array(thr_reward_batch, dtype=np.float32).reshape(-1, 1),
            device=self.device,
        )
        pu_qos_rewards = torch.as_tensor(
            np.array(pu_qos_reward_batch, dtype=np.float32).reshape(-1, 1),
            device=self.device,
        )
        nrg_rewards = torch.as_tensor(
            np.array(nrg_reward_batch, dtype=np.float32).reshape(-1, 1),
            device=self.device,
        )

        return (
            hist_seqs,
            next_hist_seqs,
            actions,
            rewards,
            dones,
            thr_rewards,
            pu_qos_rewards,
            nrg_rewards,
        )

    def __len__(self) -> int:
        return self.total_size
