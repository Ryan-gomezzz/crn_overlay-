"""
Baseline TD3 Agent Implementation.
Contains only standard TD3 components: Actor, Twin Critics, Replay Buffer (flat),
delayed updates, target smoothing, soft updates, and exploration noise.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.models import TD3_Actor, TwinCritics
from agents.buffers import SequenceReplayBuffer


class TD3Agent:
    """
    Standard Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent.
    Strictly follows baseline TD3 architecture.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device

        self.algo_cfg = config.get("training", {})
        self.algorithm_name = "TD3"

        self.gamma = self.algo_cfg.get("gamma", 0.99)
        self.tau = self.algo_cfg.get("tau", 0.005)
        self.policy_delay = self.algo_cfg.get("policy_delay", 2)

        self.expl_noise = self.algo_cfg.get("exploration_noise", 0.1)
        self.policy_noise = self.algo_cfg.get("policy_noise", 0.2)
        self.noise_clip = self.algo_cfg.get("noise_clip", 0.5)

        self.batch_size = self.algo_cfg.get("batch_size", 256)
        self.lr_actor = self.algo_cfg.get("lr_actor", 3e-4)
        self.lr_critic = self.algo_cfg.get("lr_critic", 3e-4)

        self.obs_dim = 4
        self.action_dim = 2
        self.total_it = 0

        # Shared Replay Buffer (used for standard flat transitions)
        self.replay_buffer = SequenceReplayBuffer(
            capacity=self.algo_cfg.get("buffer_size", 100000),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            sequence_length=1,  # Sequences are not used by baseline TD3
            device=self.device
        )

        # Networks
        self.actor = TD3_Actor(obs_dim=self.obs_dim, action_dim=self.action_dim).to(self.device)
        self.actor_target = TD3_Actor(obs_dim=self.obs_dim, action_dim=self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        self.critic = TwinCritics(state_dim=self.obs_dim, action_dim=self.action_dim).to(self.device)
        self.critic_target = TwinCritics(state_dim=self.obs_dim, action_dim=self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def select_action(self, obs: np.ndarray, info: dict = None, explore: bool = True) -> np.ndarray:
        """
        Select action given standard observation.
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().data.numpy().flatten()
        if explore:
            noise = np.random.normal(0, self.expl_noise, size=self.action_dim)
            action = action + noise
        return np.clip(action, 0.0, 1.0)

    def train(self, writer: SummaryWriter) -> dict:
        """
        Perform one step of TD3 optimization.
        """
        self.total_it += 1
        metrics_log = {}

        # Standard flat sampling
        obs, action, reward, next_obs, done = self.replay_buffer.sample_standard(self.batch_size)

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_obs) + noise).clamp(0.0, 1.0)

            # Compute target Q value
            target_q = reward + (1 - done) * self.gamma * torch.min(
                *self.critic_target.evaluate(next_obs, next_action)
            )

        # Get current Q estimates
        current_q1, current_q2 = self.critic.evaluate(obs, action)
        loss_critic = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(
            current_q2, target_q
        )

        # Optimize Critics
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft target updates
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            writer.add_scalar("Loss/Actor", actor_loss.item(), self.total_it)

        writer.add_scalar("Loss/Critic", loss_critic.item(), self.total_it)

        return metrics_log

    def save(self, filepath: str):
        """
        Save standard TD3 state.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "algorithm": self.algorithm_name,
            "total_it": self.total_it,
            "config": self.config,
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"TD3 Checkpoint successfully saved to: {filepath}")

    def load(self, filepath: str):
        """
        Load TD3 state.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.total_it = checkpoint["total_it"]
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        print(f"TD3 Checkpoint successfully loaded from: {filepath}")
