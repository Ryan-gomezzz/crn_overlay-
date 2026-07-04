"""
Underlay TD3 Agent Implementation (CAMO-TD3).
Faithfully implements the original CAMO-TD3 algorithm: Recurrent GRU Belief Encoder,
six critics (throughput, interference, and energy), adaptive Lagrangian optimization,
and directional exploration based on constraint gradients.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.models import GRUBeliefEncoder, CAMO_Actor, TwinCritics
from agents.buffers import SequenceReplayBuffer
from simulator.utils import dbm_to_watt


class UnderlayTD3Agent:
    """
    Underlay TD3 Agent (faithful adaptation of the CAMO-TD3 algorithm).
    Incorporates belief representation, adaptive Lagrangians, and directional exploration.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device

        self.algo_cfg = config.get("training", {})
        self.camo_cfg = config.get("camo_td3", {})
        self.algorithm_name = "UNDERLAY_TD3"

        self.gamma = self.algo_cfg.get("gamma", 0.99)
        self.tau = self.algo_cfg.get("tau", 0.005)
        self.policy_delay = self.algo_cfg.get("policy_delay", 2)

        self.expl_noise = self.algo_cfg.get("exploration_noise", 0.1)
        self.policy_noise = self.algo_cfg.get("policy_noise", 0.2)
        self.noise_clip = self.algo_cfg.get("noise_clip", 0.5)

        self.batch_size = self.algo_cfg.get("batch_size", 256)
        self.lr_actor = self.algo_cfg.get("lr_actor", 3e-4)
        self.lr_critic = self.algo_cfg.get("lr_critic", 3e-4)
        self.lr_lambda = self.camo_cfg.get("lr_lambda", 1e-3)

        self.seq_len = self.camo_cfg.get("history_length", 10)
        self.obs_dim = 7
        self.action_dim = 2
        self.total_it = 0

        # Physical limits for constraints
        self.interference_limit = dbm_to_watt(self.camo_cfg.get("interference_limit_dbm", -50.0))
        self.energy_limit = self.camo_cfg.get("energy_limit_watts", 0.1)

        # Episodic Sequence Replay Buffer
        self.replay_buffer = SequenceReplayBuffer(
            capacity=self.algo_cfg.get("buffer_size", 100000),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            sequence_length=self.seq_len,
            device=self.device
        )

        # Recurrent GRU Belief Encoders (input_dim = obs_dim + action_dim)
        self.encoder = GRUBeliefEncoder(self.obs_dim, self.action_dim, hidden_dim=64).to(self.device)
        self.encoder_target = GRUBeliefEncoder(self.obs_dim, self.action_dim, hidden_dim=64).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr_actor)

        # Actor Network
        self.actor = CAMO_Actor(belief_dim=64, action_dim=self.action_dim).to(self.device)
        self.actor_target = CAMO_Actor(belief_dim=64, action_dim=self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # Critics: 3 Twin Critic Pairs (Throughput, Interference, Energy)
        self.critic_thr = TwinCritics(state_dim=64, action_dim=self.action_dim).to(self.device)
        self.critic_thr_target = TwinCritics(state_dim=64, action_dim=self.action_dim).to(self.device)
        self.critic_thr_target.load_state_dict(self.critic_thr.state_dict())
        self.critic_thr_optimizer = optim.Adam(self.critic_thr.parameters(), lr=self.lr_critic)

        self.critic_inf = TwinCritics(state_dim=64, action_dim=self.action_dim).to(self.device)
        self.critic_inf_target = TwinCritics(state_dim=64, action_dim=self.action_dim).to(self.device)
        self.critic_inf_target.load_state_dict(self.critic_inf.state_dict())
        self.critic_inf_optimizer = optim.Adam(self.critic_inf.parameters(), lr=self.lr_critic)

        self.critic_nrg = TwinCritics(state_dim=64, action_dim=self.action_dim).to(self.device)
        self.critic_nrg_target = TwinCritics(state_dim=64, action_dim=self.action_dim).to(self.device)
        self.critic_nrg_target.load_state_dict(self.critic_nrg.state_dict())
        self.critic_nrg_optimizer = optim.Adam(self.critic_nrg.parameters(), lr=self.lr_critic)

        # Learnable Lagrangians in log scale
        init_val_inf = self.camo_cfg.get("lambda_inf_init", 0.1)
        init_val_nrg = self.camo_cfg.get("lambda_nrg_init", 0.1)

        alpha_inf_val = np.log(np.exp(init_val_inf) - 1.0 + 1e-9)
        alpha_nrg_val = np.log(np.exp(init_val_nrg) - 1.0 + 1e-9)

        self.alpha_inf = nn.Parameter(torch.tensor(alpha_inf_val, dtype=torch.float32, device=self.device, requires_grad=True))
        self.alpha_nrg = nn.Parameter(torch.tensor(alpha_nrg_val, dtype=torch.float32, device=self.device, requires_grad=True))

        self.lambda_optimizer = optim.Adam([self.alpha_inf, self.alpha_nrg], lr=self.lr_lambda)
        self.lambda_clamp_max = self.camo_cfg.get("lambda_clamp_max", 50.0)

        # Directional Exploration Parameters
        self.eta_explore_init = self.camo_cfg.get("eta_explore_init", 0.05)
        self.eta_explore_decay = self.camo_cfg.get("eta_explore_decay", 0.9999)
        self.eta_explore = self.eta_explore_init

    @property
    def lambda_inf(self) -> float:
        val = torch.functional.F.softplus(self.alpha_inf).item()
        return min(val, self.lambda_clamp_max)

    @property
    def lambda_nrg(self) -> float:
        val = torch.functional.F.softplus(self.alpha_nrg).item()
        return min(val, self.lambda_clamp_max)

    def select_action(self, obs: np.ndarray, info: dict, explore: bool = True) -> np.ndarray:
        """
        Select explored action using GRU belief state and Directional Exploration.
        """
        obs_history = torch.as_tensor(info["obs_history"], dtype=torch.float32, device=self.device).unsqueeze(0)
        act_history = torch.as_tensor(info["act_history"], dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            belief = self.encoder(obs_history, act_history)
            action = self.actor(belief).cpu().data.numpy().flatten()

        if explore:
            # 1. Standard noise
            noise = np.random.normal(0, self.expl_noise, size=self.action_dim)

            # 2. Safety Gradient Directional Exploration (decreases constraint violations)
            belief_tensor = belief.clone().detach().requires_grad_(True)
            action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_(True)

            q_inf = self.critic_inf.Q1(belief_tensor, action_tensor)
            q_nrg = self.critic_nrg.Q1(belief_tensor, action_tensor)

            grad_inf = torch.autograd.grad(q_inf.sum(), action_tensor, retain_graph=True)[0].cpu().data.numpy().flatten()
            grad_nrg = torch.autograd.grad(q_nrg.sum(), action_tensor)[0].cpu().data.numpy().flatten()

            # Moving in negative direction of constraint gradients decreases risk
            safety_bias = -self.eta_explore * (self.lambda_inf * grad_inf + self.lambda_nrg * grad_nrg)
            self.eta_explore = max(0.0, self.eta_explore * self.eta_explore_decay)

            action = action + noise + safety_bias

        return np.clip(action, 0.0, 1.0)

    def train(self, writer: SummaryWriter) -> dict:
        """
        Perform one training update for Underlay TD3.
        """
        self.total_it += 1
        metrics_log = {}

        # Sample sequences
        (
            obs_seq,
            act_seq,
            reward,
            next_obs_seq,
            next_act_seq,
            done,
            r_thr,
            r_inf,
            r_nrg,
        ) = self.replay_buffer.sample_sequences(self.batch_size)

        # Compute current and next belief states
        belief = self.encoder(obs_seq, act_seq)

        with torch.no_grad():
            next_belief = self.encoder_target(next_obs_seq, next_act_seq)
            noise = (torch.randn_like(next_act_seq[:, -1, :]) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_belief) + noise).clamp(0.0, 1.0)

            target_q_thr = r_thr + (1 - done) * self.gamma * torch.min(
                *self.critic_thr_target.evaluate(next_belief, next_action)
            )
            target_q_inf = r_inf + (1 - done) * self.gamma * torch.min(
                *self.critic_inf_target.evaluate(next_belief, next_action)
            )
            target_q_nrg = r_nrg + (1 - done) * self.gamma * torch.min(
                *self.critic_nrg_target.evaluate(next_belief, next_action)
            )

        detached_belief = belief.detach()

        # 1. Update Throughput Critics
        current_q1_thr, current_q2_thr = self.critic_thr.evaluate(detached_belief, act_seq[:, -1, :])
        loss_critic_thr = nn.functional.mse_loss(current_q1_thr, target_q_thr) + nn.functional.mse_loss(
            current_q2_thr, target_q_thr
        )
        self.critic_thr_optimizer.zero_grad()
        loss_critic_thr.backward()
        self.critic_thr_optimizer.step()

        # 2. Update Interference Critics
        current_q1_inf, current_q2_inf = self.critic_inf.evaluate(detached_belief, act_seq[:, -1, :])
        loss_critic_inf = nn.functional.mse_loss(current_q1_inf, target_q_inf) + nn.functional.mse_loss(
            current_q2_inf, target_q_inf
        )
        self.critic_inf_optimizer.zero_grad()
        loss_critic_inf.backward()
        self.critic_inf_optimizer.step()

        # 3. Update Energy Critics
        current_q1_nrg, current_q2_nrg = self.critic_nrg.evaluate(detached_belief, act_seq[:, -1, :])
        loss_critic_nrg = nn.functional.mse_loss(current_q1_nrg, target_q_nrg) + nn.functional.mse_loss(
            current_q2_nrg, target_q_nrg
        )
        self.critic_nrg_optimizer.zero_grad()
        loss_critic_nrg.backward()
        self.critic_nrg_optimizer.step()

        # 4. Update Adaptive Lagrangians (gradient ascent)
        lambda_inf_val = torch.functional.F.softplus(self.alpha_inf)
        lambda_nrg_val = torch.functional.F.softplus(self.alpha_nrg)

        violation_inf = current_q1_inf.detach() - self.interference_limit
        violation_nrg = current_q1_nrg.detach() - self.energy_limit

        loss_lambda = -lambda_inf_val * violation_inf.mean() - lambda_nrg_val * violation_nrg.mean()

        self.lambda_optimizer.zero_grad()
        loss_lambda.backward()
        self.lambda_optimizer.step()

        # 5. Delayed Policy & Encoder Updates
        if self.total_it % self.policy_delay == 0:
            detached_belief = belief.detach()
            actions_pred = self.actor(detached_belief)

            q_thr_pred = self.critic_thr.Q1(detached_belief, actions_pred)
            q_inf_pred = self.critic_inf.Q1(detached_belief, actions_pred)
            q_nrg_pred = self.critic_nrg.Q1(detached_belief, actions_pred)

            penalty_inf = self.lambda_inf * (q_inf_pred - self.interference_limit)
            penalty_nrg = self.lambda_nrg * (q_nrg_pred - self.energy_limit)

            actor_loss = -(q_thr_pred - penalty_inf - penalty_nrg).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Encoder weights
            encoder_belief = self.encoder(obs_seq, act_seq)
            self.encoder_optimizer.zero_grad()
            actor_loss_recurrent = -(self.critic_thr.Q1(encoder_belief, actions_pred.detach())).mean()
            actor_loss_recurrent.backward()
            self.encoder_optimizer.step()

            # Soft target updates
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.encoder.parameters(), self.encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_thr.parameters(), self.critic_thr_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_inf.parameters(), self.critic_inf_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_nrg.parameters(), self.critic_nrg_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            writer.add_scalar("Loss/Actor", actor_loss.item(), self.total_it)

        # Log metrics
        writer.add_scalar("Loss/Critic_Throughput", loss_critic_thr.item(), self.total_it)
        writer.add_scalar("Loss/Critic_Interference", loss_critic_inf.item(), self.total_it)
        writer.add_scalar("Loss/Critic_Energy", loss_critic_nrg.item(), self.total_it)
        writer.add_scalar("Lagrangian/Lambda_Interference", self.lambda_inf, self.total_it)
        writer.add_scalar("Lagrangian/Lambda_Energy", self.lambda_nrg, self.total_it)
        writer.add_scalar("Lagrangian/Violation_Interference_Watts", violation_inf.mean().item(), self.total_it)
        writer.add_scalar("Lagrangian/Violation_Energy_Watts", violation_nrg.mean().item(), self.total_it)

        return metrics_log

    def save(self, filepath: str):
        """
        Save Underlay TD3 agent state.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "algorithm": self.algorithm_name,
            "total_it": self.total_it,
            "config": self.config,
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "encoder_state_dict": self.encoder.state_dict(),
            "encoder_target_state_dict": self.encoder_target.state_dict(),
            "encoder_optimizer_state_dict": self.encoder_optimizer.state_dict(),
            "critic_thr_state_dict": self.critic_thr.state_dict(),
            "critic_thr_target_state_dict": self.critic_thr_target.state_dict(),
            "critic_thr_optimizer_state_dict": self.critic_thr_optimizer.state_dict(),
            "critic_inf_state_dict": self.critic_inf.state_dict(),
            "critic_inf_target_state_dict": self.critic_inf_target.state_dict(),
            "critic_inf_optimizer_state_dict": self.critic_inf_optimizer.state_dict(),
            "critic_nrg_state_dict": self.critic_nrg.state_dict(),
            "critic_nrg_target_state_dict": self.critic_nrg_target.state_dict(),
            "critic_nrg_optimizer_state_dict": self.critic_nrg_optimizer.state_dict(),
            "alpha_inf": self.alpha_inf.item(),
            "alpha_nrg": self.alpha_nrg.item(),
            "lambda_optimizer_state_dict": self.lambda_optimizer.state_dict(),
            "eta_explore": self.eta_explore,
        }
        torch.save(checkpoint, filepath)
        print(f"Underlay TD3 Checkpoint successfully saved to: {filepath}")

    def load(self, filepath: str):
        """
        Load Underlay TD3 agent state.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.total_it = checkpoint["total_it"]
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.encoder_target.load_state_dict(checkpoint["encoder_target_state_dict"])
        self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state_dict"])
        self.critic_thr.load_state_dict(checkpoint["critic_thr_state_dict"])
        self.critic_thr_target.load_state_dict(checkpoint["critic_thr_target_state_dict"])
        self.critic_thr_optimizer.load_state_dict(checkpoint["critic_thr_optimizer_state_dict"])
        self.critic_inf.load_state_dict(checkpoint["critic_inf_state_dict"])
        self.critic_inf_target.load_state_dict(checkpoint["critic_inf_target_state_dict"])
        self.critic_inf_optimizer.load_state_dict(checkpoint["critic_inf_optimizer_state_dict"])
        self.critic_nrg.load_state_dict(checkpoint["critic_nrg_state_dict"])
        self.critic_nrg_target.load_state_dict(checkpoint["critic_nrg_target_state_dict"])
        self.critic_nrg_optimizer.load_state_dict(checkpoint["critic_nrg_optimizer_state_dict"])
        
        # Load alphas
        alpha_inf_val = checkpoint.get("alpha_inf", checkpoint.get("alpha_inf"))
        alpha_nrg_val = checkpoint.get("alpha_nrg", checkpoint.get("alpha_nrg"))
        self.alpha_inf.data = torch.tensor(alpha_inf_val, dtype=torch.float32, device=self.device)
        self.alpha_nrg.data = torch.tensor(alpha_nrg_val, dtype=torch.float32, device=self.device)
        
        self.lambda_optimizer.load_state_dict(checkpoint["lambda_optimizer_state_dict"])
        self.eta_explore = checkpoint["eta_explore"]
        print(f"Underlay TD3 Checkpoint successfully loaded from: {filepath}")
