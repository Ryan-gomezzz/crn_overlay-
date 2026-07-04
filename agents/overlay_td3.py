"""
Overlay TD3 Agent Implementation.
Implements Overlay-specific improvements: 8-dimensional historical state belief
representation, six critics (throughput, primary QoS rate, and energy), adaptive
Lagrangian optimization for QoS and energy constraints, and safety gradient exploration.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.models import GRUBeliefEncoder, CAMO_Actor, TwinCritics
from agents.buffers import SequenceReplayBuffer


class OverlayTD3Agent:
    """
    Overlay TD3 Agent (custom constraints formulation for Secondary User transmission).
    Extends the GRU state model with primary user rate feedback and outage details.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device

        self.algo_cfg = config.get("training", {})
        self.camo_cfg = config.get("camo_td3", {})
        self.algorithm_name = "OVERLAY_TD3"

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

        # Physical limits and QoS settings
        self.pu_rate_threshold = self.camo_cfg.get("pu_rate_threshold", 1.5)
        self.energy_limit = self.camo_cfg.get("energy_limit_watts", 0.1)

        # Episodic Sequence Replay Buffer
        self.replay_buffer = SequenceReplayBuffer(
            capacity=self.algo_cfg.get("buffer_size", 100000),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            sequence_length=self.seq_len,
            device=self.device
        )

        # Recurrent GRU Belief Encoders (input_dim = 8 for obs, act, dec, out)
        self.encoder = GRUBeliefEncoder(self.obs_dim, self.action_dim, hidden_dim=64, input_dim=8).to(self.device)
        self.encoder_target = GRUBeliefEncoder(self.obs_dim, self.action_dim, hidden_dim=64, input_dim=8).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr_actor)

        # Actor Network
        self.actor = CAMO_Actor(belief_dim=64, action_dim=self.action_dim).to(self.device)
        self.actor_target = CAMO_Actor(belief_dim=64, action_dim=self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # Critics: 3 Twin Critic Pairs (Throughput, QoS primary rate, Energy)
        self.critic_thr = TwinCritics(state_dim=64, action_dim=self.action_dim).to(self.device)
        self.critic_thr_target = TwinCritics(state_dim=64, action_dim=self.action_dim).to(self.device)
        self.critic_thr_target.load_state_dict(self.critic_thr.state_dict())
        self.critic_thr_optimizer = optim.Adam(self.critic_thr.parameters(), lr=self.lr_critic)

        self.critic_qos = TwinCritics(state_dim=64, action_dim=self.action_dim).to(self.device)
        self.critic_qos_target = TwinCritics(state_dim=64, action_dim=self.action_dim).to(self.device)
        self.critic_qos_target.load_state_dict(self.critic_qos.state_dict())
        self.critic_qos_optimizer = optim.Adam(self.critic_qos.parameters(), lr=self.lr_critic)

        self.critic_nrg = TwinCritics(state_dim=64, action_dim=self.action_dim).to(self.device)
        self.critic_nrg_target = TwinCritics(state_dim=64, action_dim=self.action_dim).to(self.device)
        self.critic_nrg_target.load_state_dict(self.critic_nrg.state_dict())
        self.critic_nrg_optimizer = optim.Adam(self.critic_nrg.parameters(), lr=self.lr_critic)

        # Learnable Lagrangians in log scale
        init_val_qos = self.camo_cfg.get("lambda_qos_init", 0.1)
        init_val_nrg = self.camo_cfg.get("lambda_nrg_init", 0.1)

        alpha_qos_val = np.log(np.exp(init_val_qos) - 1.0 + 1e-9)
        alpha_nrg_val = np.log(np.exp(init_val_nrg) - 1.0 + 1e-9)

        self.alpha_qos = nn.Parameter(torch.tensor(alpha_qos_val, dtype=torch.float32, device=self.device, requires_grad=True))
        self.alpha_nrg = nn.Parameter(torch.tensor(alpha_nrg_val, dtype=torch.float32, device=self.device, requires_grad=True))

        self.lambda_optimizer = optim.Adam([self.alpha_qos, self.alpha_nrg], lr=self.lr_lambda)
        self.lambda_clamp_max = self.camo_cfg.get("lambda_clamp_max", 50.0)

        # Directional Exploration Parameters
        self.eta_explore_init = self.camo_cfg.get("eta_explore_init", 0.05)
        self.eta_explore_decay = self.camo_cfg.get("eta_explore_decay", 0.9999)
        self.eta_explore = self.eta_explore_init

    @property
    def lambda_qos(self) -> float:
        val = torch.functional.F.softplus(self.alpha_qos).item()
        return min(val, self.lambda_clamp_max)

    @property
    def lambda_nrg(self) -> float:
        val = torch.functional.F.softplus(self.alpha_nrg).item()
        return min(val, self.lambda_clamp_max)

    def select_action(self, obs: np.ndarray, info: dict, explore: bool = True) -> np.ndarray:
        """
        Select action given state history.
        """
        obs_history = info["obs_history"]
        act_history = info["act_history"]
        dec_history = info["dec_history"]
        out_history = info["out_history"]

        # Concatenate histories to shape (seq_len, 8)
        history = np.concatenate([obs_history, act_history, dec_history, out_history], axis=-1)
        history_tensor = torch.as_tensor(history, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            belief = self.encoder(history_tensor)
            action = self.actor(belief).cpu().data.numpy().flatten()

        if explore:
            # 1. Standard noise
            noise = np.random.normal(0, self.expl_noise, size=self.action_dim)

            # 2. Safety Gradient Directional Exploration
            belief_tensor = belief.clone().detach().requires_grad_(True)
            action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_(True)

            q_qos = self.critic_qos.Q1(belief_tensor, action_tensor)
            q_nrg = self.critic_nrg.Q1(belief_tensor, action_tensor)

            grad_qos = torch.autograd.grad(q_qos.sum(), action_tensor, retain_graph=True)[0].cpu().data.numpy().flatten()
            grad_nrg = torch.autograd.grad(q_nrg.sum(), action_tensor)[0].cpu().data.numpy().flatten()

            # Moving in positive direction of QoS gradient (maximize PU rate) and negative of Energy gradient (minimize power)
            safety_bias = self.eta_explore * (self.lambda_qos * grad_qos - self.lambda_nrg * grad_nrg)
            self.eta_explore = max(0.0, self.eta_explore * self.eta_explore_decay)

            action = action + noise + safety_bias

        return np.clip(action, 0.0, 1.0)

    def train(self, writer: SummaryWriter) -> dict:
        """
        Perform one training step of Overlay TD3.
        """
        self.total_it += 1
        metrics_log = {}

        # Sample sequences with overlay metrics
        (
            hist_seq,
            next_hist_seq,
            action_t,
            reward,
            done,
            r_thr,
            r_qos,
            r_nrg,
        ) = self.replay_buffer.sample_sequences_overlay(self.batch_size)

        # Compute current and next belief states
        belief = self.encoder(hist_seq)

        with torch.no_grad():
            next_belief = self.encoder_target(next_hist_seq)
            noise = (torch.randn_like(action_t) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_belief) + noise).clamp(0.0, 1.0)

            target_q_thr = r_thr + (1 - done) * self.gamma * torch.min(
                *self.critic_thr_target.evaluate(next_belief, next_action)
            )
            target_q_qos = r_qos + (1 - done) * self.gamma * torch.min(
                *self.critic_qos_target.evaluate(next_belief, next_action)
            )
            target_q_nrg = r_nrg + (1 - done) * self.gamma * torch.min(
                *self.critic_nrg_target.evaluate(next_belief, next_action)
            )

        detached_belief = belief.detach()

        # 1. Update Throughput Critics
        current_q1_thr, current_q2_thr = self.critic_thr.evaluate(detached_belief, action_t)
        loss_critic_thr = nn.functional.mse_loss(current_q1_thr, target_q_thr) + nn.functional.mse_loss(
            current_q2_thr, target_q_thr
        )
        self.critic_thr_optimizer.zero_grad()
        loss_critic_thr.backward()
        self.critic_thr_optimizer.step()

        # 2. Update Primary User QoS Critics
        current_q1_qos, current_q2_qos = self.critic_qos.evaluate(detached_belief, action_t)
        loss_critic_qos = nn.functional.mse_loss(current_q1_qos, target_q_qos) + nn.functional.mse_loss(
            current_q2_qos, target_q_qos
        )
        self.critic_qos_optimizer.zero_grad()
        loss_critic_qos.backward()
        self.critic_qos_optimizer.step()

        # 3. Update Energy Critics
        current_q1_nrg, current_q2_nrg = self.critic_nrg.evaluate(detached_belief, action_t)
        loss_critic_nrg = nn.functional.mse_loss(current_q1_nrg, target_q_nrg) + nn.functional.mse_loss(
            current_q2_nrg, target_q_nrg
        )
        self.critic_nrg_optimizer.zero_grad()
        loss_critic_nrg.backward()
        self.critic_nrg_optimizer.step()

        # 4. Update Adaptive Lagrangians (gradient ascent)
        lambda_qos_val = torch.functional.F.softplus(self.alpha_qos)
        lambda_nrg_val = torch.functional.F.softplus(self.alpha_nrg)

        violation_qos = self.pu_rate_threshold - current_q1_qos.detach()
        violation_nrg = current_q1_nrg.detach() - self.energy_limit

        loss_lambda = -lambda_qos_val * violation_qos.mean() - lambda_nrg_val * violation_nrg.mean()

        self.lambda_optimizer.zero_grad()
        loss_lambda.backward()
        self.lambda_optimizer.step()

        # 5. Delayed Policy & Encoder Updates
        if self.total_it % self.policy_delay == 0:
            detached_belief = belief.detach()
            actions_pred = self.actor(detached_belief)

            q_thr_pred = self.critic_thr.Q1(detached_belief, actions_pred)
            q_qos_pred = self.critic_qos.Q1(detached_belief, actions_pred)
            q_nrg_pred = self.critic_nrg.Q1(detached_belief, actions_pred)

            penalty_qos = self.lambda_qos * (self.pu_rate_threshold - q_qos_pred)
            penalty_nrg = self.lambda_nrg * (q_nrg_pred - self.energy_limit)

            actor_loss = -(q_thr_pred - penalty_qos - penalty_nrg).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Encoder weights
            encoder_belief = self.encoder(hist_seq)
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

            for param, target_param in zip(self.critic_qos.parameters(), self.critic_qos_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_nrg.parameters(), self.critic_nrg_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            writer.add_scalar("Loss/Actor", actor_loss.item(), self.total_it)

        # Log metrics
        writer.add_scalar("Loss/Critic_Throughput", loss_critic_thr.item(), self.total_it)
        writer.add_scalar("Loss/Critic_QoS", loss_critic_qos.item(), self.total_it)
        writer.add_scalar("Loss/Critic_Energy", loss_critic_nrg.item(), self.total_it)
        writer.add_scalar("Lagrangian/Lambda_QoS", self.lambda_qos, self.total_it)
        writer.add_scalar("Lagrangian/Lambda_Energy", self.lambda_nrg, self.total_it)
        writer.add_scalar("Lagrangian/Violation_QoS_bps_Hz", violation_qos.mean().item(), self.total_it)
        writer.add_scalar("Lagrangian/Violation_Energy_Watts", violation_nrg.mean().item(), self.total_it)

        return metrics_log

    def save(self, filepath: str):
        """
        Save Overlay TD3 state.
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
            "critic_qos_state_dict": self.critic_qos.state_dict(),
            "critic_qos_target_state_dict": self.critic_qos_target.state_dict(),
            "critic_qos_optimizer_state_dict": self.critic_qos_optimizer.state_dict(),
            "critic_nrg_state_dict": self.critic_nrg.state_dict(),
            "critic_nrg_target_state_dict": self.critic_nrg_target.state_dict(),
            "critic_nrg_optimizer_state_dict": self.critic_nrg_optimizer.state_dict(),
            "alpha_qos": self.alpha_qos.item(),
            "alpha_nrg": self.alpha_nrg.item(),
            "lambda_optimizer_state_dict": self.lambda_optimizer.state_dict(),
            "eta_explore": self.eta_explore,
        }
        torch.save(checkpoint, filepath)
        print(f"Overlay TD3 Checkpoint successfully saved to: {filepath}")

    def load(self, filepath: str):
        """
        Load Overlay TD3 state.
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
        self.critic_qos.load_state_dict(checkpoint["critic_qos_state_dict"])
        self.critic_qos_target.load_state_dict(checkpoint["critic_qos_target_state_dict"])
        self.critic_qos_optimizer.load_state_dict(checkpoint["critic_qos_optimizer_state_dict"])
        self.critic_nrg.load_state_dict(checkpoint["critic_nrg_state_dict"])
        self.critic_nrg_target.load_state_dict(checkpoint["critic_nrg_target_state_dict"])
        self.critic_nrg_optimizer.load_state_dict(checkpoint["critic_nrg_optimizer_state_dict"])
        
        # Load alphas
        alpha_qos_val = checkpoint.get("alpha_qos", checkpoint.get("alpha_qos"))
        alpha_nrg_val = checkpoint.get("alpha_nrg", checkpoint.get("alpha_nrg"))
        self.alpha_qos.data = torch.tensor(alpha_qos_val, dtype=torch.float32, device=self.device)
        self.alpha_nrg.data = torch.tensor(alpha_nrg_val, dtype=torch.float32, device=self.device)
        
        self.lambda_optimizer.load_state_dict(checkpoint["lambda_optimizer_state_dict"])
        self.eta_explore = checkpoint["eta_explore"]
        print(f"Overlay TD3 Checkpoint successfully loaded from: {filepath}")
