"""
Training Loop and Agent Implementation for TD3 and CAMO-TD3.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from agents.buffers import SequenceReplayBuffer
from agents.models import CAMO_Actor, GRUBeliefEncoder, TD3_Actor, TwinCritics
from simulator.utils import dbm_to_watt


class TD3Agent:
    """
    Unified TD3 & CAMO-TD3 Agent class.
    Handles online interaction, action selection, optimization, logging, and 
    checkpointing.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device

        # Parse config parameters
        self.algo_cfg = config.get("training", {})
        self.camo_cfg = config.get("camo_td3", {})

        self.algorithm_name = config.get("algorithm", {}).get("name", "TD3")
        # Standardize algorithm names for backward compatibility and presentation renames
        name_map = {
            "T3": "TD3",
            "UNDERLAY_TD3": "CAMO_TD3",
            "OVERLAY_TD3": "OVERLAY_CAMO_TD3",
        }
        self.algorithm_name = name_map.get(self.algorithm_name, self.algorithm_name)
        
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
        self.obs_dim = 4
        self.action_dim = 2

        # Physical limits for constraints in CAMO
        self.interference_limit = dbm_to_watt(
            self.camo_cfg.get("interference_limit_dbm", -50.0)
        )
        self.energy_limit = self.camo_cfg.get("energy_limit_watts", 0.1)

        # Replay Buffer
        self.replay_buffer = SequenceReplayBuffer(
            capacity=self.algo_cfg.get("buffer_size", 100000),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            sequence_length=self.seq_len,
            device=self.device,
        )

        # Setup networks based on algorithm choice
        if self.algorithm_name == "CAMO_TD3":
            # Recurrent GRU Belief Encoders
            self.encoder = GRUBeliefEncoder(
                self.obs_dim, self.action_dim, hidden_dim=64
            ).to(self.device)
            self.encoder_target = GRUBeliefEncoder(
                self.obs_dim, self.action_dim, hidden_dim=64
            ).to(self.device)
            self.encoder_target.load_state_dict(self.encoder.state_dict())
            self.encoder_optimizer = optim.Adam(
                self.encoder.parameters(), lr=self.lr_actor
            )

            # Actor Network
            self.actor = CAMO_Actor(belief_dim=64, action_dim=self.action_dim).to(self.device)
            self.actor_target = CAMO_Actor(belief_dim=64, action_dim=self.action_dim).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(), lr=self.lr_actor
            )

            # Critics: 3 Separate Twin Critic Pairs (Throughput, Interference, Energy)
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

            # Adaptive Lagrangian learnable parameters
            # Store in log scale to enforce positive lambdas via exp
            init_val_inf = self.camo_cfg.get("lambda_inf_init", 0.1)
            init_val_nrg = self.camo_cfg.get("lambda_nrg_init", 0.1)

            # softplus inverse: log(exp(x) - 1)
            alpha_inf_val = np.log(np.exp(init_val_inf) - 1.0 + 1e-9)
            alpha_nrg_val = np.log(np.exp(init_val_nrg) - 1.0 + 1e-9)

            self.alpha_inf = nn.Parameter(
                torch.tensor(
                    alpha_inf_val,
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=True,
                )
            )
            self.alpha_nrg = nn.Parameter(
                torch.tensor(
                    alpha_nrg_val,
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=True,
                )
            )

            self.lambda_optimizer = optim.Adam(
                [self.alpha_inf, self.alpha_nrg], lr=self.lr_lambda
            )
            self.lambda_clamp_max = self.camo_cfg.get("lambda_clamp_max", 50.0)

            # Directional Exploration Parameters
            self.eta_explore_init = self.camo_cfg.get("eta_explore_init", 0.05)
            self.eta_explore_decay = self.camo_cfg.get(
                "eta_explore_decay", 0.9999
            )
            self.eta_explore = self.eta_explore_init

        elif self.algorithm_name == "OVERLAY_CAMO_TD3":
            # Recurrent GRU Belief Encoders (input_dim = 8 for obs, act, dec, out)
            self.encoder = GRUBeliefEncoder(self.obs_dim, self.action_dim, hidden_dim=64, input_dim=8).to(self.device)
            self.encoder_target = GRUBeliefEncoder(self.obs_dim, self.action_dim, hidden_dim=64, input_dim=8).to(self.device)
            self.encoder_target.load_state_dict(self.encoder.state_dict())
            self.encoder_optimizer = optim.Adam(
                self.encoder.parameters(), lr=self.lr_actor
            )

            # Actor Network
            self.actor = CAMO_Actor(belief_dim=64, action_dim=self.action_dim).to(self.device)
            self.actor_target = CAMO_Actor(belief_dim=64, action_dim=self.action_dim).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(), lr=self.lr_actor
            )

            # Critics: 3 Separate Twin Critic Pairs (Throughput, QoS rate, Energy)
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

            # Adaptive Lagrangian learnable parameters
            init_val_qos = self.camo_cfg.get("lambda_qos_init", 0.1)
            init_val_nrg = self.camo_cfg.get("lambda_nrg_init", 0.1)
            self.pu_rate_threshold = self.camo_cfg.get(
                "pu_rate_threshold", 1.5
            )

            alpha_qos_val = np.log(np.exp(init_val_qos) - 1.0 + 1e-9)
            alpha_nrg_val = np.log(np.exp(init_val_nrg) - 1.0 + 1e-9)

            self.alpha_qos = nn.Parameter(
                torch.tensor(
                    alpha_qos_val,
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=True,
                )
            )
            self.alpha_nrg = nn.Parameter(
                torch.tensor(
                    alpha_nrg_val,
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=True,
                )
            )

            self.lambda_optimizer = optim.Adam(
                [self.alpha_qos, self.alpha_nrg], lr=self.lr_lambda
            )
            self.lambda_clamp_max = self.camo_cfg.get("lambda_clamp_max", 50.0)

            # Directional Exploration Parameters
            self.eta_explore_init = self.camo_cfg.get("eta_explore_init", 0.05)
            self.eta_explore_decay = self.camo_cfg.get(
                "eta_explore_decay", 0.9999
            )
            self.eta_explore = self.eta_explore_init

        else:
            # Standard TD3 Setup
            self.actor = TD3_Actor(obs_dim=self.obs_dim, action_dim=self.action_dim).to(self.device)
            self.actor_target = TD3_Actor(obs_dim=self.obs_dim, action_dim=self.action_dim).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(), lr=self.lr_actor
            )

            self.critic = TwinCritics(
                state_dim=self.obs_dim, action_dim=self.action_dim
            ).to(self.device)
            self.critic_target = TwinCritics(
                state_dim=self.obs_dim, action_dim=self.action_dim
            ).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=self.lr_critic
            )

        self.total_it = 0

    @property
    def lambda_inf(self) -> float:
        if not hasattr(self, "alpha_inf"):
            return 0.0
        val = torch.functional.F.softplus(self.alpha_inf).item()
        return min(val, self.lambda_clamp_max)

    @property
    def lambda_qos(self) -> float:
        if not hasattr(self, "alpha_qos"):
            return 0.0
        val = torch.functional.F.softplus(self.alpha_qos).item()
        return min(val, self.lambda_clamp_max)

    @property
    def lambda_nrg(self) -> float:
        if not hasattr(self, "alpha_nrg"):
            return 0.0
        val = torch.functional.F.softplus(self.alpha_nrg).item()
        return min(val, self.lambda_clamp_max)

    def select_action(
        self, obs: np.ndarray, info: dict, explore: bool = True
    ) -> np.ndarray:
        """
        Select action given state.
        If CAMO-TD3 or OVERLAY_CAMO_TD3, uses history sequences.
        """
        if self.algorithm_name == "OVERLAY_CAMO_TD3":
            obs_history = info["obs_history"]
            act_history = info["act_history"]
            dec_history = info["dec_history"]
            out_history = info["out_history"]

            # Concatenate to shape (seq_len, 8)
            history = np.concatenate([obs_history, act_history, dec_history, out_history], axis=-1)
            history_tensor = torch.as_tensor(history, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                belief = self.encoder(history_tensor)
                action = self.actor(belief).cpu().data.numpy().flatten()

            if explore:
                noise = np.random.normal(0, self.expl_noise, size=self.action_dim)
                
                # Safety gradient directional exploration
                belief_tensor = belief.clone().detach().requires_grad_(True)
                action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_(True)
                
                q_qos = self.critic_qos.Q1(belief_tensor, action_tensor)
                q_nrg = self.critic_nrg.Q1(belief_tensor, action_tensor)
                
                grad_qos = torch.autograd.grad(q_qos.sum(), action_tensor, retain_graph=True)[0].cpu().data.numpy().flatten()
                grad_nrg = torch.autograd.grad(q_nrg.sum(), action_tensor)[0].cpu().data.numpy().flatten()
                
                safety_bias = self.eta_explore * (self.lambda_qos * grad_qos - self.lambda_nrg * grad_nrg)
                self.eta_explore = max(0.0, self.eta_explore * self.eta_explore_decay)
                
                action = action + noise + safety_bias

            return np.clip(action, 0.0, 1.0)

        elif self.algorithm_name == "CAMO_TD3":
            obs_history = torch.as_tensor(
                info["obs_history"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            act_history = torch.as_tensor(
                info["act_history"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                belief = self.encoder(obs_history, act_history)
                action = self.actor(belief).cpu().data.numpy().flatten()

            if explore:
                # 1. Generate standard Gaussian exploration noise
                noise = np.random.normal(0, self.expl_noise, size=self.action_dim)
                
                # 2. Add Directional Exploration Safety Bias
                # Compute gradient of constraint Q1 values
                belief_tensor = belief.clone().detach().requires_grad_(True)
                action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_(True)
                
                # Forward constraint critics
                q_inf = self.critic_inf.Q1(belief_tensor, action_tensor)
                q_nrg = self.critic_nrg.Q1(belief_tensor, action_tensor)

                # Compute gradients of constraints w.r.t action
                grad_inf = torch.autograd.grad(q_inf.sum(), action_tensor, retain_graph=True)[0].cpu().data.numpy().flatten()
                grad_nrg = torch.autograd.grad(q_nrg.sum(), action_tensor)[0].cpu().data.numpy().flatten()
                
                # Directional explore vector: - eta * (lambda_inf * grad_inf + lambda_nrg * grad_nrg)
                # Moving in the negative direction of constraint gradients decreases violation risk.
                safety_bias = - self.eta_explore * (self.lambda_inf * grad_inf + self.lambda_nrg * grad_nrg)
                
                # Decay eta
                self.eta_explore = max(
                    0.0, self.eta_explore * self.eta_explore_decay
                )

                # Final explored action
                action = action + noise + safety_bias

            return np.clip(action, 0.0, 1.0)
        else:
            # Standard TD3 Select Action
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                action = self.actor(obs_tensor).cpu().data.numpy().flatten()
            if explore:
                noise = np.random.normal(
                    0, self.expl_noise, size=self.action_dim
                )
                action = action + noise
            return np.clip(action, 0.0, 1.0)

    def train(self, writer: SummaryWriter) -> dict:
        """
        Sample batch and perform training step.
        """
        self.total_it += 1
        metrics_log = {}

        if self.algorithm_name == "OVERLAY_CAMO_TD3":
            # OVERLAY_CAMO_TD3 Optimization Loop
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

                # Select target action with target policy smoothing noise
                noise = (
                    torch.randn_like(action_t) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                next_action = (
                    self.actor_target(next_belief) + noise
                ).clamp(0.0, 1.0)

                # Compute twin target values for the three critics
                target_q_thr = r_thr + (1 - done) * self.gamma * torch.min(
                    *self.critic_thr_target.evaluate(
                        next_belief, next_action
                    )
                )
                target_q_qos = r_qos + (1 - done) * self.gamma * torch.min(
                    *self.critic_qos_target.evaluate(
                        next_belief, next_action
                    )
                )
                target_q_nrg = r_nrg + (1 - done) * self.gamma * torch.min(
                    *self.critic_nrg_target.evaluate(
                        next_belief, next_action
                    )
                )

            # Detach belief for critic updates
            detached_belief = belief.detach()

            # 1. Update Twin Throughput Critics
            current_q1_thr, current_q2_thr = self.critic_thr.evaluate(
                detached_belief, action_t
            )
            loss_critic_thr = nn.functional.mse_loss(
                current_q1_thr, target_q_thr
            ) + nn.functional.mse_loss(current_q2_thr, target_q_thr)
            self.critic_thr_optimizer.zero_grad()
            loss_critic_thr.backward()
            self.critic_thr_optimizer.step()

            # 2. Update Twin Primary User QoS Critics
            current_q1_qos, current_q2_qos = self.critic_qos.evaluate(
                detached_belief, action_t
            )
            loss_critic_qos = nn.functional.mse_loss(
                current_q1_qos, target_q_qos
            ) + nn.functional.mse_loss(current_q2_qos, target_q_qos)
            self.critic_qos_optimizer.zero_grad()
            loss_critic_qos.backward()
            self.critic_qos_optimizer.step()

            # 3. Update Twin Energy Critics
            current_q1_nrg, current_q2_nrg = self.critic_nrg.evaluate(
                detached_belief, action_t
            )
            loss_critic_nrg = nn.functional.mse_loss(
                current_q1_nrg, target_q_nrg
            ) + nn.functional.mse_loss(current_q2_nrg, target_q_nrg)
            self.critic_nrg_optimizer.zero_grad()
            loss_critic_nrg.backward()
            self.critic_nrg_optimizer.step()

            # 4. Update Adaptive Lagrangian multipliers
            lambda_qos_val = torch.functional.F.softplus(self.alpha_qos)
            lambda_nrg_val = torch.functional.F.softplus(self.alpha_nrg)

            # Violations
            violation_qos = (
                self.pu_rate_threshold - current_q1_qos.detach()
            )
            violation_nrg = (
                current_q1_nrg.detach() - self.energy_limit
            )

            loss_lambda = (
                -lambda_qos_val * violation_qos.mean()
                - lambda_nrg_val * violation_nrg.mean()
            )

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

                # Penalty terms
                penalty_qos = self.lambda_qos * (
                    self.pu_rate_threshold - q_qos_pred
                )
                penalty_nrg = self.lambda_nrg * (
                    q_nrg_pred - self.energy_limit
                )

                actor_loss = - (q_thr_pred - penalty_qos - penalty_nrg).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update Encoder weights
                encoder_belief = self.encoder(hist_seq)
                self.encoder_optimizer.zero_grad()
                actor_loss_recurrent = - (self.critic_thr.Q1(encoder_belief, actions_pred.detach())).mean()
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

                writer.add_scalar(
                    "Loss/Actor", actor_loss.item(), self.total_it
                )

            # Log metrics
            writer.add_scalar("Loss/Critic_Throughput", loss_critic_thr.item(), self.total_it)
            writer.add_scalar("Loss/Critic_QoS", loss_critic_qos.item(), self.total_it)
            writer.add_scalar("Loss/Critic_Energy", loss_critic_nrg.item(), self.total_it)
            writer.add_scalar("Lagrangian/Lambda_QoS", self.lambda_qos, self.total_it)
            writer.add_scalar("Lagrangian/Lambda_Energy", self.lambda_nrg, self.total_it)
            writer.add_scalar("Lagrangian/Violation_QoS_bps_Hz", violation_qos.mean().item(), self.total_it)
            writer.add_scalar("Lagrangian/Violation_Energy_Watts", violation_nrg.mean().item(), self.total_it)

        elif self.algorithm_name == "CAMO_TD3":
            # CAMO-TD3 Optimization Loop
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
                
                # Select target action with target policy smoothing noise
                noise = (torch.randn_like(next_act_seq[:, -1, :]) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )
                next_action = (self.actor_target(next_belief) + noise).clamp(0.0, 1.0)

                # Compute twin target values
                target_q_thr = r_thr + (1 - done) * self.gamma * torch.min(
                    *self.critic_thr_target.evaluate(
                        next_belief, next_action
                    )
                )
                target_q_inf = r_inf + (1 - done) * self.gamma * torch.min(
                    *self.critic_inf_target.evaluate(
                        next_belief, next_action
                    )
                )
                target_q_nrg = r_nrg + (1 - done) * self.gamma * torch.min(
                    *self.critic_nrg_target.evaluate(
                        next_belief, next_action
                    )
                )

            # Detach the belief state so we don't backpropagate through the encoder multiple times
            detached_belief = belief.detach()

            # 1. Update Twin Throughput Critics
            current_q1_thr, current_q2_thr = self.critic_thr.evaluate(
                detached_belief, act_seq[:, -1, :]
            )
            loss_critic_thr = nn.functional.mse_loss(
                current_q1_thr, target_q_thr
            ) + nn.functional.mse_loss(current_q2_thr, target_q_thr)
            self.critic_thr_optimizer.zero_grad()
            loss_critic_thr.backward()
            self.critic_thr_optimizer.step()

            # 2. Update Twin Interference Critics
            current_q1_inf, current_q2_inf = self.critic_inf.evaluate(
                detached_belief, act_seq[:, -1, :]
            )
            loss_critic_inf = nn.functional.mse_loss(
                current_q1_inf, target_q_inf
            ) + nn.functional.mse_loss(current_q2_inf, target_q_inf)
            self.critic_inf_optimizer.zero_grad()
            loss_critic_inf.backward()
            self.critic_inf_optimizer.step()

            # 3. Update Twin Energy Critics
            current_q1_nrg, current_q2_nrg = self.critic_nrg.evaluate(
                detached_belief, act_seq[:, -1, :]
            )
            loss_critic_nrg = nn.functional.mse_loss(
                current_q1_nrg, target_q_nrg
            ) + nn.functional.mse_loss(current_q2_nrg, target_q_nrg)
            self.critic_nrg_optimizer.zero_grad()
            loss_critic_nrg.backward()
            self.critic_nrg_optimizer.step()

            # 4. Update Adaptive Lagrangian multiplier parameter (gradient ascent on lambdas)
            # Loss = - lambda_inf * (Q_inf - limit) - lambda_nrg * (Q_nrg - limit)
            lambda_inf_val = torch.functional.F.softplus(self.alpha_inf)
            lambda_nrg_val = torch.functional.F.softplus(self.alpha_nrg)

            # Compute constraint violations
            violation_inf = (
                current_q1_inf.detach() - self.interference_limit
            )
            violation_nrg = current_q1_nrg.detach() - self.energy_limit

            loss_lambda = (
                -lambda_inf_val * violation_inf.mean()
                - lambda_nrg_val * violation_nrg.mean()
            )

            self.lambda_optimizer.zero_grad()
            loss_lambda.backward()
            self.lambda_optimizer.step()

            # 5. Delayed Policy & Encoder Updates
            if self.total_it % self.policy_delay == 0:
                # Update Actor to maximize Throughput while respecting Interference & Energy constraints
                # Loss = - (Q_thr - lambda_inf * (Q_inf - limit) - lambda_nrg * (Q_nrg - limit))
                # Note: We detach current belief for actor gradient computation to decouple encoder/actor representations
                detached_belief = belief.detach()
                actions_pred = self.actor(detached_belief)
                
                q_thr_pred = self.critic_thr.Q1(detached_belief, actions_pred)
                q_inf_pred = self.critic_inf.Q1(detached_belief, actions_pred)
                q_nrg_pred = self.critic_nrg.Q1(detached_belief, actions_pred)

                # Penalty terms
                penalty_inf = self.lambda_inf * (
                    q_inf_pred - self.interference_limit
                )
                penalty_nrg = self.lambda_nrg * (
                    q_nrg_pred - self.energy_limit
                )

                actor_loss = - (q_thr_pred - penalty_inf - penalty_nrg).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update Encoder weights
                encoder_belief = self.encoder(obs_seq, act_seq)
                eq_thr = self.critic_thr.Q1(encoder_belief.detach(), act_seq[:, -1, :])
                eq_inf = self.critic_inf.Q1(encoder_belief.detach(), act_seq[:, -1, :])
                eq_nrg = self.critic_nrg.Q1(encoder_belief.detach(), act_seq[:, -1, :])
                
                # Encoder objective: minimize overall critic state representation reconstruction loss
                # (For simplicity and stability, we update encoder w.r.t actor loss)
                self.encoder_optimizer.zero_grad()
                actor_loss_recurrent = - (self.critic_thr.Q1(encoder_belief, actions_pred.detach())).mean()
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

                # Write tensorboard logs
                writer.add_scalar(
                    "Loss/Actor", actor_loss.item(), self.total_it
                )

            # Log metrics
            writer.add_scalar("Loss/Critic_Throughput", loss_critic_thr.item(), self.total_it)
            writer.add_scalar("Loss/Critic_Interference", loss_critic_inf.item(), self.total_it)
            writer.add_scalar("Loss/Critic_Energy", loss_critic_nrg.item(), self.total_it)
            writer.add_scalar("Lagrangian/Lambda_Interference", self.lambda_inf, self.total_it)
            writer.add_scalar("Lagrangian/Lambda_Energy", self.lambda_nrg, self.total_it)
            writer.add_scalar("Lagrangian/Violation_Interference_Watts", violation_inf.mean().item(), self.total_it)
            writer.add_scalar("Lagrangian/Violation_Energy_Watts", violation_nrg.mean().item(), self.total_it)

        else:
            # Standard TD3 Optimization Loop
            obs, action, reward, next_obs, done = self.replay_buffer.sample_standard(self.batch_size)

            with torch.no_grad():
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                next_action = (
                    self.actor_target(next_obs) + noise
                ).clamp(0.0, 1.0)

                target_q = reward + (1 - done) * self.gamma * torch.min(
                    *self.critic_target.evaluate(next_obs, next_action)
                )

            current_q1, current_q2 = self.critic.evaluate(obs, action)
            loss_critic = nn.functional.mse_loss(
                current_q1, target_q
            ) + nn.functional.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            self.critic_optimizer.step()

            if self.total_it % self.policy_delay == 0:
                actor_loss = (
                    -self.critic.Q1(obs, self.actor(obs)).mean()
                )

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft target updates
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                writer.add_scalar(
                    "Loss/Actor", actor_loss.item(), self.total_it
                )

            writer.add_scalar(
                "Loss/Critic", loss_critic.item(), self.total_it
            )

        return metrics_log

    def save(self, filepath: str):
        """
        Save full model state checkpoint.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "algorithm": self.algorithm_name,
            "total_it": self.total_it,
            "config": self.config,
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
        }

        if self.algorithm_name == "CAMO_TD3":
            checkpoint.update({
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
            })
        elif self.algorithm_name == "OVERLAY_CAMO_TD3":
            checkpoint.update({
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
            })
        else:
            checkpoint.update({
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            })

        torch.save(checkpoint, filepath)
        print(f"Checkpoint successfully saved to: {filepath}")

    def load(self, filepath: str):
        """
        Load checkpoint.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.total_it = checkpoint["total_it"]
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(
            checkpoint["actor_target_state_dict"]
        )
        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"]
        )

        if self.algorithm_name == "CAMO_TD3":
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
            self.alpha_inf.data = torch.tensor(checkpoint["alpha_inf"], dtype=torch.float32, device=self.device)
            self.alpha_nrg.data = torch.tensor(checkpoint["alpha_nrg"], dtype=torch.float32, device=self.device)
            self.lambda_optimizer.load_state_dict(checkpoint["lambda_optimizer_state_dict"])
            self.eta_explore = checkpoint["eta_explore"]
        elif self.algorithm_name == "OVERLAY_CAMO_TD3":
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
            self.alpha_qos.data = torch.tensor(checkpoint["alpha_qos"], dtype=torch.float32, device=self.device)
            self.alpha_nrg.data = torch.tensor(checkpoint["alpha_nrg"], dtype=torch.float32, device=self.device)
            self.lambda_optimizer.load_state_dict(checkpoint["lambda_optimizer_state_dict"])
            self.eta_explore = checkpoint["eta_explore"]
        else:
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        print(f"Checkpoint successfully loaded from: {filepath}")
