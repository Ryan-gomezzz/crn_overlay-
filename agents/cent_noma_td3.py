"""
Centralized Single-Agent TD3 for NOMA Overlay CRN.
Author: Ryan
"""

import copy
import math
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from agents.cent_noma_td3_networks import CentActorNetwork
from agents.matd3_networks import MACriticNetwork
from agents.ma_buffers import MAReplayBuffer


class CentNOMATD3Agent:
    def __init__(self, config: dict, device: str = "cpu"):
        self.device = torch.device(device)
        self.config = config
        
        # Hyperparameters
        train_cfg = config.get("training", {})
        self.gamma = train_cfg.get("gamma", 0.99)
        self.tau = train_cfg.get("tau", 0.005)
        self.policy_delay = train_cfg.get("policy_delay", 2)
        self.exploration_noise = train_cfg.get("exploration_noise", 0.1)
        self.policy_noise = train_cfg.get("policy_noise", 0.2)
        self.noise_clip = train_cfg.get("noise_clip", 0.5)
        self.batch_size = train_cfg.get("batch_size", 128)
        self.lr_actor = train_cfg.get("lr_actor", 3e-4)
        self.lr_critic = train_cfg.get("lr_critic", 3e-4)
        
        camo_cfg = config.get("camo_td3", {})
        self.lr_lambda = camo_cfg.get("lr_lambda", 1e-3)
        self.history_length = camo_cfg.get("history_length", 10)
        
        # Constraints (Discounted to match Q-values)
        self.pu_rate_threshold = camo_cfg.get("pu_rate_threshold", 0.5) / (1.0 - self.gamma)
        self.energy_limit_watts = camo_cfg.get("energy_limit_watts", 0.1) / (1.0 - self.gamma)
        
        mu_cfg = config.get("multi_user", {})
        self.num_agents = mu_cfg.get("num_su", 3)
        self.obs_dim = 8
        self.action_dim = self.num_agents + 1
        
        # --- Networks ---
        # Centralized Actor
        self.actor = CentActorNetwork(num_agents=self.num_agents, obs_dim=self.obs_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        
        # Centralized Critics (Thr, QoS, Nrg)
        self.critic_thr = MACriticNetwork(num_agents=self.num_agents).to(self.device)
        self.critic_thr_target = copy.deepcopy(self.critic_thr)
        
        self.critic_qos = MACriticNetwork(num_agents=self.num_agents).to(self.device)
        self.critic_qos_target = copy.deepcopy(self.critic_qos)
        
        self.critic_nrg = MACriticNetwork(num_agents=self.num_agents).to(self.device)
        self.critic_nrg_target = copy.deepcopy(self.critic_nrg)
        
        # --- Optimizers ---
        self.actor_opt = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_thr_opt = Adam(self.critic_thr.parameters(), lr=self.lr_critic)
        self.critic_qos_opt = Adam(self.critic_qos.parameters(), lr=self.lr_critic)
        self.critic_nrg_opt = Adam(self.critic_nrg.parameters(), lr=self.lr_critic)
        
        # --- Lagrangians ---
        init_qos = camo_cfg.get("lambda_qos_init", 0.1)
        init_nrg = camo_cfg.get("lambda_nrg_init", 0.1)
        alpha_qos = math.log(math.exp(init_qos) - 1 + 1e-9)
        alpha_nrg = math.log(math.exp(init_nrg) - 1 + 1e-9)
        
        self.alpha_qos = torch.tensor([alpha_qos], dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_nrg = torch.tensor([alpha_nrg], dtype=torch.float32, device=self.device, requires_grad=True)
        self.lambda_optimizer = Adam([self.alpha_qos, self.alpha_nrg], lr=self.lr_lambda)
        
        self.lambda_clamp_max = camo_cfg.get("lambda_clamp_max", 50.0)
        
        # Replay Buffer
        self.replay_buffer = MAReplayBuffer(
            capacity=train_cfg.get("buffer_size", 100000),
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            sequence_length=self.history_length,
            device=self.device
        )
        self.total_it = 0

    @property
    def lambda_qos(self) -> torch.Tensor:
        val = F.softplus(self.alpha_qos)
        return torch.clamp(val, max=self.lambda_clamp_max)

    @property
    def lambda_nrg(self) -> torch.Tensor:
        val = F.softplus(self.alpha_nrg)
        return torch.clamp(val, max=self.lambda_clamp_max)

    def select_action(self, obs: np.ndarray, info: dict, explore: bool = False) -> np.ndarray:
        h_obs = info["obs_history"]
        h_act = info["act_history"]
        h_dec = info["dec_history"]
        h_out = info["out_history"]
        
        hist = np.concatenate([h_obs, h_act, h_dec, h_out], axis=2)
        hist_tensor = torch.FloatTensor(hist).unsqueeze(0).to(self.device)  # (1, N, L, 11)
        
        self.actor.eval()
        with torch.no_grad():
            actions_tensor = self.actor(hist_tensor)  # (1, N+1)
        self.actor.train()
        
        actions = actions_tensor.cpu().numpy()[0]
        
        if explore:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            actions = actions + noise
            actions = np.clip(actions, 0.0, 1.0)
            
        return actions

    def train(self, writer: Optional[Any] = None) -> None:
        self.total_it += 1
        
        try:
            hist_seqs, next_hist_seqs, action_t, reward, done, r_thr, r_qos, r_nrg = self.replay_buffer.sample_sequences(self.batch_size)
        except Exception:
            return
            
        with torch.no_grad():
            next_global_belief = self.actor_target.get_belief(next_hist_seqs) # (B, N*64)
            n_a = self.actor_target(next_hist_seqs) # (B, N+1)
            
            noise = (torch.randn_like(n_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            n_a = (n_a + noise).clamp(0.0, 1.0)
            
            Q1_thr_tgt, Q2_thr_tgt = self.critic_thr_target.evaluate(next_global_belief, n_a)
            Q1_qos_tgt, Q2_qos_tgt = self.critic_qos_target.evaluate(next_global_belief, n_a)
            Q1_nrg_tgt, Q2_nrg_tgt = self.critic_nrg_target.evaluate(next_global_belief, n_a)
            
            target_thr = r_thr + (1 - done) * self.gamma * torch.min(Q1_thr_tgt, Q2_thr_tgt)
            target_qos = r_qos + (1 - done) * self.gamma * torch.min(Q1_qos_tgt, Q2_qos_tgt)
            target_nrg = r_nrg + (1 - done) * self.gamma * torch.min(Q1_nrg_tgt, Q2_nrg_tgt)
            
        global_belief = self.actor.get_belief(hist_seqs).detach()
        
        # Critic Updates
        Q1_thr, Q2_thr = self.critic_thr.evaluate(global_belief, action_t)
        loss_thr = F.mse_loss(Q1_thr, target_thr) + F.mse_loss(Q2_thr, target_thr)
        self.critic_thr_opt.zero_grad()
        loss_thr.backward()
        self.critic_thr_opt.step()
        
        Q1_qos, Q2_qos = self.critic_qos.evaluate(global_belief, action_t)
        loss_qos = F.mse_loss(Q1_qos, target_qos) + F.mse_loss(Q2_qos, target_qos)
        self.critic_qos_opt.zero_grad()
        loss_qos.backward()
        self.critic_qos_opt.step()
        
        Q1_nrg, Q2_nrg = self.critic_nrg.evaluate(global_belief, action_t)
        loss_nrg = F.mse_loss(Q1_nrg, target_nrg) + F.mse_loss(Q2_nrg, target_nrg)
        self.critic_nrg_opt.zero_grad()
        loss_nrg.backward()
        self.critic_nrg_opt.step()
        
        # Lagrangian Update
        val_qos = F.softplus(self.alpha_qos)
        val_nrg = F.softplus(self.alpha_nrg)
        
        violation_qos = self.pu_rate_threshold - Q1_qos.detach()
        violation_nrg = Q1_nrg.detach() - self.energy_limit_watts
        
        loss_lambda = -(val_qos * violation_qos.mean()) - (val_nrg * violation_nrg.mean())
        self.lambda_optimizer.zero_grad()
        loss_lambda.backward()
        self.lambda_optimizer.step()
        
        # Actor Update
        if self.total_it % self.policy_delay == 0:
            gb_curr = self.actor.get_belief(hist_seqs)
            a_curr = self.actor(hist_seqs)
            
            Q_thr_pi = self.critic_thr(gb_curr, a_curr)
            Q_qos_pi = self.critic_qos(gb_curr, a_curr)
            Q_nrg_pi = self.critic_nrg(gb_curr, a_curr)
            
            actor_loss = -(Q_thr_pi 
                           - self.lambda_qos.detach() * (self.pu_rate_threshold - Q_qos_pi) 
                           - self.lambda_nrg.detach() * (Q_nrg_pi - self.energy_limit_watts)).mean()
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            # Soft targets
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            self.actor.gru.flatten_parameters()
            self.actor_target.gru.flatten_parameters()
                
            for param, target_param in zip(self.critic_thr.parameters(), self.critic_thr_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.critic_qos.parameters(), self.critic_qos_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.critic_nrg.parameters(), self.critic_nrg_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        if writer is not None and self.total_it % 100 == 0:
            writer.add_scalar("train/lambda_qos", self.lambda_qos.item(), self.total_it)
            writer.add_scalar("train/lambda_nrg", self.lambda_nrg.item(), self.total_it)

    def save(self, filename: str) -> None:
        pass

    def load(self, filename: str) -> None:
        pass
