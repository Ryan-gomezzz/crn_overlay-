"""
Main training and evaluation loop.
"""

import os
import random

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from agents.train_td3 import TD3Agent
from envs.crn_env import OverlayCRNEnv


def evaluate_policy(agent, env, episodes=10):
    """Evaluate the agent's policy."""
    results = {
        "total_reward": [],
        "throughput_s": [],
        "throughput_p": [],
        "ber": [],
        "outage": [],
        "average_power": [],
        "relay_success": [],
        "qos_satisfaction": [],
        "constraint_satisfaction": [],
    }

    for _ in range(episodes):
        obs, info = env.reset()
        episode_reward = 0.0

        ep_throughput_s = []
        ep_throughput_p = []
        ep_ber = []
        ep_outage = []
        ep_power = []
        ep_relay = []
        ep_qos = []
        ep_con = []

        done = False

        while not done:
            action = agent.select_action(obs, info, explore=False)
            obs, reward, done, _, info = env.step(action)

            episode_reward += reward

            ep_throughput_s.append(info.get("throughput_s", 0.0))
            ep_throughput_p.append(info.get("throughput_p", 0.0))
            ep_ber.append(info.get("ber", 0.0))
            ep_outage.append(info.get("outage", 0.0))
            ep_power.append(info.get("average_power", 0.0))
            ep_relay.append(info.get("relay_success", 0.0))

            ep_qos.append(1.0 if info.get("throughput_p", 0.0) >= 1.5 else 0.0)

            ep_con.append(
                1.0
                if info.get("outage", 0.0) == 0.0
                and info.get("average_power", 0.0) <= env.energy_limit
                else 0.0
            )

        results["total_reward"].append(episode_reward)
        results["throughput_s"].append(np.mean(ep_throughput_s))
        results["throughput_p"].append(np.mean(ep_throughput_p))
        results["ber"].append(np.mean(ep_ber))
        results["outage"].append(np.mean(ep_outage))
        results["average_power"].append(np.mean(ep_power))
        results["relay_success"].append(np.mean(ep_relay))
        results["qos_satisfaction"].append(np.mean(ep_qos))
        results["constraint_satisfaction"].append(np.mean(ep_con))

    return {k: float(np.mean(v)) for k, v in results.items()}


def setup_config(config_path, overrides=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v

    return config


def train(
    config_path="configs/config.yaml",
    log_dir="experiments/logs",
    checkpoint_dir="experiments/checkpoints",
    overrides=None,
):
    config = setup_config(config_path, overrides)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    env = OverlayCRNEnv(config)
    agent = TD3Agent(config, device=device)

    algo_name = config.get("algorithm", {}).get("name", "TD3")

    writer = SummaryWriter(log_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)

    max_steps = config["training"]["max_training_steps"]
    eval_interval = config["training"]["eval_interval"]
    save_interval = config["training"]["save_interval"]
    eval_episodes = config["evaluation"]["eval_episodes"]

    obs, info = env.reset()
    step = 0
    best_reward = -float("inf")

    episode_reward = 0.0

    while step < max_steps:
        action = agent.select_action(obs, info, explore=True)
        next_obs, reward, done, _, info = env.step(action)

        agent.replay_buffer.add(obs, action, reward, next_obs, done, info)

        if agent.replay_buffer.size() > config["training"]["batch_size"]:
            agent.train(writer)

        episode_reward += reward

        if step % 100 == 0:
            writer.add_scalar("Reward/step", reward, step)
            print(f"Step {step} | Reward {reward:.3f}")

        # Evaluation
        if step % eval_interval == 0 and step > 0:
            eval_metrics = evaluate_policy(agent, env, eval_episodes)

            print("\nEVAL:", eval_metrics)

            if eval_metrics["total_reward"] > best_reward:
                best_reward = eval_metrics["total_reward"]
                agent.save(os.path.join(checkpoint_dir, "best_model.pth"))

        # Reset episode
        if done:
            obs, info = env.reset()
            episode_reward = 0.0
        else:
            obs = next_obs

        step += 1

    agent.save(os.path.join(checkpoint_dir, "final_model.pth"))
    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    train()