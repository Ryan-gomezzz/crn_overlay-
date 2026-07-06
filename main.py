"""
Entry point for the CRN-RL-Framework.
Responsible for orchestrating training and evaluation pipelines.
Author: Ryan
"""

import os
import random
import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from envs.crn_env import OverlayCRNEnv
from agents.train_td3 import TD3Agent


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_policy(agent: TD3Agent, env: OverlayCRNEnv, episodes: int = 5) -> dict:
    """
    Evaluate agent policy deterministically over multiple episodes.
    """
    eval_metrics = {
        "throughput_s": [],
        "throughput_p": [],
        "outage": [],
        "ber": [],
        "average_power": [],
        "total_reward": [],
        "relay_success": [],
        "qos_satisfaction": [],
        "constraint_satisfaction": [],
    }

    energy_limit = env.energy_limit

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        
        ep_throughput_s = []
        ep_throughput_p = []
        ep_outage = []
        ep_ber = []
        ep_average_power = []
        ep_relay_success = []
        ep_qos_satisfaction = []
        ep_constraint_satisfaction = []

        while not (done or truncated):
            # Deterministic action selection (explore=False)
            action = agent.select_action(obs, info, explore=False)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            ep_throughput_s.append(info["throughput_reward"])
            ep_throughput_p.append(info["primary_throughput"])
            ep_outage.append(info["outage"])
            ep_ber.append(info["ber"])
            ep_average_power.append(info["average_power"])

            dec = info.get("relay_decoded", 0.0)
            ep_relay_success.append(dec)

            qos_sat = 1.0 if info["outage"] == 0.0 else 0.0
            ep_qos_satisfaction.append(qos_sat)

            con_sat = 1.0 if (info["outage"] == 0.0 and info["average_power"] <= energy_limit) else 0.0
            ep_constraint_satisfaction.append(con_sat)

        eval_metrics["throughput_s"].append(np.mean(ep_throughput_s))
        eval_metrics["throughput_p"].append(np.mean(ep_throughput_p))
        eval_metrics["outage"].append(np.mean(ep_outage))
        eval_metrics["ber"].append(np.mean(ep_ber))
        eval_metrics["average_power"].append(np.mean(ep_average_power))
        eval_metrics["relay_success"].append(np.mean(ep_relay_success))
        eval_metrics["qos_satisfaction"].append(np.mean(ep_qos_satisfaction))
        eval_metrics["constraint_satisfaction"].append(np.mean(ep_constraint_satisfaction))
        eval_metrics["total_reward"].append(episode_reward)

    # Average metrics
    avg_metrics = {k: float(np.mean(v)) for k, v in eval_metrics.items()}
    return avg_metrics


def main_legacy():
    """
    Load configuration, setup environment, and run the pipeline.
    """
    print("Initializing CRN-RL-Framework...")

    # Load master configuration
    config_path = os.path.join("configs", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load experiment-specific overrides if present
    exp_path = os.path.join("configs", "experiment.yaml")
    if os.path.exists(exp_path):
        print(f"Loading experiment overrides from: {exp_path}")
        with open(exp_path, "r") as f:
            overrides = yaml.safe_load(f)
            if overrides:
                # Basic recursive merge for single-level dictionary keys
                for section, vals in overrides.items():
                    if section in config and isinstance(config[section], dict) and isinstance(vals, dict):
                        config[section].update(vals)
                    else:
                        config[section] = vals

    # Initialize environment and setup random seeds
    sim_cfg = config.get("simulation", {})
    seed = sim_cfg.get("seed", 42)
    set_seed(seed)

    env = OverlayCRNEnv(config)
    env.action_space.seed(seed)

    # Set compute device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Setup Agent
    agent = TD3Agent(config, device=device)

    # Setup directories
    log_cfg = config.get("logging", {})
    tb_enabled = log_cfg.get("tensorboard_enabled", True)
    
    algo_name = config.get("algorithm", {}).get("name", "TD3")
    run_name = f"{algo_name}_run_seed_{seed}"
    log_dir = os.path.join(log_cfg.get("log_dir", "experiments/runs/"), run_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir) if tb_enabled else None
    
    eval_cfg = config.get("evaluation", {})
    checkpoint_dir = eval_cfg.get("save_dir", "experiments/checkpoints/")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, f"{algo_name}_best_model.pth")
    final_checkpoint_path = os.path.join(checkpoint_dir, f"{algo_name}_final_model.pth")

    # Training parameters
    train_cfg = config.get("training", {})
    max_episodes = train_cfg.get("max_episodes", 2000)
    steps_per_episode = sim_cfg.get("time_steps_per_episode", 300)
    total_steps = max_episodes * steps_per_episode
    train_cfg["total_steps"] = total_steps
    start_steps = train_cfg.get("start_steps", 1000)
    eval_interval = eval_cfg.get("eval_interval", 500)
    eval_episodes = eval_cfg.get("eval_episodes", 5)

    print(f"Starting training pipeline for {algo_name}...")
    print(f"Training limits: {max_episodes} episodes, {steps_per_episode} steps per episode")
    
    obs, info = env.reset()
    episode_reward = 0
    episode_steps = 0
    best_eval_reward = -np.inf

    for step in range(1, total_steps + 1):
        episode_steps += 1

        # Select Action: Warmup with random actions to fill buffer, then use policy
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, info, explore=True)

        # Environment Step
        next_obs, reward, done, truncated, next_info = env.step(action)
        episode_reward += reward

        # Store transition in SequenceReplay/Flat Buffer
        agent.replay_buffer.add(obs, action, reward, next_obs, done or truncated, info)

        # Move to next state
        obs = next_obs
        info = next_info

        # Perform gradient optimization step
        if step >= start_steps:
            agent.train(writer)

        # Episode termination or truncation handling
        if done or truncated:
            # Write episode level statistics to TensorBoard
            if writer:
                writer.add_scalar("Episode/Total_Reward", episode_reward, step)
                writer.add_scalar("Episode/Average_Secondary_Throughput", info["throughput_reward"], step)
                writer.add_scalar("Episode/Primary_Throughput", info["primary_throughput"], step)
                writer.add_scalar("Episode/BER", info["ber"], step)
                writer.add_scalar("Episode/Outage_Rate", info["outage"], step)
                writer.add_scalar("Episode/Average_Power_Watts", info["average_power"], step)
                
            print(f"Step: {step} | Ep Reward: {episode_reward:.2f} | SU Throughput: {info['throughput_reward']:.3f} | PU Throughput: {info['primary_throughput']:.3f} | Power: {info['average_power']:.4f} W")
            
            # Reset Env
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0

        # Periodic Evaluation
        if step % eval_interval == 0:
            eval_env = OverlayCRNEnv(config)
            eval_metrics = evaluate_policy(agent, eval_env, episodes=eval_episodes)
            print(f"--- EVALUATION @ Step {step} ---")
            print(f"Avg Ep Reward: {eval_metrics['total_reward']:.2f}")
            print(f"Avg SU Throughput: {eval_metrics['throughput_s']:.3f} bps/Hz")
            print(f"Avg PU Throughput: {eval_metrics['throughput_p']:.3f} bps/Hz")
            print(f"Avg Outage Rate: {eval_metrics['outage']:.4f}")
            print(f"Avg Power Consumed: {eval_metrics['average_power']:.4f} W")
            print(f"Avg Relay Success: {eval_metrics.get('relay_success', 0.0):.4f}")
            print(f"Avg QoS Satisfaction: {eval_metrics.get('qos_satisfaction', 0.0):.4f}")
            print(f"Avg Constraint Satisfaction: {eval_metrics.get('constraint_satisfaction', 0.0):.4f}")

            if writer:
                writer.add_scalar("Eval/Average_Total_Reward", eval_metrics["total_reward"], step)
                writer.add_scalar("Eval/Average_Secondary_Throughput", eval_metrics["throughput_s"], step)
                writer.add_scalar("Eval/Average_Primary_Throughput", eval_metrics["throughput_p"], step)
                writer.add_scalar("Eval/Outage_Rate", eval_metrics["outage"], step)
                writer.add_scalar("Eval/Average_Power_Watts", eval_metrics["average_power"], step)
                writer.add_scalar("Eval/Relay_Success_Rate", eval_metrics.get("relay_success", 0.0), step)
                writer.add_scalar("Eval/QoS_Satisfaction_Rate", eval_metrics.get("qos_satisfaction", 0.0), step)
                writer.add_scalar("Eval/Constraint_Satisfaction_Rate", eval_metrics.get("constraint_satisfaction", 0.0), step)

            # Checkpoint: Save best model based on evaluation throughput & rewards
            if eval_metrics["total_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["total_reward"]
                agent.save(best_checkpoint_path)

    # Save final model state
    agent.save(final_checkpoint_path)
    if writer:
        writer.close()
    print("Training pipeline finished successfully.")


def main():
    import sys
    if len(sys.argv) == 1:
        # Run legacy training flow
        main_legacy()
    else:
        # Run CLI flow
        from cli.parser import get_parser
        from cli.runner import execute_cli
        parser = get_parser()
        args = parser.parse_args()
        execute_cli(args)


if __name__ == "__main__":
    main()
