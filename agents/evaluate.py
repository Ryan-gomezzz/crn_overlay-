"""
Evaluation Script.
Assignee: Aditya
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from agents.train_td3 import TD3Agent
from envs.crn_env import OverlayCRNEnv
from main import evaluate_policy, set_seed


def main():
    print("Initializing standalone evaluation...")

    # Load configuration
    config_path = os.path.join("configs", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set compute device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Set random seeds
    sim_cfg = config.get("simulation", {})
    seed = sim_cfg.get("seed", 42)
    set_seed(seed)

    # Setup environment
    env = OverlayCRNEnv(config)
    env.action_space.seed(seed)

    # Setup agent
    agent = TD3Agent(config, device=device)

    # Load checkpoint
    eval_cfg = config.get("evaluation", {})
    checkpoint_dir = eval_cfg.get("save_dir", "experiments/checkpoints/")
    algo_name = config.get("algorithm", {}).get("name", "TD3")

    checkpoint_path = os.path.join(checkpoint_dir, f"{algo_name}_best_model.pth")
    if not os.path.exists(checkpoint_path):
        # Fallback to final model
        checkpoint_path = os.path.join(checkpoint_dir, f"{algo_name}_final_model.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint model found at: {checkpoint_dir}")

    agent.load(checkpoint_path)

    # Run deterministic evaluation
    eval_episodes = eval_cfg.get("eval_episodes", 20)
    print(f"Evaluating {algo_name} policy over {eval_episodes} episodes...")

    metrics = evaluate_policy(agent, env, episodes=eval_episodes)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Algorithm:           {algo_name}")
    print(f"Checkpoint Loaded:   {checkpoint_path}")
    print(f"Average Reward:      {metrics['total_reward']:.2f}")
    print(f"Average SU Rate:     {metrics['throughput_s']:.3f} bps/Hz")
    print(f"Average PU Rate:     {metrics['throughput_p']:.3f} bps/Hz")
    print(f"Average Outage Rate: {metrics['outage']:.4f}")
    print(f"Average BER:         {metrics['ber']:.4f}")
    print(f"Average Power:       {metrics['average_power']:.4f} W")
    if "relay_success" in metrics:
        print(f"Relay Success Rate:  {metrics['relay_success']:.4f}")
    if "qos_satisfaction" in metrics:
        print(f"QoS Satisfaction:    {metrics['qos_satisfaction']:.4f}")
    if "constraint_satisfaction" in metrics:
        print(f"Constraint Satisf.:  {metrics['constraint_satisfaction']:.4f}")
    print("====================================================")


if __name__ == "__main__":
    main()
