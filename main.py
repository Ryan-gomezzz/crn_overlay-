"""
Main training and evaluation loop.
"""

import os
import torch
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.train_td3 import TD3Agent
from envs.crn_env import OverlayCRNEnv


def evaluate_policy(agent, env, episodes=10):
    """Evaluate the agent's policy.

    Args:
        agent: The agent to evaluate
        env: The environment to evaluate in
        episodes: Number of evaluation episodes

    Returns:
        dict: Evaluation metrics
    """
    total_rewards = []
    total_throughputs_s = []
    total_throughputs_p = []
    total_bers = []
    total_outages = []
    total_powers = []
    total_relay_successes = []
    total_qos_satisfaction = []
    total_constraint_satisfaction = []

    for _ in range(episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_throughput_s = 0.0
        episode_throughput_p = 0.0
        episode_ber = 0.0
        episode_outage = 0.0
        episode_power = 0.0
        episode_relay_success = 0.0
        episode_qos_sat = 0.0
        episode_con_sat = 0.0
        energy_limit = env.energy_limit

        done = False
        while not done:
            action = agent.select_action(
                obs, info, explore=False
            )
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            episode_throughput_s += info.get("throughput_s", 0.0)
            episode_throughput_p += info.get("throughput_p", 0.0)
            episode_ber += info.get("ber", 0.0)
            episode_outage += info.get("outage", 0.0)
            episode_power += info.get("average_power", 0.0)
            episode_relay_success += info.get("relay_success", 0.0)

            qos_sat = (
                1.0
                if info.get("throughput_p", 0.0) >= 1.5
                else 0.0
            )
            episode_qos_sat += qos_sat

            con_sat = (
                1.0
                if (info["outage"] == 0.0
                    and info["average_power"] <= energy_limit)
                else 0.0
            )
            episode_con_sat += con_sat

        avg_steps = len(info.get("history", {}).get("steps", []))
        if avg_steps == 0:
            avg_steps = 1

        total_rewards.append(episode_reward)
        total_throughputs_s.append(episode_throughput_s / avg_steps)
        total_throughputs_p.append(episode_throughput_p / avg_steps)
        total_bers.append(episode_ber / avg_steps)
        total_outages.append(episode_outage / avg_steps)
        total_powers.append(episode_power / avg_steps)
        total_relay_successes.append(episode_relay_success / avg_steps)
        total_qos_satisfaction.append(episode_qos_sat / avg_steps)
        total_constraint_satisfaction.append(episode_con_sat / avg_steps)

    return {
        "total_reward": np.mean(total_rewards),
        "throughput_s": np.mean(total_throughputs_s),
        "throughput_p": np.mean(total_throughputs_p),
        "ber": np.mean(total_bers),
        "outage": np.mean(total_outages),
        "average_power": np.mean(total_powers),
        "relay_success": np.mean(total_relay_successes),
        "qos_satisfaction": np.mean(total_qos_satisfaction),
        "constraint_satisfaction": np.mean(
            total_constraint_satisfaction
        ),
    }


def setup_config(config_path, overrides=None):
    """Setup configuration from file and overrides.

    Args:
        config_path: Path to YAML config file
        overrides: Dict of config overrides

    Returns:
        dict: Merged configuration
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if overrides:
        # Basic recursive merge for single-level dictionary keys
        for section, vals in overrides.items():
            if (
                section in config
                and isinstance(config[section], dict)
                and isinstance(vals, dict)
            ):
                config[section].update(vals)
            else:
                config[section] = vals

    return config


def train(
    config_path="configs/config.yaml",
    log_dir="experiments/logs",
    checkpoint_dir="experiments/checkpoints",
    overrides=None,
):
    """Main training loop.

    Args:
        config_path: Path to config file
        log_dir: Directory for logs
        checkpoint_dir: Directory for checkpoints
        overrides: Config overrides
    """
    # Setup config
    config = setup_config(config_path, overrides)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create environment
    env = OverlayCRNEnv(config)
    print(f"Environment created: {env}")

    # Create agent
    agent = TD3Agent(config, device=device)
    print(f"Agent created: {agent.algorithm_name}")

    # Create TensorBoard writer
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Setup checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    algo_name = config.get("algorithm", {}).get("name", "TD3")

    # Training parameters
    max_steps = config.get(
        "training", {}
    ).get("max_training_steps", 100000)
    eval_interval = config.get(
        "training", {}
    ).get("eval_interval", 1000)
    eval_episodes = config.get(
        "evaluation", {}
    ).get("eval_episodes", 5)
    save_interval = config.get(
        "training", {}
    ).get("save_interval", 5000)

    # Main training loop
    obs, info = env.reset()
    step = 0
    best_eval_reward = -float("inf")

    while step < max_steps:
        # Select action with exploration
        action = agent.select_action(obs, info, explore=True)

        # Environment step
        next_obs, reward, done, _, info = env.step(action)

        # Store transition
        agent.replay_buffer.add(
            obs,
            action,
            reward,
            next_obs,
            done,
            info,
        )

        # Train agent
        if (
            agent.replay_buffer.size()
            >= config.get("training", {}).get("batch_size", 256)
        ):
            metrics = agent.train(writer)

        # Log metrics
        if step % 100 == 0:
            writer.add_scalar(
                "Episode/Total_Reward", reward, step
            )
            writer.add_scalar(
                "Episode/Average_Secondary_Throughput",
                info.get("throughput_reward", 0.0),
                step,
            )
            writer.add_scalar(
                "Episode/Primary_Throughput",
                info.get("primary_throughput", 0.0),
                step,
            )
            writer.add_scalar("Episode/BER", info.get("ber", 0.0), step)
            writer.add_scalar(
                "Episode/Outage_Rate", info.get("outage", 0.0), step
            )
            writer.add_scalar(
                "Episode/Average_Power_Watts",
                info.get("average_power", 0.0),
                step,
            )

            print(
                f"Step: {step} | "
                f"Ep Reward: {reward:.2f} | "
                f"SU Throughput: "
                f"{info.get('throughput_reward', 0.0):.3f} | "
                f"PU Throughput: "
                f"{info.get('primary_throughput', 0.0):.3f} | "
                f"Power: {info.get('average_power', 0.0):.4f} W"
            )

        # Evaluation
        if step % eval_interval == 0 and step > 0:
            print(f"\nEvaluating at step {step}...")
            eval_metrics = evaluate_policy(
                agent, env, episodes=eval_episodes
            )
            print(f"Avg Reward: {eval_metrics['total_reward']:.2f}")
            print(
                f"Avg SU Throughput: "
                f"{eval_metrics['throughput_s']:.3f}"
            )
            print(
                f"Avg PU Throughput: "
                f"{eval_metrics['throughput_p']:.3f}"
            )
            print(f"Avg BER: {eval_metrics['ber']:.4f}")
            print(f"Avg Outage: {eval_metrics['outage']:.4f}")
            print(
                f"Avg Power Consumed: "
                f"{eval_metrics['average_power']:.4f} W"
            )
            print(
                f"Avg Relay Success: "
                f"{eval_metrics.get('relay_success', 0.0):.4f}"
            )
            print(
                f"Avg QoS Satisfaction: "
                f"{eval_metrics.get('qos_satisfaction', 0.0):.4f}"
            )
            print(
                f"Avg Constraint Satisfaction: "
                f"{eval_metrics.get('constraint_satisfaction', 0.0):.4f}"
            )

            if writer:
                writer.add_scalar(
                    "Eval/Average_Total_Reward",
                    eval_metrics["total_reward"],
                    step,
                )
                writer.add_scalar(
                    "Eval/Average_Secondary_Throughput",
                    eval_metrics["throughput_s"],
                    step,
                )
                writer.add_scalar(
                    "Eval/Average_Primary_Throughput",
                    eval_metrics["throughput_p"],
                    step,
                )
                writer.add_scalar(
                    "Eval/Outage_Rate",
                    eval_metrics["outage"],
                    step,
                )
                writer.add_scalar(
                    "Eval/Average_Power_Watts",
                    eval_metrics["average_power"],
                    step,
                )
                writer.add_scalar(
                    "Eval/Relay_Success_Rate",
                    eval_metrics.get("relay_success", 0.0),
                    step,
                )
                writer.add_scalar(
                    "Eval/QoS_Satisfaction_Rate",
                    eval_metrics.get("qos_satisfaction", 0.0),
                    step,
                )
                writer.add_scalar(
                    "Eval/Constraint_Satisfaction_Rate",
                    eval_metrics.get(
                        "constraint_satisfaction", 0.0
                    ),
                    step,
                )

            # Checkpoint: Save best model
            if eval_metrics["total_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["total_reward"]
                agent.save(
                    os.path.join(
                        checkpoint_dir, f"{algo_name}_best_model.pth"
                    )
                )
                print(f"Best model saved at step {step}")

        # Periodic checkpoint
        if step % save_interval == 0 and step > 0:
            agent.save(
                os.path.join(
                    checkpoint_dir, f"{algo_name}_final_model.pth"
                )
            )

        # Episode reset
        if done:
            obs, info = env.reset()
        else:
            obs = next_obs

        step += 1

    print("Training completed!")
    writer.close()


if __name__ == "__main__":
    train()
