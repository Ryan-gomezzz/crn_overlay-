"""
Entry point for the CRN-RL-Framework.
"""

import sys
import os
import random
import yaml
import numpy as np
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CRN-RL Framework - Legacy mode")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to base configuration file.")
    parser.add_argument("--experiment", type=str, default=None, help="Path to experiment override configuration file.")
    parser.add_argument("--test", action="store_true", help="Run a quick environment sanity check.")
    parser.add_argument("--info", action="store_true", help="Print system information and exit.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    # Add a fallback for unknown args to not break Aditya's CLI
    args, unknown = parser.parse_known_args()
    return args


def run_sanity_check():
    """Run a quick environment sanity check."""
    from envs.crn_env import make_crn_env
    print("Running sanity check...")
    env = make_crn_env()
    obs, info = env.reset(seed=42)
    print(f"Initial Observation: {obs}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action: {action}, Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")
        env.render()
        
    print("Sanity check completed successfully.")


def print_system_info():
    """Print dependency versions and system info."""
    print("System Information:")
    print(f"Python version: {sys.version}")
    
    try:
        import numpy
        print(f"NumPy version: {numpy.__version__}")
    except ImportError:
        print("NumPy is not installed.")
        
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch is not installed.")
        
    try:
        import gymnasium
        print(f"Gymnasium version: {gymnasium.__version__}")
    except ImportError:
        print("Gymnasium is not installed.")


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_policy(agent, env, episodes: int = 5) -> dict:
    """
    Evaluate agent policy deterministically over multiple episodes.
    """
    eval_metrics = {
        "throughput_s": [],
        "throughput_p": [],
        "outage": [],
        "su_outage": [],
        "ber": [],
        "average_power": [],
        "total_reward": [],
        "relay_success": [],
        "qos_satisfaction": [],
        "constraint_satisfaction": [],
    }

    energy_limit = env.energy_limit if hasattr(env, 'energy_limit') else 1.0

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        
        ep_throughput_s = []
        ep_throughput_p = []
        ep_outage = []
        ep_su_outage = []
        ep_ber = []
        ep_average_power = []
        ep_relay_success = []
        ep_qos_satisfaction = []
        ep_constraint_satisfaction = []

        while not (done or truncated):
            action = agent.select_action(obs, info, explore=False)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            ep_throughput_s.append(info.get("throughput_reward", 0))
            ep_throughput_p.append(info.get("primary_throughput", 0))
            ep_outage.append(info.get("outage", 0))
            ep_su_outage.append(info.get("su_outage", 0))
            ep_ber.append(info.get("ber", 0))
            ep_average_power.append(info.get("average_power", 0))

            dec = info.get("relay_decoded", 0.0)
            ep_relay_success.append(dec)

            qos_sat = 1.0 if info.get("outage", 0) == 0.0 else 0.0
            ep_qos_satisfaction.append(qos_sat)

            con_sat = 1.0 if (info.get("outage", 0) == 0.0 and info.get("average_power", 0) <= energy_limit) else 0.0
            ep_constraint_satisfaction.append(con_sat)

        eval_metrics["throughput_s"].append(np.mean(ep_throughput_s))
        eval_metrics["throughput_p"].append(np.mean(ep_throughput_p))
        eval_metrics["outage"].append(np.mean(ep_outage))
        eval_metrics["su_outage"].append(np.mean(ep_su_outage))
        eval_metrics["ber"].append(np.mean(ep_ber))
        eval_metrics["average_power"].append(np.mean(ep_average_power))
        eval_metrics["relay_success"].append(np.mean(ep_relay_success))
        eval_metrics["qos_satisfaction"].append(np.mean(ep_qos_satisfaction))
        eval_metrics["constraint_satisfaction"].append(np.mean(ep_constraint_satisfaction))
        eval_metrics["total_reward"].append(episode_reward)

    avg_metrics = {k: float(np.mean(v)) for k, v in eval_metrics.items()}
    return avg_metrics


def main_legacy():
    """
    Legacy run flow.
    """
    args = parse_args()

    # Ensure the root path is in sys.path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        
    if args.info:
        print_system_info()
    elif args.test:
        run_sanity_check()
    else:
        from experiments.pipeline import run_experiment
        run_experiment(args.config, args.experiment)


def main():
    # If standard CLI arguments are used that belong to Aditya's CLI, route to it.
    # Otherwise run legacy logic.
    if len(sys.argv) > 1 and sys.argv[1] in ["train", "eval", "benchmark", "hyperopt"]:
        # Run Aditya's CLI flow
        from cli.parser import get_parser
        from cli.runner import execute_cli
        parser = get_parser()
        args = parser.parse_args()
        execute_cli(args)
    else:
        main_legacy()


if __name__ == "__main__":
    main()
