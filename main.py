"""
Entry point for the CRN-RL-Framework.
Author: Ryan
"""
import argparse
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(description="CRN-RL Framework")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to base configuration file.")
    parser.add_argument("--experiment", type=str, default=None, help="Path to experiment override configuration file.")
    parser.add_argument("--test", action="store_true", help="Run a quick environment sanity check.")
    parser.add_argument("--info", action="store_true", help="Print system information and exit.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


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


def main():
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


if __name__ == "__main__":
    main()
