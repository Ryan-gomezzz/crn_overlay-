"""
Unified Test Runner for CRN Research Framework (Unit, Smoke, Config, Buffer).
"""

import os
import yaml
import pytest
import numpy as np
import torch
from envs.crn_env import OverlayCRNEnv
from agents.train_td3 import TD3Agent
from main import set_seed

class DummyWriter:
    """Mock TensorBoard SummaryWriter to prevent crash when TB is disabled."""
    def add_scalar(self, name, value, step):
        pass
    def close(self):
        pass

def validate_yaml_config(config_path: str = "configs/config.yaml") -> tuple[bool, list[str]]:
    """Validate config structure and required parameter keys."""
    errors = []
    if not os.path.exists(config_path):
        return False, [f"Configuration file not found at: {config_path}"]

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return False, [f"YAML Parsing Error: {e}"]

    # Required sections and keys
    required_keys = {
        "algorithm": ["name"],
        "simulation": ["seed", "time_steps_per_episode"],
        "training": ["gamma", "tau", "policy_delay", "exploration_noise", "batch_size", "lr_actor", "lr_critic", "buffer_size", "total_steps", "start_steps"],
        "camo_td3": ["history_length", "pu_rate_threshold", "interference_limit_dbm", "energy_limit_watts"],
        "logging": ["log_dir"],
        "evaluation": ["eval_episodes", "eval_interval", "save_dir"]
    }

    for section, keys in required_keys.items():
        if section not in config:
            errors.append(f"Missing required config section: '{section}'")
            continue
        for key in keys:
            if key not in config[section]:
                errors.append(f"Missing key '{key}' in section '{section}'")

    # Value constraints
    if "algorithm" in config and "name" in config["algorithm"]:
        valid_algos = ["TD3", "CAMO_TD3", "OVERLAY_CAMO_TD3"]
        if config["algorithm"]["name"] not in valid_algos:
            errors.append(f"Invalid algorithm name: '{config['algorithm']['name']}'. Choose from: {valid_algos}")

    return len(errors) == 0, errors

def run_agent_smoke_test(algo_name: str) -> bool:
    """Run a fast 5-step training smoke test for the given algorithm."""
    try:
        # Load standard config
        config_path = "configs/config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Override to fast training parameters
        config["algorithm"]["name"] = algo_name
        config["training"]["total_steps"] = 5
        config["training"]["start_steps"] = 2
        config["training"]["batch_size"] = 2
        config["simulation"]["time_steps_per_episode"] = 10
        
        set_seed(42)
        env = OverlayCRNEnv(config)
        device = "cpu" # Smoke test on CPU for speed and reliability
        agent = TD3Agent(config, device=device)
        
        obs, info = env.reset()
        
        # Run 5 steps of environment interaction and train
        for step in range(1, 6):
            action = agent.select_action(obs, info, explore=True)
            next_obs, reward, done, truncated, next_info = env.step(action)
            
            agent.replay_buffer.add(obs, action, reward, next_obs, done or truncated, info)
            
            obs = next_obs
            info = next_info
            
            if step >= config["training"]["start_steps"]:
                # Train with dummy tensorboard writer to prevent crash
                agent.train(writer=DummyWriter())
                
            if done or truncated:
                obs, info = env.reset()
                
        # Test save checkpoint
        temp_ckpt = f"experiments/checkpoints/smoke_{algo_name}_temp.pth"
        agent.save(temp_ckpt)
        
        # Test load checkpoint
        agent.load(temp_ckpt)
        
        # Cleanup
        if os.path.exists(temp_ckpt):
            os.remove(temp_ckpt)
            
        return True
    except Exception as e:
        print(f"Smoke test failed for {algo_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests() -> int:
    """Run config validation, unit tests, and smoke tests. Returns exit code."""
    print("=" * 60)
    print("           CRN FRAMEWORK UNIFIED TESTING PIPELINE")
    print("=" * 60)
    
    # 1. Config Validation
    print("\n[1/3] Running Configuration Validation...")
    ok, errors = validate_yaml_config()
    if ok:
        print("  Config validation passed successfully!")
    else:
        print("  Config validation FAILED with following errors:")
        for err in errors:
            print(f"    - {err}")
        return 1

    # 2. Replay Buffer & Unit Tests
    print("\n[2/3] Executing Unit Tests (via Pytest)...")
    pytest_exit_code = pytest.main(["-v", "tests"])
    if pytest_exit_code == 0:
        print("  All unit tests passed!")
    else:
        print(f"  Unit tests FAILED with exit code: {pytest_exit_code}")
        return pytest_exit_code

    # 3. Agent Smoke Tests
    print("\n[3/3] Running Agent Smoke Tests (5-step training runs)...")
    smoke_algos = ["TD3", "CAMO_TD3", "OVERLAY_CAMO_TD3"]
    smoke_results = {}
    for algo in smoke_algos:
        print(f"  Testing {algo}...")
        smoke_results[algo] = run_agent_smoke_test(algo)
        
    print("\n================ TEST SUMMARY ================")
    print(f"Configuration Check:  {'PASSED' if ok else 'FAILED'}")
    print(f"Pytest Unit Tests:    {'PASSED' if pytest_exit_code == 0 else 'FAILED'}")
    for algo, res in smoke_results.items():
        print(f"Smoke Test ({algo}):  {'PASSED' if res else 'FAILED'}")
    print("==============================================")
    
    # Return 0 only if all tests pass
    if ok and (pytest_exit_code == 0) and all(smoke_results.values()):
        return 0
    return 1
