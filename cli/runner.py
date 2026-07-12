"""
Command Execution Runner for the CRN Research Framework.
"""

import os
import sys
import time
import json
import yaml
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

from envs.crn_env import OverlayCRNEnv
from envs.multi_agent_crn_env import make_ma_crn_env
from envs.flat_noma_env import make_flat_noma_env
from agents.train_td3 import TD3Agent
from agents.matd3 import MATD3Agent
from agents.cent_noma_td3 import CentNOMATD3Agent
from cli.logger import ProgressLogger, print_header, print_footer
from cli.parser import AGENT_MAP, REVERSE_AGENT_MAP, VALID_SEEDS
from cli.report_generator import generate_comparison_plots, generate_markdown_report, generate_pdf_report
from cli.test_runner import run_all_tests

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def apply_overrides(config: Dict[str, Any], args: Any, agent_name: str) -> Dict[str, Any]:
    """Applies command-line overrides to config in-memory."""
    # Ensure config has proper sub-dictionaries
    if "algorithm" not in config: config["algorithm"] = {}
    if "simulation" not in config: config["simulation"] = {}
    if "training" not in config: config["training"] = {}
    if "logging" not in config: config["logging"] = {}
    if "evaluation" not in config: config["evaluation"] = {}
    if "camo_td3" not in config: config["camo_td3"] = {}

    config["algorithm"]["name"] = agent_name
    
    # Simulation overrides
    if hasattr(args, "seed") and args.seed is not None:
        config["simulation"]["seed"] = args.seed
    if hasattr(args, "steps") and args.steps is not None:
        config["simulation"]["time_steps_per_episode"] = args.steps

    # Training overrides
    if hasattr(args, "batch_size") and args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if hasattr(args, "lr") and args.lr is not None:
        config["training"]["lr_actor"] = args.lr
        config["training"]["lr_critic"] = args.lr

    # Logging overrides
    if hasattr(args, "tensorboard") and args.tensorboard is not None:
        config["logging"]["tensorboard_enabled"] = args.tensorboard
    if hasattr(args, "wandb") and args.wandb is not None:
        config["logging"]["wandb_enabled"] = args.wandb

    # Evaluation overrides
    if hasattr(args, "episodes") and args.episodes is not None and args.command in ("train", "benchmark", "resume"):
        # For training, total_steps is episodes * steps
        steps_per_ep = config["simulation"].get("time_steps_per_episode", 500)
        config["training"]["total_steps"] = args.episodes * steps_per_ep
    
    return config

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_policy(agent, env, episodes: int = 5) -> dict:
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

class HybridWriter:
    """Wrapper that logs to TensorBoard and Weights & Biases simultaneously."""
    def __init__(self, log_dir: str, tb_enabled: bool, wandb_enabled: bool, run_name: str, config: dict):
        self.tb_writer = None
        self.wandb_enabled = wandb_enabled

        if tb_enabled:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=log_dir)

        if wandb_enabled:
            import wandb
            # Safely get project name, default to crn_overlay
            project_name = config.get("logging", {}).get("wandb_project", "crn_overlay")
            wandb.init(project=project_name, name=run_name, config=config, dir=os.path.dirname(log_dir), reinit=True)

    def add_scalar(self, name, value, step):
        if self.tb_writer:
            try:
                self.tb_writer.add_scalar(name, value, step)
            except Exception:
                pass
        if self.wandb_enabled:
            import wandb
            wandb.log({name: value}, step=step)

    def close(self):
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_enabled:
            import wandb
            wandb.finish()

def run_single_train(
    agent_name: str, 
    args: Any, 
    seed: int, 
    resume_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """Run training for a single agent with a specific seed."""
    # Load base configuration
    config_path = os.path.join("configs", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    config = apply_overrides(config, args, agent_name)
    config["simulation"]["seed"] = seed
    
    set_seed(seed)
    
    # Instantiate environment & agent
    if agent_name == "MATD3":
        env = make_ma_crn_env(config_path)  # Simplified config loading
        env.action_space.seed(seed)
        device = args.device if (hasattr(args, "device") and args.device) else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nTraining {agent_name} | Seed {seed} | Device: {device}")
        agent = MATD3Agent(config, device=device)
    elif agent_name == "CENT_NOMA_TD3":
        env = make_flat_noma_env(config_path)
        env.action_space.seed(seed)
        device = args.device if (hasattr(args, "device") and args.device) else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nTraining {agent_name} | Seed {seed} | Device: {device}")
        agent = CentNOMATD3Agent(config, device=device)
    else:
        env = OverlayCRNEnv(config)
        env.action_space.seed(seed)
        device = args.device if (hasattr(args, "device") and args.device) else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nTraining {agent_name} | Seed {seed} | Device: {device}")
        agent = TD3Agent(config, device=device)
    
    # Setup Output Run Directory
    timestamp = get_timestamp()
    agent_folder = agent_name.lower()
    run_name = f"run_{timestamp}_seed_{seed}"
    output_dir = getattr(args, "output-dir", args.output_dir if hasattr(args, "output_dir") else "experiments")
    run_dir = os.path.join(output_dir, agent_folder, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Logging text file
    log_file_path = os.path.join(run_dir, "train.log")
    log_file = open(log_file_path, "w")
    
    def log_print(msg: str):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log_print(f"Run directory: {run_dir}")
    
    # Save config snapshot
    with open(os.path.join(run_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f)
        
    # Logging Integrations (TensorBoard & WandB)
    tb_enabled = config["logging"].get("tensorboard_enabled", True)
    wandb_enabled = config["logging"].get("wandb_enabled", False)
    
    tb_dir = os.path.join(run_dir, "tensorboard")
    writer = HybridWriter(log_dir=tb_dir, tb_enabled=tb_enabled, wandb_enabled=wandb_enabled, run_name=run_name, config=config)
    
    if tb_enabled:
        log_print(f"TensorBoard logging enabled: {tb_dir}")
    if wandb_enabled:
        log_print(f"Weights & Biases logging enabled.")

    # Checkpoint paths
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    final_path = os.path.join(ckpt_dir, "final_model.pth")
    
    # Resume handler
    start_episode = 1
    global_step = 0
    best_eval_reward = -np.inf
    
    if resume_checkpoint:
        log_print(f"Resuming training from checkpoint: {resume_checkpoint}")
        agent.load(resume_checkpoint)
        # Attempt to load replay buffer
        replay_path = resume_checkpoint.replace(".pth", "_replay.pkl")
        if os.path.exists(replay_path):
            from cli.runner import load_replay_buffer
            load_replay_buffer(agent.replay_buffer, replay_path)
        
        # Recover steps & episodes from total_it
        global_step = agent.total_it
        steps_per_ep = config["simulation"].get("time_steps_per_episode", 500)
        start_episode = int(global_step // steps_per_ep) + 1
        log_print(f"Recovered state: global_step={global_step}, start_episode={start_episode}")

    # Training settings
    episodes = getattr(args, "episodes", 2000)
    steps_per_episode = config["simulation"].get("time_steps_per_episode", 500)
    start_steps = config["training"].get("start_steps", 1000)
    eval_interval = config["evaluation"].get("eval_interval", 500)
    eval_episodes = config["evaluation"].get("eval_episodes", 5)
    checkpoint_every = getattr(args, "checkpoint_every", None)
    
    # Timing
    t_start = time.time()
    
    progress = ProgressLogger(agent_name, episodes)
    
    episode_reward = 0.0
    obs, info = env.reset()
    
    import math
    from scipy.special import erfc
    
    # Store evaluation metrics history
    history = {
        "episodes": [],
        "rewards": [],
        "throughput_s": [],
        "outage": [],
        "su_outage": [],
        "ber": [],
        "sinr_db_pts": [],
        "ber_pts": [],
        "pu_sinr_db_pts": [],
        "pu_ber_pts": [],
        "pu_throughput": []
    }
    
    scatter_budget = max(1, 12000 // max(episodes, 1))
    
    for ep in range(start_episode, episodes + 1):
        obs, info = env.reset()
        episode_reward = 0.0
        ep_pu_throughput = []
        
        # Sample steps for scatter plots
        import random
        sample_steps = set(random.sample(range(1, steps_per_episode + 1), min(scatter_budget, steps_per_episode)))
        
        for step in range(1, steps_per_episode + 1):
            global_step += 1
            
            # Action selection
            if global_step < start_steps and not resume_checkpoint:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, info, explore=True)
                
            next_obs, reward, done, truncated, next_info = env.step(action)
            episode_reward += reward
            
            # Log scatter metrics for PDF generation
            sinr_su = next_info.get("sinr_su", 0.0)
            sinr_pu = next_info.get("sinr_pu", 0.0)
            pu_tput = next_info.get("pu_throughput", 0.0)
            ep_pu_throughput.append(pu_tput)
            
            if step in sample_steps:
                sinr_s_db = 10.0 * math.log10(max(1e-9, sinr_su))
                ber_s = float(0.5 * erfc(math.sqrt(max(0.0, sinr_su))))
                sinr_p_db = 10.0 * math.log10(max(1e-9, sinr_pu))
                ber_p = float(0.5 * erfc(math.sqrt(max(0.0, sinr_pu))))
                
                history["sinr_db_pts"].append(sinr_s_db)
                history["ber_pts"].append(ber_s)
                history["pu_sinr_db_pts"].append(sinr_p_db)
                history["pu_ber_pts"].append(ber_p)
            
            # Store transition
            agent.replay_buffer.add(obs, action, reward, next_obs, done or truncated, info)
            
            obs = next_obs
            info = next_info
            
            # Gradient update step
            if global_step >= start_steps:
                agent.train(writer)
                
            # Periodic evaluation
            if global_step % eval_interval == 0:
                if agent_name == "MATD3":
                    eval_env = make_ma_crn_env(config_path)
                elif agent_name == "CENT_NOMA_TD3":
                    eval_env = make_flat_noma_env(config_path)
                else:
                    eval_env = OverlayCRNEnv(config)
                eval_metrics = evaluate_policy(agent, eval_env, episodes=eval_episodes)
                
                history["episodes"].append(ep)
                history["rewards"].append(eval_metrics["total_reward"])
                history["throughput_s"].append(eval_metrics["throughput_s"])
                history["outage"].append(eval_metrics["outage"])
                history["su_outage"].append(eval_metrics.get("su_outage", 0.0))
                history["ber"].append(eval_metrics["ber"])
                history["pu_throughput"].append(sum(ep_pu_throughput) / len(ep_pu_throughput) if ep_pu_throughput else 0.0)
                
                log_print(
                    f"\n[EVAL @ Step {global_step}] "
                    f"Avg Reward: {eval_metrics['total_reward']:.2f} | "
                    f"SU Thr: {eval_metrics['throughput_s']:.3f} bps/Hz | "
                    f"PU Outage: {eval_metrics['outage']:.4f}"
                )
                
                if writer:
                    writer.add_scalar("Eval/Average_Total_Reward", eval_metrics["total_reward"], global_step)
                    writer.add_scalar("Eval/Average_Secondary_Throughput", eval_metrics["throughput_s"], global_step)
                    writer.add_scalar("Eval/Outage_Rate", eval_metrics["outage"], global_step)
                    
                # Save best
                if getattr(args, "save_best", True) and (eval_metrics["total_reward"] > best_eval_reward):
                    best_eval_reward = eval_metrics["total_reward"]
                    agent.save(best_path)
                    # Also save replay buffer companion
                    save_replay_buffer(agent.replay_buffer, best_path.replace(".pth", "_replay.pkl"))
            
            if done or truncated:
                break
                
        # End of episode logging
        if writer:
            writer.add_scalar("Episode/Total_Reward", episode_reward, ep)
            writer.add_scalar("Episode/Average_Secondary_Throughput", info.get("throughput_reward", 0.0), ep)
            writer.add_scalar("Episode/Outage_Rate", info.get("outage", 0.0), ep)
            
        progress.update(
            episode=ep, 
            step=global_step, 
            reward=episode_reward, 
            throughput=info.get("throughput_reward", 0.0), 
            ber=info.get("ber", 0.0), 
            outage=info.get("outage", 0.0)
        )
        
        # Periodic checkpoint
        if checkpoint_every and ep % checkpoint_every == 0:
            p_path = os.path.join(ckpt_dir, f"checkpoint_ep_{ep}.pth")
            agent.save(p_path)
            save_replay_buffer(agent.replay_buffer, p_path.replace(".pth", "_replay.pkl"))
            log_print(f"\nPeriodic checkpoint saved at episode {ep}")
            
    progress.complete()
    t_end = time.time()
    train_duration = t_end - t_start
    
    # Save final
    if getattr(args, "save_final", True):
        agent.save(final_path)
        save_replay_buffer(agent.replay_buffer, final_path.replace(".pth", "_replay.pkl"))
        
    # Measure Inference Time over 100 steps
    obs, info = env.reset()
    t_inf_start = time.time()
    for _ in range(100):
        _ = agent.select_action(obs, info, explore=False)
    t_inf_end = time.time()
    avg_inf_time = (t_inf_end - t_inf_start) / 100.0
    
    # Load best model for final evaluation
    if os.path.exists(best_path):
        agent.load(best_path)
        
    if agent_name == "MATD3":
        eval_env = make_ma_crn_env(config_path)
    elif agent_name == "CENT_NOMA_TD3":
        eval_env = make_flat_noma_env(config_path)
    else:
        eval_env = OverlayCRNEnv(config)
        
    final_metrics = evaluate_policy(agent, eval_env, episodes=20)
    
    # Save metrics JSON
    metrics_data = {
        "agent": agent_name,
        "seed": seed,
        "train_time": train_duration,
        "inf_time": avg_inf_time,
        "eval_reward": final_metrics["total_reward"],
        "eval_su_throughput": final_metrics["throughput_s"],
        "eval_pu_outage": final_metrics["outage"],
        "eval_su_outage": final_metrics.get("su_outage", 0.0),
        "history": history
    }
    
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    # Also save a copy to centralized output checkpoints folder
    central_ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(central_ckpt_dir, exist_ok=True)
    latest_central_path = os.path.join(central_ckpt_dir, f"{agent_name}_latest.pth")
    agent.save(latest_central_path)
    save_replay_buffer(agent.replay_buffer, latest_central_path.replace(".pth", "_replay.pkl"))
    
    log_file.close()
    if writer:
        writer.close()
        
    return metrics_data

def save_replay_buffer(buffer: Any, filepath: str):
    """Serialize replay buffer transitions."""
    import pickle
    try:
        state = {
            "episodes": buffer.episodes,
            "current_episode": buffer.current_episode,
            "total_size": buffer.total_size,
            "capacity": buffer.capacity,
            "obs_dim": buffer.obs_dim,
            "action_dim": buffer.action_dim,
            "sequence_length": buffer.sequence_length
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
    except Exception as e:
        print(f"Warning: Failed to save replay buffer: {e}")

def load_replay_buffer(buffer: Any, filepath: str):
    """Deserialize replay buffer transitions."""
    import pickle
    try:
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        buffer.episodes = state["episodes"]
        buffer.current_episode = state["current_episode"]
        buffer.total_size = state["total_size"]
        print(f"Loaded {len(buffer)} transitions into replay buffer.")
    except Exception as e:
        print(f"Warning: Failed to load replay buffer: {e}")

# ==================== SUBCOMMAND HANDLERS ====================

def handle_train(args: Any):
    """Executes the train command."""
    print_header("RL Agent Training Pipeline")
    
    # Determine agents to train
    agents = []
    if args.agent:
        agents = [args.agent]
    elif args.agents:
        agents = args.agents
        
    seeds = VALID_SEEDS if args.all_seeds else [args.seed if args.seed is not None else 42]
    
    results = {}
    for agent in agents:
        results[agent] = []
        for seed in seeds:
            res = run_single_train(agent, args, seed)
            results[agent].append(res)
            
    # If all-seeds option, print stats
    if args.all_seeds:
        print_header("Reproducibility Results (All Seeds)")
        for agent, runs in results.items():
            rewards = [r["eval_reward"] for r in runs]
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            best_r = np.max(rewards)
            worst_r = np.min(rewards)
            
            print(f"\nAgent: {SHORT_NAMES_MAP.get(agent, agent)}")
            print(f"  Mean Return:  {mean_r:.2f}")
            print(f"  Std Return:   {std_r:.2f}")
            print(f"  Best Return:  {best_r:.2f}")
            print(f"  Worst Return: {worst_r:.2f}")
    print_footer()

def find_checkpoint(output_dir: str, agent_name: str) -> Optional[str]:
    """Find the checkpoint path matching the agent name with legacy fallbacks."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    
    # Standard names and legacy names mapping
    legacy_map = {
        "UNDERLAY_TD3": "CAMO_TD3",
        "OVERLAY_TD3": "OVERLAY_CAMO_TD3",
        "TD3": "TD3"
    }
    legacy_agent = legacy_map.get(agent_name, agent_name)
    
    # 1. Direct check in checkpoints directory
    for name in [agent_name, legacy_agent]:
        for suffix in ["latest.pth", "best_model.pth", "final_model.pth"]:
            path = os.path.join(ckpt_dir, f"{name}_{suffix}")
            if os.path.exists(path):
                return path
            path = os.path.join(ckpt_dir, f"{name.lower()}_{suffix}")
            if os.path.exists(path):
                return path

    # 2. Recursive search
    for root, _, files in os.walk(output_dir):
        for file in files:
            if not file.endswith(".pth"):
                continue
            
            low_file = file.lower()
            low_root = root.lower()
            full_path = os.path.join(root, file)
            
            # Verify the actual checkpoint contents
            try:
                ckpt = torch.load(full_path, map_location="cpu")
                algo = ckpt.get("algorithm", "")
                mapped_algo = legacy_map.get(algo, algo)
                if mapped_algo == agent_name:
                    return full_path
            except Exception:
                pass
            
            # Fallback if checkpoint metadata load fails (e.g. key missing)
            if agent_name == "TD3":
                if "td3" in low_file and not any(x in low_file or x in low_root for x in ["underlay", "overlay", "camo"]):
                    return full_path
            elif agent_name == "UNDERLAY_TD3":
                if ("underlay" in low_file or "camo" in low_file or "underlay" in low_root or "camo" in low_root) and "overlay" not in low_file and "overlay" not in low_root:
                    return full_path
            elif agent_name == "OVERLAY_TD3":
                if "overlay" in low_file or "overlay" in low_root:
                    return full_path
                    
    return None

def handle_evaluate(args: Any):
    """Executes evaluate command."""
    print_header(f"Deterministic Evaluation: {args.agent}")
    
    # Find model checkpoint
    checkpoint_path = find_checkpoint(args.output_dir, args.agent)
                
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found for {args.agent} in '{args.output_dir}'. Train the agent first.")
        sys.exit(1)
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load configuration
    config_path = os.path.join("configs", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config["algorithm"]["name"] = args.agent
    config["simulation"]["seed"] = args.seed
    
    set_seed(args.seed)
    
    if args.agent == "MATD3":
        env = make_ma_crn_env(config_path)
        env.action_space.seed(args.seed)
        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        agent = MATD3Agent(config, device=device)
    elif args.agent == "CENT_NOMA_TD3":
        env = make_flat_noma_env(config_path)
        env.action_space.seed(args.seed)
        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        agent = CentNOMATD3Agent(config, device=device)
    else:
        env = OverlayCRNEnv(config)
        env.action_space.seed(args.seed)
        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        agent = TD3Agent(config, device=device)
        
    agent.load(checkpoint_path)
    
    print(f"Evaluating policy over {args.episodes} episodes...")
    metrics = evaluate_policy(agent, env, episodes=args.episodes)
    
    print("\n" + "=" * 50)
    print(f"             EVALUATION REPORT: {SHORT_NAMES_MAP.get(args.agent, args.agent)}")
    print("=" * 50)
    print(f"Average Reward:      {metrics['total_reward']:.2f}")
    print(f"Average SU Rate:     {metrics['throughput_s']:.3f} bps/Hz")
    print(f"Average PU Rate:     {metrics['throughput_p']:.3f} bps/Hz")
    print(f"Average Outage Rate: {metrics['outage']:.4f}")
    print(f"Average BER:         {metrics['ber']:.4f}")
    print(f"Average Power:       {metrics['average_power']:.4f} W")
    print("=" * 50)
    print_footer()

def handle_benchmark(args: Any):
    """Executes benchmark command."""
    print_header("Multi-Agent Benchmarking Suite")
    
    agents = args.agents
    seeds = VALID_SEEDS if args.all_seeds else [args.seed if args.seed is not None else 42]
    
    print(f"Agents to benchmark: {[SHORT_NAMES_MAP.get(a, a) for a in agents]}")
    print(f"Seeds to test:       {seeds}")
    
    results = {}
    for agent in agents:
        results[agent] = []
        for seed in seeds:
            print(f"\nBenchmarking {SHORT_NAMES_MAP.get(agent, agent)} (Seed {seed})...")
            # Limit total benchmark steps if episodes not explicitly provided to avoid long wait
            res = run_single_train(agent, args, seed)
            results[agent].append(res)
            
    # Generate final summary plots & report
    output_dir = args.output_dir
    results_dir = os.path.join(os.getcwd(), "results")
    reports_dir = os.path.join(output_dir, "reports")
    
    print("\nGenerating benchmark plots...")
    generate_comparison_plots(output_dir, results_dir)
    print(f"Plots saved to: {results_dir}")
    
    print("Generating benchmark markdown report...")
    report_path = generate_markdown_report(output_dir, reports_dir)
    print(f"Markdown report saved to: {report_path}")
    
    # Summary table output
    print_header("Benchmark Summary Table")
    print("| Algorithm | Mean Return | SU Throughput (bps/Hz) | PU Outage | Train Time (s) |")
    print("|---|---|---|---|---|")
    for agent in agents:
        runs = results[agent]
        mean_ret = np.mean([r["eval_reward"] for r in runs])
        mean_su = np.mean([r["eval_su_throughput"] for r in runs])
        mean_pu = np.mean([r["eval_pu_outage"] for r in runs])
        mean_time = np.mean([r["train_time"] for r in runs])
        print(f"| {SHORT_NAMES_MAP.get(agent, agent)} | {mean_ret:.2f} | {mean_su:.4f} | {mean_pu:.4f} | {mean_time:.1f}s |")
    print_footer()

def handle_compare(args: Any):
    """Executes compare command."""
    print_header("Experiment Results Comparison")
    
    output_dir = args.output_dir
    agents = args.agents
    
    print("| Algorithm | Run Name | Final Eval Return | SU Throughput | PU Outage | Speed (Inf) |")
    print("|---|---|---|---|---|---|")
    
    from cli.report_generator import load_metrics_for_agent
    
    for agent in agents:
        runs = load_metrics_for_agent(output_dir, agent)
        if not runs:
            print(f"| {SHORT_NAMES_MAP.get(agent, agent)} | *No runs found* | - | - | - | - |")
            continue
        for r in runs:
            inf_ms = r.get("inf_time", 0.0) * 1000.0
            print(
                f"| {SHORT_NAMES_MAP.get(agent, agent)} | "
                f"{r.get('run_name', 'run')} | "
                f"{r.get('eval_reward', 0.0):.2f} | "
                f"{r.get('eval_su_throughput', 0.0):.4f} | "
                f"{r.get('eval_pu_outage', 0.0):.4f} | "
                f"{inf_ms:.3f} ms |"
            )
    print_footer()

def handle_plots(args: Any):
    """Executes plots command."""
    print_header("Generating Stored Experiment Figures")
    output_dir = args.output_dir
    plots_dir = os.path.join(output_dir, "plots")
    
    generate_comparison_plots(output_dir, plots_dir)
    print(f"All figures compiled successfully in: {plots_dir}")
    print_footer()

def handle_report(args: Any):
    """Executes report command."""
    print_header("Compiling Research Experiment Report")
    output_dir = args.output_dir
    plots_dir = os.path.join(output_dir, "plots")
    reports_dir = os.path.join(output_dir, "reports")
    
    # 1. Legacy Report
    generate_comparison_plots(output_dir, plots_dir)
    md_path = generate_markdown_report(output_dir, reports_dir)
    pdf_path = generate_pdf_report(output_dir, reports_dir)
    print(f"Generated Legacy Markdown report: {md_path}")
    print(f"Generated Legacy PDF report: {pdf_path}")
        
    # 2. NOMA Report
    noma_agents = ["MATD3", "CENT_NOMA_TD3"]
    generate_comparison_plots(output_dir, plots_dir, agents=noma_agents)
    md_path_noma = generate_markdown_report(output_dir, reports_dir, agents=noma_agents, prefix="noma_")
    pdf_path_noma = generate_pdf_report(output_dir, reports_dir, agents=noma_agents, prefix="noma_")
    print(f"Generated NOMA Markdown report: {md_path_noma}")
    print(f"Generated NOMA PDF report: {pdf_path_noma}")
    
    print_footer()

def handle_resume(args: Any):
    """Executes resume command."""
    print_header(f"Resuming Agent: {args.agent}")
    
    output_dir = args.output_dir
    agent_folder = args.agent.lower()
    legacy_agent_folder = args.agent.lower().replace("underlay_", "").replace("overlay_", "")
    
    # 1. Look for run directories of this agent
    checkpoint_path = None
    
    folder_map = {
        "TD3": ["td3"],
        "UNDERLAY_TD3": ["underlay_td3", "td3"],
        "OVERLAY_TD3": ["overlay_td3"]
    }
    folders = folder_map.get(args.agent, [args.agent.lower()])
    
    legacy_map = {
        "UNDERLAY_TD3": "CAMO_TD3",
        "OVERLAY_TD3": "OVERLAY_CAMO_TD3",
        "TD3": "TD3"
    }
    
    for folder in folders:
        agent_dir = os.path.join(output_dir, folder)
        if os.path.exists(agent_dir):
            runs = sorted(os.listdir(agent_dir))
            # Search backwards to find the latest run containing a checkpoints/final_model.pth or similar
            for run in reversed(runs):
                run_path = os.path.join(agent_dir, run)
                ckpt_dir = os.path.join(run_path, "checkpoints")
                if os.path.exists(ckpt_dir):
                    # Prefer final_model.pth or fallback to best_model.pth
                    for fname in ["final_model.pth", "best_model.pth"]:
                        path = os.path.join(ckpt_dir, fname)
                        if os.path.exists(path):
                            try:
                                ckpt = torch.load(path, map_location="cpu")
                                algo = ckpt.get("algorithm", "")
                                mapped_algo = legacy_map.get(algo, algo)
                                if mapped_algo == args.agent:
                                    checkpoint_path = path
                                    break
                            except Exception:
                                pass
                if checkpoint_path:
                    break
            if checkpoint_path:
                break
                    
    # 2. Fallback to centralized checkpoints dir
    if not checkpoint_path:
        central_path = os.path.join(output_dir, "checkpoints", f"{args.agent}_latest.pth")
        if os.path.exists(central_path):
            checkpoint_path = central_path
        else:
            legacy_map = {
                "UNDERLAY_TD3": "CAMO_TD3",
                "OVERLAY_TD3": "OVERLAY_CAMO_TD3"
            }
            legacy_agent = legacy_map.get(args.agent, args.agent)
            central_path = os.path.join(output_dir, "checkpoints", f"{legacy_agent}_latest.pth")
            if os.path.exists(central_path):
                checkpoint_path = central_path
            
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Error: Could not locate a valid checkpoint to resume training for {args.agent}.")
        sys.exit(1)
        
    print(f"Found latest checkpoint: {checkpoint_path}")
    
    # Run training loop starting from recovered step
    seed = args.seed if args.seed is not None else 42
    run_single_train(args.agent, args, seed, resume_checkpoint=checkpoint_path)
    print_footer()

def handle_test(args: Any):
    """Executes test command."""
    exit_code = run_all_tests()
    sys.exit(exit_code)

def handle_config(args: Any):
    """Executes config command."""
    print_header("Inspect Active Configuration Settings")
    config_path = os.path.join("configs", "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Error: Base configuration file missing: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    print(f"Active Algorithm (Default): {cfg.get('algorithm', {}).get('name', 'TD3')}")
    
    print("\n[Environment Settings]")
    sim = cfg.get("simulation", {})
    print(f"  Default Seed:            {sim.get('seed', 'None')}")
    print(f"  Time Steps Per Episode:  {sim.get('time_steps_per_episode', 'None')}")
    
    print("\n[Wireless Networks Settings]")
    net = cfg.get("network", {})
    print(f"  Primary Rx Coords:       {net.get('pr_coords')}")
    print(f"  SU Source Coords:        {net.get('sus_coords')}")
    print(f"  SU Dest Coords:          {net.get('sud_coords')}")
    print(f"  Max SU Power:            {net.get('p_max_su')} dBm")
    
    chan = cfg.get("channel", {})
    print(f"  Fading Model:            {chan.get('fading_type')}")
    print(f"  Path Loss Exponent:      {chan.get('path_loss_exponent')}")
    print(f"  Noise Power:             {chan.get('noise_power_dbm')} dBm")
    
    print("\n[Training Parameters]")
    tr = cfg.get("training", {})
    print(f"  Discount (Gamma):        {tr.get('gamma')}")
    print(f"  Soft Update (Tau):       {tr.get('tau')}")
    print(f"  Policy Update Delay:     {tr.get('policy_delay')}")
    print(f"  Exploration Noise:       {tr.get('exploration_noise')}")
    print(f"  Batch Size:              {tr.get('batch_size')}")
    print(f"  Actor Learning Rate:     {tr.get('lr_actor')}")
    print(f"  Critic Learning Rate:    {tr.get('lr_critic')}")
    
    print("\n[Replay Buffer & Constraints (Underlay/Overlay TD3)]")
    camo = cfg.get("camo_td3", {})
    print(f"  Sequence History Length: {camo.get('history_length')}")
    print(f"  PU Rate QoS Target:      {camo.get('pu_rate_threshold')} bps/Hz")
    print(f"  Interference Limit:      {camo.get('interference_limit_dbm')} dBm")
    print(f"  Energy Limit:            {camo.get('energy_limit_watts')} Watts")
    
    print("\n[Neural Network Architectures]")
    print("  Actor:   TD3_Actor (TD3) | CAMO_Actor (Underlay / Overlay TD3)")
    print("  Critic:  TwinCritics (3 independent pairs: Throughput, Interference, Energy/QoS)")
    print("  Encoder: GRUBeliefEncoder (L=10, state dimensionality=64)")
    print_footer()

def handle_checkpoints(args: Any):
    """Executes checkpoints command."""
    print_header("Model Checkpoints Inspector")
    output_dir = args.output_dir
    
    print("| Algorithm | Episode | Timestamp | Size (MB) | Best/Final | Path |")
    print("|---|---|---|---|---|---|")
    
    found_any = False
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".pth"):
                # Filter by agent if requested
                if args.agent:
                    agent_folder = args.agent.lower()
                    legacy_folder = args.agent.lower().replace("underlay_", "").replace("overlay_", "")
                    legacy_agent_name = "CAMO_TD3" if args.agent == "UNDERLAY_TD3" else ("OVERLAY_CAMO_TD3" if args.agent == "OVERLAY_TD3" else args.agent)
                    if (agent_folder not in root.lower()) and (legacy_folder not in root.lower()) and (args.agent not in file) and (legacy_agent_name not in file):
                        continue
                        
                filepath = os.path.join(root, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                mtime = os.path.getmtime(filepath)
                timestamp_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                
                # Check contents
                try:
                    checkpoint = torch.load(filepath, map_location="cpu")
                    algo = checkpoint.get("algorithm", "Unknown")
                    steps = checkpoint.get("total_it", 0)
                    
                    # Try to reconstruct episodes from config
                    steps_per_ep = 500
                    if "config" in checkpoint and "simulation" in checkpoint["config"]:
                        steps_per_ep = checkpoint["config"]["simulation"].get("time_steps_per_episode", 500)
                    ep = steps // steps_per_ep
                except Exception:
                    algo = "Unknown"
                    ep = "N/A"
                    
                is_best = "Best" if "best" in file.lower() else ("Final" if "final" in file.lower() else "Checkpoint")
                rel_path = os.path.relpath(filepath, output_dir)
                
                print(f"| {SHORT_NAMES_MAP.get(algo, algo)} | {ep} | {timestamp_str} | {size_mb:.2f} MB | {is_best} | {rel_path} |")
                found_any = True
                
    if not found_any:
        print("| *None* | - | - | - | - | - |")
    print_footer()

SHORT_NAMES_MAP = {
    "TD3": "TD3",
    "UNDERLAY_TD3": "Underlay TD3",
    "OVERLAY_TD3": "Overlay TD3",
    "CAMO_TD3": "Underlay TD3",
    "OVERLAY_CAMO_TD3": "Overlay TD3"
}

def execute_cli(args: Any):
    """Routes to the correct subcommand handler."""
    handlers = {
        "train": handle_train,
        "evaluate": handle_evaluate,
        "benchmark": handle_benchmark,
        "compare": handle_compare,
        "plots": handle_plots,
        "report": handle_report,
        "resume": handle_resume,
        "test": handle_test,
        "config": handle_config,
        "checkpoints": handle_checkpoints
    }
    
    if args.command in handlers:
        handlers[args.command](args)
    else:
        print(f"Error: Unknown command '{args.command}'")
        sys.exit(1)
