"""
Benchmark tool to train, evaluate, and compare TD3, CAMO-TD3,
and OVERLAY_CAMO_TD3.

Generates metrics graphs, measures training/inference speeds,
and reports summary statistics.
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.train_td3 import TD3Agent
from envs.crn_env import OverlayCRNEnv
from main import evaluate_policy, set_seed


# Load master config
def load_config():
    config_path = os.path.join("configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(algo_name: str, total_steps: int = 600, eval_interval: int = 100):
    print(f"\n================ STARTING EXPERIMENT FOR {algo_name} ================")
    config = load_config()
    config["algorithm"]["name"] = algo_name

    # Speed up training for smoke test benchmarking
    config["training"]["total_steps"] = total_steps
    config["training"]["start_steps"] = 150
    config["training"]["batch_size"] = 32

    set_seed(42)
    env = OverlayCRNEnv(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = TD3Agent(config, device=device)

    # Performance tracking
    history = {
        "steps": [],
        "rewards": [],
        "throughput_s": [],
        "throughput_p": [],
        "outage": [],
        "ber": [],
        "average_power": [],
        "lambda_qos": [],
        "lambda_nrg": [],
        "lambda_inf": [],
    }

    obs, info = env.reset()
    episode_reward = 0.0

    t_start = time.time()

    # Dummy writer for tensorboard avoidance in standalone bench
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=f"experiments/runs_bench/{algo_name}")

    for step in range(1, total_steps + 1):
        action = agent.select_action(obs, info, explore=True)
        next_obs, reward, done, truncated, next_info = env.step(action)
        episode_reward += reward

        agent.replay_buffer.add(obs, action, reward, next_obs, done or truncated, info)

        obs = next_obs
        info = next_info

        if step >= config["training"]["start_steps"]:
            agent.train(writer)

        if done or truncated:
            obs, info = env.reset()
            episode_reward = 0.0

        if step % eval_interval == 0:
            eval_env = OverlayCRNEnv(config)
            eval_metrics = evaluate_policy(agent, eval_env, episodes=3)

            history["steps"].append(step)
            history["rewards"].append(eval_metrics["total_reward"])
            history["throughput_s"].append(eval_metrics["throughput_s"])
            history["throughput_p"].append(eval_metrics["throughput_p"])
            history["outage"].append(eval_metrics["outage"])
            history["ber"].append(eval_metrics["ber"])
            history["average_power"].append(eval_metrics["average_power"])

            # Record Lagrangian values
            if algo_name == "OVERLAY_CAMO_TD3":
                history["lambda_qos"].append(agent.lambda_qos)
                history["lambda_nrg"].append(agent.lambda_nrg)
            elif algo_name == "CAMO_TD3":
                history["lambda_inf"].append(agent.lambda_inf)
                history["lambda_nrg"].append(agent.lambda_nrg)

            print(
    f"[{algo_name}] Step {step}/{total_steps} | "
    f"Eval Reward: {eval_metrics['total_reward']:.2f} | "
    f"SU Thr: {eval_metrics['throughput_s']:.3f} | "
    f"PU Outage: {eval_metrics['outage']:.4f}"
)

    t_end = time.time()
    train_time = t_end - t_start
    writer.close()

    # Save checkpoint
    checkpoint_dir = "experiments/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    agent.save(f"{checkpoint_dir}/{algo_name}_best_model.pth")

    # Measure Inference time over 100 trials
    obs, info = env.reset()
    t_inf_start = time.time()
    for _ in range(100):
        _ = agent.select_action(obs, info, explore=False)
    t_inf_end = time.time()
    avg_inf_time = (t_inf_end - t_inf_start) / 100.0

    print(
    f"[{algo_name}] Finished in {train_time:.2f}s | "
    f"Avg Inference Time: "
    f"{avg_inf_time * 1000.0:.3f}ms"
)

    return history, train_time, avg_inf_time


def generate_plots(results):
    os.makedirs("plots", exist_ok=True)
    algos = list(results.keys())

    # Color palette
    colors = {"TD3": "#e74c3c", "CAMO_TD3": "#3498db", "OVERLAY_CAMO_TD3": "#2ecc71"}

    # 1. Throughput Comparison
    plt.figure()
    for name in algos:
        plt.plot(
            results[name]["history"]["steps"],
            results[name]["history"]["throughput_s"],
            label=name,
            color=colors[name],
            marker="o",
        )
    plt.xlabel("Training Steps")
    plt.ylabel("Secondary Throughput (bps/Hz)")
    plt.title("Secondary User (SU) Throughput Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/throughput_comparison.png")
    plt.close()

    # 2. BER Comparison
    plt.figure()
    for name in algos:
        plt.plot(
            results[name]["history"]["steps"],
            results[name]["history"]["ber"],
            label=name,
            color=colors[name],
            marker="x",
        )
    plt.xlabel("Training Steps")
    plt.ylabel("Bit Error Rate (BER)")
    plt.yscale("log")
    plt.title("SU Bit Error Rate (BER) Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/ber_comparison.png")
    plt.close()

    # 3. Outage Comparison
    plt.figure()
    for name in algos:
        plt.plot(
            results[name]["history"]["steps"],
            results[name]["history"]["outage"],
            label=name,
            color=colors[name],
            marker="s",
        )
    plt.xlabel("Training Steps")
    plt.ylabel("Primary User Outage Rate")
    plt.title("Primary User (PU) Outage Rate Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/outage_comparison.png")
    plt.close()

    # 4. Convergence Comparison
    plt.figure()
    for name in algos:
        plt.plot(
            results[name]["history"]["steps"],
            results[name]["history"]["rewards"],
            label=name,
            color=colors[name],
            marker="d",
        )
    plt.xlabel("Training Steps")
    plt.ylabel("Episode Total Reward")
    plt.title("Policy Convergence Comparison (Rewards)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/convergence_comparison.png")
    plt.close()

    # 5. Lagrangian Multipliers comparison
    plt.figure()
    if "CAMO_TD3" in results:
        h = results["CAMO_TD3"]["history"]
        plt.plot(
            h["steps"],
            h["lambda_nrg"],
            label="CAMO-TD3 Energy Multiplier",
            color="#34495e",
            linestyle="--",
        )
    if "OVERLAY_CAMO_TD3" in results:
        h = results["OVERLAY_CAMO_TD3"]["history"]
        plt.plot(
            h["steps"],
            h["lambda_qos"],
            label="OVERLAY QoS Multiplier",
            color="#2ecc71",
            linestyle="-",
        )
        plt.plot(
            h["steps"],
            h["lambda_nrg"],
            label="OVERLAY Energy Multiplier",
            color="#27ae60",
            linestyle=":",
        )
    plt.xlabel("Training Steps")
    plt.ylabel("Lagrangian Multiplier Value")
    plt.title("Lagrangian Multipliers Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/lambda_comparison.png")
    plt.close()

    # 6. Training & Inference Time Comparison
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    train_times = [results[n]["train_time"] for n in algos]
    inf_times = [results[n]["inf_time"] * 1000.0 for n in algos]  # in ms

    x = np.arange(len(algos))
    width = 0.35

    ax1.bar(
        x - width / 2,
        train_times,
        width,
        label="Training Time (s)",
        color="#9b59b6",
    )

    ax2.bar(
        x + width / 2,
        inf_times,
        width,
        label="Inference Time (ms)",
        color="#f1c40f",
    )

    ax1.set_xlabel("Algorithms")
    ax1.set_ylabel("Training Time (seconds)", color="#9b59b6")
    ax2.set_ylabel("Inference Time (milliseconds)", color="#f1c40f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos)
    plt.title("Computational Resource and Speed Benchmark")
    fig.tight_layout()
    plt.savefig("plots/time_comparison.png")
    plt.close()


def main():
    results = {}

    for name in ["TD3", "CAMO_TD3", "OVERLAY_CAMO_TD3"]:
        history, train_time, inf_time = run_experiment(
            name, total_steps=600, eval_interval=100
        )
        results[name] = {
            "history": history,
            "train_time": train_time,
            "inf_time": inf_time,
        }

    generate_plots(results)

    # Print Markdown Summary Table
    print("\n\n================ BENCHMARKING SUMMARY ================")
    print(
    "| Algorithm | Training Time (s) | "
    "Avg Inference Time (ms) | "
    "Final Eval SU Rate | "
    "Final PU Outage |"
)
    print("|---|---|---|---|---|")
    for name in results:
        h = results[name]["history"]
        final_su_thr = h["throughput_s"][-1]
        final_pu_out = h["outage"][-1]
        print(
    f"| {name} | "
    f"{results[name]['train_time']:.2f}s | "
    f"{results[name]['inf_time'] * 1000.0:.3f}ms | "
    f"{final_su_thr:.4f} | "
    f"{final_pu_out:.4f} |"
)
    print("======================================================")


if __name__ == "__main__":
    main()
