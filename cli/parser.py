"""
CLI Argument Parser and Validator for CRN Research Framework.
"""

import argparse
import sys
from typing import List, Optional

# Valid algorithms mapping
AGENT_MAP = {
    "t3": "TD3",
    "underlay": "CAMO_TD3",
    "overlay": "OVERLAY_CAMO_TD3"
}

REVERSE_AGENT_MAP = {v: k for k, v in AGENT_MAP.items()}

VALID_SEEDS = [42, 123, 2026]

def parse_agent(val: str) -> str:
    """Map user-input agent string to config agent name."""
    lower_val = val.lower().strip()
    if lower_val in AGENT_MAP:
        return AGENT_MAP[lower_val]
    # Allow passing direct config names just in case
    if val in AGENT_MAP.values():
        return val
    raise argparse.ArgumentTypeError(
        f"Invalid agent: '{val}'. Choose from: {list(AGENT_MAP.keys())}"
    )

def parse_agents_list(values: List[str]) -> List[str]:
    """Map list of user-input agents."""
    return [parse_agent(v) for v in values]

def validate_episodes(val: str) -> int:
    """Validate episode count constraint."""
    try:
        ival = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Episodes must be an integer, got '{val}'")
    if not (500 <= ival <= 5000):
        raise argparse.ArgumentTypeError(f"Episodes must be between 500 and 5000 (inclusive), got {ival}")
    return ival

def validate_steps(val: str) -> int:
    """Validate steps per episode constraint."""
    try:
        ival = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Steps per episode must be an integer, got '{val}'")
    if not (200 <= ival <= 2000):
        raise argparse.ArgumentTypeError(f"Steps per episode must be between 200 and 2000 (inclusive), got {ival}")
    return ival

def validate_seed(val: str) -> int:
    """Validate seed is one of the allowed seeds."""
    try:
        ival = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Seed must be an integer, got '{val}'")
    if ival not in VALID_SEEDS:
        raise argparse.ArgumentTypeError(f"Seed must be one of {VALID_SEEDS}, got {ival}")
    return ival

def str_to_bool(val: str) -> bool:
    """Convert string truth values to bool."""
    if isinstance(val, bool):
        return val
    lower_val = val.lower().strip()
    if lower_val in ('true', 'yes', 't', 'y', '1'):
        return True
    if lower_val in ('false', 'no', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{val}'")

def add_common_overrides(parser: argparse.ArgumentParser):
    """Add training/execution parameters overrides to parser."""
    parser.add_argument("--episodes", type=validate_episodes, default=2000, help="Number of training episodes (500-5000)")
    parser.add_argument("--steps", type=validate_steps, default=500, help="Steps per episode (200-2000)")
    parser.add_argument("--seed", type=validate_seed, help=f"Random seed to use {VALID_SEEDS}")
    parser.add_argument("--all-seeds", action="store_true", help="Run execution over all valid seeds (42, 123, 2026)")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Computation device (cpu or cuda)")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--lr", type=float, help="Learning rate for actor and critic networks")
    parser.add_argument("--checkpoint-every", type=int, help="Save periodic checkpoint every N episodes")
    parser.add_argument("--save-best", type=str_to_bool, default=True, help="Save the best model based on evaluation metrics")
    parser.add_argument("--save-final", type=str_to_bool, default=True, help="Save the final model at the end of execution")
    parser.add_argument("--tensorboard", type=str_to_bool, default=True, help="Enable or disable TensorBoard logging")
    parser.add_argument("--output-dir", type=str, default="experiments", help="Output directory for experiments")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose console logs")
    parser.add_argument("--render", action="store_true", help="Enable rendering of the env (if supported)")

def get_parser() -> argparse.ArgumentParser:
    """Create and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="CRN-RL Research Framework - CLI Interface & Experiment Management",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Train Subcommand
    train_parser = subparsers.add_parser("train", help="Train one or more RL agents")
    # Support both singular and plural arguments
    agent_group = train_parser.add_mutually_exclusive_group(required=True)
    agent_group.add_argument("--agent", type=parse_agent, help="Algorithm to train (t3, underlay, overlay)")
    agent_group.add_argument("--agents", nargs="+", type=parse_agent, help="List of algorithms to train sequentially")
    add_common_overrides(train_parser)

    # 2. Evaluate Subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Deterministic policy evaluation of a trained agent")
    eval_parser.add_argument("--agent", type=parse_agent, required=True, help="Algorithm to evaluate")
    eval_parser.add_argument("--episodes", type=int, default=20, help="Number of episodes for evaluation")
    eval_parser.add_argument("--seed", type=validate_seed, default=42, help="Seed for evaluation reproducibility")
    eval_parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to load model onto")
    eval_parser.add_argument("--output-dir", type=str, default="experiments", help="Base directory where checkpoints are stored")
    eval_parser.add_argument("--render", action="store_true", help="Render environment steps during evaluation")

    # 3. Benchmark Subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Compare and benchmark performance of multiple agents")
    bench_parser.add_argument("--agents", nargs="+", type=parse_agent, default=list(AGENT_MAP.values()),
                             help="Algorithms to benchmark (defaults to all)")
    bench_parser.add_argument("--seed", type=validate_seed, help="Seed for the benchmark")
    bench_parser.add_argument("--all-seeds", action="store_true", help="Run benchmark across all predefined seeds")
    bench_parser.add_argument("--device", choices=["cpu", "cuda"], help="Computation device (cpu or cuda)")
    bench_parser.add_argument("--output-dir", type=str, default="experiments", help="Base output directory")
    bench_parser.add_argument("--episodes", type=validate_episodes, default=2000, help="Episodes per agent in benchmark (500-5000)")
    bench_parser.add_argument("--steps", type=validate_steps, default=500, help="Steps per episode (200-2000)")

    # 4. Compare Subcommand
    compare_parser = subparsers.add_parser("compare", help="Compare training/evaluation results and generate summary stats")
    compare_parser.add_argument("--agents", nargs="+", type=parse_agent, default=list(AGENT_MAP.values()),
                                help="Agents to include in comparison")
    compare_parser.add_argument("--output-dir", type=str, default="experiments", help="Base output directory")

    # 5. Plots Subcommand
    plots_parser = subparsers.add_parser("plots", help="Generate comparison plots from existing runs")
    plots_parser.add_argument("--output-dir", type=str, default="experiments", help="Base output directory")

    # 6. Report Subcommand
    report_parser = subparsers.add_parser("report", help="Generate detailed Markdown and visual summaries of experiments")
    report_parser.add_argument("--output-dir", type=str, default="experiments", help="Base output directory")

    # 7. Resume Subcommand
    resume_parser = subparsers.add_parser("resume", help="Resume training from latest checkpoint of an algorithm")
    resume_parser.add_argument("--agent", type=parse_agent, required=True, help="Algorithm to resume (t3, underlay, overlay)")
    add_common_overrides(resume_parser) # Allow modifying training specs upon resume

    # 8. Test Subcommand
    subparsers.add_parser("test", help="Run unit tests, configuration validation, and smoke tests")

    # 9. Config Subcommand
    subparsers.add_parser("config", help="Inspect active environment, network, and algorithm configurations")

    # 10. Checkpoints Subcommand
    ckpt_parser = subparsers.add_parser("checkpoints", help="List and inspect stored model checkpoints")
    ckpt_parser.add_argument("--agent", type=parse_agent, help="Filter checkpoints by agent")
    ckpt_parser.add_argument("--output-dir", type=str, default="experiments", help="Base output directory")

    return parser
