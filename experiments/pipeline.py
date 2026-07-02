"""
Experiment pipeline for running, logging, and tracking CRN-RL experiments.
Author: Ryan
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import yaml

# Optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None


class ExperimentLogger:
    """Unified logging with Python logging, TensorBoard, and W&B support."""

    def __init__(self, experiment_name: str, log_dir: str, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.config = config
        
        # 1. Setup Python logging
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, "experiment.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # 2. Setup TensorBoard
        self.tb_writer = None
        if HAS_TENSORBOARD and config.get("logging", {}).get("tensorboard", False):
            tb_dir = os.path.join(log_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            self.log_text("TensorBoard initialized.", "info")

        # 3. Setup W&B
        self.use_wandb = False
        if HAS_WANDB and config.get("logging", {}).get("wandb", False):
            wandb_proj = config.get("logging", {}).get("wandb_project", "crn-overlay-rl")
            wandb.init(project=wandb_proj, name=experiment_name, config=config, dir=log_dir)
            self.use_wandb = True
            self.log_text("W&B initialized.", "info")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar to all active backends."""
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
        if self.use_wandb:
            wandb.log({tag: value}, step=step)

    def log_scalars(self, tag_value_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        if self.tb_writer:
            for tag, value in tag_value_dict.items():
                self.tb_writer.add_scalar(tag, value, step)
        if self.use_wandb:
            wandb.log(tag_value_dict, step=step)

    def log_text(self, message: str, level: str = "info"):
        """Log a text message."""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        else:
            self.logger.debug(message)

    def close(self):
        """Close all logging backends."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class ExperimentConfig:
    """Loads and merges base + experiment YAML configs."""

    def __init__(self, base_config_path: str, experiment_config_path: Optional[str] = None):
        self.config = {}
        
        # Load base
        if os.path.exists(base_config_path):
            with open(base_config_path, "r", encoding="utf-8") as f:
                base_cfg = yaml.safe_load(f) or {}
                self._update_dict(self.config, base_cfg)
        else:
            raise FileNotFoundError(f"Base config not found: {base_config_path}")
            
        # Load experiment overrides
        if experiment_config_path and os.path.exists(experiment_config_path):
            with open(experiment_config_path, "r", encoding="utf-8") as f:
                exp_cfg = yaml.safe_load(f) or {}
                self._update_dict(self.config, exp_cfg)
                
    def _update_dict(self, d: Dict, u: Dict):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split('.')
        d = self.config
        for p in parts[:-1]:
            d = d.get(p, {})
        return d.get(parts[-1], default)

    def to_dict(self) -> Dict[str, Any]:
        return self.config


def create_experiment_directory(base_dir: str, experiment_name: str) -> str:
    """Create a structured experiment output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    
    return exp_dir


def run_experiment(config_path: str, experiment_config_path: Optional[str] = None):
    """Full experiment pipeline."""
    # 1. Load configs
    config = ExperimentConfig(config_path, experiment_config_path)
    
    exp_name = config.get("experiment.name", "experiment")
    
    # 2. Create experiment directory
    base_dir = "experiments/runs"
    exp_dir = create_experiment_directory(base_dir, exp_name)
    
    # 3. Setup logging
    logger = ExperimentLogger(exp_name, os.path.join(exp_dir, "logs"), config.to_dict())
    logger.log_text(f"Started experiment: {exp_name}")
    
    # Save a copy of the merged config
    with open(os.path.join(exp_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f)
    logger.log_text(f"Saved merged configuration to {exp_dir}/config.yaml")
    
    # 4. Create environment (import locally to avoid circular dependency)
    from envs.crn_env import make_crn_env
    env = make_crn_env(config_path=os.path.join(exp_dir, "config.yaml"))
    
    logger.log_text("Environment initialized successfully.")
    
    # Print summary
    logger.log_text(f"Observation space: {env.observation_space}")
    logger.log_text(f"Action space: {env.action_space}")
    logger.log_text("Ready for RL training (Aditya's module).")
    
    # Clean up
    env.close()
    logger.close()
    return exp_dir
