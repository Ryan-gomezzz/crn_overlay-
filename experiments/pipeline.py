"""
Experiment Pipeline and Logging Setup.
Author: Ryan
"""

import logging
import os

# import wandb


def setup_logging(experiment_name: str):
    """
    Setup Python logging, TensorBoard, and WandB integration.
    """
    log_dir = os.path.join("experiments", "logs", experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_dir, "experiment.log"),
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # TODO (Ryan): Initialize TensorBoard SummaryWriter
    # TODO (Ryan): Initialize Weights & Biases if enabled

    return logging.getLogger(experiment_name)


def run_experiment(config_path: str):
    """
    Execute a full experiment based on a config file.
    """
    pass
