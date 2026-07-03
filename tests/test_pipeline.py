"""
Tests for the experiment pipeline.
Author: Ryan
"""
import os
import shutil
import tempfile
import yaml

from experiments.pipeline import ExperimentConfig, create_experiment_directory, ExperimentLogger


def test_experiment_config_load():
    """Verify ExperimentConfig loads config.yaml."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_yaml = os.path.join(tmp_dir, "base.yaml")
        exp_yaml = os.path.join(tmp_dir, "exp.yaml")
        
        with open(base_yaml, "w") as f:
            yaml.dump({"simulation": {"seed": 42}, "network": {"d_pt_pr": 100.0}}, f)
            
        with open(exp_yaml, "w") as f:
            yaml.dump({"simulation": {"seed": 99}}, f)
            
        config = ExperimentConfig(base_yaml, exp_yaml)
        
        # Override works
        assert config.get("simulation.seed") == 99
        # Base still there
        assert config.get("network.d_pt_pr") == 100.0
        # Default value fallback works
        assert config.get("missing.key", "default") == "default"


def test_experiment_directory_creation():
    """Verify directory structure is created."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        exp_dir = create_experiment_directory(tmp_dir, "test_exp")
        
        assert os.path.exists(exp_dir)
        assert os.path.isdir(os.path.join(exp_dir, "logs"))
        assert os.path.isdir(os.path.join(exp_dir, "checkpoints"))
        assert os.path.isdir(os.path.join(exp_dir, "results"))


def test_experiment_logger_creation():
    """Verify logger initializes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_dir = os.path.join(tmp_dir, "logs")
        config = {"logging": {"tensorboard": False, "wandb": False}}
        
        logger = ExperimentLogger("test_exp", log_dir, config)
        
        assert os.path.exists(os.path.join(log_dir, "experiment.log"))
        
        logger.log_text("Test message")
        logger.close()
        
        with open(os.path.join(log_dir, "experiment.log"), "r") as f:
            content = f.read()
            assert "Test message" in content
