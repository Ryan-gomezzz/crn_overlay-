"""Shared pytest fixtures and path setup for the CRN test suite."""

import os
import sys

import pytest
import yaml

# Ensure the project root is importable when pytest is launched from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CONFIG_PATH = os.path.join(ROOT, "configs", "config.yaml")


@pytest.fixture(scope="session")
def config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def num_su(config):
    return config.get("multi_user", {}).get("num_su", 3)
