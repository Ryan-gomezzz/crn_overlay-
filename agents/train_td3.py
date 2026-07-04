"""
Unified agent dispatch facade.
Imports TD3, Underlay TD3 (CAMO-TD3), and Overlay TD3 implementations and dispatches
calls dynamically depending on configuration. Maintains backward compatibility.
"""

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.td3 import TD3Agent as BaselineTD3Agent
from agents.underlay_td3 import UnderlayTD3Agent
from agents.overlay_td3 import OverlayTD3Agent


class TD3Agent:
    """
    Facade wrapper for TD3-based agents in the CRN framework.
    Delegates all actions, optimization, logging, and checkpoint operations
    to the respective standalone algorithm agent class.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device

        raw_algo_name = config.get("algorithm", {}).get("name", "TD3")
        
        # Standardize algorithm names for backward compatibility and presentation renames
        name_map = {
            "T3": "TD3",
            "CAMO_TD3": "UNDERLAY_TD3",
            "OVERLAY_CAMO_TD3": "OVERLAY_TD3",
        }
        self.algorithm_name = name_map.get(raw_algo_name, raw_algo_name)

        if self.algorithm_name == "UNDERLAY_TD3":
            self.agent = UnderlayTD3Agent(config, device)
        elif self.algorithm_name == "OVERLAY_TD3":
            self.agent = OverlayTD3Agent(config, device)
        elif self.algorithm_name == "TD3":
            self.agent = BaselineTD3Agent(config, device)
        else:
            raise ValueError(f"Unknown standardized algorithm name: {self.algorithm_name}")

    def select_action(self, obs: np.ndarray, info: dict = None, explore: bool = True) -> np.ndarray:
        """
        Delegate action selection to the active agent.
        """
        return self.agent.select_action(obs, info, explore)

    def train(self, writer: SummaryWriter) -> dict:
        """
        Delegate policy optimization to the active agent.
        """
        return self.agent.train(writer)

    def save(self, filepath: str):
        """
        Delegate checkpoint saving to the active agent.
        """
        self.agent.save(filepath)

    def load(self, filepath: str):
        """
        Delegate checkpoint loading to the active agent.
        """
        self.agent.load(filepath)

    @property
    def total_it(self) -> int:
        return self.agent.total_it

    @total_it.setter
    def total_it(self, value: int):
        self.agent.total_it = value

    @property
    def replay_buffer(self):
        return self.agent.replay_buffer

    @property
    def lambda_inf(self) -> float:
        if hasattr(self.agent, "lambda_inf"):
            return self.agent.lambda_inf
        return 0.0

    @property
    def lambda_qos(self) -> float:
        if hasattr(self.agent, "lambda_qos"):
            return self.agent.lambda_qos
        return 0.0

    @property
    def lambda_nrg(self) -> float:
        if hasattr(self.agent, "lambda_nrg"):
            return self.agent.lambda_nrg
        return 0.0
