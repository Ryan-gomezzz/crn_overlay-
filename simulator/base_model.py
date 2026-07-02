"""
Base simulator interfaces and protocols.
Author: Ryan
"""

from typing import Any, Dict, Protocol


class BaseSimulator(Protocol):
    """
    Abstract interface for the communication simulator.
    """

    def reset(self) -> Dict[str, Any]:
        """
        Reset the simulator to an initial state.
        Returns:
            Dictionary containing initial state information.
        """
        ...

    def step(self, action: Any) -> Dict[str, Any]:
        """
        Advance the simulation by one step.
        Args:
            action: Action taken by the RL agent.
        Returns:
            Dictionary containing results of the step (SINR, Throughput, etc).
        """
        ...
