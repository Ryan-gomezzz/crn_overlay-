"""
Base simulator interfaces, protocols, and data structures.
Author: Ryan

This module defines the abstract contracts that all simulator components
must satisfy. Other team members implement these protocols:
  - Sneha: ChannelModelProtocol, PropagationModelProtocol
  - Shreya: RelayProtocol, InterferenceModelProtocol, MetricsCalculatorProtocol
  - Ryan: BaseSimulator ABC, SimulatorConfig, SimulationState, SimulationResult
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimulatorConfig:
    """All configuration parameters for the Overlay CRN simulator.

    Distances follow the standard Overlay CRN topology:
        PT -----(d_pt_pr)-----> PR
        PT -----(d_pt_relay)---> Relay
        SU -----(d_su_relay)---> Relay
        Relay --(d_relay_pr)---> PR
        Relay --(d_relay_sud)--> SU Dest
        SU -----(d_su_sud)-----> SU Dest
        PT -----(d_pt_sud)-----> SU Dest

    Attributes:
        d_pt_pr: Distance from Primary Transmitter to Primary Receiver (m).
        d_pt_relay: Distance from PT to SU Relay (m).
        d_su_relay: Distance from SU Source to SU Relay (m).
        d_relay_pr: Distance from Relay to PR (m).
        d_relay_sud: Distance from Relay to SU Destination (m).
        d_su_sud: Distance from SU Source to SU Destination (m).
        d_pt_sud: Distance from PT to SU Destination (m).
        p_max_su: Maximum SU source transmit power (Watts).
        p_max_relay: Maximum relay transmit power (Watts).
        p_pt: Fixed PT transmit power (Watts).
        noise_power: Additive white Gaussian noise power (Watts).
        bandwidth: System bandwidth (Hz).
        num_channels: Number of channel-gain values in the observation.
        max_steps: Maximum time-steps per episode.
        interference_threshold: Maximum tolerable interference at PR.
        penalty_weight: Penalty multiplier when constraint is violated.
    """

    # Network topology distances (metres)
    d_pt_pr: float = 100.0
    d_pt_relay: float = 50.0
    d_su_relay: float = 30.0
    d_relay_pr: float = 60.0
    d_relay_sud: float = 40.0
    d_su_sud: float = 70.0
    d_pt_sud: float = 90.0

    # Power constraints (Watts)
    p_max_su: float = 1.0
    p_max_relay: float = 1.0
    p_pt: float = 1.0

    # Noise
    noise_power: float = 1e-10

    # System parameters
    bandwidth: float = 1e6
    num_channels: int = 6
    max_steps: int = 200

    # PU protection
    interference_threshold: float = 0.1
    penalty_weight: float = 10.0


# ---------------------------------------------------------------------------
# Simulation State & Result
# ---------------------------------------------------------------------------


@dataclass
class SimulationState:
    """Snapshot of the simulator at a given time-step.

    Attributes:
        channel_gains: Raw channel gain coefficients keyed by link name
            (e.g. ``"pt_pr"``, ``"su_relay"``).
        path_losses: Path-loss multipliers keyed by link name.
        effective_gains: ``channel_gains * path_losses`` per link.
        power_allocation: Agent-chosen power vector ``[p_su, p_relay]``.
        su_throughput: Achieved SU throughput this step (bits/s/Hz).
        pu_throughput: Achieved PU throughput this step (bits/s/Hz).
        interference_at_pr: Total interference at PR from SU network.
        relay_decoded: Whether the relay successfully decoded the signal.
        step_count: Current step within the episode.
        sinr_su: SINR at SU destination.
        sinr_pu: SINR at PR.
    """

    channel_gains: Dict[str, float] = field(default_factory=dict)
    path_losses: Dict[str, float] = field(default_factory=dict)
    effective_gains: Dict[str, float] = field(default_factory=dict)
    power_allocation: Optional[np.ndarray] = None
    su_throughput: float = 0.0
    pu_throughput: float = 0.0
    interference_at_pr: float = 0.0
    relay_decoded: bool = False
    step_count: int = 0
    sinr_su: float = 0.0
    sinr_pu: float = 0.0


@dataclass
class SimulationResult:
    """Returned by ``BaseSimulator.step()`` for the Gymnasium wrapper.

    Attributes:
        observation: Numpy observation vector for the RL agent.
        reward: Scalar reward signal.
        terminated: ``True`` when a terminal condition is met.
        truncated: ``True`` when the episode is truncated (max steps).
        info: Auxiliary diagnostic information.
    """

    observation: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocols (interfaces for other team members)
# ---------------------------------------------------------------------------


@runtime_checkable
class ChannelModelProtocol(Protocol):
    """Interface for wireless channel generation (Sneha).

    Implementations must produce a dictionary of channel-gain
    coefficients for every link in the network.
    """

    def generate_channels(
        self, rng: np.random.Generator
    ) -> Dict[str, float]:
        """Generate channel gain coefficients for all network links.

        Args:
            rng: NumPy random generator for reproducibility.

        Returns:
            Dictionary mapping link names to channel gain values.
            Expected keys: ``"pt_pr"``, ``"pt_relay"``,
            ``"su_relay"``, ``"relay_pr"``, ``"relay_sud"``,
            ``"pt_sud"``.
        """
        ...


@runtime_checkable
class PropagationModelProtocol(Protocol):
    """Interface for path-loss computation (Sneha).

    Returns a *linear* path-loss multiplier (not in dB).
    """

    def compute_path_loss(self, distance: float) -> float:
        """Compute the path-loss multiplier for a given distance.

        Args:
            distance: Link distance in metres.

        Returns:
            Linear path-loss value (0, 1]. Smaller means more loss.
        """
        ...


@runtime_checkable
class RelayProtocol(Protocol):
    """Interface for the relay decision logic (Shreya)."""

    def can_decode(self, snr: float) -> bool:
        """Decide whether the relay can decode the received signal.

        Args:
            snr: Signal-to-Noise Ratio at the relay (linear).

        Returns:
            ``True`` if the relay successfully decodes.
        """
        ...

    def forward(
        self, channel_gain: float, power: float, noise: float
    ) -> float:
        """Compute the forwarded signal power after relay processing.

        Args:
            channel_gain: Effective channel gain on the relay-dest link.
            power: Relay transmit power (Watts).
            noise: Noise power (Watts).

        Returns:
            Received signal power at the destination from the relay.
        """
        ...


@runtime_checkable
class InterferenceModelProtocol(Protocol):
    """Interface for interference calculations (Shreya)."""

    def compute_interference(
        self,
        powers: Dict[str, float],
        gains: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute the interference power at each receiver.

        Args:
            powers: Transmit powers keyed by transmitter name.
            gains: Effective channel gains keyed by link name.

        Returns:
            Interference values keyed by receiver name
            (e.g. ``"pr"``, ``"sud"``).
        """
        ...


@runtime_checkable
class MetricsCalculatorProtocol(Protocol):
    """Interface for performance-metric computation (Shreya)."""

    def compute_sinr(
        self, signal: float, interference: float, noise: float
    ) -> float:
        """Compute the Signal-to-Interference-plus-Noise Ratio.

        Args:
            signal: Desired signal power (Watts).
            interference: Total interference power (Watts).
            noise: Noise power (Watts).

        Returns:
            SINR value (linear scale).
        """
        ...

    def compute_throughput(self, sinr: float, bandwidth: float) -> float:
        """Compute achievable throughput via Shannon capacity.

        Args:
            sinr: SINR value (linear).
            bandwidth: System bandwidth (Hz).

        Returns:
            Throughput in bits/s/Hz (normalised by bandwidth).
        """
        ...


# ---------------------------------------------------------------------------
# Abstract Base Simulator
# ---------------------------------------------------------------------------


class BaseSimulator(ABC):
    """Abstract base class every concrete simulator must extend.

    Sub-classes implement the physical-layer logic while this class
    enforces the contract required by the Gymnasium environment.

    Args:
        config: A ``SimulatorConfig`` instance.
    """

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> SimulationState:
        """Reset the simulator and return the initial state."""

    @abstractmethod
    def step(self, action: np.ndarray) -> SimulationResult:
        """Advance simulation by one step given an agent action."""

    @abstractmethod
    def get_observation(self, state: SimulationState) -> np.ndarray:
        """Extract a flat observation vector from the current state."""

    @abstractmethod
    def compute_reward(self, state: SimulationState) -> float:
        """Compute the scalar reward from the current state."""
