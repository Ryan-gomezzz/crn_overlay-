"""
Relay protocol implementation.
Assignee: Shreya
"""

# TODO (Shreya): Implement Decode-and-Forward (DF) logic.
# Expected: Logic to check if relay decodes primary/secondary signal.
# Reference: docs/team_guides/relay_module.md

from abc import ABC, abstractmethod
from simulator.metrics import calculate_sinr


class RelayProtocol(ABC):
    @abstractmethod
    def receive(self, signal: float, interference: float, noise: float):
        pass

    @abstractmethod
    def transmit(self) -> float:
        pass


class DecodeAndForward(RelayProtocol):
    def __init__(self, snr_threshold: float):
        self.snr_threshold = snr_threshold
        self.buffered_signal = 0
        self.decoded = False

    def receive(self, signal: float, interference: float, noise: float):
        snr = calculate_sinr(signal, interference, noise)

        if snr > self.snr_threshold:
            self.decoded = True
            self.buffered_signal = signal
        else:
            self.decoded = False
            self.buffered_signal = 0

    def transmit(self) -> float:
        return self.buffered_signal if self.decoded else 0.0
from typing import Protocol


class RelayProtocol(Protocol):
    """
    Protocol for Relay protocols.
    """

    def can_decode(self, sinr: float, threshold: float) -> bool:
        """
        Check if the relay can successfully decode the signal.
        """
        ...


class DecodeAndForward(RelayProtocol):
    """
    Decode-and-Forward (DF) relay protocol.
    """

    def can_decode(self, sinr: float, threshold: float) -> bool:
        """
        Return True if received SINR is above the threshold.
        """
        return sinr >= threshold
