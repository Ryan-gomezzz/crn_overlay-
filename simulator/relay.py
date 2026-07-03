"""
Relay protocol implementation.
Assignee: Shreya
"""

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
