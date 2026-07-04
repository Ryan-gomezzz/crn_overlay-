"""
Relay protocol implementation.
Assignee: Shreya
"""

from typing import Protocol

from simulator.metrics import calculate_sinr


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

    def __init__(self, snr_threshold: float = 0.0):
        """
        Initialize the relay with a default SINR threshold.
        """
        self.snr_threshold = snr_threshold
        self.last_sinr = 0.0
        self.decoded = False
        self.signal = 0.0

    def receive(
        self,
        signal: float,
        interference: float,
        noise: float,
    ) -> bool:
        """
        Receive a signal, compute its SINR,
        and determine whether decoding succeeds.
        """
        self.signal = signal

        self.last_sinr = calculate_sinr(
            signal=signal,
            interference=interference,
            noise=noise,
        )

        self.decoded = self.can_decode(self.last_sinr)
        return self.decoded

    def transmit(self) -> float:
        """
        Forward the decoded signal.

        Returns:
            Original signal if decoding succeeds,
            otherwise 0.0.
        """
        if self.decoded:
            return self.signal
        return 0.0

    def can_decode(self, sinr: float, threshold: float | None = None) -> bool:
        """
        Return True if received SINR is above the threshold.
        """
        if threshold is None:
            threshold = self.snr_threshold

        return sinr >= threshold
