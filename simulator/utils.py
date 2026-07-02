"""
Utility functions for the simulator.
"""


def dbm_to_watt(dbm: float) -> float:
    """Convert dBm to Watts."""
    return 10 ** ((dbm - 30) / 10)
