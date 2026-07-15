"""
Simulator package for the CRN-RL Framework.
Author: Ryan

Exports the NOMA CRN simulator implementation.
"""

from simulator.noma_overlay_model import NOMAOverlaySimulator, NOMAConfig

__all__ = [
    "NOMAOverlaySimulator",
    "NOMAConfig",
]
