"""
Utility functions for the CRN-RL simulator.
Author: Ryan

Provides unit-conversion helpers and array-manipulation utilities
used across the simulator and environment modules.
"""

from __future__ import annotations

import math

import numpy as np


def dbm_to_watt(dbm: float) -> float:
    """Convert power from dBm to Watts.

    Args:
        dbm: Power in dBm.

    Returns:
        Power in Watts.
    """
    return 10.0 ** ((dbm - 30.0) / 10.0)


def watt_to_dbm(watt: float) -> float:
    """Convert power from Watts to dBm.

    Args:
        watt: Power in Watts (must be > 0).

    Returns:
        Power in dBm.

    Raises:
        ValueError: If *watt* is non-positive.
    """
    if watt <= 0:
        raise ValueError(f"Power must be positive, got {watt}")
    return 10.0 * math.log10(watt) + 30.0


def db_to_linear(db: float) -> float:
    """Convert a value from dB to linear scale.

    Args:
        db: Value in dB.

    Returns:
        Linear-scale value.
    """
    return 10.0 ** (db / 10.0)


def linear_to_db(linear: float) -> float:
    """Convert a value from linear scale to dB.

    Args:
        linear: Linear-scale value (must be > 0).

    Returns:
        Value in dB.

    Raises:
        ValueError: If *linear* is non-positive.
    """
    if linear <= 0:
        raise ValueError(f"Value must be positive, got {linear}")
    return 10.0 * math.log10(linear)


def validate_power(power: float, p_max: float) -> float:
    """Clip a power value to the valid range ``[0, p_max]``.

    Args:
        power: Proposed transmit power (Watts).
        p_max: Maximum allowable power (Watts).

    Returns:
        Clipped power value.
    """
    return float(np.clip(power, 0.0, p_max))


def normalize_observation(
    obs: np.ndarray,
    obs_min: np.ndarray,
    obs_max: np.ndarray,
) -> np.ndarray:
    """Min-max normalise an observation vector to [0, 1].

    Args:
        obs: Raw observation array.
        obs_min: Per-element minimum values.
        obs_max: Per-element maximum values.

    Returns:
        Normalised observation in [0, 1].
    """
    denom = obs_max - obs_min
    # Avoid division by zero
    denom = np.where(denom == 0, 1.0, denom)
    return np.clip((obs - obs_min) / denom, 0.0, 1.0).astype(np.float32)
