"""
Tests for utility functions.
Author: Ryan
"""
import pytest
import numpy as np

from simulator.utils import (
    dbm_to_watt,
    watt_to_dbm,
    db_to_linear,
    linear_to_db,
    validate_power,
    normalize_observation
)


def test_dbm_to_watt():
    """Test dBm to Watts conversion."""
    assert pytest.approx(dbm_to_watt(30.0)) == 1.0
    assert pytest.approx(dbm_to_watt(0.0)) == 0.001
    assert pytest.approx(dbm_to_watt(10.0)) == 0.01


def test_watt_to_dbm():
    """Test Watts to dBm conversion."""
    assert pytest.approx(watt_to_dbm(1.0)) == 30.0
    assert pytest.approx(watt_to_dbm(0.001)) == 0.0
    assert pytest.approx(watt_to_dbm(0.01)) == 10.0
    
    with pytest.raises(ValueError):
        watt_to_dbm(0.0)


def test_db_to_linear():
    """Test dB to linear scale conversion."""
    assert pytest.approx(db_to_linear(0.0)) == 1.0
    assert pytest.approx(db_to_linear(10.0)) == 10.0
    assert pytest.approx(db_to_linear(-10.0)) == 0.1


def test_linear_to_db():
    """Test linear scale to dB conversion."""
    assert pytest.approx(linear_to_db(1.0)) == 0.0
    assert pytest.approx(linear_to_db(10.0)) == 10.0
    assert pytest.approx(linear_to_db(0.1)) == -10.0
    
    with pytest.raises(ValueError):
        linear_to_db(0.0)


def test_validate_power():
    """Test power validation/clipping."""
    assert validate_power(0.5, p_max=1.0) == 0.5
    assert validate_power(1.5, p_max=1.0) == 1.0
    assert validate_power(-0.5, p_max=1.0) == 0.0


def test_normalize_observation():
    """Test observation normalization."""
    obs = np.array([0.0, 5.0, 10.0])
    obs_min = np.array([0.0, 0.0, 0.0])
    obs_max = np.array([10.0, 10.0, 10.0])
    
    norm_obs = normalize_observation(obs, obs_min, obs_max)
    
    np.testing.assert_array_almost_equal(norm_obs, np.array([0.0, 0.5, 1.0]))
    
    # Test division by zero avoidance
    obs_max_zero = np.array([0.0, 0.0, 0.0])
    norm_obs_zero = normalize_observation(obs, obs_min, obs_max_zero)
    assert not np.any(np.isnan(norm_obs_zero))
