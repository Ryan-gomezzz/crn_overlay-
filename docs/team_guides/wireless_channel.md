# Wireless Channel Module Guide
**Assignee:** Sneha

## Objectives
Implement realistic wireless channel models including Rayleigh fading, path loss, distance models, and noise.

## Files to modify
- `simulator/channels.py`
- `simulator/propagation.py`
- `tests/test_channels.py`

## Expected Classes and Functions
- `class WirelessChannel(Protocol)`
- `class RayleighFading(WirelessChannel)`
- `def calculate_path_loss(distance: float) -> float`

## APIs
- The channel objects should accept parameters like distance, frequency, and output channel gains.

## Coding Standards
- Google docstrings, type hints (Python 3.11+).

## Testing Checklist
- Test channel gain variance matches expected Rayleigh distribution.
- Test path loss decreases monotonically with distance.
