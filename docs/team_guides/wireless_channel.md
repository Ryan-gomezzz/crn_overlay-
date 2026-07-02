# Wireless Channel Module Guide
**Assignee:** Sneha

## Objectives
Implement Rayleigh channel gain models, path loss, distance attenuation models, and complex noise.

## Files to modify/maintain
- `simulator/channels.py`
- `simulator/propagation.py`
- `tests/test_camo.py`

## Expected Classes and Functions
- `class RayleighFading`
- `def calculate_path_loss(distance: float, path_loss_exponent: float) -> float`

## Testing Checklist
- Run Rayleigh fading generator and verify envelope statistical gains.
- Verify path loss decreases monotonically as distance increases.
- Run `tests/test_camo.py` to check physical layers math constraints.
