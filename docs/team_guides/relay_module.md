# Relay and Communication Module Guide
**Assignee:** Shreya

## Objectives
Implement Decode-and-Forward (DF) relay logic, time-slot management, received powers, interference, SINR, capacities, and bit error rates.

## Files to modify/maintain
- `simulator/relay.py`
- `simulator/interference.py`
- `simulator/metrics.py`
- `simulator/overlay_model.py`
- `tests/test_camo.py`

## Expected Classes and Functions
- `class DecodeAndForward`
- `def calculate_sinr(signal: float, interference: float, noise: float) -> float`
- `def calculate_ber(sinr: float) -> float`
- `def calculate_capacity(sinr: float, bandwidth: float) -> float`

## Testing Checklist
- Test DF relay condition (SINR > threshold).
- Test co-channel interference aggregation.
- Run `tests/test_camo.py` to verify physical layer math calculations.
