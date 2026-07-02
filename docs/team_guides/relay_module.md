# Relay and Communication Module Guide
**Assignee:** Shreya

## Objectives
Implement Decode-and-Forward (DF) relay logic, time-slot management, signal equations, SINR, and interference.

## Files to modify
- `simulator/relay.py`
- `simulator/interference.py`
- `simulator/metrics.py`
- `tests/test_relay.py`

## Expected Classes and Functions
- `class RelayProtocol(Protocol)`
- `class DecodeAndForward(RelayProtocol)`
- `def calculate_sinr(signal: float, interference: float, noise: float) -> float`

## APIs
- Accept channel gains from Sneha's module and compute final received signals.

## Testing Checklist
- Test DF relay condition (SNR > threshold).
- Test interference aggregation.
