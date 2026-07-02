"""
Interference modeling.
Assignee: Shreya
"""

# TODO (Shreya): Implement interference constraints and calculations.
# Expected Inputs: transmission powers, channel gains.
# Expected Outputs: interference power at receivers.
# Reference: docs/team_guides/relay_module.md

def compute_interference(transmit_powers, channel_gains):
    """
    transmit_powers: list of powers [P1, P2, ...]
    channel_gains: list of gains [h1, h2, ...]
    """
    if len(transmit_powers) != len(channel_gains):
        raise ValueError("Mismatch in powers and channel gains")

    total_interference = 0.0

    for p, h in zip(transmit_powers, channel_gains):
        total_interference += p * h

    return total_interference