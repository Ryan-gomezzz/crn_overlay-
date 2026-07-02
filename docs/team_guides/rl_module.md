# Reinforcement Learning Module Guide
**Assignee:** Aditya

## Objectives
Implement and optimize RL training, evaluation, and comparative benchmarking scripts for the three supported agents:
*   **TD3** (Twin Delayed DDPG Baseline)
*   **Underlay TD3** (Adaption of original Underlay TD3 limits)
*   **Overlay TD3** (Relay and QoS-aware cooperative agent)

## Files to modify/maintain
- `agents/models.py` — GRU encoders, actors, and twin critics.
- `agents/buffers.py` — Flat, episodic, and overlay sequence replay buffers.
- `agents/train_td3.py` — Unified training loops and Lagrangian dual updates.
- `agents/evaluate.py` — Policy loader and evaluator.
- `agents/benchmark.py` — Comparative evaluation and plots generator.

## Testing Checklist
- Ensure all 4 unit tests in `tests/test_camo.py` pass.
- Verify checkpoints save and load correctly under `experiments/checkpoints/`.
- Verify comparative plots are correctly generated inside `plots/`.
