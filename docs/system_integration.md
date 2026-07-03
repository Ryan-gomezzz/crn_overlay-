# System Integration Guide

This document details the unified training, evaluation, and comparative benchmarking integration pipelines in the Overlay Cognitive Radio Network RL framework.

---

## 1. Unified Training Pipeline
All three algorithms share a common training entry point:
```bash
python main.py
```

### 1.1. Execution Flow
1.  **Configuration Parsing**: `main.py` parses coordinates, coordinate limits, and algorithm choice (`T3`, `UNDERLAY_TD3`, or `OVERLAY_TD3`) from `configs/config.yaml`.
2.  **Environment Initialization**: Starts the Gymnasium environment and sets compute seeds.
3.  **Agent Instantiation**: Initializes `TD3Agent` which automatically standardizes name switches and builds appropriate encoders, actors, and critics.
4.  **Interaction Loop**:
    *   Saves transitions to the replay buffer.
    *   Samples standard transitions (T3) or temporal sequences (Underlay/Overlay TD3).
    *   Triggers dual constraints multipliers updates and safety directional explore steps.
5.  **Periodic Evaluation & Checkpoints**: Runs deterministic evaluation and saves the best model checkpoint to `experiments/checkpoints/` if performance improves.

---

## 2. Standalone Policy Evaluation
To load and evaluate a trained policy checkpoint deterministically:
```bash
python agents/evaluate.py
```
This script reads the selected algorithm name from `configs/config.yaml`, loads the corresponding checkpoint, and displays final average reward, SU rate, PU rate, outage rate, BER, and constraint satisfaction scores.

---

## 3. Comparative Benchmarking
To perform a complete side-by-side comparison under identical conditions:
```bash
python agents/benchmark.py
```
This utility:
1.  Runs training loops sequentially for all three agents.
2.  Measures training clock-times and inference pass latencies.
3.  Aggregates tracking histories.
4.  Generates and saves performance charts to the `plots/` directory.
