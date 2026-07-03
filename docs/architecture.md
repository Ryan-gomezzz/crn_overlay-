# System Architecture

The Overlay Cognitive Radio Network RL framework consists of modular components designed to support multi-agent training, testing, and comparative benchmarking.

---

## 1. Unified Architecture Block Diagram

```
                          ┌───────────────────────┐
                          │     Config / YAML     │
                          └───────────┬───────────┘
                                      │ Parse
                                      ▼
┌──────────────────┐      ┌───────────────────────┐
│     Agent API    │◄────►│ Gymnasium Environment │
│ (T3/Underlay/    │      │   (envs/crn_env.py)   │
│  Overlay TD3)    │      └───────────┬───────────┘
└────────┬─────────┘                  │
         │                            ▼
         │                ┌───────────────────────┐
         │                │   Physical CRN Model  │
         │                │ (simulator/overlay_model)
         │                └───────────┬───────────┘
         │                            │
         ▼                            ▼
┌──────────────────┐      ┌───────────────────────┐
│  Replay Buffers  │      │  Propagation, Fading, │
│  (Flat / Seq)    │      │  Relaying, SINR, BER  │
└──────────────────┘      └───────────────────────┘
```

---

## 2. Common Infrastructure

All three agents utilize a unified communication pipeline:
1.  **Environment (`envs/crn_env.py`)**: A Gymnasium wrapper mapping physical calculations to standard state, reward, step, and reset functions.
2.  **Configurations Manager**: Automatically parses algorithm overrides from `configs/experiment.yaml` into the master configuration layout.
3.  **Physical Simulators**: Performs Rayleigh gains, distance path loss, and DF relay decoding in a single unified step.
4.  **Logging**: Episode statistics and optimization summaries are output to TensorBoard via a single SummaryWriter.

---

## 3. Agent Architecture Differences

### 3.1. T3 (Baseline)
*   **State Space**: Flat 4D channel gains.
*   **Buffer**: Flat Transition Buffer.
*   **Critics**: 1 Twin Critic pair assessing secondary throughput.
*   **Constraints**: None (uses static scalar penalty terms inside the scalar reward).

### 3.2. Underlay TD3
*   **State Space**: Recurrent 6D history embeddings ($s_t, a_{t-1}$).
*   **Buffer**: Episodic Sequential Replay Buffer.
*   **Critics**: 3 Twin Critic pairs (throughput, co-channel interference power at PR, energy).
*   **Constraints**: Underlay Lagrangian soft limits ($I_{limit}$ and $E_{limit}$).

### 3.3. Overlay TD3
*   **State Space**: Recurrent 8D history embeddings ($s_t, a_{t-1}, D_{relay, t-1}, O_{pu, t-1}$).
*   **Buffer**: Episodic Sequential Replay Buffer with extended transition attributes.
*   **Critics**: 3 Twin Critic pairs (secondary throughput, primary QoS capacity rate, energy).
*   **Constraints**: Direct Overlay QoS limit ($R_p \ge R_{threshold}$) and energy limit.
