# Research Notes & Progression

This document outlines the theoretical evolution of the RL agents in this repository and lists guidelines for academic publications.

---

## 1. Evolution Pathway

The repository traces a progression of three paradigms in spectrum-sharing optimization:

```
                  ┌──────────────────────────────┐
                  │              T3              │
                  │ (Standard flat, no safety)   │
                  └──────────────┬───────────────┘
                                 │ Recurrence & Limits
                                 ▼
                  ┌──────────────────────────────┐
                  │         Underlay TD3         │
                  │ (GRU sequence, interference) │
                  └──────────────┬───────────────┘
                                 │ Overlay Relaying
                                 ▼
                  ┌──────────────────────────────┐
                  │          Overlay TD3         │
                  │ (8D state, direct QoS limit) │
                  └──────────────────────────────┘
```

1.  **Baseline T3**: Validates that flat, standard policy gradient networks struggle under Rayleigh channel fluctuations. The lack of memory results in frequent outages since fading history is neglected.
2.  **Underlay TD3 (Original CAMO-TD3)**: Integrates GRU sequence history to handle partial observability. Evaluates soft constraints w.r.t physical limitations (interference power $I_{PR} \le I_{limit}$).
3.  **Overlay TD3**: Novel contribution. Rather than proxy bounds, it implements direct primary Quality of Service rate tracking ($R_p \ge R_{threshold}$) using a dedicated QoS Critic pair, feeding previous relay decoding success status back to the belief encoder to coordinate cooperative power splits.

---

## 2. Topics for Academic Publications

If preparing a submission to IEEE transactions or letters, we recommend framing the research under the following headings:

### 2.1. Dynamic Fading Partial Observability
*   Show how recurrent state representation ($L=10$) outperforms memoryless baselines (T3) under rapid block fading.
*   Present BER and SU throughput improvements when memory is active.

### 2.2. QoS Guarantee in Cooperative Networks
*   Evaluate how direct QoS capacity critics ($Q^{QoS}$) reduce primary user outage rate compared to general interference power limit critics ($Q^{inf}$).
*   Plot comparative convergence showing how softplus-parameterized multipliers $\lambda_{QoS}$ adapt dynamically to fading severity.
