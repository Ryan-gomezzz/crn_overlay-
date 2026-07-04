# Algorithm Comparison Matrix

This document provides a comparative analysis of the three reinforcement learning agents supported in the repository.

---

## 1. Architectural Comparison Matrix

| Feature | TD3 | Underlay TD3 | Overlay TD3 |
| :--- | :--- | :--- | :--- |
| **Recurrent History** | No (Markovian State) | Yes (Temporal sequence) | Yes (Temporal sequence) |
| **History Length** | None ($L=0$) | $L=10$ steps | $L=10$ steps |
| **Input Features** | 4D instantaneous gains | 6D ($s_t$, $a_{t-1}$) | 8D ($s_t$, $a_{t-1}$, $D_{relay, t-1}$, $O_{pu, t-1}$) |
| **Critic Networks** | 2 (Twin Q-value) | 6 (3 Twin Critic Pairs) | 6 (3 Twin Critic Pairs) |
| **Critic Objectives** | Secondary Throughput | Throughput, Interference, Energy | Throughput, QoS Rate, Energy |
| **Constrained Optimization**| No (Scalar Penalty) | Yes (Lagrangian Multipliers) | Yes (Lagrangian Multipliers) |
| **Active Constraints** | None | $I_{PR} \le I_{limit}$ and $E_{SU} \le E_{limit}$ | $R_p \ge R_{threshold}$ and $E_{SU} \le E_{limit}$ |
| **Exploration Bias** | Standard Gaussian | Safety Gradient ($v_t \propto - \nabla Q^{inf}$) | Safety Gradient ($v_t \propto + \nabla Q^{QoS}$) |

---

## 2. Theoretical Progression

```
   ┌────────────────────────────────────────────────────────┐
   │                       TD3                      │
   │   - Flat Markovian state (instantaneous gains)         │
   │   - Scalar penalties (no Lagrangian adaptation)        │
   └───────────────────────────┬────────────────────────────┘
                               │ Introduce Recurrence & Limits
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │                       Underlay TD3                     │
   │   - GRU memory sequences (obs + action)                │
   │   - Lagrangian dual limits on interference power       │
   └───────────────────────────┬────────────────────────────┘
                               │ Adapt to Cooperative DF Relay
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │                       Overlay TD3                      │
   │   - 8D state history (adds relay decoding memory)      │
   │   - Direct PU QoS constraint (capacity limits)         │
   └────────────────────────────────────────────────────────┘
```

---

## 3. Trade-offs & Discussion

1.  **TD3 (Baseline)**:
    *   *Pros*: Extremely low computational overhead; fast training and sub-millisecond inference.
    *   *Cons*: Fails to adapt to fast fading channel fluctuations due to lack of sequential history; experiences high outage rates because penalties are static.
2.  **Underlay TD3**:
    *   *Pros*: Learns dynamic multipliers to restrict power bounds under path-loss; represents sequence transitions robustly.
    *   *Cons*: The co-channel interference constraint is a proxy metric and does not directly protect PU QoS from fading-induced outages.
3.  **Overlay TD3**:
    *   *Pros*: Direct enforcement of PU license agreements ($R_p \ge R_{threshold}$); memory of relay decoding successes allows adaptive power splitting.
    *   *Cons*: Slightly higher inference latency due to 8D recurrent inputs.
