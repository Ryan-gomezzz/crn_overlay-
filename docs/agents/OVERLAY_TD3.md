# Overlay TD3 (Overlay TD3)

**Overlay TD3** is a novel research-grade reinforcement learning agent specifically redesigned to optimize secondary user access inside cooperative Decode-and-Forward (DF) Overlay Cognitive Radio Networks.

---

## 1. Algorithmic Overview
Unlike the general underlay constraints of Underlay TD3, **Overlay TD3** integrates domain-specific wireless cooperative knowledge directly into its state representation, constraint critics, and exploration dynamics:

1.  **Enhanced Belief Representation**: Includes the previous slot's relay decoding success status ($D_{relay}$) and primary outage event ($O_{pu}$) into the sequence history inputs, expanding the sequence features to 8D.
2.  **Relay and QoS-Aware Constraints**: Formulates constraints based on exact primary receiver capacity metrics ($R_p \ge R_{threshold}$) rather than static co-channel interference power limits.
3.  **Adaptive Safety Optimization**: Dynamically learns multiplier penalties ($\lambda_{QoS}$ and $\lambda_{nrg}$) to enforce strict primary user spectrum license agreements.
4.  **Overlay Directional Exploration**: Safety gradients drive exploration towards increasing the primary rate while minimizing secondary power.

---

## 2. Technical Modifications & Divergence from Underlay TD3

### 2.1. 8D Sequence Input Space
The historical steps sequence is formulated as:
$$x_i = [s_i, a_{i-1}, D_{relay, i-1}, O_{pu, i-1}] \in \mathbb{R}^8$$
This memory encoding allows the agent's GRU Belief Encoder to know whether the Decode-and-Forward relay succeeded in the previous slot, allowing it to adapt its power allocation policy dynamically.

### 2.2. Dedicated Primary QoS Critic
Instead of bounding co-channel power at the PR node, the agent maintains Twin Critics for Primary Throughput ($Q^{QoS}_1, Q^{QoS}_2$). The outage constraint is formulated as:
$$R_{threshold} - Q^{QoS}(b_t, a_t) \le 0$$
which directly relates the primary user's QoS capacity metrics to the action penalty.

### 2.3. Dual Cooperative Directional Exploration
The exploration gradient vector $v_t$ active direction is modified to bias actions towards increasing the Primary QoS rate and decreasing secondary energy usage:
$$v_t = \lambda_{QoS} \cdot \nabla_a Q^{QoS}_1(b_t, a) - \lambda_{nrg} \cdot \nabla_a Q^{nrg}_1(b_t, a)$$
$$a_{explore} = \text{clip}\left( \pi_\theta(b_t) + \mathcal{N}(0, \sigma^2) + \eta \cdot v_t, \, 0.0, \, 1.0 \right)$$

---

## 3. Comparison with Underlay TD3

| Feature | Underlay TD3 | Overlay TD3 |
| :--- | :--- | :--- |
| **History Vector** | 6D ($s_t$, $a_{t-1}$) | 8D ($s_t$, $a_{t-1}$, $D_{relay, t-1}$, $O_{pu, t-1}$) |
| **Constraint Objective** | Interference Power ($I \le I_{limit}$) | Primary QoS Rate ($R_p \ge R_{threshold}$) |
| **Twin Critics** | Throughput, Interference, Energy | Throughput, QoS Rate, Energy |
| **Exploration Bias** | Shifting away from interference | Shifting towards higher primary rate |
