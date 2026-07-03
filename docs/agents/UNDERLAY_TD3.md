# Underlay TD3 (Original CAMO-TD3 adaptation)

**Underlay TD3** is a faithful implementation of the original CAMO-TD3 algorithm adapted to run in the Overlay Cognitive Radio Network environment.

---

## 1. Algorithmic Overview
Underlay TD3 addresses partially observable constrained Markov decision processes (POMDPs) by combining recurrent state representation with Lagrangian constrained optimization:

1.  **Recurrent State Modeling**: Feeds sequential histories of observations and past actions of length $L=10$ into a **GRU Belief Encoder** to generate a continuous belief vector $b_t \in \mathbb{R}^{64}$.
2.  **Episodic Sequence Sampling**: Utilizes a custom **Sequence Replay Buffer** that maintains contiguous transitions, correctly handling boundaries and zero padding.
3.  **Multi-Objective Twin Critics**: Tracks three independent Twin Critic pairs (6 networks total):
    *   Throughput Twin Critics ($Q^{thr}_1, Q^{thr}_2$)
    *   Interference Twin Critics ($Q^{inf}_1, Q^{inf}_2$)
    *   Energy Twin Critics ($Q^{nrg}_1, Q^{nrg}_2$)
4.  **Adaptive Lagrangian Constraints**: Dynamically updates learnable multipliers $\lambda_{inf}$ and $\lambda_{nrg}$ to restrict co-channel interference and energy consumption within configurable safety thresholds ($I_{limit}$ and $E_{limit}$).
5.  **Directional Exploration Bias**: Computes safety exploration gradients from the active constraint networks to actively steer exploration steps away from boundary violations.

---

## 2. Mathematical Formulation

### 2.1. GRU Belief Encoder
Concatenates current observations and previous actions across a temporal sequence window:
$$x_i = [s_i, a_{i-1}] \in \mathbb{R}^6$$
$$b_t = GRU(x_{t-L+1:t}) \in \mathbb{R}^{64}$$

### 2.2. Lagrangian Updates
The multipliers are softplus-parameterized and updated via gradient ascent:
$$\text{Loss}(\lambda_{inf}) = - \lambda_{inf} \cdot \left( Q^{inf}_1(b_t, a_t) - I_{limit} \right)$$
$$\text{Loss}(\lambda_{nrg}) = - \lambda_{nrg} \cdot \left( Q^{nrg}_1(b_t, a_t) - E_{limit} \right)$$

### 2.3. Safety Directional Exploration
The directional bias $v_t$ shifts exploration in direction of lower constraint violation:
$$v_t = - \lambda_{inf} \cdot \nabla_a Q^{inf}_1(b_t, a) - \lambda_{nrg} \cdot \nabla_a Q^{nrg}_1(b_t, a)$$
$$a_{explore} = \text{clip}\left( \pi_\theta(b_t) + \mathcal{N}(0, \sigma^2) + \eta \cdot v_t, \, 0.0, \, 1.0 \right)$$

---

## 3. Intended Use & Application
*   **Purpose**: Serves as a standard baseline showing the performance of CAMO-TD3 when operating under traditional physical underlay power limitations.
*   **Usage Context**: Should be used when co-channel interference power limit bounds at the Primary Receiver (PR) are the main safety metric, and the agent does not receive cooperative state metrics from the relay node.
