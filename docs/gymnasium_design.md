# Gymnasium Environment Design

The Gymnasium environment (`envs/crn_env.py`) wraps the physical communication simulator to present a standardized reinforcement learning interface.

---

## 1. Action Space
The action space is a continuous Box space of size 2:
$$a_t = [P_{s1}, \, \beta] \in [0, 1]^2$$
*   **$P_{s1}$**: Transmit power scaling fraction at SUs in Time Slot 1, defining the transmit power as:
    $$P_{transmit} = P_{s1} \cdot P_{max, SU}$$
*   **$\beta$**: Relay power splitting fraction in Time Slot 2. The relay allocates:
    *   $\beta \cdot P_{rel}$ to forward the primary message.
    *   $(1 - \beta) \cdot P_{rel}$ to forward the secondary message.

---

## 2. Observation Space
The instantaneous observation space is a continuous Box space of size 4 representing Rayleigh envelope channel gains:
$$s_t = [h_{PT, PR}, \, h_{SUs, SUR}, \, h_{SUR, SUd}, \, h_{SUs, PR}] \in [0, \infty)^4$$

---

## 3. Sequential History Tracking
To address partial observability under fading channels, the environment maintains history deques of length $L=10$:
*   `obs_history`: past observations of shape `(L, 4)`
*   `act_history`: past actions of shape `(L, 2)`
*   `dec_history`: past relay decoding success outcomes of shape `(L, 1)`
*   `out_history`: past primary user outage event outcomes of shape `(L, 1)`

---

## 4. Multi-Objective Rewards Formulation
The step returns individual objective feedback returned inside `info`:
*   **Secondary Throughput Reward ($r_{thr}$)**: Secondary capacity rate $R_s$.
*   **Interference Reward ($r_{inf}$)**: Negative total co-channel interference power at the PR node (used by Underlay TD3).
*   **Primary Throughput Reward ($r_{qos}$)**: Primary capacity rate $R_p$ (used by Overlay TD3 as QoS constraint).
*   **Energy Reward ($r_{nrg}$)**: Negative average power consumed over the slots ($0.5 P_{s1} + 0.5 P_{rel}$).
*   **Standard Scalar Reward ($r_{scalar}$)**: Used by the baseline T3 agent, combining goals via fixed penalty coefficients:
    $$r_{scalar} = R_s - \alpha_{inf} \max(0, I_{PR} - I_{limit}) - \alpha_{nrg} \max(0, E - E_{limit})$$
