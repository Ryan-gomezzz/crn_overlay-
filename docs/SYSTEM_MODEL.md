# System Model: Multi-Agent NOMA Cognitive Radio Network (CRN)

This document provides a comprehensive mathematical and architectural overview of the framework implemented in this repository. 

Our system models a **Multi-Agent Non-Orthogonal Multiple Access (NOMA) Cognitive Radio Network (CRN)** utilizing a shared Decode-and-Forward (DF) relay under imperfect Channel State Information (CSI). 

---

## 1. Network Architecture
The network consists of two overlapping systems sharing the same frequency spectrum:
*   **Primary Network**: A Primary Transmitter (PT) communicating with a Primary Receiver (PR). This network owns the spectrum license and requires strict Quality-of-Service (QoS) guarantees.
*   **Secondary Network**: $N$ Secondary Users (SU sources) communicating with a common SU Destination (SUd) via a shared Secondary Relay (SUr). The SUs opportunistically use the spectrum by operating in an "overlay" mode, assisting the Primary Network in exchange for spectrum access.

---

## 2. Two-Timeslot Protocol (NOMA & SIC)
Communication occurs over a block-fading duration divided into two equal time slots $T/2$.

### Slot 1: Multiple Access Broadcast
*   The Primary Transmitter (PT) broadcasts its signal $x_p$ with power $P_{pt}$.
*   All $N$ Secondary Users simultaneously transmit their signals $x_{s,i}$ with power $P_{s,i}$ using **NOMA** in the power domain.
*   **Relay Reception**: The shared DF Relay receives a superposed signal comprising the primary message and all secondary messages, corrupted by Additive White Gaussian Noise (AWGN) $N_0$.
*   **Decoding (SIC)**: The Relay decodes the PU signal first (treating SUs as interference). Once successfully decoded and subtracted, it decodes the SU signals using **Successive Interference Cancellation (SIC)**, decoding users in descending order of their channel strengths $|h_{sr,i}|^2$.

### Slot 2: Superposition Forwarding
*   The Relay forwards a superposed mixture of the PU and SU signals. It allocates a power splitting fraction $\alpha \in [0,1]$ for the PU and $(1-\alpha)$ for the SUs.
*   The PT may also transmit new data (which acts as interference to the SU Destination).
*   **Destination Reception**: The SU Destination receives the relayed signal, decodes the PU signal, subtracts it, and then decodes the superposed SU signal.
*   **Interference at PR**: During Slot 1, all SU transmissions cause interference at the PR. During Slot 2, the Relay's SU-allocated power $(1-\alpha)P_{rel}$ causes interference at the PR.

---

## 3. Physical Channel & Imperfect CSI
The wireless channels undergo **Rayleigh fading** (Nakagami-m with $m=1$, assuming Non-Line-of-Sight) combined with distance-dependent path loss. 
The true channel power gain between any node $a$ and $b$ is:
$$|h_{ab}|^2 = \frac{|g|^2}{d_{ab}^\gamma}$$
where $g \sim \mathcal{CN}(0, 1)$ is the complex Rayleigh fading coefficient, $d$ is distance, and $\gamma$ is the path-loss exponent.

### Imperfect CSI Estimation
In practice, instantaneous global CSI is impossible to acquire perfectly. The RL agents observe an **Estimated Channel** $\hat{g}$, modeled via a perturbation error variance $\tau$:
$$\hat{g} = \sqrt{1 - \tau^2} g + \tau e$$
where $e \sim \mathcal{CN}(0, 1)$ is the estimation error.
*   The **Environment Physics** (SINR, outage, rewards) are rigidly calculated using the *true* channel $g$.
*   The **RL Agents** only receive the *noisy estimate* $\hat{g}$, forcing them to learn robust policies under partial observability.

---

## 4. Multi-Agent Reinforcement Learning (MATD3)
The system uses **Centralized Training with Decentralized Execution (CTDE)** via the MATD3 algorithm.

*   **Observation Space (per SU agent)**: An 8-Dimensional vector of the *estimated* channel power gains (converted to dB and normalized). 
    $$s_t^i = [\hat{h}_{sr_i}, \hat{h}_{sp_i}, \hat{h}_{sd_i}, \hat{h}_{pp}, \hat{h}_{pr}, \hat{h}_{pd}, \hat{h}_{rd}, \hat{h}_{rp}]$$
*   **Recurrent Encoding**: Because the observations are noisy and partial (Imperfect CSI over time-varying fading), each agent passes its last $L=10$ observations through a **Gated Recurrent Unit (GRU)** to form a latent belief state.
*   **Action Space**: The joint action space has $N+2$ continuous dimensions bound to $[0, 1]$:
    *   $N$ values: Normalized Transmit powers for each SU source.
    *   1 value: Normalized Transmit power for the shared Relay.
    *   1 value: Power splitting factor $\alpha$ used by the Relay.
*   **Reward Function**: Maximize the sum-rate of the Secondary Users: $R_{SU} = \sum_{i=1}^N \frac{1}{2} \log_2(1 + \gamma_{e2e,i})$.
*   **Constraints**:
    *   The interference caused at the PR is kept below the threshold ($I_{PR} \le I_{th}$) by a penalty term in the environment reward (`camo_td3.penalty_coef_inf`).
    *   Adaptive **Lagrangian multipliers** (learned in log/softplus space) additionally enforce the Primary-User QoS-rate and per-agent energy constraints, shaping the actor objective via the dedicated QoS and energy critics.

---

## 4a. Bit Error Rate (Per-Hop Decode-and-Forward)

BER is evaluated with **per-hop bit-level decoding** rather than a single
end-to-end SINR mapping, so that Decode-and-Forward error propagation is
captured explicitly.

For each secondary user *i*, BPSK is decoded **twice**:

1.  **Hop 1 (SU → relay).** Source bits are detected at the relay under the
    post-SIC per-user SINR $\gamma_{sr,i}$ (which already carries the NOMA
    residual interference from weaker, not-yet-cancelled users). This yields the
    relay decision — the **hop-1 / relay BER**, $P_{1,i}$.
2.  **DF forwarding.** The relay re-encodes and forwards its (possibly
    erroneous) decisions.
3.  **Hop 2 (relay → destination).** The forwarded bits are detected at the
    destination under $\gamma_{rd}$, giving error probability $P_2$.

A bit is in error end-to-end iff it is flipped an odd number of times, so the
**end-to-end BER** is

$$P_{e2e,i} = P_{1,i} + P_2 - 2\,P_{1,i}\,P_2 \;\ge\; \max(P_{1,i}, P_2),$$

with per-link BPSK BER $P = \tfrac12\,\mathrm{erfc}(\sqrt{\gamma})$. The primary
user is decoded via **selection combining** of its single-hop direct link
(PT → PR) and its DF-relayed link, taking whichever path yields the lower BER.

Both the closed-form values above and a **bit-level Monte-Carlo** simulation
(random bits transmitted, decoded at the relay, re-encoded, decoded at the
destination, then compared to the source) are recorded. Reports plot the
Monte-Carlo points against the single-hop BPSK theory line: hop-1 points fall on
the line, while end-to-end points lie above it — the DF error-propagation
penalty. Implementations: `simulator/utils.py` (`ber_bpsk_theory`,
`ber_bpsk_montecarlo`, `df_ber_theory`, `simulate_df_ber_montecarlo`) and the
per-step values exposed by `simulator/noma_overlay_model.py`.

---

## 5. Configuration Parameters (`configs/config.yaml`)

Every physical and algorithmic aspect of the system is parameterized and highly configurable. 

### Algorithm & Simulation
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `algorithm.name` | `MATD3` | The RL algorithm architecture used. |
| `simulation.seed` | `42` | Random seed for reproducibility. |
| `simulation.time_steps_per_episode` | `100` | Number of communication rounds per trajectory. |

### Network Coordinates & Power Limits
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `network.pt_coords` | `[0.0, 0.0]` | Primary Transmitter (PT) location (meters). |
| `network.pr_coords` | `[100.0, 0.0]` | Primary Receiver (PR) location. |
| `network.sus_coords` | `[10.0, 20.0]` | (Legacy/Fallback) Base coordinate for single SU generation. |
| `network.sur_coords` | `[50.0, 10.0]` | (Legacy/Fallback) Relay coordinate. |
| `network.sud_coords` | `[90.0, 20.0]` | (Legacy/Fallback) Destination coordinate. |
| `network.p_primary` | `30.0` | PT transmission power in dBm. |
| `network.p_max_su` | `20.0` | Maximum hardware transmit power limit for all SUs and the Relay (dBm). |

### Multi-User NOMA Layout
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `multi_user.num_su` | `3` | Number of Secondary Users ($N$). |
| `multi_user.su_coords` | `[[50, 200], ...]` | 2D List of [x, y] coordinates for each SU source. |
| `multi_user.sud_coords` | `[50.0, 180.0]` | Coordinate for the SU Destination. |
| `multi_user.sur_coords` | `[50.0, 190.0]` | Coordinate for the shared SU Relay. |
| `multi_user.interference_threshold_dbm` | `-50.0` | The maximum tolerable interference $I_{th}$ at the PR. |

### Channel Physics
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `channel.path_loss_exponent` | `3.5` | Path loss exponent $\gamma$ (typical for urban NLOS). |
| `channel.noise_power_dbm` | `-114.0` | AWGN noise floor $N_0$ at all receivers. |
| `channel.fading_type` | `rayleigh` | Fast fading distribution (Nakagami $m=1$). |
| `channel.csi_error_variance` | `0.1` | The estimation noise variance $\tau$ defining the Imperfect CSI. |

### MATD3 & Training Hyperparameters
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `training.gamma` | `0.99` | MDP discount factor. |
| `training.tau` | `0.005` | Soft update coefficient for target networks. |
| `training.policy_delay` | `2` | Frequency of delayed policy updates (TD3 mechanic). |
| `training.exploration_noise` | `0.1` | Gaussian noise added to action for exploration. |
| `training.policy_noise` | `0.2` | Gaussian noise added to target actions (smoothing). |
| `training.noise_clip` | `0.5` | Clip bound for the policy smoothing noise. |
| `training.batch_size` | `128` | Number of temporal sequences sampled per update. |
| `training.lr_actor` | `0.0003` | Learning rate for the Policy networks. |
| `training.lr_critic` | `0.0003` | Learning rate for the Q-value networks. |
| `training.buffer_size` | `100000` | Capacity of the episodic replay buffer. |
| `training.total_steps` | `100000` | Total environment steps over training (÷ `time_steps_per_episode` ⇒ 1000 episodes at the default 100 steps). |
| `training.start_steps` | `1000` | Pure uniform random exploration steps before training. |

### Agent Multi-Objective & Constraints (camo_td3)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `camo_td3.history_length` | `10` | The sequence length $L$ fed into the GRU encoder. |
| `camo_td3.decoding_threshold` | `0.1` | SINR target for relay decoding success criteria. |
| `camo_td3.pu_rate_threshold` | `0.5` | Minimum PU rate requirement. |
| `camo_td3.interference_limit_dbm` | `-80.0` | Objective-level interference constraint penalty threshold. |
| `camo_td3.energy_limit_watts` | `0.1` | Objective-level energy consumption constraint threshold. |
| `camo_td3.penalty_coef_inf` | `1.0e10` | Scalar steepness of the interference constraint boundary. |
| `camo_td3.penalty_coef_qos` | `10.0` | Scalar steepness of the PU QoS constraint boundary. |
| `camo_td3.penalty_coef_nrg` | `1.0` | Scalar steepness of the Energy constraint boundary. |
| `camo_td3.lr_lambda` | `0.001` | Learning rate for the adaptive Lagrangian multipliers. |
| `camo_td3.lambda_inf_init` | `1.0` | Initial dual multiplier $\lambda$ for interference limit. |
| `camo_td3.lambda_qos_init` | `1.0` | Initial dual multiplier $\lambda$ for PU QoS limit. |
| `camo_td3.lambda_nrg_init` | `0.001` | Initial dual multiplier $\lambda$ for Energy limit. |
| `camo_td3.lambda_clamp_max` | `200.0` | Maximum absolute bound for dynamic Lagrangian multipliers. |
| `camo_td3.eta_explore_init` | `0.05` | Initial exploration step size on the safety gradient manifold. |
| `camo_td3.eta_explore_decay` | `0.9999` | Exponential decay per step for safety gradient exploration. |

### Evaluation & Logging
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `logging.tensorboard_enabled` | `true` | Export loss and reward curves to Tensorboard format. |
| `logging.log_interval` | `10` | Frequency of logging (in episodes). |
| `logging.log_dir` | `experiments/runs/` | Destination directory for log files. |
| `evaluation.eval_episodes` | `5` | Number of test episodes per evaluation. |
| `evaluation.eval_interval` | `500` | Frequency of model evaluation during training (in environment steps). |
| `evaluation.save_dir` | `experiments/checkpoints/` | Destination directory for best model states. |
