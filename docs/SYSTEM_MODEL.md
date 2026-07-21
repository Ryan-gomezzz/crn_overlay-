# System Model: Multi-Agent NOMA Cognitive Radio Network (CRN)

This document provides a comprehensive mathematical and architectural overview of the framework implemented in this repository. 

Our system models a **Multi-Agent Non-Orthogonal Multiple Access (NOMA) Cognitive Radio Network (CRN)** utilizing a shared Decode-and-Forward (DF) relay under imperfect Channel State Information (CSI). 

---

## 1. Network Architecture
The network consists of two overlapping systems sharing the same frequency spectrum:
*   **Primary Network**: A Primary Transmitter (PT) communicating with a Primary Receiver (PR). This network owns the spectrum license and requires strict Quality-of-Service (QoS) guarantees.
*   **Secondary Network**: $N$ Secondary Users sharing a common SU Destination (SUd). **One of the $N$ SUs — SU$_N$ — is itself the half-duplex Decode-and-Forward relay; there is no separate relay node.** The remaining $M = N-1$ SUs are sources. The relay-SU carries its **own** traffic in addition to forwarding the sources', and the secondary network operates in "overlay" mode, assisting the Primary Network in exchange for spectrum access.

---

## 2. Two-Timeslot Protocol (NOMA & SIC)
Communication occurs over a block-fading duration divided into two equal time slots $T/2$.

### Slot 1: Multiple Access Broadcast
*   The Primary Transmitter (PT) broadcasts its signal $x_p$ with power $P_{pt}$.
*   The $M = N-1$ SU **sources** simultaneously transmit their signals $x_{s,i}$ with power $P_{s,i}$ using **NOMA** in the power domain. The relay-SU is **half-duplex**: it listens in this slot and does not transmit.
*   **Relay Reception**: The relay-SU receives a superposed signal comprising the primary message and all $M$ source messages, corrupted by AWGN $N_0$.
*   **Decoding (SIC)**: The relay-SU decodes the PU signal first (treating the sources as interference). Once decoded and subtracted, it decodes the source signals using **Successive Interference Cancellation (SIC)**, in descending order of $|h_{sr,i}|^2$.

### Slot 2: Superposition Forwarding
*   The relay-SU transmits a power-domain superposition of **three** things: the PU's message (fraction $\alpha$ of its power $P_{rel}$), its **own** data, and the $M$ decoded source messages. Within the secondary share $(1-\alpha)P_{rel}$, a fraction `own_share` goes to its own data and the remainder is split across the forwarded streams by a fixed geometric allocation ($\beta_k \propto \mu^k$ in the relay's SIC order).
*   The PT may also transmit new data (which acts as interference to the SU Destination).
*   **Destination Reception**: The SU Destination performs SIC over the superposed streams in **descending received power**, cancelling each detected stream before decoding the next.
*   **Interference at PR**: During Slot 1, the $M$ source transmissions cause interference at the PR. During Slot 2, the relay-SU's secondary payload $(1-\alpha)P_{rel}$ (own + forwarded) causes interference at the PR. The relayed PU signal is *not* interference.

### Rates
Each **source** $i$ is Decode-and-Forward and therefore limited by its weaker hop,
$\gamma_{e2e,i} = \min(\gamma_{sr,i}, \gamma_{fwd,i})$, whereas the **relay-SU's own data**
traverses a single hop (slot 2 only):

$$R_i = \tfrac12\log_2(1+\gamma_{e2e,i}),\qquad R_{\text{relay}} = \tfrac12\log_2(1+\gamma_{\text{own}}),\qquad R_{SU} = \sum_{i=1}^{M} R_i + R_{\text{relay}}.$$

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

*   **Observation Space (per SU agent)**: An 8-Dimensional vector of the *estimated* channel power gains (converted to dB and normalized). For a **source** $i$:
    $$s_t^i = [\hat{h}_{sr_i}, \hat{h}_{sp_i}, \hat{h}_{sd_i}, \hat{h}_{pp}, \hat{h}_{pr}, \hat{h}_{pd}, \hat{h}_{rd}, \hat{h}_{rp}]$$
    The **relay-SU**'s row replaces the first three with its own links $[\hat{h}_{rd}, \hat{h}_{rp}, \hat{h}_{pr}]$ (its transmit links to the destination and PR, and the PT link it must decode through).
*   **Recurrent Encoding**: Because the observations are noisy and partial (Imperfect CSI over time-varying fading), each agent passes its last $L=10$ observations through a **Gated Recurrent Unit (GRU)** to form a latent belief state.
*   **Action Space**: The joint action space has $N+2$ continuous dimensions bound to $[0, 1]$ (with $M = N-1$ sources):
    *   $M$ values: Normalized transmit powers for each SU **source**.
    *   1 value: Normalized transmit power of the **relay-SU**.
    *   1 value: Power splitting factor $\alpha$ (relay-SU power given to the PU).
    *   1 value: `own_share` — fraction of the relay-SU's secondary power spent on its **own** data (remainder forwards the sources).
*   **Agents**: $N$ GRU belief encoders (one per SU, including the relay-SU). The $M$ source actors each emit their transmit power; a centralized head consumes **all** $N$ beliefs and emits the relay-SU's $[P_{rel}, \alpha, \texttt{own\_share}]$.
*   **Reward Function**: Maximize the secondary sum-rate $R_{SU}$ defined in §2 (sources' end-to-end rates plus the relay-SU's own rate).
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

### Waveform-level BER with imperfect SIC

The per-hop model above assumes the SINRs already reflect *perfect* SIC. A
higher-fidelity **waveform Monte-Carlo** (`simulator/ber_waveform.py`,
`simulate_waveform_df_noma_ber`) removes that assumption. It forms the actual
complex-baseband superposed signals from the true complex channels
$h = g\sqrt{PL}$ and runs **real SIC**, subtracting the *detected* symbols:

- **Hop 1 (relay).** $y_r = \sqrt{P_p}h_{pr}x_p + \sum_i \sqrt{P_{s,i}}h_{sr,i}x_{s,i} + n_r$.
  The relay detects and cancels the PU, then the SUs in descending
  $|h_{sr,i}|^2$ order. A wrong decision is cancelled with the wrong symbol, so
  its residual leaks into later users — producing the characteristic **NOMA/SIC
  error floor**.
- **Hop 2 (destination).** The relay-SU re-encodes and transmits the
  superposition $x_r = \sqrt{\alpha P_r}\,\hat x_p + \sqrt{P_{\text{own}}}\,x_{\text{own}} + \sum_i \sqrt{P_{fwd,i}}\,\hat x_{s,i}$
  on the common channel $h_{rd}$. The destination performs SIC in descending
  received power, cancelling each *detected* stream. Source bits are compared to
  the *original* source bits (so hop-1 errors propagate); the relay-SU's own data
  is single-hop.
- **Primary user** uses selection combining between its direct link and the
  DF-relayed link (the relay's decoded PU forwarded on $h_{rp}$), with the
  branch chosen by the physics (higher SINR).

Both hops are genuine power-domain NOMA channels, so the imperfect-SIC error
floor appears at each. A practically important consequence the waveform exposes
and the Gaussian-SINR model cannot: when the stream being decoded does **not**
dominate the not-yet-cancelled streams, the residual is *discrete BPSK
interference*, which is markedly more damaging than an equal-power Gaussian
approximation — the measured BER then far exceeds
$\tfrac12\mathrm{erfc}(\sqrt{\gamma})$ evaluated at the same SINR. Reports overlay
the waveform points (bold markers) on the perfect-SIC Monte-Carlo and BPSK-theory
reference; the gap is the imperfect-SIC penalty.

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
| `multi_user.num_su` | `3` | **Total** Secondary Users $N$, **including the relay-SU**. |
| `multi_user.su_coords` | `[[50, 200], [52, 200]]` | Coordinates of the $N-1$ SU **sources**. |
| `multi_user.sur_coords` | `[50.0, 190.0]` | Coordinate of SU$_N$, **the relay-SU** (one of the $N$ SUs, not a separate node). |
| `multi_user.sud_coords` | `[50.0, 180.0]` | Coordinate for the SU Destination. |
| `multi_user.relay_fwd_mu` | `0.5` | Geometric ratio $\mu$ splitting the relay's forwarding power across streams. |
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
