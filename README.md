<div align="center">

# ⚡ Multi-Agent NOMA Overlay Cognitive Radio Networks with Deep RL

### A research framework for intelligent power allocation in a NOMA overlay CRN, trained with Multi-Agent TD3 (MATD3)

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Environment-green?style=for-the-badge)
![Research](https://img.shields.io/badge/Research-IEEE-blueviolet?style=for-the-badge)

</div>

---

## 📖 Overview

This repository implements a **Multi-Agent Non-Orthogonal Multiple Access (NOMA) Cognitive
Radio Network (CRN)** in *overlay* mode, and trains it with a custom **Multi-Agent TD3
(MATD3)** algorithm using **Centralized Training with Decentralized Execution (CTDE)**.

- **Primary network:** a Primary Transmitter (PT) → Primary Receiver (PR) link that owns the
  spectrum and requires interference protection.
- **Secondary network:** `N` Secondary Users share a destination. **One of them (SU_N) *is*
  the half-duplex Decode-and-Forward relay — there is no separate relay node.** The other
  `M = N-1` SUs are sources that transmit **simultaneously (NOMA, power domain)** to the
  relay-SU, which applies **Successive Interference Cancellation (SIC)**. In slot 2 the
  relay-SU forwards their decoded data **together with its own data** and the PU's data.
- The agents observe only **imperfect CSI** and learn robust power-allocation policies.

The full mathematical model is documented in **[docs/SYSTEM_MODEL.md](docs/SYSTEM_MODEL.md)** —
that document is the source of truth for the protocol, spaces, and reward.

---

## 🧠 System model (summary)

Two-timeslot block-fading protocol (see [docs/SYSTEM_MODEL.md](docs/SYSTEM_MODEL.md) for full detail):

- **Slot 1 — multiple access:** PT and the `M = N−1` SU *sources* transmit. The relay-SU is
  half-duplex (listens only); it decodes the PU first (sources as interference), subtracts it,
  then decodes the sources via SIC in descending `|h_sr,i|²` order.
- **Slot 2 — superposition forwarding:** the relay-SU transmits a power-domain superposition of
  the PU's data (share `α`), **its own data** (share `own_share` of the remaining `1−α`), and
  the decoded source data. The destination SICs the streams in descending received power.

Sources are two-hop (DF); the relay-SU's own data is single-hop:

```
γ_e2e,i = min(γ_sr,i, γ_fwd,i)          # sources
R_SU    = Σ_i ½·log₂(1+γ_e2e,i) + ½·log₂(1+γ_own)
```

Interference constraint at the PR: `I_PR = Σ_i P_s,i·|h_sp,i|² + (1−α)·P_rel·|h_rp|² ≤ I_th`.

### Spaces (per [docs/SYSTEM_MODEL.md](docs/SYSTEM_MODEL.md) §4)

| | Shape | Description |
|---|---|---|
| **Observation** (per SU) | `(8,)` | Estimated channel gains, in dB and normalized. Sources use `[ĥ_sr, ĥ_sp, ĥ_sd, …]`; the relay-SU's row uses its own links `[ĥ_rd, ĥ_rp, ĥ_pr, …]`. Each agent encodes its last `L=10` observations with a GRU. |
| **Action** (joint) | `(N+2,)` in `[0,1]` | `M = N−1` source powers, `1` relay-SU power, `α` (PU share), `own_share` (relay's own data share). |
| **Reward** | scalar | Secondary sum-rate `R_SU` (sources' e2e rates + relay-SU's own rate), with an interference-constraint penalty. |

---

## 🏗 Architecture (MATD3, CTDE)

- **Decentralized SU actors:** each SU has its own GRU belief encoder + actor producing its
  normalized transmit power ([agents/matd3_networks.py](agents/matd3_networks.py)).
- **Centralized relay actor:** consumes the concatenated SU belief states and outputs the
  relay power and `α` ([agents/matd3_networks.py](agents/matd3_networks.py) `CentralizedRelayActor`).
- **Three centralized twin critics:** throughput, PU-QoS, and energy
  (`critic_thr`, `critic_qos`, `critic_nrg`), enabling adaptive **Lagrangian** constraint handling.
- **Sequence replay buffer:** stores per-episode transitions and samples `L`-length windows
  ([agents/ma_buffers.py](agents/ma_buffers.py)).

Details: **[docs/MATD3_ARCHITECTURE.md](docs/MATD3_ARCHITECTURE.md)**.

---

## 📂 Repository structure

```
crn_overlay/
├── configs/
│   ├── config.yaml          # Master configuration (source of truth for hyperparameters)
│   └── experiment.yaml
├── docs/
│   ├── SYSTEM_MODEL.md      # Mathematical system model (protocol, spaces, reward)
│   ├── MATD3_ARCHITECTURE.md
│   ├── coding_guidelines.md
│   ├── contributing.md
│   └── development_roadmap.md
├── simulator/
│   ├── noma_overlay_model.py  # NOMA overlay physics (SIC, DF relay, interference)
│   ├── channels.py            # Rayleigh fading
│   ├── propagation.py         # Distance / path-loss
│   └── utils.py
├── envs/
│   └── multi_agent_crn_env.py # Gymnasium-style multi-agent wrapper + history tracking
├── agents/
│   ├── matd3.py               # MATD3 agent (training loop, save/load, Lagrangians)
│   ├── matd3_networks.py      # GRU actors, centralized relay actor, twin critics
│   └── ma_buffers.py          # Multi-agent sequence replay buffer
├── cli/
│   ├── parser.py              # Argument parser / validators
│   ├── runner.py              # Subcommand handlers (train, evaluate, resume, ...)
│   ├── report_generator.py    # Plots + markdown/PDF reports from metrics.json
│   ├── test_runner.py         # Config validation + pytest + smoke test
│   └── logger.py
├── tests/                     # Pytest suite (env spaces, reward, save/load round-trip)
├── experiments/               # Run outputs (gitignored): checkpoints, metrics.json, logs
├── visualizer/                # Static animation viewer for animation_trace.jsonl
├── main.py                    # CLI entry point
└── requirements.txt
```

---

## 🚀 Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Train MATD3 (uses configs/config.yaml as the default source of truth)
python main.py train --agent matd3

# Evaluate the best saved checkpoint deterministically
python main.py evaluate --agent matd3

# Resume from the latest checkpoint
python main.py resume --agent matd3

# Run across all predefined seeds (42, 123, 2026)
python main.py train --agent matd3 --all-seeds

# Validate config + run unit and smoke tests
python main.py test
```

> **Note:** `matd3` is currently the only supported agent. The CLI accepts optional overrides
> such as `--episodes`, `--steps`, `--seed`, `--device cuda`, `--batch-size`, and `--lr`;
> when omitted, values come from `configs/config.yaml`.

---

## ⚙️ CLI subcommands

| Command | Purpose |
| :--- | :--- |
| `train` | Train MATD3 (single seed or `--all-seeds`) |
| `evaluate` | Deterministic policy evaluation of a saved checkpoint |
| `benchmark` | Train + summarize (defaults to the available agents) |
| `compare` | Print a comparison table from stored `metrics.json` |
| `resume` | Resume training from the latest checkpoint |
| `plots` | Generate comparison plots from run metrics |
| `report` | Generate markdown + PDF reports from run metrics |
| `checkpoints` | List and inspect stored `.pth` checkpoints |
| `config` | Print the active configuration |
| `test` | Config validation + pytest + a 5-step smoke test |

---

## 📊 Reproducibility & outputs

- **Seeds:** predefined `42`, `123`, `2026`; evaluation uses deterministic (noise-free) actions.
- Each run under `experiments/matd3/run_<timestamp>_seed_<seed>/` archives:
  1. `config_snapshot.yaml` — exact configuration used
  2. `train.log` — text log
  3. `checkpoints/best_model.pth`, `checkpoints/final_model.pth` (+ replay-buffer companions)
  4. `tensorboard/` event logs
  5. `metrics.json` — measured returns, SU rate, PU/SU outage, etc.

Reports read **only** from each run's `metrics.json` — no synthetic values are injected.

---

## 🧩 Tech stack

Python 3.11+ · PyTorch · Gymnasium · NumPy · SciPy · Matplotlib · PyYAML · TensorBoard · pytest · black · ruff

---

## 📑 Documentation

- **[docs/SYSTEM_MODEL.md](docs/SYSTEM_MODEL.md)** — mathematical system model (authoritative).
- **[docs/MATD3_ARCHITECTURE.md](docs/MATD3_ARCHITECTURE.md)** — MATD3 network and training details.
- **[docs/coding_guidelines.md](docs/coding_guidelines.md)** · **[docs/contributing.md](docs/contributing.md)** · **[docs/development_roadmap.md](docs/development_roadmap.md)**

---

## 📜 License

Intended for academic and research purposes. See [LICENSE](LICENSE).
