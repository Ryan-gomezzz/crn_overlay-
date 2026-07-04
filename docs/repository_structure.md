# Repository Structure

This document outlines the file organization and directories mapping of the Overlay Cognitive Radio Network RL framework.

---

## 1. Directory Tree Map

```
CRN-RL-Framework/
│
├── configs/
│   ├── config.yaml          # Master configuration
│   └── experiment.yaml      # Experiment hyperparameters overrides
│
├── docs/                    # Technical specifications and guides
│   ├── agents/
│   │   ├── TD3.md            # TD3 baseline guide
│   │   ├── UNDERLAY_TD3.md  # Underlay TD3 (Original Underlay TD3) guide
│   │   └── OVERLAY_TD3.md   # Overlay TD3 guide
│   ├── architecture.md
│   ├── gymnasium_design.md
│   ├── repository_structure.md
│   ├── system_integration.md
│   ├── coding_guidelines.md
│   ├── contributing.md
│   ├── development_roadmap.md
│   ├── benchmarking.md
│   ├── algorithm_comparison.md
│   └── research_notes.md
│
├── simulator/               # Physical wireless models
│   ├── base_model.py
│   ├── overlay_model.py     # Time slots power calculations
│   ├── channels.py          # Rayleigh gains generator
│   ├── propagation.py       # Distance path loss models
│   ├── relay.py             # Decode-and-Forward protocol checks
│   ├── interference.py      # Power interference calculator
│   ├── metrics.py           # Capacity, SINR, and M-QAM BER math
│   └── utils.py
│
├── envs/                    # Gymnasium wrapper
│   └── crn_env.py           # Gymnasium step and history tracking
│
├── agents/                  # RL Agent modules
│   ├── models.py            # Neural networks (Encoder, Actor, Critics)
│   ├── buffers.py           # Sequence Replay Buffers
│   ├── train_td3.py         # Unified training loop
│   ├── evaluate.py          # Standalone checkpoint loader
│   └── benchmark.py         # Comparative benchmark script
│
├── experiments/             # Logs and checkpoints folder
│   └── checkpoints/         # Saved model weights
│
├── plots/                   # Benchmark comparative diagrams
│
├── tests/                   # Verification suite
│   └── test_camo.py         # Pytest verification cases
│
├── main.py                  # Standard pipeline orchestrator
└── requirements.txt
```
