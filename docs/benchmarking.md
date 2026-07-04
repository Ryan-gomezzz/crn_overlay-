# Benchmarking and Evaluation

This document outlines the comparative benchmarking and evaluation pipeline implemented in the repository to assess **TD3**, **Underlay TD3**, and **Overlay TD3** agents under identical wireless conditions.

---

## 1. Supported Benchmarking Metrics
We record and compare performance across multiple wireless, computational, and convergence metrics:

### 1.1. Wireless Communication Metrics
*   **Secondary User Throughput**: The rate achieved by the SU link in bps/Hz, bottlenecked by the two Decode-and-Forward hops.
*   **Bit Error Rate (BER)**: The average bit error rate at the SU Destination based on M-QAM modulation.
*   **Primary User Outage Probability**: The fraction of steps where the primary throughput drops below the minimum QoS rate ($R_{threshold}$).

### 1.2. Computational Efficiency Metrics
*   **Training Time**: Total clock time (seconds) taken to run the training steps of the agent.
*   **Average Inference Time**: Average time (milliseconds) taken by the agent to perform action selection forward passes.

### 1.3. Convergence & Constraints Metrics
*   **Total Episode Reward**: Trajectory rewards representing throughput minus soft penalties.
*   **Lagrangian Multipliers convergence**: Trajectory values of multipliers ($\lambda_{inf}$, $\lambda_{QoS}$, $\lambda_{nrg}$) over steps.
*   **QoS/Constraint Satisfaction Rates**: Average steps where constraints are fully satisfied.

---

## 2. Benchmarking Script Usage
The repository includes a dedicated benchmark script:
```bash
python agents/benchmark.py
```

### Execution Flow
1.  **TD3 baseline run**: Trains standard TD3 for 600 steps, recording evaluation milestones. Saves model checkpoint.
2.  **Underlay TD3 run**: Trains recurrent Underlay TD3 under co-channel interference and energy limit constraints.
3.  **Overlay TD3 run**: Trains recurrent Overlay TD3 under direct primary user QoS constraints and energy limit constraints.
4.  **Plots generation**: Aggregates records and saves comparative matplotlib plots under the `plots/` directory.

### Output Plots
*   `plots/throughput_comparison.png`: SU throughput comparison.
*   `plots/ber_comparison.png`: SU bit error rates comparison (log scale).
*   `plots/outage_comparison.png`: PU outage comparison.
*   `plots/convergence_comparison.png`: Policy reward learning curves.
*   `plots/lambda_comparison.png`: Converged Lagrangian dual values.
*   `plots/time_comparison.png`: Bar chart of training vs. inference latencies.
