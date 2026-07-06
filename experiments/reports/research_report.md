# CRN Reinforcement Learning Framework: Experimental Report

This report compiles performance and convergence metrics for standard **TD3**, **Underlay TD3**, and **Overlay TD3** under Rayleigh fading constraints.

## 1. Benchmarking Summary Table

| Algorithm | Mean Return | Best Seed Return | SU Throughput (bps/Hz) | PU Outage Rate | Training Time (s) | Avg Inf. Time (ms) |
|---|---|---|---|---|---|---|
| **TD3** | 967956654.41 | 968590916.03 | 4839783.2721 | 0.0000 | 1007.62s | 1.279ms |
| **Underlay TD3** | 965963549.01 | 968630143.34 | 4829817.7450 | 0.0000 | 6378.06s | 6.135ms |
| **Overlay TD3** | 977700183.96 | 988645648.87 | 4888500.9198 | 0.0000 | 5140.77s | 2.433ms |

## 2. Convergence Analysis

The convergence plots reflect the policy return trends across different seeds. Standard memoryless TD3 agents typically exhibit higher variance and outages due to the lack of history modeling. Recurrent Underlay and Overlay structures leverage temporal state representation to adapt to fading fluctuations.

### Policy Convergence
![Convergence Comparison](../plots/convergence_comparison.png)

### Metric Performance
![Metrics Comparison](../plots/metrics_comparison.png)

### Computational Efficiency
![Efficiency Comparison](../plots/efficiency_comparison.png)

