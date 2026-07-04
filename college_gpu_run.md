# Running the CRN Simulator on College GPUs

This guide explains how to execute the training pipeline on a GPU cluster using Aditya's CLI flow. The algorithm used is the newly implemented designed algorithm, `OVERLAY_TD3`, running for 3,000 episodes.

## Prerequisites

1. Ensure you have cloned the repository and are on the `main` branch.
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure PyTorch is installed with CUDA support appropriate for the cluster's GPUs)*

## Execution Command

To train the Overlay TD3 agent for 3000 episodes on a CUDA device, run the following command from the project root:

```bash
python main.py train --agent overlay --episodes 3000 --device cuda --save-best True --tensorboard True
```

### Parameters Used:
- `train`: Invokes Aditya's implemented training CLI.
- `--agent overlay`: Configures the pipeline to use the `OVERLAY_TD3` algorithm designed for this architecture.
- `--episodes 3000`: Sets the training duration to exactly 3,000 episodes as requested.
- `--device cuda`: Ensures PyTorch utilizes the GPU for training computations.
- `--save-best True`: Persists the best performing model weights to the `experiments/checkpoints/` directory.
- `--tensorboard True`: Enables TensorBoard logging in `experiments/runs/` for live metrics tracking.

## Verifying GPU Usage

When the script starts, it will output system information including:
```
PyTorch version: <version>
CUDA available: True
```
If `CUDA available` is `False`, the environment on the cluster is not configured correctly for PyTorch GPU support.

## Viewing Results
Once training begins, you can monitor the progress by starting a TensorBoard server:
```bash
tensorboard --logdir experiments/runs/
```
