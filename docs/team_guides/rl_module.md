# Reinforcement Learning Module Guide
**Assignee:** Aditya

## Objectives
Implement RL training and evaluation scripts using Stable-Baselines3.

## Files to modify
- `agents/train_dqn.py`
- `agents/train_ppo.py`
- `agents/evaluate.py`
- `baselines/*`

## Expected APIs
- Scripts should load the Gym environment from `envs.crn_env` and train SB3 models.

## Testing Checklist
- Ensure models converge on a dummy environment.
- Save models to `experiments/runs/`.
