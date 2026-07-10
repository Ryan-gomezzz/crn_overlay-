import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.multi_agent_crn_env import make_ma_crn_env
from agents.matd3 import MATD3Agent

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

print("Testing Env...")
env = make_ma_crn_env("configs/config.yaml")
obs, info = env.reset()
print("Obs shape:", obs.shape)

print("Testing Agent...")
agent = MATD3Agent(config)
action = agent.select_action(obs, info, explore=True)
print("Action shape:", action.shape)

print("Testing Step...")
n_obs, rew, done, trunc, n_info = env.step(action)
agent.replay_buffer.add(obs, action, rew, n_obs, done, info)
print("Reward:", rew)

print("Testing Train (with 128 samples to pass batch_size)...")
for _ in range(130):
    agent.replay_buffer.add(obs, action, rew, n_obs, done, info)

agent.train()
print("Train passed!")
