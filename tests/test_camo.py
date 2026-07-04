"""
Unit tests for CRN physical equations, sequence replay buffer, and Underlay/Overlay TD3 networks.
"""

import yaml
import numpy as np
import torch
import pytest

from simulator.propagation import calculate_path_loss
from simulator.channels import RayleighFading
from simulator.relay import DecodeAndForward
from simulator.interference import calculate_received_power, calculate_interference
from simulator.metrics import calculate_sinr, calculate_capacity, calculate_throughput, calculate_ber
from envs.crn_env import OverlayCRNEnv
from agents.buffers import SequenceReplayBuffer
from agents.models import GRUBeliefEncoder, CAMO_Actor, TwinCritics


def test_physical_equations():
    # 1. Path Loss
    pl_10 = calculate_path_loss(10.0, path_loss_exponent=3.0)
    assert pl_10 == pytest.approx(10.0 ** (-3.0))

    # Clamping check
    pl_zero = calculate_path_loss(0.0, path_loss_exponent=3.0)
    assert pl_zero == pytest.approx(1e-3 ** (-3.0))

    # 2. Rayleigh fading coefficients
    fading = RayleighFading()
    coeff = fading.generate_coefficient()
    assert isinstance(coeff, complex)

    gain = fading.generate_gain(distance=10.0, path_loss_exponent=3.0)
    assert gain >= 0.0

    # 3. Relay logic
    df = DecodeAndForward()
    assert df.can_decode(sinr=2.5, threshold=1.0) is True
    assert df.can_decode(sinr=0.5, threshold=1.0) is False

    # 4. Interference
    rx_p = calculate_received_power(transmit_power=2.0, channel_gain=0.5)
    assert rx_p == pytest.approx(1.0)

    inf = calculate_interference([0.2, 0.3, 0.5])
    assert inf == pytest.approx(1.0)

    # 5. SINR & Rates
    sinr = calculate_sinr(signal_power=1.0, interference_power=0.5, noise_power=0.5)
    assert sinr == pytest.approx(1.0)

    capacity = calculate_capacity(sinr=1.0, bandwidth=1.0)
    assert capacity == pytest.approx(1.0)  # log2(1+1) = 1.0

    throughput = calculate_throughput(sinr=1.0, time_fraction=0.5, bandwidth=1.0)
    assert throughput == pytest.approx(0.5)

    # BER check
    ber = calculate_ber(sinr=10.0, modulation_order=4)
    assert 0.0 <= ber <= 0.5


def test_sequence_replay_buffer():
    buffer = SequenceReplayBuffer(capacity=100, obs_dim=4, action_dim=2, sequence_length=5, device="cpu")
    
    # Fill buffer with fake transitions forming episodes
    obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    act = np.array([0.5, 0.5], dtype=np.float32)
    reward = 1.0
    next_obs = obs + 1.0
    info = {"throughput_reward": 1.0, "interference_reward": 0.1, "energy_reward": 0.2}

    # Episode 1: 3 steps
    buffer.add(obs, act, reward, next_obs, False, info)
    buffer.add(obs + 1, act, reward, next_obs + 1, False, info)
    buffer.add(obs + 2, act, reward, next_obs + 2, True, info)

    # Episode 2: 2 steps
    buffer.add(obs, act, reward, next_obs, False, info)
    buffer.add(obs + 1, act, reward, next_obs + 1, True, info)

    assert len(buffer) == 5

    # Check sequence sampling
    (
        obs_seqs,
        act_seqs,
        rewards,
        next_obs_seqs,
        next_act_seqs,
        dones,
        thr_rewards,
        inf_rewards,
        nrg_rewards,
    ) = buffer.sample_sequences(batch_size=2)

    # Sample batch dimensions
    assert obs_seqs.shape == (2, 5, 4)
    assert act_seqs.shape == (2, 5, 2)
    assert rewards.shape == (2, 1)
    assert next_obs_seqs.shape == (2, 5, 4)
    assert next_act_seqs.shape == (2, 5, 2)
    assert dones.shape == (2, 1)


def test_neural_networks():
    # Encoders
    encoder = GRUBeliefEncoder(obs_dim=4, action_dim=2, embed_dim=16, hidden_dim=32)
    obs_seq = torch.randn(2, 5, 4)
    act_seq = torch.randn(2, 5, 2)
    
    belief = encoder(obs_seq, act_seq)
    assert belief.shape == (2, 32)

    # Actor
    actor = CAMO_Actor(belief_dim=32, action_dim=2)
    actions = actor(belief)
    assert actions.shape == (2, 2)
    assert torch.all(actions >= 0.0) and torch.all(actions <= 1.0)

    # Twin Critics
    critics = TwinCritics(state_dim=32, action_dim=2)
    q1, q2 = critics.evaluate(belief, actions)
    assert q1.shape == (2, 1)
    assert q2.shape == (2, 1)


def test_overlay_td3():
    buffer = SequenceReplayBuffer(capacity=100, obs_dim=4, action_dim=2, sequence_length=5, device="cpu")
    
    # Fill buffer with fake transitions forming episodes with overlay info
    obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    act = np.array([0.5, 0.5], dtype=np.float32)
    reward = 1.0
    next_obs = obs + 1.0
    info = {
        "throughput_reward": 1.0, 
        "interference_reward": 0.1, 
        "energy_reward": 0.2,
        "primary_throughput": 2.0,
        "average_power": 0.08,
        "relay_decoded": 1.0,
        "outage": 0.0
    }

    buffer.add(obs, act, reward, next_obs, False, info)
    buffer.add(obs + 1, act, reward, next_obs + 1, True, info)

    # Sample overlay sequences
    (
        hist_seqs,
        next_hist_seqs,
        actions,
        rewards,
        dones,
        thr_rewards,
        pu_qos_rewards,
        nrg_rewards,
    ) = buffer.sample_sequences_overlay(batch_size=2)

    assert hist_seqs.shape == (2, 5, 8)
    assert next_hist_seqs.shape == (2, 5, 8)
    assert actions.shape == (2, 2)
    assert rewards.shape == (2, 1)
    assert dones.shape == (2, 1)
    assert pu_qos_rewards.shape == (2, 1)
    assert nrg_rewards.shape == (2, 1)

    # Test encoder with 8D inputs
    encoder = GRUBeliefEncoder(obs_dim=4, action_dim=2, embed_dim=16, hidden_dim=32, input_dim=8)
    belief = encoder(hist_seqs)
    assert belief.shape == (2, 32)

