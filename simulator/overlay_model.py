"""
Overlay CRN System Model — full integration orchestrator.
Author: Ryan

Implements the two-time-slot Overlay Cognitive Radio Network using
Decode-and-Forward relaying. Delegates channel generation, path loss,
relay logic, interference, and metrics to injected protocol objects.

Default stub implementations are provided so the whole pipeline runs
end-to-end without Sneha's or Shreya's code.  When their real modules
are ready they can be injected via the constructor.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from simulator.base_model import (
    BaseSimulator,
    ChannelModelProtocol,
    InterferenceModelProtocol,
    MetricsCalculatorProtocol,
    PropagationModelProtocol,
    RelayProtocol,
    SimulationResult,
    SimulationState,
    SimulatorConfig,
)


# ===================================================================
# Default stub implementations (placeholder physics)
# ===================================================================


class DefaultChannelModel:
    """Generates random uniform channel gains (placeholder for Sneha)."""

    def generate_channels(
        self, rng: np.random.Generator
    ) -> Dict[str, float]:
        """Return uniform [0.1, 2.0] channel gains for every link."""
        links = [
            "pt_pr",
            "pt_relay",
            "su_relay",
            "relay_pr",
            "relay_sud",
            "pt_sud",
        ]
        return {link: float(rng.uniform(0.1, 2.0)) for link in links}


class DefaultPropagationModel:
    """Returns unity path loss — no attenuation (placeholder for Sneha)."""

    def compute_path_loss(self, distance: float) -> float:
        """No path loss applied in the default stub."""
        return 1.0


class DefaultRelayModel:
    """Always decodes successfully (placeholder for Shreya)."""

    def can_decode(self, snr: float) -> bool:
        """Relay always decodes in the default stub."""
        return True

    def forward(
        self, channel_gain: float, power: float, noise: float
    ) -> float:
        """Forwards at full received power: gain * power."""
        return channel_gain * power


class DefaultInterferenceModel:
    """Returns zero interference (placeholder for Shreya)."""

    def compute_interference(
        self,
        powers: Dict[str, float],
        gains: Dict[str, float],
    ) -> Dict[str, float]:
        """No interference in the default stub."""
        return {"pr": 0.0, "sud": 0.0}


class DefaultMetricsCalculator:
    """Basic Shannon-capacity metrics (placeholder for Shreya)."""

    def compute_sinr(
        self, signal: float, interference: float, noise: float
    ) -> float:
        """SINR = signal / (interference + noise)."""
        denom = interference + noise
        if denom <= 0:
            return 0.0
        return signal / denom

    def compute_throughput(self, sinr: float, bandwidth: float) -> float:
        """Shannon capacity: log2(1 + SINR) in bits/s/Hz."""
        if sinr <= 0:
            return 0.0
        return math.log2(1.0 + sinr)


# ===================================================================
# Overlay CRN Simulator
# ===================================================================


class OverlaySimulator(BaseSimulator):
    """Overlay Cognitive Radio Network simulator with DF relaying.

    The system operates in **two time-slots** per step:

    **Time Slot 1** — PT transmits to PR;  SU Source transmits to Relay.
    **Time Slot 2** — PT transmits to PR;  Relay (if decoded) forwards
    to SU Destination.

    The RL agent selects ``[p_su, p_relay]`` (normalised to [0, 1]).

    Args:
        config: Simulator configuration parameters.
        channel_model: Generates wireless channel coefficients.
        propagation_model: Computes distance-dependent path loss.
        relay_model: Decode-and-Forward relay logic.
        interference_model: Interference power calculations.
        metrics_calculator: SINR and throughput computations.
    """

    def __init__(
        self,
        config: Optional[SimulatorConfig] = None,
        channel_model: Optional[ChannelModelProtocol] = None,
        propagation_model: Optional[PropagationModelProtocol] = None,
        relay_model: Optional[RelayProtocol] = None,
        interference_model: Optional[InterferenceModelProtocol] = None,
        metrics_calculator: Optional[MetricsCalculatorProtocol] = None,
    ) -> None:
        if config is None:
            config = SimulatorConfig()
        super().__init__(config)

        self.channel_model: ChannelModelProtocol = (
            channel_model or DefaultChannelModel()
        )
        self.propagation_model: PropagationModelProtocol = (
            propagation_model or DefaultPropagationModel()
        )
        self.relay_model: RelayProtocol = (
            relay_model or DefaultRelayModel()
        )
        self.interference_model: InterferenceModelProtocol = (
            interference_model or DefaultInterferenceModel()
        )
        self.metrics: MetricsCalculatorProtocol = (
            metrics_calculator or DefaultMetricsCalculator()
        )

        # Internal bookkeeping
        self._rng: np.random.Generator = np.random.default_rng()
        self._state: SimulationState = SimulationState()
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # BaseSimulator interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> SimulationState:
        """Reset the simulator for a new episode.

        Args:
            seed: Optional RNG seed for reproducibility.

        Returns:
            The initial ``SimulationState``.
        """
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        # Generate fresh channel coefficients
        channel_gains = self.channel_model.generate_channels(self._rng)

        # Compute path losses for each link
        cfg = self.config
        distances: Dict[str, float] = {
            "pt_pr": cfg.d_pt_pr,
            "pt_relay": cfg.d_pt_relay,
            "su_relay": cfg.d_su_relay,
            "relay_pr": cfg.d_relay_pr,
            "relay_sud": cfg.d_relay_sud,
            "pt_sud": cfg.d_pt_sud,
        }
        path_losses = {
            link: self.propagation_model.compute_path_loss(d)
            for link, d in distances.items()
        }

        # Effective gains = channel gain × path loss multiplier
        effective_gains = {
            link: channel_gains[link] * path_losses[link]
            for link in channel_gains
        }

        self._state = SimulationState(
            channel_gains=channel_gains,
            path_losses=path_losses,
            effective_gains=effective_gains,
            power_allocation=None,
            su_throughput=0.0,
            pu_throughput=0.0,
            interference_at_pr=0.0,
            relay_decoded=False,
            step_count=0,
            sinr_su=0.0,
            sinr_pu=0.0,
        )
        return self._state

    def step(self, action: np.ndarray) -> SimulationResult:
        """Execute one communication round (two time-slots).

        Args:
            action: Shape ``(2,)`` array with values in ``[0, 1]``.
                ``action[0]`` → normalised SU source power (slot 1).
                ``action[1]`` → normalised relay power (slot 2).

        Returns:
            A ``SimulationResult`` containing the new observation,
            reward, termination flags, and diagnostic info.
        """
        cfg = self.config
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape[0] < 2:
            action = np.append(action, action)

        # Scale normalised actions to actual power
        p_su = float(np.clip(action[0], 0.0, 1.0)) * cfg.p_max_su
        p_relay = float(np.clip(action[1], 0.0, 1.0)) * cfg.p_max_relay
        p_pt = cfg.p_pt
        noise = cfg.noise_power

        g = self._state.effective_gains

        # ==============================================================
        # Time Slot 1:  PT → PR  and  SU Source → Relay
        # ==============================================================
        signal_pt_pr_1 = p_pt * g.get("pt_pr", 0.0)
        signal_su_relay = p_su * g.get("su_relay", 0.0)

        # SNR at relay for decoding decision
        snr_at_relay = signal_su_relay / noise if noise > 0 else 0.0
        relay_decoded = self.relay_model.can_decode(snr_at_relay)

        # ==============================================================
        # Time Slot 2:  PT → PR  and  Relay → SU Dest (if decoded)
        # ==============================================================
        signal_pt_pr_2 = p_pt * g.get("pt_pr", 0.0)

        if relay_decoded:
            signal_relay_sud = self.relay_model.forward(
                g.get("relay_sud", 0.0), p_relay, noise
            )
        else:
            signal_relay_sud = 0.0

        # Combined PU signal (simple average over two slots)
        signal_pu = 0.5 * (signal_pt_pr_1 + signal_pt_pr_2)

        # ==============================================================
        # Interference
        # ==============================================================
        powers: Dict[str, float] = {
            "su": p_su,
            "relay": p_relay if relay_decoded else 0.0,
            "pt": p_pt,
        }
        interference = self.interference_model.compute_interference(
            powers, g
        )
        interference_at_pr = interference.get("pr", 0.0)
        interference_at_sud = interference.get("sud", 0.0)

        # ==============================================================
        # Metrics
        # ==============================================================
        sinr_pu = self.metrics.compute_sinr(
            signal_pu, interference_at_pr, noise
        )
        sinr_su = self.metrics.compute_sinr(
            signal_relay_sud, interference_at_sud, noise
        )

        su_throughput = self.metrics.compute_throughput(
            sinr_su, cfg.bandwidth
        )
        pu_throughput = self.metrics.compute_throughput(
            sinr_pu, cfg.bandwidth
        )

        # ==============================================================
        # Update state
        # ==============================================================
        self._step_count += 1

        # Generate new channels for the next step
        new_gains = self.channel_model.generate_channels(self._rng)
        distances_map: Dict[str, float] = {
            "pt_pr": cfg.d_pt_pr,
            "pt_relay": cfg.d_pt_relay,
            "su_relay": cfg.d_su_relay,
            "relay_pr": cfg.d_relay_pr,
            "relay_sud": cfg.d_relay_sud,
            "pt_sud": cfg.d_pt_sud,
        }
        new_pl = {
            link: self.propagation_model.compute_path_loss(d)
            for link, d in distances_map.items()
        }
        new_eff = {
            link: new_gains[link] * new_pl[link] for link in new_gains
        }

        self._state = SimulationState(
            channel_gains=new_gains,
            path_losses=new_pl,
            effective_gains=new_eff,
            power_allocation=np.array([p_su, p_relay], dtype=np.float32),
            su_throughput=su_throughput,
            pu_throughput=pu_throughput,
            interference_at_pr=interference_at_pr,
            relay_decoded=relay_decoded,
            step_count=self._step_count,
            sinr_su=sinr_su,
            sinr_pu=sinr_pu,
        )

        # Build result
        obs = self.get_observation(self._state)
        reward = self.compute_reward(self._state)
        truncated = self._step_count >= cfg.max_steps
        terminated = False

        info: Dict[str, Any] = {
            "su_throughput": su_throughput,
            "pu_throughput": pu_throughput,
            "sinr_su": sinr_su,
            "sinr_pu": sinr_pu,
            "interference_at_pr": interference_at_pr,
            "relay_decoded": relay_decoded,
            "p_su": p_su,
            "p_relay": p_relay,
            "step": self._step_count,
        }

        return SimulationResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_observation(self, state: SimulationState) -> np.ndarray:
        """Build the observation vector from effective channel gains.

        The observation order matches the canonical link ordering:
        ``[pt_pr, pt_relay, su_relay, relay_pr, relay_sud, pt_sud]``.

        Args:
            state: Current simulation state.

        Returns:
            Numpy float32 array of shape ``(num_channels,)``.
        """
        link_order = [
            "pt_pr",
            "pt_relay",
            "su_relay",
            "relay_pr",
            "relay_sud",
            "pt_sud",
        ]
        obs = np.array(
            [state.effective_gains.get(k, 0.0) for k in link_order],
            dtype=np.float32,
        )
        return obs

    def compute_reward(self, state: SimulationState) -> float:
        """Reward = SU throughput − penalty if PU constraint violated.

        Args:
            state: Current simulation state.

        Returns:
            Scalar reward value.
        """
        reward = state.su_throughput
        if state.interference_at_pr > self.config.interference_threshold:
            penalty = self.config.penalty_weight * (
                state.interference_at_pr
                - self.config.interference_threshold
            )
            reward -= penalty
        return reward
