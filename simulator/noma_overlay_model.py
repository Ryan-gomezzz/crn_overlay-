"""
Multi-User NOMA Overlay CRN Simulator — physics layer.
Author: Ryan

Implements a Non-Orthogonal Multiple Access (NOMA) overlay cognitive
radio network with N secondary users, a shared Decode-and-Forward relay,
and a primary network.

Two-timeslot protocol
---------------------
  Slot 1 (0 → T/2):
    - PT transmits x_p.
    - All N SU sources transmit simultaneously (NOMA).
    - Relay receives: y_r = Σ_i √P_si · h_sr_i · x_si + √P_p · h_pr · x_p + n_r
    - Relay performs SIC: decodes users in descending order of |h_sr_i|².

  Slot 2 (T/2 → T):
    - PT transmits x_p.
    - Relay forwards the superposed signal to SU destination.
    - Destination receives: y_d = √P_r · h_rd · x_r + √P_p · h_pd · x_p + n_d

Per-user achievable rate (DF end-to-end):
    γ_sr_i  = (P_si · |h_sr_i|²) / (interference_after_SIC + P_p · |h_pr|² + N_0)
    γ_rd    = (P_r  · |h_rd|²)   / (P_p · |h_pd|² + N_0)
    γ_e2e_i = min(γ_sr_i, γ_rd)
    R_i     = (1/2) · log2(1 + γ_e2e_i)

Interference constraint at PR:
    I_PR = Σ_i P_si · |h_sp_i|² + P_r · |h_rp|²  ≤  I_th
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulator.channels import RayleighFading
from simulator.propagation import calculate_path_loss


# =====================================================================
# Configuration dataclass for the multi-user NOMA model
# =====================================================================


@dataclass
class NOMAConfig:
    """Configuration for the multi-user NOMA overlay CRN simulator.

    All coordinates are in metres.  Distances are computed internally
    from coordinates using Euclidean distance.

    Attributes:
        num_su: Number of secondary users (N).
        pt_coords: Primary Transmitter location [x, y].
        pr_coords: Primary Receiver location [x, y].
        su_coords: List of SU source coordinates [[x1,y1], ...].
        sud_coords: SU destination coordinate [x, y].
        sur_coords: Shared relay coordinate [x, y].
        p_primary_dbm: PT transmit power in dBm.
        p_max_su_dbm: Max SU/relay transmit power in dBm.
        noise_power_dbm: AWGN noise power in dBm.
        path_loss_exponent: Path-loss exponent α.
        interference_threshold_dbm: Max tolerable interference at PR (dBm).
        max_steps: Max time-steps per episode.
        penalty_weight: Penalty multiplier for constraint violation.
    """

    # Number of secondary users
    num_su: int = 3

    # Node positions (metres)
    pt_coords: List[float] = field(default_factory=lambda: [0.0, 0.0])
    pr_coords: List[float] = field(default_factory=lambda: [100.0, 0.0])
    su_coords: List[List[float]] = field(
        default_factory=lambda: [[10.0, 20.0], [30.0, 25.0], [70.0, 15.0]]
    )
    sud_coords: List[float] = field(default_factory=lambda: [90.0, 20.0])
    sur_coords: List[float] = field(default_factory=lambda: [50.0, 10.0])

    # Power parameters (dBm → converted to Watts internally)
    p_primary_dbm: float = 30.0
    p_max_su_dbm: float = 20.0
    noise_power_dbm: float = -114.0
    interference_threshold_dbm: float = -80.0

    # Channel Estimation Error (Imperfect CSI)
    csi_error_variance: float = 0.1
    path_loss_exponent: float = 3.5

    # Episode
    max_steps: int = 200
    penalty_weight: float = 10.0

    # ---- Derived helpers (computed once) ----

    @staticmethod
    def dbm_to_watts(dbm: float) -> float:
        """Convert dBm to Watts."""
        return 10.0 ** ((dbm - 30.0) / 10.0)

    @property
    def p_pt(self) -> float:
        """PT power in Watts."""
        return self.dbm_to_watts(self.p_primary_dbm)

    @property
    def p_max_su(self) -> float:
        """Max SU / relay power in Watts."""
        return self.dbm_to_watts(self.p_max_su_dbm)

    @property
    def noise_power(self) -> float:
        """Noise power (N_0) in Watts."""
        return self.dbm_to_watts(self.noise_power_dbm)

    @property
    def interference_threshold(self) -> float:
        """Interference threshold (I_th) in Watts."""
        return self.dbm_to_watts(self.interference_threshold_dbm)


# =====================================================================
# Helper utilities
# =====================================================================


def _euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two 2-D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# =====================================================================
# NOMA Overlay Simulator
# =====================================================================


class NOMAOverlaySimulator:
    """Multi-user NOMA overlay CRN simulator with DF relaying.

    Action space
    ------------
    A vector of ``N+1`` values in [0, 1]:
        [p_su_1, p_su_2, …, p_su_N, p_relay]
    Each value is scaled by ``p_max_su`` (Watts) internally.

    Observation space (per agent *i*)
    ----------------------------------
    8-dimensional vector:
        [h_sr_i, h_sp_i, h_sd_i,  h_pp, h_pr, h_pd, h_rd, h_rp]
          ↑ 3 per-user              ↑ 5 common

    Args:
        config: A ``NOMAConfig`` instance.
    """

    # Number of per-user observation features and common features
    OBS_PER_USER = 3    # h_sr_i, h_sp_i, h_sd_i
    OBS_COMMON = 5      # h_pp,  h_pr,  h_pd,  h_rd,  h_rp
    OBS_DIM = OBS_PER_USER + OBS_COMMON  # 8

    def __init__(self, config: Optional[NOMAConfig] = None) -> None:
        self.cfg = config or NOMAConfig()

        N = self.cfg.num_su
        if len(self.cfg.su_coords) < N:
            raise ValueError(
                f"su_coords has {len(self.cfg.su_coords)} entries but "
                f"num_su={N} requires {N}."
            )

        # Channel fading generator (Sneha's RayleighFading)
        self._fading = RayleighFading()

        # Pre-compute all pairwise distances & path-loss multipliers
        self._distances: Dict[str, float] = {}
        self._path_losses: Dict[str, float] = {}
        self._precompute_distances()

        # RNG & state
        self._rng: np.random.Generator = np.random.default_rng()
        self._step_count: int = 0

        # Current effective channel gains (refreshed every step)
        self._eff_gains: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Distance & path-loss pre-computation
    # ------------------------------------------------------------------

    def _precompute_distances(self) -> None:
        """Compute Euclidean distances and path losses for all links."""
        cfg = self.cfg
        N = cfg.num_su
        alpha = cfg.path_loss_exponent

        pt = cfg.pt_coords
        pr = cfg.pr_coords
        relay = cfg.sur_coords
        dest = cfg.sud_coords

        # Primary link
        self._set_link("pp", pt, pr, alpha)

        # PT → relay, PT → dest
        self._set_link("pr_link", pt, relay, alpha)   # h_pr  (PT → Relay)
        self._set_link("pd", pt, dest, alpha)         # h_pd  (PT → SU_dest)

        # Relay → dest, Relay → PR
        self._set_link("rd", relay, dest, alpha)      # h_rd
        self._set_link("rp", relay, pr, alpha)        # h_rp  (interference)

        # Per-SU links
        for i in range(N):
            su_i = cfg.su_coords[i]
            self._set_link(f"sr_{i}", su_i, relay, alpha)   # h_sr_i
            self._set_link(f"sp_{i}", su_i, pr, alpha)      # h_sp_i (interference)
            self._set_link(f"sd_{i}", su_i, dest, alpha)    # h_sd_i (direct)

    def _set_link(
        self, name: str, a: List[float], b: List[float], alpha: float
    ) -> None:
        """Store distance and path-loss for a named link."""
        d = _euclidean(a, b)
        self._distances[name] = d
        self._path_losses[name] = calculate_path_loss(d, alpha)

    # ------------------------------------------------------------------
    # Channel generation
    # ------------------------------------------------------------------

    def _generate_all_channels(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Generate Rayleigh fading |g|² for every link and combine
        with path loss to produce effective channel gains |h|² = PL·|g|².
        Returns both true and estimated effective gains.
        """
        eff_true: Dict[str, float] = {}
        eff_est: Dict[str, float] = {}
        tau = self.cfg.csi_error_variance
        
        for name, pl in self._path_losses.items():
            g = self._fading.generate_coefficient(self._rng)
            e = self._fading.generate_coefficient(self._rng)
            
            # Imperfect CSI estimate
            g_hat = math.sqrt(1.0 - tau**2) * g + tau * e
            
            eff_true[name] = float(abs(g) ** 2) * pl
            eff_est[name] = float(abs(g_hat) ** 2) * pl
            
        return eff_true, eff_est

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the simulator and return the initial info dict.

        Args:
            seed: Optional RNG seed for reproducibility.

        Returns:
            Dictionary with ``"observations"`` (per-agent obs matrix)
            and ``"info"`` (diagnostic dict).
        """
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._eff_gains_true, self._eff_gains_est = self._generate_all_channels()

        obs = self._build_observations()
        return {
            "observations": obs,
            "info": {"step": 0, "effective_gains_true": dict(self._eff_gains_true), "effective_gains_est": dict(self._eff_gains_est)},
        }

    def step(
        self, action: np.ndarray
    ) -> Dict[str, Any]:
        """Advance one communication round (two time-slots).

        Args:
            action: Shape ``(N+1,)`` array in [0, 1].
                ``action[0:N]`` → normalised SU source powers.
                ``action[N]``   → normalised relay power.

        Returns:
            Dictionary with keys:
                ``"observations"`` — (N, 8) per-agent observation matrix
                ``"reward"``       — scalar sum-rate reward
                ``"terminated"``   — bool (always False for now)
                ``"truncated"``    — bool (True when max_steps reached)
                ``"info"``         — detailed diagnostic dict
        """
        cfg = self.cfg
        N = cfg.num_su
        action = np.asarray(action, dtype=np.float64).flatten()

        # Pad or truncate action to (N+2,)
        if action.shape[0] < N + 2:
            action = np.pad(action, (0, N + 2 - action.shape[0]))
        action = np.clip(action[: N + 2], 0.0, 1.0)

        # Convert normalised actions → Watts
        p_su = action[:N] * cfg.p_max_su    # shape (N,)
        p_relay = float(action[N]) * cfg.p_max_su
        alpha = float(action[N+1])          # Power splitting factor [0, 1]
        p_pt = cfg.p_pt
        N0 = cfg.noise_power
        g_true = self._eff_gains_true  # shorthand for physics

        # ==============================================================
        # TIME SLOT 1 — All SUs → Relay  (+ PT → Relay)
        # ==============================================================
        
        rx_power_pr = p_pt * g_true["pr_link"]
        rx_power_sr = np.array(
            [p_su[i] * g_true[f"sr_{i}"] for i in range(N)], dtype=np.float64
        )
        total_su_rx = float(rx_power_sr.sum())
        
        # Relay decodes PU first (SUs treated as interference)
        gamma_relay_pu = rx_power_pr / max(total_su_rx + N0, 1e-30)
        
        # Relay decodes SUs using SIC (assuming PU is subtracted)
        sic_order = np.argsort(-rx_power_sr)
        
        gamma_sr = np.zeros(N, dtype=np.float64)
        for rank, idx in enumerate(sic_order):
            later_users = sic_order[rank + 1:]
            noma_residual = float(rx_power_sr[later_users].sum()) if len(later_users) > 0 else 0.0
            # PU is already decoded and subtracted, so no pt_interference_relay here
            denom = noma_residual + N0
            gamma_sr[idx] = rx_power_sr[idx] / max(denom, 1e-30)

        # ==============================================================
        # TIME SLOT 2 — Relay → SU Dest  (+ PT → SU Dest as interference)
        # ==============================================================
        
        p_relay_pu = alpha * p_relay
        p_relay_su = (1.0 - alpha) * p_relay
        
        interference_dest_pt = p_pt * g_true["pd"]
        
        # Destination receives relayed PU signal and relayed SU signal
        rx_sud_pu = p_relay_pu * g_true["rd"]
        rx_sud_su = p_relay_su * g_true["rd"]
        
        # Destination decodes PU first, subtracts it, then decodes SU
        gamma_sud_pu = rx_sud_pu / max(rx_sud_su + interference_dest_pt + N0, 1e-30)
        gamma_sud_su = rx_sud_su / max(interference_dest_pt + N0, 1e-30)

        # ==============================================================
        # End-to-end DF rates
        # ==============================================================

        gamma_e2e = np.minimum(gamma_sr, gamma_sud_su)
        rates = 0.5 * np.log2(1.0 + gamma_e2e)   # shape (N,)
        sum_rate = float(rates.sum())

        # ==============================================================
        # Interference constraint at PR
        # ==============================================================
        # Slot 1 SU interference + Slot 2 SU interference (worst-case sum)
        # The relayed PU signal (p_relay_pu) is NOT interference to PR.

        i_pr_su_slot1 = float(sum(p_su[i] * g_true[f"sp_{i}"] for i in range(N)))
        i_pr_relay_su_slot2 = p_relay_su * g_true["rp"]
        interference_at_pr = i_pr_su_slot1 + i_pr_relay_su_slot2
        constraint_violated = interference_at_pr > cfg.interference_threshold

        # ==============================================================
        # Primary network throughput (Selection Combining)
        # ==============================================================
        
        signal_pu_slot1 = p_pt * g_true["pp"]
        snr_pu_direct = signal_pu_slot1 / max(i_pr_su_slot1 + N0, 1e-30)
        
        # Slot 2 Relay transmission to PR
        signal_pu_slot2_relay = p_relay_pu * g_true["rp"]
        # Slot 2 PT interference (assuming PT transmits new data)
        interference_pr_slot2 = p_pt * g_true["pp"] + i_pr_relay_su_slot2
        snr_pu_relayed_hop2 = signal_pu_slot2_relay / max(interference_pr_slot2 + N0, 1e-30)
        
        # End-to-end relayed PU SNR
        snr_pu_relayed = min(gamma_relay_pu, snr_pu_relayed_hop2)
        
        # Selection Combining at PR
        sinr_pu = max(snr_pu_direct, snr_pu_relayed)
        pu_rate = 0.5 * math.log2(1.0 + sinr_pu)

        # ==============================================================
        # Reward
        # ==============================================================
        reward = sum_rate
        if constraint_violated:
            excess = interference_at_pr - cfg.interference_threshold
            reward -= cfg.penalty_weight * excess

        # ==============================================================
        # Advance state
        # ==============================================================
        self._step_count += 1
        truncated = self._step_count >= cfg.max_steps

        # Generate new channels for the next observation
        self._eff_gains_true, self._eff_gains_est = self._generate_all_channels()
        obs = self._build_observations()

        info: Dict[str, Any] = {
            "step": self._step_count,
            "sum_rate": sum_rate,
            "per_user_rates": rates.tolist(),
            "gamma_sr": gamma_sr.tolist(),
            "gamma_rd": float(gamma_sud_su),
            "gamma_e2e": gamma_e2e.tolist(),
            "sic_order": sic_order.tolist(),
            "interference_at_pr": float(interference_at_pr),
            "interference_threshold": float(cfg.interference_threshold),
            "constraint_violated": bool(constraint_violated),
            "pu_rate": float(pu_rate),
            "sinr_pu": float(sinr_pu),
            "snr_pu_direct": float(snr_pu_direct),
            "snr_pu_relayed": float(snr_pu_relayed),
            "p_su_watts": p_su.tolist(),
            "p_relay_watts": float(p_relay),
            "alpha": float(alpha),
            "effective_gains_true": dict(self._eff_gains_true),
            "effective_gains_est": dict(self._eff_gains_est),
        }

        return {
            "observations": obs,
            "reward": float(reward),
            "terminated": False,
            "truncated": truncated,
            "info": info,
        }

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observations(self) -> np.ndarray:
        """Build the (N, 8) observation matrix.

        Per agent *i* the row is:
            [h_sr_i, h_sp_i, h_sd_i, h_pp, h_pr, h_pd, h_rd, h_rp]

        Returns:
            Numpy float32 array of shape ``(N, 8)``.
        """
        N = self.cfg.num_su
        g_est = self._eff_gains_est

        # Common features (same for every agent)
        common = np.array(
            [g_est["pp"], g_est["pr_link"], g_est["pd"], g_est["rd"], g_est["rp"]],
            dtype=np.float32,
        )

        obs = np.zeros((N, self.OBS_DIM), dtype=np.float32)
        for i in range(N):
            obs[i, 0] = g_est[f"sr_{i}"]
            obs[i, 1] = g_est[f"sp_{i}"]
            obs[i, 2] = g_est[f"sd_{i}"]
            obs[i, 3:] = common

        # Convert raw gains (~1e-10 to 1e-4) into decibels
        # Then min-max normalize to roughly [-1.0, 1.0] range
        obs_db = 10.0 * np.log10(np.clip(obs, 1e-15, None))
        obs_normalized = (obs_db + 70.0) / 50.0

        return obs_normalized.astype(np.float32)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        """Dimension of the joint action vector (N+1)."""
        return self.cfg.num_su + 1

    @property
    def obs_dim(self) -> int:
        """Per-agent observation dimension (8)."""
        return self.OBS_DIM

    @property
    def num_agents(self) -> int:
        """Number of secondary-user agents."""
        return self.cfg.num_su

    def get_distances(self) -> Dict[str, float]:
        """Return the pre-computed distance dictionary (debug helper)."""
        return dict(self._distances)

    def get_path_losses(self) -> Dict[str, float]:
        """Return the pre-computed path-loss dictionary (debug helper)."""
        return dict(self._path_losses)
