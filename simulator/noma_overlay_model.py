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
from simulator.utils import ber_bpsk_theory, df_ber_theory
from simulator.ber_waveform import simulate_waveform_df_noma_ber


# =====================================================================
# Configuration dataclass for the multi-user NOMA model
# =====================================================================


@dataclass
class NOMAConfig:
    """Configuration for the multi-user NOMA overlay CRN simulator.

    All coordinates are in metres.  Distances are computed internally
    from coordinates using Euclidean distance.

    Attributes:
        num_su: Total number of secondary users (N), INCLUDING the relay-SU.
        pt_coords: Primary Transmitter location [x, y].
        pr_coords: Primary Receiver location [x, y].
        su_coords: Coordinates of the N-1 SU *sources* [[x1,y1], ...].
        sud_coords: SU destination coordinate [x, y].
        sur_coords: Coordinate of SU_N, the relay-SU (relays + sends own data).
        relay_fwd_mu: Geometric power-split ratio across relayed streams.
        p_primary_dbm: PT transmit power in dBm.
        p_max_su_dbm: Max SU/relay transmit power in dBm.
        noise_power_dbm: AWGN noise power in dBm.
        path_loss_exponent: Path-loss exponent α.
        interference_threshold_dbm: Max tolerable interference at PR (dBm).
        max_steps: Max time-steps per episode.
        penalty_weight: Penalty multiplier for constraint violation.
    """

    # Total secondary users (N), including the relay-SU
    num_su: int = 3

    # Node positions (metres). su_coords holds the N-1 SU sources; sur_coords is
    # SU_N, the relay-SU (one of the N secondary users, not a separate node).
    pt_coords: List[float] = field(default_factory=lambda: [0.0, 0.0])
    pr_coords: List[float] = field(default_factory=lambda: [100.0, 0.0])
    su_coords: List[List[float]] = field(
        default_factory=lambda: [[10.0, 20.0], [30.0, 25.0]]
    )
    sud_coords: List[float] = field(default_factory=lambda: [90.0, 20.0])
    sur_coords: List[float] = field(default_factory=lambda: [50.0, 10.0])

    # Fixed geometric power split across the relayed streams in slot 2
    relay_fwd_mu: float = 0.5

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

    @property
    def num_sources(self) -> int:
        """Number of SU *sources* (N-1); the remaining SU is the relay-SU."""
        return max(self.num_su - 1, 0)

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
    A vector of ``N+2`` values in [0, 1]:
        [p_su_1, p_su_2, …, p_su_N, p_relay, alpha]
    The ``N`` SU powers and the relay power are scaled by ``p_max_su``
    (Watts) internally; ``alpha`` is the relay PU/SU power-splitting factor.

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

        M = self.cfg.num_sources
        if len(self.cfg.su_coords) < M:
            raise ValueError(
                f"su_coords has {len(self.cfg.su_coords)} entries but "
                f"num_su={self.cfg.num_su} (incl. the relay-SU) requires "
                f"{M} source coordinates."
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
        self._eff_channels_true: Dict[str, complex] = {}
        # Physical parameters of the most recent transmission (for waveform BER)
        self._last_tx: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Distance & path-loss pre-computation
    # ------------------------------------------------------------------

    def _precompute_distances(self) -> None:
        """Compute Euclidean distances and path losses for all links."""
        cfg = self.cfg
        M = cfg.num_sources
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

        # Per-source links (the relay-SU is SU_N and has links rd / rp instead)
        for i in range(M):
            su_i = cfg.su_coords[i]
            self._set_link(f"sr_{i}", su_i, relay, alpha)   # h_sr_i (source -> relay-SU)
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

        Also stores the *complex* true effective channel h = g·√PL for each link
        in ``self._eff_channels_true`` so the waveform-level BER simulator can
        reconstruct the exact received signals for this realization.
        """
        eff_true: Dict[str, float] = {}
        eff_est: Dict[str, float] = {}
        eff_complex_true: Dict[str, complex] = {}
        tau = self.cfg.csi_error_variance

        for name, pl in self._path_losses.items():
            g = self._fading.generate_coefficient(self._rng)
            e = self._fading.generate_coefficient(self._rng)

            # Imperfect CSI estimate
            g_hat = math.sqrt(1.0 - tau**2) * g + tau * e

            eff_true[name] = float(abs(g) ** 2) * pl
            eff_est[name] = float(abs(g_hat) ** 2) * pl
            eff_complex_true[name] = g * math.sqrt(pl)

        self._eff_channels_true = eff_complex_true
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
            action: Shape ``(N+2,)`` array in [0, 1], where ``N`` is the total
                number of secondary users (the relay-SU included) and
                ``M = N-1`` is the number of SU sources:
                ``action[0:M]``   → normalised transmit powers of the SU sources.
                ``action[M]``     → normalised transmit power of the relay-SU.
                ``action[M+1]``   → ``alpha``: relay power fraction given to the PU.
                ``action[M+2]``   → ``own_share``: fraction of the relay-SU's
                                    secondary power spent on its OWN data (the
                                    remainder forwards the decoded sources).

        Returns:
            Dictionary with keys:
                ``"observations"`` — (N, 8) per-agent observation matrix
                ``"reward"``       — scalar sum-rate reward
                ``"terminated"``   — bool (always False for now)
                ``"truncated"``    — bool (True when max_steps reached)
                ``"info"``         — detailed diagnostic dict
        """
        cfg = self.cfg
        N = cfg.num_su          # total SUs, including the relay-SU
        M = cfg.num_sources     # SU sources (N-1)
        action = np.asarray(action, dtype=np.float64).flatten()

        # Pad or truncate action to (M+3,) == (N+2,)
        n_act = M + 3
        if action.shape[0] < n_act:
            action = np.pad(action, (0, n_act - action.shape[0]))
        action = np.clip(action[:n_act], 0.0, 1.0)

        # Convert normalised actions → Watts / fractions
        p_src = action[:M] * cfg.p_max_su           # source powers, shape (M,)
        p_relay = float(action[M]) * cfg.p_max_su   # relay-SU total power
        alpha = float(action[M + 1])                # PU share of the relay power
        own_share = float(action[M + 2])            # relay's own-data share
        p_pt = cfg.p_pt
        N0 = cfg.noise_power
        g_true = self._eff_gains_true  # shorthand for physics

        # ==============================================================
        # TIME SLOT 1 — SU sources → relay-SU  (+ PT → relay-SU)
        # The relay-SU is half-duplex: it listens here, it does not transmit.
        # ==============================================================

        rx_power_pr = p_pt * g_true["pr_link"]
        rx_power_sr = np.array(
            [p_src[i] * g_true[f"sr_{i}"] for i in range(M)], dtype=np.float64
        )
        total_su_rx = float(rx_power_sr.sum())

        # Relay-SU decodes the PU first (sources treated as interference)
        gamma_relay_pu = rx_power_pr / max(total_su_rx + N0, 1e-30)

        # Then decodes the sources with SIC (PU already cancelled)
        sic_order = np.argsort(-rx_power_sr) if M > 0 else np.zeros(0, dtype=int)

        gamma_sr = np.zeros(M, dtype=np.float64)
        for rank, idx in enumerate(sic_order):
            later_users = sic_order[rank + 1:]
            noma_residual = float(rx_power_sr[later_users].sum()) if len(later_users) > 0 else 0.0
            # PU is already decoded and subtracted, so no pt_interference_relay here
            denom = noma_residual + N0
            gamma_sr[idx] = rx_power_sr[idx] / max(denom, 1e-30)

        # ==============================================================
        # TIME SLOT 2 — Relay → SU Dest  (+ PT → SU Dest as interference)
        # ==============================================================
        
        p_relay_pu = alpha * p_relay                     # cooperative PU forwarding
        p_relay_su = (1.0 - alpha) * p_relay             # relay-SU secondary payload
        p_own = own_share * p_relay_su                   # relay-SU's OWN data
        p_fwd_total = (1.0 - own_share) * p_relay_su     # forwarded source data

        # Fixed geometric split of the forwarded power across the M relayed
        # streams, assigned in the relay's SIC order (strongest source first).
        beta = np.zeros(M, dtype=np.float64)
        if M > 0:
            w = cfg.relay_fwd_mu ** np.arange(M, dtype=np.float64)
            w = w / w.sum()
            for rank, idx in enumerate(sic_order):
                beta[idx] = w[rank]
        p_fwd = beta * p_fwd_total                       # per-source forwarded power

        interference_dest_pt = p_pt * g_true["pd"]
        g_rd = g_true["rd"]

        # Destination SIC over the superposed streams (PU, the relay-SU's own
        # data, and the M relayed sources) on the common relay->dest channel:
        # decode in descending received power, weaker streams as interference.
        stream_powers = np.concatenate(([p_relay_pu, p_own], p_fwd))
        rx_streams = stream_powers * g_rd
        dest_order = np.argsort(-rx_streams)
        gamma_streams = np.zeros_like(rx_streams)
        for rank, idx in enumerate(dest_order):
            weaker = dest_order[rank + 1:]
            resid = float(rx_streams[weaker].sum()) if len(weaker) > 0 else 0.0
            gamma_streams[idx] = rx_streams[idx] / max(
                resid + interference_dest_pt + N0, 1e-30
            )
        gamma_sud_pu = float(gamma_streams[0])   # relayed PU at the destination
        gamma_own = float(gamma_streams[1])      # relay-SU's own data
        gamma_fwd = gamma_streams[2:]            # per-source hop-2 SINR

        # ==============================================================
        # Rates: sources are Decode-and-Forward (two hops); the relay-SU's own
        # data is single-hop (slot 2 only).
        # ==============================================================

        gamma_e2e = np.minimum(gamma_sr, gamma_fwd) if M > 0 else np.zeros(0)
        rates_src = 0.5 * np.log2(1.0 + gamma_e2e)          # shape (M,)
        rate_own = 0.5 * math.log2(1.0 + gamma_own)
        rates = np.concatenate((rates_src, [rate_own]))     # shape (N,)
        sum_rate = float(rates.sum())

        # Per-user BER. Sources use two-hop DF, so an end-to-end bit errs iff it
        # is flipped on an odd number of hops: P_e2e = P1 + P2 - 2*P1*P2.
        # The relay-SU's own data traverses a single hop.
        ber_src_hop1 = ber_bpsk_theory(gamma_sr) if M > 0 else np.zeros(0)
        ber_fwd_hop2 = ber_bpsk_theory(gamma_fwd) if M > 0 else np.zeros(0)
        ber_src_e2e = ber_src_hop1 + ber_fwd_hop2 - 2.0 * ber_src_hop1 * ber_fwd_hop2
        ber_own = float(ber_bpsk_theory(gamma_own))
        ber_su_hop1_per_user = np.concatenate((ber_src_hop1, [ber_own]))
        ber_su_e2e_per_user = np.concatenate((ber_src_e2e, [ber_own]))

        # ==============================================================
        # Interference constraint at PR
        # ==============================================================
        # Slot 1: the M sources. Slot 2: the relay-SU's secondary payload
        # (own + forwarded). The relayed PU signal is NOT interference to PR.

        i_pr_su_slot1 = float(sum(p_src[i] * g_true[f"sp_{i}"] for i in range(M)))
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

        # Primary BER: the PR selection-combines a single-hop direct link with a
        # two-hop DF-relayed link, i.e. it takes whichever path yields the lower
        # BER.  ber_pu_hop1 is the relay's decode of the PU (hop 1 of the relayed path).
        ber_pu_direct = float(ber_bpsk_theory(snr_pu_direct))
        ber_pu_hop1 = float(ber_bpsk_theory(gamma_relay_pu))
        _, ber_pu_relayed = df_ber_theory(gamma_relay_pu, snr_pu_relayed_hop2)
        ber_pu = min(ber_pu_direct, ber_pu_relayed)

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

        # Capture this transmission's complex channels + powers so the waveform
        # BER simulator can reconstruct the received signals for THIS realization
        # (must happen before the channels are regenerated below).
        ch = self._eff_channels_true
        self._last_tx = {
            "h_sr": np.array([ch[f"sr_{i}"] for i in range(M)]),
            "h_sp": np.array([ch[f"sp_{i}"] for i in range(M)]),
            "h_pr": ch["pr_link"],
            "h_pp": ch["pp"],
            "h_pd": ch["pd"],
            "h_rd": ch["rd"],
            "h_rp": ch["rp"],
            "p_src": p_src.copy(),
            "p_relay": p_relay,
            "alpha": alpha,
            "p_own": p_own,
            "p_fwd": p_fwd.copy(),
            "p_pt": p_pt,
            "n0": N0,
            "sic_order": sic_order.copy(),
            "pu_use_direct": bool(snr_pu_direct >= snr_pu_relayed),
        }

        # Generate new channels for the next observation
        self._eff_gains_true, self._eff_gains_est = self._generate_all_channels()
        obs = self._build_observations()

        info: Dict[str, Any] = {
            "step": self._step_count,
            "sum_rate": sum_rate,
            "per_user_rates": rates.tolist(),
            "gamma_sr": gamma_sr.tolist(),
            "gamma_rd": float(np.mean(gamma_fwd)) if M > 0 else 0.0,
            "gamma_fwd": gamma_fwd.tolist(),
            "gamma_own": float(gamma_own),
            "gamma_sud_pu": float(gamma_sud_pu),
            "gamma_e2e": gamma_e2e.tolist(),
            "rate_own": float(rate_own),
            "sic_order": sic_order.tolist(),
            "dest_order": dest_order.tolist(),
            "interference_at_pr": float(interference_at_pr),
            "interference_threshold": float(cfg.interference_threshold),
            "constraint_violated": bool(constraint_violated),
            "pu_rate": float(pu_rate),
            "sinr_pu": float(sinr_pu),
            "snr_pu_direct": float(snr_pu_direct),
            "snr_pu_relayed": float(snr_pu_relayed),
            "gamma_relay_pu": float(gamma_relay_pu),
            "snr_pu_relayed_hop2": float(snr_pu_relayed_hop2),
            # --- Bit-error rates (two-hop Decode-and-Forward, per hop + e2e) ---
            "ber_su_hop1_per_user": ber_su_hop1_per_user.tolist(),  # relay decode
            "ber_su_hop1": float(np.mean(ber_su_hop1_per_user)),
            "ber_su_per_user": ber_su_e2e_per_user.tolist(),        # end-to-end
            "ber_su": float(np.mean(ber_su_e2e_per_user)),
            "ber_pu_hop1": ber_pu_hop1,
            "ber_pu": ber_pu,
            "ber_pu_direct": ber_pu_direct,
            "ber_pu_relayed": float(ber_pu_relayed),
            "sinr_su_hop1_mean": float(np.mean(gamma_sr)) if M > 0 else 0.0,
            "sinr_su_hop2": float(np.mean(gamma_fwd)) if M > 0 else 0.0,
            "sinr_su_mean": float(np.mean(gamma_e2e)) if M > 0 else 0.0,
            "p_su_watts": p_src.tolist(),
            "p_relay_watts": float(p_relay),
            "p_own_watts": float(p_own),
            "p_fwd_watts": p_fwd.tolist(),
            "alpha": float(alpha),
            "own_share": float(own_share),
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
        M = self.cfg.num_sources
        g_est = self._eff_gains_est

        # Common features (same for every agent)
        common = np.array(
            [g_est["pp"], g_est["pr_link"], g_est["pd"], g_est["rd"], g_est["rp"]],
            dtype=np.float32,
        )

        obs = np.zeros((N, self.OBS_DIM), dtype=np.float32)
        # Rows 0..M-1: the SU sources (their transmit links).
        for i in range(M):
            obs[i, 0] = g_est[f"sr_{i}"]
            obs[i, 1] = g_est[f"sp_{i}"]
            obs[i, 2] = g_est[f"sd_{i}"]
            obs[i, 3:] = common
        # Row M: the relay-SU. Its own links are relay->dest, relay->PR and the
        # PT->relay link it must decode through.
        obs[M, 0] = g_est["rd"]
        obs[M, 1] = g_est["rp"]
        obs[M, 2] = g_est["pr_link"]
        obs[M, 3:] = common

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
        """Joint action dimension: (N-1) source powers + relay power + alpha
        + own_share, i.e. ``N+2`` where N counts the relay-SU."""
        return self.cfg.num_sources + 3

    @property
    def obs_dim(self) -> int:
        """Per-agent observation dimension (8)."""
        return self.OBS_DIM

    @property
    def num_agents(self) -> int:
        """Number of secondary-user agents."""
        return self.cfg.num_su

    def simulate_waveform_ber(
        self, n_bits: int = 4000, rng: Optional[np.random.Generator] = None
    ) -> Optional[Dict[str, Any]]:
        """Run the waveform-level (imperfect-SIC) DF-NOMA BER for the most recent
        transmission. Returns ``None`` if no step has been taken yet."""
        if self._last_tx is None:
            return None
        tx = self._last_tx
        return simulate_waveform_df_noma_ber(
            h_sr=tx["h_sr"], h_sp=tx["h_sp"], h_pr=tx["h_pr"], h_pp=tx["h_pp"],
            h_pd=tx["h_pd"], h_rd=tx["h_rd"], h_rp=tx["h_rp"],
            p_src=tx["p_src"], p_relay=tx["p_relay"], alpha=tx["alpha"],
            p_own=tx["p_own"], p_fwd=tx["p_fwd"],
            p_pt=tx["p_pt"], n0=tx["n0"], sic_order=tx["sic_order"],
            pu_use_direct=tx["pu_use_direct"], n_bits=n_bits, rng=rng,
        )

    def get_distances(self) -> Dict[str, float]:
        """Return the pre-computed distance dictionary (debug helper)."""
        return dict(self._distances)

    def get_path_losses(self) -> Dict[str, float]:
        """Return the pre-computed path-loss dictionary (debug helper)."""
        return dict(self._path_losses)
