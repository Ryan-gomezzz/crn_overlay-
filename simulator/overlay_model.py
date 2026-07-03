"""
Overlay CRN System Model Integration.
Author: Ryan
"""

from typing import Any, Dict
import numpy as np

from .base_model import BaseSimulator
from .channels import RayleighFading
from .propagation import calculate_path_loss
from .relay import DecodeAndForward
from .interference import calculate_received_power, calculate_interference
from .metrics import calculate_sinr, calculate_throughput, calculate_ber
from .utils import dbm_to_watt


class OverlaySimulator(BaseSimulator):
    """
    Implementation of the Overlay Cognitive Radio Network Simulator.
    Integrates channel, relay, propagation, and interference modules.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simulator components with config parameters.
        """
        self.config = config

        # Network topology coordinates
        net_cfg = config.get("network", {})
        self.pt_coords = np.array(net_cfg.get("pt_coords", [0.0, 0.0]))
        self.pr_coords = np.array(net_cfg.get("pr_coords", [100.0, 0.0]))
        self.sus_coords = np.array(net_cfg.get("sus_coords", [10.0, 20.0]))
        self.sur_coords = np.array(net_cfg.get("sur_coords", [50.0, 10.0]))
        self.sud_coords = np.array(net_cfg.get("sud_coords", [90.0, 20.0]))

        # Calculate distances
        self.d_pt_pr = np.linalg.norm(self.pt_coords - self.pr_coords)
        self.d_pt_sur = np.linalg.norm(self.pt_coords - self.sur_coords)
        self.d_pt_sud = np.linalg.norm(self.pt_coords - self.sud_coords)
        self.d_sus_sur = np.linalg.norm(self.sus_coords - self.sur_coords)
        self.d_sus_pr = np.linalg.norm(self.sus_coords - self.pr_coords)
        self.d_sur_pr = np.linalg.norm(self.sur_coords - self.pr_coords)
        self.d_sur_sud = np.linalg.norm(self.sur_coords - self.sud_coords)

        # Physical constants
        chan_cfg = config.get("channel", {})
        self.path_loss_exponent = chan_cfg.get("path_loss_exponent", 3.0)
        self.noise_power = dbm_to_watt(chan_cfg.get("noise_power_dbm", -114.0))

        # Transmit powers
        self.p_p = dbm_to_watt(net_cfg.get("p_primary", 30.0))
        self.p_max_su = dbm_to_watt(net_cfg.get("p_max_su", 20.0))

        # Fading model and relay protocol
        self.channel_model = RayleighFading()
        self.relay_protocol = DecodeAndForward()

        # Thresholds
        camo_cfg = config.get("camo_td3", {})
        self.decoding_threshold = camo_cfg.get("decoding_threshold", 1.0)

        # State storage
        self.channel_gains = {}

    def _generate_channels(self):
        """
        Generate small-scale fading and compute total channel gains.
        """
        self.channel_gains = {
            "pt_pr": self.channel_model.generate_gain(self.d_pt_pr, self.path_loss_exponent),
            "pt_sur": self.channel_model.generate_gain(self.d_pt_sur, self.path_loss_exponent),
            "pt_sud": self.channel_model.generate_gain(self.d_pt_sud, self.path_loss_exponent),
            "sus_sur": self.channel_model.generate_gain(self.d_sus_sur, self.path_loss_exponent),
            "sus_pr": self.channel_model.generate_gain(self.d_sus_pr, self.path_loss_exponent),
            "sur_pr": self.channel_model.generate_gain(self.d_sur_pr, self.path_loss_exponent),
            "sur_sud": self.channel_model.generate_gain(self.d_sur_sud, self.path_loss_exponent),
        }

    def reset(self) -> Dict[str, Any]:
        """
        Reset channels and return initial state observation.
        """
        self._generate_channels()
        # State vector: [|h_pt_pr|^2, |h_sus_sur|^2, |h_sur_sud|^2, |h_sus_pr|^2]
        state = np.array([
            self.channel_gains["pt_pr"],
            self.channel_gains["sus_sur"],
            self.channel_gains["sur_sud"],
            self.channel_gains["sus_pr"]
        ], dtype=np.float32)
        return {"state": state}

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Advance simulator by one step.
        action = [a_0, a_1] -> [power_fraction_sus, beta_power_fraction_sur]
        """
        # Generate new channels for this step (fading varies per step)
        self._generate_channels()

        # Parse actions
        a_0 = float(np.clip(action[0], 0.0, 1.0))
        a_1 = float(np.clip(action[1], 0.0, 1.0))

        # Physical power levels
        p_s1 = a_0 * self.p_max_su
        beta = a_1
        p_rel = self.p_max_su

        # Channel gains
        h_pt_pr = self.channel_gains["pt_pr"]
        h_pt_sur = self.channel_gains["pt_sur"]
        h_pt_sud = self.channel_gains["pt_sud"]
        h_sus_sur = self.channel_gains["sus_sur"]
        h_sus_pr = self.channel_gains["sus_pr"]
        h_sur_pr = self.channel_gains["sur_pr"]
        h_sur_sud = self.channel_gains["sur_sud"]

        # --- TIME SLOT 1 ---
        # PR receives from PT (signal) with interference from SUs
        rx_pt_pr = calculate_received_power(self.p_p, h_pt_pr)
        rx_sus_pr = calculate_received_power(p_s1, h_sus_pr)
        sinr_pr_ts1 = calculate_sinr(rx_pt_pr, rx_sus_pr, self.noise_power)

        # SUR receives from PT and SUs
        rx_pt_sur = calculate_received_power(self.p_p, h_pt_sur)
        rx_sus_sur = calculate_received_power(p_s1, h_sus_sur)
        sinr_pt_sur = calculate_sinr(rx_pt_sur, rx_sus_sur, self.noise_power)

        # Can the relay decode the primary signal?
        sur_decodes_primary = self.relay_protocol.can_decode(sinr_pt_sur, self.decoding_threshold)

        # Decode secondary signal at SUR (SIC of primary signal if successful)
        if sur_decodes_primary:
            sinr_sus_sur = calculate_sinr(rx_sus_sur, 0.0, self.noise_power)
        else:
            sinr_sus_sur = calculate_sinr(rx_sus_sur, rx_pt_sur, self.noise_power)

        # --- TIME SLOT 2 ---
        # SUR forwards combined signal if it successfully decodes both, otherwise it forwards nothing (or only primary)
        # For simplicity in this base model: if SUR can decode, it forwards. Otherwise beta=0, p_rel=0.
        sur_decodes_secondary = self.relay_protocol.can_decode(sinr_sus_sur, self.decoding_threshold)
        
        if sur_decodes_primary and sur_decodes_secondary:
            p_rel_p = beta * p_rel
            p_rel_s = (1.0 - beta) * p_rel
        else:
            p_rel_p = 0.0
            p_rel_s = 0.0

        # PR receives from PT and SUR in TS2
        rx_pt_pr_ts2 = calculate_received_power(self.p_p, h_pt_pr)
        rx_sur_pr_ts2 = calculate_received_power(p_rel_p, h_sur_pr)
        interference_pr_ts2 = calculate_received_power(p_rel_s, h_sur_pr)

        # Coherent combining of primary signal from PT and SUR
        coherent_primary_signal = (np.sqrt(rx_pt_pr_ts2) + np.sqrt(rx_sur_pr_ts2)) ** 2
        sinr_pr_ts2 = calculate_sinr(coherent_primary_signal, interference_pr_ts2, self.noise_power)

        # SUd receives secondary signal in TS2
        rx_sur_sud_ts2 = calculate_received_power(p_rel_s, h_sur_sud)
        rx_pt_sud_ts2 = calculate_received_power(self.p_p, h_pt_sud)
        interference_sud_ts2 = calculate_received_power(p_rel_p, h_sur_sud)

        # SUd decodes secondary signal. The primary signals act as interference.
        total_interference_sud = rx_pt_sud_ts2 + interference_sud_ts2
        sinr_sud_ts2 = calculate_sinr(rx_sur_sud_ts2, total_interference_sud, self.noise_power)

        # --- PERFORMANCE METRICS ---
        rate_p = calculate_throughput(sinr_pr_ts1, time_fraction=0.5) + calculate_throughput(sinr_pr_ts2, time_fraction=0.5)
        
        # Secondary rate is bottlenecked by the two hops
        rate_s1 = calculate_throughput(sinr_sus_sur, time_fraction=0.5)
        rate_s2 = calculate_throughput(sinr_sud_ts2, time_fraction=0.5)
        rate_s = min(rate_s1, rate_s2)

        # BER at SUd
        ber_s = calculate_ber(sinr_sud_ts2)

        # Next state
        next_state = np.array([
            h_pt_pr,
            h_sus_sur,
            h_sur_sud,
            h_sus_pr
        ], dtype=np.float32)

        return {
            "next_state": next_state,
            "metrics": {
                "throughput_p": rate_p,
                "throughput_s": rate_s,
                "ber_s": ber_s,
                "sinr_pr_ts1": sinr_pr_ts1,
                "sinr_pr_ts2": sinr_pr_ts2,
                "sinr_sud": sinr_sud_ts2,
                "power_s1": p_s1,
                "power_rel": p_rel,
                "beta": beta,
                "relay_decoded": float(sur_decodes_primary and sur_decodes_secondary),
            }
        }
