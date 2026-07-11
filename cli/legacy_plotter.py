import os
import math
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import erfc

TD3_COLOR = "#1f77b4"
OVERLAY_TD3_COLOR = "#2ca02c"
UNDERLAY_TD3_COLOR = "#d62728"
MATD3_COLOR = "#9467bd"
CENT_NOMA_TD3_COLOR = "#ff7f0e"
ALGO_COLORS = {"TD3": TD3_COLOR, "OVERLAY_TD3": OVERLAY_TD3_COLOR, "UNDERLAY_TD3": UNDERLAY_TD3_COLOR, "MATD3": MATD3_COLOR, "CENT_NOMA_TD3": CENT_NOMA_TD3_COLOR}
ALPHA_FILL = 0.15

@dataclass
class RunMetrics:
    name: str
    episodes: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    su_throughputs: List[float] = field(default_factory=list)
    pu_throughputs: List[float] = field(default_factory=list)
    outage_probs: List[float] = field(default_factory=list)
    avg_bers: List[float] = field(default_factory=list)
    sinr_db_pts: List[float] = field(default_factory=list)
    ber_pts: List[float] = field(default_factory=list)
    pu_sinr_db_pts: List[float] = field(default_factory=list)
    pu_ber_pts: List[float] = field(default_factory=list)
    avg_pu_bers: List[float] = field(default_factory=list)
    
    final_avg_reward: float = 0.0
    final_avg_su_tput: float = 0.0
    final_avg_pu_tput: float = 0.0
    final_outage_prob: float = 0.0
    final_avg_ber: float = 0.0
    final_avg_pu_ber: float = 0.0
    training_time_sec: float = 0.0

def theoretical_bpsk_ber(snr_db: np.ndarray) -> np.ndarray:
    snr_lin = 10.0 ** (snr_db / 10.0)
    return 0.5 * erfc(np.sqrt(snr_lin))

def nakagami_avg_ber_bpsk(snr_db: np.ndarray, m: float = 1.0) -> np.ndarray:
    snr_lin = 10.0 ** (snr_db / 10.0)
    m_int = int(round(m))
    mu = np.sqrt(snr_lin / (m_int + snr_lin))
    ber = np.zeros_like(snr_lin)
    coeff_base = ((1.0 - mu) / 2.0) ** m_int
    for k in range(m_int):
        binom = math.comb(m_int - 1 + k, k)
        ber += coeff_base * binom * ((1.0 + mu) / 2.0) ** k
    return np.clip(ber, 1e-12, 0.5)

def smooth(values: List[float], window: int = 20) -> np.ndarray:
    if len(values) == 0: return np.array([])
    arr = np.array(values, dtype=float)
    if window >= len(arr): return arr
    kernel = np.ones(window) / window
    pad = np.pad(arr, (window // 2, window - window // 2 - 1), mode="edge")
    return np.convolve(pad, kernel, mode="valid")

def _color_for(m):
    return ALGO_COLORS.get(m.name.upper(), "#555555")

def _binned_mean(x_pts, y_pts, lo=-5, hi=25, n_bins=25):
    pts = np.column_stack([x_pts, y_pts])
    bins = np.linspace(lo, hi, n_bins)
    bx, by = [], []
    for i in range(len(bins) - 1):
        mask = (pts[:, 0] >= bins[i]) & (pts[:, 0] < bins[i + 1])
        if mask.sum() > 0:
            bx.append((bins[i] + bins[i + 1]) / 2)
            by.append(np.mean(pts[mask, 1]))
    return bx, by

def generate_legacy_pdf(all_metrics: List[RunMetrics], output_path: str, n_episodes: int, steps_per_ep: int, nakagami_m: float = 1.0, pu_sinr_threshold_linear: float = 1.2589, outage_desc: str = "Outage Penalty"):
    all_metrics = [m for m in all_metrics if m.rewards]
    if not all_metrics:
        print("No metrics to report.")
        return
        
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 12, "legend.fontsize": 10, "lines.linewidth": 1.8,
    })
    
    episodes = np.arange(1, n_episodes + 1)
    algo_names = " vs ".join(m.name for m in all_metrics)
    
    with PdfPages(output_path) as pdf:
        # PAGE 1 — Title + Summary Table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        fig.text(0.5, 0.88, f"CRN Power Control: {algo_names}\nNakagami-m (m={nakagami_m:g}) Fading Channel Performance Report", ha="center", va="top", fontsize=18, fontweight="bold")
        pu_thresh_db = 10 * np.log10(pu_sinr_threshold_linear) if pu_sinr_threshold_linear > 0 else 0
        fig.text(0.5, 0.80, f"Nakagami-m = {nakagami_m}  |  Episodes = {n_episodes}  |  Steps/Episode = {steps_per_ep}  |  PU SINR Threshold = {pu_thresh_db:.1f} dB", ha="center", va="top", fontsize=11, color="#444444")
        
        col_labels = ["Metric"] + [m.name for m in all_metrics] + ["Winner"]
        metric_fields = [
            ("Avg Reward (last 100 ep)", "final_avg_reward", "max", ".4f"),
            ("SU Throughput (bits/s/Hz)","final_avg_su_tput", "max", ".4f"),
            ("PU Throughput (bits/s/Hz)","final_avg_pu_tput", "max", ".4f"),
            ("Outage Probability",       "final_outage_prob", "min", ".4f"),
            ("Average BER",              "final_avg_ber",     "min", ".6f"),
            ("Training Time (s)",        "training_time_sec", None,  ".1f"),
        ]
        
        metrics_rows = []
        for label, attr, best_fn, fmt in metric_fields:
            vals = [getattr(m, attr) for m in all_metrics]
            row = [label] + [f"{v:{fmt}}" for v in vals]
            if best_fn == "max": winner = all_metrics[int(np.argmax(vals))].name
            elif best_fn == "min": winner = all_metrics[int(np.argmin(vals))].name
            else: winner = "-"
            row.append(winner)
            metrics_rows.append(row)
            
        table = ax.table(cellText=metrics_rows, colLabels=col_labels, cellLoc="center", loc="center", bbox=[0.02, 0.08, 0.96, 0.62])
        table.auto_set_font_size(False)
        table.set_fontsize(10 if len(all_metrics) > 2 else 11)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#2c3e50")
            table[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(metrics_rows) + 1):
            winner = metrics_rows[i - 1][-1]
            cell = table[i, len(col_labels) - 1]
            wcolor = ALGO_COLORS.get(winner.upper())
            if wcolor:
                cell.set_facecolor("#e8f4e8")
                cell.set_text_props(color=wcolor, fontweight="bold")
            bg = "#f8f8f8" if i % 2 == 0 else "white"
            for j in range(len(col_labels) - 1):
                table[i, j].set_facecolor(bg)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        
        # PAGE 2 — SU SINR vs BER
        snr_range = np.linspace(-5, 25, 300)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(snr_range, theoretical_bpsk_ber(snr_range), "k--", linewidth=1.4, label="BPSK Theoretical (AWGN)", alpha=0.7)
        ax.semilogy(snr_range, nakagami_avg_ber_bpsk(snr_range, nakagami_m), "k-.", linewidth=1.4, label=f"BPSK Avg BER (Nakagami-m={nakagami_m:g})", alpha=0.7)
        for m_obj in all_metrics:
            color = _color_for(m_obj)
            if m_obj.sinr_db_pts:
                ax.scatter(m_obj.sinr_db_pts, m_obj.ber_pts, color=color, alpha=0.35, s=10, label=f"{m_obj.name} (simulated)")
                bx, by = _binned_mean(m_obj.sinr_db_pts, m_obj.ber_pts)
                if bx: ax.semilogy(bx, by, color=color, linewidth=2.2, marker="o", markersize=4, label=f"{m_obj.name} Mean BER")
        ax.set_xlabel("SINR (dB)"); ax.set_ylabel("Bit Error Rate (BER)")
        ax.set_title(f"SINR vs BER - Nakagami-m={nakagami_m:g} Fading Channel\n(BPSK Modulation, Secondary User)")
        ax.set_xlim(-5, 25); ax.set_ylim(1e-6, 0.6); ax.legend(loc="lower left"); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        
        # PAGE 3 — PU SINR vs BER
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(snr_range, theoretical_bpsk_ber(snr_range), "k--", linewidth=1.4, label="BPSK Theoretical (AWGN)", alpha=0.7)
        ax.semilogy(snr_range, nakagami_avg_ber_bpsk(snr_range, nakagami_m), "k-.", linewidth=1.4, label=f"BPSK Avg BER (Nakagami-m={nakagami_m:g})", alpha=0.7)
        for m_obj in all_metrics:
            color = _color_for(m_obj)
            if m_obj.pu_sinr_db_pts:
                ax.scatter(m_obj.pu_sinr_db_pts, m_obj.pu_ber_pts, color=color, alpha=0.25, s=10, label=f"{m_obj.name} PU (simulated)")
                bx, by = _binned_mean(m_obj.pu_sinr_db_pts, m_obj.pu_ber_pts)
                if bx: ax.semilogy(bx, by, color=color, linewidth=2.2, marker="s", markersize=4, label=f"{m_obj.name} PU Mean BER")
        ax.axvline(x=pu_thresh_db, color="green", ls="--", lw=1.4, label=f"PU SINR Threshold ({pu_sinr_threshold_linear:.2f} linear)")
        ax.set_xlabel("PU SINR (dB)"); ax.set_ylabel("Bit Error Rate (BER)")
        ax.set_title(f"Primary User SINR vs BER - Nakagami-m={nakagami_m:g} Fading Channel\n(BPSK Modulation | Effect of SU Interference)")
        ax.set_xlim(-5, 25); ax.set_ylim(1e-6, 0.6); ax.legend(loc="lower left"); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        
        # PAGE 4 — SU Throughput
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for m_obj in all_metrics:
            color = _color_for(m_obj)
            raw = np.array(m_obj.su_throughputs); sm = smooth(raw, window=20)
            eps_arr = np.array(m_obj.episodes) if len(m_obj.episodes) == len(raw) else np.arange(1, len(raw) + 1)
            ax.plot(eps_arr, raw, color=color, alpha=0.25, linewidth=0.8)
            ax.plot(eps_arr, sm,  color=color, linewidth=2.0, label=f"{m_obj.name} (smoothed)")
            ax.fill_between(eps_arr, np.maximum(0, sm - raw.std()), sm + raw.std(), color=color, alpha=ALPHA_FILL)
        ax.set_xlabel("Episode"); ax.set_ylabel("Throughput (bits/s/Hz)"); ax.set_title("Secondary User (SU) Throughput vs Episodes")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        
        # PAGE 5 — PU Throughput
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for m_obj in all_metrics:
            color = _color_for(m_obj)
            raw = np.array(m_obj.pu_throughputs); sm = smooth(raw, window=20)
            if len(raw) > 0:
                eps_arr = np.array(m_obj.episodes) if len(m_obj.episodes) == len(raw) else np.arange(1, len(raw) + 1)
                ax.plot(eps_arr, raw, color=color, alpha=0.25, linewidth=0.8)
                ax.plot(eps_arr, sm,  color=color, linewidth=2.0, label=f"{m_obj.name} (smoothed)")
                ax.fill_between(eps_arr, np.maximum(0, sm - raw.std()), sm + raw.std(), color=color, alpha=ALPHA_FILL)
        ax.set_xlabel("Episode"); ax.set_ylabel("Throughput (bits/s/Hz)"); ax.set_title("Primary User (PU) Throughput vs Episodes")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        
        # PAGE 6 — Outage Probability
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for m_obj in all_metrics:
            color = _color_for(m_obj)
            raw = np.array(m_obj.outage_probs); sm = smooth(raw, window=20)
            eps_arr = np.array(m_obj.episodes) if len(m_obj.episodes) == len(raw) else np.arange(1, len(raw) + 1)
            ax.plot(eps_arr, raw, color=color, alpha=0.25, linewidth=0.8)
            ax.plot(eps_arr, sm,  color=color, linewidth=2.0, label=f"{m_obj.name} (smoothed)")
        ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=1.2, label="5% target")
        ax.set_xlabel("Episode"); ax.set_ylabel("Outage Probability")
        ax.set_title(f"Outage Probability vs Episodes\n({outage_desc})")
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.0)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        
        # PAGE 7 — Individual Reward Curves
        n_algos = len(all_metrics)
        cols = min(n_algos, 3)
        rows_grid = math.ceil(n_algos / cols)
        fig, axes = plt.subplots(rows_grid, cols, figsize=(5 * cols, 5 * rows_grid), sharey=False, squeeze=False)
        for idx, m_obj in enumerate(all_metrics):
            ax = axes[idx // cols][idx % cols]
            color = _color_for(m_obj)
            raw = np.array(m_obj.rewards); sm = smooth(raw, window=20)
            eps_arr = np.array(m_obj.episodes) if len(m_obj.episodes) == len(raw) else np.arange(1, len(raw) + 1)
            ax.plot(eps_arr, raw, color=color, alpha=0.2, linewidth=0.7)
            ax.plot(eps_arr, sm,  color=color, linewidth=2.2, label="Smoothed reward")
            ax.set_xlabel("Episode"); ax.set_ylabel("Episode Reward")
            ax.set_title(f"{m_obj.name} Reward Curve"); ax.legend(); ax.grid(True, alpha=0.3)
        for idx in range(n_algos, rows_grid * cols): axes[idx // cols][idx % cols].set_visible(False)
        fig.suptitle(f"Episode Reward Curves - {algo_names}", fontsize=14, y=1.01)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        
        # PAGE 8 — Combined overlay reward comparison
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for m_obj in all_metrics:
            color = _color_for(m_obj)
            raw = np.array(m_obj.rewards); sm = smooth(raw, window=20)
            eps_arr = np.array(m_obj.episodes) if len(m_obj.episodes) == len(raw) else np.arange(1, len(raw) + 1)
            ax.plot(eps_arr, raw, color=color, alpha=0.18, linewidth=0.7)
            ax.plot(eps_arr, sm,  color=color, linewidth=2.2, label=f"{m_obj.name}")
        ax.set_xlabel("Episode"); ax.set_ylabel("Episode Reward")
        ax.set_title(f"{algo_names} - Reward Comparison")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
