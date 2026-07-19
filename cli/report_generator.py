"""
Report and Plot Generation Engine for the CRN Research Framework.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, List, Any, Optional

from simulator.utils import ber_bpsk_theory

# Premium color palette for all agents
COLORS = {
    "TD3": "#1f77b4",          # Blue
    "UNDERLAY_TD3": "#ff7f0e", # Orange
    "OVERLAY_TD3": "#2ca02c",  # Green
    "MATD3": "#9467bd",        # Purple
    "CENT_NOMA_TD3": "#d62728" # Red
}

SHORT_NAMES = {
    "TD3": "TD3",
    "UNDERLAY_TD3": "Underlay TD3",
    "OVERLAY_TD3": "Overlay TD3",
    "MATD3": "MATD3",
    "CENT_NOMA_TD3": "CENT_NOMA_TD3"
}

def load_metrics_for_agent(experiments_dir: str, agent: str) -> List[Dict[str, Any]]:
    """Scan and load all metrics.json files for a given agent."""
    agent_dir = os.path.join(experiments_dir, agent.lower())
    metrics_list = []
    if not os.path.exists(agent_dir):
        # Fallback in case folder is named differently
        pass

    if os.path.exists(agent_dir):
        for run_name in sorted(os.listdir(agent_dir)):
            run_path = os.path.join(agent_dir, run_name)
            metrics_file = os.path.join(run_path, "metrics.json")
            if os.path.isdir(run_path) and os.path.exists(metrics_file):
                try:
                    with open(metrics_file, "r") as f:
                        data = json.load(f)
                        data["run_name"] = run_name
                        data["agent"] = agent
                        metrics_list.append(data)
                except Exception as e:
                    print(f"Warning: Failed to load metrics from {metrics_file}: {e}")
    return metrics_list

def smooth_curve(points, factor=0.9):
    """Exponential moving average for smoothing."""
    smoothed = []
    for point in points:
        if smoothed:
            prev = smoothed[-1]
            smoothed.append(prev * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return np.array(smoothed)

# System bandwidth (Hz) used by the simulator to scale throughput.
# Throughput logged in metrics.json is time_fraction * B * log2(1+SINR) in bits/s;
# dividing by BANDWIDTH_HZ recovers spectral efficiency R = 1/2 * log2(1+SINR) in bits/s/Hz,
# which is exactly the achievable secondary rate R_s in the reference diagram.
BANDWIDTH_HZ = 1e6


def _agent_series(runs, key):
    """Return (episodes, mean_values) across a single agent's runs for a history key.

    Averages element-wise over runs (trimming to the shortest run). Returns
    ``(None, None)`` when no run carries the key. No smoothing or noise is added —
    the values are exactly what training measured.
    """
    series = []
    episode_axes = []
    for run in runs:
        hist = run.get("history", {})
        if key in hist and hist[key] and hist.get("episodes"):
            series.append(hist[key])
            episode_axes.append(hist["episodes"])
    if not series:
        return None, None
    min_len = min(len(s) for s in series)
    if min_len == 0:
        return None, None
    values = np.mean([s[:min_len] for s in series], axis=0)
    episodes = episode_axes[0][:min_len]
    return np.asarray(episodes), np.asarray(values)


def generate_comparison_plots(experiments_dir: str, output_plots_dir: str, agents=None):
    """Generate and save overlaid plots comparing the specified agents.
    If none provided, defaults to the legacy 3 agents.
    """
    os.makedirs(output_plots_dir, exist_ok=True)

    if agents is None:
        agents = ["MATD3"]
    metrics_by_agent = {a: load_metrics_for_agent(experiments_dir, a) for a in agents}
    active = [a for a in agents if metrics_by_agent[a]]

    plt.style.use('default')
    plt.rcParams.update({'figure.dpi': 300, 'axes.grid': True, 'grid.alpha': 0.3})

    # Plot 1: Episode return (learning curve) — real, with light EMA overlay
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    plotted = False
    for agent in active:
        ep, rew = _agent_series(metrics_by_agent[agent], "rewards")
        if rew is None:
            continue
        plotted = True
        ax1.plot(ep, rew, color=COLORS[agent], alpha=0.30, linewidth=0.8)
        ax1.plot(ep, smooth_curve(rew, factor=0.9), color=COLORS[agent],
                 linewidth=2, label=f'{SHORT_NAMES[agent]} (EMA)')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Mean Evaluation Return', fontsize=12)
    ax1.set_title('Learning Curve — Mean Evaluation Return vs Episode\n(measured, faint = raw eval points, bold = EMA smoothing)', fontsize=14)
    if plotted:
        ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_plots_dir, "real_reward.png"))
    plt.close(fig1)

    # Plot 2: Secondary spectral efficiency (bits/s/Hz) — real
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    plotted = False
    for agent in active:
        ep, th = _agent_series(metrics_by_agent[agent], "throughput_s")
        if th is None:
            continue
        plotted = True
        se = th
        ax2.plot(ep, se, color=COLORS[agent], alpha=0.30, linewidth=0.8)
        ax2.plot(ep, smooth_curve(se, factor=0.9), color=COLORS[agent],
                 linewidth=2, label=f'{SHORT_NAMES[agent]} (EMA)')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Secondary Rate $R_s$ (bits/s/Hz)', fontsize=12)
    ax2.set_title('Achievable Secondary Rate vs Episode\n$R_s = \\frac{1}{2}\\log_2(1+\\gamma_{e2e})$, measured during evaluation', fontsize=14)
    if plotted:
        ax2.legend(loc='best')
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_plots_dir, "real_su_throughput.png"))
    plt.close(fig2)

    # Plot 3: PU and SU outage probability vs episode — real
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    plotted = False
    for agent in active:
        ep, pu = _agent_series(metrics_by_agent[agent], "outage")
        if pu is not None:
            plotted = True
            ax3.plot(ep, pu, color=COLORS[agent], linewidth=2,
                     label=f'{SHORT_NAMES[agent]} — PU outage')
        ep_s, su = _agent_series(metrics_by_agent[agent], "su_outage")
        if su is not None:
            plotted = True
            ax3.plot(ep_s, su, color=COLORS[agent], linewidth=1.5, linestyle='--',
                     label=f'{SHORT_NAMES[agent]} — SU outage')
    ax3.set_ylim(-0.02, 1.0)
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Outage Probability', fontsize=12)
    ax3.set_title('Primary / Secondary Outage Probability vs Episode\n(measured; PU outage = interference at PR exceeds $I_{th}$)', fontsize=14)
    if plotted:
        ax3.legend(loc='best', fontsize=9)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_plots_dir, "real_outage.png"))
    plt.close(fig3)

def _run_summary(runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Collapse an agent's runs into a single real summary (mean over seeds)."""
    if not runs:
        return None
    def _mean(key):
        vals = [r.get(key) for r in runs if isinstance(r.get(key), (int, float))]
        return float(np.mean(vals)) if vals else None
    episodes = 0
    for r in runs:
        eps = r.get("history", {}).get("episodes") or []
        if eps:
            episodes = max(episodes, int(max(eps)))
    seeds = sorted({r.get("seed") for r in runs if r.get("seed") is not None})
    return {
        "eval_reward": _mean("eval_reward"),
        "eval_su_throughput": _mean("eval_su_throughput"),
        "eval_pu_outage": _mean("eval_pu_outage"),
        "eval_su_outage": _mean("eval_su_outage"),
        "eval_average_power": _mean("eval_average_power"),
        "eval_relay_success": _mean("eval_relay_success"),
        "train_time": _mean("train_time"),
        "inf_time": _mean("inf_time"),
        "episodes": episodes,
        "seeds": seeds,
        "n_runs": len(runs),
    }


def generate_markdown_report(experiments_dir: str, output_dir: str, agents=None, prefix="") -> str:
    """Create a markdown report summarizing the real measured metrics per agent."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{prefix}research_report.md")
    if agents is None:
        agents = ["MATD3"]
    lines = ["# CRN Overlay Framework — Experimental Report", "",
             "Metrics below are read directly from each run's `metrics.json`.", ""]
    lines.append("| Algorithm | Episodes | Seeds | Mean Return | SU Rate (bits/s/Hz) | PU Outage | SU Outage | Avg Power (W) | Relay Success | Train Time (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    if "MATD3" in agents:
        lines.append("")
        lines.append("> **Note:** The secondary sum-rate objective is optimized under an interference "
                     "constraint at the PR, enforced jointly by an environment penalty "
                     "(`camo_td3.penalty_coef_inf`) and adaptive Lagrangian multipliers "
                     "(QoS/energy). Exact values are recorded in each run's `config_snapshot.yaml`.")
        lines.append("")
    for agent in agents:
        s = _run_summary(load_metrics_for_agent(experiments_dir, agent))
        if not s:
            lines.append(f"| {SHORT_NAMES[agent]} | *no runs* | - | - | - | - | - | - |")
            continue
        se = s["eval_su_throughput"] if s["eval_su_throughput"] is not None else None
        lines.append(
            f"| {SHORT_NAMES[agent]} | {s['episodes']} | {s['seeds']} | "
            f"{s['eval_reward']:.3e} | {se:.4f} | {s['eval_pu_outage']:.4f} | "
            f"{(s['eval_su_outage'] if s['eval_su_outage'] is not None else 0.0):.4f} | "
            f"{(s['eval_average_power'] if s.get('eval_average_power') is not None else 0.0):.4f} | "
            f"{(s['eval_relay_success'] if s.get('eval_relay_success') is not None else 0.0):.4f} | "
            f"{s['train_time']:.1f} |"
        )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path

def generate_pdf_report(experiments_dir: str, output_dir: str, agents=None, prefix=""):
    """Generate a comprehensive multi-page PDF report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{prefix}research_report.pdf")
    
    if agents is None:
        agents = ["MATD3"]
    metrics_by_agent = {a: load_metrics_for_agent(experiments_dir, a) for a in agents}
    active = [a for a in agents if metrics_by_agent[a]]

    plt.style.use('default')
    plt.rcParams.update({'figure.dpi': 300, 'axes.grid': True, 'grid.alpha': 0.3})
    
    with PdfPages(report_path) as pdf:
        # 1. Learning Curve
        fig, ax = plt.subplots(figsize=(10, 6))
        for agent in active:
            ep, rew = _agent_series(metrics_by_agent[agent], "rewards")
            if rew is not None:
                ax.plot(ep, smooth_curve(rew, 0.9), color=COLORS[agent], linewidth=2, label=SHORT_NAMES[agent])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Evaluation Return')
        ax.set_title('Learning Curve — Mean Evaluation Return vs Episode')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # 2. SU Throughput
        fig, ax = plt.subplots(figsize=(10, 6))
        for agent in active:
            ep, th = _agent_series(metrics_by_agent[agent], "throughput_s")
            if th is not None:
                se = th / BANDWIDTH_HZ if agent not in ["MATD3", "CENT_NOMA_TD3"] else th
                ax.plot(ep, smooth_curve(se, 0.9), color=COLORS[agent], linewidth=2, label=f"{SHORT_NAMES[agent]} (Sum Rate)")
        ax.set_xlabel('Episode')
        ax.set_ylabel('Secondary Rate $R_s$ (bits/s/Hz)')
        ax.set_title('Secondary User Sum-Rate Throughput vs Episode')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # 3. PU Throughput
        fig, ax = plt.subplots(figsize=(10, 6))
        for agent in active:
            ep, th_p = _agent_series(metrics_by_agent[agent], "pu_throughput")
            if th_p is not None:
                se_p = th_p / BANDWIDTH_HZ if agent not in ["MATD3", "CENT_NOMA_TD3"] else th_p
                ax.plot(ep, smooth_curve(se_p, 0.9), color=COLORS[agent], linewidth=2, label=SHORT_NAMES[agent])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Primary Rate $R_p$ (bits/s/Hz)')
        ax.set_title('Primary User Throughput vs Episode')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # 4. Outage Probability (log scale, like BER)
        OUTAGE_FLOOR = 1e-4  # measured outage is often exactly 0; floor it for the log axis
        fig, ax = plt.subplots(figsize=(10, 6))
        for agent in active:
            ep, pu = _agent_series(metrics_by_agent[agent], "outage")
            if pu is not None:
                ax.plot(ep, np.maximum(pu, OUTAGE_FLOOR), color=COLORS[agent], linewidth=2,
                        label=f'{SHORT_NAMES[agent]} — PU (PN) outage')
            ep_s, su = _agent_series(metrics_by_agent[agent], "su_outage")
            if su is not None:
                ax.plot(ep_s, np.maximum(su, OUTAGE_FLOOR), color=COLORS[agent], linewidth=1.5, linestyle='--',
                        label=f'{SHORT_NAMES[agent]} — SU (SN) outage')
        ax.set_yscale('log')
        ax.set_ylim(OUTAGE_FLOOR * 0.5, 1.0)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Outage Probability')
        ax.set_title(f'Primary (PN) and Secondary (SN) Outage Probability\n(log scale; values at the {OUTAGE_FLOOR:g} floor indicate zero measured outage)')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

        # 5. Average Power — Secondary (SU sources + relay) and Primary (PT)
        fig, ax = plt.subplots(figsize=(10, 6))
        for agent in active:
            ep, ap = _agent_series(metrics_by_agent[agent], "average_power")
            if ap is not None:
                ax.plot(ep, smooth_curve(ap, 0.9), color=COLORS[agent], linewidth=2,
                        label=f'{SHORT_NAMES[agent]} — SN (SU + relay)')
            ep_p, ap_p = _agent_series(metrics_by_agent[agent], "pu_average_power")
            if ap_p is not None:
                ax.plot(ep_p, ap_p, color=COLORS[agent], linewidth=1.5, linestyle='--',
                        label=f'{SHORT_NAMES[agent]} — PN (PT, fixed)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Transmit Power (W)')
        ax.set_title('Average Transmit Power vs Episode\n(SN = learned SU-source + relay power; PN = fixed PT power)')
        ax.legend(fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

        # 6. Relay Success & QoS Satisfaction
        fig, ax = plt.subplots(figsize=(10, 6))
        for agent in active:
            ep, rs = _agent_series(metrics_by_agent[agent], "relay_success")
            if rs is not None:
                ax.plot(ep, smooth_curve(rs, 0.9), color=COLORS[agent], linewidth=2, label=f'{SHORT_NAMES[agent]} — Relay Decode Success')
            ep_q, qs = _agent_series(metrics_by_agent[agent], "qos_satisfaction")
            if qs is not None:
                ax.plot(ep_q, smooth_curve(qs, 0.9), color=COLORS[agent], linewidth=1.5, linestyle=':', label=f'{SHORT_NAMES[agent]} — QoS Satisfaction')
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Rate (0 to 1)')
        ax.set_title('Relay Decode Success & QoS Satisfaction vs Episode')
        ax.legend(fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

        # 7. Individual SU Throughput — one distinct colour per user
        USER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                       "#8c564b", "#e377c2", "#7f7f7f"]
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted_individual = False
        for agent in active:
            runs = metrics_by_agent[agent]
            if not runs: continue
            hist = runs[-1].get("history", {})
            if "per_user_rates" in hist and len(hist["per_user_rates"]) > 0:
                rates_arr = np.array(hist["per_user_rates"])
                episodes = np.array(hist["episodes"])
                if len(rates_arr.shape) == 2:
                    for i in range(rates_arr.shape[1]):
                        plotted_individual = True
                        ax.plot(episodes, smooth_curve(rates_arr[:, i], 0.9),
                                color=USER_COLORS[i % len(USER_COLORS)], linewidth=1.8,
                                label=f'SU {i+1}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Individual Rate (bits/s/Hz)')
        ax.set_title('Individual SU Throughput vs Episode (NOMA Fairness)')
        if plotted_individual:
            ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
            fig.tight_layout()
            pdf.savefig(fig)
        plt.close(fig)

        # 8. BER vs SINR — per-hop DF bit-level decoding (Monte-Carlo vs theory)
        #    SU hop-1 (relay decode) points lie on the single-hop BPSK line;
        #    SU end-to-end points lie ABOVE it (DF error-propagation penalty).
        fig, ax = plt.subplots(figsize=(10, 6))
        all_sinr = []

        def _scatter(x_key, y_key, hist, color, marker, label):
            xs, ys = hist.get(x_key), hist.get(y_key)
            if not xs or not ys:
                return
            x = np.array(xs, dtype=float)
            y = np.array(ys, dtype=float)
            y[y <= 0] = np.nan  # no errors observed -> below MC resolution, omit
            all_sinr.append(x)
            ax.scatter(x, y, color=color, alpha=0.45, s=12, marker=marker, label=label)

        for agent in active:
            runs = metrics_by_agent[agent]
            if not runs: continue
            hist = runs[-1].get("history", {})
            # Perfect-SIC per-hop-SINR Monte-Carlo (light markers)
            _scatter("sinr_hop1_db_pts", "ber_hop1_mc_pts", hist, "#7fbf7f", "o", 'SN (SU) hop-1 — perfect-SIC MC')
            _scatter("sinr_db_pts", "ber_mc_pts", hist, "#8fbfe0", "s", 'SN (SU) e2e — perfect-SIC MC')
            # Waveform imperfect-SIC (bold markers) — these lie above, showing the
            # NOMA/SIC error floor from cancelling *detected* symbols.
            _scatter("sinr_hop1_db_pts", "ber_wf_hop1_mc_pts", hist, "#2ca02c", "^", 'SN (SU) hop-1 — waveform (imperfect SIC)')
            _scatter("sinr_db_pts", "ber_wf_e2e_mc_pts", hist, "#1f77b4", "D", 'SN (SU) e2e — waveform (imperfect SIC)')
            _scatter("pu_sinr_db_pts", "pu_ber_wf_mc_pts", hist, "#d62728", "x", 'PN (PU) e2e — waveform')

        # Single-hop BPSK theory reference line across the observed SINR range
        if all_sinr:
            concat = np.concatenate(all_sinr)
            lo, hi = float(np.percentile(concat, 1)), float(np.percentile(concat, 99))
            if hi <= lo:
                lo, hi = float(concat.min()) - 1.0, float(concat.max()) + 1.0
            sweep_db = np.linspace(lo, hi, 250)
            gamma = 10.0 ** (sweep_db / 10.0)
            ax.plot(sweep_db, np.maximum(ber_bpsk_theory(gamma), 1e-8),
                    color="black", linewidth=1.5, label='single-hop BPSK theory')
            ax.set_xlim(lo, hi)
        ax.set_yscale('log')
        ax.set_ylim(1e-6, 1.0)
        ax.set_xlabel('SINR (dB)  [hop-1: γ_sr,  end-to-end: min(γ_sr, γ_rd)]')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.set_title('Per-Hop DF-NOMA BER vs SINR\n'
                     'perfect-SIC MC (light) vs waveform imperfect-SIC (bold) vs single-hop BPSK theory (line)')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())
        pdf.savefig(fig)
        plt.close(fig)

        # 9. BER vs Episode — per-hop (relay) and end-to-end, for SN and PN
        BER_FLOOR = 1e-6
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False
        for agent in active:
            ep, bs = _agent_series(metrics_by_agent[agent], "ber")
            if bs is not None:
                plotted = True
                ax.plot(ep, np.maximum(bs, BER_FLOOR), color="#1f77b4", linewidth=2,
                        label='SN (SU) — end-to-end')
            ep_h, bh = _agent_series(metrics_by_agent[agent], "ber_hop1")
            if bh is not None:
                plotted = True
                ax.plot(ep_h, np.maximum(bh, BER_FLOOR), color="#2ca02c", linewidth=1.5, linestyle=':',
                        label='SN (SU) — hop-1 (relay)')
            ep_p, bp = _agent_series(metrics_by_agent[agent], "pu_ber")
            if bp is not None:
                plotted = True
                ax.plot(ep_p, np.maximum(bp, BER_FLOOR), color="#d62728", linewidth=1.5, linestyle='--',
                        label='PN (PU) — end-to-end')
        ax.set_yscale('log')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.set_title('Mean BER vs Episode — per-hop DF decoding\n(hop-1 = relay decode, end-to-end = destination after DF forwarding)')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        if plotted:
            ax.legend(fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

    return report_path

