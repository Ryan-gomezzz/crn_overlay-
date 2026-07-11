"""
Report and Plot Generation Engine for the CRN Research Framework.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Premium color palette for all agents
COLORS = {
    "TD3": "#1f77b4",          # Deep Blue
    "UNDERLAY_TD3": "#d62728", # Crimson Red
    "OVERLAY_TD3": "#2ca02c",  # Forest Green
    "MATD3": "#9467bd",        # Purple
    "CENT_NOMA_TD3": "#ff7f0e" # Orange
}

SHORT_NAMES = {
    "TD3": "TD3",
    "UNDERLAY_TD3": "Underlay TD3",
    "OVERLAY_TD3": "Overlay TD3",
    "MATD3": "MATD3",
    "CENT_NOMA_TD3": "Cent NOMA TD3"
}

def load_metrics_for_agent(experiments_dir: str, agent: str) -> List[Dict[str, Any]]:
    """Scan and load all metrics.json files for a given agent."""
    agent_dir = os.path.join(experiments_dir, agent.lower())
    metrics_list = []
    if not os.path.exists(agent_dir):
        folder_map = {
            "TD3": "td3",
            "UNDERLAY_TD3": "underlay_td3",
            "OVERLAY_TD3": "overlay_td3",
            "CAMO_TD3": "underlay_td3",
            "OVERLAY_CAMO_TD3": "overlay_td3",
            "T3": "td3"
        }
        agent_dir = os.path.join(experiments_dir, folder_map.get(agent, agent.lower()))

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
        agents = ["TD3", "UNDERLAY_TD3", "OVERLAY_TD3"]
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
        se = th / BANDWIDTH_HZ
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
        agents = ["TD3", "UNDERLAY_TD3", "OVERLAY_TD3"]
    lines = ["# CRN Overlay Framework — Experimental Report", "",
             "Metrics below are read directly from each run's `metrics.json`.", ""]
    lines.append("| Algorithm | Episodes | Seeds | Mean Return | SU Rate (bits/s/Hz) | PU Outage | SU Outage | Train Time (s) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for agent in agents:
        s = _run_summary(load_metrics_for_agent(experiments_dir, agent))
        if not s:
            lines.append(f"| {SHORT_NAMES[agent]} | *no runs* | - | - | - | - | - | - |")
            continue
        se = (s["eval_su_throughput"] / BANDWIDTH_HZ) if s["eval_su_throughput"] is not None else None
        lines.append(
            f"| {SHORT_NAMES[agent]} | {s['episodes']} | {s['seeds']} | "
            f"{s['eval_reward']:.3e} | {se:.4f} | {s['eval_pu_outage']:.4f} | "
            f"{(s['eval_su_outage'] if s['eval_su_outage'] is not None else 0.0):.4f} | "
            f"{s['train_time']:.1f} |"
        )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path

def generate_pdf_report(md_path: str, pdf_path: str, agents=None) -> str:
    """Compile the professional PDF report matching the legacy Matplotlib layout."""
    import os
    try:
        from cli.legacy_plotter import generate_legacy_pdf, RunMetrics
    except ImportError as e:
        print(f"Warning: Legacy plotter not found: {e}")
        return ""
        
    try:
        experiments_dir = os.path.abspath(os.path.join(os.path.dirname(md_path), ".."))
        if agents is None:
            agents = ["TD3", "UNDERLAY_TD3", "OVERLAY_TD3"]
        
        all_metrics = []
        n_eps = 0
        steps_per_ep = 200 # Default fallback
        nakagami_m = 1.0
        pu_sinr_threshold_linear = 1.2589
        outage_desc = "Outage Penalty"
        
        for agent in agents:
            runs = load_metrics_for_agent(experiments_dir, agent)
            if not runs:
                continue
            
            run = runs[-1] # Use the latest run for plots
            
            # Extract config dynamically from the first valid run found
            config_path = os.path.join(run.get("path", ""), "config_snapshot.yaml")
            if os.path.exists(config_path):
                import yaml
                try:
                    with open(config_path, "r") as f:
                        cfg = yaml.safe_load(f)
                    fading = cfg.get("channel", {}).get("fading_type", "rayleigh")
                    nakagami_m = 1.0 if fading == "rayleigh" else 3.0
                    
                    pu_rate = cfg.get("camo_td3", {}).get("pu_rate_threshold", 0.5)
                    pu_sinr_threshold_linear = (2.0 ** pu_rate) - 1.0
                    
                    int_thresh = cfg.get("camo_td3", {}).get("interference_limit_dbm", -80.0)
                    outage_desc = f"Interference at PR > {int_thresh} dBm"
                    steps_per_ep = cfg.get("simulation", {}).get("time_steps_per_episode", 200)
                except Exception as ex:
                    print("Could not parse config:", ex)
                    
            hist = run.get("history", {})
            m = RunMetrics(name=agent)
            
            # Map metrics.json to legacy format
            m.episodes = hist.get("episodes", [])
            m.rewards = hist.get("rewards", [])
            m.su_throughputs = [x / BANDWIDTH_HZ for x in hist.get("throughput_s", [])]
            m.pu_throughputs = hist.get("pu_throughput", [])
            m.outage_probs = hist.get("outage", [])
            m.avg_bers = hist.get("ber", [])
            
            m.sinr_db_pts = hist.get("sinr_db_pts", [])
            m.ber_pts = hist.get("ber_pts", [])
            m.pu_sinr_db_pts = hist.get("pu_sinr_db_pts", [])
            m.pu_ber_pts = hist.get("pu_ber_pts", [])
            
            # Fill summary table stats
            m.final_avg_reward = run.get("eval_reward", 0.0)
            m.final_avg_su_tput = run.get("eval_su_throughput", 0.0) / BANDWIDTH_HZ if run.get("eval_su_throughput") else 0.0
            
            if len(m.pu_throughputs) > 0:
                m.final_avg_pu_tput = sum(m.pu_throughputs[-100:]) / min(100, len(m.pu_throughputs))
            else:
                m.final_avg_pu_tput = 0.0
                
            m.final_outage_prob = run.get("eval_pu_outage", 0.0)
            m.final_avg_ber = hist.get("ber", [0.0])[-1] if hist.get("ber") else 0.0
            m.training_time_sec = run.get("train_time", 0.0)
            
            eps_list = hist.get("episodes", [])
            if eps_list and len(eps_list) > n_eps:
                n_eps = len(eps_list)
                
            all_metrics.append(m)
            
        generate_legacy_pdf(
            all_metrics, 
            pdf_path, 
            n_episodes=n_eps, 
            steps_per_ep=steps_per_ep,
            nakagami_m=nakagami_m,
            pu_sinr_threshold_linear=pu_sinr_threshold_linear,
            outage_desc=outage_desc
        )
        
        # Rename the file dynamically based on specifications
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Provide fallback if int_thresh isn't parsed
        try:
            ith_val = int_thresh
        except NameError:
            ith_val = "unknown"
            
        is_noma = "MATD3" in agents
        prefix = "noma_report" if is_noma else "report"
            
        new_name = f"{prefix}_ep{n_eps}_nak{nakagami_m}_Ith{ith_val}_{timestamp}.pdf"
        new_pdf_path = os.path.join(os.path.dirname(pdf_path), new_name)
        if os.path.exists(pdf_path):
            os.rename(pdf_path, new_pdf_path)
            return new_pdf_path
        return pdf_path
    except Exception as e:
        print(f"Warning: PDF generation failed: {e}")
        return ""
