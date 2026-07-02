"""
Report and Plot Generation Engine for the CRN Research Framework.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Premium color palette
COLORS = {
    "TD3": "#e74c3c",          # Crimson Red
    "CAMO_TD3": "#3498db",     # Vibrant Blue
    "OVERLAY_CAMO_TD3": "#2ecc71" # Emerald Green
}

SHORT_NAMES = {
    "TD3": "T3",
    "CAMO_TD3": "Underlay TD3",
    "OVERLAY_CAMO_TD3": "Overlay TD3"
}

def load_metrics_for_agent(experiments_dir: str, agent: str) -> List[Dict[str, Any]]:
    """Scan and load all metrics.json files for a given agent."""
    agent_dir = os.path.join(experiments_dir, agent.lower())
    metrics_list = []
    if not os.path.exists(agent_dir):
        # Check standard folder mapping just in case
        folder_map = {
            "TD3": "t3",
            "CAMO_TD3": "underlay_td3",
            "OVERLAY_CAMO_TD3": "overlay_td3"
        }
        agent_dir = os.path.join(experiments_dir, folder_map.get(agent, agent.lower()))

    if os.path.exists(agent_dir):
        for run_name in os.listdir(agent_dir):
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

def generate_comparison_plots(experiments_dir: str, output_plots_dir: str):
    """Generate professional Matplotlib comparison plots from stored metrics."""
    os.makedirs(output_plots_dir, exist_ok=True)
    
    # Load all metrics
    agents = ["TD3", "CAMO_TD3", "OVERLAY_CAMO_TD3"]
    metrics_by_agent = {}
    for agent in agents:
        metrics_by_agent[agent] = load_metrics_for_agent(experiments_dir, agent)

    # 1. Convergence Curve (Episode rewards)
    plt.figure(figsize=(10, 6))
    for agent in agents:
        runs = metrics_by_agent[agent]
        if not runs:
            continue
        
        # We can align by episode steps or use the run that has history
        # Let's check if there is an episode history
        all_rewards = []
        for run in runs:
            if "history" in run and "rewards" in run["history"]:
                all_rewards.append(run["history"]["rewards"])
        
        if all_rewards:
            # Pad/truncate to same length
            min_len = min(len(r) for r in all_rewards)
            trimmed_rewards = [r[:min_len] for r in all_rewards]
            mean_rewards = np.mean(trimmed_rewards, axis=0)
            std_rewards = np.std(trimmed_rewards, axis=0)
            
            episodes = np.arange(1, min_len + 1)
            plt.plot(episodes, mean_rewards, label=SHORT_NAMES[agent], color=COLORS[agent], linewidth=2)
            plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, color=COLORS[agent], alpha=0.15)
            
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Episode Return", fontsize=12)
    plt.title("Policy Convergence & Training Returns", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_dir, "convergence_comparison.png"), dpi=150)
    plt.close()

    # 2. SU Throughput vs Outage Tradeoff
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for agent in agents:
        runs = metrics_by_agent[agent]
        if not runs:
            continue
        
        all_th = []
        all_out = []
        for r in runs:
            if "history" in r:
                if "throughput_s" in r["history"]:
                    all_th.append(r["history"]["throughput_s"])
                if "outage" in r["history"]:
                    all_out.append(r["history"]["outage"])
        
        if all_th and all_out:
            min_len = min(len(all_th), len(all_out))
            min_steps = min(len(x) for x in all_th[:min_len])
            th_arr = np.mean([x[:min_steps] for x in all_th], axis=0)
            out_arr = np.mean([x[:min_steps] for x in all_out], axis=0)
            steps = np.arange(1, min_steps + 1)
            
            ax1.plot(steps, th_arr, label=SHORT_NAMES[agent], color=COLORS[agent], marker='o', markevery=max(1, min_steps//10))
            ax2.plot(steps, out_arr, label=SHORT_NAMES[agent], color=COLORS[agent], marker='s', markevery=max(1, min_steps//10))

    ax1.set_xlabel("Evaluation Milestones", fontsize=11)
    ax1.set_ylabel("SU Throughput (bps/Hz)", fontsize=11)
    ax1.set_title("Secondary User (SU) Average Throughput", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.set_xlabel("Evaluation Milestones", fontsize=11)
    ax2.set_ylabel("PU Outage Rate", fontsize=11)
    ax2.set_title("Primary User (PU) QoS Outage Rate", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_dir, "metrics_comparison.png"), dpi=150)
    plt.close()

    # 3. Bar Chart of Computation Speeds (Train & Inference)
    plt.figure(figsize=(10, 5))
    available_agents = [a for a in agents if metrics_by_agent[a]]
    if available_agents:
        train_times = []
        inf_times = []
        for agent in available_agents:
            t_times = [r.get("train_time", 0.0) for r in metrics_by_agent[agent] if r.get("train_time")]
            i_times = [r.get("inf_time", 0.0) * 1000.0 for r in metrics_by_agent[agent] if r.get("inf_time")] # ms
            
            train_times.append(np.mean(t_times) if t_times else 0.0)
            inf_times.append(np.mean(i_times) if i_times else 0.0)
            
        x = np.arange(len(available_agents))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        rects1 = ax1.bar(x - width/2, train_times, width, label="Training Time (s)", color="#9b59b6")
        rects2 = ax2.bar(x + width/2, inf_times, width, label="Inference Time (ms)", color="#f1c40f")
        
        ax1.set_xlabel("Algorithms", fontsize=12)
        ax1.set_ylabel("Avg Training Time (seconds)", color="#9b59b6", fontsize=12)
        ax2.set_ylabel("Avg Inference Time (milliseconds)", color="#f1c40f", fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels([SHORT_NAMES[a] for a in available_agents])
        plt.title("Computational Efficiency Benchmark", fontsize=14, fontweight="bold")
        fig.tight_layout()
        plt.savefig(os.path.join(output_plots_dir, "efficiency_comparison.png"), dpi=150)
        plt.close()

def generate_markdown_report(experiments_dir: str, output_dir: str) -> str:
    """Create a premium markdown report summarizing all experiments."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "research_report.md")
    
    agents = ["TD3", "CAMO_TD3", "OVERLAY_CAMO_TD3"]
    metrics_by_agent = {}
    for agent in agents:
        metrics_by_agent[agent] = load_metrics_for_agent(experiments_dir, agent)
        
    with open(report_path, "w") as f:
        f.write("# CRN Reinforcement Learning Framework: Experimental Report\n\n")
        f.write("This report compiles performance and convergence metrics for standard **T3** (TD3), **Underlay TD3** (CAMO-TD3), and **Overlay TD3** (Overlay-CAMO-TD3) under Rayleigh fading constraints.\n\n")
        
        f.write("## 1. Benchmarking Summary Table\n\n")
        f.write("| Algorithm | Mean Return | Best Seed Return | SU Throughput (bps/Hz) | PU Outage Rate | Training Time (s) | Avg Inf. Time (ms) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for agent in agents:
            runs = metrics_by_agent[agent]
            if not runs:
                f.write(f"| {SHORT_NAMES[agent]} | *No Runs Found* | - | - | - | - | - |\n")
                continue
                
            returns = [r.get("eval_reward", 0.0) for r in runs if "eval_reward" in r]
            best_ret = max(returns) if returns else 0.0
            mean_ret = np.mean(returns) if returns else 0.0
            
            su_th = np.mean([r.get("eval_su_throughput", 0.0) for r in runs])
            pu_out = np.mean([r.get("eval_pu_outage", 0.0) for r in runs])
            
            t_times = [r.get("train_time", 0.0) for r in runs if r.get("train_time")]
            i_times = [r.get("inf_time", 0.0) * 1000.0 for r in runs if r.get("inf_time")]
            
            mean_t = np.mean(t_times) if t_times else 0.0
            mean_i = np.mean(i_times) if i_times else 0.0
            
            f.write(
                f"| **{SHORT_NAMES[agent]}** | "
                f"{mean_ret:.2f} | "
                f"{best_ret:.2f} | "
                f"{su_th:.4f} | "
                f"{pu_out:.4f} | "
                f"{mean_t:.2f}s | "
                f"{mean_i:.3f}ms |\n"
            )
            
        f.write("\n## 2. Convergence Analysis\n\n")
        f.write("The convergence plots reflect the policy return trends across different seeds. Standard memoryless T3 agents typically exhibit higher variance and outages due to the lack of history modeling. Recurrent Underlay and Overlay structures leverage temporal state representation to adapt to fading fluctuations.\n\n")
        
        # Reference generated charts
        f.write("### Policy Convergence\n")
        f.write("![Convergence Comparison](../plots/convergence_comparison.png)\n\n")
        
        f.write("### Metric Performance\n")
        f.write("![Metrics Comparison](../plots/metrics_comparison.png)\n\n")
        
        f.write("### Computational Efficiency\n")
        f.write("![Efficiency Comparison](../plots/efficiency_comparison.png)\n\n")

    return report_path

def generate_pdf_report(md_path: str, pdf_path: str) -> bool:
    """Attempts to compile the Markdown report to PDF using reportlab if available."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
    except ImportError:
        return False
        
    try:
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        story = []
        
        # Quick title formatting
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=20
        )
        
        h2_style = ParagraphStyle(
            'H2Style',
            parent=styles['Heading2'],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor('#34495e'),
            spaceBefore=15,
            spaceAfter=10
        )
        
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['Normal'],
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=8
        )
        
        story.append(Paragraph("CRN Reinforcement Learning Framework: Experimental Report", title_style))
        story.append(Spacer(1, 10))
        
        intro_text = "This report compiles performance and convergence metrics for standard T3, Underlay TD3, and Overlay TD3 under Rayleigh fading constraints. It is compiled automatically by the research framework CLI."
        story.append(Paragraph(intro_text, body_style))
        story.append(Spacer(1, 15))
        
        story.append(Paragraph("1. Benchmarking Performance Summary", h2_style))
        story.append(Spacer(1, 10))
        
        # Simple summary table template
        table_data = [
            ["Algorithm", "Throughput (bps/Hz)", "PU Outage", "Avg Return"],
            ["T3 Baseline", "TBD", "TBD", "TBD"],
            ["Underlay TD3", "TBD", "TBD", "TBD"],
            ["Overlay TD3", "TBD", "TBD", "TBD"]
        ]
        
        t = Table(table_data, colWidths=[150, 120, 100, 100])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#ecf0f1')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9f9f9')])
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("2. Plots and Figures Reference", h2_style))
        story.append(Paragraph("The visual charts have been exported successfully to the `plots/` subdirectory. Please refer to: `convergence_comparison.png`, `metrics_comparison.png`, and `efficiency_comparison.png` for a graphical summary of learning rates, Lagrangians, and step counts.", body_style))
        
        doc.build(story)
        return True
    except Exception as e:
        print(f"Warning: PDF generation failed: {e}")
        return False
