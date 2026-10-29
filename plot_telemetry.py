import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- 1. CONFIGURATION & COLOR PALETTE ---
TIER_COLORS = {
    "P1_SURVIVAL": "#E63946",     # Red: LiDAR obstacle/box evasion
    "P2_MISSION": "#2A9D8F",      # Teal/Green: Visual + Map Pin servoing
    "P3_EXPLORATION": "#E9C46A",  # Gold: VLM room sweep & search
}

TIER_LEVELS = {
    "P3_EXPLORATION": 1,
    "P2_MISSION": 2,
    "P1_SURVIVAL": 3,
}

FRIENDLY_NAMES = {
    "test1_open_los.csv": "Test 1: Open LoS",
    "test2_occlusion.csv": "Test 2: Narrow-Gap Flank",
    "test3_blind_search.csv": "Test 3: Blind Search",
    "test4a_redock.csv": "Test 4A: Live Re-Dock",
    "test4b_midroute.csv": "Test 4B: Phantom-Pin Clear",
}


def load_telemetry_csv(filepath: str):
    """Loads a telemetry CSV using standard library + NumPy (no pandas required)."""
    steps, tiers, rx, ry, front_m, bore_m, vlm_states, yolo_ms = [], [], [], [], [], [], [], []
    try:
        with open(filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "step" not in reader.fieldnames:
                return None
            for row in reader:
                steps.append(int(float(row["step"])))
                tiers.append(row["tier"].strip())
                rx.append(float(row["robot_x"]))
                ry.append(float(row["robot_y"]))
                front_m.append(float(row["front_lidar_m"]))
                bore_m.append(float(row["boresight_m"]))
                vlm_states.append(row["vlm_state"].strip())
                yolo_ms.append(float(row["yolo_ms"]))
    except Exception as e:
        print(f"[SKIP] Could not parse {filepath}: {e}")
        return None

    if len(steps) == 0:
        return None

    return {
        "filename": os.path.basename(filepath),
        "label": FRIENDLY_NAMES.get(os.path.basename(filepath), os.path.splitext(os.path.basename(filepath))[0]),
        "step": np.array(steps),
        "tier": np.array(tiers),
        "x": np.array(rx),
        "y": np.array(ry),
        "front_m": np.array(front_m),
        "bore_m": np.array(bore_m),
        "vlm": np.array(vlm_states),
        "yolo_ms": np.array(yolo_ms),
    }


def generate_individual_deep_dive(data: dict):
    """Generates a 3-panel deep-dive figure for a single telemetry run."""
    fig = plt.figure(figsize=(14, 6.2), dpi=300)
    fig.suptitle(f"Autonomous Navigation Telemetry Deep-Dive — {data['label']}", fontsize=14, fontweight="bold")

    # Left Subplot: 2D Trajectory Color-Coded by Active Subsumption Tier
    ax_map = fig.add_subplot(1, 2, 1)
    pt_colors = [TIER_COLORS.get(t, "#888888") for t in data["tier"]]
    ax_map.plot(data["x"], data["y"], color="#CCCCCC", linewidth=1.2, zorder=1)
    ax_map.scatter(data["x"], data["y"], c=pt_colors, s=14, zorder=2)

    # Mark Start and Final Dock Positions
    ax_map.scatter(data["x"][0], data["y"][0], c="#1D3557", s=110, marker="^", zorder=4, label="Start (0, 0)")
    ax_map.scatter(data["x"][-1], data["y"][-1], c="#FF0055", s=140, marker="*", zorder=4,
                   label=f"Final Dock ({data['x'][-1]:.2f}m, {data['y'][-1]:.2f}m)")

    ax_map.set_title("2D World-Frame Trajectory (Colored by Active Tier)", fontsize=11, fontweight="bold")
    ax_map.set_xlabel("World X (meters)")
    ax_map.set_ylabel("World Y (meters)")
    ax_map.grid(True, linestyle="--", alpha=0.4)
    ax_map.set_aspect("equal", adjustable="datalim")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="P1: LiDAR Survival (Bypass)", markerfacecolor=TIER_COLORS["P1_SURVIVAL"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="P2: YOLO+HSV+Map Mission", markerfacecolor=TIER_COLORS["P2_MISSION"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="P3: VLM Exploration", markerfacecolor=TIER_COLORS["P3_EXPLORATION"], markersize=8),
        Line2D([0], [0], marker="^", color="w", label="Robot Start", markerfacecolor="#1D3557", markersize=9),
        Line2D([0], [0], marker="*", color="w", label=f"Goal Dock ({data['bore_m'][-1]:.3f}m)", markerfacecolor="#FF0055", markersize=11),
    ]
    ax_map.legend(handles=legend_elements, loc="best", fontsize=8.5)

    # Top-Right Subplot: LiDAR Clearances vs. Safety & Goal Thresholds
    ax_lidar = fig.add_subplot(2, 2, 2)
    ax_lidar.plot(data["step"], data["bore_m"], color="#2A9D8F", linewidth=1.3, label="Boresight LiDAR (m)")
    ax_lidar.plot(data["step"], data["front_m"], color="#E63946", linewidth=1.0, alpha=0.75, label="Closest Front/Shoulder (m)")
    ax_lidar.axhline(0.20, color="#1D3557", linestyle="--", linewidth=1.2, label="Goal Stop Dist (0.20m)")
    ax_lidar.axhline(0.17, color="#F4A261", linestyle=":", linewidth=1.2, label="Shoulder Bypass Trigger (0.17m)")
    ax_lidar.axhline(0.12, color="#D62828", linestyle=":", linewidth=1.1, label="Emergency Trigger (0.12m)")

    ax_lidar.set_title("LiDAR Range Profile & Safety Thresholds", fontsize=11, fontweight="bold")
    ax_lidar.set_ylabel("Distance (meters)")
    ax_lidar.set_ylim(0, min(2.2, max(data["bore_m"]) + 0.15))
    ax_lidar.grid(True, linestyle="--", alpha=0.4)
    ax_lidar.legend(loc="upper right", fontsize=7.5, ncol=2)

    # Bottom-Right Subplot: Subsumption State Machine Timeline
    ax_tier = fig.add_subplot(2, 2, 4, sharex=ax_lidar)
    y_levels = [TIER_LEVELS.get(t, 1) for t in data["tier"]]
    ax_tier.step(data["step"], y_levels, where="post", color="#264653", linewidth=1.2)
    ax_tier.scatter(data["step"], y_levels, c=pt_colors, s=10, zorder=3)
    ax_tier.set_yticks([1, 2, 3])
    ax_tier.set_yticklabels(["P3: Explore", "P2: Mission", "P1: Survival"])
    ax_tier.set_ylim(0.5, 3.5)
    ax_tier.set_title("Subsumption Arbitration State Timeline", fontsize=11, fontweight="bold")
    ax_tier.set_xlabel("Control Loop Step")
    ax_tier.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_name = f"{os.path.splitext(data['filename'])[0]}_deep_dive.png"
    plt.savefig(out_name, dpi=300)
    plt.close(fig)
    print(f"[SAVED] Individual run analysis -> {out_name}")


def generate_comparison_dashboard(datasets: list):
    """Generates a 4-panel multi-scenario comparison figure across all loaded CSVs."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), dpi=300)
    fig.suptitle("TurtleBot3 Autonomous Navigation — Multi-Scenario Telemetry Benchmark", fontsize=15, fontweight="bold")

    # Panel 1 (Top-Left): Overlay of 2D World Trajectories Across Scenarios
    ax1 = axes[0, 0]
    cmap = plt.get_cmap("tab10")
    for idx, d in enumerate(datasets):
        color = cmap(idx % 10)
        ax1.plot(d["x"], d["y"], linewidth=2.0, label=f"{d['label']} ({len(d['step'])} steps)", color=color)
        ax1.scatter(d["x"][-1], d["y"][-1], color=color, s=70, marker="X", zorder=4)
    ax1.scatter(0, 0, color="black", s=100, marker="^", zorder=5, label="Start Origin (0,0)")
    ax1.set_title("A. 2D Odometry Trajectories Across Test Scenarios", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("World X (meters)")
    ax1.set_ylabel("World Y (meters)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(loc="best", fontsize=8)

    # Panel 2 (Top-Right): Stacked Bar Chart of Subsumption Tier Percentages
    ax2 = axes[0, 1]
    labels = [d["label"] for d in datasets]
    p1_pct, p2_pct, p3_pct = [], [], []
    for d in datasets:
        total = float(len(d["tier"]))
        p1_pct.append(100.0 * np.sum(d["tier"] == "P1_SURVIVAL") / total)
        p2_pct.append(100.0 * np.sum(d["tier"] == "P2_MISSION") / total)
        p3_pct.append(100.0 * np.sum(d["tier"] == "P3_EXPLORATION") / total)

    x_pos = np.arange(len(labels))
    p1_arr = np.array(p1_pct)
    p2_arr = np.array(p2_pct)
    p3_arr = np.array(p3_pct)

    ax2.bar(x_pos, p1_arr, width=0.55, label="P1: LiDAR Survival", color=TIER_COLORS["P1_SURVIVAL"])
    ax2.bar(x_pos, p2_arr, width=0.55, bottom=p1_arr, label="P2: Map+YOLO+HSV", color=TIER_COLORS["P2_MISSION"])
    ax2.bar(x_pos, p3_arr, width=0.55, bottom=p1_arr + p2_arr, label="P3: VLM Exploration", color=TIER_COLORS["P3_EXPLORATION"])

    for i in range(len(labels)):
        if p1_arr[i] > 6:
            ax2.text(i, p1_arr[i] / 2, f"{p1_arr[i]:.1f}%", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        if p2_arr[i] > 6:
            ax2.text(i, p1_arr[i] + p2_arr[i] / 2, f"{p2_arr[i]:.1f}%", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        if p3_arr[i] > 6:
            ax2.text(i, p1_arr[i] + p2_arr[i] + p3_arr[i] / 2, f"{p3_arr[i]:.1f}%", ha="center", va="center", color="black", fontsize=8.5, fontweight="bold")

    ax2.set_title("B. Subsumption Tier Activation Breakdown (%)", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    ax2.set_ylabel("Percentage of Mission Steps (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.5)

    # Panel 3 (Bottom-Left): Boresight Docking Convergence (Normalized Progress)
    ax3 = axes[1, 0]
    for idx, d in enumerate(datasets):
        norm_progress = np.linspace(0, 100, len(d["bore_m"]))
        ax3.plot(norm_progress, d["bore_m"], linewidth=1.4, label=f"{d['label']} (Stop: {d['bore_m'][-1]:.3f}m)", color=cmap(idx % 10))
    ax3.axhline(0.20, color="black", linestyle="--", linewidth=1.3, label="Goal Stop Threshold (0.20m)")
    ax3.set_title("C. Target Boresight Range Convergence (0% -> 100% Mission)", fontsize=11.5, fontweight="bold")
    ax3.set_xlabel("Normalized Mission Progress (%)")
    ax3.set_ylabel("Boresight LiDAR Distance (meters)")
    ax3.set_ylim(0, 2.2)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=8)

    # Panel 4 (Bottom-Right): YOLOv8n Fast-Loop Inference Latency Boxplot
    ax4 = axes[1, 1]
    yolo_data = [d["yolo_ms"][d["yolo_ms"] < 80.0] for d in datasets]
    bp = ax4.boxplot(yolo_data, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [cmap(i % 10) for i in range(len(datasets))]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax4.axhline(32.0, color="#E63946", linestyle="--", linewidth=1.2, label="32ms Webots Timestep Budget")
    ax4.set_title("D. YOLOv8n Fast-Loop Inference Latency Distribution", fontsize=11.5, fontweight="bold")
    ax4.set_xticks(np.arange(1, len(labels) + 1))
    ax4.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    ax4.set_ylabel("Inference Time (ms)")
    ax4.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    out_file = "telemetry_comparison_dashboard.png"
    plt.savefig(out_file, dpi=300)
    print(f"[SAVED] Master 4-panel comparison chart -> {out_file}")
    plt.show()


if __name__ == "__main__":
    csv_files = sorted(glob.glob("*.csv"))
    filtered_files = [
        f for f in csv_files
        if not (f.startswith("telemetry_v2_") and "mission_telemetry.csv" in csv_files)
    ]

    loaded_datasets = []
    for fpath in filtered_files:
        parsed = load_telemetry_csv(fpath)
        if parsed is not None:
            loaded_datasets.append(parsed)
            generate_individual_deep_dive(parsed)

    if not loaded_datasets:
        print("[ERROR] No valid telemetry CSV files found in the current directory!")
    else:
        print(f"\n[INFO] Successfully processed {len(loaded_datasets)} telemetry dataset(s).")
        generate_comparison_dashboard(loaded_datasets)