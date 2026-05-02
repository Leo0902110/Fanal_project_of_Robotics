from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONDITION_ORDER = ["Clean", "Pseudo-Blur", "Active-Probe"]
COLORS = {
    "Clean": "#4C78A8",
    "Pseudo-Blur": "#F58518",
    "Active-Probe": "#54A24B",
}


def _load_results(results_dir: Path) -> pd.DataFrame:
    merged_path = results_dir / "mvp_results.csv"
    frames = []
    for csv_path in sorted(results_dir.glob("*/mvp_results.csv")):
        frames.append(pd.read_csv(csv_path))
    if not frames:
        if merged_path.exists():
            return pd.read_csv(merged_path)
        raise FileNotFoundError(f"No mvp_results.csv files found under {results_dir}")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(merged_path, index=False)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "condition" not in df.columns:
        if "experiment" in df.columns:
            mapping = {
                "baseline_clean": "Clean",
                "clean": "Clean",
                "baseline_pseudo_blur": "Pseudo-Blur",
                "pseudo_blur": "Pseudo-Blur",
                "active_tactile_mvp": "Active-Probe",
                "active_probe": "Active-Probe",
            }
            df["condition"] = df["experiment"].map(mapping).fillna(df["experiment"])
        else:
            raise ValueError("Results must contain either 'condition' or 'experiment'.")
    if "success_rate" not in df.columns and "success" in df.columns:
        df["success_rate"] = df["success"]
    return df


def plot_results(df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("success_rate", "Success Rate"),
        ("total_reward", "Total Reward"),
        ("trigger_count", "Active Probe Count"),
        ("mean_uncertainty", "Mean Uncertainty"),
    ]
    if "final_boundary_confidence" in df.columns:
        metrics[-1] = ("final_boundary_confidence", "Boundary Confidence")
    missing = [column for column, _ in metrics if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    grouped = df.groupby("condition", as_index=False)[[column for column, _ in metrics]].mean()
    grouped["condition"] = pd.Categorical(grouped["condition"], CONDITION_ORDER, ordered=True)
    grouped = grouped.sort_values("condition")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.reshape(-1)
    bar_colors = [COLORS.get(str(condition), "#777777") for condition in grouped["condition"]]

    for ax, (column, title) in zip(axes, metrics):
        ax.bar(grouped["condition"].astype(str), grouped[column], color=bar_colors)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("MVP Performance under Visual Pseudo-Blur", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_decision_flow(df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("visual_trigger_count", "Visual Trigger"),
        ("decision_ambiguity_count", "Decision Ambiguity"),
        ("probe_request_count", "Probe Request"),
        ("refinement_count", "Boundary Refinement"),
        ("contact_resolved_count", "Contact Resolved"),
    ]
    available = [(column, title) for column, title in metrics if column in df.columns]
    if not available:
        return

    grouped = df.groupby("condition", as_index=False)[[column for column, _ in available]].mean()
    grouped["condition"] = pd.Categorical(grouped["condition"], CONDITION_ORDER, ordered=True)
    grouped = grouped.sort_values("condition")

    x = range(len(grouped))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, (column, title) in enumerate(available):
        offsets = [value + (index - (len(available) - 1) / 2) * width for value in x]
        ax.bar(offsets, grouped[column], width=width, label=title)

    ax.set_xticks(list(x))
    ax.set_xticklabels(grouped["condition"].astype(str), rotation=12)
    ax.set_title("Decision Ambiguity to Tactile Probe Flow")
    ax.set_ylabel("Step Count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot local MVP experiment results.")
    parser.add_argument("--results-dir", default="results/local_mvp_rgbd")
    parser.add_argument("--output", default="results/mvp_performance_chart.png")
    parser.add_argument("--decision-output", default="results/mvp_decision_flow_chart.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    df = _normalize_columns(_load_results(results_dir))
    plot_results(df, output_path)
    plot_decision_flow(df, Path(args.decision_output))
    print(f"Saved chart: {output_path}")
    print(f"Saved decision flow chart: {args.decision_output}")


if __name__ == "__main__":
    main()
