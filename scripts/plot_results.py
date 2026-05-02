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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot local MVP experiment results.")
    parser.add_argument("--results-dir", default="results/local_mvp_rgbd")
    parser.add_argument("--output", default="results/mvp_performance_chart.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    df = _normalize_columns(_load_results(results_dir))
    plot_results(df, output_path)
    print(f"Saved chart: {output_path}")


if __name__ == "__main__":
    main()
