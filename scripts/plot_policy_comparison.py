from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "sine": "#F58518",
    "fallback": "#F58518",
    "bc": "#4C78A8",
    "joint_scripted": "#54A24B",
}


def load_results(csv_paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in csv_paths]
    return pd.concat(frames, ignore_index=True)


def plot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("total_reward", "Total Reward"),
        ("probe_request_count", "Probe Requests"),
        ("final_boundary_confidence", "Boundary Confidence"),
        ("mean_uncertainty", "Mean Uncertainty"),
    ]
    missing = [column for column, _ in metrics if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    grouped = df.groupby("policy", as_index=False)[[column for column, _ in metrics]].mean()
    grouped["policy"] = pd.Categorical(grouped["policy"], ["sine", "fallback", "joint_scripted", "bc"], ordered=True)
    grouped = grouped.sort_values("policy")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.reshape(-1)
    colors = [COLORS.get(str(policy), "#777777") for policy in grouped["policy"]]
    for ax, (column, title) in zip(axes, metrics):
        ax.bar(grouped["policy"].astype(str), grouped[column], color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("")

    fig.suptitle("BC vs Sine vs Joint-Scripted Policy", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BC, sine, and joint-scripted policy comparison.")
    parser.add_argument("--fallback-csv", default="results/policy_comparison/fallback/fallback_eval_results.csv")
    parser.add_argument("--sine-csv", default="")
    parser.add_argument("--joint-csv", default="")
    parser.add_argument("--bc-csv", default="results/policy_comparison/bc/bc_eval_results.csv")
    parser.add_argument("--output", default="results/policy_comparison/policy_comparison_chart.png")
    parser.add_argument("--merged-output", default="results/policy_comparison/policy_comparison_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = [Path(args.bc_csv)]
    if args.sine_csv:
        csv_paths.append(Path(args.sine_csv))
    elif args.fallback_csv:
        csv_paths.append(Path(args.fallback_csv))
    if args.joint_csv:
        csv_paths.append(Path(args.joint_csv))
    df = load_results(csv_paths)
    merged_output = Path(args.merged_output)
    merged_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(merged_output, index=False)
    plot_comparison(df, Path(args.output))
    print(f"Saved merged comparison: {merged_output}")
    print(f"Saved policy comparison chart: {args.output}")


if __name__ == "__main__":
    main()
