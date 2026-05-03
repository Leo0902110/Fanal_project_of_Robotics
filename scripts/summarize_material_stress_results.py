from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROFILE_ORDER = ["transparent", "dark", "reflective", "low_texture"]
PROFILE_LABELS = {
    "transparent": "Transparent",
    "dark": "Dark",
    "reflective": "Reflective",
    "low_texture": "Low-texture",
}
COLORS = ["#4C78A8", "#54A24B", "#F58518", "#B279A2"]


def wilson_interval(successes: float, total: float, z: float = 1.96) -> tuple[float, float]:
    total = float(total)
    if total <= 0:
        return 0.0, 0.0
    proportion = float(successes) / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * np.sqrt((proportion * (1.0 - proportion) / total) + (z * z / (4.0 * total * total))) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def load_results(results_dir: Path) -> pd.DataFrame:
    paths = sorted(results_dir.glob("*/tdp_eval_results.csv"))
    if not paths:
        raise FileNotFoundError(f"No tdp_eval_results.csv files found under {results_dir}")
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["result_dir"] = str(path.parent)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    run_sizes = (
        df.groupby(["object_profile", "result_dir"])
        .size()
        .rename("run_episodes")
        .reset_index()
    )
    max_run_sizes = run_sizes.groupby("object_profile")["run_episodes"].transform("max")
    selected_runs = run_sizes.loc[run_sizes["run_episodes"] == max_run_sizes, ["object_profile", "result_dir"]]
    return df.merge(selected_runs, on=["object_profile", "result_dir"], how="inner")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("object_profile")
        .agg(
            success_rate=("success_rate", "mean"),
            grasp_rate=("max_is_grasped", "mean"),
            mean_steps=("steps", "mean"),
            mean_reward=("total_reward", "mean"),
            episodes=("episode", "count"),
            successes=("success", "sum"),
            grasps=("max_is_grasped", "sum"),
            seed_min=("seed", "min"),
            seed_max=("seed", "max"),
        )
        .reindex(PROFILE_ORDER)
    )
    summary = summary.dropna(how="all")
    intervals = [wilson_interval(row.successes, row.episodes) for row in summary.itertuples()]
    grasp_intervals = [wilson_interval(row.grasps, row.episodes) for row in summary.itertuples()]
    summary["success_ci95_low"] = [low for low, _high in intervals]
    summary["success_ci95_high"] = [high for _low, high in intervals]
    summary["grasp_ci95_low"] = [low for low, _high in grasp_intervals]
    summary["grasp_ci95_high"] = [high for _low, high in grasp_intervals]
    return summary


def plot_success_grasp(summary: pd.DataFrame, output_path: Path) -> None:
    profiles = list(summary.index)
    labels = [PROFILE_LABELS.get(profile, profile) for profile in profiles]
    x = np.arange(len(profiles))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    success_values = summary["success_rate"].to_numpy(dtype=float)
    grasp_values = summary["grasp_rate"].to_numpy(dtype=float)
    success_yerr = np.vstack(
        [
            success_values - summary["success_ci95_low"].to_numpy(dtype=float),
            summary["success_ci95_high"].to_numpy(dtype=float) - success_values,
        ]
    )
    bars_success = ax.bar(
        x - width / 2,
        success_values,
        width,
        yerr=success_yerr,
        capsize=5,
        label="Success (95% CI)",
        color="#4C78A8",
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    bars_grasp = ax.bar(x + width / 2, grasp_values, width, label="Grasp", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Rate")
    ax.set_title("Material-Stress Results: Residual Tactile Diffusion Policy (50 eps/profile)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    for bars, counts_col in [(bars_success, "successes"), (bars_grasp, "grasps")]:
        for bar, rate, count, total in zip(bars, bars.datavalues, summary[counts_col], summary["episodes"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(float(bar.get_height()) + 0.08, 1.16),
                f"{int(count)}/{int(total)}\n{rate:.0%}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_performance(summary: pd.DataFrame, output_path: Path) -> None:
    profiles = list(summary.index)
    labels = [PROFILE_LABELS.get(profile, profile) for profile in profiles]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    axes = axes.reshape(-1)
    metrics = [
        ("success_rate", "Success Rate", 0, 1.05, "{:.0%}"),
        ("grasp_rate", "Grasp Rate", 0, 1.05, "{:.0%}"),
        ("mean_steps", "Mean Steps to Finish", 0, max(50.0, float(summary["mean_steps"].max()) * 1.2), "{:.1f}"),
        ("mean_reward", "Mean Total Reward", 0, float(summary["mean_reward"].max()) * 1.2, "{:.1f}"),
    ]
    for ax, (column, title, ymin, ymax, fmt) in zip(axes, metrics):
        values = summary[column].to_numpy(dtype=float)
        colors = COLORS[: len(values)]
        bars = ax.bar(labels, values, color=colors, width=0.65)
        ax.set_title(title)
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(bar.get_height()) + (ymax - ymin) * 0.025,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("Final Performance under Material-Matched Visual Stress", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_summary_markdown(summary: pd.DataFrame, output_path: Path) -> None:
    total_episodes = int(summary["episodes"].sum())
    total_successes = int(summary["successes"].sum())
    total_grasps = int(summary["grasps"].sum())
    overall_success_rate = total_successes / total_episodes if total_episodes else 0.0
    overall_grasp_rate = total_grasps / total_episodes if total_episodes else 0.0
    lines = [
        "# Material-Stress Evaluation Summary",
        "",
        "This evaluation combines material-level render profiles with material-matched observation pseudo-blur.",
        "Each profile should use a distinct seed range to avoid identical trajectories across object materials.",
        "When smoke-test and full-run outputs coexist, the summary uses the largest evaluation run per material profile.",
        "Rates are reported with Wilson 95% confidence intervals.",
        "",
        f"Overall success: {total_successes}/{total_episodes} ({overall_success_rate:.0%}); "
        f"overall grasp: {total_grasps}/{total_episodes} ({overall_grasp_rate:.0%}).",
        "",
        "| Object profile | Seed range | Success | Success rate | 95% CI | Grasp | Grasp rate | Mean steps | Mean reward |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile, row in summary.iterrows():
        label = PROFILE_LABELS.get(profile, profile)
        lines.append(
            f"| {label} | {int(row.seed_min)}-{int(row.seed_max)} | "
            f"{int(row.successes)}/{int(row.episodes)} | {row.success_rate:.0%} | "
            f"{row.success_ci95_low:.0%}-{row.success_ci95_high:.0%} | "
            f"{int(row.grasps)}/{int(row.episodes)} | {row.grasp_rate:.0%} | "
            f"{row.mean_steps:.1f} | {row.mean_reward:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "These results are stronger than render-only material profiles because the observation stream is also stressed.",
            "They are still simulation stress tests, not real-world transparent/depth-sensor validation.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize material-stress tactile diffusion evaluation results.")
    parser.add_argument("--results-dir", default="results/material_stress_eval")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    df = load_results(results_dir)
    summary = summarize(df)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "material_stress_plot_summary.csv")
    plot_success_grasp(summary, results_dir / "material_stress_success_grasp_chart.png")
    plot_performance(summary, results_dir / "material_stress_performance_chart.png")
    write_summary_markdown(summary, results_dir / "material_stress_summary.md")
    print(summary)
    print(f"Saved summary under: {results_dir}")


if __name__ == "__main__":
    main()
