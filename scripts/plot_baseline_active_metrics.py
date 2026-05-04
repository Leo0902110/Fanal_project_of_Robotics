from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("visual_uncertainty", "Visual uncertainty"),
    ("boundary_confidence", "Boundary confidence"),
    ("contact_strength", "Contact strength"),
]


def _load_trace(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "step" not in df.columns:
        df = df.reset_index().rename(columns={"index": "step"})
    df["condition"] = label
    for column, _ in METRICS:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["step"] = df["step"].where(df["step"].notna(), pd.Series(range(len(df)), index=df.index)).astype(int)
    return df


def _summary(df: pd.DataFrame, label: str) -> dict[str, float | str]:
    return {
        "condition": label,
        "steps": int(len(df)),
        "max_visual_uncertainty": float(df["visual_uncertainty"].max()),
        "mean_visual_uncertainty": float(df["visual_uncertainty"].mean()),
        "max_boundary_confidence": float(df["boundary_confidence"].max()),
        "final_boundary_confidence": float(df["boundary_confidence"].iloc[-1]) if len(df) else 0.0,
        "max_contact_strength": float(df["contact_strength"].max()),
        "contact_step_count": int((df["contact_strength"] > 0).sum()),
        "probe_request_count": (
            int((df["should_probe"].astype(str).str.lower() == "true").sum()) if "should_probe" in df.columns else 0
        ),
    }


def plot_metrics(baseline_trace: Path, active_trace: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_trace(baseline_trace, "Baseline: vision only")
    active = _load_trace(active_trace, "Active: vision + tactile")

    colors = {
        "Baseline: vision only": "#d94e41",
        "Active: vision + tactile": "#2f7f6f",
    }
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(12, 9), sharex=True)
    for ax, (column, title) in zip(axes, METRICS):
        for df in (baseline, active):
            label = str(df["condition"].iloc[0])
            ax.plot(df["step"], df[column], label=label, linewidth=2.2, color=colors[label])
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Step")
    axes[0].legend(loc="upper right")
    fig.suptitle("Baseline vs Active Tactile Perception Metrics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = output_dir / "baseline_vs_active_metrics.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = pd.DataFrame([_summary(baseline, "Baseline: vision only"), _summary(active, "Active: vision + tactile")])
    summary_path = output_dir / "baseline_vs_active_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline-vs-active wrist-camera demo metrics.")
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--active-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline_active_comparison"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_metrics(args.baseline_trace, args.active_trace, args.output_dir)


if __name__ == "__main__":
    main()
