from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
import pandas as pd


COLORS = {
    "visual_uncertainty": "#D94E41",
    "ambiguity_score": "#4C78A8",
    "boundary_confidence": "#54A24B",
    "post_probe_uncertainty": "#B279A2",
    "probe": "#F58518",
}


def _bool_series(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    return values.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "ambiguity_score": 0.0,
        "visual_uncertainty": 0.0,
        "boundary_confidence": 0.0,
        "post_probe_uncertainty": 0.0,
        "should_probe": False,
        "state": "observe",
        "reason": "",
        "refinement_reason": "",
        "reward": 0.0,
    }
    for column, value in defaults.items():
        if column not in df.columns:
            df[column] = value
    if "step" not in df.columns:
        df["step"] = np.arange(len(df))
    return df


def render_trace(input_csv: Path, output_path: Path, fps: int, dpi: int) -> None:
    if output_path.suffix.lower() != ".gif":
        try:
            import imageio_ffmpeg

            mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    df = _ensure_columns(pd.read_csv(input_csv))
    steps = df["step"].to_numpy()
    should_probe = _bool_series(df["should_probe"])

    fig = plt.figure(figsize=(11, 6))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1.25], width_ratios=[3, 1.15])
    ax = fig.add_subplot(grid[0, 0])
    side = fig.add_subplot(grid[:, 1])
    timeline = fig.add_subplot(grid[1, 0])

    ax.set_xlim(float(steps.min()), float(steps.max()) if len(steps) > 1 else 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.25)

    lines = {}
    for column, label in [
        ("visual_uncertainty", "Visual uncertainty"),
        ("ambiguity_score", "Decision ambiguity"),
        ("boundary_confidence", "Boundary confidence"),
        ("post_probe_uncertainty", "Post-probe uncertainty"),
    ]:
        (line,) = ax.plot([], [], lw=2.2, color=COLORS[column], label=label)
        lines[column] = line
    probe_scatter = ax.scatter([], [], s=80, color=COLORS["probe"], marker="D", label="Probe request", zorder=5)
    cursor = ax.axvline(0, color="#333333", lw=1.2, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)

    timeline.set_xlim(float(steps.min()), float(steps.max()) if len(steps) > 1 else 1.0)
    timeline.set_ylim(0, 1)
    timeline.set_yticks([])
    timeline.set_xlabel("Decision timeline")
    timeline.grid(axis="x", alpha=0.2)

    side.axis("off")
    fig.suptitle("Active Perception Trace: Ambiguity to Tactile Probe", fontsize=14)

    def draw_frame(frame: int):
        upto = frame + 1
        x = steps[:upto]
        for column, line in lines.items():
            line.set_data(x, df[column].to_numpy(dtype=float)[:upto])

        probe_x = steps[:upto][should_probe[:upto]]
        probe_y = df["ambiguity_score"].to_numpy(dtype=float)[:upto][should_probe[:upto]]
        probe_scatter.set_offsets(np.column_stack([probe_x, probe_y]) if len(probe_x) else np.empty((0, 2)))
        cursor.set_xdata([steps[frame], steps[frame]])

        timeline.clear()
        timeline.set_xlim(float(steps.min()), float(steps.max()) if len(steps) > 1 else 1.0)
        timeline.set_ylim(0, 1)
        timeline.set_yticks([])
        timeline.set_xlabel("Decision timeline")
        timeline.grid(axis="x", alpha=0.2)
        for idx in range(upto):
            color = COLORS["probe"] if should_probe[idx] else "#D0D0D0"
            timeline.barh(0.5, 0.8, left=float(steps[idx]) - 0.4, height=0.45, color=color)
        timeline.axvline(float(steps[frame]), color="#333333", lw=1.2)

        row = df.iloc[frame]
        side.clear()
        side.axis("off")
        text = "\n".join(
            [
                f"Step: {int(row['step'])}",
                f"State: {row['state']}",
                f"Probe: {bool(should_probe[frame])}",
                "",
                f"Reason: {row['reason']}",
                f"Refine: {row['refinement_reason']}",
                "",
                f"Ambiguity: {float(row['ambiguity_score']):.3f}",
                f"Visual unc.: {float(row['visual_uncertainty']):.3f}",
                f"Boundary conf.: {float(row['boundary_confidence']):.3f}",
                f"Post-probe unc.: {float(row['post_probe_uncertainty']):.3f}",
                f"Reward: {float(row['reward']):.3f}",
            ]
        )
        side.text(0.02, 0.96, text, va="top", ha="left", fontsize=10, family="monospace")
        return [*lines.values(), probe_scatter, cursor]

    animation = FuncAnimation(fig, draw_frame, frames=len(df), interval=1000 / fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".gif":
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        animation.save(output_path, writer=FFMpegWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved trace animation: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an active-perception decision trace as an animation.")
    parser.add_argument(
        "--input",
        default="results/local_mvp_rgbd/active_probe/active_probe_decision_trace.csv",
    )
    parser.add_argument("--output", default="results/active_probe_decision_trace.mp4")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_trace(Path(args.input), Path(args.output), fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
