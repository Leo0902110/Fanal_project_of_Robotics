from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation


STAGE_COLORS = {
    "observe": "#4C78A8",
    "request_probe": "#F58518",
    "refine_boundary": "#54A24B",
    "grasp": "#72B7B2",
    "transfer": "#B279A2",
    "release": "#59A14F",
}


def _bool(value) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _safe_float(row: pd.Series, column: str, default: float = 0.0) -> float:
    try:
        return float(row.get(column, default))
    except Exception:
        return default


def _stage(row: pd.Series) -> tuple[str, str]:
    phase = str(row.get("phase", "observe"))
    should_probe = _bool(row.get("should_probe", False))
    refined = _bool(row.get("refined", False))
    state = str(row.get("state", "observe"))

    if should_probe or state == "request_probe":
        return "request_probe", "Visual ambiguity triggers tactile probe"
    if refined:
        return "refine_boundary", "Tactile boundary confidence updated"
    if phase in {"close_gripper", "grasp"}:
        return "grasp", "Gripper closes using refined boundary"
    if phase in {"transfer", "lift"}:
        return "transfer", "Object is transferred to target"
    if phase == "release":
        return "release", "Object released at target"
    return "observe", "RGB-D pseudo-blur is evaluated"


def _load_video_frames(video_path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(video_path)
    try:
        frames = [frame for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames found in {video_path}")
    return frames


def _sample_frame(frames: list[np.ndarray], frame_index: int, trace_len: int) -> np.ndarray:
    if trace_len <= 1:
        return frames[min(frame_index, len(frames) - 1)]
    video_index = int(round(frame_index * (len(frames) - 1) / (trace_len - 1)))
    return frames[np.clip(video_index, 0, len(frames) - 1)]


def render_demo(video_path: Path, trace_path: Path, output_path: Path, fps: int, dpi: int) -> None:
    try:
        import imageio_ffmpeg

        mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    frames = _load_video_frames(video_path)
    trace = pd.read_csv(trace_path)
    if trace.empty:
        raise ValueError(f"Trace CSV is empty: {trace_path}")
    if "step" not in trace.columns:
        trace["step"] = np.arange(len(trace))

    steps = trace["step"].to_numpy()
    uncertainty = trace.get("visual_uncertainty", pd.Series(np.zeros(len(trace)))).to_numpy(dtype=float)
    ambiguity = trace.get("ambiguity_score", pd.Series(np.zeros(len(trace)))).to_numpy(dtype=float)
    boundary = trace.get("boundary_confidence", pd.Series(np.zeros(len(trace)))).to_numpy(dtype=float)
    post_probe = trace.get("post_probe_uncertainty", pd.Series(np.zeros(len(trace)))).to_numpy(dtype=float)
    reward = trace.get("reward", pd.Series(np.zeros(len(trace)))).to_numpy(dtype=float)

    fig = plt.figure(figsize=(14, 7.5), facecolor="#F8F8F5")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 0.42])
    video_ax = fig.add_subplot(grid[0, 0])
    panel_ax = fig.add_subplot(grid[0, 1])
    timeline_ax = fig.add_subplot(grid[1, :])

    fig.suptitle("Active Perception Grasping Demo: Vision Uncertainty -> Tactile Probe -> Grasp", fontsize=15)
    video_ax.axis("off")
    panel_ax.axis("off")

    image_artist = video_ax.imshow(_sample_frame(frames, 0, len(trace)))
    video_ax.set_title("ManiSkill simulated arm", fontsize=12)

    timeline_ax.set_xlim(float(steps.min()), float(steps.max()) if len(steps) > 1 else 1.0)
    timeline_ax.set_ylim(-0.05, 1.05)
    timeline_ax.set_xlabel("Step")
    timeline_ax.set_ylabel("Score")
    timeline_ax.grid(alpha=0.22)
    (unc_line,) = timeline_ax.plot([], [], color="#D94E41", lw=2.0, label="Visual uncertainty")
    (amb_line,) = timeline_ax.plot([], [], color="#4C78A8", lw=2.0, label="Decision ambiguity")
    (bound_line,) = timeline_ax.plot([], [], color="#54A24B", lw=2.0, label="Boundary confidence")
    (post_line,) = timeline_ax.plot([], [], color="#B279A2", lw=2.0, label="Post-probe uncertainty")
    cursor = timeline_ax.axvline(float(steps[0]), color="#222222", lw=1.3, alpha=0.7)
    probe_points = timeline_ax.scatter([], [], color="#F58518", marker="D", s=70, label="Probe")
    timeline_ax.legend(loc="upper right", ncol=5, fontsize=8)

    def draw(frame_index: int):
        row = trace.iloc[frame_index]
        stage, stage_text = _stage(row)
        color = STAGE_COLORS.get(stage, "#333333")

        image_artist.set_data(_sample_frame(frames, frame_index, len(trace)))

        upto = frame_index + 1
        x = steps[:upto]
        unc_line.set_data(x, uncertainty[:upto])
        amb_line.set_data(x, ambiguity[:upto])
        bound_line.set_data(x, boundary[:upto])
        post_line.set_data(x, post_probe[:upto])
        cursor.set_xdata([steps[frame_index], steps[frame_index]])

        probe_mask = trace.iloc[:upto]["should_probe"].astype(str).str.lower().isin(["true", "1", "yes"])
        probe_x = steps[:upto][probe_mask.to_numpy()]
        probe_y = ambiguity[:upto][probe_mask.to_numpy()]
        probe_points.set_offsets(
            np.column_stack([probe_x, probe_y]) if len(probe_x) else np.empty((0, 2))
        )

        panel_ax.clear()
        panel_ax.axis("off")
        panel_ax.text(0.0, 0.98, "Algorithm State", fontsize=14, fontweight="bold", va="top")
        panel_ax.text(
            0.0,
            0.86,
            stage.replace("_", " ").title(),
            fontsize=18,
            fontweight="bold",
            color=color,
            va="top",
        )
        panel_ax.text(0.0, 0.76, stage_text, fontsize=10.5, va="top", wrap=True)

        metrics = [
            ("Phase", str(row.get("phase", ""))),
            ("Visual uncertainty", f"{_safe_float(row, 'visual_uncertainty'):.3f}"),
            ("Decision ambiguity", f"{_safe_float(row, 'ambiguity_score'):.3f}"),
            ("Should probe", str(_bool(row.get("should_probe", False)))),
            ("Boundary confidence", f"{_safe_float(row, 'boundary_confidence'):.3f}"),
            ("Post-probe uncertainty", f"{_safe_float(row, 'post_probe_uncertainty'):.3f}"),
            ("Contact detected", str(_bool(row.get("contact_detected", False)))),
            ("Reward", f"{reward[frame_index]:.3f}"),
        ]

        y = 0.62
        for label, value in metrics:
            panel_ax.text(0.0, y, label, fontsize=9.5, color="#555555", va="center")
            panel_ax.text(0.58, y, value, fontsize=10.5, fontweight="bold", va="center")
            y -= 0.065

        reason = str(row.get("reason", ""))
        refine = str(row.get("refinement_reason", ""))
        panel_ax.text(0.0, 0.08, f"Decision: {reason}", fontsize=9, color="#333333", wrap=True)
        panel_ax.text(0.0, 0.02, f"Tactile update: {refine}", fontsize=9, color="#333333", wrap=True)
        return [image_artist, unc_line, amb_line, bound_line, post_line, cursor, probe_points]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, draw, frames=len(trace), interval=1000 / fps, blit=False)
    animation.save(output_path, writer=FFMpegWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved full algorithm demo: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine ManiSkill robot video and active-perception trace into one presentation MP4."
    )
    parser.add_argument("--video", default="results/colab_render_active_probe/active_probe.mp4")
    parser.add_argument("--trace", default="results/colab_render_active_probe/active_probe_decision_trace.csv")
    parser.add_argument("--output", default="results/full_active_perception_demo.mp4")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_demo(
        video_path=Path(args.video),
        trace_path=Path(args.trace),
        output_path=Path(args.output),
        fps=args.fps,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
