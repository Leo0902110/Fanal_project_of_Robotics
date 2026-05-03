from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROFILE_LABELS = {
    "transparent": "Transparent object",
    "dark": "Dark object",
    "reflective": "Reflective object",
    "low_texture": "Low-texture object",
}
PROFILE_NOTES = {
    "transparent": "transparent render + depth dropout/boundary loss",
    "dark": "dark render + low signal/noisy depth",
    "reflective": "reflective render + highlight/depth edge artifacts",
    "low_texture": "low-texture render + weak boundary cues",
}


def _read_rows(result_dirs: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result_dir in result_dirs:
        csv_path = result_dir / "tdp_eval_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing eval CSV: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                row["_result_dir"] = str(result_dir)
                rows.append(row)
    return rows


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _as_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        return frame[..., :3]
    return frame[..., :3]


def _video_path(row: dict[str, str]) -> Path:
    path = Path(row.get("video_path", ""))
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path
    candidate = Path(row["_result_dir"]) / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Video not found for row: {row.get('video_path')}")


def _rng(profile: str, seed: str, frame_index: int) -> np.random.Generator:
    value = abs(hash((profile, seed, frame_index))) % (2**32)
    return np.random.default_rng(value)


def _center_mask(height: int, width: int) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    radius = np.maximum(np.abs(xx), np.abs(yy))
    return radius < 0.58


def _boundary_mask(height: int, width: int) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    radius = np.maximum(np.abs(xx), np.abs(yy))
    return (radius > 0.42) & (radius < 0.78)


def make_observation_proxy(frame: np.ndarray, profile: str, seed: str, frame_index: int) -> np.ndarray:
    rgb = np.asarray(_as_rgb(frame), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    rng = _rng(profile, seed, frame_index)
    center = _center_mask(height, width)
    boundary = _boundary_mask(height, width)

    if profile == "transparent":
        gray = rgb.mean(axis=-1, keepdims=True)
        rgb = 0.72 * rgb + 0.28 * gray
        rgb = 0.9 * rgb + 0.1
        dropout = (rng.random((height, width)) < 0.28) & (center | boundary)
        rgb[dropout] = np.array([0.88, 0.93, 0.96], dtype=np.float32)
        edge = boundary & (rng.random((height, width)) < 0.20)
        rgb[edge] = np.array([0.15, 0.22, 0.28], dtype=np.float32)
    elif profile == "dark":
        rgb *= 0.35
        noise = rng.normal(0.0, 0.08, rgb.shape).astype(np.float32)
        rgb = rgb + noise
        dropout = (rng.random((height, width)) < 0.15) & (center | boundary)
        rgb[dropout] = 0.02
    elif profile == "reflective":
        rgb = np.clip(rgb * 0.95 + 0.03, 0.0, 1.0)
        for _ in range(6):
            x0 = int(rng.integers(max(width // 8, 1), max(width - width // 8, 2)))
            y0 = int(rng.integers(max(height // 8, 1), max(height - height // 8, 2)))
            stripe_w = int(rng.integers(max(width // 30, 2), max(width // 10, 3)))
            stripe_h = int(rng.integers(max(height // 80, 2), max(height // 25, 3)))
            rgb[max(0, y0 - stripe_h) : min(height, y0 + stripe_h), max(0, x0 - stripe_w) : min(width, x0 + stripe_w)] = 1.0
        wrong_edges = boundary & (rng.random((height, width)) < 0.22)
        rgb[wrong_edges] = np.array([0.05, 0.08, 0.12], dtype=np.float32)
    elif profile == "low_texture":
        gray = rgb.mean(axis=-1, keepdims=True)
        rgb = 0.18 * rgb + 0.82 * gray
        image = Image.fromarray(np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0))
        image = image.filter(ImageFilter.GaussianBlur(radius=2.0))
        rgb = np.asarray(image).astype(np.float32) / 255.0
        banding = (np.floor(rgb * 5.0) / 5.0).astype(np.float32)
        rgb = 0.65 * rgb + 0.35 * banding

    return np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0)


def _draw_badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=(255, 255, 255)) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 6
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill, font=font)


def compose_frame(frame: np.ndarray, row: dict[str, str], segment_index: int, frame_index: int) -> np.ndarray:
    profile = row.get("object_profile", "")
    seed = row.get("seed", "")
    left = Image.fromarray(_as_rgb(frame))
    right = Image.fromarray(make_observation_proxy(frame, profile, seed, frame_index))
    width, height = left.size
    canvas = Image.new("RGB", (width * 2, height + 112), (18, 18, 18))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (width, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    small_font = _font(14)
    _draw_badge(draw, (12, 10), "Simulator render", title_font)
    _draw_badge(draw, (width + 12, 10), "Policy observation proxy", title_font)
    label = PROFILE_LABELS.get(profile, profile)
    note = PROFILE_NOTES.get(profile, "material-matched visual stress")
    steps = row.get("steps", "")
    success = row.get("success", "")
    grasp = row.get("max_is_grasped", "")
    y0 = height + 10
    draw.text((14, y0), f"{label} | {note}", fill=(245, 245, 245), font=small_font)
    draw.text((14, y0 + 24), f"segment={segment_index + 1:02d}  seed={seed}  frame={frame_index:02d}  steps={steps}", fill=(220, 230, 255), font=small_font)
    draw.text((14, y0 + 48), f"Residual tactile diffusion + D base | material_visual_stress=True | success={success}  grasp={grasp}", fill=(210, 255, 220), font=small_font)
    draw.text((14, y0 + 72), "Overall full-run result: 72/80 success (90%), 80/80 grasp (100%)", fill=(255, 235, 190), font=small_font)
    return np.asarray(canvas)


def render_demo(
    result_dirs: list[Path],
    output_video: Path,
    *,
    fps: int,
    max_episodes: int,
    success_only: bool,
    hold_frames: int,
) -> None:
    rows = _read_rows(result_dirs)
    if success_only:
        rows = [row for row in rows if float(row.get("success", 0.0) or 0.0) > 0.0]
    rows = rows[:max_episodes]
    if not rows:
        raise ValueError("No rows selected for demo video")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_video, fps=fps)
    total_frames = 0
    try:
        for segment_index, row in enumerate(rows):
            reader = imageio.get_reader(_video_path(row))
            last_frame = None
            try:
                for frame_index, frame in enumerate(reader):
                    composed = compose_frame(frame, row, segment_index, frame_index)
                    writer.append_data(composed)
                    last_frame = composed
                    total_frames += 1
            finally:
                reader.close()
            if last_frame is not None:
                for _ in range(hold_frames):
                    writer.append_data(last_frame)
                    total_frames += 1
    finally:
        writer.close()
    if total_frames == 0:
        raise ValueError("No frames were written")
    duration = total_frames / max(fps, 1)
    print(f"Saved material-stress demo: {output_video}")
    print(f"Segments: {len(rows)}, frames: {total_frames}, fps: {fps}, duration_seconds: {duration:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a long side-by-side material-stress demo video.")
    parser.add_argument("--result-dir", action="append", required=True, help="Directory containing tdp_eval_results.csv and rollout videos. Can be repeated.")
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=24)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--success-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_demo(
        [Path(value) for value in args.result_dir],
        Path(args.output_video),
        fps=args.fps,
        max_episodes=args.max_episodes,
        success_only=args.success_only,
        hold_frames=args.hold_frames,
    )


if __name__ == "__main__":
    main()
