from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def _as_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        return frame[..., :3]
    return frame[..., :3]


def _draw_badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=(255, 255, 255)) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 6
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill, font=font)


def _compose(frame: np.ndarray, row: dict[str, str], segment_index: int, frame_index: int) -> np.ndarray:
    image = Image.fromarray(_as_rgb(frame))
    width, height = image.size
    canvas = Image.new("RGB", (width, height + 74), (16, 16, 16))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    small_font = _font(14)
    profile = row.get("object_profile") or row.get("pseudo_blur_profile", "")
    episode = row.get("episode", "")
    seed = row.get("seed", "")
    steps = row.get("steps", "")
    success = row.get("success", "")
    grasp = row.get("max_is_grasped", "")
    _draw_badge(draw, (12, 10), f"Material object: {profile}", title_font)
    y0 = height + 10
    draw.text(
        (14, y0),
        f"randomized position demo {segment_index + 1} | episode={episode} seed={seed} frame={frame_index:02d}",
        fill=(240, 240, 240),
        font=small_font,
    )
    draw.text(
        (14, y0 + 24),
        f"steps={steps}  success={success}  grasp={grasp}  scene=material_object",
        fill=(210, 255, 220),
        font=small_font,
    )
    return np.asarray(canvas)


def render_position_demo(
    result_dirs: list[Path],
    output_video: Path,
    *,
    fps: int,
    max_episodes: int,
    success_only: bool,
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
            try:
                for frame_index, frame in enumerate(reader):
                    writer.append_data(_compose(frame, row, segment_index, frame_index))
                    total_frames += 1
            finally:
                reader.close()
    finally:
        writer.close()
    if total_frames == 0:
        raise ValueError("No frames were written")
    print(f"Saved multi-position material demo: {output_video}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate material-object rollouts into a randomized-position demo video.")
    parser.add_argument("--result-dir", action="append", required=True, help="Directory containing tdp_eval_results.csv and rollout videos. Can be repeated.")
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=4)
    parser.add_argument("--success-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_position_demo(
        [Path(value) for value in args.result_dir],
        Path(args.output_video),
        fps=args.fps,
        max_episodes=args.max_episodes,
        success_only=args.success_only,
    )


if __name__ == "__main__":
    main()