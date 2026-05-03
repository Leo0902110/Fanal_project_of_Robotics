from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROFILE_NOTES = {
    "transparent": "depth dropout + missing object boundary",
    "dark": "low RGB signal + noisy depth",
    "reflective": "specular highlights + wrong depth edges",
    "low_texture": "weak texture + ambiguous boundary",
}


def _read_trace(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


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


def _rng(profile: str, frame_index: int) -> np.random.Generator:
    seed = abs(hash((profile, frame_index))) % (2**32)
    return np.random.default_rng(seed)


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


def make_observation_proxy(frame: np.ndarray, profile: str, frame_index: int) -> np.ndarray:
    rgb = _as_rgb(frame).astype(np.float32) / 255.0
    height, width = rgb.shape[:2]
    rng = _rng(profile, frame_index)
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
        img = Image.fromarray(np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0))
        img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
        rgb = np.asarray(img).astype(np.float32) / 255.0
        banding = (np.floor(rgb * 5.0) / 5.0).astype(np.float32)
        rgb = 0.65 * rgb + 0.35 * banding
    else:
        rgb = np.clip(rgb, 0.0, 1.0)

    return np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0)


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=(255, 255, 255)) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 5
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill, font=font)


def compose_frame(frame: np.ndarray, proxy: np.ndarray, trace_row: dict[str, str], profile: str, frame_index: int) -> np.ndarray:
    left = Image.fromarray(_as_rgb(frame))
    right = Image.fromarray(proxy)
    width, height = left.size
    canvas = Image.new("RGB", (width * 2, height + 92), (18, 18, 18))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (width, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    small_font = _font(14)
    _draw_label(draw, (12, 10), "Simulator render", title_font)
    _draw_label(draw, (width + 12, 10), f"Policy observation proxy: {profile}", title_font)

    uncertainty = float(trace_row.get("uncertainty", 0.0) or 0.0)
    phase = trace_row.get("phase", "")
    success = trace_row.get("success", "0")
    grasp = trace_row.get("is_grasped", "0")
    note = PROFILE_NOTES.get(profile, "observation-level pseudo-blur")
    y0 = height + 10
    draw.text((14, y0), f"profile={profile} | {note}", fill=(245, 245, 245), font=small_font)
    draw.text((14, y0 + 24), f"step={frame_index:02d}  phase={phase}  uncertainty={uncertainty:.2f}", fill=(220, 230, 255), font=small_font)
    draw.text((14, y0 + 48), f"condition_clip_sigma=2.0  success={success}  grasp={grasp}", fill=(210, 255, 220), font=small_font)
    bar_x = width + 14
    bar_y = y0 + 34
    bar_w = width - 28
    draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + 14), outline=(210, 210, 210))
    draw.rectangle((bar_x, bar_y, bar_x + int(bar_w * min(max(uncertainty, 0.0), 1.0)), bar_y + 14), fill=(255, 150, 70))
    draw.text((bar_x, bar_y + 22), "visual uncertainty", fill=(230, 230, 230), font=small_font)
    return np.asarray(canvas)


def render_demo(input_video: Path, trace_csv: Path, output_video: Path, profile: str, fps: int) -> None:
    reader = imageio.get_reader(input_video)
    trace_rows = _read_trace(trace_csv)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_video, fps=fps)
    frame_count = 0
    try:
        for idx, frame in enumerate(reader):
            row = trace_rows[min(idx, len(trace_rows) - 1)] if trace_rows else {}
            proxy = make_observation_proxy(frame, profile, idx)
            writer.append_data(compose_frame(frame, proxy, row, profile, idx))
            frame_count += 1
    finally:
        writer.close()
        reader.close()
    if frame_count == 0:
        raise ValueError(f"No frames found in {input_video}")
    print(f"Saved annotated demo: {output_video}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a side-by-side presentation video for pseudo-blur demos.")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_NOTES), required=True)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_demo(Path(args.input_video), Path(args.trace_csv), Path(args.output_video), args.profile, args.fps)


if __name__ == "__main__":
    main()