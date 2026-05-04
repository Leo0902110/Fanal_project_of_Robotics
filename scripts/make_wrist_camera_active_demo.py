from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _bool(value) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _safe_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _load_video(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        frames = [np.asarray(frame[..., :3], dtype=np.uint8) for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames found in {path}")
    return frames


def _sample_frame(frames: list[np.ndarray], index: int, trace_len: int) -> np.ndarray:
    if trace_len <= 1:
        return frames[min(index, len(frames) - 1)]
    video_index = int(round(index * (len(frames) - 1) / (trace_len - 1)))
    return frames[int(np.clip(video_index, 0, len(frames) - 1))]


def _wrist_proxy(frame: np.ndarray, uncertainty: float, step: int) -> Image.Image:
    image = Image.fromarray(frame)
    width, height = image.size

    # Approximate a wrist/eye-in-hand crop by focusing the central workspace.
    crop_w = int(width * 0.66)
    crop_h = int(height * 0.66)
    left = max((width - crop_w) // 2, 0)
    top = max(int(height * 0.22), 0)
    crop = image.crop((left, top, min(left + crop_w, width), min(top + crop_h, height)))
    crop = crop.resize((720, 540), Image.Resampling.BICUBIC)

    # Create a policy-observation style degradation when uncertainty is high.
    arr = np.asarray(crop).astype(np.float32)
    rng = np.random.default_rng(1000 + int(step))
    gray = arr.mean(axis=-1, keepdims=True)
    arr = 0.62 * arr + 0.38 * gray
    if uncertainty > 0.25:
        noise = rng.normal(0.0, 22.0 * uncertainty, arr.shape)
        arr = arr + noise
        mask = rng.random(arr.shape[:2]) < (0.10 * uncertainty)
        arr[mask] = 235.0
        crop = Image.fromarray(np.uint8(np.clip(arr, 0, 255))).filter(
            ImageFilter.GaussianBlur(radius=0.8 + 1.4 * uncertainty)
        )
    else:
        crop = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))

    draw = ImageDraw.Draw(crop)
    font = _font(18)
    small = _font(14)
    draw.rectangle((0, 0, 720, 38), fill=(0, 0, 0))
    draw.text((14, 9), "Wrist-camera proxy: blurred object boundary", fill=(245, 245, 245), font=font)

    # Reticle and ambiguous boundary marks.
    cx, cy = 360, 285
    draw.ellipse((cx - 58, cy - 58, cx + 58, cy + 58), outline=(255, 210, 70), width=3)
    draw.line((cx - 78, cy, cx - 22, cy), fill=(255, 210, 70), width=2)
    draw.line((cx + 22, cy, cx + 78, cy), fill=(255, 210, 70), width=2)
    draw.line((cx, cy - 78, cx, cy - 22), fill=(255, 210, 70), width=2)
    draw.line((cx, cy + 22, cx, cy + 78), fill=(255, 210, 70), width=2)
    for i in range(7):
        x0 = int(250 + 22 * i + 8 * np.sin(step + i))
        y0 = int(225 + 12 * np.cos(0.5 * step + i))
        draw.rectangle((x0, y0, x0 + 18, y0 + 6), outline=(255, 90, 65), width=2)

    draw.rectangle((14, 488, 706, 526), fill=(0, 0, 0))
    draw.text(
        (24, 499),
        f"visual uncertainty={uncertainty:.3f}  boundary cues degraded by pseudo-blur",
        fill=(255, 235, 190),
        font=small,
    )
    return crop


def _stage(row: pd.Series, index: int, probe_indices: list[int], pedagogical_contact: bool) -> tuple[str, str, tuple[int, int, int]]:
    phase = str(row.get("phase", "observe"))
    should_probe = _bool(row.get("should_probe", False))
    refined = _bool(row.get("refined", False))
    contact = _bool(row.get("contact_detected", False))

    if should_probe:
        return "REQUEST PROBE", "Visual boundary is ambiguous; spend one tactile probing action.", (245, 133, 24)

    if pedagogical_contact and any(0 < index - p <= 3 for p in probe_indices):
        return "TACTILE CHECK", "Probe motion touches/clears the hypothesized boundary region.", (76, 120, 168)

    if refined or contact:
        return "REFINE BOUNDARY", "Contact feedback updates boundary confidence and grasp target.", (84, 162, 75)

    if phase in {"close_gripper", "grasp"}:
        return "GRASP", "Gripper closes using the current object boundary estimate.", (114, 183, 178)
    if phase in {"transfer", "lift"}:
        return "TRANSFER", "Object is lifted and moved toward the target.", (178, 121, 162)
    if phase == "release":
        return "RELEASE", "Object is released at the goal.", (89, 161, 79)
    return "OBSERVE", "Eye-in-hand view estimates boundary uncertainty.", (217, 78, 65)


def _draw_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, value: float, color: tuple[int, int, int]) -> None:
    draw.rectangle((x, y, x + w, y + h), outline=(190, 190, 190), width=1)
    fill_w = int(np.clip(value, 0.0, 1.0) * w)
    draw.rectangle((x, y, x + fill_w, y + h), fill=color)


def _compose(
    frame: np.ndarray,
    row: pd.Series,
    index: int,
    probe_indices: list[int],
    pedagogical_contact: bool,
) -> np.ndarray:
    uncertainty = _safe_float(row, "visual_uncertainty")
    ambiguity = _safe_float(row, "ambiguity_score")
    boundary = _safe_float(row, "boundary_confidence")
    post_probe = _safe_float(row, "post_probe_uncertainty")

    wrist = _wrist_proxy(frame, uncertainty, int(row.get("step", index)))
    render = Image.fromarray(frame).resize((640, 480), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (1440, 820), (248, 248, 245))
    canvas.paste(wrist, (35, 80))
    canvas.paste(render, (765, 80))

    draw = ImageDraw.Draw(canvas)
    title_font = _font(32)
    h_font = _font(24)
    body_font = _font(18)
    small_font = _font(15)
    draw.text((35, 24), "Eye-in-Hand Active Perception Demo", fill=(20, 20, 20), font=title_font)
    draw.text((765, 48), "ManiSkill third-person render", fill=(50, 50, 50), font=h_font)

    stage, stage_text, color = _stage(row, index, probe_indices, pedagogical_contact)
    draw.rounded_rectangle((35, 645, 1405, 790), radius=10, fill=(255, 255, 255), outline=(220, 220, 220))
    draw.text((60, 668), stage, fill=color, font=h_font)
    draw.text((60, 704), stage_text, fill=(40, 40, 40), font=body_font)
    draw.text(
        (60, 744),
        f"phase={row.get('phase', '')}  should_probe={_bool(row.get('should_probe', False))}  "
        f"decision={row.get('reason', '')}  tactile_update={row.get('refinement_reason', '')}",
        fill=(70, 70, 70),
        font=small_font,
    )

    x0, y0 = 760, 590
    draw.text((x0, y0), "Live signals", fill=(20, 20, 20), font=h_font)
    metrics = [
        ("visual uncertainty", uncertainty, (217, 78, 65)),
        ("decision ambiguity", ambiguity, (76, 120, 168)),
        ("boundary confidence", boundary, (84, 162, 75)),
        ("post-probe uncertainty", post_probe, (178, 121, 162)),
    ]
    for i, (label, value, metric_color) in enumerate(metrics):
        y = y0 + 42 + i * 38
        draw.text((x0, y), f"{label:24s} {value:.3f}", fill=(55, 55, 55), font=small_font)
        _draw_bar(draw, x0 + 260, y + 4, 320, 16, value, metric_color)

    if _bool(row.get("should_probe", False)):
        draw.ellipse((1268, 658, 1338, 728), fill=(245, 133, 24))
        draw.text((1288, 678), "P", fill=(255, 255, 255), font=h_font)
        draw.text((1200, 735), "Probe request", fill=(245, 133, 24), font=body_font)

    return np.asarray(canvas)


def render_demo(
    video_path: Path,
    trace_path: Path,
    output_path: Path,
    *,
    fps: int,
    pedagogical_contact: bool,
) -> None:
    frames = _load_video(video_path)
    trace = pd.read_csv(trace_path)
    if trace.empty:
        raise ValueError(f"Trace CSV is empty: {trace_path}")
    if "step" not in trace.columns:
        trace["step"] = np.arange(len(trace))

    probe_mask = trace["should_probe"].astype(str).str.lower().isin(["true", "1", "yes"])
    probe_indices = list(np.flatnonzero(probe_mask.to_numpy()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for index, row in trace.iterrows():
            frame = _sample_frame(frames, int(index), len(trace))
            writer.append_data(_compose(frame, row, int(index), probe_indices, pedagogical_contact))
    finally:
        writer.close()
    print(f"Saved wrist-camera active perception demo: {output_path}")
    print(f"frames={len(trace)}, fps={fps}, probe_requests={len(probe_indices)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a presentation video that approximates an eye-in-hand camera view for the active-perception rollout."
    )
    parser.add_argument("--video", required=True, help="Input ManiSkill rollout mp4.")
    parser.add_argument("--trace", required=True, help="Input active_probe_decision_trace.csv.")
    parser.add_argument("--output", required=True, help="Output presentation mp4.")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument(
        "--pedagogical-contact",
        action="store_true",
        help="Show a short tactile-check stage after probe requests even when contact evidence is absent in this rollout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_demo(
        Path(args.video),
        Path(args.trace),
        Path(args.output),
        fps=args.fps,
        pedagogical_contact=args.pedagogical_contact,
    )


if __name__ == "__main__":
    main()
