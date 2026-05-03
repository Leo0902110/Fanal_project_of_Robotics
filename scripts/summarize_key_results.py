from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPERIMENTS = [
    ("joint_scripted", "policy_comparison_joint500/joint_scripted/joint_scripted_eval_results.csv", "seeds 600-619"),
    ("BC all-demo", "policy_comparison_joint500/bc/bc_eval_results.csv", "seeds 600-619"),
    ("BC success-only", "policy_comparison_success_only/bc_success_only/bc_eval_results.csv", "seeds 600-619"),
    ("BC oracle-geometry", "policy_comparison_oracle_geometry/bc_oracle_geometry/bc_eval_results.csv", "seeds 600-619"),
    ("BC oracle + assist", "success90_oracle_geometry/bc_oracle_geometry_assist_pd_joint_pos/bc_eval_results.csv", "seeds 600-619"),
    ("D MLP 500", "d_policy_assist_teacher_500_eval_seed3000_scale036/d_eval_results.csv", "seeds 3000-3019"),
    ("D MLP 500", "d_policy_assist_teacher_500_eval_seed1200_scale036/d_eval_results.csv", "seeds 1200-1219"),
    ("Direct tactile DP", "tactile_dp_teacher_500_phase_eval_seed3000_noise02_5ep/tdp_eval_results.csv", "seeds 3000-3004"),
    ("Residual DP scale=0.00", "tactile_dp_residual_d500_h1_seed3000_scale000_fixed_20ep/tdp_eval_results.csv", "seeds 3000-3019"),
    ("Residual DP scale=0.00", "tactile_dp_residual_d500_h1_seed1200_scale000_fixed_20ep/tdp_eval_results.csv", "seeds 1200-1219"),
    ("Residual DP scale=0.05", "tactile_dp_residual_d500_h1_seed3000_scale005_fixed_20ep/tdp_eval_results.csv", "seeds 3000-3019"),
    ("Residual DP scale=0.05", "tactile_dp_residual_d500_h1_seed1200_scale005_fixed_20ep/tdp_eval_results.csv", "seeds 1200-1219"),
    ("Residual DP scale=0.10", "tactile_dp_residual_d500_h1_seed3000_scale010_fixed_20ep/tdp_eval_results.csv", "seeds 3000-3019"),
    ("Residual DP scale=0.10", "tactile_dp_residual_d500_h1_seed1200_scale010_fixed_20ep/tdp_eval_results.csv", "seeds 1200-1219"),
]


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def summarize_csv(path: Path) -> dict[str, float | int | str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    episodes = len(rows)
    success_count = sum(_float(row, "success") > 0.0 for row in rows)
    if rows and "max_is_grasped" in rows[0]:
        grasp_count = sum(_float(row, "max_is_grasped") > 0.0 for row in rows)
        grasp_rate = grasp_count / max(episodes, 1)
    else:
        grasp_count = ""
        grasp_rate = ""
    mean_reward = sum(_float(row, "total_reward") for row in rows) / max(episodes, 1)
    return {
        "episodes": episodes,
        "success_count": int(success_count),
        "success_rate": success_count / max(episodes, 1),
        "grasp_count": grasp_count,
        "grasp_rate": grasp_rate,
        "mean_reward": mean_reward,
    }


def format_rate(value: float | str) -> str:
    if value == "":
        return ""
    return f"{float(value):.2f}"


def write_outputs(rows: list[dict[str, str | int | float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "key_results_summary.csv"
    md_path = output_dir / "key_results_summary.md"
    json_path = output_dir / "key_results_summary.json"
    fieldnames = ["method", "split", "episodes", "success", "success_rate", "grasp", "grasp_rate", "mean_reward", "source"]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Key PickCube Results",
        "",
        "| Method | Split | Success | Success Rate | Grasp | Grasp Rate | Mean Reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {split} | {success} | {success_rate} | {grasp} | {grasp_rate} | {mean_reward:.3f} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "json": str(json_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize key PickCube evaluation results into CSV/Markdown tables.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_rows = []
    for method, relative_path, split in EXPERIMENTS:
        path = results_dir / relative_path
        if not path.exists():
            print(f"Skipping missing result: {path}")
            continue
        summary = summarize_csv(path)
        success = f"{summary['success_count']}/{summary['episodes']}"
        grasp = "" if summary["grasp_count"] == "" else f"{summary['grasp_count']}/{summary['episodes']}"
        output_rows.append(
            {
                "method": method,
                "split": split,
                "episodes": summary["episodes"],
                "success": success,
                "success_rate": format_rate(summary["success_rate"]),
                "grasp": grasp,
                "grasp_rate": format_rate(summary["grasp_rate"]),
                "mean_reward": round(float(summary["mean_reward"]), 6),
                "source": str(path),
            }
        )
    write_outputs(output_rows, Path(args.output_dir))


if __name__ == "__main__":
    main()