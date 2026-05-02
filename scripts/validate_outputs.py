from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_mvp_results(path: Path, min_success: float, require_policy: str | None) -> None:
    rows = _load_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No MVP result rows found in {path}")

    max_success = max(float(row.get("success_rate", row.get("success", 0.0)) or 0.0) for row in rows)
    if max_success < min_success:
        raise ValueError(
            f"MVP scripted baseline check failed: max success_rate={max_success:.3f} < required {min_success:.3f}."
        )

    if require_policy is not None:
        bad = [
            row for row in rows if str(row.get("effective_policy", row.get("requested_policy", ""))) != require_policy
        ]
        if bad:
            raise ValueError(
                f"MVP policy validation failed: found rows not using effective_policy={require_policy} in {path}."
            )

    print(
        json.dumps(
            {
                "validated_file": str(path),
                "type": "mvp_results",
                "num_rows": len(rows),
                "max_success_rate": max_success,
                "required_success_rate": min_success,
                "required_policy": require_policy,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def validate_manifest(path: Path, min_mean_success: float, require_policy: str | None) -> None:
    rows = _load_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No manifest rows found in {path}")

    successes = [float(row.get("success", 0.0) or 0.0) for row in rows]
    mean_success = sum(successes) / len(successes)
    max_success = max(successes)

    if mean_success < min_mean_success:
        raise ValueError(
            f"Demo manifest validation failed: mean success={mean_success:.3f} < required {min_mean_success:.3f}."
        )

    if require_policy is not None:
        bad = [row for row in rows if str(row.get("effective_policy", row.get("requested_policy", ""))) != require_policy]
        if bad:
            raise ValueError(
                f"Demo manifest policy validation failed: found rows not using effective_policy={require_policy} in {path}."
            )

    print(
        json.dumps(
            {
                "validated_file": str(path),
                "type": "manifest",
                "num_rows": len(rows),
                "mean_success": mean_success,
                "max_success": max_success,
                "required_mean_success": min_mean_success,
                "required_policy": require_policy,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def validate_csv_success(path: Path, min_success: float) -> None:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No CSV rows found in {path}")
    max_success = max(float(row.get("success_rate", row.get("success", 0.0)) or 0.0) for row in rows)
    if max_success < min_success:
        raise ValueError(f"CSV success validation failed: max success_rate={max_success:.3f} < required {min_success:.3f}.")
    print(
        json.dumps(
            {
                "validated_file": str(path),
                "type": "csv_success",
                "num_rows": len(rows),
                "max_success_rate": max_success,
                "required_success_rate": min_success,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated MVP and demo outputs before continuing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mvp_parser = subparsers.add_parser("mvp-results", help="Validate main.py MVP JSON results.")
    mvp_parser.add_argument("--path", required=True)
    mvp_parser.add_argument("--min-success", type=float, default=0.1)
    mvp_parser.add_argument("--require-policy", default="scripted")

    manifest_parser = subparsers.add_parser("manifest", help="Validate collect_demos manifest.json.")
    manifest_parser.add_argument("--path", required=True)
    manifest_parser.add_argument("--min-mean-success", type=float, default=0.5)
    manifest_parser.add_argument("--require-policy", default="scripted")

    csv_parser = subparsers.add_parser("csv-success", help="Validate any CSV with success_rate column.")
    csv_parser.add_argument("--path", required=True)
    csv_parser.add_argument("--min-success", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "mvp-results":
        validate_mvp_results(Path(args.path), args.min_success, args.require_policy)
    elif args.command == "manifest":
        validate_manifest(Path(args.path), args.min_mean_success, args.require_policy)
    elif args.command == "csv-success":
        validate_csv_success(Path(args.path), args.min_success)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
