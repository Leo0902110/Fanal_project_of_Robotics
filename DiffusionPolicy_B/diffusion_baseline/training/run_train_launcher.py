# 用途: 使用 git_setup.py 创建的虚拟环境 Python 启动 dummy train 和 eval。
# Purpose: Launch dummy training and evaluation with the venv Python created by git_setup.py.

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_venv_python() -> Path:
    if os.name == "nt" or sys.platform.startswith("win"):
        return Path(os.path.expanduser("~/workspace/.venv/Scripts/python.exe"))
    return Path(os.path.expanduser("~/workspace/.venv/bin/python"))


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    print("\n$ " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch dummy train/eval using ~/workspace/.venv.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=Path("diffusion_baseline/checkpoints/dummy_ckpt.pt"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    venv_python = get_venv_python()

    print(f"platform={sys.platform}")
    print(f"venv_python={venv_python}")
    print("Windows PowerShell:")
    print(
        rf'  & "$HOME\workspace\.venv\Scripts\python.exe" '
        rf'diffusion_baseline\training\run_dummy_train.py --batch_size {args.batch_size} --num_steps {args.num_steps}'
    )
    print("Linux:")
    print(
        f"  ~/workspace/.venv/bin/python "
        f"diffusion_baseline/training/run_dummy_train.py --batch_size {args.batch_size} --num_steps {args.num_steps}"
    )

    if not venv_python.exists():
        print(f"VENV_PYTHON_NOT_FOUND: {venv_python}")
        return 1

    train_cmd = [
        str(venv_python),
        "diffusion_baseline/training/run_dummy_train.py",
        "--batch_size",
        str(args.batch_size),
        "--num_steps",
        str(args.num_steps),
        "--device",
        args.device,
        "--checkpoint",
        str(args.checkpoint),
    ]
    train_result = run_command(train_cmd, cwd=repo_root)
    if train_result.returncode != 0:
        print("DUMMY_TRAIN_LAUNCH_FAILED")
        return train_result.returncode

    eval_cmd = [
        str(venv_python),
        "diffusion_baseline/training/eval_dummy.py",
        "--device",
        args.device,
        "--checkpoint",
        str(args.checkpoint),
    ]
    eval_result = run_command(eval_cmd, cwd=repo_root)
    if eval_result.returncode != 0:
        print("DUMMY_EVAL_LAUNCH_FAILED")
        return eval_result.returncode

    print(f"checkpoint_path={args.checkpoint}")
    print("DUMMY_TRAIN_AND_EVAL_LAUNCHER_SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
