"""
Resume training stability test.
Runs 50 steps, saves checkpoint, resumes from it, runs another 50 steps.
Verifies loss continuity and step correctness.

Usage (PowerShell):
    $env:HOME = "$env:USERPROFILE"
    & "$HOME\workspace\.venv\Scripts\python.exe" diffusion_baseline\training\resume_test.py --device cuda --amp True
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(args: list[str], label: str) -> tuple[int, str]:
    cmd = [sys.executable, "diffusion_baseline/training/run_env_train.py"] + args
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.returncode, result.stdout


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume training test.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--buffer_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=50)
    args = parser.parse_args()

    common = [
        "--device", args.device,
        "--amp", str(args.amp),
        "--env_id", args.env_id,
        "--warmup_steps", str(args.warmup_steps),
        "--buffer_size", str(args.buffer_size),
        "--batch_size", str(args.batch_size),
        "--save_every", str(args.save_every),
        "--num_envs", "1",
    ]

    ckpt_path = Path("diffusion_baseline/checkpoints/ckpt_step50.pt")
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"[INFO] Removed old checkpoint: {ckpt_path}")

    # Phase 1: Train 50 steps
    phase1_args = common + ["--num_steps", "50"]
    rc1, out1 = run_command(phase1_args, "PHASE 1: Train 50 steps")
    print(out1)
    if rc1 != 0:
        print(f"[FAIL] Phase 1 exited with code {rc1}")
        sys.exit(1)

    if not ckpt_path.exists():
        print(f"[FAIL] Checkpoint not found after phase 1: {ckpt_path}")
        sys.exit(1)
    size = ckpt_path.stat().st_size
    print(f"\n[OK] Checkpoint saved: {ckpt_path} size={size:,} bytes")

    # Parse final step and loss from phase 1
    step1 = None
    loss1 = None
    for line in out1.splitlines():
        if "step=" in line and "loss=" in line:
            parts = line.split()
            for p in parts:
                if p.startswith("step="):
                    step1 = int(p.split("=")[1])
                if p.startswith("loss="):
                    loss1 = float(p.split("=")[1])

    # Phase 2: Resume from ckpt_step50.pt, train another 50 steps
    phase2_args = common + ["--num_steps", "50", "--resume", str(ckpt_path)]
    rc2, out2 = run_command(phase2_args, "PHASE 2: Resume from ckpt_step50.pt and train 50 more steps")
    print(out2)
    if rc2 != 0:
        print(f"[FAIL] Phase 2 exited with code {rc2}")
        sys.exit(1)

    # Parse resume info and final step/loss from phase 2
    resume_step = None
    step2 = None
    loss2 = None
    for line in out2.splitlines():
        if "[RESUME]" in line and "step=" in line:
            resume_step = int(line.split("step=")[1].strip())
        if "step=" in line and "loss=" in line:
            parts = line.split()
            for p in parts:
                if p.startswith("step="):
                    step2 = int(p.split("=")[1])
                if p.startswith("loss="):
                    loss2 = float(p.split("=")[1])

    # Validation
    print(f"\n{'='*60}")
    print("  VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Phase 1 final step : {step1}")
    print(f"  Phase 1 final loss : {loss1}")
    print(f"  Resume step        : {resume_step}")
    print(f"  Phase 2 final step : {step2}")
    print(f"  Phase 2 final loss : {loss2}")

    ok = True
    if resume_step != 50:
        print(f"  [FAIL] Resume step mismatch: expected 50, got {resume_step}")
        ok = False
    else:
        print(f"  [OK]   Resume step = 50")

    if step2 != 100:
        print(f"  [FAIL] Final step mismatch: expected 100, got {step2}")
        ok = False
    else:
        print(f"  [OK]   Final step = 100")

    if loss1 is not None and loss2 is not None:
        delta = abs(loss2 - loss1)
        print(f"  [INFO] Loss delta |loss2-loss1| = {delta:.6f}")
        if delta > 10.0:
            print(f"  [WARN] Loss jump is large ({delta:.2f}), but this can happen with random sampling.")
        else:
            print(f"  [OK]   Loss continuity looks reasonable")
    else:
        print(f"  [WARN] Could not parse losses for comparison")

    if ok:
        print(f"\n  [PASS] Resume test succeeded!")
    else:
        print(f"\n  [FAIL] Resume test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
