# 用途: 自动定位 git_setup.py 创建的虚拟环境 Python，并运行 diffusion_baseline/tests。
# Purpose: Locate the venv Python created by git_setup.py and run diffusion_baseline/tests.

from __future__ import annotations

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


def ensure_pytest(venv_python: Path, repo_root: Path) -> None:
    check = run_command([str(venv_python), "-m", "pytest", "--version"], cwd=repo_root)
    if check.returncode == 0:
        return
    install = run_command(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "pytest",
            "-i",
            "https://pypi.tuna.tsinghua.edu.cn/simple",
        ],
        cwd=repo_root,
    )
    if install.returncode != 0:
        print("PYTEST_INSTALL_FAILED")
        sys.exit(install.returncode)


def main() -> int:
    package_root = Path(__file__).resolve().parent
    repo_root = package_root.parent
    tests_dir = package_root / "tests"
    venv_python = get_venv_python()

    print(f"platform={sys.platform}")
    print(f"venv_python={venv_python}")
    if not venv_python.exists():
        print(f"VENV_PYTHON_NOT_FOUND: {venv_python}")
        print("Expected the virtual environment created by git_setup.py at ~/workspace/.venv")
        return 1

    ensure_pytest(venv_python, repo_root)
    result = run_command(
        [
            str(venv_python),
            "-m",
            "pytest",
            str(tests_dir),
            "-q",
            "-vv",
            "--tb=long",
        ],
        cwd=repo_root,
    )
    if result.returncode == 0:
        print("ALL_TESTS_PASSED")
    else:
        print("TESTS_FAILED")
        print("Pytest output above contains the failed test names and full Python tracebacks.")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
