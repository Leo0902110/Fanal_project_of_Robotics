from __future__ import annotations

import importlib.util
import json
import platform
import sys


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    report = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "modules": {
            "numpy": has_module("numpy"),
            "torch": has_module("torch"),
            "gymnasium": has_module("gymnasium"),
            "mani_skill": has_module("mani_skill"),
            "imageio": has_module("imageio"),
        },
    }
    report["recommendation"] = []
    if sys.version_info >= (3, 13):
        report["recommendation"].append("Local simulator stack is safer on Python 3.12 than 3.13.")
    if not report["modules"]["torch"]:
        report["recommendation"].append("Install torch before running BC training/evaluation.")
    if not report["modules"]["mani_skill"]:
        report["recommendation"].append("ManiSkill missing; mock env fallback will be used for pipeline validation.")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
