from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadrisk.ui.tables import assert_app_ready


def run(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True)


def run_artifact_check() -> None:
    print("$ artifact readiness check")
    assert_app_ready()
    print("App artifacts are present and non-empty.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Also verify required data/app and models/registry artifacts.",
    )
    args = parser.parse_args()
    run([sys.executable, "-m", "ruff", "check", "."])
    run([sys.executable, "-m", "pytest", "-q"])
    if args.include_artifacts:
        run_artifact_check()


if __name__ == "__main__":
    main()
