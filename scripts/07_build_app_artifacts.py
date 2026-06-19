from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadrisk.config import Mode, PipelineConfig, YearRange, project_root_from
from roadrisk.data.build_app_tables import build_app_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.MODELING.value)
    parser.add_argument("--include-monitoring", action="store_true")
    parser.add_argument("--project-root", type=Path, default=project_root_from())
    args = parser.parse_args()
    config = PipelineConfig(
        mode=Mode(args.mode),
        year_range=YearRange(args.start_year, args.end_year),
        project_root=Path(args.project_root).resolve(),
    ).validate()
    paths = build_app_artifacts(config, include_monitoring=args.include_monitoring)
    print("App artifacts written:")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
