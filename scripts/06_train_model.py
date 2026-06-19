from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadrisk.config import project_root_from
from roadrisk.models.train import train_model, validate_modeling_year_range


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--model-name", default="severity_model")
    parser.add_argument("--project-root", type=Path, default=project_root_from())
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    validate_modeling_year_range(project_root, args.start_year, args.end_year)
    metrics = train_model(project_root, model_name=args.model_name)
    print(f"Trained {metrics['model_name']} on chronological split {metrics['split']}.")


if __name__ == "__main__":
    main()
