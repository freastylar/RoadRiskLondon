from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadrisk.config import build_arg_parser, config_from_args
from roadrisk.features.build_collision_features import build_modeling_table


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    df = build_modeling_table(config_from_args(args))
    print(f"Modeling table written with {len(df)} rows.")


if __name__ == "__main__":
    main()
