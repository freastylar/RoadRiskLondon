from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadrisk.config import build_arg_parser, config_from_args
from roadrisk.features.road_risk import build_road_risk_table


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    df = build_road_risk_table(config_from_args(args))
    print(
        f"Road-risk table written: {len(df)} rows, {df['road_label'].nunique()} named roads, "
        f"{df['road_area'].nunique()} road areas, modes={sorted(df['mode'].unique().tolist())}, "
        f"low-confidence={(df['confidence'] == 'Low').mean():.0%}."
    )


if __name__ == "__main__":
    main()
