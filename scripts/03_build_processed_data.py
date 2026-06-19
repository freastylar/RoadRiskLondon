from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadrisk.config import build_arg_parser, config_from_args
from roadrisk.data.build_processed_data import build_processed_data


def main() -> None:
    parser = build_arg_parser()
    parser.add_argument("--allow-coordinate-fallback", action="store_true")
    args = parser.parse_args()
    paths = build_processed_data(
        config_from_args(args), allow_coordinate_fallback=args.allow_coordinate_fallback
    )
    print("Processed data written:")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
