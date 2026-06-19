from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadrisk.config import build_arg_parser, config_from_args
from roadrisk.data.build_historical_trends import build_historical_trends


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    paths = build_historical_trends(config_from_args(args))
    print("Historical trend artifacts written:")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
