from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadrisk.config import build_arg_parser, config_from_args
from roadrisk.data.inspect_schema import inspect_sources


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    manifest = inspect_sources(config_from_args(args))
    print(f"Inspected tables: {', '.join(manifest['tables'])}")


if __name__ == "__main__":
    main()
