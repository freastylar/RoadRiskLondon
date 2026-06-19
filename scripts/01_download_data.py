from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadrisk.config import build_arg_parser, config_from_args
from roadrisk.data.download import download_sources


def main() -> None:
    parser = build_arg_parser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = download_sources(config_from_args(args), force=args.force)
    print(f"Downloaded or verified {len(rows)} source files.")


if __name__ == "__main__":
    main()
