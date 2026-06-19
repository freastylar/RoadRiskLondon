from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Mode(StrEnum):
    MVP = "mvp"
    MODELING = "modeling"
    TRENDS = "trends"


DEFAULT_LATEST_FINAL_YEAR = 2024
MIN_OPEN_DATA_YEAR = 1979


@dataclass(frozen=True)
class YearRange:
    start_year: int
    end_year: int

    def validate(self, latest_final_year: int = DEFAULT_LATEST_FINAL_YEAR) -> YearRange:
        if self.start_year < MIN_OPEN_DATA_YEAR:
            msg = f"start_year must be >= {MIN_OPEN_DATA_YEAR}"
            raise ValueError(msg)
        if self.end_year > latest_final_year:
            msg = f"end_year must be <= latest final year {latest_final_year}"
            raise ValueError(msg)
        if self.start_year > self.end_year:
            raise ValueError("start_year must be <= end_year")
        return self

    @property
    def years(self) -> list[int]:
        return list(range(self.start_year, self.end_year + 1))

    @property
    def label(self) -> str:
        return f"{self.start_year}_{self.end_year}"


@dataclass(frozen=True)
class PipelineConfig:
    mode: Mode
    year_range: YearRange
    project_root: Path
    latest_final_year: int = DEFAULT_LATEST_FINAL_YEAR

    def validate(self) -> PipelineConfig:
        self.year_range.validate(self.latest_final_year)
        if self.mode == Mode.MVP and self.year_range.start_year < 2020:
            raise ValueError("mvp mode should start in 2020 or later")
        if self.mode == Mode.MODELING and self.year_range.start_year < 2015:
            raise ValueError("modeling mode should start in 2015 or later")
        return self


def project_root_from(path: Path | None = None) -> Path:
    if path is not None:
        return path.resolve()
    return Path(__file__).resolve().parents[2]


def build_arg_parser(require_mode: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    if require_mode:
        parser.add_argument("--mode", choices=[m.value for m in Mode], required=True)
    parser.add_argument("--project-root", type=Path, default=project_root_from())
    return parser


def config_from_args(args: argparse.Namespace, default_mode: Mode | None = None) -> PipelineConfig:
    mode_value = getattr(args, "mode", None) or default_mode
    if mode_value is None:
        raise ValueError("mode is required")
    mode = mode_value if isinstance(mode_value, Mode) else Mode(mode_value)
    config = PipelineConfig(
        mode=mode,
        year_range=YearRange(args.start_year, args.end_year),
        project_root=Path(args.project_root).resolve(),
    )
    return config.validate()
