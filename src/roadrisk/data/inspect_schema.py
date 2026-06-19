from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from roadrisk.config import PipelineConfig
from roadrisk.data_sources import TABLES
from roadrisk.utils.paths import raw_dir, reports_dir
from roadrisk.utils.validation import DataValidationError, require_file


@dataclass(frozen=True)
class TableFiles:
    table: str
    paths: list[Path]


def load_source_manifest(root: Path) -> dict:
    path = raw_dir(root) / "source_manifest.json"
    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def table_files_from_manifest(root: Path, table: str) -> TableFiles:
    manifest = load_source_manifest(root)
    files = [
        Path(row["path"])
        for row in manifest.get("files", [])
        if row.get("table") == table and row.get("final") and not row.get("provisional")
    ]
    if not files:
        raise DataValidationError(f"No raw files listed for table {table}")
    for path in files:
        require_file(path)
    return TableFiles(table=table, paths=files)


def read_csv_sample(path: Path, nrows: int | None = None) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path, low_memory=False, nrows=nrows)


def inspect_dataframe(df: pd.DataFrame, table: str) -> dict:
    if df.empty:
        raise DataValidationError(f"Raw {table} data is empty")
    years = []
    year_column = "accident_year" if "accident_year" in df.columns else "collision_year"
    if year_column in df.columns:
        years = sorted(pd.to_numeric(df[year_column], errors="coerce").dropna().astype(int).unique().tolist())
    date_range = None
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        if parsed.notna().any():
            date_range = [str(parsed.min().date()), str(parsed.max().date())]
    severity_distribution = {}
    severity_column = (
        "accident_severity"
        if "accident_severity" in df.columns
        else "collision_severity"
        if "collision_severity" in df.columns
        else "casualty_severity"
    )
    if severity_column in df.columns:
        severity_distribution = df[severity_column].value_counts(dropna=False).head(20).to_dict()
    london_candidates = {}
    for column in ["local_authority_ons_district", "local_authority_district", "local_authority_highway", "police_force"]:
        if column in df.columns:
            london_candidates[column] = df[column].value_counts(dropna=False).head(20).to_dict()
    return {
        "table": table,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missingness": {column: int(df[column].isna().sum()) for column in df.columns},
        "date_range": date_range,
        "available_years": years,
        "severity_distribution": severity_distribution,
        "key_categorical_values": {
            column: df[column].value_counts(dropna=False).head(15).to_dict()
            for column in df.columns
            if df[column].nunique(dropna=False) <= 30
        },
        "london_candidate_counts": london_candidates,
        "linking_candidates": [
            column
            for column in [
                "accident_index",
                "accident_year",
                "accident_reference",
                "collision_index",
                "collision_year",
                "collision_ref_no",
                "vehicle_reference",
            ]
            if column in df.columns
        ],
        "warnings": [],
    }


def _markdown_report(summary: dict) -> str:
    lines = [
        f"# Schema Report: {summary['table'].title()}",
        "",
        f"- Row count: {summary['row_count']}",
        f"- Date range: {summary['date_range']}",
        f"- Available years: {summary['available_years']}",
        f"- Linking candidates: {summary['linking_candidates']}",
        "",
        "## Columns",
        "",
    ]
    lines.extend(f"- `{column}`: {summary['dtypes'][column]}" for column in summary["columns"])
    lines.extend(["", "## Missingness", ""])
    lines.extend(f"- `{column}`: {count}" for column, count in summary["missingness"].items())
    lines.extend(["", "## Severity Distribution", ""])
    if summary["severity_distribution"]:
        lines.extend(f"- `{key}`: {value}" for key, value in summary["severity_distribution"].items())
    else:
        lines.append("- Not available in this table.")
    lines.extend(["", "## London Candidate Counts", ""])
    if summary["london_candidate_counts"]:
        for column, values in summary["london_candidate_counts"].items():
            lines.append(f"### {column}")
            lines.extend(f"- `{key}`: {value}" for key, value in values.items())
    else:
        lines.append("- No direct London candidate column found.")
    lines.extend(["", "## Warnings / Issues", ""])
    lines.extend(f"- {warning}" for warning in summary["warnings"]) or lines.append("- None.")
    return "\n".join(lines) + "\n"


def inspect_table_files(table_files: TableFiles) -> dict:
    frames = [read_csv_sample(path) for path in table_files.paths]
    df = pd.concat(frames, ignore_index=True)
    summary = inspect_dataframe(df, table_files.table)
    summary["source_files"] = [str(path) for path in table_files.paths]
    column_sets = [set(frame.columns) for frame in frames]
    if len(column_sets) > 1 and len({tuple(sorted(columns)) for columns in column_sets}) > 1:
        summary["warnings"].append("Schema differs across selected source files.")
    return summary


def inspect_sources(config: PipelineConfig) -> dict:
    report_dir = reports_dir(config.project_root)
    manifest = {"mode": config.mode.value, "tables": {}}
    for table in TABLES:
        table_files = table_files_from_manifest(config.project_root, table)
        summary = inspect_table_files(table_files)
        manifest["tables"][table] = summary
        report_path = report_dir / f"schema_{table}.md"
        report_path.write_text(_markdown_report(summary), encoding="utf-8")
    manifest_path = report_dir / "schema_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return manifest
