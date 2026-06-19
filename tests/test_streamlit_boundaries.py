from __future__ import annotations

import ast
from pathlib import Path

STREAMLIT_ENTRYPOINTS = [Path("app.py"), *sorted(Path("pages").glob("*.py"))]
FORBIDDEN_IMPORT_PREFIXES = (
    "roadrisk.data.build_",
    "roadrisk.data.clean_",
    "roadrisk.data.download",
    "roadrisk.data.inspect_schema",
    "roadrisk.features.build_",
    "roadrisk.models.train",
    "roadrisk.models.predict",
)
FORBIDDEN_CALL_NAMES = {
    "build_app_artifacts",
    "build_historical_trends",
    "build_modeling_table",
    "build_processed_data",
    "download_sources",
    "inspect_sources",
    "predict_probabilities",
    "read_csv",
    "to_csv",
    "to_parquet",
    "train_model",
    "urlopen",
}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_streamlit_entrypoints_do_not_import_pipeline_or_training_modules():
    violations = []
    for path in STREAMLIT_ENTRYPOINTS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    violations.append(f"{path}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                        violations.append(f"{path}: import {alias.name}")
    assert not violations


def test_streamlit_entrypoints_do_not_run_pipeline_or_file_write_calls():
    violations = []
    for path in STREAMLIT_ENTRYPOINTS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node)
                if name in FORBIDDEN_CALL_NAMES:
                    violations.append(f"{path}: calls {name}()")
    assert not violations
