from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(root: Path, name: str) -> Path:
    return ensure_dir(root / "data" / name)


def raw_dir(root: Path) -> Path:
    return data_dir(root, "raw")


def interim_dir(root: Path) -> Path:
    return data_dir(root, "interim")


def processed_dir(root: Path) -> Path:
    return data_dir(root, "processed")


def app_data_dir(root: Path) -> Path:
    return data_dir(root, "app")


def reports_dir(root: Path) -> Path:
    return ensure_dir(root / "models" / "reports")


def registry_dir(root: Path) -> Path:
    return ensure_dir(root / "models" / "registry")
