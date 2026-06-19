from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from roadrisk.config import PipelineConfig
from roadrisk.data_sources import (
    SourceFile,
    discover_open_data_sources,
    documentation_sources,
    select_sources,
)
from roadrisk.utils.paths import raw_dir
from roadrisk.utils.validation import require_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(source: SourceFile, destination: Path, force: bool = False) -> dict:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return {
            **source.to_dict(),
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "downloaded_at": None,
            "skipped_existing": True,
        }
    request = Request(source.url, headers={"User-Agent": "RoadRiskLondon/0.1"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    require_file(destination)
    return {
        **source.to_dict(),
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "downloaded_at": datetime.now(UTC).isoformat(),
        "skipped_existing": False,
    }


def download_sources(config: PipelineConfig, force: bool = False) -> list[dict]:
    destination_dir = raw_dir(config.project_root)
    discovered = discover_open_data_sources()
    selected = select_sources(config, discovered)
    manifest_rows = []
    for source in selected:
        destination = destination_dir / source.filename
        manifest_rows.append(download_file(source, destination, force=force))
    manifest = {
        "mode": config.mode.value,
        "start_year": config.year_range.start_year,
        "end_year": config.year_range.end_year,
        "latest_final_year": config.latest_final_year,
        "generated_at": datetime.now(UTC).isoformat(),
        "documentation": documentation_sources(),
        "files": manifest_rows,
    }
    manifest_path = destination_dir / "source_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_rows
