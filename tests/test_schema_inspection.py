from __future__ import annotations

from roadrisk.data.inspect_schema import inspect_sources


def test_schema_reports_created(sample_config):
    manifest = inspect_sources(sample_config)
    assert set(manifest["tables"]) == {"collisions", "vehicles", "casualties"}
    assert (sample_config.project_root / "models/reports/schema_collisions.md").exists()
    assert "accident_index" in manifest["tables"]["collisions"]["linking_candidates"]
