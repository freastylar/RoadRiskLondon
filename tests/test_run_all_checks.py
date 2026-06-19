from __future__ import annotations

import importlib

checks = importlib.import_module("scripts.08_run_all_checks")


def test_run_artifact_check_uses_app_ready_gate(monkeypatch):
    called = {"value": False}

    def fake_assert_app_ready() -> None:
        called["value"] = True

    monkeypatch.setattr(checks, "assert_app_ready", fake_assert_app_ready)
    checks.run_artifact_check()
    assert called["value"]
