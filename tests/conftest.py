"""Shared fixtures for HumaneProxy test suite."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the escalation DB to a temp directory for every test."""
    db_path = str(tmp_path / "test_escalations.db")
    monkeypatch.setenv("HUMANE_PROXY_DB_PATH", db_path)

    # Force config reload so the DB path picks up.
    from humane_proxy.config import reload_config
    reload_config()

    # Drop the cached store singleton — otherwise every test after the first
    # keeps using the FIRST test's temp DB and the per-test isolation above
    # is illusory.
    from humane_proxy.storage.factory import reset_store
    reset_store()

    # Re-init DB in temp location.
    from humane_proxy.escalation.local_db import init_db
    init_db()

    yield

    # Don't leak a store bound to this test's (soon-deleted) tmp dir.
    reset_store()


@pytest.fixture()
def tmp_config(tmp_path: Path):
    """Create a temporary config file and return its path."""
    config_path = tmp_path / "humane_proxy.yaml"
    config_path.write_text(
        "safety:\n  risk_threshold: 0.5\n", encoding="utf-8"
    )
    return config_path
