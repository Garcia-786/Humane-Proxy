"""Tests for fail-safe setup: auto stage resolution and startup nudges."""

from __future__ import annotations

from unittest.mock import patch

from humane_proxy.classifiers import pipeline as pl
from humane_proxy.classifiers.pipeline import SafetyPipeline


class TestResolveStages:
    def test_auto_includes_stage2_when_backend_available(self):
        with patch.object(pl, "stage2_backend_available", return_value=True):
            assert SafetyPipeline._resolve_stages("auto") == [1, 2]

    def test_auto_is_stage1_only_without_backend(self):
        with patch.object(pl, "stage2_backend_available", return_value=False):
            assert SafetyPipeline._resolve_stages("auto") == [1]

    def test_auto_never_enables_stage3(self):
        with patch.object(pl, "stage2_backend_available", return_value=True):
            assert 3 not in SafetyPipeline._resolve_stages("auto")

    def test_auto_case_insensitive(self):
        with patch.object(pl, "stage2_backend_available", return_value=True):
            assert SafetyPipeline._resolve_stages("  AUTO ") == [1, 2]

    def test_explicit_list_passthrough(self):
        assert SafetyPipeline._resolve_stages([1, 2, 3]) == [1, 2, 3]

    def test_invalid_falls_back_to_stage1(self):
        assert SafetyPipeline._resolve_stages("garbage") == [1]
        assert SafetyPipeline._resolve_stages(None) == [1]


class TestDiagnose:
    def test_reports_expected_keys(self):
        from humane_proxy.cli import _diagnose

        d = _diagnose()
        for key in (
            "enabled_stages", "stage2_backend_available", "stage3_provider_ready",
            "storage_backend", "alert_channels", "startup_warnings",
        ):
            assert key in d
        assert isinstance(d["enabled_stages"], list)

    def test_stage3_ready_follows_keys(self, monkeypatch):
        from humane_proxy.cli import _diagnose

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert _diagnose()["stage3_provider_ready"] is False

        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        assert _diagnose()["stage3_provider_ready"] is True


class TestSetupWarnings:
    def _run(self, diag):
        from click.testing import CliRunner
        import click
        import humane_proxy.cli as cli

        @click.command()
        def probe():
            cli._print_setup_warnings()

        with patch.object(cli, "_diagnose", return_value=diag):
            return CliRunner().invoke(probe)

    def test_warns_when_stage2_off_and_no_backend(self):
        out = self._run({
            "enabled_stages": [1], "stage2_backend_available": False,
            "stage3_provider_ready": False, "alert_channels": [],
            "startup_warnings": True,
        }).output
        assert "Stage 2" in out and "pip install humane-proxy[onnx]" in out

    def test_silent_when_warnings_disabled(self):
        out = self._run({
            "enabled_stages": [1], "stage2_backend_available": False,
            "stage3_provider_ready": False, "alert_channels": [],
            "startup_warnings": False,
        }).output
        assert out.strip() == ""

    def test_no_stage2_warning_when_already_enabled(self):
        out = self._run({
            "enabled_stages": [1, 2, 3], "stage2_backend_available": True,
            "stage3_provider_ready": True, "alert_channels": ["slack"],
            "startup_warnings": True,
        }).output
        assert "Stage 2" not in out
        assert "Stage 3" not in out
