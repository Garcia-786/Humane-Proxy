"""Tests for the fail-open-loudly policy and the hp doctor command."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from humane_proxy.classifiers.pipeline import SafetyPipeline
from humane_proxy.cli import main


class TestFailOpenLoudly:
    def test_stage1_error_fails_open_and_logs(self, caplog):
        pipe = SafetyPipeline({"pipeline": {"enabled_stages": [1]}})
        with patch.object(
            SafetyPipeline, "_run_stage1", side_effect=RuntimeError("boom")
        ):
            with caplog.at_level(logging.ERROR, logger="humane_proxy.pipeline"):
                result = pipe.classify_sync("anything", "s1")

        # Failed open (not blocked) ...
        assert result.classification.category == "safe"
        assert "stage1_error" in result.classification.triggers
        # ... and loudly (exception logged).
        assert any("Stage-1" in r.getMessage() for r in caplog.records)

    def test_stage2_error_fails_open_and_logs(self, caplog):
        pipe = SafetyPipeline({"pipeline": {"enabled_stages": [1, 2]}})

        class Boom:
            def classify(self, text):
                raise RuntimeError("model exploded")

        pipe._stage2 = Boom()
        with caplog.at_level(logging.ERROR, logger="humane_proxy.pipeline"):
            result = pipe.classify_sync("some safe text", "s2")

        assert result.classification.category == "safe"
        assert "stage2_error" in result.classification.triggers
        assert any("Stage-2" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_stage3_error_fails_open_and_logs(self, caplog):
        # Stage 2 enabled with a benign passthrough so a safe message
        # flows past the early exit and reaches Stage 3 (stage3_on_safe).
        from humane_proxy.classifiers.models import ClassificationResult

        pipe = SafetyPipeline({"pipeline": {"enabled_stages": [1]}})
        pipe.enabled_stages = [1, 2, 3]

        class BenignStage2:
            def classify(self, text):
                return ClassificationResult(category="safe", score=0.0, stage=2)

        class Boom:
            async def classify(self, text, prior):
                raise RuntimeError("provider down")

        pipe._stage2 = BenignStage2()
        pipe._stage3 = Boom()
        with caplog.at_level(logging.ERROR, logger="humane_proxy.pipeline"):
            result = await pipe.classify("some text", "s3")

        assert result.classification.category == "safe"
        assert "stage3_error" in result.classification.triggers
        assert any("Stage-3" in r.getMessage() for r in caplog.records)


class TestDoctor:
    def test_doctor_reports_posture(self):
        result = CliRunner().invoke(main, ["doctor"])
        assert result.exit_code == 0
        assert "Protection posture" in result.output
        assert "Stage 1 heuristics" in result.output
        assert "Stage 2 embeddings" in result.output
        assert "Stage 3 reasoning" in result.output
        assert "Alert channels" in result.output
