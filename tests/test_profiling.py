"""Tests for the benchmark profiling helpers."""

from __future__ import annotations

from humane_proxy import _profiling


class TestCaptureEnvironment:
    def test_has_core_fields(self):
        env = _profiling.capture_environment()
        for key in ("platform", "processor", "cpu_count", "python", "psutil_available"):
            assert key in env
        assert isinstance(env["cpu_count"], int)


class TestResourceSampler:
    def test_context_manager_runs_and_reports(self):
        import time

        with _profiling.ResourceSampler(interval=0.01) as sampler:
            # Sleep well past the sample interval so at least a few samples
            # are guaranteed regardless of machine speed.
            time.sleep(0.1)
        stats = sampler.stats()
        if _profiling._PSUTIL:
            assert stats is not None
            assert stats["samples"] > 0
            assert stats["peak_rss_mb"] > 0
            assert "cpu_percent_mean" in stats
        else:
            assert stats is None

    def test_no_op_without_psutil(self, monkeypatch):
        monkeypatch.setattr(_profiling, "_PSUTIL", False)
        monkeypatch.setattr(_profiling, "psutil", None)
        with _profiling.ResourceSampler() as sampler:
            pass
        assert sampler.stats() is None
