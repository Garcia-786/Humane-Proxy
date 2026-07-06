"""Tests for the Redis trajectory backend and its dispatch/fallback logic.

The redis package is not required: everything runs against MagicMock
clients, matching the storage backend test approach.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

import humane_proxy.risk.redis_trajectory as redis_traj_mod
import humane_proxy.risk.trajectory as trajectory
from humane_proxy._json import dumps as json_dumps
from humane_proxy.classifiers.models import TrajectoryResult
from humane_proxy.risk.redis_trajectory import _ANALYZE_LUA, RedisTrajectoryBackend


@pytest.fixture(autouse=True)
def _reset_backend():
    trajectory.reset_backend()
    trajectory._cfg_snapshot = None
    yield
    trajectory.reset_backend()
    trajectory._cfg_snapshot = None


def _redis_config(**traj_overrides):
    traj = {
        "backend": "redis",
        "window_size": 5,
        "spike_delta": 0.35,
        "decay_half_life_hours": 24.0,
        "redis": {"url": "", "key_prefix": "humane_proxy:traj:", "ttl_seconds": 0},
    }
    traj.update(traj_overrides)
    return {
        "trajectory": traj,
        "storage": {"redis": {"url": "redis://storage-host:6379/0"}},
    }


def _make_backend(config, mock_client):
    with patch.object(redis_traj_mod, "_REDIS_AVAILABLE", True), \
         patch.object(redis_traj_mod, "_redis", create=True) as mock_redis:
        mock_redis.Redis.from_url.return_value = mock_client
        backend = RedisTrajectoryBackend(config)
    return backend


def _script_response(now, entries):
    return json_dumps({"now": now, "entries": entries})


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------

class TestBackendConstruction:
    def test_requires_redis_package(self):
        with patch.object(redis_traj_mod, "_REDIS_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="redis"):
                RedisTrajectoryBackend(_redis_config())

    def test_empty_url_reuses_storage_redis(self):
        mock_client = MagicMock()
        with patch.object(redis_traj_mod, "_REDIS_AVAILABLE", True), \
             patch.object(redis_traj_mod, "_redis", create=True) as mock_redis:
            mock_redis.Redis.from_url.return_value = mock_client
            RedisTrajectoryBackend(_redis_config())
            url = mock_redis.Redis.from_url.call_args[0][0]
        assert url == "redis://storage-host:6379/0"

    def test_explicit_url_wins_over_storage(self):
        cfg = _redis_config(redis={"url": "redis://traj-host:6379/1"})
        with patch.object(redis_traj_mod, "_REDIS_AVAILABLE", True), \
             patch.object(redis_traj_mod, "_redis", create=True) as mock_redis:
            mock_redis.Redis.from_url.return_value = MagicMock()
            RedisTrajectoryBackend(cfg)
            url = mock_redis.Redis.from_url.call_args[0][0]
        assert url == "redis://traj-host:6379/1"

    def test_lua_script_registered_with_atomic_ops(self):
        mock_client = MagicMock()
        _make_backend(_redis_config(), mock_client)
        assert mock_client.register_script.called
        lua = mock_client.register_script.call_args[0][0]
        for op in ("TIME", "ZRANGE", "ZADD", "ZREMRANGEBYRANK", "EXPIRE", "INCR"):
            assert op in lua
        assert lua == _ANALYZE_LUA


class TestTtlPolicy:
    def test_auto_ttl_is_twice_half_life(self):
        backend = _make_backend(_redis_config(), MagicMock())
        assert backend._ttl == 2 * 24 * 3600

    def test_auto_ttl_min_one_hour(self):
        backend = _make_backend(
            _redis_config(decay_half_life_hours=0.1), MagicMock()
        )
        assert backend._ttl == 3600

    def test_auto_ttl_decay_disabled_is_seven_days(self):
        backend = _make_backend(
            _redis_config(decay_half_life_hours=0), MagicMock()
        )
        assert backend._ttl == 7 * 24 * 3600

    def test_explicit_ttl_wins(self):
        backend = _make_backend(
            _redis_config(redis={"url": "", "ttl_seconds": 42}), MagicMock()
        )
        assert backend._ttl == 42


# ---------------------------------------------------------------------------
# analyze() semantics (mirroring the in-memory backend)
# ---------------------------------------------------------------------------

class TestAnalyze:
    def _backend_with_script(self, response, **traj_overrides):
        mock_client = MagicMock()
        script = MagicMock(return_value=response)
        mock_client.register_script.return_value = script
        backend = _make_backend(_redis_config(**traj_overrides), mock_client)
        return backend, mock_client, script

    def test_first_message_no_spike(self):
        backend, client, script = self._backend_with_script(
            _script_response(1000.0, [])
        )
        result = backend.analyze("s1", 0.9, "self_harm")
        assert result.spike_detected is False
        assert result.window_scores == [0.9]
        assert result.category_counts == {"self_harm": 1}
        assert result.message_count == 1
        assert result.trend == "stable"

    def test_spike_over_stable_baseline(self):
        entries = [[997.0, 0.1, "safe"], [998.0, 0.1, "safe"], [999.0, 0.1, "safe"]]
        backend, client, script = self._backend_with_script(
            _script_response(1000.0, entries)
        )
        result = backend.analyze("s1", 0.9, "self_harm")
        assert result.spike_detected is True
        assert result.window_scores == [0.1, 0.1, 0.1, 0.9]
        assert result.category_counts == {"safe": 3, "self_harm": 1}
        assert result.message_count == 4

    def test_no_spike_below_delta(self):
        entries = [[999.0, 0.3, "safe"]]
        backend, _, _ = self._backend_with_script(_script_response(1000.0, entries))
        result = backend.analyze("s1", 0.5, "safe")
        assert result.spike_detected is False

    def test_script_called_with_keys_and_args(self):
        backend, client, script = self._backend_with_script(
            _script_response(1000.0, [])
        )
        backend.analyze("sess-42", 0.5, "safe")
        kwargs = script.call_args.kwargs
        assert kwargs["keys"] == [
            "humane_proxy:traj:sess-42",
            "humane_proxy:traj:sess-42:seq",
        ]
        assert kwargs["args"][0] == 0.5
        assert kwargs["args"][1] == "safe"
        assert kwargs["args"][2] == 5  # window size

    def test_spike_flag_persisted_for_snapshot(self):
        entries = [[999.0, 0.1, "safe"]]
        backend, client, _ = self._backend_with_script(
            _script_response(1000.0, entries)
        )
        backend.analyze("s1", 0.9, "self_harm")
        args, kwargs = client.set.call_args
        assert args == ("humane_proxy:traj:s1:spike", "1")
        assert kwargs["ex"] == backend._ttl

    def test_trend_escalating_from_returned_window(self):
        entries = [
            [996.0, 0.1, "safe"],
            [997.0, 0.1, "safe"],
            [998.0, 0.5, "safe"],
        ]
        backend, _, _ = self._backend_with_script(_script_response(1000.0, entries))
        result = backend.analyze("s1", 0.6, "safe")
        assert result.trend == "escalating"

    def test_decay_weights_old_baseline(self):
        # Two 3-day-old low scores with a 24h half-life carry ~12.5% weight
        # each; the weighted mean is still 0.1, so a 0.9 spikes.
        old_ts = 1000.0
        now = old_ts + 3 * 24 * 3600
        entries = [[old_ts, 0.1, "safe"], [old_ts + 1, 0.1, "safe"]]
        backend, _, _ = self._backend_with_script(_script_response(now, entries))
        result = backend.analyze("s1", 0.9, "self_harm")
        assert result.spike_detected is True

    def test_cjson_empty_object_entries(self):
        # cjson encodes an empty Lua array as {} -- must not crash.
        backend, _, _ = self._backend_with_script('{"now": 1000.0, "entries": {}}')
        result = backend.analyze("s1", 0.2, "safe")
        assert result.spike_detected is False
        assert result.message_count == 1


# ---------------------------------------------------------------------------
# snapshot() / forget_session()
# ---------------------------------------------------------------------------

class TestSnapshotAndForget:
    def test_snapshot_reads_window_and_spike_flag(self):
        mock_client = MagicMock()
        pipe = MagicMock()
        pipe.execute.return_value = (
            [
                json_dumps({"s": 0.1, "c": "safe", "u": 1}),
                json_dumps({"s": 0.8, "c": "self_harm", "u": 2}),
            ],
            "1",
        )
        mock_client.pipeline.return_value = pipe
        backend = _make_backend(_redis_config(), mock_client)

        result = backend.snapshot("s1")
        assert result.spike_detected is True
        assert result.window_scores == [0.1, 0.8]
        assert result.category_counts == {"safe": 1, "self_harm": 1}
        assert result.message_count == 2

    def test_snapshot_empty_session_is_neutral(self):
        mock_client = MagicMock()
        pipe = MagicMock()
        pipe.execute.return_value = ([], None)
        mock_client.pipeline.return_value = pipe
        backend = _make_backend(_redis_config(), mock_client)

        result = backend.snapshot("nobody")
        assert result.spike_detected is False
        assert result.window_scores == []
        assert result.category_counts == {}
        assert result.message_count == 0
        assert result.trend == "stable"

    def test_forget_session_deletes_all_keys(self):
        mock_client = MagicMock()
        mock_client.delete.return_value = 3
        backend = _make_backend(_redis_config(), mock_client)

        assert backend.forget_session("s1") is True
        mock_client.delete.assert_called_once_with(
            "humane_proxy:traj:s1",
            "humane_proxy:traj:s1:seq",
            "humane_proxy:traj:s1:spike",
        )

    def test_forget_unknown_session_returns_false(self):
        mock_client = MagicMock()
        mock_client.delete.return_value = 0
        backend = _make_backend(_redis_config(), mock_client)
        assert backend.forget_session("ghost") is False


# ---------------------------------------------------------------------------
# Dispatch in risk/trajectory.py
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_memory_is_default(self):
        result = trajectory.analyze("dispatch-mem-default", 0.5, "safe")
        assert isinstance(result, TrajectoryResult)
        assert "dispatch-mem-default" in trajectory.session_history
        assert trajectory._backend == "memory"

    def test_redis_backend_selected_and_used(self, monkeypatch):
        cfg = _redis_config()
        monkeypatch.setattr(trajectory, "get_config", lambda: cfg)

        fake_backend = MagicMock()
        fake_backend.analyze.return_value = TrajectoryResult()
        with patch.object(
            redis_traj_mod, "RedisTrajectoryBackend", return_value=fake_backend
        ):
            trajectory.analyze("dispatch-redis", 0.5, "safe")

        fake_backend.ping.assert_called_once()
        fake_backend.analyze.assert_called_once_with("dispatch-redis", 0.5, "safe")
        assert "dispatch-redis" not in trajectory.session_history

    def test_snapshot_and_forget_dispatch(self, monkeypatch):
        cfg = _redis_config()
        monkeypatch.setattr(trajectory, "get_config", lambda: cfg)

        fake_backend = MagicMock()
        fake_backend.snapshot.return_value = TrajectoryResult()
        fake_backend.forget_session.return_value = True
        with patch.object(
            redis_traj_mod, "RedisTrajectoryBackend", return_value=fake_backend
        ):
            trajectory.snapshot("dispatch-snap")
            assert trajectory.forget_session("dispatch-snap") is True

        fake_backend.snapshot.assert_called_once_with("dispatch-snap")
        fake_backend.forget_session.assert_called_once_with("dispatch-snap")

    def test_fallback_when_redis_lib_missing(self, monkeypatch, caplog):
        cfg = _redis_config()
        monkeypatch.setattr(trajectory, "get_config", lambda: cfg)

        with patch.object(redis_traj_mod, "_REDIS_AVAILABLE", False):
            with caplog.at_level(logging.WARNING, logger="humane_proxy.risk.trajectory"):
                result = trajectory.analyze("fallback-lib", 0.5, "safe")

        assert isinstance(result, TrajectoryResult)
        assert "fallback-lib" in trajectory.session_history
        assert any("falling back to in-memory" in r.getMessage() for r in caplog.records)

    def test_fallback_when_ping_fails_and_no_retry(self, monkeypatch, caplog):
        cfg = _redis_config()
        monkeypatch.setattr(trajectory, "get_config", lambda: cfg)

        fake_backend = MagicMock()
        fake_backend.ping.side_effect = ConnectionError("refused")
        with patch.object(
            redis_traj_mod, "RedisTrajectoryBackend", return_value=fake_backend
        ) as ctor:
            with caplog.at_level(logging.WARNING, logger="humane_proxy.risk.trajectory"):
                trajectory.analyze("fallback-ping", 0.5, "safe")
                trajectory.analyze("fallback-ping", 0.6, "safe")

        # Constructed once; the failure is cached as "memory" for the process.
        assert ctor.call_count == 1
        assert trajectory._backend == "memory"
        assert any("falling back to in-memory" in r.getMessage() for r in caplog.records)

    def test_env_var_selects_backend(self, monkeypatch):
        from humane_proxy.config import reload_config

        monkeypatch.setenv("HUMANE_PROXY_TRAJECTORY_BACKEND", "redis")
        cfg = reload_config()
        assert cfg["trajectory"]["backend"] == "redis"
