"""Tests for swappable storage backend implementations."""

import pytest
import sqlite3
from unittest.mock import MagicMock, patch

from humane_proxy.storage.factory import _create_store
from humane_proxy.storage.sqlite import SQLiteStore


def test_sqlite_store(tmp_path):
    """Test the real SQLite backend."""
    db_path = tmp_path / "test.db"
    config = {
        "storage": {
            "backend": "sqlite",
            "sqlite": {"path": str(db_path)}
        }
    }
    
    store = _create_store(config)
    assert isinstance(store, SQLiteStore)
    
    store.init()
    
    # Test rate limiting empty
    assert store.check_rate_limit("user-1") is True
    
    # Write a log
    store.log("user-1", "self_harm", 0.95, ["keyword"])
    
    # Query
    results = store.query(session_id="user-1")
    assert len(results) == 1
    assert results[0]["session_id"] == "user-1"
    assert results[0]["category"] == "self_harm"
    assert results[0]["risk_score"] == 0.95
    
    # Count stats
    assert store.count() == 1
    stats = store.stats()
    assert stats["total_escalations"] == 1
    assert stats["by_category"]["self_harm"] == 1
    
    # Delete
    deleted = store.delete_session("user-1")
    assert deleted == 1
    assert store.count() == 0


@patch("humane_proxy.storage.redis._redis", create=True)
@patch("humane_proxy.storage.redis._REDIS_AVAILABLE", True)
def test_redis_store_creation(mock_redis):
    """Verify Redis store is created correctly when configured."""
    config = {
        "storage": {
            "backend": "redis",
            "redis": {"url": "redis://localhost/0"}
        }
    }
    
    from humane_proxy.storage.redis import RedisStore
    store = _create_store(config)
    assert isinstance(store, RedisStore)
    assert mock_redis.Redis.from_url.called


@patch("humane_proxy.storage.postgres.psycopg", create=True)
@patch("humane_proxy.storage.postgres._PG_AVAILABLE", True)
def test_postgres_store_creation(mock_psycopg):
    """Verify Postgres store is created correctly when configured."""
    config = {
        "storage": {
            "backend": "postgres",
            "postgres": {"dsn": "postgres://user@localhost/db"}
        }
    }
    
    from humane_proxy.storage.postgres import PostgresStore
    store = _create_store(config)
    assert isinstance(store, PostgresStore)


# ---------------------------------------------------------------------------
# Redis rate limiting — regression for the never-incrementing counter
# ---------------------------------------------------------------------------

def _make_redis_store(mock_client):
    """Build a RedisStore around a fully mocked redis client."""
    import humane_proxy.storage.redis as redis_mod

    with patch.object(redis_mod, "_REDIS_AVAILABLE", True), \
         patch.object(redis_mod, "_redis", create=True) as mock_redis:
        mock_redis.Redis.from_url.return_value = mock_client
        store = redis_mod.RedisStore(
            {"storage": {"redis": {"url": "redis://localhost/0"}}},
            rate_limit_max=3,
            rate_limit_window_hours=1,
        )
    return store


def test_redis_rate_limit_increments_and_blocks():
    """The old implementation SETEX'd the counter to 1 and never incremented
    it, so the limit could never trigger.  The atomic Lua script must return
    an increasing count and the store must block once it exceeds the max."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    counter = {"n": 0}

    def fake_script(keys, args):
        counter["n"] += 1
        return counter["n"]

    mock_client.register_script.return_value = fake_script
    store = _make_redis_store(mock_client)

    results = [store.check_rate_limit("sess-rl") for _ in range(5)]
    assert results == [True, True, True, False, False]


def test_redis_rate_limit_registers_atomic_script():
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    _make_redis_store(mock_client)

    assert mock_client.register_script.called
    lua = mock_client.register_script.call_args[0][0]
    assert "INCR" in lua and "EXPIRE" in lua


def test_redis_delete_session_cleans_category_index():
    """delete_session used to leave ids dangling in the category:{cat}
    sorted sets, inflating counts and breaking the right to erasure."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.zrange.return_value = ["7"]
    mock_client.hget.return_value = "self_harm"
    pipe = MagicMock()
    mock_client.pipeline.return_value = pipe

    store = _make_redis_store(mock_client)
    deleted = store.delete_session("sess-del")

    assert deleted == 1
    zrem_keys = [call.args[0] for call in pipe.zrem.call_args_list]
    assert "humane_proxy:category:self_harm" in zrem_keys
    assert "humane_proxy:esc_timeline" in zrem_keys


# ---------------------------------------------------------------------------
# Empty-string filters — regression for SQL/params desync
# ---------------------------------------------------------------------------

def test_sqlite_empty_string_filters_do_not_desync(tmp_path):
    """query(category="") used to build a WHERE clause (is-not-None check)
    while the params builder skipped it (truthiness check), raising
    'Incorrect number of bindings supplied' — e.g. /admin/escalations?category=."""
    from humane_proxy.storage.sqlite import SQLiteStore

    store = SQLiteStore(
        {"storage": {"sqlite": {"path": str(tmp_path / "empty_filter.db")}},
         "escalation": {}}
    )
    store.init()
    store.log("sess-ef", "self_harm", 1.0, ["t"])

    rows = store.query(category="", session_id="")
    assert len(rows) == 1
    assert store.count(category="", session_id="") == 1

    # Real filters still work.
    assert store.count(category="self_harm") == 1
    assert store.count(category="criminal_intent") == 0
