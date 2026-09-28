"""Tests for the internal JSON shim (orjson fast path + stdlib fallback)."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone

import pytest

from humane_proxy import _json


class TestAvailabilityFlag:
    def test_flag_matches_importability(self):
        installed = importlib.util.find_spec("orjson") is not None
        assert _json.ORJSON_AVAILABLE == installed


@pytest.fixture(params=[True, False], ids=["orjson", "stdlib"])
def shim_path(request, monkeypatch):
    """Run a test against both shim paths (skips orjson if not installed)."""
    if request.param and not _json.ORJSON_AVAILABLE:
        pytest.skip("orjson not installed")
    monkeypatch.setattr(_json, "ORJSON_AVAILABLE", request.param)
    return request.param


class TestRoundtrip:
    SAMPLE = {
        "list": [1, 2.5, "three"],
        "unicode": "café écoute",
        "nested": {"a": None, "b": True},
        "float": 0.35,
    }

    def test_dumps_loads_roundtrip(self, shim_path):
        encoded = _json.dumps(self.SAMPLE)
        assert isinstance(encoded, str)
        assert _json.loads(encoded) == self.SAMPLE

    def test_loads_accepts_bytes(self, shim_path):
        assert _json.loads(b'{"k": [1, 2]}') == {"k": [1, 2]}

    def test_loads_invalid_raises_valueerror(self, shim_path):
        # orjson.JSONDecodeError subclasses json.JSONDecodeError, which
        # subclasses ValueError -- handlers work on both paths.
        with pytest.raises(ValueError):
            _json.loads("not json {")


class TestDumpsPretty:
    def test_indented_and_parseable(self, shim_path):
        out = _json.dumps_pretty({"a": [1, 2], "b": "x"})
        assert "\n" in out
        assert _json.loads(out) == {"a": [1, 2], "b": "x"}

    def test_datetime_serializes(self, shim_path):
        # orjson handles datetimes natively (RFC 3339); stdlib goes through
        # default=str. Either way serialization must succeed and parse back
        # to a string value.
        out = _json.dumps_pretty({"t": datetime(2026, 1, 1, tzinfo=timezone.utc)})
        assert isinstance(_json.loads(out)["t"], str)

    def test_non_json_types_coerced_via_str(self, shim_path):
        class Odd:
            def __str__(self) -> str:
                return "odd-repr"

        out = _json.dumps_pretty({"o": Odd()})
        assert _json.loads(out)["o"] == "odd-repr"


class TestInterceptorResponseClass:
    def test_orjson_response_renders_bytes(self):
        if not _json.ORJSON_AVAILABLE:
            pytest.skip("orjson not installed")
        from fastapi.responses import JSONResponse
        from humane_proxy.middleware.interceptor import _select_response_class

        cls = _select_response_class()
        assert cls is not JSONResponse
        assert issubclass(cls, JSONResponse)
        resp = cls(content={"status": "ok", "score": 0.35})
        assert _json.loads(resp.body) == {"status": "ok", "score": 0.35}

    def test_fallback_selects_json_response(self, monkeypatch):
        from fastapi.responses import JSONResponse
        from humane_proxy.middleware.interceptor import _select_response_class

        monkeypatch.setattr(_json, "ORJSON_AVAILABLE", False)
        assert _select_response_class() is JSONResponse
