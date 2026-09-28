"""Internal JSON helpers with an optional orjson fast path.

orjson (Rust-backed, installed via the ``perf`` extra) is used when
available; otherwise these helpers fall back to the stdlib ``json``
module with identical semantics. Callers never need to know which
backend is active.

Error compatibility: ``orjson.JSONDecodeError`` subclasses both
``json.JSONDecodeError`` and ``ValueError``, so existing
``except json.JSONDecodeError`` handlers work on either path.
"""

from __future__ import annotations

import json as _stdlib_json
from typing import Any

try:
    import orjson as _orjson

    ORJSON_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _orjson = None
    ORJSON_AVAILABLE = False


def dumps(obj: Any) -> str:
    """Serialize *obj* to a compact JSON string."""
    if ORJSON_AVAILABLE:
        return _orjson.dumps(obj).decode("utf-8")
    return _stdlib_json.dumps(obj)


def dumps_pretty(obj: Any) -> str:
    """Serialize *obj* with 2-space indentation.

    Non-JSON-native types (datetimes, dataclass reprs, ...) are coerced
    with ``str()`` — used for human-readable tool output in the agent
    integrations.
    """
    if ORJSON_AVAILABLE:
        return _orjson.dumps(
            obj, option=_orjson.OPT_INDENT_2, default=str
        ).decode("utf-8")
    return _stdlib_json.dumps(obj, indent=2, default=str)


def loads(data: str | bytes) -> Any:
    """Deserialize a JSON string or bytes payload."""
    if ORJSON_AVAILABLE:
        return _orjson.loads(data)
    return _stdlib_json.loads(data)
