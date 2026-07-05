"""Security-focused MCP helper tests."""

import pytest

from humane_proxy.escalation.query import normalize_escalation_query
from humane_proxy.mcp_server import (
    MCP_DEFAULT_HOST,
    MCP_TOKEN_ENV,
    _get_mcp_auth_provider,
    _is_public_bind_host,
    serve_http,
)


def test_http_mcp_defaults_to_localhost():
    assert MCP_DEFAULT_HOST == "127.0.0.1"
    assert serve_http.__defaults__[0] == "127.0.0.1"


def test_mcp_auth_provider_uses_configured_bearer_token(monkeypatch):
    """Must build a provider from the REAL fastmcp package.

    The previous version of this test injected a fake
    ``fastmcp.server.auth.BearerTokenAuth`` module into sys.modules — a
    class no fastmcp release actually exports — so the suite passed while
    the documented HTTP-auth flow crashed at import time with real fastmcp.
    """
    pytest.importorskip("fastmcp", reason="requires the [mcp] extra")

    monkeypatch.setenv(MCP_TOKEN_ENV, "test-mcp-secret")

    auth = _get_mcp_auth_provider()

    from fastmcp.server.auth import AuthProvider
    from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

    assert isinstance(auth, StaticTokenVerifier)
    assert isinstance(auth, AuthProvider)
    assert "test-mcp-secret" in auth.tokens


def test_mcp_app_accepts_auth_provider(monkeypatch):
    """The provider must be accepted by the real FastMCP constructor."""
    pytest.importorskip("fastmcp", reason="requires the [mcp] extra")

    monkeypatch.setenv(MCP_TOKEN_ENV, "test-mcp-secret")

    from fastmcp import FastMCP

    app = FastMCP("humane-proxy-auth-probe", auth=_get_mcp_auth_provider())
    assert app.name == "humane-proxy-auth-probe"


def test_mcp_auth_provider_is_optional(monkeypatch):
    monkeypatch.delenv(MCP_TOKEN_ENV, raising=False)
    assert _get_mcp_auth_provider() is None


@pytest.mark.parametrize(
    ("raw_limit", "expected"),
    [
        (0, 1),
        (-50, 1),
        (25, 25),
        (500, 100),
        ("not-a-number", 20),
    ],
)
def test_escalation_query_limit_is_clamped(raw_limit, expected):
    limit, category = normalize_escalation_query(raw_limit, "self_harm")

    assert limit == expected
    assert category == "self_harm"


def test_escalation_query_rejects_unknown_categories():
    with pytest.raises(ValueError, match="category must be one of"):
        normalize_escalation_query(20, "all_data")


def test_escalation_query_treats_whitespace_category_as_unfiltered():
    limit, category = normalize_escalation_query(20, "   ")

    assert limit == 20
    assert category is None


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", False),
        ("::1", False),
        ("localhost", False),
        ("0.0.0.0", True),
        ("::", True),
        ("[::]", True),
        ("192.168.1.10", True),
        ("10.0.0.5", True),
        ("mcp.example.com", True),
        ("", True),
    ],
)
def test_public_bind_detection_flags_non_loopback_hosts(host, expected):
    assert _is_public_bind_host(host) is expected
