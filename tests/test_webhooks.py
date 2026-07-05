"""Tests for humane_proxy.escalation.webhooks."""

from unittest.mock import AsyncMock, patch

import pytest

from humane_proxy.escalation.webhooks import (
    dispatch_webhooks,
    send_discord,
    send_pagerduty,
    send_slack,
)


@pytest.mark.asyncio
class TestSlack:
    async def test_send_slack_success(self):
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_slack("https://hooks.slack.com/test", "sess-1", 0.9, ["t1"], "self_harm")
            mock.assert_called_once()
            payload = mock.call_args[0][1]
            assert "blocks" in payload

    async def test_send_slack_error_swallowed(self):
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=Exception("network fail"),
        ):
            await send_slack("https://hooks.slack.com/test", "sess-1", 0.9, ["t1"], "self_harm")

    async def test_slack_includes_category(self):
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_slack("https://hooks.slack.com/test", "sess-1", 0.9, ["t1"], "self_harm")
            payload = mock.call_args[0][1]
            header_text = payload["blocks"][0]["text"]["text"]
            assert "self_harm" in header_text


@pytest.mark.asyncio
class TestDiscord:
    async def test_send_discord_success(self):
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_discord("https://discord.com/test", "sess-1", 0.9, ["t1"], "criminal_intent")
            mock.assert_called_once()
            payload = mock.call_args[0][1]
            assert "embeds" in payload


@pytest.mark.asyncio
class TestPagerDuty:
    async def test_send_pagerduty_success(self):
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_pagerduty("routing-key", "sess-1", 0.9, ["t1"], "self_harm")
            mock.assert_called_once()
            payload = mock.call_args[0][1]
            assert payload["routing_key"] == "routing-key"
            assert payload["event_action"] == "trigger"
            assert "self_harm" in payload["payload"]["summary"]


@pytest.mark.asyncio
class TestDispatch:
    async def test_dispatch_no_webhooks_configured(self):
        """No crash when all webhook URLs are empty."""
        config = {"escalation": {"webhooks": {"slack_url": "", "discord_url": "", "pagerduty_routing_key": ""}}}
        await dispatch_webhooks(config, "sess-1", 0.9, ["t1"], "self_harm")

    async def test_dispatch_calls_configured(self):
        config = {"escalation": {"webhooks": {
            "slack_url": "https://hooks.slack.com/test",
            "discord_url": "",
            "pagerduty_routing_key": "",
        }}}
        with patch("humane_proxy.escalation.webhooks.send_slack", new_callable=AsyncMock) as mock:
            await dispatch_webhooks(config, "sess-1", 0.9, ["t1"], "self_harm")
            mock.assert_called_once()


class TestUrlSanitization:
    """Webhook URLs carry routing tokens — logs must only show the host (#33)."""

    def test_sanitize_strips_path_query_fragment(self):
        from humane_proxy.escalation.webhooks import _sanitize_url

        url = "https://hooks.slack.com/services/T0000/B0000/SECRETTOKEN?x=1#frag"
        assert _sanitize_url(url) == "https://hooks.slack.com"

    def test_sanitize_strips_userinfo(self):
        from humane_proxy.escalation.webhooks import _sanitize_url

        assert _sanitize_url("https://user:pass@example.com/hook") == "https://example.com"

    def test_sanitize_handles_garbage(self):
        from humane_proxy.escalation.webhooks import _sanitize_url

        assert _sanitize_url("not a url at all") == "<invalid-url>"
        assert _sanitize_url("") == "<invalid-url>"

    async def test_error_logs_never_contain_token(self, caplog):
        import logging
        from unittest.mock import AsyncMock, MagicMock
        from humane_proxy.escalation.webhooks import _post

        url = "https://hooks.slack.com/services/T0000/B0000/SECRETTOKEN"
        resp = MagicMock()
        resp.status_code = 400
        resp.text = "invalid_payload SECRETBODY"

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            with caplog.at_level(logging.WARNING, logger="humane_proxy.escalation.webhooks"):
                await _post(url, {"k": "v"})

        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "SECRETTOKEN" not in joined
        assert "SECRETBODY" not in joined
        assert "hooks.slack.com" in joined
        assert "400" in joined

    async def test_exception_logs_never_contain_token(self, caplog):
        import logging
        from unittest.mock import AsyncMock
        from humane_proxy.escalation.webhooks import _post

        url = "https://discord.com/api/webhooks/12345/SECRETTOKEN"
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
                   side_effect=Exception("boom")):
            with caplog.at_level(logging.WARNING, logger="humane_proxy.escalation.webhooks"):
                await _post(url, {"k": "v"})

        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "SECRETTOKEN" not in joined
        assert "discord.com" in joined
