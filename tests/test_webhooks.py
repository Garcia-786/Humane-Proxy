"""Tests for humane_proxy.escalation.webhooks."""

from unittest.mock import AsyncMock, patch

import pytest

from humane_proxy.escalation.webhooks import (
    dispatch_webhooks,
    send_discord,
    send_pagerduty,
    send_slack,
    send_teams,
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
        assert "Webhook https://hooks.slack.com returned HTTP 400" in joined

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
        assert "Webhook dispatch to https://discord.com failed" in joined


@pytest.mark.asyncio
class TestNotificationInjection:
    """Regression: session_id is attacker-controlled (unauthenticated,
    read straight off the /chat request body — see
    middleware/interceptor.py) and must not be able to inject markdown
    links, channel-wide mentions, or break out of the backtick code span
    it's wrapped in, in any of the chat-based webhook payloads.
    """

    MALICIOUS_LINK = "[Click here for help](https://evil.example/phish)"
    MENTION_PAYLOAD = "@everyone urgent <!channel> <@12345>"
    BACKTICK_BREAKOUT = "innocent`*BREAKOUT* <!channel>"

    async def test_slack_link_syntax_neutralized(self):
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_slack("https://hooks.slack.com/test", self.MALICIOUS_LINK, 0.9, ["t"], "self_harm")
            payload = mock.call_args[0][1]
            session_field = payload["blocks"][1]["fields"][0]["text"]
        assert "[Click here for help](https://evil.example/phish)" not in session_field
        assert "[" not in session_field and "]" not in session_field
        assert "［" in session_field and "］" in session_field  # fullwidth substitutes present

    async def test_slack_mention_syntax_neutralized(self):
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_slack("https://hooks.slack.com/test", self.MENTION_PAYLOAD, 0.9, ["t"], "self_harm")
            payload = mock.call_args[0][1]
            session_field = payload["blocks"][1]["fields"][0]["text"]
        assert "<!channel>" not in session_field
        assert "<@12345>" not in session_field

    async def test_slack_backtick_breakout_neutralized(self):
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_slack("https://hooks.slack.com/test", self.BACKTICK_BREAKOUT, 0.9, ["t"], "self_harm")
            payload = mock.call_args[0][1]
            session_field = payload["blocks"][1]["fields"][0]["text"]
        # The field must be wrapped by exactly the two literal backticks
        # HumaneProxy itself added — none of the attacker's own backticks
        # may reach the payload as a real backtick character at all
        # (substituted for a fullwidth lookalike instead of escaped, since
        # Slack's mrkdwn doesn't reliably honor backslash escapes here).
        assert session_field.count("`") == 2
        assert "｀" in session_field  # attacker's backtick, defanged

    async def test_discord_mention_and_link_syntax_neutralized(self):
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_discord(
                "https://discord.com/test",
                self.MALICIOUS_LINK + " " + self.MENTION_PAYLOAD,
                0.9, ["t"], "self_harm",
            )
            payload = mock.call_args[0][1]
            session_value = payload["embeds"][0]["fields"][0]["value"]
        assert "[Click here for help](https://evil.example/phish)" not in session_value
        assert "<!channel>" not in session_value
        assert "@everyone" not in session_value  # broken up by zero-width space
        assert "everyone" in session_value  # content preserved, just defanged

    async def test_teams_factset_link_syntax_neutralized(self):
        """Teams had NO escaping at all before this fix — the sharpest case,
        since Adaptive Card FactSet values can render markdown links."""
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_teams("https://outlook.office.com/test", self.MALICIOUS_LINK, 0.9, ["t"], "self_harm")
            payload = mock.call_args[0][1]
            facts = payload["attachments"][0]["content"]["body"][1]["facts"]
            session_fact = next(f for f in facts if f["title"] == "Session")
        assert "[Click here for help](https://evil.example/phish)" not in session_fact["value"]
        assert "[" not in session_fact["value"] and "]" not in session_fact["value"]

    async def test_normal_session_id_unaffected(self):
        """Sanitization must not alter ordinary, non-adversarial values."""
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_slack("https://hooks.slack.com/test", "sess-abc-123", 0.9, ["t"], "self_harm")
            payload = mock.call_args[0][1]
            session_field = payload["blocks"][1]["fields"][0]["text"]
        assert session_field == "*Session:*\n`sess-abc-123`"
