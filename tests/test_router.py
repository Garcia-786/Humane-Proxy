"""Tests for humane_proxy.escalation.router."""

from unittest.mock import patch

from humane_proxy.escalation.router import escalate


class TestEscalation:
    def test_basic_escalation(self):
        result = escalate("esc-sess", 0.95, ["trigger1"], "self_harm")
        assert result["escalated"] is True
        assert result["reason"] == "logged"
        assert result["risk_score"] == 0.95
        assert result["triggers"] == ["trigger1"]
        assert result["category"] == "self_harm"

    def test_category_in_result(self):
        result = escalate("cat-sess", 0.9, ["t"], "criminal_intent")
        assert result["category"] == "criminal_intent"

    def test_defensive_triggers_copy(self):
        original = ["mutable"]
        result = escalate("copy-sess", 0.9, original, "self_harm")
        original.append("sneaky")
        assert "sneaky" not in result["triggers"]

    def test_none_triggers_handled(self):
        result = escalate("none-sess", 0.9, None, "self_harm")  # type: ignore[arg-type]
        assert result["escalated"] is True


class TestRateLimiting:
    def test_rate_limit_blocks_after_max(self):
        sid = "ratelimit-sess"
        for _ in range(3):
            escalate(sid, 0.9, ["t"], "self_harm")
        result = escalate(sid, 0.9, ["t"], "self_harm")
        assert result["escalated"] is False
        assert result["reason"] == "rate_limited"


class TestDbFailure:
    def test_db_failure_graceful(self):
        with patch(
            "humane_proxy.escalation.router.log_escalation",
            side_effect=Exception("DB boom"),
        ):
            result = escalate("fail-sess", 0.99, ["boom"], "self_harm")
            assert result["escalated"] is True
            assert result["reason"] == "logged_with_db_error"


class TestWebhookGate:
    """Regression: _fire_webhooks must fire when ONLY email is configured.

    The gate used to check a non-existent ``webhooks.email_to`` key while
    the dispatcher reads ``webhooks.email.to`` — email-only configs never
    dispatched anything.
    """

    def _cfg(self, webhooks: dict) -> dict:
        return {
            "escalation": {
                "rate_limit_max": 100,
                "rate_limit_window_hours": 1,
                "webhooks": webhooks,
            }
        }

    def _fire(self, cfg: dict) -> list:
        from humane_proxy.escalation import router as router_mod
        from humane_proxy.escalation import webhooks as webhooks_mod

        calls: list = []

        async def fake_dispatch(*args, **kwargs):
            calls.append(args)

        with patch.object(router_mod, "get_config", return_value=cfg), \
             patch.object(webhooks_mod, "dispatch_webhooks", fake_dispatch):
            router_mod._fire_webhooks("gate-sess", 0.9, ["t"], "self_harm")
        return calls

    def test_email_only_config_dispatches(self):
        calls = self._fire(self._cfg({
            "slack_url": "",
            "discord_url": "",
            "pagerduty_routing_key": "",
            "teams_url": "",
            "email": {"host": "smtp.example.com", "to": ["ops@example.com"]},
        }))
        assert calls, "email-only webhook config should trigger dispatch"

    def test_no_webhooks_configured_skips_dispatch(self):
        calls = self._fire(self._cfg({
            "slack_url": "",
            "discord_url": "",
            "pagerduty_routing_key": "",
            "teams_url": "",
            "email": {"host": "", "to": []},
        }))
        assert not calls

    def test_email_none_block_handled(self):
        # `email:` left empty in YAML parses to None — must not crash the gate.
        calls = self._fire(self._cfg({
            "slack_url": "",
            "email": None,
        }))
        assert not calls
