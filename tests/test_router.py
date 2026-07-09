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
    def test_rate_limit_suppresses_alerts_after_max(self):
        sid = "ratelimit-sess"
        for _ in range(3):
            result = escalate(sid, 0.9, ["t"], "self_harm")
            assert result["alerted"] is True
        result = escalate(sid, 0.9, ["t"], "self_harm")
        # The event is still recorded — only alerting is rate limited.
        assert result["escalated"] is True
        assert result["alerted"] is False
        assert result["reason"] == "logged_alerts_rate_limited"

    def test_rate_limited_events_still_reach_audit_log(self):
        from humane_proxy.storage.factory import get_store

        sid = "ratelimit-audit-sess"
        for _ in range(5):
            escalate(sid, 0.9, ["t"], "self_harm")
        # All 5 events must be persisted even though only 3 alerted.
        assert get_store().count(session_id=sid) == 5

    def test_no_webhooks_fired_when_rate_limited(self):
        sid = "ratelimit-webhook-sess"
        for _ in range(3):
            escalate(sid, 0.9, ["t"], "self_harm")

        with patch("humane_proxy.escalation.router._fire_webhooks") as mock_fire:
            escalate(sid, 0.9, ["t"], "self_harm")
            mock_fire.assert_not_called()


class TestGlobalRateLimitBackstop:
    """Regression: rotating session_id must NOT bypass alert rate limiting.

    Previously, check_rate_limit() was keyed only on session_id, which is
    caller-supplied and unauthenticated (interceptor.py reads it straight
    off the request body with no validation). An attacker could send a
    fresh session_id on every request and get unlimited operator alerts.
    """

    def _cfg(self, global_max: int, window_s: int = 60):
        return {
            "escalation": {
                "global_rate_limit_max": global_max,
                "global_rate_limit_window_seconds": window_s,
                "webhooks": {},
            }
        }

    def setup_method(self):
        from humane_proxy.escalation.router import _reset_global_rate_limit
        _reset_global_rate_limit()

    def teardown_method(self):
        from humane_proxy.escalation.router import _reset_global_rate_limit
        _reset_global_rate_limit()

    def test_rotating_session_id_no_longer_bypasses_rate_limit(self):
        from humane_proxy.escalation import router as router_mod

        with patch.object(router_mod, "get_config", return_value=self._cfg(global_max=5)):
            results = [
                escalate(f"attacker-session-{i}", 0.95, ["t"], "self_harm")
                for i in range(20)
            ]

        alerted = [r for r in results if r["alerted"]]
        suppressed = [r for r in results if not r["alerted"]]
        # Every request used a brand-new session_id (own fresh per-session
        # quota each time), yet the global ceiling still caps total alerts.
        assert len(alerted) == 5
        assert len(suppressed) == 15
        assert all(r["reason"] == "logged_alerts_globally_rate_limited" for r in suppressed)
        # Audit trail is still complete for every event, same guarantee as
        # the existing per-session limiter.
        assert all(r["escalated"] is True for r in results)

    def test_global_limit_disabled_when_zero(self):
        from humane_proxy.escalation import router as router_mod

        with patch.object(router_mod, "get_config", return_value=self._cfg(global_max=0)):
            results = [
                escalate(f"disabled-check-session-{i}", 0.95, ["t"], "self_harm")
                for i in range(10)
            ]
        assert all(r["alerted"] is True for r in results)

    def test_global_window_expires_and_allows_more(self):
        from humane_proxy.escalation import router as router_mod

        with patch.object(router_mod, "get_config", return_value=self._cfg(global_max=2, window_s=60)):
            results = [
                escalate(f"window-session-{i}", 0.95, ["t"], "self_harm")
                for i in range(4)
            ]
        assert sum(1 for r in results if r["alerted"]) == 2

        # Simulate the window elapsing by backdating recorded timestamps.
        with router_mod._global_alert_lock:
            router_mod._global_alert_timestamps.clear()

        with patch.object(router_mod, "get_config", return_value=self._cfg(global_max=2, window_s=60)):
            result = escalate("window-session-after-reset", 0.95, ["t"], "self_harm")
        assert result["alerted"] is True

    def test_session_limited_event_does_not_consume_global_slot(self):
        """A request already blocked by its own session quota shouldn't
        also burn a global-quota slot — it was never going to alert."""
        from humane_proxy.escalation import router as router_mod
        from humane_proxy.storage.factory import get_store

        # The per-session cap lives on the already-instantiated storage
        # singleton (set once from real config at process start) — read
        # the live value rather than assuming one.
        session_cap = get_store()._rate_limit_max

        with patch.object(router_mod, "get_config", return_value=self._cfg(global_max=5)):
            sid = "same-session-repeated"
            for _ in range(session_cap):
                escalate(sid, 0.95, ["t"], "self_harm")  # burn its own session quota
            for _ in range(10):
                result = escalate(sid, 0.95, ["t"], "self_harm")
                assert result["reason"] == "logged_alerts_rate_limited"

            # Global quota (5) should be untouched by the 10 session-limited
            # calls above — a fresh session can still alert.
            fresh = escalate("fresh-session", 0.95, ["t"], "self_harm")
            assert fresh["alerted"] is True


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
