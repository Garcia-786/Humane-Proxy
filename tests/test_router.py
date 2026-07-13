"""Tests for humane_proxy.escalation.router."""

from unittest.mock import patch

import pytest

from humane_proxy.escalation.router import escalate


@pytest.fixture(autouse=True)
def _reset_alert_backstops():
    """Isolate the in-process IP/global limiter state between tests.

    Both live as module-level counters in router.py, so without this,
    tests that don't pass client_ip share a single "unknown" IP bucket and
    can bleed into each other's quotas depending on run order.
    """
    from humane_proxy.escalation.router import _reset_global_rate_limit, _reset_ip_rate_limit
    _reset_global_rate_limit()
    _reset_ip_rate_limit()
    yield
    _reset_global_rate_limit()
    _reset_ip_rate_limit()


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
    """Regression: rotating session_id (AND client_ip) must not bypass
    alert rate limiting entirely — the global outer ceiling still caps it.

    Previously, check_rate_limit() was keyed only on session_id, which is
    caller-supplied and unauthenticated (interceptor.py reads it straight
    off the request body with no validation). An attacker could send a
    fresh session_id on every request and get unlimited operator alerts.

    ip_rate_limit_max is disabled (0) in these tests to isolate the global
    layer specifically — see TestIpRateLimitBackstop for that layer.
    """

    def _cfg(self, global_max: int, window_s: int = 60):
        return {
            "escalation": {
                "ip_rate_limit_max": 0,
                "global_rate_limit_max": global_max,
                "global_rate_limit_window_seconds": window_s,
                "webhooks": {},
            }
        }

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


class TestIpRateLimitBackstop:
    """The per-IP layer: keyed on client_ip, checked on every request.

    This is the layer that actually neutralizes session_id rotation from a
    single network origin — the global ceiling (above) is the last-resort
    catch-all for a *distributed* attacker with many real IPs.

    global_rate_limit_max is set generously high in these tests to isolate
    the IP layer specifically.
    """

    def _cfg(self, ip_max: int, window_s: int = 60):
        return {
            "escalation": {
                "ip_rate_limit_max": ip_max,
                "ip_rate_limit_window_seconds": window_s,
                "global_rate_limit_max": 10_000,
                "webhooks": {},
            }
        }

    def test_rotating_session_id_same_ip_still_capped(self):
        """Same attacker IP, fresh session_id every request — the per-IP
        layer catches what the per-session limiter alone would miss."""
        from humane_proxy.escalation import router as router_mod

        with patch.object(router_mod, "get_config", return_value=self._cfg(ip_max=4)):
            results = [
                escalate(
                    f"attacker-session-{i}", 0.95, ["t"], "self_harm",
                    client_ip="203.0.113.7",
                )
                for i in range(10)
            ]

        alerted = [r for r in results if r["alerted"]]
        suppressed = [r for r in results if not r["alerted"]]
        assert len(alerted) == 4
        assert len(suppressed) == 6
        assert all(r["reason"] == "logged_alerts_ip_rate_limited" for r in suppressed)
        assert all(r["escalated"] is True for r in results)  # audit trail intact

    def test_different_ips_each_get_their_own_quota(self):
        """Two distinct real IPs each get their own fresh per-IP quota —
        expected/legitimate behavior; the global ceiling is the backstop
        for the distributed case, not this layer."""
        from humane_proxy.escalation import router as router_mod

        with patch.object(router_mod, "get_config", return_value=self._cfg(ip_max=2)):
            results_a = [
                escalate(f"sess-a-{i}", 0.95, ["t"], "self_harm", client_ip="203.0.113.7")
                for i in range(2)
            ]
            results_b = [
                escalate(f"sess-b-{i}", 0.95, ["t"], "self_harm", client_ip="198.51.100.42")
                for i in range(2)
            ]
        assert all(r["alerted"] for r in results_a)
        assert all(r["alerted"] for r in results_b)

    def test_ip_limit_disabled_when_zero(self):
        from humane_proxy.escalation import router as router_mod

        with patch.object(router_mod, "get_config", return_value=self._cfg(ip_max=0)):
            results = [
                escalate(
                    f"disabled-check-{i}", 0.95, ["t"], "self_harm",
                    client_ip="203.0.113.7",
                )
                for i in range(15)
            ]
        assert all(r["alerted"] is True for r in results)

    def test_missing_client_ip_falls_back_to_shared_unknown_bucket(self):
        """Backward compatibility: callers that don't pass client_ip (e.g.
        any external caller of escalate() written before this change)
        still work, sharing one 'unknown' bucket rather than crashing."""
        from humane_proxy.escalation import router as router_mod

        with patch.object(router_mod, "get_config", return_value=self._cfg(ip_max=3)):
            results = [
                escalate(f"no-ip-sess-{i}", 0.95, ["t"], "self_harm")
                for i in range(5)
            ]
        assert sum(1 for r in results if r["alerted"]) == 3

    def test_session_limited_event_does_not_consume_ip_slot(self):
        """Mirrors the equivalent global-layer test: a request already
        blocked by its own session quota shouldn't burn an IP-quota slot."""
        from humane_proxy.escalation import router as router_mod
        from humane_proxy.storage.factory import get_store

        session_cap = get_store()._rate_limit_max
        ip = "203.0.113.7"
        # IP quota must have headroom beyond session_cap, or burning the
        # session's own quota (below) would itself exhaust the IP bucket
        # and confound the thing this test is isolating.
        ip_max = session_cap + 5

        with patch.object(router_mod, "get_config", return_value=self._cfg(ip_max=ip_max)):
            sid = "same-session-repeated-ip-test"
            for _ in range(session_cap):
                escalate(sid, 0.95, ["t"], "self_harm", client_ip=ip)
            for _ in range(10):
                result = escalate(sid, 0.95, ["t"], "self_harm", client_ip=ip)
                assert result["reason"] == "logged_alerts_rate_limited"

            # IP quota should be untouched by the 10 session-limited calls
            # above — a fresh session from the same IP can still alert.
            fresh = escalate("fresh-session-ip-test", 0.95, ["t"], "self_harm", client_ip=ip)
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
