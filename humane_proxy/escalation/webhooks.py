"""Enhanced webhooks — Slack, Discord, PagerDuty, Microsoft Teams, Email.

All dispatchers are **fire-and-forget**: they never raise exceptions to the
caller, and they never block the request pipeline.
"""

from __future__ import annotations

import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import urlparse


logger = logging.getLogger("humane_proxy.escalation.webhooks")

# ---------------------------------------------------------------------------
# Notification-injection sanitization
# ---------------------------------------------------------------------------
# session_id is attacker-controlled: it's read straight off the /chat
# request body with no validation (middleware/interceptor.py) and flows
# unmodified into these webhook payloads. Slack/Discord/Teams all render
# markdown in the fields below, so an unsanitized session_id can:
#   - break out of the backtick code span already wrapping it in the
#     Slack/Discord payloads (if it contains a backtick itself)
#   - inject Slack link syntax (<https://evil.example|Click here>) or
#     Discord/Slack mention syntax (<!channel>, <@id>, <#id>) — both use
#     literal <...> — turning a safety alert into a phishing link or a
#     channel-wide mention-spam vector aimed at the incident-response team
#   - render as a clickable masked markdown link ([text](url)) in the
#     Microsoft Teams FactSet "Session" value, which had no escaping at
#     all prior to this fix
#
# This is a lightweight character-substitution mitigation, not a markdown
# parser — it targets the specific structural syntax each platform uses
# to create links/mentions/emphasis/code-breakout, applied to the one
# field (session_id) that's actually attacker-controlled. Trigger text and
# category are server-generated from the classifier's own keyword/pattern
# config, not user input, so they're left as-is.
#
# Backslash-escaping (e.g. "\`") was considered and rejected: Slack's
# mrkdwn dialect doesn't reliably honor backslash escapes for these
# characters the way Discord/CommonMark do, so a backslash-escaped
# backtick can still terminate a Slack code span. Substituting each risky
# character for a visually near-identical fullwidth Unicode form instead
# works the same way on every platform, since the substitute genuinely
# isn't the syntax character — nothing to fail to honor.
_NOTIFICATION_ESCAPE_MAP = str.maketrans({
    "`": "｀",  # U+FF40 FULLWIDTH GRAVE ACCENT
    "*": "＊",  # U+FF0A FULLWIDTH ASTERISK
    "_": "＿",  # U+FF3F FULLWIDTH LOW LINE
    "~": "～",  # U+FF5E FULLWIDTH TILDE
    "|": "｜",  # U+FF5C FULLWIDTH VERTICAL LINE
    "<": "＜",  # U+FF1C FULLWIDTH LESS-THAN SIGN
    ">": "＞",  # U+FF1E FULLWIDTH GREATER-THAN SIGN
    "[": "［",  # U+FF3B FULLWIDTH LEFT SQUARE BRACKET
    "]": "］",  # U+FF3D FULLWIDTH RIGHT SQUARE BRACKET
})


def _sanitize_for_notification(text: str) -> str:
    """Neutralize markdown/mention-triggering characters in *text*.

    Substitutes fullwidth Unicode lookalikes for Slack/Discord `<...>` link
    and mention syntax, Teams/Discord `[text](url)` masked links, backtick
    code-span breakout, and `*_~` emphasis markers — the substitutes render
    visually similarly but have no special meaning to any of the three
    platforms' renderers. Also breaks Discord's literal (bracket-free)
    `@everyone` / `@here` mass-mention keywords with a zero-width space,
    since those aren't covered by the character substitution above.
    """
    if not isinstance(text, str):
        return text
    text = text.translate(_NOTIFICATION_ESCAPE_MAP)
    text = text.replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
    return text


def _category_label(category: str) -> str:
    """Return the bracketed severity tag used across all alert channels."""
    return "[SELF-HARM]" if category == "self_harm" else "[ALERT]"


def _sanitize_url(url: str) -> str:
    """Return only scheme + host of *url* for safe logging.

    Slack/Discord/Teams webhook URLs carry routing tokens in their path,
    so the path, query, fragment, and any userinfo must never be logged
    (issue #33).  The host alone still identifies which integration failed.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme and parsed.hostname:
            return f"{parsed.scheme}://{parsed.hostname}"
    except ValueError:
        pass
    return "<invalid-url>"


async def _post(url: str, payload: dict, *, headers: dict | None = None) -> None:
    """POST JSON to *url*, swallowing all errors."""
    try:
        from humane_proxy.http_client import get_async_client

        client = get_async_client()
        resp = await client.post(
            url, json=payload, headers=headers or {}, timeout=10.0
        )
        if resp.status_code >= 400:
            # Response bodies can echo the request or contain provider
            # details — log only status + length; full body at DEBUG.
            logger.warning(
                "Webhook %s returned HTTP %d (len=%d)",
                _sanitize_url(url), resp.status_code, len(resp.text),
            )
            logger.debug("Webhook error body: %s", resp.text[:500])
    except Exception:
        logger.exception("Webhook dispatch to %s failed", _sanitize_url(url))


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

async def send_slack(
    webhook_url: str,
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
) -> None:
    """Send a Slack Block Kit formatted alert."""
    safe_session_id = _sanitize_for_notification(session_id)
    trigger_text = "\n".join(f"• {t}" for t in triggers) or "(none)"
    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{_category_label(category)} HumaneProxy Alert — {category}",
                    "emoji": False,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Session:*\n`{safe_session_id}`"},
                    {"type": "mrkdwn", "text": f"*Risk Score:*\n`{risk_score:.2f}`"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Triggers:*\n{trigger_text}"},
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Time: {datetime.now(timezone.utc).isoformat()}"},
                ],
            },
        ]
    }
    await _post(webhook_url, payload)


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

async def send_discord(
    webhook_url: str,
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
) -> None:
    """Send a Discord embed formatted alert."""
    safe_session_id = _sanitize_for_notification(session_id)
    color = 15158332 if category == "self_harm" else 16744192
    trigger_text = "\n".join(f"• {t}" for t in triggers) or "(none)"
    payload = {
        "embeds": [
            {
                "title": f"{_category_label(category)} HumaneProxy Alert — {category}",
                "color": color,
                "fields": [
                    {"name": "Session", "value": f"`{safe_session_id}`", "inline": True},
                    {"name": "Risk Score", "value": f"`{risk_score:.2f}`", "inline": True},
                    {"name": "Category", "value": f"`{category}`", "inline": True},
                    {"name": "Triggers", "value": trigger_text, "inline": False},
                ],
                "footer": {"text": datetime.now(timezone.utc).isoformat()},
            }
        ]
    }
    await _post(webhook_url, payload)


# ---------------------------------------------------------------------------
# PagerDuty (Events API v2)
# ---------------------------------------------------------------------------

async def send_pagerduty(
    routing_key: str,
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
) -> None:
    """Send a PagerDuty Events API v2 trigger event."""
    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "payload": {
            "summary": f"HumaneProxy: [{category}] session {session_id} flagged (score={risk_score:.2f})",
            "severity": "critical",
            "source": "humane-proxy",
            "custom_details": {
                "session_id": session_id,
                "category": category,
                "risk_score": risk_score,
                "triggers": triggers,
            },
        },
    }
    await _post(
        "https://events.pagerduty.com/v2/enqueue",
        payload,
        headers={"Content-Type": "application/json"},
    )


# ---------------------------------------------------------------------------
# Microsoft Teams (Adaptive Card via Incoming Webhook)
# ---------------------------------------------------------------------------

async def send_teams(
    webhook_url: str,
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
) -> None:
    """Send a Microsoft Teams adaptive card alert."""
    safe_session_id = _sanitize_for_notification(session_id)
    trigger_text = "\n\n".join(f"• {t}" for t in triggers) or "(none)"
    color = "FF0000" if category == "self_harm" else "FF8C00"
    payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": f"{_category_label(category)} HumaneProxy Alert — {category}",
                            "weight": "Bolder",
                            "size": "Large",
                            "color": "Attention" if category == "self_harm" else "Warning",
                        },
                        {
                            "type": "FactSet",
                            "facts": [
                                {"title": "Session", "value": safe_session_id},
                                {"title": "Risk Score", "value": f"{risk_score:.2f}"},
                                {"title": "Category", "value": category},
                                {"title": "Time", "value": datetime.now(timezone.utc).isoformat()},
                            ],
                        },
                        {
                            "type": "TextBlock",
                            "text": f"**Triggers:**\n\n{trigger_text}",
                            "wrap": True,
                        },
                    ],
                },
            }
        ],
    }
    await _post(webhook_url, payload)


# ---------------------------------------------------------------------------
# Email (via smtplib — stdlib, zero extra deps)
# ---------------------------------------------------------------------------

async def send_email(
    smtp_config: dict,
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
) -> None:
    """Send an email alert using stdlib smtplib (runs in thread pool)."""
    import asyncio

    def _send_sync() -> None:
        host = smtp_config.get("host", "localhost")
        port = smtp_config.get("port", 587)
        user = smtp_config.get("username", "")
        password = smtp_config.get("password", "")
        from_addr = smtp_config.get("from", user)
        to_addrs = smtp_config.get("to", [])

        if not to_addrs:
            return

        category_label = _category_label(category)
        trigger_list = "\n".join(f"  • {t}" for t in triggers) or "  (none)"
        body = (
            f"{category_label} HumaneProxy Safety Alert\n"
            f"{'=' * 50}\n\n"
            f"Category  : {category}\n"
            f"Session   : {session_id}\n"
            f"Risk Score: {risk_score:.2f}\n"
            f"Time      : {datetime.now(timezone.utc).isoformat()}\n\n"
            f"Triggers:\n{trigger_list}\n"
        )

        msg = MIMEMultipart()
        msg["Subject"] = f"[HumaneProxy] {category_label} {category} alert — session {session_id}"
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)
        msg.attach(MIMEText(body, "plain", "utf-8"))

        try:
            with smtplib.SMTP(host, port, timeout=10) as smtp:
                if smtp_config.get("use_tls", True):
                    smtp.starttls()
                if user and password:
                    smtp.login(user, password)
                smtp.sendmail(from_addr, to_addrs, msg.as_string())
        except Exception:
            logger.exception("Email alert failed to send")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _send_sync)
    except Exception:
        logger.exception("Email dispatch setup failed")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

async def dispatch_webhooks(
    config: dict,
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
) -> None:
    """Fire all configured webhooks.  Called from the escalation router."""
    webhooks = config.get("escalation", {}).get("webhooks", {})

    slack_url = webhooks.get("slack_url", "")
    if slack_url:
        await send_slack(slack_url, session_id, risk_score, triggers, category)

    discord_url = webhooks.get("discord_url", "")
    if discord_url:
        await send_discord(discord_url, session_id, risk_score, triggers, category)

    pd_key = webhooks.get("pagerduty_routing_key", "")
    if pd_key:
        await send_pagerduty(pd_key, session_id, risk_score, triggers, category)

    teams_url = webhooks.get("teams_url", "")
    if teams_url:
        await send_teams(teams_url, session_id, risk_score, triggers, category)

    smtp_cfg = webhooks.get("email", {})
    if smtp_cfg and smtp_cfg.get("to"):
        await send_email(smtp_cfg, session_id, risk_score, triggers, category)
