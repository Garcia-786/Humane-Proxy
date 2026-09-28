"""AutoGen (AG2) integration for HumaneProxy.

Provides convenience functions to register HumaneProxy safety tools
with AutoGen agents.

Quick start::

    from humane_proxy.integrations.autogen import get_safety_functions, register_safety_tools

    # Option A: get raw functions to register manually
    functions = get_safety_functions()

    # Option B: auto-register with agents
    register_safety_tools(assistant, user_proxy)

Requires::

    pip install humane-proxy[autogen]
"""

from __future__ import annotations

import logging
from typing import Any

from humane_proxy._json import dumps_pretty

logger = logging.getLogger("humane_proxy.integrations.autogen")

_proxy = None


def _get_proxy():
    """Return the process-level HumaneProxy singleton (avoids rebuilding
    the whole pipeline — and Stage-2 setup — on every tool call)."""
    global _proxy
    if _proxy is None:
        from humane_proxy import HumaneProxy
        _proxy = HumaneProxy()
    return _proxy


# ---------------------------------------------------------------------------
# Tool functions — these are the raw callables that AutoGen will invoke.
# Type hints and docstrings are critical for AG2 schema generation.
# ---------------------------------------------------------------------------


def check_message_safety(message: str, session_id: str = "autogen-default") -> str:
    """Classify a user message for self-harm or criminal intent.

    Args:
        message: The user message to classify.
        session_id: Session identifier for trajectory tracking.

    Returns:
        JSON string with safety verdict, category, score, and triggers.
    """
    result = _get_proxy().check(message, session_id=session_id)
    return dumps_pretty(result)


def get_session_risk(session_id: str) -> str:
    """Get the current risk trajectory for a session.

    Args:
        session_id: The session identifier to query.

    Returns:
        JSON string with spike detection, trend, and category distribution.
    """
    from humane_proxy.risk.trajectory import snapshot, to_dict

    return dumps_pretty(to_dict(snapshot(session_id)))


def list_recent_escalations(limit: int = 20, category: str = "") -> str:
    """Query recent escalation events from the safety audit log.

    Args:
        limit: Maximum number of events to return.
        category: Filter by category (self_harm or criminal_intent). Empty for all.

    Returns:
        JSON string with list of escalation records.
    """
    from humane_proxy.escalation.query import normalize_escalation_query
    from humane_proxy.storage.factory import get_store

    limit, category = normalize_escalation_query(limit, category)
    store = get_store()
    results = store.query(
        category=category,
        limit=limit,
    )
    return dumps_pretty(results)


def get_safety_functions() -> list[dict[str, Any]]:
    """Return HumaneProxy safety tool definitions for manual registration.

    Returns
    -------
    list[dict]
        Each dict has ``"function"`` (the callable), ``"name"``, and
        ``"description"`` keys suitable for AG2 tool registration.
    """
    return [
        {
            "function": check_message_safety,
            "name": "check_message_safety",
            "description": (
                "Classify a user message for self-harm or criminal intent. "
                "Returns safety verdict, category, score, and triggers."
            ),
        },
        {
            "function": get_session_risk,
            "name": "get_session_risk",
            "description": (
                "Get the current risk trajectory for a session including "
                "spike detection, trend, and category distribution."
            ),
        },
        {
            "function": list_recent_escalations,
            "name": "list_recent_escalations",
            "description": "Query recent escalation events from the safety audit log.",
        },
    ]


def register_safety_tools(assistant: Any, user_proxy: Any) -> None:
    """Register all HumaneProxy safety tools with an AG2 agent pair.

    Parameters
    ----------
    assistant:
        The ``AssistantAgent`` that will use the tools (LLM side).
    user_proxy:
        The ``UserProxyAgent`` that will execute the tools.

    Raises
    ------
    ImportError
        If ``autogen-agentchat`` is not installed.
    """
    try:
        # Verify autogen is available.
        import autogen  # noqa: F401
    except ImportError:
        raise ImportError(
            "AutoGen integration requires autogen-agentchat.\n"
            "Install with: pip install humane-proxy[autogen]"
        )

    for tool in get_safety_functions():
        fn = tool["function"]
        name = tool["name"]
        desc = tool["description"]

        assistant.register_for_llm(name=name, description=desc)(fn)
        user_proxy.register_for_execution(name=name)(fn)

    logger.info("Registered %d safety tools with AutoGen agents", len(get_safety_functions()))
