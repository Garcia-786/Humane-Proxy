"""RiskTracker — trajectory-based spike detection and trend analysis.

Supports **exponential time-decay** so that stale scores from hours or
days ago naturally fade toward zero, giving returning users a fair
baseline while still catching rapid within-session escalation.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import TYPE_CHECKING

from humane_proxy.config import get_config
from humane_proxy.classifiers.models import TrajectoryResult

if TYPE_CHECKING:
    from humane_proxy.risk.redis_trajectory import RedisTrajectoryBackend

logger = logging.getLogger("humane_proxy.risk.trajectory")

# ---------------------------------------------------------------------------
# Configuration
#
# These globals are (re)built by :func:`_refresh_config` from the **merged**
# configuration (package defaults -> user humane_proxy.yaml -> HUMANE_PROXY_*
# env vars such as HUMANE_PROXY_DECAY_HALF_LIFE).  They refresh automatically
# whenever the cached config object changes — previously this module read
# only the package defaults, once, at import time, silently ignoring all
# documented overrides.
# ---------------------------------------------------------------------------
_cfg_snapshot: dict | None = None

_WINDOW_SIZE: int = 5
_SPIKE_DELTA: float = 0.35

# Decay half-life in hours.  After this many hours a historical score
# contributes only 50 % of its original weight to the rolling baseline.
# Set to 0 or negative to disable decay entirely.
_DECAY_HALF_LIFE_HOURS: float = 24.0

# Precomputed lambda: λ = ln(2) / half_life.
_DECAY_LAMBDA: float = math.log(2) / (_DECAY_HALF_LIFE_HOURS * 3600)


def _refresh_config() -> None:
    """Re-read trajectory settings when the merged config changes.

    Note: ``window_size`` changes only apply to sessions created after the
    change — existing deques keep their original ``maxlen``.
    """
    global _cfg_snapshot, _WINDOW_SIZE, _SPIKE_DELTA
    global _DECAY_HALF_LIFE_HOURS, _DECAY_LAMBDA, _backend

    cfg = get_config().get("trajectory", {})
    if cfg is _cfg_snapshot:
        return
    _cfg_snapshot = cfg
    _backend = None  # re-select the storage backend for the new config

    _WINDOW_SIZE = cfg.get("window_size", 5)
    _SPIKE_DELTA = cfg.get("spike_delta", 0.35)
    _DECAY_HALF_LIFE_HOURS = cfg.get("decay_half_life_hours", 24.0)
    _DECAY_LAMBDA = (
        math.log(2) / (_DECAY_HALF_LIFE_HOURS * 3600)
        if _DECAY_HALF_LIFE_HOURS > 0
        else 0.0
    )

# Maximum distinct sessions to track before eviction (memory-leak prevention).
_MAX_SESSIONS: int = 1000

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
# "memory" (the module-level dicts below, per-process) or a
# RedisTrajectoryBackend instance shared across workers. Resolved lazily on
# first use and re-resolved whenever the merged config changes.
_backend: "RedisTrajectoryBackend | str | None" = None


def _get_backend() -> "RedisTrajectoryBackend | str":
    """Return the active trajectory backend ("memory" or a Redis backend).

    Selecting ``trajectory.backend: redis`` requires the redis package and
    a reachable server; on any failure the process falls back to in-memory
    tracking with a single logged warning (safety checks must not fail
    because a shared-state optimisation is unavailable).
    """
    global _backend
    _refresh_config()
    if _backend is not None:
        return _backend

    cfg = get_config()
    if cfg.get("trajectory", {}).get("backend", "memory") != "redis":
        _backend = "memory"
        return _backend

    try:
        from humane_proxy.risk.redis_trajectory import RedisTrajectoryBackend

        backend = RedisTrajectoryBackend(cfg)
        backend.ping()
        _backend = backend
    except Exception as exc:
        logger.warning(
            "Redis trajectory backend unavailable (%s: %s); "
            "falling back to in-memory tracking for this process",
            type(exc).__name__, exc,
        )
        _backend = "memory"
    return _backend


def reset_backend() -> None:
    """Clear the cached backend so the next call re-reads the config."""
    global _backend
    _backend = None

# ---------------------------------------------------------------------------
# In-memory session stores
# ---------------------------------------------------------------------------
# Each entry is (score, timestamp_seconds).
session_history: dict[str, deque[tuple[float, float]]] = {}
_category_history: dict[str, deque[str]] = {}
_last_spike_by_session: dict[str, bool] = {}


def _evict_oldest_sessions() -> None:
    """Pop roughly 10 % of sessions (FIFO order) when we exceed the cap.

    ``dict`` in CPython 3.7+ preserves insertion order, so popping the
    first *n* keys approximates LRU well enough for v0.1.
    """
    evict_count = max(1, len(session_history) // 10)
    for _ in range(evict_count):
        oldest_key = next(iter(session_history))
        del session_history[oldest_key]
        _category_history.pop(oldest_key, None)
        _last_spike_by_session.pop(oldest_key, None)


# ---------------------------------------------------------------------------
# Decay-weighted mean
# ---------------------------------------------------------------------------

def _weighted_mean(
    history: deque[tuple[float, float]] | list[tuple[float, float]],
    now: float,
    lam: float | None = None,
) -> float:
    """Compute the exponentially time-decayed weighted mean of *history*.

    Each entry ``(score, ts)`` is weighted by ``e^{-λ(now-ts)}``.
    When decay is disabled (λ = 0), this collapses to a plain mean.

    *lam* defaults to the module-level decay constant; the Redis backend
    passes its own so both backends share this single implementation.

    Returns 0.0 when *history* is empty (should never happen in practice
    because callers gate on ``len(history) == 0`` first).
    """
    if lam is None:
        lam = _DECAY_LAMBDA

    if lam == 0.0:
        # Fast path: decay disabled — plain mean.
        return sum(s for s, _ in history) / len(history) if history else 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for score, ts in history:
        dt = now - ts  # seconds elapsed
        w = math.exp(-lam * dt)
        weighted_sum += score * w
        total_weight += w

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def _trend_for_scores(scores: list[float]) -> str:
    """Return the trend label for a list of recent raw scores."""
    if len(scores) < 4:
        return "stable"

    mid = len(scores) // 2
    first_half_avg = sum(scores[:mid]) / mid
    second_half_avg = sum(scores[mid:]) / (len(scores) - mid)
    trend_delta = second_half_avg - first_half_avg
    if trend_delta > 0.15:
        return "escalating"
    if trend_delta < -0.15:
        return "declining"
    return "stable"


def _category_counts(session_id: str) -> dict[str, int]:
    """Return the category distribution for a tracked session."""
    cat_counts: dict[str, int] = {}
    for c in _category_history.get(session_id, []):
        cat_counts[c] = cat_counts.get(c, 0) + 1
    return cat_counts


def detect_spike(session_id: str, current_score: float) -> bool:
    """Return ``True`` if the current score spikes above the recent average.

    Note: this low-level helper always uses the in-memory store, even when
    ``trajectory.backend: redis`` is configured — the pipeline routes
    through :func:`analyze`, which dispatches to the configured backend.

    Math
    ----
    ``delta = current_score - weighted_mean(last N scores)``

    Historical scores are weighted by ``e^{-λ Δt}`` (exponential
    time-decay with a configurable half-life, default 24 h).

    If ``delta > _SPIKE_DELTA`` (default **0.35**), the interaction is
    considered a behavioural spike.

    Parameters
    ----------
    session_id:
        The session / user identifier.
    current_score:
        The heuristic risk score for the current message.

    Returns
    -------
    bool
        Whether a spike was detected.
    """
    _refresh_config()
    now = time.time()

    # --- memory-leak guard ---
    if len(session_history) > _MAX_SESSIONS and session_id not in session_history:
        _evict_oldest_sessions()

    # Initialise the deque on first encounter.
    if session_id not in session_history:
        session_history[session_id] = deque(maxlen=_WINDOW_SIZE)

    history = session_history[session_id]

    # Not enough history to calculate a meaningful delta yet.
    if len(history) == 0:
        history.append((current_score, now))
        return False

    avg = _weighted_mean(history, now)
    delta = current_score - avg

    # Always record *after* computing the delta so the current score
    # doesn't influence the baseline it's compared against.
    history.append((current_score, now))

    return delta > _SPIKE_DELTA


# ---------------------------------------------------------------------------
# Enhanced trajectory analysis (Phase 2)
# ---------------------------------------------------------------------------

def analyze(
    session_id: str,
    score: float,
    category: str = "safe",
) -> TrajectoryResult:
    """Record score + category, run spike detection, and compute trend.

    This is the preferred entry point for the pipeline.  It calls
    :func:`detect_spike` internally, so callers should use **either**
    ``analyze()`` **or** ``detect_spike()`` for a given session — never both.

    Parameters
    ----------
    session_id:
        The session / user identifier.
    score:
        The risk score for the current message.
    category:
        The detected category for the current message.

    Returns
    -------
    TrajectoryResult
        Rich trajectory analysis including spike detection, trend, and
        category distribution.
    """
    backend = _get_backend()
    if backend != "memory":
        return backend.analyze(session_id, score, category)

    # Run spike detection (this also appends the score to session_history
    # and refreshes config-derived settings).
    spike = detect_spike(session_id, score)
    _last_spike_by_session[session_id] = spike

    # Track category history.
    if session_id not in _category_history:
        _category_history[session_id] = deque(maxlen=_WINDOW_SIZE)
    _category_history[session_id].append(category)

    # Get current window (extract raw scores for the public API).
    history = session_history.get(session_id, deque())
    scores = [s for s, _ in history]

    return TrajectoryResult(
        spike_detected=spike,
        trend=_trend_for_scores(scores),
        window_scores=scores,
        category_counts=_category_counts(session_id),
        message_count=len(scores),
    )


def forget_session(session_id: str) -> bool:
    """Erase all in-memory trajectory state for a session.

    Called by the admin ``DELETE /admin/sessions/{id}`` endpoint so the
    right to erasure covers live trajectory data, not just the DB records.

    Returns ``True`` if the session had any tracked state.
    """
    backend = _get_backend()
    if backend != "memory":
        # Clear any in-memory residue too (e.g. state recorded before the
        # backend switched or while Redis was briefly unavailable).
        memory_found = session_id in session_history or session_id in _category_history
        session_history.pop(session_id, None)
        _category_history.pop(session_id, None)
        _last_spike_by_session.pop(session_id, None)
        return backend.forget_session(session_id) or memory_found

    found = session_id in session_history or session_id in _category_history
    session_history.pop(session_id, None)
    _category_history.pop(session_id, None)
    _last_spike_by_session.pop(session_id, None)
    return found


def snapshot(session_id: str) -> TrajectoryResult:
    """Return the current trajectory state without recording a new event."""
    backend = _get_backend()
    if backend != "memory":
        return backend.snapshot(session_id)

    history = session_history.get(session_id, deque())
    scores = [s for s, _ in history]

    return TrajectoryResult(
        spike_detected=_last_spike_by_session.get(session_id, False),
        trend=_trend_for_scores(scores),
        window_scores=scores,
        category_counts=_category_counts(session_id),
        message_count=len(scores),
    )


def to_dict(result: TrajectoryResult) -> dict:
    """Serialize a trajectory result for MCP and agent integrations."""
    return {
        "spike_detected": result.spike_detected,
        "trend": result.trend,
        "window_scores": result.window_scores,
        "category_counts": result.category_counts,
        "message_count": result.message_count,
    }
