"""Redis-backed trajectory state for multi-worker deployments.

The in-memory trajectory store in :mod:`humane_proxy.risk.trajectory` is
per-process: with ``uvicorn --workers N`` each worker sees only its own
slice of a session's history. This backend keeps the rolling window in a
Redis sorted set (scored by server timestamp) so every worker shares one
consistent view.

Atomicity: a single Lua script reads the pre-append window, appends the
new entry, and trims to the window size in one server-side step, so the
spike baseline can never interleave with another worker's append. The
decay / spike / trend math itself runs in Python on the returned window,
reusing the same functions as the in-memory backend.

Requires: pip install humane-proxy[redis]
"""

from __future__ import annotations

import logging
import math

from humane_proxy._json import loads as _json_loads
from humane_proxy.classifiers.models import TrajectoryResult

logger = logging.getLogger("humane_proxy.risk.redis_trajectory")

try:
    import redis as _redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    _redis = None  # type: ignore[assignment]

# Atomic read-baseline + append + trim. Uses Redis server TIME for both
# the zset score and the "now" passed back for decay math, so multiple
# workers never disagree on clocks. Members are JSON objects carrying a
# per-session sequence number ("u") so identical score/category pairs in
# the same microsecond remain distinct zset members.
#
# KEYS[1] = window zset, KEYS[2] = sequence counter
# ARGV[1] = score, ARGV[2] = category, ARGV[3] = window size, ARGV[4] = TTL (0 = none)
# Returns cjson {now = <float seconds>, entries = {{ts, score, category}, ...}}
# where entries is the PRE-append window, oldest first.
_ANALYZE_LUA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local raw = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
local entries = {}
for i = 1, #raw, 2 do
    local m = cjson.decode(raw[i])
    entries[#entries + 1] = { tonumber(raw[i + 1]), m.s, m.c }
end
local seq = redis.call('INCR', KEYS[2])
local member = cjson.encode({ s = tonumber(ARGV[1]), c = ARGV[2], u = seq })
redis.call('ZADD', KEYS[1], now, member)
redis.call('ZREMRANGEBYRANK', KEYS[1], 0, -(tonumber(ARGV[3]) + 1))
local ttl = tonumber(ARGV[4])
if ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
end
return cjson.encode({ now = now, entries = entries })
"""

_DEFAULT_KEY_PREFIX = "humane_proxy:traj:"

# TTL when decay is disabled and no explicit ttl_seconds is configured.
_TTL_NO_DECAY_S = 7 * 24 * 3600
_TTL_MIN_S = 3600


class RedisTrajectoryBackend:
    """Shared trajectory state in Redis (single-instance / replicated).

    Not Redis-Cluster-safe: the analyze script touches two keys without
    hash tags, matching the posture of the Redis escalation store.
    """

    def __init__(self, config: dict):
        if not _REDIS_AVAILABLE:
            raise RuntimeError(
                "Redis trajectory backend requires the redis package. "
                "Install with: pip install humane-proxy[redis]"
            )

        traj_cfg = config.get("trajectory", {})
        redis_cfg = traj_cfg.get("redis", {}) or {}

        # Empty URL -> share the escalation store's Redis instance.
        url = redis_cfg.get("url") or (
            config.get("storage", {}).get("redis", {}) or {}
        ).get("url") or "redis://localhost:6379/0"

        self._prefix = redis_cfg.get("key_prefix", _DEFAULT_KEY_PREFIX)
        self._window = int(traj_cfg.get("window_size", 5))
        self._spike_delta = float(traj_cfg.get("spike_delta", 0.35))

        half_life_hours = float(traj_cfg.get("decay_half_life_hours", 24.0))
        self._decay_lambda = (
            math.log(2) / (half_life_hours * 3600)
            if half_life_hours > 0
            else 0.0
        )

        ttl = int(redis_cfg.get("ttl_seconds", 0) or 0)
        if ttl <= 0:
            if half_life_hours > 0:
                ttl = max(int(2 * half_life_hours * 3600), _TTL_MIN_S)
            else:
                ttl = _TTL_NO_DECAY_S
        self._ttl = ttl

        self._client = _redis.Redis.from_url(url, decode_responses=True)
        self._analyze_script = self._client.register_script(_ANALYZE_LUA)

    def ping(self) -> None:
        """Raise if the Redis server is unreachable (fail-fast probe)."""
        self._client.ping()

    # -- key helpers --------------------------------------------------

    def _zkey(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def _seqkey(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}:seq"

    def _spikekey(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}:spike"

    # -- public API (mirrors the in-memory trajectory semantics) ------

    def analyze(self, session_id: str, score: float, category: str = "safe") -> TrajectoryResult:
        from humane_proxy.risk.trajectory import _trend_for_scores, _weighted_mean

        raw = self._analyze_script(
            keys=[self._zkey(session_id), self._seqkey(session_id)],
            args=[score, category, self._window, self._ttl],
        )
        data = _json_loads(raw)
        now = float(data["now"])
        # cjson encodes an empty Lua table as {} rather than [].
        entries = data.get("entries") or []
        if isinstance(entries, dict):
            entries = []

        # Spike baseline = pre-append window, matching detect_spike().
        history = [(float(e[1]), float(e[0])) for e in entries]
        if history:
            delta = score - _weighted_mean(history, now, lam=self._decay_lambda)
            spike = delta > self._spike_delta
        else:
            spike = False

        self._client.set(
            self._spikekey(session_id), "1" if spike else "0", ex=self._ttl
        )

        window = (entries + [[now, float(score), category]])[-self._window:]
        scores = [float(e[1]) for e in window]
        counts: dict[str, int] = {}
        for e in window:
            counts[e[2]] = counts.get(e[2], 0) + 1

        return TrajectoryResult(
            spike_detected=spike,
            trend=_trend_for_scores(scores),
            window_scores=scores,
            category_counts=counts,
            message_count=len(scores),
        )

    def snapshot(self, session_id: str) -> TrajectoryResult:
        from humane_proxy.risk.trajectory import _trend_for_scores

        pipe = self._client.pipeline()
        pipe.zrange(self._zkey(session_id), 0, -1)
        pipe.get(self._spikekey(session_id))
        members, spike_raw = pipe.execute()

        scores: list[float] = []
        counts: dict[str, int] = {}
        for member in members or []:
            try:
                m = _json_loads(member)
            except ValueError:
                continue
            scores.append(float(m["s"]))
            counts[m["c"]] = counts.get(m["c"], 0) + 1

        return TrajectoryResult(
            spike_detected=spike_raw == "1",
            trend=_trend_for_scores(scores),
            window_scores=scores,
            category_counts=counts,
            message_count=len(scores),
        )

    def forget_session(self, session_id: str) -> bool:
        deleted = self._client.delete(
            self._zkey(session_id),
            self._seqkey(session_id),
            self._spikekey(session_id),
        )
        return int(deleted) > 0
