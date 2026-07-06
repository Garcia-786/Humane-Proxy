# Configuration Reference

HumaneProxy layers configuration from three sources. Highest priority wins:

1. `HUMANE_PROXY_*` environment variables
2. `humane_proxy.yaml` in your working directory (or the path in `HUMANE_PROXY_CONFIG`)
3. Package-bundled defaults

Scaffold a starter config with `humane-proxy init`, or copy the annotated
example [`humane_proxy.yaml`](../humane_proxy.yaml) from the repository root.

## Table of Contents

- [Reference Table](#reference-table)
- [Alert Webhooks](#alert-webhooks)
- [Storage Backends](#storage-backends)
- [Privacy](#privacy)

---

## Reference Table

| YAML key | Env var | Default | Description |
|---|---|---|---|
| `safety.risk_threshold` | `HUMANE_PROXY_RISK_THRESHOLD` | `0.5` | Score threshold for criminal_intent escalation (calibrated `[0,1]` scale) |
| `safety.categories.self_harm.escalate_threshold` | `HUMANE_PROXY_SELF_HARM_THRESHOLD` | `0.5` | Score threshold for self_harm escalation |
| `safety.categories.self_harm.response_mode` | — | `"block"` | `"block"` or `"forward"` |
| `safety.categories.self_harm.region` | — | `""` | Optional ISO country code; surfaces that country's crisis resources first in block mode |
| `safety.spike_boost` | `HUMANE_PROXY_SPIKE_BOOST` | `0.25` | Score boost on trajectory spike |
| `server.host` | `HUMANE_PROXY_HOST` | `127.0.0.1` | Proxy bind address |
| `server.port` | `HUMANE_PROXY_PORT` | `8000` | Proxy port |
| `pipeline.enabled_stages` | `HUMANE_PROXY_ENABLED_STAGES` | `[1]` | Active stages (e.g. `1,2,3`) |
| `pipeline.stage1_ceiling` | `HUMANE_PROXY_STAGE1_CEILING` | `0.3` | Early exit after Stage 1 |
| `pipeline.stage2_ceiling` | `HUMANE_PROXY_STAGE2_CEILING` | `0.4` | Early exit after Stage 2 |
| `pipeline.stage3_on_safe` | — | `true` | Run Stage 3 on messages Stages 1-2 marked safe (fail-safe net). `false` restores the cost-saving early exit |
| `stage2.model` | — | `"all-MiniLM-L6-v2"` | Embedding model name |
| `stage2.backend` | `HUMANE_PROXY_STAGE2_BACKEND` | `"auto"` | Stage 2 inference: `"auto"`, `"onnx"`, `"sentence-transformers"` |
| `stage2.safe_threshold` | — | `0.35` | Cosine similarity below this is safe |
| `stage2.score_ceiling` | — | `0.65` | Cosine at/above this calibrates to score `1.0` |
| `stage3.provider` | `HUMANE_PROXY_STAGE3_PROVIDER` | `"auto"` | Stage 3 provider |
| `stage3.timeout` | `HUMANE_PROXY_STAGE3_TIMEOUT` | `10` | Stage 3 timeout (s) |
| `stage3.openai_moderation.model` | — | `"omni-moderation-latest"` | Moderation model (omni emits `illicit` categories) |
| `stage3.openai_chat.max_tokens` | — | `1024` | Reply budget; reasoning models need room before the JSON verdict |
| `stage3.openai_chat.json_mode` | — | `false` | Strict JSON mode; leave off for reasoning models |
| `trajectory.window_size` | — | `5` | Messages in the rolling risk window |
| `trajectory.spike_delta` | — | `0.35` | Delta threshold for spike detection |
| `trajectory.decay_half_life_hours` | `HUMANE_PROXY_DECAY_HALF_LIFE` | `24.0` | Time-decay half-life; `0` disables decay |
| `trajectory.backend` | `HUMANE_PROXY_TRAJECTORY_BACKEND` | `"memory"` | Trajectory state: `"memory"` (per-process) or `"redis"` (shared) |
| `privacy.store_message_text` | — | `false` | Store raw text (vs SHA-256 hash) |
| `escalation.rate_limit_max` | `HUMANE_PROXY_RATE_LIMIT_MAX` | `3` | Max alerts per session/window |
| `escalation.db_path` | `HUMANE_PROXY_DB_PATH` | package dir | SQLite database path |
| `storage.backend` | `HUMANE_PROXY_STORAGE_BACKEND` | `"sqlite"` | `"sqlite"`, `"redis"`, `"postgres"` |
| `storage.redis.url` | `HUMANE_PROXY_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `storage.postgres.dsn` | `HUMANE_PROXY_POSTGRES_DSN` | — | PostgreSQL DSN |
| `telemetry.enabled` | `HUMANE_PROXY_TELEMETRY_ENABLED` | `false` | OpenTelemetry tracing |

Webhook URLs can also be set via `HUMANE_PROXY_SLACK_URL`,
`HUMANE_PROXY_DISCORD_URL`, and `HUMANE_PROXY_PAGERDUTY_KEY`.

---

## Alert Webhooks

Configure in `humane_proxy.yaml`:

```yaml
escalation:
  rate_limit_max: 3            # max alerts per session per window
  rate_limit_window_hours: 1

  webhooks:
    slack_url: "https://hooks.slack.com/services/..."
    discord_url: "https://discord.com/api/webhooks/..."
    pagerduty_routing_key: "your-routing-key"
    teams_url: "https://outlook.office.com/webhook/..."

    # Email alerts via SMTP (stdlib, no extra deps)
    email:
      host: "smtp.gmail.com"
      port: 587
      use_tls: true
      username: "your@gmail.com"
      password: "app-password"
      from: "humane-proxy@yourorg.com"
      to:
        - "safety-team@yourorg.com"
        - "oncall@yourorg.com"
```

The rate limit caps operator *notifications* only — every escalation is
always persisted to the audit log regardless of the quota.

---

## Storage Backends

```yaml
storage:
  backend: "sqlite"   # or "redis", "postgres"

  redis:
    url: "redis://localhost:6379/0"

  postgres:
    dsn: "postgresql://user:pass@localhost/humane_proxy"
```

- **SQLite** (default) — zero-config, WAL mode, full query/sort/stats support.
- **Redis** — fast, atomic Lua rate limiting; some advanced stats are limited. Requires `pip install humane-proxy[redis]`.
- **PostgreSQL** — full query support for multi-instance deployments. Requires `pip install humane-proxy[postgres]`.

---

## Privacy

By default HumaneProxy **never stores raw message text**. Only a SHA-256 hash is persisted for correlation. The escalation DB stores:

- `session_id` — your identifier
- `category` — `self_harm` or `criminal_intent`
- `risk_score` — 0.0-1.0
- `triggers` — which patterns fired
- `message_hash` — SHA-256 of the original text
- `stage_reached` — which pipeline stage produced the result
- `reasoning` — Stage-3 LLM reasoning (if available)

To enable raw text storage (e.g. for human review):

```yaml
privacy:
  store_message_text: true
```

`DELETE /admin/sessions/{id}` implements the right to erasure: it removes
all stored records for a session, live in-memory trajectory state, and
Redis trajectory keys when the Redis backend is active.
