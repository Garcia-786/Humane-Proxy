# Deployment & Operations

Running HumaneProxy in production: the CLI, the admin API, CI/CD gating,
and observability.

## Table of Contents

- [CLI Reference](#cli-reference)
- [REST Admin API](#rest-admin-api)
- [GitHub Action — CI/CD Safety Gate](#github-action--cicd-safety-gate)
- [OpenTelemetry Tracing](#opentelemetry-tracing)

---

## CLI Reference

All commands are available via both `humane-proxy` and the shorthand `hp`.

```bash
# Safety check
hp check "I want to end my life"
# [FLAGGED] self_harm
# Score   : 1.0
# Category: self_harm

# Run benchmark evaluation
hp benchmark --dataset evals/sample.json
hp benchmark --dataset evals/sample.json --ci  # exit code 1 on failure

# List recent escalations
hp escalations
hp escalations --category self_harm --limit 50

# Session risk history
hp session user-42

# Start proxy server
hp start [--host 0.0.0.0] [--port 8000]

# MCP server (requires [mcp] extra)
hp mcp-serve
```

### Proxy environment variables

The reverse proxy (`hp start`) forwards safe messages to your upstream
LLM and needs to know where it lives:

```bash
export LLM_API_KEY=sk-...
export LLM_API_URL=https://api.your-llm.com/v1/chat/completions
hp start
```

These are only needed for the proxy server. The Python library, CLI
checks, and MCP server work without them.

HumaneProxy does **not** auto-load `.env` files. If you keep these values
in the `.env` scaffolded by `humane-proxy init`, load it into the
environment yourself — e.g. `set -a; source .env; set +a` on Unix shells,
or run via a process manager / `docker --env-file` that injects it.

---

## REST Admin API

Mounted at `/admin`, secured with `HUMANE_PROXY_ADMIN_KEY` Bearer token:

```bash
export HUMANE_PROXY_ADMIN_KEY=your-secret-key

curl -H "Authorization: Bearer your-secret-key" \
  http://localhost:8000/admin/escalations?category=self_harm&limit=10

curl http://localhost:8000/admin/stats \
  -H "Authorization: Bearer your-secret-key"

# Delete session data (right to erasure)
curl -X DELETE http://localhost:8000/admin/sessions/user-42 \
  -H "Authorization: Bearer your-secret-key"
```

| Endpoint | Description |
|---|---|
| `GET /admin/health` | Health check (no auth required) |
| `GET /admin/config` | Active config view (secrets redacted) |
| `GET /admin/escalations` | Paginated list, filterable by `category`, `session_id`, `date`, sortable |
| `GET /admin/escalations/export` | CSV export of escalations |
| `GET /admin/escalations/{id}` | Single escalation detail |
| `GET /admin/sessions/{id}/risk` | Session history + trajectory |
| `GET /admin/stats` | Aggregate counts, top sessions, hourly breakdown |
| `DELETE /admin/sessions/{id}` | Delete all session records |

---

## GitHub Action — CI/CD Safety Gate

Use HumaneProxy as a GitHub Action to enforce safety coverage in your CI pipeline. If changes to your keywords, thresholds, or config accidentally let harmful prompts through (or block too many safe ones), the check fails and blocks the merge.

```yaml
# .github/workflows/safety-benchmark.yml
name: Safety Benchmark
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: Vishisht16/Humane-Proxy@v0.5.0
        with:
          dataset: evals/sample.json
```

| Input | Required | Default | Description |
|---|---|---|---|
| `dataset` | Yes | — | Path to JSON evaluation dataset |
| `python-version` | No | `3.12` | Python version to use |
| `extra` | No | `ml` | pip extras. Defaults to `ml` — the benchmark runs stages 1+2 and Stage 2 needs the embeddings dependency. Pass `""` for heuristics-only |

---

## OpenTelemetry Tracing

HumaneProxy can export distributed traces to Jaeger, Grafana Tempo, or Datadog, giving full visibility into pipeline latency and safety decisions per request.

### Install

```bash
pip install humane-proxy[telemetry]
```

### Enable

In `humane_proxy.yaml`:

```yaml
telemetry:
  enabled: true
  endpoint: "http://localhost:4317"   # OTLP gRPC endpoint
```

Or via environment variable (wins over yaml):

```bash
export HUMANE_PROXY_TELEMETRY_ENABLED=true
```

### Span hierarchy

Every request produces a trace like this:

```
pipeline.classify                  [root — full request latency]
  ├── stage1.heuristics            [< 1ms — keyword + regex]
  ├── stage2.embeddings            [~100ms — embeddings backend]
  └── stage3.reasoning_llm         [1-3s — Groq/OpenAI — only when ambiguous]
```

Early-exit messages only produce child spans for stages that actually ran — making it immediately obvious where the pipeline terminated.

### Span attributes

| Attribute | Type | Description |
|---|---|---|
| `humane_proxy.session_id` | string | Your session identifier |
| `humane_proxy.category` | string | `safe`, `self_harm`, or `criminal_intent` |
| `humane_proxy.final_score` | float | Risk score 0.0-1.0 |
| `humane_proxy.stage_reached` | int | Last pipeline stage executed (1, 2, or 3) |
| `humane_proxy.triggers_count` | int | Number of Stage 1 keyword/regex triggers |
| `humane_proxy.message_hash` | string | SHA-256 of the original message |

> **Privacy:** Raw message text is never added to spans. `humane_proxy.message_hash` lets you correlate spans with your own audit logs without storing the original text in your tracing backend.

### Validate locally with Jaeger

```bash
# Start Jaeger all-in-one (OTLP gRPC on port 4317, UI on port 16686)
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest

# Enable telemetry and start HumaneProxy
export HUMANE_PROXY_TELEMETRY_ENABLED=true
humane-proxy start

# Send a test message
hp check "I want to end my life"

# Open Jaeger UI -> http://localhost:16686
# Select service: humane_proxy to see the full trace
```

### Zero overhead when disabled

When `telemetry.enabled: false` (the default), a `NoOpTracerProvider` is registered. All OTel API calls are pure no-ops at the library level — no `if enabled` checks anywhere in the pipeline hot path.
