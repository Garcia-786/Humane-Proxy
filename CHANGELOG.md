# Changelog

All notable changes to HumaneProxy will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

Fail-safe defaults and a working, benchmarked Stage-3 cascade. On
SimpleSafetyTests the full pipeline now detects **92%** of unsafe prompts
(Stage 1+2 alone: 21%) while holding a **1.2%** false-positive rate on
XSTest's safe prompts. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for
methodology, machine specs, and per-stage latency.

### Added

- **Stage-2 score calibration** (`stage2.score_ceiling`, default `0.65`) —
  raw cosine similarity in `[safe_threshold, score_ceiling]` is mapped
  onto `[0, 1]`. Escalation thresholds were tuned to Stage-1 keyword
  scores; embedding scores for clear harm top out around 0.55-0.65, so
  without calibration the thresholds were effectively unreachable by
  Stage 2. Ambiguity dampening still keys off the raw cosine scale.
- **Stage-3 safety net** (`pipeline.stage3_on_safe`, default `true`) —
  when Stage 3 is enabled it evaluates messages Stages 1-2 marked safe.
  Embeddings score much criminal content near zero (it is semantically
  far from the anchors), so this is where Stage 3 earns its keep. Set
  `false` to restore the cost-saving early exit for paid providers.
- **Benchmark profiling** — `hp benchmark` now records the machine
  (CPU, RAM, OS, Python), per-run CPU/RSS via `psutil`, throughput,
  and per-stage latency percentiles, all included in `--json-out`. New
  `--delay` flag spaces calls to stay under a Stage-3 provider's rate
  limit. New public-dataset fetcher (`evals/fetch.py`) for XSTest and
  SimpleSafetyTests, plus harm-recall / false-positive-rate / verdict-
  stage metrics in the report.
- **Published benchmarks** — [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### Changed

- **Fail-safe escalation threshold** — criminal-intent `risk_threshold`
  lowered `0.7` -> `0.5`. On the calibrated `[0, 1]` scale, 0.7 sat near
  the embedding ceiling and effectively never triggered; 0.5 roughly
  doubles criminal recall while holding the XSTest false-positive rate
  at 1.2%.
- **Expanded detection coverage** — Stage-2 anchors and Stage-1
  keywords/intent patterns gained clusters for eating disorders,
  violence against persons, scams and fraud, child safety, and weapons
  acquisition (previously scored near zero).
- **Stage-3 OpenAI Chat provider is now reasoning-model-ready** — the
  endpoint and key are already configurable (works with Groq, Together,
  a local server; key falls back from `OPENAI_API_KEY` to `LLM_API_KEY`).
  Reply budget raised (`stage3.openai_chat.max_tokens`, default `1024`)
  and JSON mode made optional (`stage3.openai_chat.json_mode`, default
  `false`) because reasoning models emit chain-of-thought before the
  JSON verdict — strict JSON mode rejected those replies outright. The
  parser now tolerantly extracts the JSON object from free-form text.
  The classifier system prompt no longer instructs the model to lean
  safe on criminal intent.

### Fixed

- **Stage-3 providers no longer discard their own detections** — OpenAI
  Moderation and the chat classifier returned a harmful category with a
  raw confidence score (often below 0.5) that the escalation threshold
  then dropped back to safe. A dedicated safety classifier's *category*
  is the verdict, so a flagged harmful category is now floored to a
  confident score.
- **OpenAI Moderation criminal-intent coverage** — the `illicit` and
  `illicit/violent` categories (drugs, weapons, fraud, other crimes) are
  now mapped to `criminal_intent`; they were previously ignored, sending
  such prompts to safe. The provider now requests
  `omni-moderation-latest` (configurable), which emits those categories;
  the older `text-moderation-*` models do not.
- **Stage-3 chat reply truncation** — the previous 200-token reply cap
  truncated reasoning models mid-answer (and, with JSON mode on, caused
  the provider to reject the whole call), silently failing open to safe.
- Benchmark output falls back cleanly when stdout has no raw buffer and
  no longer crashes under test runners (carried over with the profiling
  work).

### Docs

- README split into focused guides under `docs/` (Pipeline,
  Configuration, Integrations, Deployment); the README is now a concise
  landing page. Added a Node.js / TypeScript integration recipe.
- `SECURITY.md` supported-versions updated (0.6.x supported; 0.5.6-0.5.7
  critical fixes only; earlier versions are unsupported pre-releases).
- `humane_proxy.yaml` marked as a copy-me example template.

---

## [0.6.0] - 2026-07-06

### Added

- **ONNX Runtime Stage-2 backend** (`stage2.backend`, env
  `HUMANE_PROXY_STAGE2_BACKEND`) — Stage 2 can now run on the model's
  pre-exported ONNX graph via a new `onnx` extra (`onnxruntime` +
  `tokenizers` + `huggingface_hub`), with no PyTorch dependency
  (~2 GB lighter install, faster CPU inference). The default `"auto"`
  prefers ONNX when installed and falls back to sentence-transformers;
  both backends produce numerically equivalent embeddings (verified by
  an equivalence test suite, including truncation parity for long
  inputs). Model/anchor/result caches are keyed per backend.
- **Redis-backed trajectory analysis** (`trajectory.backend: redis`, env
  `HUMANE_PROXY_TRAJECTORY_BACKEND`) — session risk windows move to Redis
  sorted sets with an atomic Lua read-append-trim, so every uvicorn
  worker shares one consistent view of a session's trajectory (the
  in-memory default tracks per-process). Reuses `storage.redis.url`
  unless `trajectory.redis.url` is set; sessions auto-expire via TTL
  (default 2x the decay half-life); falls back to in-memory tracking
  with a logged warning when Redis is unavailable. `DELETE
  /admin/sessions/{id}` erasure covers the Redis trajectory keys.

### Changed

- **orjson JSON fast path** — a new optional `perf` extra installs
  `orjson` (Rust-backed JSON); when present, the proxy serializes every
  `/chat` response through `ORJSONResponse`, and storage trigger
  serialization, Stage-3 LLM response parsing, and integration tool
  output all route through an internal shim (`humane_proxy._json`).
  Without the extra, everything falls back to the stdlib `json` module
  with identical behavior.

---

## [0.5.7] - 2026-07-06

### Added

- Brand assets: `docs/assets/banner.png` (README header) and
  `docs/assets/logo.png` (canonical logo, also used by the dashboard
  templates). Referenced via raw GitHub URLs so PyPI and Glama render them.

### Security

- Upstream LLM connection failures no longer echo exception details
  (which can include internal URLs and network information) to proxy
  clients — details go to server logs; clients get a generic 503 message
  (CodeQL #2).
- The tests workflow's `GITHUB_TOKEN` is now restricted to
  `contents: read` (CodeQL #6).

### Fixed

- Webhook log-sanitization tests assert the exact sanitized log line
  instead of a bare domain substring (CodeQL #14, #15).
- The no-emoji scan's character class no longer contains overlapping
  ranges (CodeQL #16).

### Removed

- `dashboard/public/logo.png` — superseded by `docs/assets/logo.png`.

---

## [0.5.6] - 2026-07-06

### Changed

- **No-emoji policy across the repository** — all emoji characters removed
  from source, output strings, docs, and CI files:
  - Severity markers in webhook alerts (Slack/Discord/Teams/email), the
    CRITICAL log banner, and CLI output are now bracketed ASCII tags
    (`[SELF-HARM]`, `[ALERT]`, `[FLAGGED]`, `[SAFE]`, `[OK]`, `[WARN]`,
    `[INFO]`, `[ERROR]`), matching the benchmark's existing `[PASS]`/`[FAIL]`
    convention. Discord/Teams urgency remains color-coded.
  - The self-harm care response lists crisis resources under plain country
    names (flag emojis removed; helpline content unchanged).
  - Docs tables (README, COMPLIANCE, LAUNCHGUIDE, SECURITY) use
    "Yes"/"No"/"Conditional" instead of symbols.
  - The policy is CI-enforced by a new `tests/test_no_emoji.py`, which scans
    every tracked file; the single sanctioned exemption is CONTRIBUTING.md's
    AI-PR title marker.
- **Repository layout decluttered** — `COMPLIANCE.md` and `LAUNCHGUIDE.md`
  moved to `docs/`; `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` moved to
  `.github/` alongside `SECURITY.md`; the Glama marketplace `Dockerfile` and
  `.dockerignore` moved to `deploy/glama/` with a README (HumaneProxy is not
  served via Docker — the container definition exists only for the Glama
  listing).

### Removed

- `requirements.txt` — duplicated `pyproject.toml` dependencies and had
  already drifted once; `pyproject.toml` is the single source of truth.
- `.github/labels.yml` — one-time label-sync input, no longer consumed by
  any workflow (existing GitHub labels are unaffected).

### Fixed

- `humane-proxy init` crashed with `UnicodeEncodeError` on default Windows
  (cp1252) consoles — CLI output is now console-encoding-safe.

---

## [0.5.5] - 2026-07-06

### Changed

- **Context reducers are now span-aware** — a keyword and an intent pattern
  firing on the *same phrase* count as one signal, so reducers can neutralize
  false positives like "how to make a bomb in minecraft" (previously scored
  1.0 because two triggers disabled reduction). Two or more *separate*
  harmful expressions in one message still disable reduction entirely.
- **The audit log is exempt from the alert rate limit** — every escalation is
  now persisted; `escalation.rate_limit_max` only caps operator alerts
  (webhooks + CRITICAL log) per session per window. Previously events past
  the quota were dropped entirely, blinding the audit trail for exactly the
  sessions escalating hardest. `escalate()` results gain an `alerted` field.

### Security

- **Webhook URLs and response bodies no longer leak into logs** (#33) —
  logs show only the webhook's scheme + host (Slack/Discord/Teams tokens
  live in the URL path), and error logs record status + length with bodies
  demoted to `DEBUG`.
- **Raw upstream LLM bodies are no longer echoed to clients** (#33) —
  non-JSON upstream responses return a generic error; the body is available
  at `DEBUG` for operators.

### Fixed

- `.env.example` and the `humane-proxy init` scaffold now include the
  documented `OPENAI_API_KEY`, `GROQ_API_KEY`, and `HUMANE_PROXY_ADMIN_KEY`
  secrets with purpose comments (#63).
- **Malformed pipeline config no longer crashes classification** — a
  non-list `enabled_stages` (string/None) falls back to `[1]`, junk entries
  are filtered out, and non-numeric thresholds coerce to safe defaults, all
  with logged warnings. The three long-standing `xfail` config-validation
  tests are now regular passes.

---

## [0.5.4] - 2026-07-05

### Performance

- **Shared HTTP connection pool** — the reverse proxy, all webhook
  dispatchers, and the three Stage-3 classifiers now reuse one process-level
  `httpx.AsyncClient` instead of opening a fresh TCP+TLS connection per call.
- **Embedding caches** — Stage-2 anchor sentences are encoded once per
  process (per model) instead of per classifier instance, and a bounded
  5-minute TTL cache serves repeated identical messages without re-encoding.
- **Singleton pipelines** — the MCP `check_message_safety` tool and the
  LlamaIndex/CrewAI/AutoGen integrations reuse one pipeline instead of
  rebuilding it (including Stage-2 setup) on every tool call.

### Fixed

- **`stage1.heuristics` OTel span now exists** — the documented span
  hierarchy listed it, but Stage 1 was never traced. The `session_id` span
  attribute is now populated too (classify call sites pass it as a keyword).
- `to_dict()` now includes `should_escalate`, matching the MCP tool's
  documented return shape.
- FastAPI app version is derived from `humane_proxy.__version__` instead of
  a hardcoded string.
- `requirements.txt` was missing `numpy` (already a core dependency in
  `pyproject.toml`); removed dead imports left by the shared-client refactor.

---

## [0.5.3] - 2026-07-05

### Fixed

- **HTTP MCP auth crashed with real fastmcp** — the server imported a
  `fastmcp.server.auth.BearerTokenAuth` class that no fastmcp release
  exports, so setting `HUMANE_PROXY_ADMIN_KEY` (exactly as the README
  instructs) made `mcp-serve` fail at import. Auth now uses fastmcp's
  `StaticTokenVerifier`; the `[mcp]` extra requires `fastmcp>=2.11`.
  The masking test that injected a fake fastmcp module into `sys.modules`
  was replaced with tests against the real package.
- **`.env` guidance corrected** — README Quick Start and `humane-proxy init`
  said to put `LLM_API_KEY`/`LLM_API_URL` "in .env", but HumaneProxy never
  loads `.env` files; following the docs verbatim produced a 503 on every
  safe message. Docs now show explicit exports (with `source .env` guidance),
  and the interceptor reads both vars at request time instead of import time.

### Tests

- `conftest.py` now resets the storage-factory singleton per test — the
  cached store previously kept the first test's temp DB path for the whole
  session, making per-test DB isolation illusory.

---

## [0.5.2] - 2026-07-05

### Fixed

- **User configuration finally reaches the classifiers** — the heuristic
  classifier and risk trajectory read config through a legacy package-only
  loader at import time, so `heuristics:`/`trajectory:` overrides in
  `humane_proxy.yaml` and the documented `HUMANE_PROXY_DECAY_HALF_LIFE` env
  var were silently ignored. Both modules now rebuild their settings from the
  merged config (`config.get_config()`) whenever it changes; the legacy
  `humane_proxy.load_config()` is deprecated.
- **Duplicate top-level keys removed from package `config.yaml`** — the file
  declared `safety:` and `escalation:` twice; PyYAML silently keeps only the
  last block, leaving the first ones as dead config.
- **Empty-string filters no longer 500** — `GET /admin/escalations?category=`
  desynced the SQL WHERE clause from the parameter list in the SQLite and
  PostgreSQL stores ("incorrect number of bindings"). Empty strings are now
  treated as "no filter".
- **GitHub Action defaults to the `ml` extra** — `hp benchmark` runs stages
  1+2 by default but Stage 2 silently no-ops without `sentence-transformers`,
  so the action failed its own sample dataset in `--ci` mode when installed
  without extras.

---

## [0.5.1] - 2026-07-05

### Fixed

- **Email-only alert configs never dispatched** — the webhook gate checked a
  non-existent `webhooks.email_to` key while the dispatcher reads
  `webhooks.email.to`. Deployments with only SMTP alerts configured silently
  sent nothing.
- **LlamaGuard category mapping** — S9 (Indiscriminate Weapons) was mapped to
  `safe` and S10 (Hate) to `self_harm`, so weapons queries passed Stage 3
  unflagged while hate speech received a suicide-crisis care response. S9 now
  maps to `criminal_intent`, S10 to `safe` (out of scope). An `unsafe` verdict
  with no in-scope codes now returns score 0.0 with an auditable
  `llamaguard:unsafe_out_of_scope` trigger instead of a score-inflating 0.85.
- **Redis rate limit never triggered** — the counter was set to 1 once and
  never incremented, so per-session alert quotas were unlimited. Replaced with
  an atomic Lua `INCR`+`EXPIRE` script that also removes the multi-worker race
  (issue #5 pattern).
- **Redis `delete_session` left dangling ids** in the `category:{cat}` indexes,
  inflating counts after erasure. Category indexes are now cleaned up.
- **Right to erasure now covers live trajectory state** — `DELETE
  /admin/sessions/{id}` clears the in-memory risk trajectory (new
  `trajectory.forget_session()`), so `/admin/sessions/{id}/risk` no longer
  returns data for erased sessions.

---

## [0.5.0] - 2026-07-05

### Added

- **OpenTelemetry distributed tracing** (closes #7) — new `[telemetry]` install extra, `telemetry.py` module owning all OTel logic, `@traced_stage` spans across the 3-stage pipeline, `HUMANE_PROXY_TELEMETRY_ENABLED` env override, zero-overhead no-op tracer when disabled, and a README section with Jaeger validation steps.
- **Region-aware care response** — optional `safety.categories.self_harm.region` (ISO country code) surfaces that country's crisis resources first in block mode while keeping the full international list.
- **Crisis helplines for 10 new countries** (#26) — Japan, South Korea, Spain, Italy, Mexico, New Zealand and more, alongside the existing US/IN/GB/AU/CA/DE/FR/BR/ZA resources.
- **Web dashboard template** — initial `dashboard/` scaffold (template + logo) for a future escalations UI.
- **Date filtering and sorting** for escalation queries — `EscalationStore.query()`/`count()` extended with `date_from`, `date_to`, `sort_by`, `sort_order` across SQLite, Redis, and PostgreSQL backends.
- Table of contents in README.
- Test coverage: heuristic edge cases (Unicode, leet-speak, boundary scoring — #11), pipeline config validation (#35), malformed `/chat` payloads, `_extract_last_user_message` multimodal cases, `_weighted_mean` edge cases (#60).

### Fixed

- **Storage-backend bypass** (#37) — admin API, CLI, and MCP tools now route through `get_store()` instead of direct SQLite calls, so Redis/Postgres deployments see consistent data everywhere.
- **Multimodal content arrays** (#44, #45, #48) — `/chat` now extracts text parts from OpenAI-style content arrays and no longer crashes on string or malformed message content.
- Empty request body on `/chat` returns a clean 400 instead of an unhandled error (#41).
- CodeQL SQL-injection alerts resolved via statically generated SQL templates; timezone conversion bug and CSV `triggers` serialization fixed in escalation export.

### Security

- **Timing-attack fix** in admin authentication — Bearer token comparison now uses `hmac.compare_digest` (#18, #21).
- **Hardened HTTP MCP** — binds to `127.0.0.1` by default, warns on public binds without a token, optional Bearer auth via `HUMANE_PROXY_ADMIN_KEY`, and bounded/validated `list_recent_escalations` queries (#17).

### Docs

- Architecture diagram replaced with Mermaid.js flowchart, including corrected Stage-1 transition labels (#4).

---

## [0.4.0] - 2026-04-18

### Added

- **`hp benchmark` CLI command** — run evaluation datasets through the safety pipeline with per-category precision, recall, F1, confusion matrix, and latency stats. Supports `--ci` flag for CI/CD exit-code gating. Uses `rich` for beautiful terminal rendering.
- **GitHub Action** (`action.yml`) — `uses: Vishisht16/Humane-Proxy@v0.4.0` — run safety benchmarks in CI pipelines to catch regressions before production.
- **`hp` CLI alias** — all commands now available via `hp` shorthand (e.g., `hp check`, `hp benchmark`, `hp start`).
- **Sample evaluation dataset** (`evals/sample.json`) — 20 test cases: 5 self-harm, 5 criminal intent, 10 safe (including false-positive bait like "kill a process" and "dying of laughter").
- `COMPLIANCE.md` — HIPAA, GDPR, and SOC 2 readiness assessment with operator responsibilities.
- `.github/SECURITY.md` — vulnerability disclosure policy with supported versions and coordinated disclosure window.
- `LAUNCHGUIDE.md` for MCP Marketplace extended "About" section with full pipeline docs, MCP tool reference, and setup guides.
- `.github/FUNDING.yml` with GitHub Sponsors and Ko-fi links.
- `.github/CODEOWNERS` locking architecture and config files to maintainer review.
- AI Automation Policy and anti-spam rules in `CONTRIBUTING.md`.
- Contributor License Agreement (CLA) notice in `CONTRIBUTING.md`.
- "Available On" section in README with Glama AAA rating and MCP Marketplace links.
- "As an MCP Server" quick start section in README.
- MCP Marketplace badge in README.

### Fixed

- FastAPI dependency bumped to `>=0.109.1` to patch Content-Type Header ReDoS vulnerability (GHSA-qf9m-vfgh-m389).
- `server.json` env vars now include `"required": false` for cross-marketplace parser compatibility.

### Security

- Branch protection rules documented and recommended for `main`.
- CODEOWNERS prevents unauthorized modification of `pyproject.toml`, `CHANGELOG.md`, `Dockerfile`, and pipeline code.
- Honeypot mechanism in CONTRIBUTING.md to identify unsupervised AI agent PRs.

### Dependencies

- Added `rich>=13.0.0` as a core dependency for CLI benchmark rendering.

---

## [0.3.1] - 2026-04-03

### Fixed

- Fixed `mcp-serve` stdout logging bug breaking the MCP `stdio` JSON-RPC transport protocol.

### Added

- **Exponential time-decay** for risk trajectory: historical scores are weighted by `e^{-λΔt}` with a configurable half-life (default 24 h). Prevents stale history from penalising returning users while keeping rapid-escalation detection intact.
- Environment variable `HUMANE_PROXY_DECAY_HALF_LIFE` to configure the decay half-life in hours.
- Added multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) recommendation in documentation for non-English coverage.

---

## [0.3.0] — 2026-04-03

### Added

- **Swappable Storage Backend:** Implemented the Repository Pattern for escalation storage. Now supports `"sqlite"` (default), `"redis"`, and `"postgres"`. Configurable via `storage.backend` in `humane_proxy.yaml` (`[redis]` and `[postgres]` install extras available).
- **Framework Integrations:** First-class support for LlamaIndex (`get_safety_tools()`), CrewAI (`get_safety_tools()`), and AutoGen (`register_safety_tools()`). Available via respective install extras.
- **Configurable Self-Harm Threshold:** Added `escalate_threshold` to `self_harm` category config (defaults to `0.5`). Replaces the previous behaviour of unconditionally forcing all self-harm scores to 1.0.
- **Stage 2 Ambiguity Dampening:** Semantic embedding scores in the "grey zone" (0.30–0.55) for self-harm are now compared against benign anchors (e.g. "no point in continuing this project"). If benign semantics are competitive, the score is halved to prevent false positives.
- **Enhanced Admin API:** Added `/admin/health` (no auth), `/admin/config` (sanitised config view), and `/admin/escalations/export` (CSV export). Enhanced `/admin/stats` with `top_sessions`, `by_stage`, and `hourly_last_24h` breakdowns. Added date filtering and sorting to `/admin/escalations`.
- **Container Files:** Added  `Dockerfile` and `.dockerignore` to allow building Docker image with ~500KB context size.

### Changed

- **Stage 2 Model Caching:** The sentence-transformer model is now a process-level singleton. `humane-proxy check` no longer reads the 80MB model from disk on every call. Added a warm-up encode step to eliminate lazy-init latency.
- FastMCP constructor gracefully removes `description` kwarg to support `fastmcp>=0.4.0`.

---

## [0.2.3] — 2026-04-01

### Fixed

- **Stage 2 never ran:** The pipeline's early-exit logic was too aggressive — when Stage 1 scored a message as safe (score 0.0), it would early-exit before the embedding classifier had a chance to evaluate it. This defeated Stage 2's entire purpose: catching *semantically* dangerous messages that keyword matching misses. Now, when Stage 2 is enabled, all messages that Stage 1 does not flag as `self_harm` proceed to the embedding classifier.
- **Stage 3 warning shown incorrectly:** Users with `enabled_stages: [1]` or `[1, 2]` saw a Stage 3 "DISABLED" warning even though they never configured Stage 3. The warning now only appears when `3` is in `enabled_stages` but no API key / provider is available.
- **`HUMANE_PROXY_ENABLED_STAGES` env var not wired up:** Documented in README but not implemented in the config loader. Now accepts comma-separated ints (e.g. `"1,2"`).
- **FastAPI app version** pinned to `0.2.0` — updated to `0.2.3`.

### Added

- **`glama.json`** metadata for Glama MCP directory listing.
- **Real embedding model tests:** New `TestEmbeddingClassifierReal` test class that exercises the full `all-MiniLM-L6-v2` classify flow (guarded by `pytest.importorskip`, auto-skipped in CI).
- **Pipeline early-exit regression tests:** `TestStage2EarlyExitFix` class verifying that Stage 2 is always invoked when enabled, even for messages heuristics considers safe.
- **Stage 3 warning tests:** Tests that verify the warning is only shown when Stage 3 is in `enabled_stages`.
- **Glama badges** in README for MCP server card and quality score.

### Changed

- README updated to clarify Stage 2 behaviour: when enabled, all messages flow through the embedding classifier. Stage 1 heuristics becomes an early-exit optimisation for clear self-harm only, not a safety determiner.
- README updated to clarify `LLM_API_KEY` / `LLM_API_URL` are only needed for the reverse proxy server, not for the library API or MCP server.

---

## [0.2.2] — 2026-03-31

### Added

- **MCP HTTP transport mode:** `humane-proxy mcp-serve --transport http` exposes tools over Streamable HTTP for remote clients and registry listing.
- **Official MCP Registry integration:** `server.json` metadata + `publish-mcp.yml` GitHub Actions workflow for automated publishing via OIDC.
- **LangChain integration:** `humane_proxy.integrations.langchain` module with `get_safety_tools()` and `get_langchain_mcp_config()` helpers. New `[langchain]` install extra.
- **`<!-- mcp-name -->`** marker in README for MCP Registry discovery.

### Removed

- `smithery.yaml` — Smithery no longer supports GitHub-based stdio imports; HTTP transport is used instead.

### Changed

- `humane-proxy mcp-serve` now accepts `--transport` (`stdio` | `http`), `--host`, and `--port` options.
- `[all]` install extra now includes `[langchain]`.

### Fixed

- `server.json` now uses the correct MCP Registry schema (`static.modelcontextprotocol.io`) and `packages` format.
- `publish-mcp.yml` uses proper `login github-oidc` two-step auth.
- `mcp-name` marker uses full reverse-DNS namespace for PyPI ownership validation.

---

## [0.2.0] — 2026-03-31

### Added

- **3-Stage Cascade Safety Pipeline** — fully configurable per stage:
  - **Stage 1 (Heuristics):** Always-on, sub-millisecond keyword + regex classifier.
  - **Stage 2 (Embeddings):** Semantic similarity using `sentence-transformers` (`pip install humane-proxy[ml]`).
  - **Stage 3 (Reasoning LLM):** Optional; auto-detected from API keys. Supports LlamaGuard (Groq), OpenAI Moderation API, or any OpenAI-compatible chat model.
- **International Self-Harm Care Response System:**
  - **Block mode (default):** Replies with an empathetic message and crisis resources for 10+ countries (US, India, UK, Australia, Canada, Germany, France, Brazil, South Africa + IASP/Befrienders international).
  - **Forward mode:** Injects a care-context system prompt before forwarding to the upstream LLM.
  - Both modes are configurable via `humane_proxy.yaml`.
- **MCP Server Integration:** Expose safety tools via the Model Context Protocol (`pip install humane-proxy[mcp]`). Tools: `check_message_safety`, `get_session_risk`, `list_recent_escalations`. Smithery-compatible.
- **REST Admin API:** Mounted at `/admin`, secured with `HUMANE_PROXY_ADMIN_KEY` Bearer token. Endpoints: list escalations (paginated + filterable), single record, session risk history, aggregate stats, session data deletion (privacy right to erasure).
- **Enhanced CLI commands:**
  - `humane-proxy escalations [--category] [--limit] [--session]` — audit log viewer.
  - `humane-proxy session <id>` — per-session risk history.
  - `humane-proxy mcp-serve` — starts the MCP stdio server.
- **Microsoft Teams webhook** (adaptive card format) alongside existing Slack, Discord, PagerDuty.
- **Email alerts** via SMTP (stdlib `smtplib`, zero extra deps).
- **Privacy controls:** SHA-256 message hashing, `stage_reached` and `reasoning` stored per escalation.
- **Enhanced Risk Trajectory:** Trend detection (escalating / stable / declining), category distribution per session, spike detection.
- **BYOK Stage-3:** Auto-detects `OPENAI_API_KEY` → OpenAI Moderation, `GROQ_API_KEY` → LlamaGuard; prints clear setup guidance if neither is found.
- **PyPI publish workflow** (`.github/workflows/pypi.yml`) via Trusted Publishers (OIDC, no token needed).

### Changed

- Risk threshold default lowered 0.8 → 0.7 (better recall for human safety cases).
- Self-harm keyword score raised 0.5 → 0.7 (these are high-intent by nature).
- Classification pipeline now returns structured `ClassificationResult` with `category`, `score`, `triggers`, `stage`, and `reasoning`.
- `PipelineResult.to_dict()` includes `stage_reached` for explainability.
- Interceptor `/chat` endpoint now uses the full async 3-stage pipeline.
- Escalation DB schema extended: `message_hash`, `stage_reached`, `reasoning` columns with auto-migration.

---

## [0.1.0] — 2026-03-31

### Added

- Heuristic keyword + regex classifier specifically for **self-harm** and **criminal intent** (not jailbreaks).
- Context-aware false positive reduction — e.g. `"I want to die laughing"` → `safe`.
- First-person intent patterns and method-seeking patterns.
- Per-session risk trajectory with spike detection.
- SQLite-backed escalation logging with configurable per-session rate limiting.
- Webhook alerts: Slack (Block Kit), Discord (embeds), PagerDuty (Events API v2).
- FastAPI reverse proxy middleware (`humane-proxy start`) that intercepts, classifies, and optionally forwards.
- CLI: `init`, `start`, `check`, `version`.
- Programmatic `HumaneProxy` class usable as a Python library.
- Layered configuration: package defaults → `humane_proxy.yaml` → `HUMANE_PROXY_*` env vars.
- Apache 2.0 licence. Attribution to Vishisht Mishra (@Vishisht16).
- 94 unit & integration tests.
