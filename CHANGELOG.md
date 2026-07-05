# Changelog

All notable changes to HumaneProxy will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
