"""HumaneProxy CLI — developer-friendly command-line interface.

Install the package and run::

    humane-proxy init          # scaffold config + .env in your project
    humane-proxy start         # start the proxy server
    humane-proxy check "text"  # quick safety check from terminal
    humane-proxy benchmark     # run evaluation dataset through the pipeline
    humane-proxy version       # print version info

The ``hp`` alias is also available::

    hp check "text"
    hp benchmark --dataset evals/sample.json
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import click

_BANNER = r"""
  _   _                                  ____
 | | | |_   _ _ __ ___   __ _ _ __   ___|  _ \ _ __ _____  ___   _
 | |_| | | | | '_ ` _ \ / _` | '_ \ / _ \ |_) | '__/ _ \ \/ / | | |
 |  _  | |_| | | | | | | (_| | | | |  __/  __/| | | (_) >  <| |_| |
 |_| |_|\__,_|_| |_| |_|\__,_|_| |_|\___|_|   |_|  \___/_/\_\\__, |
                                                                |___/
"""

_DEFAULT_YAML_CONTENT = """\
# HumaneProxy — project-level configuration.
# Values here override the package defaults.
# See: https://github.com/your-org/humane-proxy#configuration

server:
  host: "0.0.0.0"
  port: 8000
  reload: true          # auto-reload on code changes (dev mode)

safety:
  risk_threshold: 0.7
  spike_boost: 0.25

heuristics:
  self_harm_keywords:
    - "want to die"
    - "kill myself"
    - "end my life"
    - "suicide"
    - "suicidal"
    - "want to kill myself"
    - "can't go on"
    - "no reason to live"
    - "overdose on"
    - "slit my wrists"
    - "hang myself"
    - "cutting myself"
    - "hurt myself"
    - "self harm"
    - "self-harm"
    # Add your own keywords below:

  criminal_keywords:
    - "how to make a bomb"
    - "how to build a bomb"
    - "how to poison someone"
    - "how to make poison"
    - "how to kill someone"
    - "how to get away with murder"
    - "how to make meth"
    - "how to synthesize fentanyl"
    - "how to cook meth"
    - "child pornography"
    - "explosive device"
    - "how to make ricin"
    # Add your own keywords below:

  self_harm_keyword_score: 0.7
  criminal_keyword_score: 0.6
  intent_pattern_score: 0.7

  context_reducers:
    - "laughing"
    - "of laughter"
    - "of embarrassment"
    - "of boredom"
    - "in the game"
    - "in minecraft"
    - "for my character"
    - "in fiction"
    - "the villain"
    - "the character"
    - "in a novel"
    - "in a movie"
    - "in a book"
    - "in a story"
    - "my character"
    - "warning signs"
    - "prevent"
    - "prevention"
    - "how to help"
    - "help someone"
    - "help a friend"
    - "awareness"

trajectory:
  window_size: 5
  spike_delta: 0.35

escalation:
  rate_limit_max: 3
  rate_limit_window_hours: 1
  webhooks:
    slack_url: ""
    discord_url: ""
    pagerduty_routing_key: ""
"""

_DEFAULT_ENV_CONTENT = """\
# HumaneProxy environment variables.
# Rename this file to .env and fill in your values.
#
# NOTE: HumaneProxy does NOT auto-load .env files. Load it into your
# environment before `humane-proxy start`, e.g.:
#   set -a; source .env; set +a        (bash/zsh)
# or inject it via your process manager / docker --env-file.

# Upstream LLM (required for the reverse proxy server only).
LLM_API_KEY=
LLM_API_URL=

# Stage-3 reasoning providers (optional — enables the third pipeline stage).
# "auto" provider detection checks OPENAI_API_KEY first, then GROQ_API_KEY.
# OPENAI_API_KEY=sk-...
# GROQ_API_KEY=gsk_...

# Admin API / HTTP MCP bearer token (optional).
# Secures the /admin REST endpoints and HTTP-mode MCP tool access.
# HUMANE_PROXY_ADMIN_KEY=your-secret-token

# Optional overrides (uncomment to use):
# HUMANE_PROXY_PORT=8000
# HUMANE_PROXY_RISK_THRESHOLD=0.7
# HUMANE_PROXY_SLACK_URL=https://hooks.slack.com/services/...
# HUMANE_PROXY_DISCORD_URL=https://discord.com/api/webhooks/...
# HUMANE_PROXY_PAGERDUTY_KEY=your-routing-key
# HUMANE_PROXY_DB_PATH=/path/to/escalations.db
"""


@click.group()
def main() -> None:
    """HumaneProxy — AI safety middleware that protects humans."""
    pass


@main.command()
def init() -> None:
    """Scaffold humane_proxy.yaml and .env.example in the current directory."""
    cwd = Path.cwd()
    created: list[str] = []

    yaml_path = cwd / "humane_proxy.yaml"
    if yaml_path.exists():
        click.echo(f"  [WARN] {yaml_path.name} already exists, skipping.")
    else:
        yaml_path.write_text(_DEFAULT_YAML_CONTENT, encoding="utf-8")
        created.append(yaml_path.name)

    env_path = cwd / ".env.example"
    if env_path.exists():
        click.echo(f"  [WARN] {env_path.name} already exists, skipping.")
    else:
        env_path.write_text(_DEFAULT_ENV_CONTENT, encoding="utf-8")
        created.append(env_path.name)

    if created:
        click.echo(f"\n  [OK] Created: {', '.join(created)}")
        click.echo("\n  Next steps:")
        click.echo("    1. Copy .env.example -> .env and fill in your LLM_API_KEY / LLM_API_URL")
        click.echo("    2. Load it into your environment (HumaneProxy does not auto-load .env):")
        click.echo("         set -a; source .env; set +a")
        click.echo("    3. Edit humane_proxy.yaml to customise thresholds & keywords")
        click.echo("    4. Run: humane-proxy start")
    else:
        click.echo("\n  [INFO] Nothing to create — files already exist.")


def _diagnose() -> dict:
    """Summarize the active protection posture without loading any model.

    Returns a dict describing resolved stages, Stage-2 backend
    availability, Stage-3 provider readiness, storage, and alerting —
    shared by the startup nudges and ``hp doctor``.
    """
    import os

    from humane_proxy.config import get_config
    from humane_proxy.classifiers.pipeline import (
        SafetyPipeline,
        stage2_backend_available,
    )

    cfg = get_config()
    pipeline_cfg = cfg.get("pipeline", {})
    raw_stages = pipeline_cfg.get("enabled_stages", "auto")
    stages = SafetyPipeline._resolve_stages(raw_stages)

    stage2_ok = stage2_backend_available()

    stage3_cfg = cfg.get("stage3", {})
    provider = stage3_cfg.get("provider", "auto")
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    stage3_provider_ready = (
        provider not in ("auto", "none") or has_openai or has_groq
    )

    webhooks = cfg.get("escalation", {}).get("webhooks", {}) or {}
    email_cfg = webhooks.get("email") or {}
    # A channel counts only when it carries a real value — the default
    # config ships empty url strings and an empty email block.
    alert_channels = [
        name for name, val in (
            ("slack", webhooks.get("slack_url")),
            ("discord", webhooks.get("discord_url")),
            ("teams", webhooks.get("teams_url")),
            ("pagerduty", webhooks.get("pagerduty_routing_key")),
            ("email", email_cfg.get("host") and email_cfg.get("to")),
        ) if val
    ]

    return {
        "enabled_stages": stages,
        "stage2_backend_available": stage2_ok,
        "stage3_provider": provider,
        "stage3_provider_ready": stage3_provider_ready,
        "has_openai_key": has_openai,
        "has_groq_key": has_groq,
        "storage_backend": cfg.get("storage", {}).get("backend", "sqlite"),
        "trajectory_backend": cfg.get("trajectory", {}).get("backend", "memory"),
        "alert_channels": alert_channels,
        "startup_warnings": bool(cfg.get("startup_warnings", True)),
    }


def _print_setup_warnings(err: bool = False) -> None:
    """Print fail-safe nudges when stronger protection is available but off.

    Gated by ``startup_warnings`` (default true). Never blocks startup.
    Pass ``err=True`` to route to stderr (required for stdio MCP, whose
    stdout carries the protocol).
    """
    d = _diagnose()
    if not d["startup_warnings"]:
        return

    stages = d["enabled_stages"]

    if 2 not in stages and not d["stage2_backend_available"]:
        click.echo(
            "  [WARN] Stage 2 (semantic embeddings) is OFF — install a "
            "backend for far stronger detection:\n"
            "         pip install humane-proxy[onnx]   (no PyTorch, ~5ms/msg)",
            err=err,
        )
    elif 2 not in stages and d["stage2_backend_available"]:
        click.echo(
            "  [WARN] Stage 2 backend is installed but Stage 2 is OFF — set "
            "pipeline.enabled_stages: \"auto\" (or [1,2]) to enable it.",
            err=err,
        )

    if 3 not in stages:
        if d["stage3_provider_ready"]:
            click.echo(
                "  [INFO] Stage 3 (reasoning LLM) is available — add 3 to "
                "pipeline.enabled_stages for maximum recall (benchmarked 92%).",
                err=err,
            )
        else:
            click.echo(
                "  [INFO] Stage 3 (reasoning LLM) is OFF — set OPENAI_API_KEY "
                "(free OpenAI Moderation) or GROQ_API_KEY, then enable stage 3.",
                err=err,
            )

    if not d["alert_channels"]:
        click.echo(
            "  [INFO] No alert channels configured — operators will not be "
            "notified on escalation. See escalation.webhooks in your config.",
            err=err,
        )
    click.echo("", err=err)


@main.command()
@click.option("--host", default=None, help="Bind host (default: from config)")
@click.option("--port", "-p", default=None, type=int, help="Bind port (default: from config)")
@click.option("--reload/--no-reload", default=None, help="Auto-reload on changes")
def start(host: str | None, port: int | None, reload: bool | None) -> None:
    """Start the HumaneProxy proxy server."""
    click.echo(_BANNER)

    from humane_proxy.config import get_config

    cfg = get_config()
    server_cfg = cfg.get("server", {})

    final_host = host or server_cfg.get("host", "0.0.0.0")
    final_port = port or server_cfg.get("port", 8000)
    final_reload = reload if reload is not None else server_cfg.get("reload", False)

    click.echo(f"  Starting HumaneProxy on {final_host}:{final_port}")
    if final_reload:
        click.echo("  [INFO] Auto-reload enabled")
    click.echo("")

    _print_setup_warnings()

    import uvicorn

    uvicorn.run(
        "humane_proxy.middleware.interceptor:app",
        host=final_host,
        port=final_port,
        reload=final_reload,
    )


@main.command()
def doctor() -> None:
    """Diagnose the active protection posture (stages, backends, alerting)."""
    click.echo(_BANNER)
    d = _diagnose()

    def line(ok: bool, label: str, detail: str) -> None:
        tag = "[OK]  " if ok else "[WARN]"
        click.echo(f"  {tag} {label:<22} {detail}")

    stages = d["enabled_stages"]
    click.echo("  Protection posture\n")

    line(1 in stages, "Stage 1 heuristics", "always on" if 1 in stages else "OFF")
    line(
        2 in stages, "Stage 2 embeddings",
        "enabled" if 2 in stages else (
            "backend installed but OFF" if d["stage2_backend_available"]
            else "OFF — pip install humane-proxy[onnx]"
        ),
    )
    line(
        3 in stages, "Stage 3 reasoning",
        "enabled" if 3 in stages else (
            "available (provider ready) but OFF" if d["stage3_provider_ready"]
            else "OFF — set OPENAI_API_KEY or GROQ_API_KEY"
        ),
    )
    line(
        3 not in stages or d["stage3_provider_ready"], "Stage 3 provider",
        f"{d['stage3_provider']} "
        f"(openai_key={'yes' if d['has_openai_key'] else 'no'}, "
        f"groq_key={'yes' if d['has_groq_key'] else 'no'})",
    )
    line(True, "Storage backend", d["storage_backend"])
    line(True, "Trajectory backend", d["trajectory_backend"])
    line(
        bool(d["alert_channels"]), "Alert channels",
        ", ".join(d["alert_channels"]) if d["alert_channels"]
        else "none configured — operators won't be notified",
    )

    click.echo("")
    if 3 in stages:
        click.echo("  [OK]   Full 3-stage cascade active.")
    elif 2 in stages:
        click.echo("  [INFO] Stages 1+2 active. Enable Stage 3 for maximum recall (benchmarked 92%).")
    else:
        click.echo("  [WARN] Stage 1 only — install an embedding backend for far stronger detection.")
    click.echo("")


@main.command()
@click.argument("text")
@click.option("--session", "-s", default="cli", help="Session ID for trajectory tracking")
def check(text: str, session: str) -> None:
    """Quick safety check on TEXT from the terminal."""
    from humane_proxy import HumaneProxy

    proxy = HumaneProxy()
    result = proxy.check(text, session_id=session)

    category = result.get("category", "safe")

    if category == "self_harm":
        icon = "[FLAGGED]"
        label = "self_harm"
    elif category == "criminal_intent" and not result["safe"]:
        icon = "[FLAGGED]"
        label = "criminal_intent"
    elif result["safe"]:
        icon = "[SAFE]"
        label = ""
    else:
        icon = "[FLAGGED]"
        label = f"{category}"

    click.echo(f"\n  {icon} {label}".rstrip())
    click.echo(f"  Score   : {result['score']}")
    click.echo(f"  Category: {category}")
    if result["triggers"]:
        click.echo(f"  Triggers: {', '.join(result['triggers'])}")
    else:
        click.echo("  Triggers: (none)")
    click.echo("")


@main.command()
def version() -> None:
    """Print HumaneProxy version."""
    from humane_proxy import __version__
    click.echo(f"HumaneProxy v{__version__}")


@main.command()
@click.option("--category", "-c", default=None,
              help="Filter by category: self_harm | criminal_intent")
@click.option("--limit", "-n", default=20, type=int, help="Max records (default 20)")
@click.option("--session", "-s", default=None, help="Filter by session ID")
def escalations(category: str | None, limit: int, session: str | None) -> None:
    """List recent escalation events from the audit log."""
    from humane_proxy.storage.factory import get_store

    store = get_store()
    rows = store.query(category=category, session_id=session, limit=limit, offset=0)

    if not rows:
        click.echo("  [INFO] No escalations found.")
        return

    click.echo(f"\n  {'ID':<6} {'Session':<28} {'Category':<18} {'Score':<7} {'When'}")
    click.echo("  " + "-" * 75)
    for rec in rows:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(rec["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        cat = rec["category"]
        click.echo(f"  {rec['id']:<6} {rec['session_id']:<28} {cat:<18} {rec['risk_score']:.2f}  {dt}")
    click.echo("")


@main.command()
@click.argument("session_id")
def session(session_id: str) -> None:
    """Show risk trajectory and escalation history for a session."""
    from humane_proxy.storage.factory import get_store

    store = get_store()
    rows = store.query(session_id=session_id, limit=500, offset=0)

    click.echo(f"\n  Session: {session_id}")
    click.echo(f"  Escalation count: {len(rows)}\n")

    if not rows:
        click.echo("  [INFO] No escalations recorded for this session.")
        return

    for rec in rows:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(rec["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        cat = rec["category"]
        trigs = rec.get("triggers", [])
        if isinstance(trigs, str):
            import json
            try:
                trigs = json.loads(trigs)
            except Exception:
                trigs = []
        click.echo(f"  {dt}  score={rec['risk_score']:.2f}  category={cat}")
        if trigs:
            click.echo(f"     triggers: {', '.join(trigs[:3])}")

    click.echo("")


@main.command()
@click.option("--dataset", "-d", required=True,
              type=click.Path(exists=True),
              help="Path to JSON evaluation dataset.")
@click.option("--ci", is_flag=True, default=False,
              help="CI mode: exit with code 1 if any test case fails.")
@click.option("--stages", default="1,2",
              help="Comma-separated pipeline stages to run. Default: '1,2'")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show every test case. Default: full table only for small "
                   "datasets; large runs list failures only.")
@click.option("--json-out", type=click.Path(), default=None,
              help="Write metrics and per-case results to a JSON file.")
@click.option("--delay", type=float, default=0.0,
              help="Seconds to wait between messages. Use to stay under a "
                   "Stage-3 provider's rate limit (e.g. free tiers).")
def benchmark(dataset: str, ci: bool, stages: str, verbose: bool,
              json_out: str | None, delay: float) -> None:
    """Run an evaluation dataset through the safety pipeline and report results.

    The dataset must be a JSON file containing an array of objects, each with
    'message' (str) and 'expected' (str: safe | self_harm | criminal_intent).

    Example::

        hp benchmark --dataset evals/sample.json
        hp benchmark --dataset evals/sample.json --ci
    """
    import asyncio
    import json
    import time

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        _RICH = True
    except ImportError:
        _RICH = False

    # --- Load dataset ---
    with open(dataset, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if not isinstance(cases, list) or not cases:
        click.echo("  [ERROR] Dataset must be a non-empty JSON array.")
        sys.exit(1)

    for i, case in enumerate(cases):
        if "message" not in case or "expected" not in case:
            click.echo(f"  [ERROR] Entry {i} missing 'message' or 'expected' field.")
            sys.exit(1)

    click.echo(_BANNER)
    click.echo(f"  [*] Benchmark: {dataset}")
    click.echo(f"  [*] Test cases: {len(cases)}\n")

    # --- Run pipeline ---
    import os
    os.environ["HUMANE_PROXY_ENABLED_STAGES"] = stages
    
    from humane_proxy import HumaneProxy
    proxy = HumaneProxy()

    results = []

    async def _run_all():
        import asyncio as _asyncio
        for i, case in enumerate(cases):
            if delay > 0 and i > 0:
                await _asyncio.sleep(delay)
            msg = case["message"]
            expected = case["expected"]
            t0 = time.perf_counter()
            result = await proxy.check_async(msg, session_id=f"bench-{i}")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            actual = result.get("category", "safe")
            # Treat safe=True results as "safe" regardless of category field
            if result.get("safe", True) and actual == "safe":
                actual = "safe"
            elif result.get("safe", True):
                actual = "safe"

            passed = actual == expected
            results.append({
                "message": msg,
                "expected": expected,
                "actual": actual,
                "passed": passed,
                "score": result.get("score", 0.0),
                "stage": result.get("stage_reached", 1),
                "latency_ms": elapsed_ms,
            })

    from humane_proxy._profiling import ResourceSampler, capture_environment

    environment = capture_environment()
    wall_t0 = time.perf_counter()
    with ResourceSampler() as sampler:
        asyncio.run(_run_all())
    wall_seconds = time.perf_counter() - wall_t0
    resource_usage = sampler.stats()

    # --- Compute metrics ---
    categories = ["safe", "self_harm", "criminal_intent"]
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = total - passed_count

    def _percentiles(values: list[float]) -> dict:
        s = sorted(values)
        n = len(s)
        if n == 0:
            return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
        return {
            "avg": sum(s) / n,
            "p50": s[n // 2],
            "p95": s[min(n - 1, int(n * 0.95))],
            "min": s[0],
            "max": s[-1],
        }

    lat = _percentiles([r["latency_ms"] for r in results])
    avg_latency, min_latency, max_latency = lat["avg"], lat["min"], lat["max"]
    p50_latency, p95_latency = lat["p50"], lat["p95"]
    throughput = total / wall_seconds if wall_seconds > 0 else 0.0

    # Latency grouped by the stage that produced each verdict, so the cost
    # of stage-1-only vs escalating to stage 2/3 is visible separately.
    latency_by_stage = {
        str(s): _percentiles([r["latency_ms"] for r in results if r["stage"] == s])
        for s in sorted({r["stage"] for r in results})
    }

    # Binary safe/unsafe metrics. Category confusion between the two harm
    # classes still counts as detection here; a safety proxy's first job
    # is not letting harm through, and its second is not over-blocking.
    unsafe_expected = [r for r in results if r["expected"] != "safe"]
    safe_expected = [r for r in results if r["expected"] == "safe"]
    detected = sum(1 for r in unsafe_expected if r["actual"] != "safe")
    false_positives = sum(1 for r in safe_expected if r["actual"] != "safe")
    harm_recall = detected / len(unsafe_expected) if unsafe_expected else None
    false_positive_rate = false_positives / len(safe_expected) if safe_expected else None

    # Which stage produced the final verdict, per message.
    stage_counts: dict[int, int] = {}
    for r in results:
        stage_counts[r["stage"]] = stage_counts.get(r["stage"], 0) + 1

    # Per-category precision/recall/F1
    cat_metrics = {}
    for cat in categories:
        tp = sum(1 for r in results if r["expected"] == cat and r["actual"] == cat)
        fp = sum(1 for r in results if r["expected"] != cat and r["actual"] == cat)
        fn = sum(1 for r in results if r["expected"] == cat and r["actual"] != cat)
        tn = sum(1 for r in results if r["expected"] != cat and r["actual"] != cat)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        cat_metrics[cat] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    accuracy = passed_count / total if total > 0 else 0.0

    # --- Display results ---
    if _RICH:
        import io
        # Force UTF-8 output to avoid Windows cp1252 encoding errors.
        # Falls back to a plain Console when stdout has no raw buffer
        # (e.g. under test runners that substitute sys.stdout).
        utf8_stdout = None
        try:
            utf8_stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
        except (AttributeError, io.UnsupportedOperation, ValueError):
            pass
        console = Console(file=utf8_stdout) if utf8_stdout else Console()

        # Individual results table. Large runs collapse to failures only
        # unless --verbose is passed.
        show_all = verbose or total <= 40
        shown = results if show_all else [r for r in results if not r["passed"]][:25]
        title = "Test Results" if show_all else (
            f"Failures (first {len(shown)} of {failed_count} — use --verbose for all cases)"
        )

        if shown:
            detail_table = Table(title=title, show_lines=True)
            detail_table.add_column("#", style="dim", width=4)
            detail_table.add_column("Message", max_width=50)
            detail_table.add_column("Expected", style="cyan")
            detail_table.add_column("Actual", style="cyan")
            detail_table.add_column("Score", justify="right")
            detail_table.add_column("Latency", justify="right")
            detail_table.add_column("Result", justify="center")

            for i, r in enumerate(shown):
                result_text = Text("PASS", style="green bold") if r["passed"] else Text("FAIL", style="red bold")
                actual_style = "green" if r["passed"] else "red bold"
                detail_table.add_row(
                    str(i + 1),
                    r["message"][:50],
                    r["expected"],
                    Text(r["actual"], style=actual_style),
                    f"{r['score']:.2f}",
                    f"{r['latency_ms']:.1f}ms",
                    result_text,
                )

            console.print(detail_table)
            console.print()

        # Per-category metrics table
        metrics_table = Table(title="Per-Category Metrics")
        metrics_table.add_column("Category", style="cyan bold")
        metrics_table.add_column("TP", justify="right")
        metrics_table.add_column("FP", justify="right")
        metrics_table.add_column("FN", justify="right")
        metrics_table.add_column("Precision", justify="right")
        metrics_table.add_column("Recall", justify="right")
        metrics_table.add_column("F1", justify="right")

        for cat in categories:
            m = cat_metrics[cat]
            metrics_table.add_row(
                cat,
                str(m["tp"]),
                str(m["fp"]),
                str(m["fn"]),
                f"{m['precision']:.1%}",
                f"{m['recall']:.1%}",
                f"{m['f1']:.1%}",
            )

        console.print(metrics_table)
        console.print()

        # Summary panel
        acc_style = "green bold" if accuracy >= 0.9 else ("yellow bold" if accuracy >= 0.7 else "red bold")
        summary_text = Text()
        summary_text.append(f"Accuracy: {accuracy:.1%}", style=acc_style)
        summary_text.append(f"  |  Passed: {passed_count}/{total}")
        summary_text.append(f"  |  Failed: {failed_count}")
        if harm_recall is not None:
            summary_text.append(f"\nHarm detection rate: {harm_recall:.1%} ({detected}/{len(unsafe_expected)} unsafe prompts flagged)")
        if false_positive_rate is not None:
            summary_text.append(f"\nFalse positive rate: {false_positive_rate:.1%} ({false_positives}/{len(safe_expected)} safe prompts flagged)")
        stage_parts = "  ".join(
            f"stage {s}: {n} ({n / total:.0%})" for s, n in sorted(stage_counts.items())
        )
        summary_text.append(f"\nVerdict stage — {stage_parts}")
        summary_text.append(
            f"\nLatency — avg: {avg_latency:.1f}ms  p50: {p50_latency:.1f}ms  "
            f"p95: {p95_latency:.1f}ms  min: {min_latency:.1f}ms  max: {max_latency:.1f}ms"
        )
        summary_text.append(
            f"\nThroughput: {throughput:.1f} msg/s over {wall_seconds:.2f}s wall"
        )
        if resource_usage is not None:
            summary_text.append(
                f"\nResources — CPU avg {resource_usage['cpu_percent_mean']}% "
                f"(peak {resource_usage['cpu_percent_peak']}%, {resource_usage['cpu_count']} cores)  "
                f"peak RSS {resource_usage['peak_rss_mb']} MB"
            )
        summary_text.append(
            f"\nMachine — {environment['processor']}  "
            f"{environment['cpu_count']} cores  "
            f"{environment.get('ram_total_gb', '?')} GB RAM  "
            f"Python {environment['python']}"
        )

        panel_style = "green" if failed_count == 0 else "red"
        console.print(Panel(summary_text, title="Benchmark Summary", border_style=panel_style))

        if utf8_stdout is not None:
            # Release the raw buffer without closing it — later click.echo
            # calls still write to the original stdout.
            utf8_stdout.flush()
            utf8_stdout.detach()

    else:
        # Fallback plain text output
        show_all = verbose or total <= 40
        shown = results if show_all else [r for r in results if not r["passed"]][:25]
        if not show_all:
            click.echo(f"  Failures (first {len(shown)} of {failed_count} — use --verbose for all cases):")
        else:
            click.echo("  Results:")
        click.echo(f"  {'#':<4} {'Expected':<18} {'Actual':<18} {'Score':<8} {'Latency':<10} {'Result'}")
        click.echo("  " + "-" * 80)
        for i, r in enumerate(shown):
            status = "PASS" if r["passed"] else "FAIL"
            click.echo(
                f"  {i+1:<4} {r['expected']:<18} {r['actual']:<18} "
                f"{r['score']:<8.2f} {r['latency_ms']:<10.1f} {status}"
            )

        click.echo(f"\n  Accuracy: {accuracy:.1%} ({passed_count}/{total})")
        if harm_recall is not None:
            click.echo(f"  Harm detection rate: {harm_recall:.1%} ({detected}/{len(unsafe_expected)})")
        if false_positive_rate is not None:
            click.echo(f"  False positive rate: {false_positive_rate:.1%} ({false_positives}/{len(safe_expected)})")
        click.echo("  Verdict stage — " + "  ".join(
            f"stage {s}: {n} ({n / total:.0%})" for s, n in sorted(stage_counts.items())
        ))
        click.echo(
            f"  Latency — avg: {avg_latency:.1f}ms  p50: {p50_latency:.1f}ms  "
            f"p95: {p95_latency:.1f}ms  min: {min_latency:.1f}ms  max: {max_latency:.1f}ms"
        )
        click.echo(f"  Throughput: {throughput:.1f} msg/s over {wall_seconds:.2f}s wall")
        if resource_usage is not None:
            click.echo(
                f"  Resources — CPU avg {resource_usage['cpu_percent_mean']}% "
                f"(peak {resource_usage['cpu_percent_peak']}%, {resource_usage['cpu_count']} cores)  "
                f"peak RSS {resource_usage['peak_rss_mb']} MB"
            )
        click.echo(
            f"  Machine — {environment['processor']}  {environment['cpu_count']} cores  "
            f"{environment.get('ram_total_gb', '?')} GB RAM  Python {environment['python']}"
        )

        for cat in categories:
            m = cat_metrics[cat]
            click.echo(
                f"  {cat}: precision={m['precision']:.1%} recall={m['recall']:.1%} f1={m['f1']:.1%} "
                f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})"
            )

    click.echo("")

    if json_out:
        payload = {
            "dataset": dataset,
            "stages": stages,
            "total": total,
            "accuracy": accuracy,
            "harm_recall": harm_recall,
            "false_positive_rate": false_positive_rate,
            "stage_counts": {str(k): v for k, v in sorted(stage_counts.items())},
            "wall_seconds": round(wall_seconds, 3),
            "throughput_msgs_per_sec": round(throughput, 1),
            "latency_ms": {k: round(v, 3) for k, v in lat.items()},
            "latency_by_stage_ms": {
                s: {k: round(v, 3) for k, v in p.items()}
                for s, p in latency_by_stage.items()
            },
            "environment": environment,
            "resource_usage": resource_usage,
            "categories": cat_metrics,
            "results": results,
        }
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        click.echo(f"  [INFO] Metrics written to {json_out}\n")

    if ci and failed_count > 0:
        click.echo(f"  [FAIL] CI mode: {failed_count} test case(s) failed. Exiting with code 1.")
        sys.exit(1)

    if failed_count == 0:
        click.echo("  [PASS] All test cases passed!")
    else:
        click.echo(f"  [WARN] {failed_count} test case(s) failed.")


@main.command("mcp-serve")
@click.option("--transport", "-t", default="stdio",
              type=click.Choice(["stdio", "http"]),
              help="Transport mode: stdio (default) or http")
@click.option("--host", default="127.0.0.1", help="HTTP bind host (default: 127.0.0.1)")
@click.option("--port", "-p", default=3000, type=int, help="HTTP bind port (default: 3000)")
def mcp_serve(transport: str, host: str, port: int) -> None:
    """Start the MCP server (requires [mcp] extra).

    Use --transport stdio (default) for local integration with agents.
    Use --transport http for HTTP access. Set HUMANE_PROXY_ADMIN_KEY
    before exposing HTTP MCP beyond localhost.
    """
    # Nudges to stderr — stdio MCP's stdout carries the protocol.
    _print_setup_warnings(err=True)
    try:
        if transport == "http":
            from humane_proxy.mcp_server import serve_http
            click.echo(f"  Starting HumaneProxy MCP server (HTTP) on {host}:{port}...")
            serve_http(host=host, port=port)
        else:
            from humane_proxy.mcp_server import serve
            click.echo("  Starting HumaneProxy MCP server (stdio)...", err=True)
            serve()
    except RuntimeError as exc:
        click.echo(f"\n  [ERROR] {exc}\n", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()