"""HumaneProxy interceptor — FastAPI proxy with full safety pipeline + care response."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from humane_proxy import __version__
from humane_proxy.telemetry import setup_telemetry
from humane_proxy.escalation.local_db import init_db
from humane_proxy.escalation.router import escalate, get_self_harm_response
from json import JSONDecodeError

logger = logging.getLogger("humane_proxy")

_pipeline = None


def _get_pipeline():
    """Return the process-level SafetyPipeline singleton, creating it on first call."""
    global _pipeline
    if _pipeline is None:
        from humane_proxy.config import get_config
        from humane_proxy.classifiers.pipeline import SafetyPipeline
        _pipeline = SafetyPipeline(get_config())
    return _pipeline


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from humane_proxy.config import get_config
    setup_telemetry(get_config())
    init_db()
    _get_pipeline()
    logger.info("[HumaneProxy] Database initialised. Server is ready.")

    # Mount admin router.
    try:
        from humane_proxy.api.admin import router as admin_router
        app.include_router(admin_router)
        logger.info("[HumaneProxy] Admin API mounted at /admin")
    except Exception:
        logger.warning("[HumaneProxy] Admin API could not be loaded.")

    try:
        yield
    finally:
        # Close the shared upstream HTTP connection pool.
        from humane_proxy.http_client import aclose
        await aclose()

        # Flush and shut down the OTel tracer provider cleanly on exit.
        try:
            from opentelemetry import trace
            provider = trace.get_tracer_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
        except Exception:
            pass


app = FastAPI(
    title="HumaneProxy",
    version=__version__,
    description="Lightweight AI safety middleware that protects humans.",
    lifespan=_lifespan,
)


def _resolve_session_id(payload: dict[str, Any], request: Request) -> str:
    """Return the session_id from the payload, falling back to the client IP."""
    return payload.get("session_id") or (
        request.client.host if request.client else "unknown"
    )


def _extract_last_user_message(payload: dict[str, Any]) -> str:
    """Return the content of the last user-role message in the payload."""
    messages: list[dict[str, str]] = payload.get("messages", [])
    if not isinstance(messages, list):
        return ""
        
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
            
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
            
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text", ""), str)
            ]
            return " ".join(text_parts)
            
        return ""
        
    return ""


@app.post("/chat")
async def chat(request: Request) -> JSONResponse:
    """Intercept a chat request, evaluate safety, then forward or respond."""
    try:
        payload: dict[str, Any] = await request.json()
    except (JSONDecodeError, ValueError):
        return JSONResponse(
           status_code=400,
           content={
              "status": "error",
               "message": "Request body must contain valid JSON",
            },
        )
 

    session_id = _resolve_session_id(payload, request)
    user_message = _extract_last_user_message(payload)

    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "No user message found in payload."},
        )

    pipeline = _get_pipeline()
    result = await pipeline.classify(user_message, session_id=session_id)

    if result.should_escalate:
        cls = result.classification

        esc = escalate(
            session_id,
            cls.score,
            cls.triggers,
            cls.category,
            message_hash=result.message_hash,
            stage_reached=cls.stage,
            reasoning=cls.reasoning,
        )

        # Self-harm: return care response instead of generic flagged message.
        if cls.category == "self_harm":
            care = get_self_harm_response(payload)

            if care["mode"] == "block":
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "care_response",
                        "category": "self_harm",
                        "message": care["message"],
                        "escalation": esc,
                    },
                )
            else:
                # Forward mode: inject care context and pass to LLM.
                payload = care["payload"]

        else:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "flagged",
                    "category": cls.category,
                    "message": "Content flagged for review.",
                    "stage_reached": cls.stage,
                    "escalation": esc,
                },
            )

    # Safe (or forward mode) — forward to upstream LLM.
    # Read at request time (not import time) so the vars work no matter
    # when or how they are set — e.g. exported after the app module loads.
    llm_api_url = os.environ.get("LLM_API_URL", "")
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    if not llm_api_url:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "LLM_API_URL is not configured."},
        )

    headers = {
        "Authorization": f"Bearer {llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        # Shared pooled client — connections are reused across requests
        # instead of a fresh TCP+TLS handshake per /chat call.
        from humane_proxy.http_client import get_async_client
        client = get_async_client()
        llm_response = await client.post(
            llm_api_url, headers=headers, json=payload, timeout=30.0
        )
        try:
            body = llm_response.json()
        except (ValueError, TypeError):
            # Never forward raw upstream bodies to clients — they can carry
            # sensitive or unexpected content (issue #33).  Operators can
            # enable DEBUG logging to inspect the payload.
            logger.debug(
                "Upstream non-JSON body (HTTP %d): %s",
                llm_response.status_code, llm_response.text[:500],
            )
            body = {
                "status": "error",
                "message": f"Upstream returned non-JSON (HTTP {llm_response.status_code}).",
            }
        return JSONResponse(status_code=llm_response.status_code, content=body)

    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": f"Upstream LLM unavailable: {type(exc).__name__}: {exc}",
            },
        )