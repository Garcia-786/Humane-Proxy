# Copyright 2026 Vishisht Mishra (Vishisht16)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared process-level ``httpx.AsyncClient``.

Creating a fresh ``AsyncClient`` per request forces a new TCP + TLS
handshake on every upstream call and defeats HTTP connection pooling.
This module keeps one client per event loop and hands it to the
interceptor, the webhook dispatchers, and the Stage-3 classifiers.
Callers pass their own per-request ``timeout=`` — pooling is shared,
deadlines are not.

The client is bound to the event loop it was created on.  Long-lived
servers use a single loop, so the pool persists for the process; contexts
that spin up short-lived loops (``asyncio.run`` in the CLI, benchmarks)
transparently get a fresh client for each new loop.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger("humane_proxy.http_client")

_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None


def get_async_client() -> httpx.AsyncClient:
    """Return the shared ``AsyncClient`` for the running event loop.

    Must be called from within a running loop.  A new client is created
    the first time and whenever the running loop changes (the previous
    loop's client is unusable once its loop is gone).
    """
    global _client, _client_loop

    loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not loop:
        _client = httpx.AsyncClient()
        _client_loop = loop
    return _client


async def aclose() -> None:
    """Close the shared client (called from the server's lifespan shutdown)."""
    global _client, _client_loop
    if _client is not None and not _client.is_closed:
        try:
            await _client.aclose()
        except Exception:
            logger.debug("Shared httpx client close failed", exc_info=True)
    _client = None
    _client_loop = None
