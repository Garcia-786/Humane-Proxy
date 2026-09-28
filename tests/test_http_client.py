"""Tests for the shared process-level httpx.AsyncClient."""

import asyncio

from humane_proxy import http_client


class TestSharedClient:
    def test_same_instance_within_one_loop(self):
        async def run():
            a = http_client.get_async_client()
            b = http_client.get_async_client()
            assert a is b
            await http_client.aclose()

        asyncio.run(run())

    def test_new_client_after_close(self):
        async def run():
            a = http_client.get_async_client()
            await http_client.aclose()
            b = http_client.get_async_client()
            assert a is not b
            assert a.is_closed and not b.is_closed
            await http_client.aclose()

        asyncio.run(run())

    def test_new_client_for_new_loop(self):
        """A client created on a dead loop must not be reused (CLI /
        benchmark contexts call asyncio.run repeatedly)."""
        holder = {}

        async def first():
            holder["a"] = http_client.get_async_client()

        async def second():
            holder["b"] = http_client.get_async_client()
            await http_client.aclose()

        asyncio.run(first())
        asyncio.run(second())
        assert holder["a"] is not holder["b"]
