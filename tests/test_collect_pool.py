"""Proof collection must not let axons that never answer delay the others.

Reproduces the mechanism behind the finney incident (e00026350/e00026351,
fugal_subnet/collect): more than 100 axons that hold their connection without
answering, then one honest miner. With bittensor's default dendrite session
(aiohttp's 100-connection pool) the honest miner is not contacted until the
hanging connections time out. On the validator that is the full 120 s, which is
what the affected miners logged: the request arrived ~127 s after the query.
The collection dendrite reaches it at once.
"""
from __future__ import annotations

import asyncio
import time

import aiohttp
import bittensor as bt
import pytest
from aiohttp import web

from fugal_subnet.collect import COLLECT_CONNECTION_LIMIT, CollectDendrite

HANGING = 110          # more than aiohttp's default limit of 100
TIMEOUT_S = 2.0        # the validator uses 120 s; the mechanism is the same


async def _serve_hanging():
    """A TCP server that accepts every connection and never answers."""
    held = []

    async def handler(reader, writer):
        held.append(writer)
        await asyncio.Event().wait()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1], held


async def _serve_honest():
    """A miner that answers at once."""
    async def ok(_request):
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/proof", ok)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


async def _collect(session: aiohttp.ClientSession, hang_port: int, ok_port: int):
    """Query the hanging axons first and the honest one last, as one burst
    (dendrite.query gathers all axons at once, in UID order). Returns whether
    the honest miner answered, and how many seconds after the burst began."""
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_S)
    t0 = time.monotonic()

    async def post(port: int):
        try:
            async with session.post(f"http://127.0.0.1:{port}/proof", json={},
                                    timeout=timeout) as resp:
                ok = (await resp.json()).get("ok", False)
        except (asyncio.TimeoutError, aiohttp.ClientError):
            ok = False
        return ok, time.monotonic() - t0

    results = await asyncio.gather(*[post(hang_port) for _ in range(HANGING)], post(ok_port))
    return results[-1]


def _run(dendrite_cls):
    async def main():
        server, hang_port, held = await _serve_hanging()
        runner, ok_port = await _serve_honest()
        dendrite = dendrite_cls(wallet=bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic()))
        try:
            return await _collect(await dendrite.session, hang_port, ok_port)
        finally:
            await dendrite.aclose_session()
            for w in held:
                w.close()
            server.close()
            await runner.cleanup()
    return asyncio.run(main())


def test_default_dendrite_pool_delays_the_honest_miner():
    """The bug, reproduced: the honest miner is not contacted until the
    hanging connections have used up their whole timeout."""
    _ok, elapsed = _run(bt.Dendrite)
    assert elapsed >= TIMEOUT_S


def test_collection_dendrite_reaches_the_honest_miner_at_once():
    """The fix: an unbounded pool, so no miner waits behind another."""
    ok, elapsed = _run(CollectDendrite)
    assert ok is True
    assert elapsed < TIMEOUT_S / 2


def test_collection_pool_is_unbounded():
    assert COLLECT_CONNECTION_LIMIT == 0

    async def main():
        d = CollectDendrite(wallet=bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic()))
        try:
            return (await d.session).connector.limit
        finally:
            await d.aclose_session()

    assert asyncio.run(main()) == 0


@pytest.mark.parametrize("cls", [CollectDendrite])
def test_the_validator_collects_with_the_collection_dendrite(cls):
    """Guard against a revert to bt.Dendrite in the validator."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "neurons" / "validator.py").read_text()
    assert f"dendrite = {cls.__name__}(wallet=wallet)" in src
    assert "bt.Dendrite(wallet=wallet)" not in src
