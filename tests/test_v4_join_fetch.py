"""Protocol v4: the root's fetch of an epoch's proposal documents
(join_step.ProposalFetcher, driven by join_step.gather), against real
loopback servers answering the proposal synapse.

Every document is fetched at once in one event loop; a box that hangs,
refuses, redirects or sends too much costs only its own proposal, which is
then "unfetched" and waits on chain. Loopback only.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import bittensor as bt
import pytest
from aiohttp import web

from fugal_subnet import chain_proposals as cp
from fugal_subnet import join_step
from tests import menu_fixtures as fx

TIMEOUT_S = 2.0
BUDGET_S = 6.0


class _Boxes:
    """Loopback proposer boxes, one port each, served from one thread."""

    def __init__(self, behaviours: dict):
        self.behaviours, self.ports = behaviours, {}
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._serve, daemon=True).start()
        assert self._ready.wait(10)

    def _handler(self, name):
        kind, doc = self.behaviours[name]

        async def h(request):
            body = await request.json()
            if kind == "hang":
                await asyncio.sleep(60)
            if kind == "redirect":
                raise web.HTTPTemporaryRedirect(f"http://127.0.0.1:{self.ports['ok-0']}"
                                                "/FugalProposalSynapse")
            if kind == "endless":
                resp = web.StreamResponse()
                await resp.prepare(request)
                while True:
                    await resp.write(b"x" * 8192)
            body["document"] = doc.decode()
            if kind == "oversized":
                body["padding"] = "y" * (join_step.RESPONSE_MAX_BYTES + 1)
            return web.json_response(body)
        return h

    def _serve(self):
        asyncio.set_event_loop(self._loop)

        async def start():
            for name, (kind, _doc) in self.behaviours.items():
                if kind == "refuse":
                    continue
                app = web.Application()
                app.router.add_post("/FugalProposalSynapse", self._handler(name))
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                self.ports[name] = site._server.sockets[0].getsockname()[1]
        self._loop.run_until_complete(start())
        self._ready.set()
        self._loop.run_forever()


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def dendrite(monkeypatch):
    from bittensor.utils import networking

    monkeypatch.setattr(networking, "get_external_ip", lambda: "203.0.113.9")
    return bt.Dendrite(wallet=bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic()))


def _world(behaviours: dict, dendrite):
    """(read, fetcher, {hotkey: raw document}) for boxes named by hotkey."""
    docs = {hk: fx.document(f"lab-c/model-{i}", "high", proposer=hk)[1]
            for i, hk in enumerate(behaviours)}
    boxes = _Boxes({hk: (kind, docs[hk]) for hk, kind in behaviours.items()})
    axons = {hk: bt.AxonInfo(version=1, ip="127.0.0.1",
                             port=boxes.ports.get(hk) or _free_port(), ip_type=4,
                             hotkey=hk, coldkey=f"ck-{hk}") for hk in behaviours}
    base = fx.wide_state(epoch=101)
    read = cp.ChainRead(epoch=base.epoch, block=base.epoch * 360 - cp.JOIN_READ_MARGIN_BLOCKS,
                        block_hash="ab" * 32,
                        proposals=tuple(sorted((hk, cp.proposal_hash(d)) for hk, d in docs.items())),
                        coldkey_of={hk: f"ck-{hk}" for hk in docs},
                        serving=frozenset(docs) | {f"other-{i}" for i in range(10)},
                        reveal=("e00000099", "cd" * 32))
    fetcher = join_step.ProposalFetcher(dendrite, axons, timeout_s=TIMEOUT_S)
    return base, read, fetcher, docs


def _listings(docs: dict) -> dict:
    return {cp.proposal_hash(raw): fx.listing(cp.parse_document(raw).provider_order)
            for raw in docs.values()}


@pytest.mark.parametrize("n", [2, 3, 8, 16])
def test_every_document_of_an_epoch_is_fetched(n, dendrite):
    base, read, fetcher, docs = _world({f"hk-{i:03d}": "ok" for i in range(n)}, dendrite)
    got, _ = join_step.gather(read, fetch_document=fetcher, fetch_listing=lambda m: None,
                              budget_s=BUDGET_S)
    assert got == {cp.proposal_hash(d): d for d in docs.values()}
    # The fetcher stays usable afterwards, one call at a time.
    hk, p = read.proposals[0]
    assert fetcher(hk, p) == docs[hk]


def test_a_failing_box_costs_only_its_own_proposal(dendrite):
    behaviours = {f"ok-{i}": "ok" for i in range(5)}
    behaviours.update({"hang": "hang", "refuse": "refuse", "endless": "endless",
                       "oversized": "oversized", "redirect": "redirect"})
    base, read, fetcher, docs = _world(behaviours, dendrite)
    t0 = time.monotonic()
    got, _ = join_step.gather(read, fetch_document=fetcher, fetch_listing=lambda m: None,
                              budget_s=BUDGET_S)
    assert time.monotonic() - t0 < BUDGET_S
    assert got == {cp.proposal_hash(docs[hk]): docs[hk] for hk in docs if hk.startswith("ok-")}
    # The admitted set is the admission order's choice among what was fetched,
    # the same as with every good document handed over directly; the rest are
    # unfetched and wait on chain.
    delta = cp.admit(base, read, got, _listings(docs), fx.ALLOW, fx.PARAMS)
    direct = cp.admit(base, read, {cp.proposal_hash(docs[hk]): docs[hk] for hk in docs
                                   if hk.startswith("ok-")}, _listings(docs), fx.ALLOW, fx.PARAMS)
    assert delta == direct
    verdicts = {c.hotkey: c.verdict for c in delta.considered}
    assert {verdicts[hk] for hk in ("hang", "refuse", "endless", "oversized", "redirect")} == {
        cp.UNFETCHED}
    assert sum(v == cp.JOINED for v in verdicts.values()) == 2


def test_with_every_document_fetched_the_admission_matches_the_order(dendrite):
    base, read, fetcher, docs = _world({f"hk-{i:03d}": "ok" for i in range(8)}, dendrite)
    got, _ = join_step.gather(read, fetch_document=fetcher, fetch_listing=lambda m: None,
                              budget_s=BUDGET_S)
    delta = cp.admit(base, read, got, _listings(docs), fx.ALLOW, fx.PARAMS)
    order = [c.proposal for c in delta.considered]
    assert order == sorted(order, key=lambda p: cp.place(read.block_hash, p))
    joined = [c.proposal for c in delta.joined()]
    assert joined == order[:2]
    assert cp.UNFETCHED not in {c.verdict for c in delta.considered}


def test_fetches_outstanding_when_the_budget_ends_are_unfetched(dendrite):
    behaviours = {"ok-0": "ok", "ok-1": "ok", "hang": "hang"}
    base, read, _fetcher, docs = _world(behaviours, dendrite)
    slow = join_step.ProposalFetcher(dendrite, _fetcher.axon_of, timeout_s=30.0)
    t0 = time.monotonic()
    got, _ = join_step.gather(read, fetch_document=slow, fetch_listing=lambda m: None,
                              budget_s=1.5)
    assert time.monotonic() - t0 < 1.5 + 1.5
    assert set(got) == {cp.proposal_hash(docs["ok-0"]), cp.proposal_hash(docs["ok-1"])}


def test_an_answer_whose_document_is_not_a_short_string_is_nothing():
    of = join_step.ProposalFetcher.document_of
    assert of(json.dumps({"document": "abc"}).encode()) == b"abc"
    for bad in (None, b"not json", b"[]", json.dumps({"document": 5}).encode(),
                json.dumps({"document": ""}).encode(),
                json.dumps({"document": "a" * 1025}).encode()):
        assert of(bad) is None


def test_a_real_axon_serves_the_fetcher_as_it_serves_a_dendrite(dendrite):
    """The signed request is the dendrite's own: a bittensor axon (as the
    miner runs) verifies it and answers."""
    from fugal_subnet.protocol import FugalProposalSynapse

    def kp():
        return bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())

    wallet = SimpleNamespace(hotkey=kp(), coldkeypub=kp())
    wallet.coldkey = wallet.coldkeypub
    doc = fx.document(proposer="hk")[1]

    def forward(s: FugalProposalSynapse) -> FugalProposalSynapse:
        if s.proposal == cp.proposal_hash(doc):
            s.document = doc.decode()
        return s
    # The axon reads the classes themselves (this file's annotations are strings).
    forward.__annotations__ = {"s": FugalProposalSynapse, "return": FugalProposalSynapse}

    port = _free_port()
    axon = bt.Axon(wallet=wallet, ip="127.0.0.1", port=port, external_ip="127.0.0.1",
                   external_port=port)
    axon.attach(forward_fn=forward)
    axon.start()
    try:
        info = bt.AxonInfo(version=1, ip="127.0.0.1", port=port, ip_type=4,
                           hotkey=wallet.hotkey.ss58_address, coldkey="c")
        fetcher = join_step.ProposalFetcher(dendrite, {"hk": info}, timeout_s=5.0)
        for _ in range(50):                      # the server thread may still be starting
            if fetcher("hk", cp.proposal_hash(doc)) is not None:
                break
            time.sleep(0.1)
        got = fetcher.fetch_many([("hk", cp.proposal_hash(doc)), ("hk", "00" * 32)], 5.0)
        assert got == {("hk", cp.proposal_hash(doc)): doc, ("hk", "00" * 32): None}
    finally:
        axon.stop()


def test_the_validator_fetches_through_the_one_loop_fetcher(dendrite):
    from neurons import validator as V

    mg = SimpleNamespace(hotkeys=["hk"], axons=[bt.AxonInfo(version=1, ip="127.0.0.1", port=9,
                                                            ip_type=4, hotkey="hk",
                                                            coldkey="c")])
    fetch = V._v4_proposal_fetcher(dendrite, SimpleNamespace(metagraph=lambda n, block: mg), 1, 7)
    assert isinstance(fetch, join_step.ProposalFetcher)
    assert set(fetch.axon_of) == {"hk"}
