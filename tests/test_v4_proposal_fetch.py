"""The root fetches an epoch's proposal documents at once, each from its
proposer's axon, through the validator's `_v4_proposal_fetcher`.

A real `bt.Dendrite` cannot serve two threads at once: `query` runs its
request on the calling thread's event loop but keeps one aiohttp session for
the instance, bound to the loop that made it, so a second thread's request
fails ("got Future ... attached to a different loop"). The local-chain v4
rehearsal (scripts/dress_rehearsal_v4_seat.py) found every fetch failing
once two proposals were on chain. The fetcher now posts every request in one
event loop with an HTTP session of its own (join_step.ProposalFetcher,
join_step.post_all) and uses the validator's dendrite only to sign: the
dendrite here refuses concurrent `query` use as a real one does, and the
documents come from real loopback servers."""
from __future__ import annotations

import threading
import types

import bittensor as bt
import pytest

from fugal_subnet import chain_proposals as cp
from fugal_subnet import join_step
from tests.test_v4_join_fetch import _Boxes


class _OneLoopDendrite(bt.Dendrite):
    """A real dendrite whose `query` raises when one instance is used from
    two threads at once (as the real one fails) and counts its uses."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._busy = threading.Lock()
        self.queries = 0

    def query(self, *a, **k):
        self.queries += 1
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("got Future attached to a different loop")
        try:
            return super().query(*a, **k)
        finally:
            self._busy.release()


@pytest.fixture
def dendrite(monkeypatch):
    from bittensor.utils import networking

    monkeypatch.setattr(networking, "get_external_ip", lambda: "203.0.113.9")
    return _OneLoopDendrite(wallet=bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic()))


def test_two_proposals_are_fetched_at_once_without_sharing_one_dendrite(dendrite):
    from neurons import validator

    docs = {}
    for hk, model in (("5Fhk-a", "lab-a/model-x"), ("5Fhk-b", "lab-b/model-y")):
        docs[hk] = cp.document_bytes(cp.ProposalDoc(
            model=model, reasoning=None, provider_order=("host-a",), quantization="fp8",
            proposer=hk))
    boxes = _Boxes({hk: ("ok", d) for hk, d in docs.items()})
    mg = types.SimpleNamespace(
        hotkeys=list(docs),
        axons=[bt.AxonInfo(version=1, ip="127.0.0.1", port=boxes.ports[hk], ip_type=4,
                           hotkey=hk, coldkey=f"ck-{hk}") for hk in docs])
    subtensor = types.SimpleNamespace(metagraph=lambda netuid, block=None: mg)
    fetch = validator._v4_proposal_fetcher(dendrite, subtensor, 7, 100)
    assert isinstance(fetch, join_step.ProposalFetcher)
    read = types.SimpleNamespace(
        proposals=tuple((hk, cp.proposal_hash(d)) for hk, d in docs.items()),
        serving=frozenset(docs))
    got, _listings = join_step.gather(read, fetch_document=fetch,
                                      fetch_listing=lambda model: None, budget_s=10.0)
    assert got == {cp.proposal_hash(d): d for d in docs.values()}
    assert dendrite.queries == 0        # the dendrite signs; it does not carry the requests
