from __future__ import annotations

import base64
import hashlib
from types import SimpleNamespace

from fugal_subnet.v2 import chain
from fugal_subnet.v2.chain import historical_chain_resolver, query_head_submissions
from fugal_subnet.v2.commitments import (
    HEAD_COMMITMENT_ID,
    HistoricalCommitmentReceipt,
    commitment_payload,
)


class FakeSubtensor:
    def __init__(self, historical):
        self.historical = historical
        self.weight_call = None
        self.substrate = SimpleNamespace(
            get_chain_finalised_head=lambda: "0x" + f"{200:064x}",
            get_block_number=lambda _hash: 200,
        )

    def get_commitment(self, netuid, uid, block=None):
        return self.historical.get((uid, block), "")

    def get_block_hash(self, block):
        return "0x" + f"{block:064x}"

    def set_weights(self, **kwargs):
        self.weight_call = kwargs
        return True, "finalized"

    def metagraph(self, netuid):
        del netuid
        return SimpleNamespace(n=3, W=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.25]])


class FakeDendrite:
    def __init__(self, responses):
        self.responses = responses

    def query(self, axons, synapse, timeout):
        assert synapse.epoch_id == "v2-100"
        assert timeout == 10
        return self.responses


def test_head_query_binds_boundary_and_late_commitments(monkeypatch):
    artifacts = [b"head-zero", b"head-one", b"head-two"]
    hashes = [hashlib.sha256(item).hexdigest() for item in artifacts]
    payloads = [commitment_payload("head", HEAD_COMMITMENT_ID, item) for item in hashes]
    boundary = {"hk0": (payloads[0], 99)}
    current = {"hk0": (payloads[0], 99), "hk1": (payloads[1], 101)}

    def commitments(_subtensor, _netuid, *, block=None):
        return boundary if block == 100 else current

    monkeypatch.setattr(chain, "get_commitments_with_blocks", commitments)
    subtensor = FakeSubtensor({(0, 99): payloads[0], (1, 101): payloads[1]})
    responses = [
        SimpleNamespace(
            head_npz_b64=base64.b64encode(artifact).decode(),
            model_pool=["provider/a"],
        )
        for artifact in artifacts
    ]
    metagraph = SimpleNamespace(
        n=3,
        hotkeys=["hk0", "hk1", "hk2"],
        axons=[object(), object(), object()],
    )
    result = query_head_submissions(
        dendrite=FakeDendrite(responses),
        subtensor=subtensor,
        metagraph=metagraph,
        network="test",
        netuid=11,
        epoch_id="v2-100",
        benchmark_hash="a" * 64,
        boundary_block=100,
        timeout=10,
    )
    assert [item.uid for item in result.submissions] == [0, 1]
    assert [item.commitment_receipt.block for item in result.submissions] == [99, 101]
    assert result.uncommitted_uids == {2}
    assert result.responding_uids == {0, 1, 2}


def test_chain_resolver_checks_exact_block_hash_and_weight_finality():
    payload = commitment_payload("report", "v2-100", "a" * 64)
    subtensor = FakeSubtensor({(7, 105): payload})
    receipt = HistoricalCommitmentReceipt(
        network="test",
        netuid=11,
        hotkey="builder",
        uid=7,
        block=105,
        block_hash=f"{105:064x}",
        namespace="report",
        epoch_id="v2-100",
        artifact_hash="a" * 64,
        payload=payload,
    )
    assert historical_chain_resolver(subtensor, receipt) == payload
    assert chain.submit_exact_weights(
        subtensor,
        object(),
        netuid=11,
        weights={"9": "0.25", "2": "0.75"},
    ) is True
    assert subtensor.weight_call["uids"] == [2, 9]
    assert subtensor.weight_call["wait_for_finalization"] is True


def test_weight_submission_is_idempotent_when_chain_row_already_matches():
    subtensor = FakeSubtensor({})
    assert chain.submit_exact_weights(
        subtensor,
        object(),
        netuid=11,
        validator_uid=1,
        weights={"0": "0.75", "2": "0.25"},
    ) is False
    assert subtensor.weight_call is None
