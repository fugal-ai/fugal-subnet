from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from fugal_subnet.v2.commitments import (
    CommitmentError,
    capture_historical_receipt,
    collect_historical_receipts,
    commitment_at_block,
    commitment_payload,
    find_historical_receipt,
    set_commitment_finalized,
    submit_commitment_with_receipt,
    verify_historical_receipt,
)


class FakeSubtensor:
    def __init__(self, historical: dict[int, str], latest: str = "wrong"):
        self.historical = historical
        self.latest = latest
        self.queries: list[tuple[int, int, int | None]] = []
        self.set_calls = []
        finalized = max(historical, default=0)
        self.substrate = SimpleNamespace(
            get_chain_finalised_head=lambda: f"0x{finalized:064x}",
            get_block_number=lambda _hash: finalized,
        )

    def get_commitment(self, netuid, uid, block=None):
        self.queries.append((netuid, uid, block))
        return self.latest if block is None else self.historical.get(block, "")

    def get_block_hash(self, block):
        return "0x" + f"{block:064x}"

    def set_commitment(self, wallet, netuid, payload, **kwargs):
        self.set_calls.append((wallet, netuid, payload, kwargs))
        return True, "finalized"


class TransitionSubtensor:
    def __init__(self, transitions=None, current_block=110):
        self.transitions = transitions or {}
        self.current_block = current_block
        self.substrate = SimpleNamespace(
            get_chain_finalised_head=lambda: f"0x{self.current_block:064x}",
            get_block_number=lambda _hash: self.current_block,
        )

    def get_commitment(self, netuid, uid, block=None):
        at = self.current_block if block is None else block
        values = self.transitions.get(uid, {})
        candidates = [height for height in values if height <= at]
        return values[max(candidates)] if candidates else ""

    def get_block_hash(self, block):
        return "0x" + f"{block:064x}"

    def get_current_block(self):
        return self.current_block

    def set_commitment(self, wallet, netuid, payload, **kwargs):
        self.current_block += 1
        self.transitions.setdefault(wallet.uid, {})[self.current_block] = payload
        return True, "finalized"


def test_namespaced_payload_is_bounded_and_domain_separated():
    artifact_hash = "a" * 64
    questions = commitment_payload("questions", "epoch-100", artifact_hash)
    report = commitment_payload("report", "epoch-100", artifact_hash)
    assert questions.startswith("fugal:v2:q:")
    assert report.startswith("fugal:v2:r:")
    assert questions != report
    assert len(questions.encode("ascii")) <= 128


def test_historical_receipt_queries_exact_block_not_latest_state():
    payload = commitment_payload("questions", "epoch-100", "a" * 64)
    subtensor = FakeSubtensor({123: payload}, latest="later-overwrite")
    receipt = capture_historical_receipt(
        subtensor,
        network="test",
        netuid=11,
        hotkey="5builder",
        uid=7,
        block=123,
        namespace="questions",
        epoch_id="epoch-100",
        artifact_hash="a" * 64,
    )
    assert subtensor.queries == [(11, 7, 123)]
    assert receipt.payload == payload
    assert len(receipt.receipt_hash) == 64
    verify_historical_receipt(receipt)
    assert type(receipt).from_dict(asdict(receipt)) == receipt


def test_historical_storage_query_uses_bound_hotkey_without_metagraph_lookup():
    payload = commitment_payload("questions", "epoch-100", "a" * 64)

    class DirectSubtensor:
        def get_commitment_metadata(self, netuid, hotkey, block):
            assert (netuid, hotkey, block) == (11, "bound-hotkey", 123)
            return {
                "info": {
                    "fields": [{"Raw": "0x" + payload.encode().hex()}],
                },
            }

        def get_commitment(self, *_args, **_kwargs):
            raise AssertionError("UID/metagraph convenience path must not be used")

    assert commitment_at_block(
        DirectSubtensor(),
        netuid=11,
        uid=7,
        hotkey="bound-hotkey",
        block=123,
    ) == payload


def test_historical_receipt_rejects_missing_or_overwritten_value():
    subtensor = FakeSubtensor({}, latest=commitment_payload("report", "epoch", "b" * 64))
    with pytest.raises(CommitmentError, match="historical"):
        capture_historical_receipt(
            subtensor,
            network="test",
            netuid=11,
            hotkey="5builder",
            uid=7,
            block=123,
            namespace="report",
            epoch_id="epoch",
            artifact_hash="b" * 64,
        )


def test_submission_requires_inclusion_and_finalization():
    subtensor = FakeSubtensor({})
    wallet = object()
    set_commitment_finalized(
        subtensor,
        wallet,
        netuid=11,
        namespace="report",
        epoch_id="epoch",
        artifact_hash="b" * 64,
    )
    assert subtensor.set_calls[0][3] == {
        "wait_for_inclusion": True,
        "wait_for_finalization": True,
    }


def test_invalid_hash_or_namespace_is_rejected():
    with pytest.raises(CommitmentError):
        commitment_payload("unsupported", "epoch", "a" * 64)
    with pytest.raises(CommitmentError):
        commitment_payload("report", "epoch", "ABC")


def test_transition_scan_finds_exact_commit_block_and_collects_committee():
    epoch_id = "epoch-100"
    artifact_hash = "c" * 64
    payload = commitment_payload("questions", epoch_id, artifact_hash)
    subtensor = TransitionSubtensor({
        7: {102: payload},
        8: {104: payload},
    })
    receipt = find_historical_receipt(
        subtensor,
        network="test",
        netuid=11,
        uid=7,
        hotkey="builder-b",
        namespace="questions",
        epoch_id=epoch_id,
        start_block=100,
        end_block=109,
        artifact_hash=artifact_hash,
    )
    assert receipt is not None and receipt.block == 102
    receipts = collect_historical_receipts(
        subtensor,
        network="test",
        netuid=11,
        builders=[(7, "builder-b"), (8, "builder-a")],
        namespace="questions",
        epoch_id=epoch_id,
        start_block=100,
        end_block=109,
        artifact_hash=artifact_hash,
    )
    assert [(item.hotkey, item.block) for item in receipts] == [
        ("builder-a", 104), ("builder-b", 102),
    ]


def test_report_scan_discovers_hash_and_rejects_equivocation():
    epoch_id = "epoch-100"
    first = commitment_payload("report", epoch_id, "d" * 64)
    second = commitment_payload("report", epoch_id, "e" * 64)
    subtensor = TransitionSubtensor({7: {105: first}})
    receipt = find_historical_receipt(
        subtensor,
        network="test",
        netuid=11,
        uid=7,
        hotkey="builder",
        namespace="report",
        epoch_id=epoch_id,
        start_block=101,
        end_block=109,
    )
    assert receipt is not None and receipt.artifact_hash == "d" * 64

    subtensor.transitions[7][108] = second
    with pytest.raises(CommitmentError, match="multiple matching"):
        find_historical_receipt(
            subtensor,
            network="test",
            netuid=11,
            uid=7,
            hotkey="builder",
            namespace="report",
            epoch_id=epoch_id,
            start_block=101,
            end_block=109,
        )


def test_submission_returns_exact_historical_receipt():
    subtensor = TransitionSubtensor(current_block=50)
    wallet = SimpleNamespace(
        uid=7,
        hotkey=SimpleNamespace(ss58_address="builder"),
    )
    receipt = submit_commitment_with_receipt(
        subtensor,
        wallet,
        network="test",
        netuid=11,
        uid=7,
        namespace="report",
        epoch_id="epoch-50",
        artifact_hash="f" * 64,
    )
    assert receipt.block == 51
    assert receipt.hotkey == "builder"


def test_submission_never_uses_reorgable_best_chain_height():
    class BestChainAhead(TransitionSubtensor):
        def get_current_block(self):
            return 9_999

    subtensor = BestChainAhead(current_block=50)
    wallet = SimpleNamespace(
        uid=7,
        hotkey=SimpleNamespace(ss58_address="builder"),
    )
    receipt = submit_commitment_with_receipt(
        subtensor,
        wallet,
        network="test",
        netuid=11,
        uid=7,
        namespace="questions",
        epoch_id="epoch-finalized",
        artifact_hash="a" * 64,
    )
    assert receipt.block == 51


def test_historical_scan_rejects_unfinalized_end_block():
    subtensor = TransitionSubtensor(current_block=50)
    with pytest.raises(CommitmentError, match="exceeds finality"):
        find_historical_receipt(
            subtensor,
            network="test",
            netuid=11,
            uid=7,
            hotkey="builder",
            namespace="report",
            epoch_id="epoch",
            start_block=50,
            end_block=51,
        )
