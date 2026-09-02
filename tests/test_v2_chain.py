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
    def __init__(self, historical, stored_weights=((1, ((0, 49151), (2, 16384))),)):
        self.historical = historical
        self.weight_call = None
        self.stored_weights = stored_weights
        self.weight_query_block = None
        self.commit_reveal_enabled = False
        self.timelock_commits: tuple = ()
        self.substrate = SimpleNamespace(
            get_chain_finalised_head=lambda: "0x" + f"{200:064x}",
            get_block_number=lambda _hash: 200,
            query_map=self._query_map,
            query=self._query,
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
        # Bittensor 10.5 returns an empty derived matrix on a local chain even
        # after finalized set_weights calls; the adapter must not depend on it.
        return SimpleNamespace(n=3, W=[])

    def _query(self, module, storage_function, params):
        assert module == "SubtensorModule"
        if storage_function == "CommitRevealWeightsEnabled":
            return self.commit_reveal_enabled
        raise AssertionError(storage_function)

    def _query_map(self, module, storage_function, params):
        assert module == "SubtensorModule"
        if storage_function == "TimelockedWeightCommits":
            return list(self.timelock_commits)
        raise AssertionError(storage_function)

    def weights(self, netuid, block=None):
        del netuid
        self.weight_query_block = block
        # Sparse (uid, [(target_uid, u16)]) rows, as SubtensorModule.Weights
        # exposes them. 49151/16384 is the u16 encoding of 0.75/0.25.
        return list(self.stored_weights)


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
    """An empty metagraph.W must not defeat the finalized-storage restart check."""
    subtensor = FakeSubtensor({})
    assert subtensor.metagraph(11).W == []
    assert chain.submit_exact_weights(
        subtensor,
        object(),
        netuid=11,
        validator_uid=1,
        weights={"0": "0.75", "2": "0.25"},
    ) is False
    assert subtensor.weight_call is None
    # The row must be read at the finalized height, never at the best head.
    assert subtensor.weight_query_block == 200


def test_finalized_weight_row_normalizes_the_u16_row():
    subtensor = FakeSubtensor({})
    row = chain.finalized_weight_row(subtensor, netuid=11, validator_uid=1)
    assert row is not None
    assert sorted(row) == [0, 2]
    assert abs(sum(row.values()) - 1.0) < 1e-12
    assert abs(row[0] - 0.75) < chain.WEIGHT_MATCH_TOLERANCE


def test_absent_finalized_row_resubmits_rather_than_assuming_a_match():
    subtensor = FakeSubtensor({}, stored_weights=())
    assert chain.finalized_weight_row(subtensor, netuid=11, validator_uid=1) is None
    assert chain.submit_exact_weights(
        subtensor,
        object(),
        netuid=11,
        validator_uid=1,
        weights={"0": "0.75", "2": "0.25"},
    ) is True
    assert subtensor.weight_call["uids"] == [0, 2]


def test_differing_finalized_row_is_not_treated_as_a_match():
    subtensor = FakeSubtensor({}, stored_weights=((1, ((0, 32768), (2, 32767))),))
    assert chain.submit_exact_weights(
        subtensor,
        object(),
        netuid=11,
        validator_uid=1,
        weights={"0": "0.75", "2": "0.25"},
    ) is True
    assert subtensor.weight_call is not None


def test_finalized_row_with_an_extra_uid_is_drift():
    # A whole extra recipient, well beyond the u16 round-trip tolerance.
    subtensor = FakeSubtensor(
        {}, stored_weights=((1, ((0, 44151), (2, 14384), (3, 7000))),)
    )
    assert chain._chain_weights_match(
        subtensor, netuid=11, validator_uid=1, target={0: 0.75, 2: 0.25}
    ) is False


def test_explicit_zero_weight_matches_a_row_the_pallet_did_not_store():
    subtensor = FakeSubtensor({})
    assert chain._chain_weights_match(
        subtensor,
        netuid=11,
        validator_uid=1,
        target={0: 0.75, 1: 0.0, 2: 0.25},
    ) is True


def test_malformed_finalized_rows_fail_closed():
    for stored in (
        ((1, ((0, 49151), (0, 16384))),),        # duplicate target uid
        ((1, ((0, -1), (2, 16384))),),           # negative weight
        ((1, ((0, 0), (2, 0))),),                # zero total
        ((1, ("not-a-pair",)),),                 # malformed entry
    ):
        subtensor = FakeSubtensor({}, stored_weights=stored)
        assert chain.finalized_weight_row(
            subtensor, netuid=11, validator_uid=1
        ) is None


def _commit_subtensor(commit_block, hotkey="val-hot"):
    subtensor = FakeSubtensor({}, stored_weights=())
    subtensor.commit_reveal_enabled = True
    subtensor.timelock_commits = ((214, ((hotkey, commit_block, "0xdeadbeef"),)),)
    return subtensor


class FakeWallet:
    def __init__(self, hotkey="val-hot"):
        self.hotkey = SimpleNamespace(ss58_address=hotkey)


def test_commit_reveal_submission_is_invisible_to_the_plaintext_read():
    """A commit-reveal subnet writes no plaintext row, so weights() sees nothing."""
    subtensor = _commit_subtensor(commit_block=500)
    assert chain.read_finalized_weight_row(
        subtensor, netuid=11, validator_uid=1
    ) is None
    assert chain.commit_reveal_is_enabled(subtensor, 11) is True
    assert chain.pending_timelock_commit_block(
        subtensor, netuid=11, hotkey="val-hot"
    ) == 500


def test_pending_commit_at_this_epoch_suppresses_a_duplicate_submission():
    subtensor = _commit_subtensor(commit_block=500)
    assert chain.submit_exact_weights(
        subtensor,
        FakeWallet(),
        netuid=11,
        validator_uid=1,
        weights={"0": "0.75", "2": "0.25"},
        epoch_start_block=480,
    ) is False
    assert subtensor.weight_call is None


def test_commit_from_a_previous_epoch_does_not_suppress_this_epoch():
    """An older commit belongs to a finished epoch and must not block a new one."""
    subtensor = _commit_subtensor(commit_block=400)
    assert chain.submit_exact_weights(
        subtensor,
        FakeWallet(),
        netuid=11,
        validator_uid=1,
        weights={"0": "0.75", "2": "0.25"},
        epoch_start_block=480,
    ) is True
    assert subtensor.weight_call is not None


def test_another_hotkeys_commit_never_suppresses_our_submission():
    subtensor = _commit_subtensor(commit_block=500, hotkey="someone-else")
    assert chain.pending_timelock_commit_block(
        subtensor, netuid=11, hotkey="val-hot"
    ) is None
    assert chain.submit_exact_weights(
        subtensor,
        FakeWallet(),
        netuid=11,
        validator_uid=1,
        weights={"0": "0.75", "2": "0.25"},
        epoch_start_block=480,
    ) is True


def test_unreadable_commit_reveal_flag_assumes_the_deferring_path():
    """An unreadable flag must never cause a duplicate submission."""
    subtensor = FakeSubtensor({}, stored_weights=())
    subtensor.substrate.query = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("x"))
    assert chain.commit_reveal_is_enabled(subtensor, 11) is True
