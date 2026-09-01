from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from fugal_subnet.v2.journal import EpochJournal
from neurons import validator_v2


class _Subtensor:
    def __init__(self, finalized_blocks):
        self._blocks = iter(finalized_blocks)
        self._current = 0
        self.hash_requests = []
        self.substrate = SimpleNamespace(
            get_chain_finalised_head=self._head,
            get_block_number=lambda _block_hash: self._current,
        )

    def _head(self):
        self._current = next(self._blocks)
        return f"0x{self._current:064x}"

    def get_block_hash(self, block):
        self.hash_requests.append(block)
        return f"0x{block:064x}"


def test_wait_until_uses_finalized_head(monkeypatch):
    subtensor = _Subtensor([8, 9, 10, 10])
    monkeypatch.setattr(validator_v2.time, "sleep", lambda _seconds: None)

    assert validator_v2._wait_until(subtensor, 10) == 10
    assert subtensor.hash_requests == [10]


def test_commitment_collection_waits_for_finalized_deadline(monkeypatch):
    events = []
    late_receipt = object()

    def wait_until(_subtensor, deadline):
        events.append(("deadline", deadline))
        return deadline

    def collect():
        events.append(("collect", None))
        return (object(), object(), object(), late_receipt)

    monkeypatch.setattr(validator_v2, "_wait_until", wait_until)
    receipts = validator_v2._collect_after_finalized_deadline(
        object(), 600, collect,
    )

    assert events == [("deadline", 600), ("collect", None)]
    assert len(receipts) == 4
    assert receipts[-1] is late_receipt


def test_report_release_clock_is_monotonic_and_chain_io_free():
    clock = validator_v2._FinalizedBlockClock(100)
    assert clock.current() == 100
    assert clock.advance(125) == 125
    assert clock.current() == 125
    with pytest.raises(RuntimeError, match="cannot move backward"):
        clock.advance(124)


def test_schedule_requires_ordered_positive_deadlines():
    consensus = {"committee": {
        "epoch_blocks": 360,
        "precommit_deadline_offset_blocks": 60,
        "report_deadline_offset_blocks": 300,
        "slice_size": 100,
        "api_concurrency": 4,
    }}
    assert validator_v2._schedule(consensus) == (360, 60, 300, 100, 4)
    consensus["committee"]["report_deadline_offset_blocks"] = 400
    with pytest.raises(RuntimeError, match="deadlines"):
        validator_v2._schedule(consensus)


def test_live_mode_requires_budget_before_any_runtime_initialization():
    result = CliRunner().invoke(
        validator_v2.main,
        ["--live", "--grader-socket", "/run/fugal/grader.sock"],
    )
    assert result.exit_code == 2
    assert "explicit positive epoch budget" in result.output


def test_epoch_windows_are_relative_to_activation_not_genesis():
    protocol = SimpleNamespace(
        activation_blocks={"local": 101, "test": 500, "finney": None}
    )
    assert validator_v2._epoch_window(
        "test", protocol, 861, 360, 60, 300,
    ) == (1, 860, 920, 1160)
    with pytest.raises(RuntimeError, match="activation"):
        validator_v2._epoch_window("test", protocol, 499, 360, 60, 300)
    assert validator_v2._next_epoch_boundary("test", protocol, 861, 360) == 1220


def test_local_backbone_lock_is_rejected_on_public_network(monkeypatch, tmp_path):
    monkeypatch.setenv("FUGAL_LOCAL_BACKBONE_LOCK", str(tmp_path / "qwen.lock"))
    monkeypatch.setenv("FUGAL_LOCAL_BACKBONE_CACHE", str(tmp_path / "cache"))
    assert validator_v2._local_backbone_lock("local") == tmp_path / "qwen.lock"
    assert validator_v2._local_backbone_cache("local") == tmp_path / "cache"
    with pytest.raises(RuntimeError, match="local/mock-only"):
        validator_v2._local_backbone_lock("test")
    with pytest.raises(RuntimeError, match="local/mock-only"):
        validator_v2._local_backbone_cache("test")


def test_local_backbone_cache_computes_once(monkeypatch, tmp_path):
    import fugal_subnet.v2.backbone as backbone

    calls = []

    def compute(prompts):
        calls.append(tuple(prompts))
        return np.ones((len(prompts), backbone.HIDDEN_DIM), dtype=np.float32)

    monkeypatch.setattr(backbone, "compute_hidden_states", compute)
    lock = tmp_path / "qwen.lock"
    cache = tmp_path / "cache"
    cache.mkdir()

    first = validator_v2._compute_local_serialized(["one"], lock, cache)
    second = validator_v2._compute_local_serialized(["one"], lock, cache)

    assert np.array_equal(first, second)
    assert calls == [("one",)]


def test_uid_transfers_and_metagraph_size_changes_are_forced_zero():
    assert validator_v2._changed_uids(
        ["alice", "bob", "carol"],
        ["alice", "mallory", "carol", "new"],
    ) == {1, 3}


def test_stale_active_journals_are_aborted(tmp_path):
    stale = EpochJournal(tmp_path, "v2-000000000001")
    stale.initialize(
        manifest_hash="a" * 64,
        boundary_block=100,
        boundary_hash="boundary",
        budget_usd="0",
    )
    current = EpochJournal(tmp_path, "v2-000000000002")
    current.initialize(
        manifest_hash="a" * 64,
        boundary_block=200,
        boundary_hash="boundary-2",
        budget_usd="0",
    )

    assert validator_v2._abort_stale_journals(tmp_path, 200) == 1
    assert stale.read()["status"] == "aborted"
    assert current.read()["status"] == "active"
    assert validator_v2._abort_stale_journals(tmp_path, 200) == 0


def test_stale_journal_symlink_is_rejected(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    epoch = tmp_path / "v2-000000000001"
    epoch.mkdir()
    (epoch / "journal.json").symlink_to(target)

    with pytest.raises(RuntimeError, match="symbolic link"):
        validator_v2._abort_stale_journals(tmp_path, 200)
