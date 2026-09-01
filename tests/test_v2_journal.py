"""Crash/restart and tamper vectors for the inactive v2 epoch journal."""

from __future__ import annotations

import json
import os
import threading

import pytest

from fugal_subnet.v2.journal import MAX_RESPONSE_BYTES, EpochJournal, JournalError

MANIFEST_HASH = "a" * 64
BOUNDARY_HASH = "0xboundary"


def initialized(tmp_path, epoch_id: str = "e00000001", budget: str = "1") -> EpochJournal:
    journal = EpochJournal(tmp_path, epoch_id)
    journal.initialize(
        manifest_hash=MANIFEST_HASH,
        boundary_block=100,
        boundary_hash=BOUNDARY_HASH,
        budget_usd=budget,
    )
    return journal


def test_atomic_journal_permissions_and_binding(tmp_path):
    journal = initialized(tmp_path)
    state = journal.read()

    assert state["schema_version"] == 1
    assert state["manifest_hash"] == MANIFEST_HASH
    assert state["boundary"] == {"block": 100, "hash": BOUNDARY_HASH}
    assert os.stat(journal.path).st_mode & 0o777 == 0o600
    assert [path.name for path in journal.epoch_dir.glob(".journal.*")] == [
        ".journal.lock"
    ]

    # Initialization is idempotent only for the exact same epoch identity.
    journal.initialize(
        manifest_hash=MANIFEST_HASH,
        boundary_block=100,
        boundary_hash=BOUNDARY_HASH,
        budget_usd="1",
    )
    with pytest.raises(JournalError, match="does not match"):
        journal.initialize(
            manifest_hash="b" * 64,
            boundary_block=100,
            boundary_hash=BOUNDARY_HASH,
            budget_usd="1",
        )


def test_completed_cell_is_cached_idempotently_without_double_spend(tmp_path):
    journal = initialized(tmp_path)
    assert journal.reserve_cell("q1", "provider/model", "0.4") is True
    assert journal.complete_cell(
        "q1", "provider/model", "bounded response", 10, 4, "0.1"
    ) is True

    # A restart sees the completed response and must not schedule it again.
    restarted = EpochJournal(tmp_path, "e00000001")
    assert restarted.reserve_cell("q1", "provider/model", "0.4") is False
    assert restarted.complete_cell(
        "q1", "provider/model", "bounded response", 10, 4, "0.1"
    ) is False

    state = restarted.read()
    assert state["spend"] == {
        "budget_usd": "1",
        "reserved_usd": "0",
        "actual_usd": "0.1",
    }
    cell = next(iter(state["cells"].values()))
    assert cell["response_text"] == "bounded response"
    assert cell["response_sha256"]

    with pytest.raises(JournalError, match="conflicting response"):
        restarted.complete_cell(
            "q1", "provider/model", "different response", 10, 4, "0.1"
        )


def test_crash_left_reservation_is_never_repeated_automatically(tmp_path):
    journal = initialized(tmp_path)
    journal.reserve_cell("q1", "provider/model", "0.25")

    restarted = EpochJournal(tmp_path, "e00000001")
    assert restarted.has_inflight_cells() is True
    assert restarted.reserve_cell("q1", "provider/model", "0.25") is False

    # The orchestrator must conservatively forfeit/abort rather than call again.
    assert restarted.forfeit_cell("q1", "provider/model") is True
    assert restarted.forfeit_cell("q1", "provider/model") is False
    assert restarted.has_inflight_cells() is False
    assert restarted.read()["spend"] == {
        "budget_usd": "1",
        "reserved_usd": "0",
        "actual_usd": "0.25",
    }


def test_concurrent_reservations_are_serialized_against_budget(tmp_path):
    journal = initialized(tmp_path, budget="1")
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def reserve(question: str):
        barrier.wait()
        try:
            journal.reserve_cell(question, "provider/model", "0.6")
            outcomes.append("reserved")
        except JournalError:
            outcomes.append("rejected")

    threads = [
        threading.Thread(target=reserve, args=("q1",)),
        threading.Thread(target=reserve, args=("q2",)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["rejected", "reserved"]
    assert journal.read()["spend"]["reserved_usd"] == "0.6"


def test_phase_commitment_report_and_terminal_rules(tmp_path):
    journal = initialized(tmp_path)
    journal.advance_phase("precommit")
    commitment_hash = "c" * 64
    journal.record_finalized_commitment("questions", commitment_hash, 101)
    journal.record_finalized_commitment("questions", commitment_hash, 101)
    journal.update_report(
        artifact_hash="d" * 64,
        committed_block=120,
        chunk_count=3,
        received_hotkeys=["builder-b", "builder-a"],
    )
    state = journal.advance_phase("complete")

    assert state["status"] == "complete"
    assert state["commitments"] == [
        {"namespace": "questions", "hash": commitment_hash, "block": 101}
    ]
    assert state["report"]["received_hotkeys"] == ["builder-a", "builder-b"]
    with pytest.raises(JournalError, match="terminal"):
        journal.advance_phase("complete")

    other = initialized(tmp_path, epoch_id="e00000002")
    other.advance_phase("matrix_build")
    with pytest.raises(JournalError, match="backwards"):
        other.advance_phase("miner_query")
    aborted = other.abort("quorum unavailable")
    assert aborted["status"] == "aborted"
    assert aborted["abort_reason"] == "quorum unavailable"


def test_response_bounds_and_actual_cost_are_enforced(tmp_path):
    journal = initialized(tmp_path)
    journal.reserve_cell("q1", "provider/model", "0.2")
    with pytest.raises(JournalError, match="exceeds journal bound"):
        journal.complete_cell(
            "q1", "provider/model", "x" * (MAX_RESPONSE_BYTES + 1), 1, 1, "0.1"
        )
    with pytest.raises(JournalError, match="exceeds its reservation"):
        journal.complete_cell(
            "q1", "provider/model", "response", 1, 1, "0.3"
        )


def test_response_tampering_is_detected_on_reload(tmp_path):
    journal = initialized(tmp_path)
    journal.reserve_cell("q1", "provider/model", "0.2")
    journal.complete_cell("q1", "provider/model", "original", 1, 1, "0.1")

    raw = json.loads(journal.path.read_text(encoding="utf-8"))
    cell = next(iter(raw["cells"].values()))
    cell["response_text"] = "tampered"
    journal.path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(JournalError, match="response hash mismatch"):
        journal.read()


def test_epoch_path_traversal_is_rejected(tmp_path):
    for epoch_id in ("../escape", "/absolute", "", "a" * 65):
        with pytest.raises(JournalError, match="invalid epoch_id"):
            EpochJournal(tmp_path, epoch_id)
