from __future__ import annotations

import json

import pytest

from fugal_subnet.v2.validator_state import ValidatorStateError, ValidatorStateStore


def test_liveness_forces_zero_after_three_misses_and_recovers(tmp_path):
    store = ValidatorStateStore(tmp_path / "state.json")
    hotkeys = {1: "hk1", 2: "hk2"}
    for epoch, expected_misses in (("e1", 1), ("e2", 2)):
        state = store.update_epoch(epoch, hotkeys, {1}, {1: "a" * 64})
        assert state.records[2].epochs_missed == expected_misses
        assert state.forced_zero_uids == set()
    state = store.update_epoch("e3", hotkeys, {1}, {1: "a" * 64})
    assert state.forced_zero_uids == {2}
    assert state.eligible_uids == {1}
    state = store.update_epoch(
        "e4", hotkeys, {1, 2}, {1: "a" * 64, 2: "b" * 64}
    )
    assert state.records[2].epochs_missed == 0
    assert state.eligible_uids == {1, 2}


def test_uid_transfer_resets_all_inherited_history(tmp_path):
    store = ValidatorStateStore(tmp_path / "state.json")
    store.update_epoch("e1", {4: "old-hotkey"}, {4}, {4: "c" * 64})
    state = store.update_epoch("e2", {4: "new-hotkey"}, set())
    assert state.reset_uids == {4}
    assert state.records[4].hotkey == "new-hotkey"
    assert state.records[4].epochs_seen == 0
    assert state.records[4].epochs_missed == 1
    assert state.records[4].current_head_hash == ""


def test_tampered_state_fails_closed(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"schema_version": 1, "records": [{"uid": 1}]}))
    with pytest.raises(ValidatorStateError, match="schema"):
        ValidatorStateStore(path).update_epoch("e1", {1: "hk"}, {1})


def test_same_epoch_resume_does_not_double_count(tmp_path):
    store = ValidatorStateStore(tmp_path / "state.json")
    first = store.update_epoch("e1", {1: "hk"}, {1})
    second = store.update_epoch("e1", {1: "hk"}, {1})
    assert first.records[1].epochs_seen == second.records[1].epochs_seen == 1


def test_preview_does_not_persist_an_epoch_that_may_abort(tmp_path):
    path = tmp_path / "state.json"
    store = ValidatorStateStore(path)

    preview = store.preview_epoch("e1", {1: "hk"}, {1}, {1: "a" * 64})

    assert preview.records[1].epochs_seen == 1
    assert not path.exists()


def test_commit_after_preview_matches_and_persists(tmp_path):
    path = tmp_path / "state.json"
    store = ValidatorStateStore(path)
    preview = store.preview_epoch("e1", {1: "hk"}, {1}, {1: "a" * 64})

    committed = store.update_epoch("e1", {1: "hk"}, {1}, {1: "a" * 64})

    assert committed == preview
    assert path.exists()
