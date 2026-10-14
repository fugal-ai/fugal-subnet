"""Dedup seniority is derived from chain state, identically on every validator.

Diligence 2026-10, additional finding A4. Seniority used to be the earliest
commitment block each validator had ever SEEN for a hotkey, persisted in its
own state file: deterministic per validator and different for each, so two
validators could disqualify different members of the same cluster. It is now
`seniority_key(registration block, commitment block)` from the metagraph and
the commitments pinned at the epoch boundary.
"""
from __future__ import annotations

from fugal_subnet.commitments import seniority_key
from fugal_subnet.dedup import find_duplicates
from tests.test_non_interference import _decisions


def test_registration_outranks_commitment_and_commitment_breaks_ties():
    assert seniority_key(1000, 5000) < seniority_key(1001, 1)
    assert seniority_key(1000, 1) < seniority_key(1000, 2)


def test_a_retrained_author_keeps_rank_over_the_copier_of_its_old_head():
    """The case the per-validator first-seen map existed for, now from chain."""
    behaviour = _decisions([0, 1, 2, 1, 0, 2, 1, 0] * 4)
    heads = {1: behaviour, 2: behaviour.copy()}
    # Author registered at 900 and re-committed at 2000 after retraining; the
    # copier registered at 1050 and committed the copy at 1100.
    seniority = {1: seniority_key(900, 2000), 2: seniority_key(1050, 1100)}
    assert find_duplicates(heads, seniority) == {2}


def test_the_validator_does_not_persist_a_first_commit_map(tmp_path, monkeypatch):
    from neurons import validator

    monkeypatch.setattr(validator, "STATE_PATH", str(tmp_path / "state.json"))
    validator.save_state({}, [], [], 3, None)
    state = validator.load_state(dict)
    assert "first_commit_blocks" not in state
    assert state["last_epoch_index"] == 3
