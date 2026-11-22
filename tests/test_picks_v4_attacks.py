"""The protocol v4 picks cases of the miner attack suite
(fugal_subnet/attacks/run_miner_attacks.py, PICKS_ATTACKS; SEAT_MINIMAL_V4.md
3.4, 3.7, 5.4), each held, and each shown to break once the rule that holds
it is taken out: the rule, not the shape of the case, is what holds."""
from __future__ import annotations

import pytest

from fugal_subnet import picks_v4 as pv
from fugal_subnet.attacks import run_miner_attacks as suite
from fugal_subnet.tee import verify as verify_mod


def _no_scored_rule(monkeypatch):
    """verify._v4_scored_check taken out: the scored results are not held to
    the proof's `scored` field, nor to the whole slice."""
    monkeypatch.setattr(verify_mod, "_v4_scored_check", lambda *a, **k: None)


def _any_epoch_counts(monkeypatch):
    """An epoch counts for a head whenever it is recorded, answered or not."""
    monkeypatch.setattr(pv, "answered_in", lambda records, head, epoch: int(epoch) in records)
    monkeypatch.setattr(pv, "answered_run", lambda records, head, through: tuple(
        e for e in sorted(records) if e <= through))


def _gaps_allowed(monkeypatch):
    """The run is every answered epoch, gaps and all: the consecutive rule out."""
    monkeypatch.setattr(pv, "answered_run", lambda records, head, through: tuple(
        e for e in sorted(records) if e <= through and head in records[e].values))


def _no_failure_bound(monkeypatch):
    monkeypatch.setattr(pv, "over_bound", lambda failed, calls: False)


RULES = {
    "a proof with a partial scored set": _no_scored_rule,
    "exploration-only proof, foreign scored results": _no_scored_rule,
    "a challenge counted in an epoch without one": _any_epoch_counts,
    "challenge only when it suits": _gaps_allowed,
    "junk exploration past the failure bound": _no_failure_bound,
}


def test_every_picks_case_names_its_rule():
    assert {n for n, *_ in suite.PICKS_ATTACKS} == set(RULES)


@pytest.mark.parametrize("case", suite.PICKS_ATTACKS, ids=[c[0] for c in suite.PICKS_ATTACKS])
def test_each_picks_case_is_held_by_its_rule(case, monkeypatch):
    name, fn, gain, inv = case
    held = suite._run_input(name, fn, gain, inv)
    assert held["verdict"] == "BLOCKED", held["detail"]
    RULES[name](monkeypatch)
    broken = suite._run_input(name, fn, gain, inv)
    assert broken["verdict"] == "BROKEN", broken["detail"]


def test_the_exploration_only_and_full_controls_verify():
    (control,) = [c for c in suite._v4_controls() if "exploration-only" in c["name"]]
    assert control["verdict"] == "CONTROL", control["detail"]
