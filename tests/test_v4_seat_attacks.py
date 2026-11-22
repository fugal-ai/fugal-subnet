"""The protocol v4 seat cases of the miner attack suite
(fugal_subnet/attacks/run_miner_attacks.py, SEAT_ATTACKS; SEAT_MINIMAL_V4.md
3.5 to 3.7, 4.4), each held, and each shown to break once the rule that
holds it is taken out (rule C, the standing window, byte-identical
succession, a successor reading its predecessors' odds together with the
late-odds rounds serving the seat's heads first, each of which holds it
alone); and the residual the design accepts (3.9: a copy of a
head absent for more than the standing window no longer is barred by it)."""
from __future__ import annotations

import pytest

from fugal_subnet import labels_v4, seat_v4
from fugal_subnet.attacks import run_miner_attacks as suite
from fugal_subnet.attacks.seat_v4_world import World


def _no_rule_c(monkeypatch):
    """Rule C taken out: a qualifier beats the bar (and R) and nothing else."""
    monkeypatch.setattr(seat_v4, "must_beat",
                        lambda order, i, *, open_seat: [] if open_seat else [None])


def _no_standing(monkeypatch):
    """The standing window taken down to the epoch itself: a head that is
    not verified this epoch no longer counts."""
    monkeypatch.setattr(seat_v4, "STANDING_EPOCHS", 1)


def _any_successor(monkeypatch):
    """Succession without the byte-identical rows: any later head of the
    same hotkey and coldkey inherits its predecessor's place."""
    monkeypatch.setattr(seat_v4, "is_successor", lambda new_rows, pred, menu_ids: True)


def _no_predecessor_odds(monkeypatch):
    """A successor no longer reads its predecessors' stored odds: before its
    commit it has only the late odds computed for it."""
    monkeypatch.setattr(labels_v4, "predecessor_odds", lambda window, side, epoch: None)


def _no_late_odds_priority(monkeypatch):
    """Late-odds rounds no longer serve the seat's heads first: every head
    takes its turn."""
    monkeypatch.setattr(seat_v4, "late_odds_first", lambda state, window, *, epoch: frozenset())


def _no_predecessor_odds_nor_priority(monkeypatch):
    """Both: a successor has before its commit only the late odds it gets a
    turn for among every other head."""
    _no_predecessor_odds(monkeypatch)
    _no_late_odds_priority(monkeypatch)


RULES = {
    "v4 seat: a near-copy of an earlier head": _no_rule_c,
    "v4 seat: a near-copy while the original misses": _no_standing,
    "v4 seat: borrow an earlier commit's precedence": _any_successor,
    "v4 seat: a head that is re-committed each epoch keeps its evidence":
        _no_predecessor_odds_nor_priority,
}


def test_every_seat_case_names_its_rule():
    assert {n for n, *_ in suite.SEAT_ATTACKS} == set(RULES)


@pytest.mark.parametrize("case", suite.SEAT_ATTACKS, ids=[c[0] for c in suite.SEAT_ATTACKS])
def test_each_seat_case_is_held_by_its_rule(case, monkeypatch):
    name, fn, gain, inv = case
    held = suite._run_input(name, fn, gain, inv)
    assert held["verdict"] == "BLOCKED", held["detail"]
    RULES[name](monkeypatch)
    broken = suite._run_input(name, fn, gain, inv)
    assert broken["verdict"] == "BROKEN", broken["detail"]


@pytest.mark.parametrize("remove", [_no_predecessor_odds, _no_late_odds_priority])
def test_the_re_committed_holder_case_is_held_by_either_rule_alone(remove, monkeypatch):
    """The predecessors' stored odds and the late-odds priority each hold it
    alone: with one taken out the challenger still takes the seat when it
    would from a static holder."""
    remove(monkeypatch)
    held, detail = suite.atk_seat_recommitted_head_keeps_its_evidence()
    assert held, detail


def test_the_near_copy_case_is_held_by_rule_c_and_not_by_the_standing_window(monkeypatch):
    """The first case does not need the standing window (the original is
    served every epoch), and the second is not the first again: rule C alone
    taken out breaks the first while the window is whole."""
    _no_standing(monkeypatch)
    held, _ = suite.atk_seat_near_copy_of_an_earlier_head()
    assert held


def test_a_copy_of_the_holder_after_an_outage_longer_than_the_window_takes_the_seat():
    """The residual of 3.9, accepted and documented: a holder absent for
    more than STANDING_EPOCHS epochs no longer bars a near-copy of itself,
    which takes the open seat as the latest qualifier. Inside the window it
    cannot (the cases above)."""
    w = World()
    w.add("h", skill=0.4)
    w.add("copy", skill=0.4, block=9, noise=6, owner="ck-other")
    for e in range(1, 6):
        w.epoch(e, served=["h"], answered=["h"])
    for e in range(6, 11):
        w.epoch(e, served=["h", "copy"], answered=["h", "copy"])
        assert w.holder() == "h"
    events = []
    for e in range(11, 15):
        d = w.epoch(e, served=["copy"], answered=["copy"])
        events.append((d.event, w.holder()))
    assert events == [("hold", "h"), ("hold", "h"), ("take", "copy"), ("hold", "copy")]
