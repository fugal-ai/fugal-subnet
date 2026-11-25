"""Protocol v5 seat (fugal_subnet/seat_v5.py) on the synthetic world
(fugal_subnet/attacks/seat_v5_world.py): every verdict set by construction."""
from __future__ import annotations

import random

import pytest

from fugal_subnet import seat_v5
from fugal_subnet.attacks.seat_v5_world import Head, World

S = seat_v5.SeatStateV5


def test_empty_seat_fills_on_merit_not_on_order():
    w = World()
    weak = w.add(Head("weak", "ck-a", 10, 0.70))           # no better than B*
    strong = w.add(Head("strong", "ck-b", 20, 0.85))
    d = w.decide(S())
    assert d.event == "fill" and d.holder == strong.hash
    assert d.holder_since == 100 and d.payable == (strong.hash,)
    assert "does not beat the best single model" in d.evidence["heads"][weak.hash]["reasons"]


def test_no_head_beats_the_baseline_seat_stays_empty_and_burns():
    w = World(baseline_accuracy=0.85)
    w.add(Head("a", "ck-a", 10, 0.80))
    w.add(Head("b", "ck-b", 11, 0.84))
    d = w.decide(S())
    assert d.event == "empty" and d.holder is None and d.payable == ()
    uids, weights = seat_v5.weights(d, {}, burn_uid=0)
    assert uids == [0] and weights == [1.0]


def test_better_challenger_takes_and_holder_goes_to_history():
    w = World()
    h = w.add(Head("holder", "ck-h", 10, 0.80))
    c = w.add(Head("challenger", "ck-c", 50, 0.92))
    d = w.decide(S(h.hash, 90, ()))
    assert d.event == "take" and d.holder == c.hash and d.history == (h.hash,)
    assert d.holder_since == 100
    uids, weights = seat_v5.weights(d, {c.hash: 7, h.hash: 8}, burn_uid=0)
    assert dict(zip(uids, weights)) == pytest.approx({7: 0.90, 8: 0.10})   # the empty third seat goes to the holder


def test_near_copy_farm_cannot_pass_the_margin():
    w = World()
    h = w.add(Head("holder", "ck-h", 10, 0.85))
    for i in range(12):
        w.add(Head(f"copy{i}", f"ck-farm{i}", 20 + i, 0.85, seed=i + 1))
    d = w.decide(S(h.hash, 90, ()))
    assert d.event == "hold" and d.holder == h.hash and d.holder_since == 90


def test_lucky_late_copy_cannot_beat_the_earlier_original():
    w = World()
    orig = w.add(Head("original", "ck-o", 10, 0.88))
    late = w.add(Head("late-copy", "ck-l", 30, 0.89, seed=7))   # a point "luckier"
    d = w.decide(S())
    assert d.event == "fill" and d.holder == orig.hash
    assert any("by the margin" in r for r in d.evidence["heads"][late.hash]["reasons"])


def test_holder_may_upgrade_when_not_worse_and_keeps_its_place():
    w = World()
    h = w.add(Head("v1", "ck-h", 10, 0.85, noise=0.0))
    s = w.add(Head("v2", "ck-h", 40, 0.86, noise=0.0, seed=3))   # a retrain, a point better
    d = w.decide(S(h.hash, 77, ("old",)))
    assert d.event == "upgrade" and d.holder == s.hash
    assert d.holder_since == 77 and d.history == ("old",)


def test_near_identical_retrain_upgrades_not_shown_worse():
    # a same-coldkey retrain that differs on ~2% of 6,000 questions cannot be
    # shown worse by the 1/4 margin, so it replaces its own holder (gentle rule)
    w = World()
    h = w.add(Head("v1", "ck-h", 10, 0.85, noise=0.0))
    s = w.add(Head("v1b", "ck-h", 40, 0.85, noise=0.02, seed=5))
    d = w.decide(S(h.hash, 77, ()))
    assert d.event == "upgrade" and d.holder == s.hash and d.holder_since == 77
    assert not d.evidence["heads"][s.hash]["holder_over_upgrade"]["passes"]


def test_upgrade_must_itself_be_payable():
    w = World(baseline_accuracy=0.85)
    h = w.add(Head("v1", "ck-h", 10, 0.90, noise=0.0))
    s = w.add(Head("v2", "ck-h", 40, 0.84, noise=0.0, seed=3))   # below B*
    d = w.decide(S(h.hash, 77, ()))
    assert d.holder == h.hash
    assert any("upgrade refused" in r for r in d.evidence["heads"][s.hash]["reasons"])


def test_holder_that_stops_standing_is_replaced_on_merit_and_does_not_resume():
    w = World()
    h = w.add(Head("old", "ck-a", 10, 0.90, noise=0.0, standing=False))
    c = w.add(Head("new", "ck-b", 40, 0.85, noise=0.0, seed=2))
    d = w.decide(S(h.hash, 50, ()))
    assert d.event == "fill" and d.holder == c.hash and h.hash in d.history
    # it returns: it walks as an ordinary contender (earlier commit, so first),
    # and needs the margin over the new holder, which it has here
    h.standing = True
    d2 = w.decide(d.state)
    assert d2.event == "take" and d2.holder == h.hash


def test_worse_upgrade_is_refused_and_holder_holds():
    w = World()
    h = w.add(Head("v1", "ck-h", 10, 0.88))
    s = w.add(Head("v2", "ck-h", 40, 0.80, seed=3))
    d = w.decide(S(h.hash, 77, ()))
    assert d.event == "hold" and d.holder == h.hash
    assert any("upgrade refused" in r for r in d.evidence["heads"][s.hash]["reasons"])


def test_holder_worse_than_baseline_loses_the_seat():
    w = World(baseline_accuracy=0.88)
    h = w.add(Head("holder", "ck-h", 10, 0.70))
    d = w.decide(S(h.hash, 50, ()))
    assert d.event == "empty" and d.holder is None and d.history == (h.hash,)
    assert d.payable == ()


def test_holder_level_with_baseline_holds_but_burns():
    w = World(baseline_accuracy=0.80)
    h = w.add(Head("holder", "ck-h", 10, 0.80))
    d = w.decide(S(h.hash, 50, ()))
    assert d.event == "burn" and d.holder == h.hash and d.payable == ()


def test_copies_memo_refused_and_absent_heads_never_contend():
    w = World()
    w.add(Head("copy", "ck-a", 10, 0.95, copy=True))
    w.add(Head("memo", "ck-b", 11, 0.95, memo_refused=True))
    w.add(Head("gone", "ck-c", 12, 0.95, standing=False))
    d = w.decide(S())
    assert d.event == "empty" and d.evidence["order"] == []


def test_one_contender_per_coldkey_newest_and_the_cap():
    w = World()
    old = w.add(Head("old", "ck-a", 10, 0.95))
    new = w.add(Head("new", "ck-a", 20, 0.85))
    pool = seat_v5.select([h.contender() for h in w.heads.values()], None)
    assert [c.head for c in pool] == [new.hash]
    assert old.hash not in [c.head for c in pool]
    many = [Head(f"h{i}", f"ck{i}", 100 + i, 0.8).contender() for i in range(40)]
    holder = Head("H", "ckH", 999, 0.8).contender()
    pool = seat_v5.select(many + [holder], holder.head)
    assert len(pool) == seat_v5.CONTENDERS_MAX and holder in pool


def test_decision_does_not_depend_on_contender_order():
    w = World()
    h = w.add(Head("holder", "ck-h", 10, 0.80))
    for i in range(6):
        w.add(Head(f"c{i}", f"ck{i}", 20 + i, 0.80 + 0.03 * i, seed=i))
    names = list(w.heads)
    a = w.decide(S(h.hash, 90, ()), order=names)
    random.Random(1).shuffle(names)
    b = w.decide(S(h.hash, 90, ()), order=names)
    assert a == b


def test_state_round_trips_through_the_reveal():
    st = S("a" * 64, 12, ("b" * 64,))
    assert S.from_reveal(st.to_reveal()) == st
    with pytest.raises(ValueError):
        S.from_reveal({"holder": 5})


def test_day_one_all_below_baseline_burns_then_fills_the_first_epoch_one_passes():
    w = World(baseline_accuracy=0.86)
    heads = [w.add(Head(f"h{i}", f"ck{i}", 10 + i, 0.82 + 0.01 * i, seed=i)) for i in range(4)]
    d1 = w.decide(S(), epoch=1)
    assert d1.event == "empty" and d1.holder is None and d1.payable == ()
    assert d1.evidence["burned"] is True
    for h in heads:
        r = d1.evidence["heads"][h.hash]
        assert r["vs_baseline"]["passes"] is False
        assert "does not beat the best single model" in r["reasons"]
    assert seat_v5.weights(d1, {h.hash: i + 1 for i, h in enumerate(heads)}, burn_uid=0) == ([0], [1.0])
    better = w.add(Head("better", "ck-new", 99, 0.95))
    d2 = w.decide(d1.state, epoch=2)
    assert d2.event == "fill" and d2.holder == better.hash and d2.holder_since == 2
    assert d2.payable == (better.hash,) and d2.evidence["burned"] is False
