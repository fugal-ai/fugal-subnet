"""Protocol v4: late-odds rounds serve first the heads a seat decision reads
(v2_epoch.late_epochs `first`; seat_v4.late_odds_first, its predicate
seat_v4.challenged_recently).

1. The predicate: the seat holder, and every head that (itself or through a
   predecessor it succeeded, seat_v4.lineage_records) has a counted
   challenge in this epoch or one of the STANDING_EPOCHS - 1 before it (the
   contenders' condition).
2. The schedule: those heads take their rounds first, in the schedule's
   order among themselves, then every other head with the slots left; with
   no such head, or with slots for everyone, it chooses what it chose
   before.
3. A function of its inputs: the order of the heads, of the window's odds,
   of `since` and of the priority set never changes what is chosen.
4. Delay: a challenger that answered recently and missed 5 epochs gets
   their odds the epoch it is back, whatever the number of other heads with
   backlogs (up to 255), where the turn-taking schedule alone delays it by
   up to 34 epochs at 128 or more.
5. The seat's world (as the live epoch) hands the priority set over.
"""
from __future__ import annotations

import dataclasses
import hashlib
import random

import numpy as np
import pytest

from fugal_subnet import config, labels_v4, seat_v4
from fugal_subnet.champion import EpochPairs, Window
from fugal_subnet.picks_v4 import PicksEpoch
from fugal_subnet.v2_epoch import late_epochs

SLOTS = config.LATE_ODDS_SLOTS_V4
WINDOW = config.SEAT_WINDOW_EPOCHS_V4


def _pairs(n: int = 2) -> labels_v4.LabelPairs:
    return labels_v4.label_pairs(
        EpochPairs(question_ids=tuple(f"q{i}" for i in range(n)), models=("m",) * n,
                   benchmarks=("b",) * n, categories=("c",) * n, y=np.zeros(n, np.int32),
                   prompt_tokens=np.zeros(n, np.int32), completion_tokens=np.zeros(n, np.int32)),
        [("ck",)] * n, np.full(n, -1, np.int16))


P = _pairs()


def _h(tag) -> str:
    return hashlib.sha256(f"late-priority:{tag}".encode()).hexdigest()


def _old_schedule(window, heads, since, slots):
    """The schedule before the priority (every head in turn), as a reference."""
    newest = max(window.pairs, default=0)
    wanted = {}
    for h in sorted(heads, key=lambda h: hashlib.sha256(f"{newest}|{h}".encode()).hexdigest()):
        have = window.odds.get(h, {})
        wanted[h] = [e for e in sorted(window.pairs, reverse=True)
                     if e >= int(since[h]) and e not in have]
    chosen, used, depth = {}, 0, 0
    while any(depth < len(v) for v in wanted.values()):
        for h, todo in wanted.items():
            if depth < len(todo) and (slots is None or used < slots):
                chosen.setdefault(h, []).append(todo[depth])
                used += 1
        depth += 1
        if slots is not None and used >= slots:
            break
    return chosen


def _random_window(rng, trial):
    heads = [_h(f"r{trial}-{i}") for i in range(rng.randint(1, 60))]
    since = {x: rng.randint(0, 47) for x in heads}
    odds = {x: {e: np.zeros(2, np.int16) for e in range(48) if rng.random() < 0.4} for x in heads}
    w = dataclasses.replace(Window.empty(48), pairs={e: P for e in range(48)}, odds=odds)
    return w, heads, since


# --- 1. the predicate ----------------------------------------------------------------------


def _hs(name: str, preds=(), since: int = 1) -> seat_v4.HeadState:
    return seat_v4.HeadState(head=_h(name), owner=f"ck-{name}", hotkey=f"hk-{name}", block=10,
                             since=since, seen=10, origin=_h(name), origin_block=10,
                             origin_owner=f"ck-{name}", predecessors=tuple(_h(p) for p in preds),
                             predecessor_since=tuple(range(len(preds))))


def test_the_holder_and_the_heads_that_answered_recently_come_first():
    g = seat_v4.STANDING_EPOCHS
    names = ["holder", "now", "edge", "gone", "never", "line"]
    state = seat_v4.SeatStateV4(champion=_h("holder"), champion_since=1,
                                heads=tuple(_hs(n, ("line-old",) if n == "line" else (),
                                                # a predecessor's challenges count
                                                # before its successor's commit
                                                since=10 if n == "line" else 1)
                                            for n in names), epoch=10)
    answered = {10: ["now"], 10 - g + 1: ["edge", "line-old"], 10 - g: ["gone"]}
    picks = {e: PicksEpoch(e, ("c",), {_h(n): (0,) for n in ns}, False)
             for e, ns in answered.items()}
    window = dataclasses.replace(Window.empty(48), picks=picks)
    first = seat_v4.late_odds_first(state, window, epoch=10)
    # The holder whether or not it answered; a successor through its predecessor.
    assert first == frozenset(_h(n) for n in ("holder", "now", "edge", "line"))
    records = seat_v4.lineage_records(window.picks, state)
    assert [seat_v4.challenged_recently(records, _h(n), epoch=10) for n in names] == [
        False, True, True, False, False, True]
    # No holder, no records: nobody.
    assert seat_v4.late_odds_first(seat_v4.SeatStateV4(), Window.empty(48), epoch=10) == frozenset()


# --- 2. the schedule -----------------------------------------------------------------------


def test_the_first_heads_take_their_rounds_before_any_other():
    x = _h("x")
    others = [_h(f"o{i}") for i in range(40)]
    since = {h: 0 for h in [x, *others]}
    odds = {h: {e: np.zeros(2, np.int16) for e in range(40, 48)} for h in others}
    odds[x] = {e: np.zeros(2, np.int16) for e in range(48) if not 41 <= e <= 45}
    w = dataclasses.replace(Window.empty(48), pairs={e: P for e in range(48)}, odds=odds)
    heads = {h: None for h in since}
    got = late_epochs(w, heads, since, SLOTS, first={x})
    assert got[x] == [45, 44, 43, 42, 41]
    assert sum(map(len, got.values())) == SLOTS
    rest = {h: v for h, v in got.items() if h != x}
    assert rest == _old_schedule(dataclasses.replace(w, odds={h: odds[h] for h in others}),
                                 {h: None for h in others}, since, SLOTS - 5)
    # Several first heads share their rounds in the schedule's order.
    y = _h("y")
    odds[y] = {}
    since[y] = 0
    w2 = dataclasses.replace(w, odds=odds)
    two = late_epochs(w2, {**heads, y: None}, since, SLOTS, first={x, y})
    assert set(two) == {x, y} and sum(map(len, two.values())) == SLOTS
    assert sorted(two[x]) == [41, 42, 43, 44, 45] and two[y] == list(range(47, 47 - 27, -1))
    assert two == _old_schedule(dataclasses.replace(w2, odds={x: odds[x], y: {}}),
                                {x: None, y: None}, since, SLOTS)


def test_without_competition_for_slots_nothing_changes():
    rng = random.Random(70)
    for trial in range(150):
        w, heads, since = _random_window(rng, trial)
        hd = {h: None for h in heads}
        first = set(rng.sample(heads, rng.randint(0, len(heads))))
        slots = rng.choice([None, 0, 1, 7, 32, 10_000])
        before = _old_schedule(w, hd, since, slots)
        # No first head: exactly what it chose before, in the same order.
        assert late_epochs(w, hd, since, slots) == before
        assert late_epochs(w, hd, since, slots, first=()) == before
        total = sum(1 for h in heads for e in w.pairs if e >= since[h] and e not in w.odds[h])
        if slots is None or total <= slots:
            got = late_epochs(w, hd, since, slots, first=first)
            assert {h: sorted(v) for h, v in got.items()} == {
                h: sorted(v) for h, v in before.items()}, trial


# --- 3. a function of its inputs -----------------------------------------------------------


def test_the_choice_does_not_depend_on_the_order_of_its_inputs():
    rng = random.Random(71)
    for trial in range(200):
        w, heads, since = _random_window(rng, trial)
        first = rng.sample(heads, rng.randint(0, len(heads)))
        slots = rng.choice([1, 7, 32])
        a = late_epochs(w, {x: None for x in heads}, since, slots, first=frozenset(first))
        shuffled = heads[:]
        rng.shuffle(shuffled)
        odds2 = {x: dict(reversed(list(w.odds[x].items()))) for x in reversed(heads)}
        w2 = dataclasses.replace(w, pairs=dict(reversed(list(w.pairs.items()))), odds=odds2)
        first2 = first[:]
        rng.shuffle(first2)
        b = late_epochs(w2, {x: None for x in shuffled}, {x: since[x] for x in shuffled}, slots,
                        first=first2 + [_h("not-a-head")])
        assert a == b, trial
        assert sum(map(len, a.values())) <= slots


# --- 4. the delay --------------------------------------------------------------------------


def _delay(n_others: int, *, priority: bool, backlog: int = 40, missed: int = 5,
           epochs: int = 90) -> int | None:
    """N other heads are committed at 0 and first verified at `backlog`
    (each owes `backlog` epochs), then every epoch; the challenger X, which
    answers the scored questions, is verified every epoch but the `missed`
    before `backlog`. The epochs after its return until its missed epochs
    all have odds (None: some left the window unfilled)."""
    x = _h("challenger")
    others = [_h(f"g{i}") for i in range(n_others)]
    w = Window.empty(WINDOW)
    gone = set(range(backlog - missed, backlog))
    filled: dict[int, int] = {}
    for e in range(epochs):
        verified = ([] if e in gone else [x]) + (others if e >= backlog else [])
        pw = {k: p for k, p in {**w.pairs, e: P}.items() if k > e - WINDOW}
        odds = {h: {k: o for k, o in per.items() if k in pw} for h, per in w.odds.items()}
        for h in verified:
            odds.setdefault(h, {})[e] = np.zeros(2, np.int16)
        w = dataclasses.replace(w, pairs=pw, odds=odds)
        chosen = late_epochs(w, {h: None for h in verified}, {h: 0 for h in verified}, SLOTS,
                             first={x} if priority and x in verified else ())
        for h, eps in chosen.items():
            for k in eps:
                odds[h][k] = np.zeros(2, np.int16)
                if h == x:
                    filled[k] = e
    if any(k not in filled for k in gone):
        return None
    return max(filled[k] for k in gone) - backlog


@pytest.mark.parametrize("n", [64, 128, 255])
def test_a_recent_challengers_late_odds_are_never_delayed(n):
    assert _delay(n, priority=True) == 0
    # The turn-taking schedule alone delays it (the reason for the priority).
    assert _delay(n, priority=False) not in (0,)


# --- 5. the seat's world hands it over ------------------------------------------------------


def test_the_seats_world_serves_a_challenger_first():
    """On the seat's world with 32 slots: a challenger that answers, back
    after 5 missed epochs, among 40 heads that each owe 30 epochs, has all
    its odds the epoch it is back."""
    from tests.seat_v4_world import World

    w = World(late_slots=SLOTS, labels=20, questions=4)
    w.add("x", skill=0.5, hotkey="hk-x", owner="ck-x", block=5)
    for i in range(40):
        w.add(f"g{i}", skill=0.1, hotkey=f"hk-g{i}", owner="ck-g", block=5)
    for e in range(1, 31):
        w.epoch(e, served=[] if 26 <= e <= 30 else ["x"], answered=[] if 26 <= e else ["x"])
    w.epoch(31, served=["x", *(f"g{i}" for i in range(40))], answered=["x"])
    x = w.heads["x"].hash
    assert all(k in w.window.odds[x] for k in range(26, 31))
    gs = [w.heads[f"g{i}"].hash for i in range(40)]
    assert sum(len(w.window.odds[g]) - 1 for g in gs) == SLOTS - 5
