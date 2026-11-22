"""A missing row counts from the head's line, not from its latest commit
(labels_v4.Side.line_since; FAST_CROWN_V4.md 18): a holder that lacks a
full entry's row and re-commits a successor every epoch (the same rows,
other bytes) is scored on that entry exactly as a holder that never
re-commits, and loses the seat in the same epoch. A successor that adds the
row is scored on it from its own commit. A head under another coldkey, or a
row that changed, inherits nothing.

The seat's world (fugal_subnet/attacks/seat_v4_world.py) marks the new
entry's labels before it is full as provisional here, as section 18 does."""
from __future__ import annotations

import numpy as np
import pytest

from fugal_subnet import labels_v4, seat_v4
from fugal_subnet.attacks.seat_v4_world import BPE, ENTRIES, World

NEW = ENTRIES[1]
FULL_FROM = 10


@pytest.fixture(autouse=True)
def _provisional_before_full(monkeypatch):
    real = labels_v4.label_pairs

    def marked(pairs, voters, missing):
        m = np.asarray(missing)
        return real(pairs, voters, np.where(m > 0, labels_v4.PROVISIONAL_LABEL, m).astype(np.int16))
    monkeypatch.setattr(labels_v4, "label_pairs", marked)


def _run(*, recommit_from=None, add_at=None, last=24, owner_from=None):
    """The holder h0 (a row for ENTRIES[0] only) fills the seat while NEW is
    provisional; NEW is full from FULL_FROM. From `recommit_from` the holder
    commits a successor every epoch with the same rows; at `add_at` one that
    adds NEW's row; `owner_from` re-commits under another coldkey."""
    w = World()
    w.missing_after[NEW] = FULL_FROM
    w.add("h0", skill=0.3, rows=(ENTRIES[0],), hotkey="hk-h", owner="ck-h", block=5)
    cur, events = "h0", {}
    for e in range(1, last):
        new = None
        if recommit_from is not None and e >= recommit_from:
            new = dict(rows=(ENTRIES[0],), owner="ck-h")
        if add_at is not None and e == add_at:
            new = dict(rows=ENTRIES, owner="ck-h")
        if owner_from is not None and e == owner_from:
            new = dict(rows=(ENTRIES[0],), owner="ck-other")
        if new is not None:
            cur = f"h{e}"
            w.add(cur, skill=0.3, hotkey="hk-h", block=(e - 1) * BPE + 1, **new)
        d = w.epoch(e, served=[cur], answered=[cur])
        events[e] = d.event
    return w, cur, events


def _strata(w, name, now):
    hs = w.state.head(w.heads[name].hash)
    ref = labels_v4.reference(w.window, now=now)
    return labels_v4.gain_strata(w.window, ref, seat_v4.side_of(hs), labels_v4.REFERENCE, now=now)


def test_a_holder_that_recommits_every_epoch_is_scored_as_a_static_one_and_loses_the_seat_with_it():
    static_w, static_h, static = _run()
    again_w, again_h, again = _run(recommit_from=FULL_FROM + 1)
    hs = again_w.state.head(again_w.heads[again_h].hash)
    assert hs is not None and hs.predecessors and hs.since > FULL_FROM + 1     # a successor
    opened = min(e for e, ev in static.items() if ev == "open")
    assert opened > FULL_FROM and again == static
    # Exactly the same evidence against R, label for label.
    now = opened - 1
    assert _strata(again_w, again_h, now) == _strata(static_w, static_h, now)


def test_a_successor_that_adds_the_row_is_scored_on_it_from_its_commit():
    _w, _h, static = _run()
    w, h, added = _run(add_at=FULL_FROM + 4)
    assert "open" in static.values() and "open" not in added.values()
    hs = w.state.head(w.heads[h].hash)
    side = seat_v4.side_of(hs)
    assert NEW not in hs.rows_since and side.line_since() < side.since
    # Its labels from its commit on are read on its own row.
    pairs = w.window.pairs[hs.since]
    keep = labels_v4.eligible(pairs, hs.since, side, labels_v4.REFERENCE, now=hs.since,
                              raw_x=w.window.odds[hs.head][hs.since])
    assert keep[np.array([m == NEW for m in pairs.models])].any()


def _side_world():
    """A two-epoch window: epoch 5 and 6 labels of both entries, a head's
    stored odds with or without NEW's row."""
    from fugal_subnet.champion import EpochPairs, Window

    n = 8
    w = Window.empty(48)
    pairs, odds = {}, {}
    for e in (5, 6):
        models = tuple(ENTRIES[i % 2] for i in range(n))
        p = labels_v4.label_pairs(EpochPairs(
            question_ids=tuple(f"q{e}-{i}" for i in range(n)), models=models,
            benchmarks=("b",) * n, categories=("c",) * n,
            y=np.array([labels_v4.ODDS_SCALE * (i % 3 == 0) for i in range(n)], np.int32),
            prompt_tokens=np.zeros(n, np.int32), completion_tokens=np.zeros(n, np.int32)),
            [("ck-v",)] * n, np.full(n, -1, np.int16))
        pairs[e] = p
    import dataclasses
    return dataclasses.replace(w, pairs=pairs), odds


def test_the_rule_on_one_epochs_labels():
    """Line from 5, successor's commit 6. A missing row is read from 5 at odds
    0; an added row from 6 on its own odds and at odds 0 at 5; a changed row
    (neither inherited nor added) from 6 only; a head of another coldkey (no
    inherited rows) from its own commit."""
    import dataclasses

    w, _ = _side_world()
    lacks = np.where([m == NEW for m in w.pairs[5].models], -1, 7000).astype(np.int16)
    has = np.full(len(lacks), 7000, np.int16)
    succ_lacks = labels_v4.Side("s", "ck-h", 6, {ENTRIES[0]: 5})
    succ_adds = labels_v4.Side("s", "ck-h", 6, {ENTRIES[0]: 5}, added=frozenset({NEW}))
    succ_changed = labels_v4.Side("s", "ck-h", 6, {ENTRIES[0]: 5})
    other = labels_v4.Side("s", "ck-x", 6, {})
    on_new = np.array([m == NEW for m in w.pairs[5].models])

    def read(side, raw, epoch):
        win = dataclasses.replace(w, odds={"s": {5: raw, 6: raw}})
        keep = labels_v4.eligible(w.pairs[epoch], epoch, side, labels_v4.REFERENCE, now=6,
                                  raw_x=raw)
        return keep, labels_v4._scored(win, None, side, epoch)

    keep, o = read(succ_lacks, lacks, 5)
    assert keep[on_new].all() and (o[on_new] == 0).all()             # from the line, odds 0
    keep, o = read(succ_adds, has, 5)
    assert keep[on_new].all() and (o[on_new] == 0).all()             # the line had no row
    keep, o = read(succ_adds, has, 6)
    assert keep[on_new].all() and (o[on_new] == 7000).all()          # its own row
    keep, _o = read(succ_changed, has, 5)
    assert not keep[on_new].any() and keep[~on_new].all()            # a changed row: from 6
    keep, _o = read(other, lacks, 5)
    assert not keep.any()                                            # nothing before its commit
    assert succ_lacks.line_since() == 5 and other.line_since() == 6
    with pytest.raises(ValueError):
        labels_v4.Side(None, added=frozenset({NEW}))


def test_a_head_under_another_coldkey_inherits_nothing():
    w, h, events = _run(owner_from=FULL_FROM + 1)
    hs = w.state.head(w.heads[h].hash)
    assert hs is not None and hs.predecessors == () and hs.rows_since == {}
    assert seat_v4.side_of(hs).line_since() == hs.since


def test_a_row_the_line_added_is_published_and_read_the_same_by_every_later_generation():
    """seat_v4.observe records the row a successor adds (`HeadState.added`),
    the state publishes it and the line origin's commit ("hash:since"), and
    side_of hands both to labels_v4.Side: a second successor that inherits
    the added row still scores its labels from the line's start up to the
    adder's commit at odds 0, exactly as the adder does, and a rebuild of the
    published state gives the same sides."""
    add_at, again_at = FULL_FROM + 3, FULL_FROM + 5
    w = World()
    w.missing_after[NEW] = FULL_FROM
    w.add("h0", skill=0.3, rows=(ENTRIES[0],), hotkey="hk-h", owner="ck-h", block=5)
    cur = "h0"
    for e in range(1, again_at + 3):
        if e in (add_at, again_at):
            cur = f"h{e}"
            w.add(cur, skill=0.3, rows=ENTRIES, hotkey="hk-h", owner="ck-h",
                  block=(e - 1) * BPE + 1)
        w.epoch(e, served=[cur], answered=[cur])
        if e == add_at:
            adder = w.state.head(w.heads[cur].hash)
    hs = w.state.head(w.heads[cur].hash)
    assert adder.added == frozenset({NEW}) and NEW not in adder.rows_since
    assert hs.predecessors[0] == adder.head and NEW in hs.rows_since and NEW in hs.added
    back = seat_v4.from_dict(seat_v4.to_dict(w.state))
    assert back == w.state and seat_v4.side_of(back.head(hs.head)) == seat_v4.side_of(hs)
    side = seat_v4.side_of(hs)
    origin_since = hs.predecessor_since[-1]
    assert side.line == origin_since and side.line_since() == origin_since
    # Between the line's start and the adder's commit NEW is scored at odds 0
    # for both generations, and read by both.
    ref = labels_v4.reference(w.window, now=again_at + 2)
    checked = 0
    for e in range(FULL_FROM, adder.since):
        if e not in w.window.pairs:
            continue
        on_new = np.array([m == NEW for m in w.window.pairs[e].models])
        for s in (side, seat_v4.side_of(adder)):
            o = labels_v4._scored(w.window, ref, s, e)
            if o is None:
                continue
            assert (o[on_new] == 0).all()
            checked += int(on_new.any())
            keep = labels_v4.eligible(w.window.pairs[e], e, s, labels_v4.REFERENCE,
                                      now=again_at + 2, raw_x=labels_v4._raw(w.window, s, e))
            assert keep[on_new & (w.window.pairs[e].missing != labels_v4.PROVISIONAL_LABEL)].all()
    assert checked >= 2
