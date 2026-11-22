"""Protocol v4: a successor's odds before its own commit are its
predecessors' stored odds (SEAT_MINIMAL_V4.md 4.4: a successor is its
predecessor for the rules, its odds on their rows are theirs;
labels_v4.predecessor_odds, read by labels_v4._scored and by
v2_epoch.late_epochs).

1. The lookup: the most recent predecessor that has odds for the epoch and
   was its line's head then, two deep and MAX_PREDECESSORS (the evidence window) deep, none
   back; a predecessor with gaps; odds a predecessor got after it was
   succeeded are not read; a head that is not a successor (a changed row,
   another coldkey, another hotkey) reads nothing.
2. Exact: a successor's evidence read through its predecessors equals its
   evidence on its own bytes' odds, including when it added rows.
3. Late odds: `late_epochs` never lists an epoch the predecessors cover, and
   with no successor it lists what it listed before.
4. The seat at the design's late-odds bound (32 head-epochs a reveal, 31
   other hotkeys re-committing each epoch to fill it): a better challenger
   takes the seat from a holder that re-commits its head each epoch in the
   epoch it takes it from one that does not, and R opens the seat on such a
   holder in the epoch it opens it on the other.
5. The published seat state carries each predecessor's commit epoch.
"""
from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest

from fugal_subnet import labels_v4, seat_v4
from fugal_subnet.attacks import run_miner_attacks as suite
from fugal_subnet.champion import EpochPairs, Window
from fugal_subnet.labels_v4 import REFERENCE, Side
from fugal_subnet.v2_epoch import late_epochs
from tests.seat_v4_world import ENTRIES, World, hash_of

# --- a window built by hand ----------------------------------------------------------------


def _pairs(n: int = 6) -> labels_v4.LabelPairs:
    return labels_v4.label_pairs(
        EpochPairs(question_ids=tuple(f"q{i}" for i in range(n)),
                   models=tuple(ENTRIES[i % 2] for i in range(n)), benchmarks=("b",) * n,
                   categories=("c",) * n, y=np.zeros(n, np.int32),
                   prompt_tokens=np.zeros(n, np.int32), completion_tokens=np.zeros(n, np.int32)),
        [("ck-v",)] * n, np.full(n, 5000, np.int16))


def _window(epochs, odds_at: dict[str, list[int]]) -> Window:
    """A window over `epochs` whose head h holds odds (all equal to a value
    naming h and the epoch) at the epochs odds_at[h] lists."""
    pairs = {e: _pairs() for e in epochs}
    odds = {h: {e: np.full(6, (sum(map(ord, h)) * 7 + e) % 9000, np.int16) for e in es}
            for h, es in odds_at.items()}
    return dataclasses.replace(Window.empty(48), pairs=pairs, odds=odds)


def _h(tag: str) -> str:
    return hash_of(f"successor-odds:{tag}")


def test_the_most_recent_predecessor_that_was_head_then_is_read_two_deep():
    s, p1, p2 = _h("s"), _h("p1"), _h("p2")
    # p2 was the line's head until 15, p1 from 15 until 20, s from 20.
    side = Side(s, "ck", 20, {ENTRIES[0]: 10}, chain=((p1, 20), (p2, 15)))
    w = _window(range(5, 25), {s: [20, 21], p1: [16, 17, 12], p2: [10, 11, 12, 13, 14, 17, 18]})
    got = {e: labels_v4.predecessor_odds(w, side, e) for e in range(5, 25)}
    assert got[16] is w.odds[p1][16] and got[17] is w.odds[p1][17]
    assert got[12] is w.odds[p1][12]               # its own late odds come first
    for e in (10, 11, 13, 14):
        assert got[e] is w.odds[p2][e]
    # A gap in p1's own epochs is not filled from p2 (p2's odds at 18 were
    # computed after it was succeeded: rows added since may count by then).
    for e in (15, 18, 19):
        assert got[e] is None
    for e in (5, 6, 7, 8, 9):                         # nobody holds them
        assert got[e] is None
    for e in range(20, 25):                           # its own epochs are its own
        assert got[e] is None
    # _scored reads the same, and its own stored odds before them.
    ref = None
    assert np.array_equal(labels_v4._scored(w, ref, side, 16), w.odds[p1][16].astype(np.int64))
    assert np.array_equal(labels_v4._scored(w, ref, side, 21), w.odds[s][21].astype(np.int64))
    assert labels_v4._scored(w, ref, side, 15) is None


def test_the_cap_is_the_evidence_window():
    """A line re-committed every epoch keeps a predecessor for every epoch
    of the evidence window (seat_v4.MAX_PREDECESSORS)."""
    from fugal_subnet import config

    assert seat_v4.MAX_PREDECESSORS == config.SEAT_WINDOW_EPOCHS_V4 == labels_v4.SEAT_WINDOW_EPOCHS


def test_max_predecessors_deep_and_none_beyond():
    cap = seat_v4.MAX_PREDECESSORS
    s = _h("s-cap")
    preds = [_h(f"p-cap-{j}") for j in range(1, cap + 3)]
    since = cap + 30
    # p_j was the line's head for the single epoch since - j.
    chain = tuple((preds[j - 1], since - j + 1) for j in range(1, cap + 1))
    side = Side(s, "ck", since, {ENTRIES[0]: 10}, chain=chain)
    w = _window(range(10, since + 1), {p: [since - j] for j, p in enumerate(preds, 1)})
    for j in range(1, cap + 1):
        assert labels_v4.predecessor_odds(w, side, since - j) is w.odds[preds[j - 1]][since - j]
    for j in (cap + 1, cap + 2):                      # beyond the cap: not read
        assert labels_v4.predecessor_odds(w, side, since - j) is None
    late = late_epochs(w, {s: None}, {s: 10}, None, {s: side})
    assert sorted(late[s]) == [e for e in range(10, since + 1) if not since - cap <= e < since]


def _line(last: int, every: int = 1) -> World:
    """One hotkey's line: l0 from epoch 1, then a successor committed every
    `every` epochs through `last`, each served (and answering) from its
    commit; no late odds, so a head's own odds are those of the epochs it
    was verified in."""
    w = World(late_slots=0, labels=20, questions=4)
    w.add("l0", skill=0.3, hotkey="hk-l", owner="ck-l", block=5)
    head = "l0"
    for e in range(1, last + 1):
        if e > 1 and (e - 1) % every == 0:
            head = f"l{e}"
            w.add(head, skill=0.3, hotkey="hk-l", owner="ck-l", block=e * 10 - 1)
        w.epoch(e, served=[head], answered=[head])
    w.line_head = head
    return w


def test_the_seat_keeps_max_predecessors_and_their_commit_epochs():
    cap = seat_v4.MAX_PREDECESSORS
    last = cap + 4
    w = _line(last)
    hs = w.state.head(hash_of(f"l{last}"))
    kept = range(last - 1, last - 1 - cap, -1)
    assert hs.predecessors == tuple(hash_of(f"l{e}") for e in kept)
    assert hs.predecessor_since == tuple(kept) and hs.since == last
    side = seat_v4.side_of(hs)
    assert side.chain == tuple((hash_of(f"l{e}"), e + 1) for e in kept)
    for e in kept:
        assert labels_v4.predecessor_odds(w.window, side, e) is w.window.odds[hash_of(f"l{e}")][e]
    for e in range(1, last - cap):
        assert labels_v4.predecessor_odds(w.window, side, e) is None
    # What the cap leaves out is the successor's own late odds to compute.
    late = late_epochs(w.window, {hs.head: None}, {hs.head: hs.odds_since}, None,
                       {hs.head: side})
    assert late == {hs.head: list(range(last - cap - 1, 0, -1))}


@pytest.mark.parametrize("every", [1, 2, 5])
def test_a_line_longer_than_the_window_keeps_the_whole_windows_evidence(every):
    """A line re-committed every epoch (or less often) for longer than the
    window: every epoch of the window has odds for its head, its own or a
    predecessor's, and the predecessors it no longer keeps were its line's
    head only in epochs that have left the window."""
    window = labels_v4.SEAT_WINDOW_EPOCHS
    last = window + 12
    w = _line(last, every)
    hs = w.state.head(hash_of(w.line_head))
    assert len(hs.predecessors) <= seat_v4.MAX_PREDECESSORS
    side = seat_v4.side_of(hs)
    in_window = [e for e in sorted(w.window.pairs) if labels_v4.in_window(e, last)]
    assert len(in_window) == window
    held = [e for e in in_window if e in w.window.odds.get(hs.head, {})
            or labels_v4.predecessor_odds(w.window, side, e) is not None]
    assert held == in_window
    # The oldest predecessor kept was the line's head at the window's start
    # (or the line reaches back to its first head).
    chain = seat_v4.predecessor_chain(hs)
    oldest = w.state.head(hs.head).predecessors[-1]
    assert oldest == hash_of("l0") or min(
        e for e in w.window.odds[oldest]) <= last - window + 1, chain[-1]


@pytest.mark.parametrize("case", ["row", "coldkey", "hotkey"])
def test_a_head_that_is_not_a_successor_reads_nothing(case):
    w = World(late_slots=0)
    w.add("h", skill=0.3, rows=(ENTRIES[0],), hotkey="hk-h", owner="ck-h", block=5)
    for e in range(1, 7):
        w.epoch(e, served=["h"], answered=["h"])
    kw = dict(skill=0.3, rows=ENTRIES, hotkey="hk-h", owner="ck-h", block=65)
    if case == "row":
        kw["row_salt"] = {ENTRIES[0]: "changed"}
    elif case == "coldkey":
        kw["owner"] = "ck-someone-else"
    else:
        kw["hotkey"] = "hk-second"
        w.commits.pop("hk-h")
    w.add("h2", **kw)
    w.epoch(7, served=["h2"], answered=["h2"])
    hs = w.state.head(hash_of("h2"))
    side = seat_v4.side_of(hs)
    assert not hs.predecessors and side.chain == ()
    assert all(labels_v4.predecessor_odds(w.window, side, e) is None for e in range(1, 8))
    assert sorted(w.window.odds[hs.head]) == [7]


def test_the_reference_and_a_malformed_chain_are_refused():
    with pytest.raises(ValueError):
        Side(None, chain=((_h("p"), 3),))
    with pytest.raises(ValueError):
        Side(_h("s"), "ck", 5, chain=((_h("p"), "3"),))
    with pytest.raises(ValueError):
        dataclasses.replace(seat_v4.HeadState(
            head=_h("s"), owner="ck", hotkey="hk", block=1, since=1, seen=1, origin=_h("s"),
            origin_block=1, origin_owner="ck"), predecessors=(_h("p"),))


# --- exact: the evidence read through the predecessors is the successor's own ----------------


def _line_world(late_slots, *, added_row: bool):
    """A head whose hotkey re-commits it every epoch from 6 (its rows byte
    for byte; with `added_row` the first successor adds the second entry's
    row), served every epoch, against a better head committed later."""
    w = World(late_slots=late_slots, labels=400)
    if added_row:
        w.missing_after[ENTRIES[1]] = 1000
    first_rows = (ENTRIES[0],) if added_row else ENTRIES
    w.add("x0", skill=0.4, rows=first_rows, hotkey="hk-x", owner="ck-x", block=5)
    w.add("y", skill=0.6, block=31)
    for e in range(1, 6):
        w.epoch(e, served=["x0"] + (["y"] if e >= 4 else []), answered=["x0"])
    for e in range(6, 13):
        w.add(f"x{e}", skill=0.4, rows=ENTRIES, hotkey="hk-x", owner="ck-x", block=e * 10 - 1)
        w.epoch(e, served=[f"x{e}", "y"], answered=[f"x{e}", "y"])
    return w


@pytest.mark.parametrize("added_row", [False, True])
def test_a_successors_evidence_through_its_predecessors_equals_its_own(added_row):
    computed = _line_world(None, added_row=added_row)     # every late odds computed
    read = _line_world(0, added_row=added_row)            # none: read or nothing
    hs = read.state.head(hash_of("x12"))
    assert hs == computed.state.head(hash_of("x12")) and len(hs.predecessors) == 7
    assert sorted(read.window.odds[hs.head]) == [12]
    assert sorted(computed.window.odds[hs.head]) == list(range(hs.odds_since, 13))
    side, y = seat_v4.side_of(hs), seat_v4.side_of(read.state.head(hash_of("y")))
    if added_row:
        # The added row counts from the commit of the successor that added
        # it (6); before it the line had no row: its labels are read from the
        # line's start and scored as a missing row is on each pair (here the
        # entry's rate, 5000), for every later generation too.
        assert hs.rows_since == {ENTRIES[0]: 1, ENTRIES[1]: 6} and ENTRIES[1] in hs.added
        x0 = read.window.odds[hash_of("x0")][3]
        e1 = np.array([m == ENTRIES[1] for m in read.window.pairs[3].models])
        assert labels_v4.predecessor_odds(read.window, side, 3) is x0 and (x0[e1] == 5000).all()
        assert labels_v4.eligible(read.window.pairs[3], 3, side, REFERENCE, now=12,
                                  raw_x=labels_v4._raw(read.window, side, 3))[e1].all()
        assert (labels_v4._scored(read.window, None, side, 3)[e1] == 5000).all()
    for x_, y_ in ((side, REFERENCE), (side, y), (y, side)):
        a = labels_v4.gain_strata(read.window, labels_v4.reference(read.window, now=12),
                                  x_, y_, now=12)
        b = labels_v4.gain_strata(computed.window,
                                  labels_v4.reference(computed.window, now=12), x_, y_, now=12)
        assert a == b and sum(s.n for s in a.values()) > 0
    assert read.decisions[12].evidence == computed.decisions[12].evidence


# --- late odds -----------------------------------------------------------------------------


def test_late_epochs_never_lists_an_epoch_the_predecessors_cover():
    rng = random.Random(24)
    for trial in range(150):
        epochs = range(rng.randint(0, 20), 48)
        heads, sides, since, odds_at = {}, {}, {}, {}
        for i in range(rng.randint(1, 12)):
            h = _h(f"t{trial}-{i}")
            heads[h] = None
            since[h] = rng.choice(epochs)
            odds_at[h] = [e for e in epochs if rng.random() < 0.3]
            if rng.random() < 0.6:
                own = rng.randint(since[h], 47)
                chain, until = [], own
                for j in range(rng.randint(1, seat_v4.MAX_PREDECESSORS)):
                    p = _h(f"t{trial}-{i}-p{j}")
                    chain.append((p, until))
                    odds_at[p] = [e for e in epochs if rng.random() < 0.5]
                    until = max(since[h], until - rng.randint(1, 4))
                sides[h] = Side(h, "ck", own, chain=tuple(chain))
        w = _window(epochs, odds_at)
        slots = rng.choice([None, 0, 1, 7, 32])
        got = late_epochs(w, heads, since, slots, sides)
        for h, es in got.items():
            assert all(labels_v4.predecessor_odds(w, sides[h], e) is None
                       for e in es if h in sides), trial
        # With no successor it is what it was: every missing epoch, newest first.
        assert late_epochs(w, heads, since, slots, {}) == late_epochs(w, heads, since, slots)


# --- the seat at the design's late-odds bound ----------------------------------------------


def _helpers(w: World, e: int, n: int = 31) -> list[str]:
    """`n` hotkeys of one coldkey, each re-committing its head at epoch e."""
    out = []
    for i in range(n):
        w.add(f"helper{i}-{e}", skill=0.3, hotkey=f"hk-helper{i}", owner="ck-helpers",
              block=e * 10 - 1)
        out.append(f"helper{i}-{e}")
    return out


def test_a_challenger_takes_from_a_re_committing_holder_when_it_would_from_a_static_one():
    """run_miner_attacks.recommit_take_epoch: the holder re-commits its head
    every epoch from 12, 31 other hotkeys theirs."""
    static, _ = suite.recommit_take_epoch(False)
    took, seen = suite.recommit_take_epoch(True)
    assert static is not None and took == static, (static, took, seen)


def r_opens(recommit: bool, last: int = 40) -> int | None:
    """The epoch R opens the seat on a holder whose second row is worse than
    R: its labels of that entry were supplied by the holder's own coldkey
    until 6 (so not read for it) and it took the seat on the first; the
    holder re-commits each epoch from 8 (or keeps its head), with 31 other
    hotkeys filling the late-odds slots."""
    w = World(late_slots=32, labels=600)
    w.add("a0", skill=0.3, row_skill={ENTRIES[1]: -0.4}, hotkey="hk-a", owner="ck-a", block=5)
    own = ("ck-voter-1", "ck-a")                      # the second entry's labels are its own
    for e in range(1, 6):
        w.epoch(e, served=["a0"], answered=["a0"], voters=own)
    assert w.holder() == "a0"
    a = "a0"
    for e in range(6, last + 1):
        if recommit and e >= 8:
            a = f"a{e}"
            w.add(a, skill=0.3, row_skill={ENTRIES[1]: -0.4}, hotkey="hk-a", owner="ck-a",
                  block=e * 10 - 1)
        d = w.epoch(e, served=[a, *_helpers(w, e)], answered=[a])
        if d.evidence["open"]:
            return e
    return None


def test_r_opens_the_seat_on_a_re_committing_holder_when_it_would_on_a_static_one():
    static = r_opens(False)
    assert static is not None and r_opens(True) == static
    assert static - 8 <= seat_v4.MAX_PREDECESSORS        # every successor's chain reaches a0


# --- the published state -------------------------------------------------------------------


def test_the_published_state_carries_each_predecessors_commit_epoch():
    w = World(late_slots=0)
    w.add("l0", skill=0.3, hotkey="hk-l", owner="ck-l", block=5)
    w.epoch(1, served=["l0"], answered=["l0"])
    for e in range(2, 5):
        w.add(f"l{e}", skill=0.3, hotkey="hk-l", owner="ck-l", block=e * 10 - 1)
        w.epoch(e, served=[f"l{e}"], answered=[f"l{e}"])
    doc = seat_v4.to_dict(w.state)
    hs = w.state.head(hash_of("l4"))
    assert doc["heads"][hs.head]["predecessors"] == [
        f"{hash_of('l3')}:3", f"{hash_of('l2')}:2", f"{hash_of('l0')}:1"]
    assert seat_v4.from_dict(doc) == w.state
    p = hash_of("l3")
    for item in (p, f"{p}:", f"{p}:03", f"{p}:-3", f"{p}:3:4", f"{p.upper()}:3", f"{p}:３",
                 f"{p}:{10**12}", 3, None):
        bad = dict(doc, heads={**doc["heads"], hs.head: dict(doc["heads"][hs.head],
                                                             predecessors=[item])})
        with pytest.raises(ValueError, match="predecessors"):
            seat_v4.from_dict(bad)
