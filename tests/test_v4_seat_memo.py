"""Protocol v4: the seat reads the memorisation verdict (SEAT_MINIMAL_V4.md
3.6, 3.8, 6.2; fugal_subnet/seat_v4.py, fugal_subnet/memo_v4.py).

When `run_v2_epoch` is handed the epoch's memo, every head the seat keeps
(serving or not: the holder, the history and every standing head) is judged
on the window as it stands after the epoch is added, from the seat's own
view of it (a successor from its inherited rows' commit), before the seat is
decided. A refused holder is not the bar and the seat opens to R; a refused
head is no contender and cannot take the seat; a refused head is not paid.
The published `memo_v4` evidence is the verdict the seat read. Without a
memo the epoch is decided with no head refused and carries no memorisation
fields.

The epochs run on fugal_subnet/attacks/memo_seat_world.py: a window of
published and fresh labels where `mem` and `line1` know published outcomes
and nothing fresh, `honest` knows a third of every outcome.
"""
from __future__ import annotations

import json

import pytest

from fugal_subnet import labels_v4, memo_v4, seat_v4
from fugal_subnet.attacks import memo_seat_world as msw
from fugal_subnet.v2_epoch import _digest_of, load_window, save_window


@pytest.fixture(scope="module")
def world():
    return msw.World()


def H(name):
    return msw.hash_of(name)


def _refused(r):
    return {h for h, ev in r.reveal["memo_v4"]["evidence"].items() if ev["refused"]}


def _digests_cover_the_decision(r):
    rv = r.reveal
    assert rv["v2_state"]["memo_digest"] == _digest_of(rv["memo_v4"])
    assert rv["v2_state"][seat_v4.DIGEST_KEY] == seat_v4.digest(rv[seat_v4.REVEAL_KEY])


def test_a_refused_holder_is_not_the_bar_and_is_not_paid(world):
    plain = world.run(holder="mem", kept=["mem"], served=["mem"], memo=False)
    assert plain.decision.event == "hold" and plain.decision.bar == H("mem")
    assert dict(zip(plain.uids, plain.weights)) == {msw.UID["mem"]: 1.0}
    assert "memo_v4" not in plain.reveal and "memo_digest" not in plain.reveal["v2_state"]
    assert plain.reveal[seat_v4.REVEAL_KEY]["holder"]["refused"] is False

    r = world.run(holder="mem", kept=["mem"], served=["mem"], memo=True)
    ev = r.reveal[seat_v4.REVEAL_KEY]
    assert _refused(r) == {H("mem")}
    # The seat opens to R; the holder keeps it (no qualifier) and is not paid.
    assert r.decision.event == "open" and r.decision.bar is None and ev["open"]
    assert r.decision.champion == H("mem") and ev["holder"]["refused"] is True
    assert ev["paid"] == {H("mem"): False} and H("mem") not in r.decision.payable
    assert dict(zip(r.uids, r.weights)) == {0: 1.0}
    _digests_cover_the_decision(r)


def test_a_refused_head_cannot_take_the_seat(world):
    plain = world.run(holder=None, kept=["mem"], served=["mem"], memo=False)
    assert plain.decision.event == "fill" and plain.decision.champion == H("mem")
    assert dict(zip(plain.uids, plain.weights)) == {msw.UID["mem"]: 1.0}

    r = world.run(holder=None, kept=["mem"], served=["mem"], memo=True)
    doc = r.reveal[seat_v4.REVEAL_KEY]["heads"][H("mem")]
    assert r.decision.event == "empty" and r.decision.champion is None
    assert doc["refused"] is True and not doc["contender"] and not doc["qualifies"]
    assert doc["reasons"] == ["refused by the memorisation check"]
    assert r.reveal[seat_v4.REVEAL_KEY]["order"] == []
    assert dict(zip(r.uids, r.weights)) == {0: 1.0}
    _digests_cover_the_decision(r)


def test_a_refused_head_is_no_earlier_contender_for_another(world):
    """`honest` is earlier than `mem`. Both stand; `mem` is refused, so it is
    not in rule C's order, and `honest` fills the seat either way."""
    plain = world.run(holder=None, kept=["mem", "honest"], served=["mem", "honest"], memo=False)
    r = world.run(holder=None, kept=["mem", "honest"], served=["mem", "honest"], memo=True)
    assert H("mem") in plain.reveal[seat_v4.REVEAL_KEY]["order"]
    assert r.reveal[seat_v4.REVEAL_KEY]["order"] == [H("honest")]
    assert _refused(r) == {H("mem")}
    assert r.decision.event == "fill" and r.decision.champion == H("honest")
    assert dict(zip(r.uids, r.weights)) == {msw.UID["honest"]: 1.0}


def test_a_refused_holder_that_is_not_serving_is_still_refused(world):
    plain = world.run(holder="mem", kept=["mem"], served=[], memo=False)
    assert plain.decision.event == "hold" and plain.decision.bar == H("mem")

    r = world.run(holder="mem", kept=["mem"], served=[], memo=True)
    assert H("mem") not in r.reveal["v2_state"]["serving"]
    # Its evidence is published, and the seat read it.
    assert set(r.reveal["memo_v4"]["evidence"]) == {H("mem")}
    assert _refused(r) == {H("mem")}
    assert r.decision.event == "open" and r.decision.bar is None
    assert r.reveal[seat_v4.REVEAL_KEY]["holder"]["refused"] is True
    _digests_cover_the_decision(r)


def test_a_successor_is_judged_from_its_inherited_rows_commit(world):
    """line2 succeeds the holder line1 this epoch (its rows byte for byte,
    committed in the epoch before). Its menu rows count from line1's commit,
    so the check reads the whole window's labels and refuses it as it would
    line1; judged from its own commit it would have almost no labels and
    would not be refused."""
    plain = world.run(holder="line1", kept=["line1"], served=["line2"], memo=False)
    hs = plain.decision.state.head(H("line2"))
    assert plain.decision.champion == H("line2") and hs.predecessors == (H("line1"),)
    assert hs.rows_since == {e: msw.COMMIT for e in msw.ENTRIES} and hs.since == msw.NOW - 1
    assert plain.decision.event == "hold"

    r = world.run(holder="line1", kept=["line1"], served=["line2"], memo=True)
    published = r.reveal["memo_v4"]["evidence"][H("line2")]
    ref = labels_v4.reference(r.window, now=msw.NOW)
    inherited = memo_v4.memo_evidence(
        r.window, ref, labels_v4.Side(H("line2"), "ck-line", hs.since, dict(hs.rows_since)),
        world.published(), now=msw.NOW)
    own = memo_v4.memo_evidence(r.window, ref, labels_v4.Side(H("line2"), "ck-line", hs.since),
                                world.published(), now=msw.NOW)
    assert published == memo_v4.evidence_for_reveal(inherited) and published["refused"] is True
    assert not own.refused and own.seen_labels + own.unseen_labels < memo_v4.MEMO_MIN_LABELS
    assert r.decision.event == "open" and r.decision.champion == H("line2")
    assert dict(zip(r.uids, r.weights)) == {0: 1.0}
    _digests_cover_the_decision(r)


def test_a_successors_evidence_reads_its_predecessors_odds_when_it_has_none_of_its_own(
        world, monkeypatch):
    """The same epoch with line2's odds before its commit (NOW - 1) not in
    the window: they are line1's stored odds, read through the seat's view
    of it (seat_v4.side_of, labels_v4.predecessor_odds), and none is
    computed as late odds. The memorisation evidence and the seat's are the
    inherited view's, label for label; without the read it would be judged
    on the late odds one epoch computes, fewer labels, and not refused."""
    import dataclasses

    full = world.run(holder="line1", kept=["line1"], served=["line2"], memo=True)
    bare = world.window(["line1"])
    line2 = H("line2")
    bare = dataclasses.replace(bare, odds={**bare.odds, line2: {
        e: o for e, o in bare.odds[line2].items() if e >= msw.NOW - 1}})
    r = world.run(holder="line1", kept=["line1"], served=["line2"], memo=True, window=bare)
    assert sorted(r.window.odds[line2]) == [msw.NOW - 1, msw.NOW]
    assert min(full.window.odds[line2]) < msw.NOW - 1
    assert line2 not in r.reveal["late_odds"] and line2 not in full.reveal["late_odds"]
    assert r.reveal["memo_v4"] == full.reveal["memo_v4"]
    assert r.reveal["memo_v4"]["evidence"][line2]["refused"] is True
    assert r.reveal[seat_v4.REVEAL_KEY] == full.reveal[seat_v4.REVEAL_KEY]
    assert r.decision.event == "open"

    monkeypatch.setattr(labels_v4, "predecessor_odds", lambda *a: None)
    starved = world.run(holder="line1", kept=["line1"], served=["line2"], memo=True, window=bare)
    ev = starved.reveal["memo_v4"]["evidence"][line2]
    got = r.reveal["memo_v4"]["evidence"][line2]
    assert ev["refused"] is False and ev["seen"] + ev["unseen"] < got["seen"] + got["unseen"]


def test_a_refused_holder_not_serving_loses_the_seat_to_an_honest_qualifier(world):
    """Without the memo the holder keeps the seat while it stands (unpaid:
    it did not answer this epoch); with it the seat opens and the honest
    head, which beats R, takes it at once."""
    plain = world.run(holder="mem", kept=["mem", "honest"], served=["honest"], memo=False)
    assert plain.decision.event == "hold" and plain.decision.champion == H("mem")
    r = world.run(holder="mem", kept=["mem", "honest"], served=["honest"], memo=True)
    assert r.decision.event == "take" and r.decision.champion == H("honest")
    assert r.decision.state.history[0] == H("mem")
    # The former holder's history share is unpaid (refused, and not live), so
    # it goes to the holder (3.8).
    assert r.reveal[seat_v4.REVEAL_KEY]["paid"] == {H("honest"): True, H("mem"): False}
    assert dict(zip(r.uids, r.weights)) == {msw.UID["honest"]: 1.0}


def test_the_published_evidence_is_every_kept_head_and_honest_heads_pass(world):
    r = world.run(holder="mem", kept=["mem", "honest"], served=["honest"], memo=True)
    evidence = r.reveal["memo_v4"]["evidence"]
    assert set(evidence) == {hs.head for hs in r.decision.state.heads} == {H("mem"), H("honest")}
    assert evidence[H("honest")]["refused"] is False
    assert evidence[H("mem")]["refused"] is True
    assert r.reveal[seat_v4.REVEAL_KEY]["heads"][H("honest")]["refused"] is False


SCENARIOS = [dict(holder="mem", kept=["mem"], served=["mem"]),
             dict(holder=None, kept=["mem", "honest"], served=["mem", "honest"]),
             dict(holder="mem", kept=["mem", "honest"], served=["honest"]),
             dict(holder="line1", kept=["line1"], served=["line2"])]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_root_and_follower_with_the_same_memo_publish_the_same_bytes(world, scenario, tmp_path):
    """A follower handed every input in reverse order, with its window and
    seat read back from its saved state (as a restarted or synced node holds
    them), publishes byte-identical `memo_v4`, `seat_v4`, `v2_state` (both
    digests) and weights."""
    root = world.run(memo=True, **scenario)
    path = str(tmp_path / "w.npz")
    save_window(path, world.window(scenario["kept"], scenario["holder"]))
    seat = seat_v4.from_dict(json.loads(json.dumps(seat_v4.to_dict(
        world.seat(scenario["holder"], scenario["kept"])))))
    follower = world.run(memo=True, fed=lambda xs: list(reversed(list(xs))),
                         window=load_window(path, 48), seat=seat, **scenario)
    for key in ("memo_v4", seat_v4.REVEAL_KEY, "v2_state", "champion"):
        assert json.dumps(root.reveal[key], sort_keys=True) == json.dumps(
            follower.reveal[key], sort_keys=True), key
    assert (root.uids, root.weights) == (follower.uids, follower.weights)
    assert _refused(root)            # every scenario refuses a head


def test_without_a_memo_nothing_is_refused_and_no_memo_field_is_written(world):
    for sc in SCENARIOS:
        r = world.run(memo=False, **sc)
        ev = r.reveal[seat_v4.REVEAL_KEY]
        assert not any(d["refused"] for d in ev["heads"].values())
        assert ev["holder"] is None or ev["holder"]["refused"] is False
        assert "memo_v4" not in r.reveal and "memo_digest" not in r.reveal["v2_state"]
