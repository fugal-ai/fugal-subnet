"""Protocol v4: when a follower re-anchors on the root, and what a replay
reads (docs/INVARIANTS.md, the v4 replay after a re-anchor).

- The root's first reveal after a gap in its reveals (the root was down
  while the followers ran) is a re-anchor point: a follower whose state
  differs re-anchors on it at once and replays its own epochs since, so it
  publishes the root's state from the first epoch that reveal can be read
  in. During the outage each follower published its own best knowledge.
- A follower restarted without its window file re-anchors at once on the
  root's newest reveal and replays its own epochs since: from its first
  epoch back it publishes the root's state.
- A replayed epoch whose seat names a holder the node's record of the epoch
  does not hold reads the holder's head from the node's head store, then
  the root's head publication, each checked by hash.

The loop scenarios run the shipped validator loop (tests/v4_multi_loop.py).
No network, no spend.
"""
from __future__ import annotations

import hashlib
import logging

import pytest

from fugal_subnet import seat_v4, v4_replay
from neurons import validator
from tests import v4_world as w

# --- the root's first reveal after an outage -----------------------------------------------


def _reveals(*epochs):
    return {e: (f"/x/{e}", {"epoch_id": f"e{e:08d}"}) for e in epochs}


def test_the_roots_first_reveal_after_a_gap_it_was_down_for_is_a_reanchor_point():
    own = _reveals(24, 25, 26, 27, 28)
    assert validator._root_came_back(_reveals(24, 25, 27), own, 27, None)
    # no gap in the root's reveals, or a gap this node did not run either
    assert not validator._root_came_back(_reveals(24, 25, 26, 27), own, 27, None)
    assert not validator._root_came_back(_reveals(24, 25, 27), _reveals(24, 25, 27, 28), 27,
                                         None)
    # nothing before it to tell a gap from a start, or re-anchored since
    assert not validator._root_came_back(_reveals(27), own, 27, None)
    assert not validator._root_came_back(_reveals(24, 25, 27), own, 27, 29)


# --- a replay's holder head ------------------------------------------------------------------


def _store(tmp_path, held: dict):
    store = v4_replay.Store(tmp_path / "replay")
    (tmp_path / "replay" / "heads").mkdir(parents=True, exist_ok=True)
    for sha, data in held.items():
        (tmp_path / "replay" / "heads" / sha).write_bytes(data)
    return store


def _kw(heads=None):
    from fugal_subnet.v2_epoch import PicksInput

    return {"picks": PicksInput(state=None, heads=dict(heads or {}))}


def test_a_replay_reads_a_holder_its_record_lacks_from_the_head_store_then_the_publication(
        tmp_path):
    a, b = b"head-a", b"head-b"
    ha, hb = hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest()
    seat = seat_v4.SeatStateV4(champion=ha, champion_since=1, holders=((9, hb),))
    doc = {"odds": {"heads": {}}}
    store = _store(tmp_path, {ha: a})
    asked = []

    def publication(h):
        asked.append(h)
        return {hb: b}.get(h)
    got = v4_replay._holder_heads(_kw(), doc, seat, 11, store, publication)
    assert got["picks"].heads == {ha: a, hb: b}
    assert asked == [hb]                 # the store first; the publication only for what it lacks


def test_bytes_that_do_not_hash_to_the_holder_are_not_taken(tmp_path):
    a = b"head-a"
    ha = hashlib.sha256(a).hexdigest()
    seat = seat_v4.SeatStateV4(champion=ha, champion_since=1, holders=((9, ha),))
    got = v4_replay._holder_heads(_kw(), {"odds": {"heads": {}}}, seat, 11,
                                  _store(tmp_path, {}), lambda h: b"other bytes")
    assert got["picks"].heads == {}


def test_a_holder_the_record_holds_or_that_served_is_not_sought(tmp_path):
    a = b"head-a"
    ha = hashlib.sha256(a).hexdigest()
    seat = seat_v4.SeatStateV4(champion=ha, champion_since=1, holders=((9, ha),))

    def boom(h):
        raise AssertionError("not sought")
    kw = _kw({ha: a})
    assert v4_replay._holder_heads(kw, {"odds": {"heads": {}}}, seat, 11,
                                   _store(tmp_path, {}), boom) is kw
    kw = _kw()
    assert v4_replay._holder_heads(kw, {"odds": {"heads": {"3": ha}}}, seat, 11,
                                   _store(tmp_path, {}), boom) is kw


# --- the shipped loop ------------------------------------------------------------------------


def _run(tmp_path, override, last_offset=14):
    from fugal_subnet import menu_rules as mr
    from tests import v4_multi_loop as ml
    from tests.signed_list_fixtures import Signer
    from tests.test_menu_publish import LIST_BYTES

    logging.getLogger().setLevel(logging.WARNING)
    patch = pytest.MonkeyPatch()
    try:
        s = Signer(tmp_path, "release")
        patch.setattr(mr, "list_signers", lambda: s.signers_line().encode())
        (tmp_path / "w").mkdir()
        return ml.run_seed(0, tmp_path / "w", last_offset, s.sign(LIST_BYTES), patch,
                           override=override)
    finally:
        patch.undo()


def _clean(res):
    assert not res.get("crash"), res.get("tb")
    assert res["errors"] == [] and res["clock_errors"] == {} and res["future_reads"] == []


def test_a_take_during_a_root_outage_agrees_from_the_first_epoch_the_roots_return_is_read(
        tmp_path):
    """The root is down for the one epoch in which a sharper head answers a
    1,000-question challenge; the followers run it and seat that head. The
    root comes back at the next epoch without it. Its reveal of that epoch is
    read two epochs later: from then on every follower publishes the root's
    state and weights, with no drift alarm."""
    switch = w.SEAT_FIRST + 6

    def override(sc):
        sc = dict(sc)
        sc.update(holder_quiet=w.SEAT_FIRST + 4, switch=switch, switch_size="1000",
                  succeed=10**6, other_at=10**6, misses={})
        sc["restarts"] = {"root": [(switch * w.BPE - 1, (switch + 1) * w.BPE + 2, ())]}
        sc["kinds"] = {"root": [(switch, "outage1")]}
        return sc
    res = _run(tmp_path, override)
    _clean(res)
    back = switch + 1
    assert res["missing"] == [(switch, "root")]
    late = [(k, n) for k, n, _what in res["diff"] + res["recovery_diff"] if k >= back + 2]
    assert late == [], res["diff"] + res["recovery_diff"]
    assert not [a for a in res["alarms"] if "ALARM" in a[1]], res["alarms"]


def test_a_follower_that_lost_its_window_publishes_the_roots_state_from_its_first_epoch_back(
        tmp_path):
    at = w.SEAT_FIRST + 5

    def override(sc):
        sc = dict(sc)
        sc.update(holder_quiet=w.SEAT_FIRST + 4, switch=w.SEAT_FIRST + 9, switch_size="1000",
                  succeed=10**6, other_at=10**6, misses={})
        sc["restarts"] = {"f1": [(at * w.BPE + 3, at * w.BPE + 5, ("window",))]}
        sc["kinds"] = {"f1": [(at, "no-miss", ("window",))]}
        return sc
    res = _run(tmp_path, override)
    _clean(res)
    assert ("f1#1", "window") in [(n, what) for n, what, _b in res["deleted"]]
    assert res["diff"] == [] and res["recovery_diff"] == [], res["diff"] + res["recovery_diff"]
    assert not [a for a in res["alarms"] if "ALARM" in a[1]], res["alarms"]


@pytest.mark.slow
@pytest.mark.parametrize("seed", [9, 10, 13])
def test_seeded_restarts_and_root_outages_agree_outside_recovery(tmp_path, seed):
    from fugal_subnet import menu_rules as mr
    from tests import v4_multi_loop as ml
    from tests.signed_list_fixtures import Signer
    from tests.test_menu_publish import LIST_BYTES

    patch = pytest.MonkeyPatch()
    try:
        s = Signer(tmp_path, "release")
        patch.setattr(mr, "list_signers", lambda: s.signers_line().encode())
        (tmp_path / "w").mkdir()
        res = ml.run_seed(seed, tmp_path / "w", 16, s.sign(LIST_BYTES), patch)
    finally:
        patch.undo()
    _clean(res)
    assert res["diff"] == [] and res["end_state"] == [] and res["td_menu_split"] == [], res

