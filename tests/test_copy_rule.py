"""Protocol v2 copy rule: first-commit seniority, coldkey ownership, ghost incumbents.

docs/INVARIANTS.md, "I3/I4 — the v2 copy rule: a head must beat every senior
head by the margin".
"""
import dataclasses
import hashlib
import json
import random

import numpy as np
import pytest

from fugal_subnet import copy_rule as cr
from fugal_subnet.champion import Candidate, HeadEpoch, SeatState, seat_step
from fugal_subnet.commitments import seniority_key
from tests.test_champion import (
    PARAMS,
    Q,
    build_window,
    make_slice,
    noisy,
    oracle_router,
    seated,
)

OFF, RET = PARAMS.ghost_off, PARAMS.ghost_retire


def H(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def C(hotkey, coldkey, head, block):
    return cr.Commitment(hotkey=hotkey, coldkey=coldkey, head_hash=H(head), block=block)


def obs(state, epoch, *commitments, boundary=None, ghost_retire=RET):
    return cr.observe(state, epoch, boundary if boundary is not None else 10**9,
                      commitments, ghost_retire=ghost_retire)


def cand(state, name, eligible=True, burned_in=True, since=0):
    return Candidate(head_hash=H(name), seniority=cr.seniority(state, H(name)),
                     eligible=eligible, since_epoch=since, burned_in=burned_in)


def bars(state, epoch, window, names, holders, ghost_retire=RET):
    return cr.senior_bars(state, epoch, window, [H(n) for n in names],
                          holders=[H(n) for n in holders],
                          ghost_off=OFF, ghost_retire=ghost_retire)


def seat(champion=None, history=()):
    return SeatState(champion=None if champion is None else H(champion),
                     champion_since=None if champion is None else 0,
                     history=tuple(H(h) for h in history))


# --- rule 1: seniority is the head's first commit ---------------------------


def test_an_older_hotkey_that_copies_a_newer_head_is_junior():
    # The author registered late (block 5,000) and committed at 6,000; the
    # copier registered early (block 100) and committed its tweak at 6,500.
    # v1 ranks the copier senior; v2 ranks the author senior.
    assert seniority_key(100, 6500) < seniority_key(5000, 6000)
    s = obs(cr.CopyState(), 1, C("hk-author", "ck-a", "author", 6000),
            C("hk-copier", "ck-c", "tweak", 6500))
    assert cr.seniority(s, H("author")) < cr.seniority(s, H("tweak"))


def test_seniority_is_the_first_commit_of_the_hash_not_the_current_one():
    s = obs(cr.CopyState(), 1, C("hk1", "ck-a", "x", 100))
    # The owner re-commits the same hash later (a re-registration, a retry):
    # the record keeps the first block.
    s = obs(s, 2, C("hk2", "ck-a", "x", 900))
    assert s.get(H("x")).first_block == 100


def test_a_tie_in_one_block_is_broken_by_digest_not_by_input_order_or_hotkey():
    a, b = C("hk-zzz", "ck-a", "same", 50), C("hk-aaa", "ck-b", "same", 50)
    s1 = obs(cr.CopyState(), 1, a, b)
    s2 = obs(cr.CopyState(), 1, b, a)
    assert s1 == s2
    expect = min(("ck-a", "ck-b"),
                 key=lambda ck: hashlib.sha256(f"fugal-v2-owner:{H('same')}:{ck}".encode()).digest())
    assert s1.get(H("same")).owner == expect


def test_an_unknown_head_is_junior_to_every_known_one():
    s = obs(cr.CopyState(), 1, C("hk", "ck", "x", 10**8))
    assert cr.seniority(s, H("x")) < cr.seniority(s, H("never-committed"))


# --- rule 1b: a hash belongs to the coldkey that first committed it ----------


def test_the_same_hash_under_another_coldkey_is_a_copy():
    s = obs(cr.CopyState(), 1, C("hk-a", "ck-a", "x", 100), C("hk-b", "ck-b", "x", 101))
    v = cr.classify(s, {1: C("hk-a", "ck-a", "x", 100), 2: C("hk-b", "ck-b", "x", 101)})
    assert v == {1: "owner", 2: "copy"}


def test_two_hotkeys_of_the_owner_serving_one_hash_are_one_head():
    s = obs(cr.CopyState(), 1, C("hk-a", "ck-a", "x", 100), C("hk-a2", "ck-a", "x", 120))
    v = cr.classify(s, {7: C("hk-a2", "ck-a", "x", 120), 3: C("hk-a", "ck-a", "x", 100)})
    assert v == {3: "owner", 7: "duplicate"}


def test_an_old_hash_recommitted_by_another_coldkey_is_still_a_copy():
    s = obs(cr.CopyState(), 1, C("hk-a", "ck-a", "x", 100))
    s = obs(s, 2, C("hk-a", "ck-a", "y", 200))            # the author moves on
    s = obs(s, 3, C("hk-a", "ck-a", "y", 200), C("hk-b", "ck-b", "x", 300))
    assert s.get(H("x")).owner == "ck-a" and s.get(H("x")).first_block == 100
    assert cr.classify(s, {9: C("hk-b", "ck-b", "x", 300)}) == {9: "copy"}


def test_a_pruned_author_that_re_registers_keeps_its_head():
    s = obs(cr.CopyState(), 1, C("hk-old", "ck-a", "x", 100))
    s = obs(s, 2)                                          # hotkey deregistered
    assert not s.get(H("x")).active and s.get(H("x")).retired_at == 2
    s = obs(s, 3, C("hk-new", "ck-a", "x", 500))          # new hotkey, same coldkey
    r = s.get(H("x"))
    assert r.active and r.first_block == 100 and r.owner == "ck-a"
    assert cr.classify(s, {4: C("hk-new", "ck-a", "x", 500)}) == {4: "owner"}


def test_the_wiring_names():
    # An older hotkey (UID registered first) serving the author's exact hash
    # is a copy; the owner is the first committer and seniority is its block.
    s = obs(cr.CopyState(), 1, C("hk-author", "ck-a", "x", 6000),
            C("hk-old", "ck-old", "x", 6500), C("hk-old2", "ck-old", "t", 6400))
    assert cr.owner(s, H("x")) == ("ck-a", "hk-author")
    assert cr.first_commit_block(s, H("x")) == 6000
    assert cr.is_exact_copy(s, "ck-old", H("x")) and not cr.is_exact_copy(s, "ck-a", H("x"))
    assert not cr.is_exact_copy(s, "ck-a", H("unknown"))
    assert cr.owner(s, H("unknown")) is None and cr.first_commit_block(s, H("unknown")) is None
    assert cr.v2_seniority(s, H("x")) < cr.v2_seniority(s, H("t"))
    assert cr.v2_seniority(s, H("x")) == cr.seniority(s, H("x"))


def test_an_author_that_swaps_coldkey_keeps_its_head():
    s = obs(cr.CopyState(), 1, C("hk-author", "ck-old", "x", 100))
    # A chain coldkey swap: the same hotkey, the same commitment, a new coldkey.
    s = obs(s, 2, C("hk-author", "ck-new", "x", 100))
    r = s.get(H("x"))
    assert r.active and r.retired_at is None and r.first_block == 100
    assert cr.owner(s, H("x")) == ("ck-new", "hk-author")
    assert not cr.is_exact_copy(s, "ck-new", H("x"), hotkey="hk-author")
    assert cr.classify(s, {3: C("hk-author", "ck-new", "x", 100)}) == {3: "owner"}
    # Before the state has seen the swap, the hotkey alone keeps it eligible.
    s1 = obs(cr.CopyState(), 1, C("hk-author", "ck-old", "x", 100))
    assert not cr.is_exact_copy(s1, "ck-new", H("x"), hotkey="hk-author")


def test_a_different_hotkey_under_a_new_coldkey_is_still_a_copy():
    s = obs(cr.CopyState(), 1, C("hk-author", "ck-old", "x", 100))
    s = obs(s, 2, C("hk-author", "ck-new", "x", 100), C("hk-other", "ck-x", "x", 150))
    assert cr.is_exact_copy(s, "ck-x", H("x"), hotkey="hk-other")
    assert cr.is_exact_copy(s, "ck-old", H("x"), hotkey="hk-other")
    assert cr.classify(s, {3: C("hk-author", "ck-new", "x", 100),
                           4: C("hk-other", "ck-x", "x", 150)}) == {3: "owner", 4: "copy"}


def test_one_coldkeys_two_hotkeys_in_one_block_record_the_lowest_hotkey():
    a, b = C("hk-b", "ck", "x", 50), C("hk-a", "ck", "x", 50)
    assert obs(cr.CopyState(), 1, a, b) == obs(cr.CopyState(), 1, b, a)
    assert cr.owner(obs(cr.CopyState(), 1, a, b), H("x")) == ("ck", "hk-a")


# --- rule 1c: retirement and the state's lifetime ---------------------------


def test_a_retired_head_is_kept_for_the_retire_window_then_dropped():
    s = obs(cr.CopyState(), 1, C("hk", "ck", "x", 100), ghost_retire=3)
    s = obs(s, 2, C("hk", "ck", "y", 200), ghost_retire=3)
    assert s.get(H("x")).retired_at == 2
    assert cr.retired_hashes(s) == {H("x")}
    for e in (3, 4):
        s = obs(s, e, C("hk", "ck", "y", 200), ghost_retire=3)
        assert H("x") in s
    s = obs(s, 5, C("hk", "ck", "y", 200), ghost_retire=3)
    assert H("x") not in s


def test_a_copier_holding_a_retired_hash_keeps_it_a_copy_after_the_window():
    s = obs(cr.CopyState(), 1, C("hk", "ck", "x", 100), C("hk-c", "ck-c", "x", 150),
            ghost_retire=3)
    for e in range(2, 10):
        s = obs(s, e, C("hk", "ck", "y", 200), C("hk-c", "ck-c", "x", 150), ghost_retire=3)
    # Still on record, still the author's: the copier never inherits block 100.
    assert s.get(H("x")).owner == "ck" and not s.get(H("x")).active
    assert cr.classify(s, {5: C("hk-c", "ck-c", "x", 150)}) == {5: "copy"}


def test_after_the_record_is_gone_a_fresh_commit_is_a_new_head():
    s = obs(cr.CopyState(), 1, C("hk", "ck", "x", 100), ghost_retire=2)
    for e in range(2, 5):
        s = obs(s, e, C("hk", "ck", "y", 200), ghost_retire=2)
    assert H("x") not in s
    s = obs(s, 5, C("hk", "ck", "y", 200), C("hk-c", "ck-c", "x", 4000), ghost_retire=2)
    assert s.get(H("x")).first_block == 4000 and s.get(H("x")).owner == "ck-c"
    assert cr.seniority(s, H("y")) < cr.seniority(s, H("x"))


def test_the_retire_window_counts_from_the_first_retirement_only():
    s = obs(cr.CopyState(), 1, C("hk", "ck", "x", 100))
    s = obs(s, 2, C("hk", "ck", "y", 200))               # retired at 2
    s = obs(s, 3, C("hk", "ck", "x", 300))               # back
    s = obs(s, 4, C("hk", "ck", "y", 400))               # retired again
    assert s.get(H("x")).retired_at == 2


def test_the_shipped_retire_window_is_24_epochs():
    from fugal_subnet.config import CHAMPION_GHOST_OFF, CHAMPION_GHOST_RETIRE
    assert (CHAMPION_GHOST_RETIRE, CHAMPION_GHOST_OFF) == (24, 3)


# --- bounds and determinism -------------------------------------------------


def test_malformed_commitments_are_ignored():
    good = C("hk", "ck", "x", 100)
    bad = [
        cr.Commitment("hk1", "ck", H("x").upper(), 100),        # not lowercase hex
        cr.Commitment("hk2", "ck", "deadbeef", 100),            # not 64 chars
        cr.Commitment("hk3", "ck", H("x") + "00", 100),
        cr.Commitment("hk4", "ck", H("x"), 0),                  # no block
        cr.Commitment("hk5", "ck", H("x"), 10**9 + 1),          # after the boundary
        cr.Commitment("hk6", "ck", H("x"), True),               # a bool is not a block
        cr.Commitment("hk7", "", H("x"), 100),                  # no coldkey
        cr.Commitment("h" * 65, "ck", H("x"), 100),             # not an address
        cr.Commitment("hk8", None, H("x"), 100),
    ]
    assert cr.valid_commitments([good, *bad], 10**9) == [good]


def test_a_snapshot_is_capped_and_one_entry_per_hotkey():
    many = [C(f"hk{i}", "ck", f"h{i}", 100 + i) for i in range(cr.MAX_COMMITMENTS + 50)]
    assert len(cr.valid_commitments(many, 10**9)) == cr.MAX_COMMITMENTS
    twice = [C("hk", "ck", "late", 200), C("hk", "ck", "early", 100)]
    assert [c.head_hash for c in cr.valid_commitments(twice, 10**9)] == [H("early")]


def test_observe_is_order_independent_and_applied_once():
    snap = [C(f"hk{i}", f"ck{i % 3}", f"h{i % 5}", 100 + (i % 4)) for i in range(40)]
    first = obs(cr.CopyState(), 1, *snap)
    for seed in range(5):
        shuffled = list(snap)
        random.Random(seed).shuffle(shuffled)
        assert obs(cr.CopyState(), 1, *shuffled) == first
    with pytest.raises(ValueError):
        obs(first, 1, *snap)


def test_the_state_is_bounded_by_hotkeys_times_the_retire_window():
    s = cr.CopyState()
    for e in range(1, 60):                    # 10 hotkeys switch head every epoch
        s = obs(s, e, *[C(f"hk{i}", f"ck{i}", f"h{i}-{e}", 100 * e + i) for i in range(10)],
                ghost_retire=5)
    assert len(s.heads) <= 10 * (5 + 1)


def test_the_state_round_trips_through_the_reveal_form():
    s = obs(cr.CopyState(), 1, C("hk", "ck", "x", 100))
    s = obs(s, 2, C("hk", "ck", "y", 200))
    d = s.to_reveal()
    assert set(d) == {"epoch", "first_commit", "ownership", "first_hotkey", "active",
                      "retired_at"}
    assert d["first_hotkey"] == {H("x"): "hk", H("y"): "hk"}
    assert d["first_commit"] == {H("x"): 100, H("y"): 200}
    assert d["ownership"] == {H("x"): "ck", H("y"): "ck"}
    assert d["active"] == [H("y")] and d["retired_at"] == {H("x"): 2}
    assert cr.CopyState.from_reveal(json.loads(json.dumps(d))) == s


@pytest.mark.parametrize("mutate", [
    lambda d: d["first_commit"].update({"../x": 1}),                    # not a hash
    lambda d: d["first_commit"].update({H("x"): True}),                  # bool block
    lambda d: d["first_commit"].update({H("x"): 0}),
    lambda d: d["ownership"].update({H("x"): "c" * 65}),                 # not an address
    lambda d: d["ownership"].pop(H("x")),                                # maps disagree
    lambda d: d["first_hotkey"].pop(H("x")),
    lambda d: d["first_hotkey"].update({H("x"): ""}),
    lambda d: d["active"].append(H("nope")),
    lambda d: d.update(active=[]) or d["retired_at"].pop(H("x")),       # inactive, no retirement
    lambda d: d["retired_at"].update({H("x"): -1}),
    lambda d: d.update(first_commit=[]),
])
def test_a_malformed_reveal_state_is_refused(mutate):
    s = obs(cr.CopyState(), 1, C("hk", "ck", "x", 100))
    s = obs(s, 2, C("hk", "ck", "y", 200))
    d = json.loads(json.dumps(s.to_reveal()))
    mutate(d)
    with pytest.raises(ValueError):
        cr.CopyState.from_reveal(d)


def test_an_oversized_reveal_state_is_refused(monkeypatch):
    monkeypatch.setattr(cr, "MAX_STATE_HEADS", 1)
    s = obs(cr.CopyState(), 1, C("hk", "ck", "x", 100), C("hk2", "ck", "y", 100))
    with pytest.raises(ValueError):
        cr.CopyState.from_reveal(s.to_reveal())


def test_first_epoch_is_the_first_boundary_at_or_after_the_commit():
    s = obs(cr.CopyState(), 1, C("hk", "ck", "x", 360), C("hk2", "ck", "y", 361))
    assert cr.first_epoch(s, H("x"), 360) == 1
    assert cr.first_epoch(s, H("y"), 360) == 2
    assert cr.first_epoch(s, H("z"), 360) is None


# --- rebuild from reveals ---------------------------------------------------


def _write_reveal(root, epoch, state, with_state=True):
    d = root / f"e{epoch:08d}"
    d.mkdir()
    reveal = {"epoch_id": f"e{epoch:08d}"}
    if with_state:
        reveal["protocol_version"] = 2
        reveal["v2_state"] = {cr.REVEAL_KEY: state.to_reveal()}
    (d / "reveal.json").write_text(json.dumps(reveal))
    return d / "reveal.json"


SNAPS = [
    (10, 3600, [C("hk", "ck", "x", 100), C("hk-c", "ck-c", "x", 150)]),
    (11, 3960, [C("hk", "ck", "y", 3700), C("hk-c", "ck-c", "x", 150)]),
    (13, 4680, [C("hk", "ck", "y", 3700), C("hk2", "ck2", "z", 4000),
                cr.Commitment("hk-bad", "ck", "not-a-hash", 1)]),
]


def test_rebuild_loads_the_newest_published_state(tmp_path):
    live, states = cr.CopyState(), {}
    for e, b, cs in SNAPS:
        live = cr.observe(live, e, b, cs, ghost_retire=RET)
        states[e] = live
    paths = [_write_reveal(tmp_path, e, st) for e, st in states.items()]
    _write_reveal(tmp_path, 9, None, with_state=False)    # a v1 reveal does nothing
    assert cr.rebuild_from_reveals(tmp_path) == live
    # File order does not matter; the epoch id does.
    assert cr.rebuild_from_reveals(list(reversed(paths))) == live
    # A validator that rebuilds and keeps folding agrees with one that never stopped.
    nxt = [C("hk", "ck", "y", 3700)]
    assert (cr.observe(cr.rebuild_from_reveals(tmp_path), 14, 5040, nxt, ghost_retire=RET)
            == cr.observe(live, 14, 5040, nxt, ghost_retire=RET))


def test_rebuild_with_no_v2_state_is_none(tmp_path):
    _write_reveal(tmp_path, 9, None, with_state=False)
    assert cr.rebuild_from_reveals(tmp_path) is None


def test_a_reveal_whose_state_is_for_another_epoch_is_refused(tmp_path):
    st = cr.observe(cr.CopyState(), 10, 3600, SNAPS[0][2], ghost_retire=RET)
    _write_reveal(tmp_path, 11, st)
    with pytest.raises(ValueError):
        cr.rebuild_from_reveals(tmp_path)


def test_replay_of_chain_snapshots_matches_the_live_fold():
    live = cr.CopyState()
    for e, b, cs in SNAPS:
        live = cr.observe(live, e, b, cs, ghost_retire=RET)
    assert cr.replay(reversed(SNAPS), ghost_retire=RET) == live


# --- rule 4: a junior must beat every senior seat holder by the champion margin


def _state(*entries, epoch=9):
    return obs(cr.CopyState(), epoch, *entries)


def test_seat_holders_are_the_paid_seats_and_the_champions_ghosts():
    from fugal_subnet.champion import Ghost

    st = SeatState(champion=H("c"), champion_since=0, history=(H("p1"), H("p2"), H("p3")))
    assert cr.seat_holders(st, 3) == {H("c"), H("p1"), H("p2")}
    empty = SeatState(history=(H("p1"), H("p2"), H("p3")),
                      ghosts=(Ghost(H("p1"), 12, "retired"),))
    assert cr.seat_holders(empty, 3) == {H("p1"), H("p2")}


def test_a_near_copy_of_a_former_champion_does_not_take_the_seat():
    # S held the seat before the current champion (seat 2); J is a near-copy
    # of S. Without the copy rule J takes the seat on S's quality. With it, J
    # must beat S by the margin, and a copy cannot.
    w = build_window({H("champ"): noisy(oracle_router, 0.6, 5), H("S"): oracle_router,
                      H("J"): noisy(oracle_router, 0.002, 11)})
    s = _state(C("hk0", "ck0", "champ", 50), C("hk1", "ck1", "S", 100),
               C("hk2", "ck2", "J", 200))
    st = seat("champ", ["S"])
    cands = {H("champ"): cand(s, "champ"), H("S"): cand(s, "S", burned_in=False),
             H("J"): cand(s, "J")}
    without = seat_step(st, 9, "e9", w, cands, params=PARAMS)
    assert without.event == "take" and without.champion == H("J")
    sb = cr.senior_bars(s, 9, w, [H("champ"), H("S"), H("J")],
                        holders=cr.seat_holders(st, 3), ghost_off=OFF, ghost_retire=RET)
    assert sb[H("J")] == (H("champ"), H("S"))
    d = seat_step(st, 9, "e9", w, cands, params=PARAMS, senior_bars=sb)
    assert d.event == "hold" and d.champion == H("champ")
    assert any("does not beat senior" in msg for h, msg in d.detail if h == H("J"))


def test_a_senior_that_holds_no_seat_does_not_bar():
    # The narrowing: a better head is not put through a test against every
    # older head, only against the seats. S is senior, strong and unseated.
    w = build_window({H("champ"): noisy(oracle_router, 0.6, 5), H("S"): oracle_router,
                      H("J"): noisy(oracle_router, 0.002, 11)})
    s = _state(C("hk0", "ck0", "champ", 50), C("hk1", "ck1", "S", 100),
               C("hk2", "ck2", "J", 200))
    st = seat("champ")
    sb = cr.senior_bars(s, 9, w, [H("J")], holders=cr.seat_holders(st, 3),
                        ghost_off=OFF, ghost_retire=RET)
    assert sb[H("J")] == (H("champ"),)
    cands = {H("champ"): cand(s, "champ"), H("S"): cand(s, "S", burned_in=False),
             H("J"): cand(s, "J")}
    assert seat_step(st, 9, "e9", w, cands, params=PARAMS,
                     senior_bars=sb).champion == H("J")


def test_a_junior_that_beats_every_senior_seat_holder_takes_the_seat():
    w = build_window({H("champ"): noisy(oracle_router, 0.6, 5),
                      H("S"): noisy(oracle_router, 0.4, 7), H("J"): oracle_router})
    s = _state(C("hk0", "ck0", "champ", 50), C("hk1", "ck1", "S", 100),
               C("hk2", "ck2", "J", 200))
    cands = {H(n): cand(s, n) for n in ("champ", "S", "J")}
    d = seat_step(seat("champ", ["S"]), 9, "e9", w, cands, params=PARAMS,
                  senior_bars=bars(s, 9, w, ["champ", "S", "J"], ["champ", "S"]))
    assert d.event == "take" and d.champion == H("J")


def test_a_junior_never_bars_a_senior():
    w = build_window({H("champ"): noisy(oracle_router, 0.6, 5), H("S"): oracle_router,
                      H("J"): noisy(oracle_router, 0.002, 11)})
    s = _state(C("hk0", "ck0", "champ", 50), C("hk1", "ck1", "S", 100),
               C("hk2", "ck2", "J", 200))
    sb = bars(s, 9, w, ["champ", "S", "J"], ["champ", "S", "J"])
    assert H("J") not in sb[H("S")] and sb[H("champ")] == ()
    cands = {H("champ"): cand(s, "champ"), H("S"): cand(s, "S"), H("J"): cand(s, "J")}
    d = seat_step(seated(H("champ")), 9, "e9", w, cands, params=PARAMS, senior_bars=sb)
    assert d.event == "take" and d.champion == H("S")


def test_an_empty_seat_does_not_go_to_a_copy_of_a_former_champion():
    # S held a seat and its owner retired it; the seat is empty with no ghost
    # (S's champion ghost has expired), but S is still a paid seat in the
    # history and inside its copy-rule retire window.
    w = build_window({H("S"): oracle_router, H("J"): noisy(oracle_router, 0.002, 11)})
    s = _state(C("hk1", "ck1", "S", 100), C("hk2", "ck2", "J", 200), epoch=8)
    s = obs(s, 9, C("hk1", "ck1", "S2", 900), C("hk2", "ck2", "J", 200))
    w2 = _extend(w, [10], {H("J"): noisy(oracle_router, 0.002, 11)})
    s = obs(s, 10, C("hk1", "ck1", "S2", 900), C("hk2", "ck2", "J", 200))
    st = seat(None, ["S"])
    cands = {H("J"): cand(s, "J")}
    assert seat_step(st, 10, "e10", w2, cands, params=PARAMS).champion == H("J")
    d = seat_step(st, 10, "e10", w2, cands, params=PARAMS,
                  senior_bars=bars(s, 10, w2, ["J"], ["S"]))
    assert d.event == "empty" and d.champion is None


def test_a_committed_but_never_scored_senior_bars_nobody():
    # Committing a hash early costs nothing; without evidence it cannot bar.
    w = build_window({H("champ"): noisy(oracle_router, 0.6, 5), H("J"): oracle_router})
    s = _state(C("hk-x", "ck-x", "squat", 1), C("hk0", "ck0", "champ", 50),
               C("hk2", "ck2", "J", 200))
    sb = bars(s, 9, w, ["champ", "J"], ["squat", "champ"])
    assert H("squat") not in sb[H("J")]
    cands = {H("champ"): cand(s, "champ"), H("J"): cand(s, "J")}
    assert seat_step(seated(H("champ")), 9, "e9", w, cands, params=PARAMS,
                     senior_bars=sb).champion == H("J")


def _extend(w, epochs, policies):
    for e in epochs:
        sl, diff = make_slice(e)
        w = w.add_epoch(sl, {
            h: HeadEpoch(np.asarray(p(diff), dtype=np.int64), np.zeros(Q, bool))
            for h, p in policies.items()
        })
    return w


def test_a_retired_seat_holder_bars_its_copies_for_the_retire_window_then_stops():
    champ, J = noisy(oracle_router, 0.6, 5), noisy(oracle_router, 0.002, 11)
    w = build_window({H("champ"): champ, H("S"): oracle_router, H("J"): J})
    base = [C("hk0", "ck0", "champ", 50), C("hk2", "ck2", "J", 200)]
    s = obs(cr.CopyState(), 9, *base, C("hk1", "ck1", "S", 100), ghost_retire=3)
    # S held seat 2. Epoch 10: S's author commits a new head (not yet scored).
    st = seat("champ", ["S"])
    events = []
    for e in range(10, 14):
        s = obs(s, e, *base, C("hk1", "ck1", "S2", 3000 + e), ghost_retire=3)
        w = _extend(w, [e], {H("champ"): champ, H("J"): J})
        cs = cr.comparison_set(s, e, w, ghost_off=OFF, ghost_retire=3)
        cands = {H("champ"): cand(s, "champ"), H("J"): cand(s, "J")}
        sb = cr.senior_bars(s, e, w, cands, holders=cr.seat_holders(st, 3),
                            ghost_off=OFF, ghost_retire=3)
        d = seat_step(st, e, f"e{e}", w, cands, retired=cr.retired_hashes(s),
                      params=PARAMS, senior_bars=sb)
        events.append((cs.get(H("S")), d.event, d.champion))
        st = d.state
    # Retired at 10, ghost through 12: the copy cannot take. At 13 the ghost
    # is gone, and J (now the best live router) takes the seat.
    assert events[:3] == [("retired", "hold", H("champ"))] * 3
    assert events[3] == (None, "take", H("J"))


def test_a_fresh_copy_of_a_retired_seat_holder_is_barred_with_no_paired_evidence():
    champ, S = noisy(oracle_router, 0.6, 5), oracle_router
    w = build_window({H("champ"): champ, H("S"): S})
    s = obs(cr.CopyState(), 9, C("hk0", "ck0", "champ", 50), C("hk1", "ck1", "S", 100))
    # S retires at 10; a copy J of S starts serving at 10 and is burned in by 19.
    w = _extend(w, range(10, 20), {H("champ"): champ, H("J"): noisy(S, 0.002, 11)})
    for e in range(10, 20):
        s = obs(s, e, C("hk0", "ck0", "champ", 50), C("hk1", "ck1", "S2", 3000),
                C("hk2", "ck2", "J", 3500))
    cands = {H("champ"): cand(s, "champ"), H("J"): cand(s, "J", since=10)}
    sb = bars(s, 19, w, ["champ", "J"], ["champ", "S"])
    assert H("S") in sb[H("J")]
    d = seat_step(seat("champ", ["S"]), 19, "e19", w, cands, params=PARAMS, senior_bars=sb)
    assert d.event == "hold"
    assert any(f"vs senior {H('S')[:16]}: n=0" in msg for h, msg in d.detail if h == H("J"))


def test_a_senior_that_missed_bars_only_for_the_missed_window():
    w = build_window({H("S"): oracle_router, H("J"): noisy(oracle_router, 0.002, 11)})
    s = obs(cr.CopyState(), 9, C("hk1", "ck1", "S", 100), C("hk2", "ck2", "J", 200))
    status = []
    for e in range(10, 10 + OFF + 1):
        w = _extend(w, [e], {H("J"): noisy(oracle_router, 0.002, 11)})
        s = obs(s, e, C("hk1", "ck1", "S", 100), C("hk2", "ck2", "J", 200))
        status.append(cr.comparison_set(s, e, w, ghost_off=OFF, ghost_retire=RET).get(H("S")))
    assert status == ["missed"] * OFF + [None]


def test_no_senior_bars_is_the_seat_rule_unchanged():
    w = build_window({H("champ"): noisy(oracle_router, 0.6, 5), H("S"): oracle_router,
                      H("J"): noisy(oracle_router, 0.002, 11)})
    s = _state(C("hk0", "ck0", "champ", 50), C("hk1", "ck1", "S", 100),
               C("hk2", "ck2", "J", 200))
    cands = {H(n): cand(s, n, burned_in=(n != "S")) for n in ("champ", "S", "J")}
    base = seat_step(seated(H("champ")), 9, "e9", w, cands, params=PARAMS)
    assert seat_step(seated(H("champ")), 9, "e9", w, cands, params=PARAMS,
                     senior_bars={}) == base
    assert seat_step(seated(H("champ")), 9, "e9", w, cands, params=PARAMS,
                     senior_bars=None) == base


def test_senior_bars_do_not_depend_on_input_order():
    w = build_window({H(n): noisy(oracle_router, 0.1 * i, i) for i, n in enumerate("abcde")})
    snap = [C(f"hk{i}", f"ck{i}", n, 100 + 10 * i) for i, n in enumerate("abcde")]
    s1 = _state(*snap)
    s2 = _state(*reversed(snap))
    b1 = bars(s1, 9, w, list("abcde"), list("abcd"))
    b2 = bars(s2, 9, w, list(reversed("abcde")), list(reversed("abcd")))
    assert b1 == b2
    assert b1[H("e")] == tuple(H(n) for n in "abcd") and b1[H("a")] == ()


def test_the_copy_rule_uses_the_champion_margin_and_no_other():
    # Same heads, a margin just above what J clears over S: J does not take.
    w = build_window({H("S"): noisy(oracle_router, 0.4, 7), H("J"): oracle_router})
    s = _state(C("hk1", "ck1", "S", 100), C("hk2", "ck2", "J", 200))
    cands = {H("S"): cand(s, "S", burned_in=False), H("J"): cand(s, "J")}
    sb = bars(s, 9, w, ["S", "J"], ["S"])
    st = seat(None, ["S"])
    assert seat_step(st, 9, "e9", w, cands, params=PARAMS, senior_bars=sb).champion == H("J")
    huge = dataclasses.replace(PARAMS, margin_points=1000.0)
    assert seat_step(st, 9, "e9", w, cands, params=huge, senior_bars=sb).champion is None


def test_the_state_plugs_into_the_single_v2_state_block(tmp_path):
    # One channel: the copy rule publishes through v2_state_for_reveal's
    # `extra`, and the rebuild reads it back from there.
    from fugal_subnet.v2_epoch import v2_state_for_reveal

    st = cr.observe(cr.CopyState(), 10, 3600, SNAPS[0][2], ghost_retire=RET)
    block = v2_state_for_reveal(seat=SeatState(), owners={}, serving={}, burn_in={},
                                extra={cr.REVEAL_KEY: st.to_reveal()})
    d = tmp_path / "e00000010"
    d.mkdir()
    (d / "reveal.json").write_text(json.dumps(
        {"epoch_id": "e00000010", "protocol_version": 2, "v2_state": block}))
    assert cr.rebuild_from_reveals(tmp_path) == st
