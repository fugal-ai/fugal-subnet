"""Protocol v4's seat as a pure function (fugal_subnet/seat_v4.py;
docs/decisions/SEAT_MINIMAL_V4.md 3.3 to 3.8, 4.3, 4.4, as
docs/decisions/FAST_CROWN_V4.md changes them): every clause of the seat step
on a synthetic world (tests/seat_v4_world.py) whose heads' odds and challenge
items are set by construction, at the design's constants (300 labels, 0.1
Brier points, 500 items, 1.5 points, n0 100, G = 3). A head named in
`answered` has a counted challenge that epoch, its items paired with the
holder after the epoch two before (seat_v4.challenge_holder)."""
from __future__ import annotations

import dataclasses
import random

import pytest

from fugal_subnet import config, payout, seat_v4
from tests.seat_v4_world import ENTRIES, hash_of
from tests.seat_v4_world import World as _World


def World(**kw):
    """The seat's world with challenges of 100 questions an epoch, so rule
    B's 500 items take five epochs of an unbroken run (as the tests below
    are timed)."""
    return _World(**{"questions": 100, **kw})

SHARES = (0.85, 0.10, 0.05)


def _run(w: World, epochs, **kw):
    out = None
    for e in epochs:
        out = w.epoch(e, **kw)
    return out


def _weights(w: World, d: seat_v4.SeatDecisionV4) -> dict[str, float]:
    uids, ws = payout.seat_weights(d.champion, d.state.history, w.serving, set(d.payable),
                                   shares=SHARES, paid_fraction=1.0, burn_uid=0)
    by_uid = {u: x for u, x in zip(uids, ws)}
    name = {uid: w.name(h) for h, uid in w.serving.items()}
    return {("burn" if u == 0 else name[u]): x for u, x in by_uid.items()}


# --- constants -----------------------------------------------------------------------------


def test_the_constants_are_the_designs():
    assert seat_v4.STANDING_EPOCHS == config.STANDING_EPOCHS_V4 == 3
    assert seat_v4.ODDS_MARGIN_UNITS == 100_000                    # 0.1 Brier points
    assert seat_v4.ODDS_MIN_PAIRS == config.ODDS_MIN_PAIRS_V4 == 300
    assert seat_v4.PICKS_MARGIN_UNITS == 1_500_000                 # 1.5 points
    assert seat_v4.PICKS_MIN_QUESTIONS == 500
    assert seat_v4.BOUND_N0 == config.BOUND_N0_V4 == 100
    assert config.PAYOUT_SEAT_SHARES == SHARES


# --- the fill and the bar ------------------------------------------------------------------


def test_an_empty_seat_is_filled_once_a_head_beats_r_and_has_500_challenge_items():
    w = World()
    w.add("good", skill=0.3)
    w.add("flat", skill=0.0, block=3)
    for e in range(1, 5):
        d = w.epoch(e, served=["good", "flat"], answered=["good", "flat"])
        assert d.event == "empty" and d.champion is None and not d.payable
        assert _weights(w, d) == {"burn": 1.0}
    good = d.evidence["heads"][hash_of("good")]
    assert good["vs_bar"]["passes"] and good["picks"]["questions"] == 400
    assert "fewer than 500 challenge items" in good["reasons"][0]
    d = w.epoch(5, served=["good", "flat"], answered=["good", "flat"])
    assert d.event == "fill" and w.holder() == "good" and d.state.champion_since == 5
    assert _weights(w, d) == {"good": 1.0}
    # A head with no edge over R never fills it, however long it answers.
    w2 = World(seed=1)
    w2.add("flat", skill=0.0)
    for e in range(1, 30):
        assert w2.epoch(e, served=["flat"], answered=["flat"]).event == "empty"


def test_the_holder_is_the_bar_while_it_answers_and_r_does_not_beat_it():
    w = World()
    w.add("h", skill=0.3)
    _run(w, range(1, 6), served=["h"], answered=["h"])
    assert w.holder() == "h"
    d = w.epoch(6, served=["h"], answered=["h"])
    assert d.event == "hold" and d.bar == hash_of("h") and not d.evidence["open"]


@pytest.mark.parametrize("missed", [1, 2])
def test_a_holder_missing_one_or_two_epochs_keeps_the_seat_and_is_unpaid_for_them(missed):
    w = World()
    w.add("h", skill=0.3)
    w.add("c", skill=0.6, block=9)                       # better, and committed later
    _run(w, range(1, 6), served=["h"], answered=["h"])
    _run(w, range(6, 9), served=["h", "c"], answered=["h", "c"])
    assert w.holder() == "h"
    for e in range(9, 9 + missed):
        d = w.epoch(e, served=["c"], answered=["c"])
        # The holder still stands: it answered in one of the last three epochs.
        assert d.event == "hold" and w.holder() == "h"
        assert hash_of("h") not in d.payable and _weights(w, d) == {"burn": 1.0}
    d = w.epoch(9 + missed, served=["h"], answered=["h"])
    assert d.event == "hold" and _weights(w, d) == {"h": 1.0}


def test_after_three_missed_epochs_the_seat_opens_and_goes_to_the_latest_qualifier_at_once():
    w = World()
    w.add("h", skill=0.3)
    w.add("c1", skill=0.25, block=8)                     # worse than the holder
    w.add("c2", skill=0.3, block=9, noise=7)             # its equal, better than c1
    _run(w, range(1, 6), served=["h"], answered=["h"])
    # The challengers answer, but neither beats the holder's odds.
    for e in range(6, 12):
        d = w.epoch(e, served=["h", "c1", "c2"], answered=["h", "c1", "c2"])
    assert w.holder() == "h" and d.event == "hold"
    d = w.epoch(12, served=["c1", "c2"], answered=["c1", "c2"])
    d = w.epoch(13, served=["c1", "c2"], answered=["c1", "c2"])
    assert w.holder() == "h" and d.event == "hold"
    d = w.epoch(14, served=["c1", "c2"], answered=["c1", "c2"])
    # The holder last answered in 11: after three epochs the bar is R, the
    # seat open, and the latest qualifier (c2, which beats c1) takes it now.
    assert d.evidence["open"] and d.event == "take" and w.holder() == "c2"
    assert d.state.history == (hash_of("h"),)
    assert d.evidence["qualifiers"] == [hash_of("c1"), hash_of("c2")]
    assert _weights(w, d) == {"c2": 0.85 + 0.10 + 0.05}


def test_an_open_seat_with_no_qualifier_stays_with_the_holder_unpaid():
    w = World()
    w.add("h", skill=0.3)
    _run(w, range(1, 6), served=["h"], answered=["h"])
    for e in range(6, 10):
        d = w.epoch(e, served=[], answered=[])
    assert d.event == "open" and w.holder() == "h" and d.bar is None
    assert _weights(w, d) == {"burn": 1.0}
    # It comes back and answers: it is the bar again, and paid.
    d = w.epoch(10, served=["h"], answered=["h"])
    assert d.event == "hold" and _weights(w, d) == {"h": 1.0}


def test_r_beating_the_holder_opens_the_seat_and_the_holder_is_unpaid():
    w = World()
    w.add("h", skill=0.3, rows=(ENTRIES[0],))            # no row for the second entry
    w.missing_after[ENTRIES[1]] = 6                      # whose allowance ends at 6
    _run(w, range(1, 6), served=["h"], answered=["h"])
    assert w.holder() == "h"
    for e in range(6, 20):
        d = w.epoch(e, served=["h"], answered=["h"])
        if d.evidence["holder"]["r_beats"]["passes"]:
            break
    assert d.event == "open" and d.bar is None and w.holder() == "h"
    assert _weights(w, d) == {"burn": 1.0}


def test_a_refused_holder_opens_the_seat_is_unpaid_and_a_refused_head_cannot_take():
    w = World()
    w.add("h", skill=0.3)
    w.add("m", skill=0.9, block=9)
    _run(w, range(1, 6), served=["h"], answered=["h"])
    _run(w, range(6, 11), served=["h", "m"], answered=["h", "m"], memo=["m"])
    assert w.holder() == "h"
    d = w.decisions[10]
    assert "refused by the memorisation check" in d.evidence["heads"][hash_of("m")]["reasons"]
    d = w.epoch(11, served=["h", "m"], answered=["h", "m"], memo=["h", "m"])
    assert d.evidence["open"] and d.event == "open" and _weights(w, d) == {"burn": 1.0}
    d = w.epoch(12, served=["h", "m"], answered=["h", "m"])
    assert d.event == "take" and w.holder() == "m"


# --- rule C: precedence ----------------------------------------------------------------------


def test_the_latest_qualifier_takes_and_beat_every_other_and_order_is_not_read():
    w = World()
    for name, block, skill in (("a", 3, 0.2), ("b", 4, 0.5), ("c", 5, 0.8)):
        w.add(name, skill=skill, block=block)
    d = _run(w, range(1, 6), served=["a", "b", "c"], answered=["a", "b", "c"])
    assert d.event == "fill" and w.holder() == "c"
    assert d.evidence["qualifiers"] == [hash_of(x) for x in ("a", "b", "c")]
    # It beat every other qualifier on odds (rule C), each the margin.
    earlier = {row[0]: row[3] for row in seat_v4.earlier_rows(d.evidence, hash_of("c"))}
    assert earlier == {hash_of("a"): True, hash_of("b"): True}
    # The same epoch decided from heads and serving given in any order.
    for seed in range(3):
        w2 = World()
        names = [("a", 3, 0.2), ("b", 4, 0.5), ("c", 5, 0.8)]
        random.Random(seed).shuffle(names)
        for name, block, skill in names:
            w2.add(name, skill=skill, block=block)
        served = [n for n, *_ in names]
        d2 = _run(w2, range(1, 6), served=served, answered=served)
        assert seat_v4.to_dict(d2.state) == seat_v4.to_dict(d.state)
        assert d2.evidence == d.evidence


def test_equal_independent_heads_the_earlier_takes():
    w = World()
    w.add("early", skill=0.4, block=3, noise=1)
    w.add("late", skill=0.4, block=6, noise=2)
    d = _run(w, range(1, 6), served=["early", "late"], answered=["early", "late"])
    assert w.holder() == "early"
    late = d.evidence["heads"][hash_of("late")]
    assert late["contender"] and not late["qualifies"]
    last = seat_v4.earlier_rows(d.evidence, hash_of("late"))[-1]
    assert last[0] == hash_of("early") and not last[3]


def _odd(e):
    return e % 2 == 1


def test_a_near_copy_of_an_earlier_contender_never_passes_it_even_when_it_misses_an_epoch():
    w = World()
    w.add("h", skill=0.2)
    w.add("orig", skill=0.6, block=7)
    w.add("copy", skill=0.6, block=8, noise=5, owner="ck-other")
    _run(w, range(1, 6), served=["h"], answered=["h"])
    # The original answers the scored questions every other epoch: a
    # contender (it answered in the last three epochs) that never has the
    # 1,000 questions to take. Its near-copy answers every epoch.
    for e in range(6, 16):
        missed = e in (12, 13)                       # the original misses two epochs
        served = ["h", "copy"] + ([] if missed else ["orig"])
        answered = ["h", "copy"] + (["orig"] if e in (7, 9, 11, 14) else [])
        d = w.epoch(e, served=served, answered=answered)
        assert w.holder() == "h", e
        c = d.evidence["heads"][hash_of("copy")]
        assert c["contender"] and not c["qualifies"]
        if c["picks"] and c["picks"]["passes"]:
            last = seat_v4.earlier_rows(d.evidence, hash_of("copy"))[-1]
            assert [last[0], last[3]] == [hash_of("orig"), False]
    assert d.evidence["heads"][hash_of("copy")]["picks"]["passes"]
    # Once the original no longer bars (it last answered in 14, so it bars
    # through 16) the copy is just a better head.
    events = [w.epoch(e, served=["h", "copy"], answered=["h", "copy"]).event for e in (16, 17, 18)]
    assert events == ["hold", "take", "hold"] and w.holder() == "copy"


def test_a_contender_that_stops_answering_stops_barring_after_three_epochs():
    w = World()
    w.add("h", skill=0.2)
    w.add("early", skill=0.6, block=7)
    w.add("late", skill=0.6, block=8, noise=3)
    _run(w, range(1, 6), served=["h"], answered=["h"])
    for e in range(6, 12):
        w.epoch(e, served=["h", "early", "late"],
                answered=["h", "late"] + (["early"] if _odd(e) else []))
    assert w.holder() == "h"
    # From 12 the earlier head is served but never answers (odds only): it
    # last answered in 11, so it bars through 13.
    holders = []
    for e in range(12, 16):
        d = w.epoch(e, served=["h", "early", "late"], answered=["h", "late"])
        holders.append(w.holder())
        if e == 14:
            bars = d.evidence["heads"][hash_of("early")]
    assert holders == ["h", "h", "late", "late"]
    assert "bars no one" in bars["reasons"][0]


def test_a_contender_shown_to_route_worse_stops_barring():
    w = World()
    w.add("h", skill=0.2)
    w.add("early", skill=0.6, block=7)
    w.add("late", skill=0.6, block=8, noise=3)
    _run(w, range(1, 6), served=["h"], answered=["h"])
    # Challenges pair with the holder after the epoch two before: from 7.
    d = _run(w, range(6, 12), served=["h", "early", "late"], answered=["h", "early", "late"],
             value={"early": -3})
    # "early" routes 3 points worse than the holder: shown worse, no bar.
    early = d.evidence["heads"][hash_of("early")]
    assert early["routes_worse"]["passes"] and not early["contender"]
    assert d.event == "take" and w.holder() == "late"


def test_an_earlier_head_that_never_answers_bars_nobody():
    w = World()
    w.add("never", skill=0.6, block=3)
    w.add("late", skill=0.6, block=8, noise=3)
    d = _run(w, range(1, 6), served=["never", "late"], answered=["late"])
    assert d.event == "fill" and w.holder() == "late"


def test_better_odds_and_picks_two_points_worse_never_take():
    w = World()
    w.add("h", skill=0.2)
    w.add("c", skill=0.9, block=9)
    _run(w, range(1, 6), served=["h"], answered=["h"])
    for e in range(6, 40):
        d = w.epoch(e, served=["h", "c"], answered=["h", "c"], value={"c": -2})
    assert w.holder() == "h"
    assert not d.evidence["heads"][hash_of("c")]["contender"]


def test_precedence_ties_pre_activation_commits_at_the_floor_by_the_salted_digest():
    def order(salt):
        w = World(floor_block=100, salt=salt)
        w.add("a", block=10)
        w.add("b", block=20)
        w.add("post", block=150)
        w.epoch(20, served=["a", "b", "post"])
        hs = w.state.by_head()
        keys = {w.name(h): seat_v4.precedence(hs[h], floor_block=100, salt=salt) for h in hs}
        return sorted(keys, key=keys.get), keys

    orders = {tuple(order(s)[0]) for s in ("s1", "s2", "s3", "s4", "s5", "s6")}
    assert all(o[-1] == "post" for o in orders)
    assert len(orders) == 2                               # the salt decides a and b's order
    _o, keys = order("s1")
    assert keys["a"][0] == keys["b"][0] == 100 and keys["post"][0] == 150


def test_no_take_while_the_window_is_not_trusted():
    w = World()
    w.add("good", skill=0.3)
    d = _run(w, range(1, 7), served=["good"], answered=["good"], allow_take=False)
    assert d.event == "empty" and "not trusted" in d.evidence["heads"][hash_of("good")]["reasons"][0]


# --- pay -------------------------------------------------------------------------------------


def test_pay_is_85_10_5_with_unpaid_shares_to_the_holder_and_an_unpaid_holder_burning():
    w = World()
    w.add("a", skill=0.2, block=3)
    w.add("b", skill=0.4, block=6)
    w.add("c", skill=0.7, block=9)
    # Each challenger's items pair with the holder after the epoch two before.
    _run(w, range(1, 6), served=["a"], answered=["a"])
    _run(w, range(6, 12), served=["a", "b"], answered=["a", "b"])
    assert w.holder() == "b"
    _run(w, range(12, 18), served=["a", "b", "c"], answered=["a", "b", "c"])
    assert w.holder() == "c" and [w.name(h) for h in w.state.history] == ["b", "a"]
    # Every seat is paid on its verification alone: no challenge is needed.
    d = w.epoch(18, served=["a", "b", "c"], answered=[])
    assert _weights(w, d) == pytest.approx({"c": 0.85, "b": 0.10, "a": 0.05})
    # The second former holder is not verified: its share goes to the holder.
    d = w.epoch(19, served=["b", "c"], answered=[])
    assert _weights(w, d) == pytest.approx({"c": 0.90, "b": 0.10})
    # The holder is not verified: its share burns, and so does the share of
    # the seat that is not paid; the verified former holder is paid.
    d = w.epoch(20, served=["b"], answered=[])
    assert w.holder() == "c" and _weights(w, d) == pytest.approx({"burn": 0.90, "b": 0.10})


# --- succession ------------------------------------------------------------------------------


def test_a_holder_that_adds_a_row_by_succession_keeps_its_seat_precedence_and_evidence():
    w = World()
    w.missing_after[ENTRIES[1]] = 1000                   # the second entry is new
    w.add("h", skill=0.3, rows=(ENTRIES[0],), hotkey="hk-h", owner="ck-h", block=5)
    _run(w, range(1, 7), served=["h"], answered=["h"])
    assert w.holder() == "h"
    old = w.state.head(hash_of("h"))
    w.add("h2", skill=0.3, rows=ENTRIES, hotkey="hk-h", owner="ck-h", block=65)
    d = w.epoch(7, served=["h2"], answered=["h2"])
    assert w.holder() == "h2" and d.event == "hold"
    hs = w.state.head(hash_of("h2"))
    assert hs.origin == hash_of("h") and hs.origin_block == 5
    assert hs.rows_since == {ENTRIES[0]: old.since} and hs.since == 7
    assert hs.odds_since == old.since and hs.predecessors == (hash_of("h"),)
    assert w.state.head(hash_of("h")) is None
    # Its precedence is its predecessor's; its picks run carries on.
    assert seat_v4.precedence(hs, floor_block=None, salt="")[:2] == \
        seat_v4.precedence(old, floor_block=None, salt="")[:2]
    from fugal_subnet.picks_v4 import answered_run
    recs = seat_v4.lineage_records(w.window.picks, w.state)
    assert answered_run(recs, hash_of("h2"), 7) == tuple(range(1, 8))
    assert _weights(w, d) == {"h2": 1.0}


def test_a_changed_row_another_coldkey_or_another_hotkey_is_a_new_head():
    for case in ("row", "coldkey", "hotkey"):
        w = World()
        w.missing_after[ENTRIES[1]] = 1000
        w.add("h", skill=0.3, rows=(ENTRIES[0],), hotkey="hk-h", owner="ck-h", block=5)
        _run(w, range(1, 7), served=["h"], answered=["h"])
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
        assert hs.origin == hash_of("h2") and hs.rows_since == {} and not hs.predecessors, case
        assert w.holder() == "h", case


def test_a_successor_of_a_contender_keeps_its_place_against_a_later_near_copy():
    w = World()
    w.missing_after[ENTRIES[1]] = 1000
    w.add("h", skill=0.2)
    w.add("orig", skill=0.6, block=7, rows=(ENTRIES[0],), hotkey="hk-o", owner="ck-o")
    w.add("copy", skill=0.6, block=8, noise=5, owner="ck-other", rows=(ENTRIES[0],))
    _run(w, range(1, 6), served=["h"], answered=["h"])
    for e in range(6, 10):
        w.epoch(e, served=["h", "orig", "copy"],
                answered=["h", "copy"] + (["orig"] if _odd(e) else []))
    # The original adds a row by succession, after its near-copy's commit.
    w.add("orig2", skill=0.6, block=95, rows=ENTRIES, hotkey="hk-o", owner="ck-o")
    for e in range(10, 16):
        d = w.epoch(e, served=["h", "orig2", "copy"],
                    answered=["h", "copy"] + (["orig2"] if _odd(e) else []))
        assert w.holder() == "h", e
    assert d.evidence["order"] == [hash_of("orig2"), hash_of("copy")]


# --- a holder lacking a row when an entry's allowance ends (4.3) -------------------------


def _allowance_world():
    w = World()
    w.add("h", skill=0.3, rows=(ENTRIES[0],), hotkey="hk-h", owner="ck-h")
    w.missing_after[ENTRIES[1]] = 10                     # the second entry's allowance ends
    _run(w, range(1, 6), served=["h"], answered=["h"])
    assert w.holder() == "h"
    return w


def test_a_holder_lacking_a_row_loses_the_seat_to_a_qualifier_with_it_with_no_burn():
    w = _allowance_world()
    # The challenger is the holder's equal, with a row that says the entry's
    # measured rate: equal while the allowance runs.
    w.add("c", skill=0.3, noise=4, block=55, row_skill={ENTRIES[1]: 0.0})
    for e in range(6, 10):
        d = w.epoch(e, served=["h", "c"], answered=["h", "c"])
        assert w.holder() == "h" and _weights(w, d) == {"h": 1.0}
    for e in range(10, 30):
        d = w.epoch(e, served=["h", "c"], answered=["h", "c"])
        # No epoch with the seat open and nobody seated: it goes at once.
        assert not d.evidence["open"] or d.event == "take", e
        if d.event == "take":
            break
    assert w.holder() == "c" and d.state.history == (hash_of("h"),)
    # No burn: the new holder is paid at once; the former holder is paid its
    # history seat while R does not beat it, and its share goes to the
    # holder once R does.
    paid = _weights(w, d)
    assert "burn" not in paid and paid["c"] >= 0.85
    assert paid.get("h", 0.0) == (0.0 if d.evidence["paid"][hash_of("h")] is False else 0.10)


def test_a_holder_lacking_a_row_with_no_qualifier_keeps_the_seat_unpaid():
    w = _allowance_world()
    # A missing row is scored at odds 0 (100 y^2 a label, FAST_CROWN_V4.md 18):
    # on this world's coin-flip labels R overtakes the holder within ten epochs.
    d = _run(w, range(6, 22), served=["h"], answered=["h"])
    assert w.holder() == "h" and d.event == "open" and _weights(w, d) == {"burn": 1.0}


def test_a_holder_that_adds_the_row_by_succession_before_the_allowance_ends_keeps_everything():
    w = _allowance_world()
    w.add("h2", skill=0.3, rows=ENTRIES, hotkey="hk-h", owner="ck-h", block=75)
    d = _run(w, range(8, 14), served=["h2"], answered=["h2"])
    assert w.holder() == "h2" and d.event == "hold" and _weights(w, d) == {"h2": 1.0}


# --- the published state -------------------------------------------------------------------


def test_the_state_round_trips_and_a_malformed_one_is_refused():
    w = World()
    w.missing_after[ENTRIES[1]] = 1000                   # the second entry is new
    w.add("h", skill=0.3, rows=(ENTRIES[0],), hotkey="hk-h", owner="ck-h")
    _run(w, range(1, 7), served=["h"], answered=["h"])
    w.add("h2", skill=0.3, rows=ENTRIES, hotkey="hk-h", owner="ck-h", block=65)
    w.epoch(7, served=["h2"], answered=["h2"])
    raw = seat_v4.to_dict(w.state)
    assert seat_v4.from_dict(raw) == w.state and seat_v4.is_v4_dict(raw)
    import json
    assert seat_v4.from_dict(json.loads(json.dumps(raw))) == w.state
    bad = [
        {**raw, "extra": 1},
        {**raw, "champion": "0" * 64},                      # a seat the state does not keep
        {**raw, "champion_since": None},
        {**raw, "history": ["x"]},
        {**raw, "entries": list(reversed(raw["entries"]))},
        {**raw, "heads": {"nothex": next(iter(raw["heads"].values()))}},
    ]
    one = dict(next(iter(raw["heads"].values())))
    for field, value in (("rows", f"99:{'0' * 64}"), ("seen", -1), ("block", 0),
                         ("rows", "0:zz"), ("rows", f"1:{'0' * 64} 0:{'0' * 64}"),
                         ("rows", f"0:{'0' * 64} 0:{'1' * 64}"), ("rows", [[0, "0" * 64]]),
                         ("rows_since", "0:x"), ("rows_since", "0:01"),
                         ("predecessors", ["x"] * 2), ("predecessors", ["0" * 64] * 9),
                         ("added", "5"), ("added", "1 0"), ("added", "01"), ("added", [0])):
        bad.append({**raw, "heads": {next(iter(raw["heads"])): {**one, field: value}}})
    for b in bad:
        with pytest.raises(ValueError):
            seat_v4.from_dict(b)


def test_row_digests_are_the_rows_values_and_entry():
    import numpy as np
    W = np.arange(8, dtype=np.float32).reshape(2, 4)
    b = np.array([0.5, -1.0], dtype=np.float32)
    d = seat_v4.row_digests(W, b, ["e1", "e2"])
    assert d == seat_v4.row_digests(W.astype(np.float64), b.astype(np.float64), ["e1", "e2"])
    assert d["e1"] != seat_v4.row_digests(W, b, ["e2", "e1"])["e2"]
    W2 = W.copy()
    W2[1, 3] += 1
    d2 = seat_v4.row_digests(W2, b, ["e1", "e2"])
    assert d2["e1"] == d["e1"] and d2["e2"] != d["e2"]


def test_a_head_that_no_longer_stands_and_holds_no_seat_is_dropped():
    w = World()
    w.add("h", skill=0.3)
    w.add("x", skill=0.0, block=7)
    _run(w, range(1, 6), served=["h", "x"], answered=["h"])
    _run(w, range(6, 9), served=["h"], answered=["h"])
    assert w.state.head(hash_of("x")) is None and w.state.head(hash_of("h")) is not None
    d = _run(w, range(9, 14), served=[], answered=[])
    assert w.state.head(hash_of("h")) is not None and d.event == "open"


def test_dataclass_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        seat_v4.SeatStateV4().champion = "x"


# --- a follower's re-anchor (v2_sync.resync) ----------------------------------------------------


def test_a_reanchored_state_is_brought_forward_over_the_own_reveals_after_the_snapshot():
    """resync takes the reference's seat state at k and this validator's own
    reveals after it: the heads those reveals list as served are seen then,
    and a head first seen after k (here a successor of the holder, and a new
    head) is taken as the own reveal's seat state records it."""
    w = World()
    w.missing_after[ENTRIES[1]] = 1000
    w.add("h", skill=0.3, rows=(ENTRIES[0],), hotkey="hk-h", owner="ck-h")
    w.add("x", skill=0.0, block=7)
    _run(w, range(1, 7), served=["h", "x"], answered=["h"])
    snapshot = w.state
    own = []
    w.add("h2", skill=0.3, rows=ENTRIES, hotkey="hk-h", owner="ck-h", block=65)
    w.add("n", skill=0.0, block=66)
    for e, served in ((7, ["h2", "x", "n"]), (8, ["h2", "n"])):
        w.epoch(e, served=served, answered=["h2"])
        own.append((e, {"v2_state": {"serving": dict(w.serving),
                                     "seat": seat_v4.to_dict(w.state)}}))
    got = seat_v4.carry_seen(snapshot, own)
    assert got.champion == hash_of("h2") == w.state.champion
    assert got.head(hash_of("h")) is None
    for name in ("h2", "x", "n"):
        assert got.head(hash_of(name)) == w.state.head(hash_of(name)), name
    assert got.head(hash_of("x")).seen == 7
    # Nothing after the snapshot: the snapshot itself.
    assert seat_v4.carry_seen(snapshot, []) == snapshot


# --- what v4 no longer runs (SEAT_MINIMAL_V4.md 12) ------------------------------------------


REMOVED_CONSTANTS = (
    "CHAMPION_GHOST", "CHAMPION_CONFIRM", "CHAMPION_MIN_ACCURACY", "CHAMPION_HARD",
    "CHAMPION_MIN_PAIRED_QUESTIONS", "CHAMPION_VALUE_PER_CORRECT_USD", "CHAMPION_TIME_GATE",
    "CHAMPION_CONFIDENCE", "V2_BURN_IN_QUESTIONS", "ODDS_CONFIRM_PAIRS", "NV_NONINFERIORITY",
    "WILSON_CONFIDENCE", "EXPLORE_DISAGREE_", "ENTRY_GRACE_EPOCHS", "MAX_WEIGHT_DELTA",
    "BRIER_MARGIN", "BOOTSTRAP",
)


def test_the_v4_seat_reads_none_of_the_removed_rules_or_their_constants():
    """The v4 seat's modules, and the v4 branch of the live epoch, name none
    of the constants of the rules section 12 removes (the bootstrap, the
    confirmation phase, ghosts, burn-in, the high-water marks, the accuracy
    floor, the hard weight, the senior bars, the agreement detector, grace,
    the weight cap), and call none of those rules. What now stops each one's
    cheat is tested above and in tests/test_v4_seat_attacks.py."""
    import inspect

    from fugal_subnet import labels_v4, memo_v4, picks_v4, seat_bound, v2_epoch

    src = {m.__name__: inspect.getsource(m)
           for m in (seat_v4, seat_bound, picks_v4, labels_v4, memo_v4)}
    whole = inspect.getsource(v2_epoch.run_v2_epoch)
    branch = whole[whole.index("    if v4:\n        # The seat's view"):
                   whole.index("    new_window = window.add_epoch(")]
    assert "_memo_verdicts(" in branch
    src["run_v2_epoch (v4)"] = (branch + inspect.getsource(v2_epoch._v4_result)
                                + inspect.getsource(v2_epoch._memo_verdicts))
    rules = ("burn_in_complete", "senior_bars", "seat_holders", "comparison_set",
             "_passes_gates", "_live_pending", "paired_test", "cap_weight_change", "Ghost(")
    for name, text in src.items():
        for word in REMOVED_CONSTANTS + rules:
            assert word not in text, (name, word)
