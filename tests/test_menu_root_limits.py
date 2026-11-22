"""Protocol v4: what a committed menu cannot do, whoever writes it
(docs/decisions/LIVE_MENU_V4.md, 4.3 "what the root can and cannot do").

Each case is a menu a root could commit; every node's reader
(`menu_for_epoch`, through `check_transition`) refuses it and the epoch runs
on the carried menu:

- a new filing for an id whose cooldown is still running, or from a
  coldkey whose cooldown is still running, or identical to the entry on the
  menu;
- an entry that leaves without the `retired` record a retirement writes;
- a proposal that leaves measurement or the queue without what the step
  would write for it (the queue, the menu, or its record);
- a price, answer limit, token mean or exploration-pair count that moves
  outside a refresh epoch, for any entry but the one that joins or is
  updated in that step;
- a refreshed price that moves by more than a factor of
  `MENU_PRICE_STEP` (4) up or down.
"""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from fugal_subnet import menu_state as ms
from tests import menu_fixtures as fx

CADENCE = dataclasses.replace(fx.PARAMS, refresh_epochs=ms.MENU_REFRESH_EPOCHS)
K, NEW = 100, 102
HIGH, TWO = "lab-a/model-one@high", "lab-b/model-two"


def verdict(prev, nxt, params=CADENCE):
    raw = ms.canonical_json(nxt)
    try:
        got = ms.menu_for_epoch(prev, hashlib.sha256(raw).hexdigest(), raw, allowlist=fx.ALLOW,
                                params=params)
    except ms.MenuRefused as e:
        assert e.carried == ms.carry(prev)
        return f"refused: {e}"
    assert got == nxt
    return "accepted"


def nxt(prev, **over):
    return dataclasses.replace(ms.carry(prev), serial=prev.serial + 1, **over)


# --- filings ---------------------------------------------------------------------------
# --- what leaves leaves its record ---------------------------------------------------------


def test_an_entry_that_leaves_without_its_record_is_refused():
    prev = fx.state()
    gone = prev.entry(HIGH)
    bare = nxt(prev, entries=tuple(e for e in prev.entries if e.id != HIGH))
    assert "record" in verdict(prev, bare)
    for wrong in (ms.Retired(HIGH, NEW, ms.REASON_GONE, NEW + 1),            # short cooldown
                  ms.Retired(HIGH, NEW - 5, ms.REASON_GONE, NEW + 48),        # another epoch
                  ms.Retired(HIGH, NEW, ms.REASON_GATE_USEFUL, NEW + 48),     # not an entry's
                  ms.Retired(HIGH, NEW, ms.REASON_LAB_REMOVED, NEW + 48)):    # no newer list
        assert verdict(prev, dataclasses.replace(bare, retired=(wrong,))).startswith("refused")
    ok = dataclasses.replace(bare, retired=(ms.Retired(gone.id, NEW, ms.REASON_GONE, NEW + 48),))
    assert verdict(prev, ok) == "accepted"
# --- the refresh cadence ---------------------------------------------------------------------


def test_a_structural_change_no_longer_licenses_a_refresh_of_every_other_entry():
    prev = fx.state()                                                   # 102: no refresh epoch
    gone = prev.entry(HIGH)
    rest = tuple(dataclasses.replace(e, price_in=e.price_in * 2) for e in prev.entries
                 if e.id != HIGH)
    moved = nxt(prev, entries=rest, retired=(ms.Retired(gone.id, NEW, ms.REASON_GONE, NEW + 48),))
    assert "refresh" in verdict(prev, moved)
    with_list = nxt(prev, labs=ms.AllowlistRef(3, fx.allowlist(3).sha256),
                    entries=tuple(dataclasses.replace(e, explore_pairs=5) for e in prev.entries))
    assert "refresh" in verdict(prev, with_list)


def test_the_step_refreshes_only_in_a_refresh_epoch():
    # A structural change (a provisional entry decided) is no refresh.
    s = ms.carry(fx.provisional(fx.state(epoch=100)))
    reads = (fx.read("lab-a/model-one", "host-a/fp8", price_in=150_000),
             fx.read("lab-a/model-one", "host-b/fp8", price_in=150_000))
    out = ms.step(s, fx.inputs(host_reads=reads, gates=(fx.gate("lab-c/model-new@high"),)),
                  CADENCE)
    assert not out.entry("lab-c/model-new@high").provisional
    assert out.entry(HIGH).price_in == s.entry(HIGH).price_in
    assert verdict(s, out) == "accepted"
    at = fx.state(epoch=119)
    out = ms.step(at, fx.inputs(epoch=118, host_reads=reads), CADENCE)
    assert out.entry(HIGH).price_in == 150_000


# --- a refreshed price moves at most a factor of four -------------------------------------


def test_a_refreshed_price_moves_at_most_a_factor_of_four():
    assert ms.MENU_PRICE_STEP == 4
    prev = fx.state(epoch=119)                                          # 120 refreshes
    e = prev.entry(TWO)
    for price_in, price_out, ok in ((e.price_in * 4, e.price_out, True),
                                    (e.price_in * 4 + 1, e.price_out, False),
                                    (e.price_in, -(-e.price_out // 4), True),
                                    (e.price_in, e.price_out // 4 - 1, False)):
        moved = nxt(prev, entries=tuple(
            dataclasses.replace(x, price_in=price_in, price_out=price_out) if x.id == TWO else x
            for x in prev.entries))
        assert (verdict(prev, moved) == "accepted") is ok, (price_in, price_out)


@pytest.mark.parametrize("factor", [100, 1])
def test_the_step_moves_a_price_a_factor_of_four_a_refresh(factor):
    """A real larger move arrives over consecutive refreshes (below the
    price ceiling, menu_rules.check_price, which test_menu_price_ceiling.py
    covers)."""
    s = fx.state(epoch=119)
    s = dataclasses.replace(s, entries=tuple(
        dataclasses.replace(x, price_in=200_000) if x.id == TWO else x for x in s.entries))
    e = s.entry(TWO)
    price = e.price_in * factor if factor > 1 else 1
    read = (fx.read(e.model, "host-c", price_in=price, price_out=e.price_out),)
    seen = []
    for k in (118, 138, 158, 178):
        s = ms.step(dataclasses.replace(s, epoch=k + 1), fx.inputs(epoch=k, host_reads=read),
                    CADENCE)
        seen.append(s.entry(TWO).price_in)
    if factor > 1:
        assert seen[:3] == [e.price_in * 4, e.price_in * 16, e.price_in * 64]
        assert seen[3] == price
    else:
        assert seen[0] == -(-e.price_in // 4)


# --- the records and cooldowns are the step's -----------------------------------------------


def test_a_retired_record_is_written_only_for_what_ends_in_the_step():
    """A record decides who may file: a committed menu cannot invent one for
    an id that did not leave, give one another epoch or reason, or lengthen
    one that is there."""
    prev = fx.state()
    for invented in (ms.Retired("lab-z/never@high", NEW, ms.REASON_GATE_USEFUL, NEW + 48),
                     ms.Retired(HIGH, NEW, ms.REASON_GATE_USEFUL, NEW + 48)):  # still on the menu
        assert "record" in verdict(prev, nxt(prev, retired=(invented,)))
    rest = tuple(e for e in prev.entries if e.id != HIGH)
    for wrong in (ms.Retired(HIGH, NEW, ms.REASON_GONE, ms.MAX_INT),
                  ms.Retired(HIGH, NEW, ms.REASON_GONE, NEW + 49)):
        assert verdict(prev, nxt(prev, entries=rest, retired=(wrong,))).startswith("refused")
    # an existing record, lengthened or re-dated in a step where its id did nothing
    old = ms.Retired("lab-z/left@high", 90, ms.REASON_FAILING, 138)
    prev = fx.state(retired=(old,))
    for longer in (dataclasses.replace(old, cooldown_until=ms.MAX_INT),
                   ms.Retired(old.id, NEW, ms.REASON_FAILING, NEW + 48)):
        assert "record" in verdict(prev, nxt(prev, retired=(longer,)))
    assert verdict(prev, nxt(prev, retired=(old,))) == "accepted"


def test_a_coldkey_cooldown_is_written_only_by_a_failed_gate_in_the_step():
    prev = fx.state()
    for banned in ((ms.ColdkeyCooldown("5Fany-rival", ms.MAX_INT),),
                   (ms.ColdkeyCooldown(fx.COLDKEY, NEW + 48),)):
        assert "cooldown" in verdict(prev, nxt(prev, coldkey_cooldowns=banned))
    held = ms.ColdkeyCooldown("5Fck-held", 130)
    prev = fx.state(coldkey_cooldowns=(held,))
    assert "cooldown" in verdict(prev, nxt(prev, coldkey_cooldowns=(
        dataclasses.replace(held, until=ms.MAX_INT),)))
    # a failed gate writes its proposer's coldkey's cooldown at the compiled
    # length, no longer, and a list retirement of a provisional entry writes none
    base = fx.provisional(fx.state(epoch=100))
    prev = ms.carry(base)
    eid = "lab-c/model-new@high"
    rest = tuple(e for e in prev.entries if e.id != eid)
    gate = ms.Retired(eid, NEW, ms.REASON_GATE_USEFUL, NEW + 48)
    assert "cooldown" in verdict(prev, nxt(prev, entries=rest, provisional=(), retired=(gate,),
                                           coldkey_cooldowns=(
                                               ms.ColdkeyCooldown(fx.COLDKEY, ms.MAX_INT),)))
    assert "cooldown" in verdict(prev, nxt(prev, entries=rest, provisional=(), retired=(gate,)))
    assert verdict(prev, nxt(prev, entries=rest, provisional=(), retired=(gate,),
                             coldkey_cooldowns=(ms.ColdkeyCooldown(fx.COLDKEY, NEW + 48),))
                   ) == "accepted"


def test_records_leave_only_by_the_steps_own_pruning():
    """A record whose cooldown ran out stays until `retired` is over its size,
    and then the oldest such records leave first; a running one never does."""
    small = dataclasses.replace(CADENCE, retired_keep=2)
    spent_old = ms.Retired("lab-y/old@low", 40, ms.REASON_FAILING, 88)
    spent_new = ms.Retired("lab-y/new@low", 50, ms.REASON_FAILING, 98)
    prev = fx.state(retired=(spent_new, spent_old))
    assert "retired record" in verdict(prev, nxt(prev, retired=(spent_new,)), small)
    rest = tuple(e for e in prev.entries if e.id != HIGH)
    gone = ms.Retired(HIGH, NEW, ms.REASON_GONE, NEW + 48)
    # one more record: the oldest that ran out makes room, and only it
    ok = nxt(prev, entries=rest, retired=tuple(sorted((gone, spent_new), key=lambda r: r.id)))
    assert verdict(prev, ok, small) == "accepted"
    assert ms.step(prev, fx.inputs(health=(ms.HealthStat(HIGH, 1, unlisted_epochs=6),)),
                   dataclasses.replace(small, min_entries=1)).retired == ok.retired
    wrong = nxt(prev, entries=rest, retired=tuple(sorted((gone, spent_old), key=lambda r: r.id)))
    assert "retired record" in verdict(prev, wrong, small)


def test_a_menu_holds_no_more_records_or_coldkeys_than_the_caps():
    assert ms.coldkey_cooldown_cap() == ms.MENU_MAX_PROVISIONAL * (
        ms.PROPOSAL_COOLDOWN_EPOCHS + ms.MENU_LAG)
    s = fx.state(retired=tuple(ms.Retired(f"lab-z/r{i:03d}@low", 90, ms.REASON_FAILING, 140)
                               for i in range(ms.RETIRED_KEEP + 1)))
    assert "retired record" in ms.check_caps(s, CADENCE)
    s = fx.state(coldkey_cooldowns=tuple(ms.ColdkeyCooldown(f"5Fck{i:04d}", 140)
                                         for i in range(ms.coldkey_cooldown_cap() + 1)))
    assert "coldkeys" in ms.check_caps(s, CADENCE)


# --- a list reason is the list's verdict ------------------------------------------------------


def _list_step(prev, newer, leaving, reason):
    """`prev` one step on, adopting `newer` and retiring `leaving` for `reason`."""
    return nxt(prev, labs=ms.AllowlistRef(newer.serial, newer.sha256),
               entries=tuple(e for e in prev.entries if e.id not in leaving),
               retired=tuple(ms.Retired(i, NEW, reason, NEW + 48) for i in sorted(leaving)))


def offered(prev, state, allowlist):
    raw = ms.canonical_json(state)
    try:
        ms.menu_for_epoch(prev, hashlib.sha256(raw).hexdigest(), raw, allowlist=allowlist,
                          params=CADENCE)
    except ms.MenuRefused as e:
        return f"refused: {e}"
    return "accepted"


def test_an_entry_retires_for_a_list_reason_only_when_the_newer_list_refuses_it():
    prev = fx.wide_state()
    adds_a_host = fx.allowlist(3, hosts=fx.HOSTS + ("host-x",))         # refuses nothing
    healthy = {e.id for e in prev.entries if e.model.startswith("lab-d/")}
    assert "list" in offered(prev, _list_step(prev, adds_a_host, healthy, ms.REASON_LAB_REMOVED),
                             adds_a_host)
    drops_lab_b = fx.allowlist(3, labs=tuple(x for x in fx.LABS if x != "lab-b"))
    assert offered(prev, _list_step(prev, drops_lab_b, {TWO}, ms.REASON_LAB_REMOVED),
                   drops_lab_b) == "accepted"
    # the reason is the rule it fails: its lab, not one of its hosts
    assert "list" in offered(prev, _list_step(prev, drops_lab_b, {TWO}, ms.REASON_HOST_REMOVED),
                             drops_lab_b)
    # and it cannot take a healthy entry along with one the list refuses
    assert "list" in offered(prev, _list_step(prev, drops_lab_b, {TWO, HIGH},
                                              ms.REASON_LAB_REMOVED), drops_lab_b)
# --- what joins is its proposal --------------------------------------------------------------


def _with(prev, *entries, drop=(), **over):
    keep = tuple(e for e in prev.entries if e.id not in {x.id for x in entries} | set(drop))
    return nxt(prev, entries=tuple(sorted(keep + entries, key=lambda e: e.id)), **over)
# --- the queue's wait ---------------------------------------------------------------------------
# --- make-up measurement is bounded ---------------------------------------------------------

