"""Protocol v4: a menu every node accepts is a menu every node can run an
epoch on (docs/decisions/LIVE_MENU_V4.md, 4.3, 4.4 and 9.2).

The epoch's exploration draw (`epoch_plan.explore_shares_v4`) and the head's
picks (`head_odds.picks_and_odds_v4`) run in every TD and every validator
before any proof exists. They need a menu that is not empty and an expected
cost above zero for every entry. So:

- the reader refuses an entry whose expected cost is not positive, and a
  menu with fewer than `MENU_MIN_ENTRIES` entries;
- the step never prices an entry at 0 (a voted price of 0 is no change) and
  never takes the menu below the minimum;
- health retirements ("failing" and "gone") are at most
  max(2, a tenth of the menu) a step, the rest waiting for later steps, and
  "gone" has an outage guard like "failing": an epoch in which the listing
  shows too many pinned hosts unlisted at once is not counted;
- for any menu the reader's per-menu checks pass, the draw and the picks
  compute.
"""
from __future__ import annotations

import dataclasses
import hashlib
import random

import numpy as np
import pytest

from fugal_subnet import epoch_plan
from fugal_subnet import menu_inputs as mi
from fugal_subnet import menu_state as ms
from tests import menu_fixtures as mf

TWO = "lab-b/model-two"
# The fixtures' parameters with the compiled minimum menu size.
PARAMS = dataclasses.replace(mf.PARAMS, min_entries=ms.MENU_MIN_ENTRIES)


def offer(prev, nxt, params=None):
    raw = ms.canonical_json(nxt)
    return ms.menu_for_epoch(prev, hashlib.sha256(raw).hexdigest(), raw, allowlist=mf.ALLOW,
                             params=params)


def next_menu(prev, **over):
    return dataclasses.replace(ms.carry(prev), serial=prev.serial + 1, **over)


# --- the reader ---------------------------------------------------------------------


def test_the_compiled_minimum_is_small_and_far_below_the_seed():
    assert ms.MENU_MIN_ENTRIES == 8
    assert ms.READER_PARAMS.min_entries == ms.MENU_MIN_ENTRIES
    assert len(ms.load_seed().entries) >= 6 * ms.MENU_MIN_ENTRIES
    assert ms.check_caps(ms.seed_state(100)) is None          # the seed passes the reader's caps


def test_an_entry_with_no_expected_cost_is_refused():
    prev = mf.wide_state(epoch=119)              # the menu of 120 is a refresh epoch
    for zero in (dict(price_in=0, price_out=0), dict(mean_in=0, mean_out=0)):
        entries = tuple(dataclasses.replace(e, **zero) if e.id == TWO else e
                        for e in prev.entries)
        with pytest.raises(ms.MenuRefused, match="expected cost"):
            offer(prev, next_menu(prev, entries=entries))
        assert "expected cost" in ms.check_caps(next_menu(prev, entries=entries))
    # A price of 0 on one side is fine while the call still costs something.
    entries = tuple(dataclasses.replace(e, price_in=0) if e.id == TWO else e for e in prev.entries)
    assert ms.check_caps(next_menu(prev, entries=entries)) is None


def test_a_menu_below_the_compiled_minimum_is_refused():
    prev = mf.wide_state(fillers=5)              # eight entries: the minimum
    assert len(prev.entries) == ms.MENU_MIN_ENTRIES
    gone = prev.entries[-1]
    nxt = next_menu(prev, entries=prev.entries[:-1],
                    retired=(ms.Retired(gone.id, 102, ms.REASON_FAILING, 102 + 48),))
    with pytest.raises(ms.MenuRefused, match="fewer than"):
        offer(prev, nxt)
    with pytest.raises(ms.MenuRefused):
        offer(prev, next_menu(prev, entries=()))
    # One more entry, and the same retirement leaves the minimum: accepted.
    prev9 = mf.wide_state(fillers=6)
    gone = prev9.entries[-1]
    nxt = next_menu(prev9, entries=prev9.entries[:-1],
                    retired=(ms.Retired(gone.id, 102, ms.REASON_FAILING, 102 + 48),))
    assert offer(prev9, nxt) == nxt


def test_more_health_retirements_than_a_step_makes_are_refused():
    prev = mf.wide_state(fillers=9)               # twelve entries: two a step
    leave = prev.entries[-3:]
    nxt = next_menu(prev, entries=prev.entries[:-3],
                    retired=tuple(ms.Retired(e.id, 102, ms.REASON_GONE, 102 + 48)
                                  for e in sorted(leave, key=lambda e: e.id)))
    with pytest.raises(ms.MenuRefused, match="health"):
        offer(prev, nxt)
    two = next_menu(prev, entries=prev.entries[:-2],
                    retired=tuple(ms.Retired(e.id, 102, ms.REASON_GONE, 102 + 48)
                                  for e in sorted(prev.entries[-2:], key=lambda e: e.id)))
    assert offer(prev, two) == two


# --- the step -------------------------------------------------------------------------


def test_a_voted_price_of_zero_is_no_change():
    prev = mf.wide_state(epoch=119)
    e = prev.entry(TWO)
    reads = (mf.read(e.model, "host-c", price_in=0, price_out=0, voters=5),)
    res = ms.step_detail(prev, mf.inputs(epoch=118, host_reads=reads), PARAMS)
    after = res.state.entry(TWO)
    assert (after.price_in, after.price_out) == (e.price_in, e.price_out)
    assert ms.entry_cost(after) > 0
    # One side at zero keeps that side; the other moves.
    reads = (mf.read(e.model, "host-c", price_in=0, price_out=e.price_out * 2, voters=5),)
    after = ms.step(prev, mf.inputs(epoch=118, host_reads=reads), PARAMS).entry(TWO)
    assert (after.price_in, after.price_out) == (e.price_in, e.price_out * 2)
    # and the reader accepts what the step derives
    nxt = ms.step(prev, mf.inputs(epoch=118, host_reads=reads), PARAMS)
    assert offer(prev, nxt) == nxt


def _health(state, ids, **kw):
    return tuple(ms.HealthStat(eid, state.entry(eid).rev, **kw) for eid in sorted(ids))


def test_health_retirements_are_bounded_per_step_and_the_rest_wait():
    prev = mf.wide_state(fillers=9)               # twelve: max(2, 12 // 10) = 2 a step
    ids = [e.id for e in prev.entries[:5]]
    res = ms.step_detail(prev, mf.inputs(health=_health(prev, ids, eligible=6, failing=6)),
                         PARAMS)
    retired = [x.subject for x in res.events if x.kind == "retired"]
    assert retired == sorted(ids)[:2]
    assert offer(prev, res.state) == res.state
    # the next step retires the next two
    nxt = res.state
    res2 = ms.step_detail(nxt, dataclasses.replace(
        mf.inputs(epoch=101, health=_health(nxt, sorted(ids)[2:], eligible=6, failing=6))),
        PARAMS)
    assert [x.subject for x in res2.events if x.kind == "retired"] == sorted(ids)[2:4]


def test_health_never_takes_the_menu_below_the_minimum():
    prev = mf.wide_state(fillers=6)               # nine entries
    ids = [e.id for e in prev.entries]
    res = ms.step_detail(prev, mf.inputs(health=_health(prev, ids, unlisted_epochs=99)), PARAMS)
    assert len(res.state.entries) == ms.MENU_MIN_ENTRIES
    assert offer(prev, res.state) == res.state
    again = ms.step(res.state, mf.inputs(epoch=101, health=_health(
        res.state, res.state.ids(), unlisted_epochs=99)), PARAMS)
    assert len(again.entries) == ms.MENU_MIN_ENTRIES


def test_a_list_that_would_leave_fewer_than_the_minimum_is_not_adopted():
    prev = mf.wide_state(fillers=6)               # six of the nine are lab-d's
    newer = mf.allowlist(3, labs=tuple(x for x in mf.LABS if x != "lab-d"))
    res = ms.step_detail(prev, mf.inputs(lab_list=newer), PARAMS)
    assert res.state.labs == prev.labs and len(res.state.entries) == 9
    assert any(x.kind == "lab_list_held" for x in res.events)
    # A newer list that leaves enough is adopted as before.
    fine = mf.allowlist(3, labs=tuple(x for x in mf.LABS if x != "lab-b"))
    res = ms.step_detail(prev, mf.inputs(lab_list=fine), PARAMS)
    assert res.state.labs.serial == 3 and TWO not in res.state.ids()


def test_while_a_newer_list_is_held_back_joins_go_on_under_the_list_in_force():
    """An honest root that holds a newer list back (adopting it would take the
    menu below the minimum) still admits joins, under the list in force (the
    base menu's), so the menu can grow until the newer list leaves enough;
    then the list is adopted and what it refuses retires. Until then the
    removed lab's entries stay."""
    from fugal_subnet import chain_proposals as cp

    gone = mf.entry("lab-d/filler-5", None, provider_order=("host-d",))
    prev = mf.wide_state(fillers=5)                                    # 8: lab-b's one among them
    assert gone.id not in prev.ids() and len(prev.entries) == ms.MENU_MIN_ENTRIES
    newer = mf.allowlist(3, labs=tuple(x for x in mf.LABS if x != "lab-b"))
    res = ms.step_detail(prev, mf.inputs(lab_list=newer, lab_list_in_force=mf.ALLOW), PARAMS)
    assert any(x.kind == "lab_list_held" for x in res.events)
    assert res.state.labs == prev.labs and TWO in res.state.ids()
    assert offer(prev, res.state, PARAMS) == res.state
    # A proposal of a lab the held list would refuse is still judged under the
    # list in force, and joins.
    doc, raw = mf.document("lab-b/model-new", "high", proposer="hk")
    read = cp.ChainRead(epoch=res.state.epoch, block=0, block_hash="ab" * 32,
                        proposals=(("hk", cp.proposal_hash(raw)),), coldkey_of={"hk": "ck"},
                        serving=frozenset({"hk"}), reveal=None)
    delta = cp.admit(res.state, read, {cp.proposal_hash(raw): raw},
                     {cp.proposal_hash(raw): mf.listing()}, mf.ALLOW, PARAMS)
    assert [c.verdict for c in delta.considered] == [cp.JOINED]
    # The list in force handed in must be the one the menu names.
    with pytest.raises(ValueError, match="in force"):
        ms.step_detail(prev, mf.inputs(lab_list=newer, lab_list_in_force=newer), PARAMS)
    # Once the menu has room to lose lab-b's entry, the list is adopted.
    grown = mf.wide_state(gone, fillers=5)
    res = ms.step_detail(grown, mf.inputs(lab_list=newer, lab_list_in_force=mf.ALLOW), PARAMS)
    assert res.state.labs.serial == 3 and TWO not in res.state.ids()


# --- the minimum counts full entries ------------------------------------------------------
#
# The gates may retire every provisional entry in one step, whatever else the
# step does, so a provisional entry holds no place under the floor: the step
# never leaves fewer than `min_entries` full entries, and the reader refuses a
# menu with fewer.


def _with_provisional(base: ms.MenuState, n: int) -> ms.MenuState:
    """`base` with `n` provisional entries of lab-c joined at its epoch."""
    for i in range(n):
        base = mf.provisional(base, f"lab-c/model-{i}", proposer=f"hk-{i}", coldkey=f"ck-{i}")
    return base


def test_provisional_entries_do_not_make_up_the_minimum_for_a_newer_list():
    """Nine full entries, six of them lab-d's, and five provisional ones that
    the gates decide in this step. A list without lab-d leaves three full
    entries and five provisional: eight entries, and before the minimum
    counted full entries it was adopted; the gates then retired all five and
    the root derived a menu of three, which every node refused."""
    prev = ms.carry(_with_provisional(mf.wide_state(fillers=6, epoch=100), 5))   # menu of 101
    newer = mf.allowlist(3, labs=tuple(x for x in mf.LABS if x != "lab-d"))
    res = ms.step_detail(prev, mf.inputs(lab_list=newer, lab_list_in_force=mf.ALLOW), PARAMS)
    assert ("lab_list_held", "3") in [(x.kind, x.subject) for x in res.events]
    assert res.state.labs == prev.labs and not res.state.provisional
    assert ms.full_count(res.state.entries) == 9
    assert offer(prev, res.state, PARAMS) == res.state
    # The menu that adopts the list and keeps the provisional entries holds
    # eight entries but three full ones: the reader refuses it.
    lab_d = {e.id for e in prev.entries if e.model.startswith("lab-d/")}
    adopted = dataclasses.replace(prev, entries=tuple(e for e in prev.entries if e.id not in lab_d))
    assert len(adopted.entries) == ms.MENU_MIN_ENTRIES
    assert "3 full entries" in ms.check_caps(adopted, PARAMS)


def test_at_the_minimum_a_full_menu_displaces_nothing_and_health_waits():
    """Eight full entries and two provisional ones on a menu at its cap of
    ten: nothing is displaced and nothing retires for health, whatever the
    use and the health say. With a ninth full entry, one is displaced."""
    params = dataclasses.replace(PARAMS, menu_cap=10)
    use = dict(use=(), use_total=100)                         # every entry at zero use
    at_min = _with_provisional(mf.wide_state(fillers=5), 2)     # joined at 101: not decided
    health = _health(at_min, [e.id for e in at_min.entries if not e.provisional],
                     eligible=6, failing=6)
    res = ms.step_detail(at_min, mf.inputs(health=health, **use), params)
    assert res.state.ids() == at_min.ids() and ms.full_count(res.state.entries) == 8
    assert ("menu_full", "10") in [(x.kind, x.subject) for x in res.events]
    assert sum(x.kind == "retire_waits" for x in res.events) == 8
    one_over = _with_provisional(mf.wide_state(fillers=6), 1)
    res = ms.step_detail(one_over, mf.inputs(**use), params)
    assert [x.detail for x in res.events if x.kind == "retired"] == [ms.REASON_DISPLACED]
    assert ms.full_count(res.state.entries) == 8
    assert offer(one_over, res.state, params) == res.state


def test_every_entry_failing_at_once_leaves_the_minimum_of_full_entries():
    """An outage: every full entry fails, every provisional entry fails its
    gate, and a newer list would remove most of the menu, all in one step."""
    prev = ms.carry(_with_provisional(mf.wide_state(fillers=6, epoch=100), 4))
    full = [e.id for e in prev.entries if not e.provisional]
    newer = mf.allowlist(3, labs=tuple(x for x in mf.LABS if x != "lab-d"))
    res = ms.step_detail(prev, mf.inputs(
        lab_list=newer, lab_list_in_force=mf.ALLOW, use_total=100,
        health=_health(prev, full, eligible=6, failing=6, unlisted_epochs=99)), PARAMS)
    assert not res.state.provisional and res.state.labs == prev.labs
    assert len(res.state.entries) == ms.full_count(res.state.entries) == ms.MENU_MIN_ENTRIES
    assert offer(prev, res.state, PARAMS) == res.state


# --- "gone" has an outage guard ----------------------------------------------------------


def _votes_unlisted(state, models_hosts):
    return tuple(sorted({ms.HostRead(model=m, host=h, voters=6, listed=False)
                         for m, h in models_hosts}, key=lambda r: (r.model, r.host)))


def _all_pins(state):
    return [(e.model, h) for e in state.entries for h in e.provider_order]


def test_an_epoch_with_most_of_the_listing_unlisted_is_not_counted_for_gone():
    state = mf.wide_state(epoch=119)
    window = [mi.EpochSummary(epoch=k, votes=_votes_unlisted(state, _all_pins(state)),
                              calls=(), pairs=(), use=()) for k in range(110, 119)]
    health = mi.health_stats(state, window, through=118)
    assert all(h.unlisted_epochs == 0 for h in health)
    res = ms.step_detail(state, mf.inputs(epoch=118, health=health), PARAMS)
    assert not [x for x in res.events if x.kind == "retired"]


def test_one_entry_gone_while_the_listing_is_healthy_still_retires():
    state = mf.wide_state(epoch=119)
    e = state.entry(TWO)
    listed = tuple(sorted({ms.HostRead(model=m, host=h, voters=6, listed=True,
                                       price_in=1, price_out=1, limit=1000)
                           for m, h in _all_pins(state) if m != e.model},
                          key=lambda r: (r.model, r.host)))
    votes = tuple(sorted(listed + _votes_unlisted(state, [(e.model, "host-c")]),
                         key=lambda r: (r.model, r.host)))
    window = [mi.EpochSummary(epoch=k, votes=votes, calls=(), pairs=(), use=())
              for k in range(110, 119)]
    health = {h.entry_id: h for h in mi.health_stats(state, window, through=118)}
    assert health[TWO].unlisted_epochs == 9
    assert ms.health_retires(health[TWO], PARAMS) == ms.REASON_GONE


# --- any menu the reader accepts can run an epoch ----------------------------------------


def _head(ids):
    from fugal_subnet.head_eval import load_head_v4
    from tests.test_tee_e2e import _head as head_bytes
    return load_head_v4(head_bytes(list(ids)))


def _random_menus(n: int, seed: int):
    """Menus with arbitrary prices, token means and exploration-pair counts
    in range, from the pinned seed's entries, of arbitrary sizes; an entry's
    pick-cost table dropped, or kept with its means at the new call means or
    at arbitrary values."""
    seed_state = ms.load_seed()
    rnd = random.Random(seed)
    extremes = [0, 1, 2, 7, 10**3, 10**6, 10**9, 10**12, ms.MAX_INT]

    def value():
        return rnd.choice(extremes) if rnd.random() < 0.5 else rnd.randint(
            0, ms.MAX_INT >> rnd.randint(0, 52))

    def pick(e, keep):
        if e.pick is None or not keep or rnd.random() < 0.4:
            return None
        means = (e.mean_in, e.mean_out) if rnd.random() < 0.8 else (value(), value())
        return dataclasses.replace(e.pick, mean_in=means[0], mean_out=means[1], categories=tuple(
            dataclasses.replace(c, mean_in=means[0], mean_out=means[1])
            for c in e.pick.categories))

    for _ in range(n):
        size = rnd.choice([0, 1, 2, 7, 8, 9, 30, len(seed_state.entries)])
        entries = rnd.sample(list(seed_state.entries), size)
        entries = [dataclasses.replace(e, price_in=value(), price_out=value(), mean_in=value(),
                                       mean_out=value(), explore_pairs=value()) for e in entries]
        # A quarter of the menus keep tables: a table beside call means past
        # the answer cap is one no cells could give, so most of those are refused.
        keep = rnd.random() < 0.25
        entries = [dataclasses.replace(e, pick=pick(e, keep)) for e in entries]
        yield dataclasses.replace(seed_state, epoch=1000,
                                  entries=tuple(sorted(entries, key=lambda e: e.id)))


def test_every_menu_the_readers_per_menu_checks_pass_can_draw_and_pick():
    """The reader's checks of one menu (`validate`, `check_caps`) are what
    every accepted menu passes whatever menu it follows, so a menu that
    passes them is all the draw and the picks may rely on."""
    head = _head([e.id for e in ms.load_seed().entries])
    hidden = np.random.RandomState(0).randn(3, 1024).astype(np.float32)
    passed = refused = 0
    from fugal_subnet.head_odds import picks_and_odds_v4

    for state in _random_menus(160, seed=5):
        try:
            ms.validate(state)
        except ValueError:
            refused += 1
            continue
        if ms.check_caps(state) is not None:
            refused += 1
            continue
        passed += 1
        shares = epoch_plan.explore_shares_v4(state)
        shares.shared_weights()
        shares.private_weights()
        assert len(picks_and_odds_v4(head, state, hidden).picks) == 3
    assert passed > 40 and refused > 20, (passed, refused)


def test_every_menu_the_reader_accepts_after_a_refresh_can_draw_and_pick():
    """The same through `menu_for_epoch` itself: committed menus that move
    every refreshed value as far as one refresh may, from a menu of the
    compiled minimum size."""
    prev = mf.wide_state(epoch=119, fillers=5)
    head = _head(prev.ids())
    hidden = np.random.RandomState(1).randn(2, 1024).astype(np.float32)
    rnd = random.Random(9)
    from fugal_subnet.head_odds import picks_and_odds_v4

    accepted = 0
    for _ in range(120):
        def price(old):          # mostly within one refresh's step, sometimes not
            return old * rnd.choice([1, 4]) // rnd.choice([1, 4]) if rnd.random() < 0.97 \
                else rnd.choice([0, old * 5, ms.MAX_INT])

        entries = tuple(dataclasses.replace(
            e, price_in=price(e.price_in), price_out=price(e.price_out),
            mean_in=rnd.choice([1, e.mean_in, ms.MAX_INT]) if rnd.random() < 0.97 else 0,
            mean_out=rnd.choice([0, 1, e.mean_out, ms.MAX_INT]),
            explore_pairs=rnd.choice([0, 1, ms.MAX_INT])) for e in prev.entries)
        try:
            got = offer(prev, next_menu(prev, entries=entries))
        except ms.MenuRefused:
            continue
        accepted += 1
        epoch_plan.explore_shares_v4(got).private_weights()
        picks_and_odds_v4(head, got, hidden)
    assert accepted > 10, accepted
