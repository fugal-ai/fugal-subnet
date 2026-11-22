"""Protocol v4: the pick-cost table in the published menu (W13;
docs/decisions/SEAT_MINIMAL_V4.md 3.4, the earlier seat design's 8.2).

An entry has two measured costs. The call cost (`menu_state.costs`: token
means over its exploration pairs) is what one call costs on a question drawn
at random; exploration shares, weights and spend caps read it. The pick cost
(`menu_state.pick_costs`) is what a call costs on the questions routers
actually send the entry: the token means of its graded scored-question
cells, shrunk toward the call cost as if PICK_COST_PRIOR_CELLS_ENTRY more
cells had cost exactly that, and per category toward the entry's pick cost
by PICK_COST_PRIOR_CELLS_CATEGORY. Picks, the product and the real-picks
check read it.

The table is published state: exact integers, hashed with the rest of the
menu, refreshed only at the menu's cadence, and reset to the call cost for
an entry that joins or whose request changes. Every node checks that from
the two menus alone. No network, no spend.
"""
from __future__ import annotations

import dataclasses
import json
from fractions import Fraction

import numpy as np
import pytest

from fugal_subnet import epoch_plan, head_odds
from fugal_subnet import menu_state as ms
from tests import menu_fixtures as fx

HIGH, LOW, TWO = "lab-a/model-one@high", "lab-a/model-one@low", "lab-b/model-two"
CADENCE = dataclasses.replace(fx.PARAMS, refresh_epochs=ms.MENU_REFRESH_EPOCHS)


def cell(category, cells, tokens_in, tokens_out):
    return ms.PickCell(category=category, cells=cells, tokens_in=tokens_in,
                       tokens_out=tokens_out)


# The worked example of the shrinkage: HIGH (call means 250 / 900 tokens)
# with 10 graded code cells of 300 / 2,500 tokens and 20 math cells of
# 200 / 5,000.
EXAMPLE = (cell("code", 10, 3_000, 25_000), cell("math", 20, 4_000, 100_000))


def run(state, params=fx.PARAMS, **inputs) -> ms.StepResult:
    return ms.step_detail(state, fx.inputs(epoch=state.epoch - 1, **inputs), params)


def flat(cells, mean_in, mean_out):
    """A table whose cells are all of one category, at the entry's values."""
    return ms.PickTable(cells=cells, mean_in=mean_in, mean_out=mean_out,
                        categories=(ms.PickCategory("code", cells, mean_in, mean_out),))


def with_pick(state, entry_id, pick):
    return dataclasses.replace(state, entries=tuple(
        dataclasses.replace(e, pick=pick) if e.id == entry_id else e for e in state.entries))


# --- the constants ---------------------------------------------------------------------


def test_a_table_holds_at_most_eight_categories_of_at_most_32_characters():
    assert (ms.PICK_MAX_CATEGORIES, ms.PICK_CATEGORY_MAX_CHARS) == (8, 32)
    assert len(epoch_plan.mix_v4().menu_mix()) < ms.PICK_MAX_CATEGORIES
    assert all(len(c) <= ms.PICK_CATEGORY_MAX_CHARS for c, _ in epoch_plan.mix_v4().menu_mix())


def test_the_prior_is_thirty_cells_per_entry_and_twenty_per_category():
    assert (ms.PICK_COST_PRIOR_CELLS_ENTRY, ms.PICK_COST_PRIOR_CELLS_CATEGORY) == (30, 20)
    p = ms.MenuParams(mix=fx.MIX, min_category_accuracy_bp=0)
    assert (p.pick_prior_cells_entry, p.pick_prior_cells_category) == (30, 20)
    assert (ms.READER_PARAMS.pick_prior_cells_entry,
            ms.READER_PARAMS.pick_prior_cells_category) == (30, 20)
    with pytest.raises(ValueError):
        dataclasses.replace(fx.PARAMS, pick_prior_cells_entry=-1)


# --- the shrinkage -------------------------------------------------------------------


def test_the_shrinkage_is_exact_integer_arithmetic_rounded_half_up():
    e = fx.entry()
    table = ms.pick_table(EXAMPLE, e, fx.PARAMS)
    # Entry: (1000 x token sums + 30 x call means) / (30 cells + 30).
    assert (table.cells, table.mean_in, table.mean_out) == (30, 241_667, 2_533_333)
    want_in = Fraction(1000 * 7_000 + 30 * e.mean_in, 60)
    want_out = Fraction(1000 * 125_000 + 30 * e.mean_out, 60)
    assert table.mean_in == int((2 * want_in + 1) // 2)
    assert table.mean_out == int((2 * want_out + 1) // 2)
    # Each category toward the entry's published pick means, with 20 cells.
    code, math = table.categories
    assert (code.category, code.cells, code.mean_in, code.mean_out) == (
        "code", 10, 261_111, 2_522_222)
    # 8,833,340 / 40 = 220,833.5 and 150,666,660 / 40 = 3,766,666.5: half up.
    assert (math.category, math.cells, math.mean_in, math.mean_out) == (
        "math", 20, 220_834, 3_766_667)


def test_an_entry_with_no_graded_cells_is_priced_at_its_call_cost():
    e = fx.entry()
    assert ms.pick_table((), e, fx.PARAMS) is None
    assert ms.pick_table((cell("code", 0, 0, 0),), e, fx.PARAMS) is None
    s = fx.state()
    assert all(x.pick is None for x in s.entries)
    assert ms.pick_costs(s) == ms.costs(s)
    for eid in s.ids():
        assert ms.pick_cost(s, eid, "code") == ms.pick_cost(s, eid) == ms.costs(s)[eid]


def test_many_cells_approach_their_own_mean_and_few_stay_near_the_call_cost():
    e = fx.entry()
    few = ms.pick_table((cell("code", 1, 10_000, 50_000),), e, fx.PARAMS)
    many = ms.pick_table((cell("code", 30_000, 30_000 * 10_000, 30_000 * 50_000),), e, fx.PARAMS)
    assert abs(few.mean_out - e.mean_out) < abs(few.mean_out - 50_000_000)
    assert abs(many.mean_out - 50_000_000) < 50_000_000 // 500


def test_a_missing_category_is_priced_at_the_entrys_pick_cost():
    e = dataclasses.replace(fx.entry(), pick=ms.pick_table(EXAMPLE, fx.entry(), fx.PARAMS))
    s = fx.state(e, fx.entry("lab-a/model-one", "low"))
    entry_cost = ms.pick_costs(s)[HIGH]
    assert entry_cost == ms._cost(e.price_in, e.price_out, e.pick.mean_in, e.pick.mean_out)
    assert ms.pick_cost(s, HIGH, "knowledge") == entry_cost
    assert ms.pick_cost(s, HIGH, None) == entry_cost
    code = e.pick.categories[0]
    assert ms.pick_cost(s, HIGH, "code") == ms._cost(e.price_in, e.price_out, code.mean_in,
                                                    code.mean_out)
    assert ms.pick_cost(s, HIGH, "code") != entry_cost
    # The call cost is untouched: exploration reads it.
    assert ms.costs(s)[HIGH] == ms.entry_cost(e) != entry_cost
    with pytest.raises(KeyError):
        ms.pick_cost(s, "lab-z/not-on-the-menu")


def test_pick_means_that_would_make_a_pick_cost_nothing_are_not_taken():
    free_in = fx.entry(price_in=0, mean_out=1)
    # Cells with no completion tokens drive the out mean to zero.
    zero = ms.pick_table((cell("code", 10**6, 10**6, 0),), free_in, fx.PARAMS)
    assert zero is None


# --- the published state ---------------------------------------------------------------


def test_a_pick_table_is_hashed_and_has_one_encoding():
    s = fx.state()
    priced = with_pick(s, HIGH, ms.pick_table(EXAMPLE, s.entry(HIGH), fx.PARAMS))
    assert ms.menu_hash(priced) != ms.menu_hash(s)
    raw = ms.canonical_json(priced)
    assert ms.from_json(raw) == priced
    doc = json.loads(raw)
    high = next(d for d in doc["entries"] if d["id"] == HIGH)
    assert high["pick"] == {"cells": 30, "mean_in": 241_667, "mean_out": 2_533_333,
                            "categories": [
                                {"category": "code", "cells": 10, "mean_in": 261_111,
                                 "mean_out": 2_522_222},
                                {"category": "math", "cells": 20, "mean_in": 220_834,
                                 "mean_out": 3_766_667}]}
    # An entry without graded cells carries no pick field at all.
    assert all("pick" not in d for d in doc["entries"] if d["id"] != HIGH)


@pytest.mark.parametrize("break_it", [
    lambda p: dict(p, cells=31),                                    # categories do not add up
    lambda p: dict(p, cells=0, categories=[]),                      # a table without cells
    lambda p: dict(p, categories=list(reversed(p["categories"]))),  # out of order
    lambda p: dict(p, categories=[p["categories"][0]] * 2, cells=20),  # a category twice
    lambda p: dict(p, categories=[dict(p["categories"][0], cells=0)] + p["categories"][1:],
                   cells=20),                                       # an empty category
    lambda p: dict(p, mean_in=-1),
    lambda p: dict(p, mean_out="2533333"),
    lambda p: dict(p, extra=1),
    lambda p: {k: v for k, v in p.items() if k != "categories"},
    lambda p: dict(p, categories=[dict(p["categories"][0], category="a b")] + p["categories"][1:]),
    # A name past PICK_CATEGORY_MAX_CHARS, and more categories than
    # PICK_MAX_CATEGORIES: every list in a menu is capped.
    lambda p: dict(p, categories=[dict(p["categories"][0], category="c" * 33)]
                   + p["categories"][1:]),
    lambda p: dict(p, cells=9, categories=[
        {"category": f"c{i}", "cells": 1, "mean_in": 1, "mean_out": 1} for i in range(9)]),
])
def test_a_malformed_pick_table_is_not_a_menu(break_it):
    s = fx.state()
    priced = with_pick(s, HIGH, ms.pick_table(EXAMPLE, s.entry(HIGH), fx.PARAMS))
    doc = json.loads(ms.canonical_json(priced))
    for d in doc["entries"]:
        if d["id"] == HIGH:
            d["pick"] = break_it(d["pick"])
    with pytest.raises(ValueError):
        ms.from_dict(doc)


def test_the_reader_holds_a_pick_table_to_what_its_cells_can_give():
    """Every graded cell's tokens are at most the answer cap, so a table is
    between its prior's share of the call means (every cell free) and that
    share plus every cell at the cap. A root's table outside that range, or
    one that prices a pick at nothing, is refused like any other cap."""
    s = fx.state(epoch=119)
    good = with_pick(s, HIGH, ms.pick_table(EXAMPLE, s.entry(HIGH), fx.PARAMS))
    assert ms.check_caps(good, CADENCE) is None
    table = good.entry(HIGH).pick
    k = CADENCE.pick_prior_cells_entry
    floor_in = int((2 * Fraction(k * s.entry(HIGH).mean_in, table.cells + k) + 1) // 2)
    cap_out = int((2 * Fraction(table.cells * ms.MILLI * ms.menu_rules.MAX_ANSWER_TOKENS
                                + k * s.entry(HIGH).mean_out, table.cells + k) + 1) // 2)
    for bad in (dataclasses.replace(table, mean_in=floor_in - 1),
                dataclasses.replace(table, mean_out=cap_out + 1),
                dataclasses.replace(table, categories=(
                    dataclasses.replace(table.categories[0], mean_out=ms.MAX_INT),
                    table.categories[1])),
                dataclasses.replace(table, mean_in=0, mean_out=0)):
        why = ms.check_caps(with_pick(s, HIGH, bad), CADENCE)
        assert why is not None and "pick" in why
    edge = dataclasses.replace(table, mean_in=floor_in)
    assert ms.check_caps(with_pick(s, HIGH, edge), CADENCE) is None


# --- the step: refreshed at the cadence, reset by a join or an update -------------------


def test_at_a_refresh_the_step_computes_the_table_from_the_graded_cells():
    s = fx.state(epoch=119)                           # the step derives epoch 120
    out = run(s, CADENCE, pick_stats=(ms.PickStat(HIGH, 1, EXAMPLE),)).state
    assert out.entry(HIGH).pick == ms.pick_table(EXAMPLE, s.entry(HIGH), CADENCE)
    assert out.entry(LOW).pick is None and out.entry(TWO).pick is None
    assert ms.check_transition(s, out, CADENCE) is None


def test_the_table_shrinks_toward_the_call_means_refreshed_in_the_same_step():
    s = fx.state(epoch=119)
    stat = ms.TokenStat(HIGH, 1, (ms.TokenCell("code", 40, 40 * 400, 40 * 1_200),))
    out = run(s, CADENCE, token_stats=(stat,), pick_stats=(ms.PickStat(HIGH, 1, EXAMPLE),)).state
    assert out.entry(HIGH).mean_out != s.entry(HIGH).mean_out
    assert out.entry(HIGH).pick == ms.pick_table(EXAMPLE, out.entry(HIGH), CADENCE)


def test_outside_the_cadence_the_table_stays_as_published():
    s = fx.state(epoch=101)
    out = run(s, CADENCE, pick_stats=(ms.PickStat(HIGH, 1, EXAMPLE),))
    assert out.state == ms.carry(s) and out.events == ()
    priced = with_pick(fx.state(epoch=101), HIGH,
                       ms.pick_table(EXAMPLE, s.entry(HIGH), CADENCE))
    again = run(priced, CADENCE).state                 # no stat: still not a refresh epoch
    assert again.entry(HIGH).pick == priced.entry(HIGH).pick


def test_cells_of_another_revision_are_not_read_and_an_entry_without_cells_is_reset():
    s = with_pick(fx.state(epoch=119), HIGH,
                  ms.pick_table(EXAMPLE, fx.state().entry(HIGH), CADENCE))
    out = run(s, CADENCE, pick_stats=(ms.PickStat(HIGH, 2, EXAMPLE),)).state
    assert out.entry(HIGH).pick is None
    assert ms.pick_costs(out)[HIGH] == ms.costs(out)[HIGH]


def test_the_step_refuses_pick_inputs_that_contradict_themselves():
    s = fx.state(epoch=101)                            # checked outside a refresh too
    cap = ms.menu_rules.MAX_ANSWER_TOKENS
    for bad in ((ms.PickStat(HIGH, 1, EXAMPLE), ms.PickStat(HIGH, 1, EXAMPLE)),
                (ms.PickStat(HIGH, 1, (cell("code", 1, 1, 1), cell("code", 1, 1, 1))),),
                (ms.PickStat(HIGH, 1, (cell("code", 0, 0, 5),)),),
                (ms.PickStat(HIGH, 1, (cell("code", 2, 0, 2 * cap + 1),)),),
                (ms.PickStat(HIGH, 1, (cell("code", 2, 2 * cap + 1, 0),)),),
                (ms.PickStat(HIGH, 1, (cell("code", -1, 0, 0),)),),
                (ms.PickStat(HIGH, 0, EXAMPLE),),
                (ms.PickStat(HIGH, 1, (cell("", 1, 1, 1),)),),
                (ms.PickStat(HIGH, 1, (cell("c" * 33, 1, 1, 1),)),),
                (ms.PickStat(HIGH, 1, tuple(cell(f"c{i}", 1, 1, 1) for i in range(9))),)):
        with pytest.raises(ValueError):
            run(s, CADENCE, pick_stats=bad)


def test_a_joining_entry_starts_at_its_call_cost_and_keeps_it_until_a_refresh():
    """An entry a join delta adds has no pick table; the step that makes it
    full outside a refresh epoch gives it none either. (Requests to change
    an entry on the menu are not proposals: chain_proposals refuses them as
    already_on_menu, so no revision changes outside a step's own rules.)"""
    new = "lab-c/model-new@high"
    joined = fx.provisional(fx.state(epoch=100))
    assert joined.entry(new).pick is None
    assert ms.pick_costs(joined)[new] == ms.costs(joined)[new]
    out = run(ms.carry(joined), CADENCE, gates=(fx.gate(new),)).state      # derives 102
    assert not out.entry(new).provisional and out.entry(new).pick is None
    assert ms.pick_costs(out)[new] == ms.costs(out)[new]


# --- the reader --------------------------------------------------------------------------


def test_the_reader_refuses_a_table_moved_outside_the_cadence():
    prev = fx.state()                                  # epoch 101: 102 is no refresh epoch
    nxt = dataclasses.replace(ms.carry(prev), serial=prev.serial + 1)
    moved = with_pick(nxt, HIGH, ms.pick_table(EXAMPLE, prev.entry(HIGH), CADENCE))
    why = ms.check_transition(prev, moved, CADENCE)
    assert why is not None and "refresh" in why
    with pytest.raises(ms.MenuRefused, match="refresh"):
        ms.menu_for_epoch(prev, ms.menu_hash(moved), ms.canonical_json(moved),
                          allowlist=fx.ALLOW, params=CADENCE)
    # Dropped, or one category moved: the same refusal.
    priced = with_pick(prev, HIGH, ms.pick_table(EXAMPLE, prev.entry(HIGH), CADENCE))
    for other in (with_pick(nxt, HIGH, None),
                  with_pick(nxt, HIGH, dataclasses.replace(
                      priced.entry(HIGH).pick, categories=(
                          dataclasses.replace(priced.entry(HIGH).pick.categories[0],
                                              mean_out=2_522_223),
                          priced.entry(HIGH).pick.categories[1])))):
        assert "refresh" in (ms.check_transition(priced, other, CADENCE) or "")
    # At the cadence it is a refresh.
    at = fx.state(epoch=119)
    assert ms.check_transition(at, dataclasses.replace(moved, epoch=120), CADENCE) is None


def test_the_reader_refuses_an_entry_made_full_with_a_table():
    new = "lab-c/model-new@high"
    start = ms.carry(fx.provisional(fx.state(epoch=100)))
    honest = run(start, CADENCE, gates=(fx.gate(new),)).state
    assert ms.check_transition(start, honest, CADENCE) is None
    e = honest.entry(new)
    bad = with_pick(honest, new, flat(1, e.mean_in, e.mean_out))
    why = ms.check_transition(start, bad, CADENCE)
    assert why is not None and ("call cost" in why or "refresh" in why), why


# --- picks read the pick table, exploration the call table ------------------------------------


def _flat_head(models, logits):
    import io

    from fugal_subnet import success_test_fixtures as sf
    from fugal_subnet.head_eval import load_head_v4

    z = dict(sf.arrays(np.zeros((1, 1024), np.float32), np.zeros(1, np.float32), ["lab/x"]))
    for k in ("mean_in_tokens", "mean_out_tokens", "cost_profile_id"):
        z.pop(k, None)
    z.update(W=np.zeros((len(models), 1024), np.float32),
             b=np.asarray(logits, dtype=np.float32), models=np.asarray(list(models), "U64"))
    buf = io.BytesIO()
    np.savez(buf, **z)
    return load_head_v4(buf.getvalue())


def _hidden(n):
    h = np.random.RandomState(3).randn(n, 1024).astype(np.float32)
    return h / np.linalg.norm(h, axis=1, keepdims=True)


def test_changing_the_pick_table_changes_picks_and_the_call_table_changes_only_shares():
    s = fx.wide_state()
    ids = list(s.ids())
    # Equal odds on HIGH and LOW: the cheaper pick wins.
    head = _flat_head([HIGH, LOW], [1.0, 1.0])
    hidden = _hidden(4)
    base = head_odds.picks_and_odds_v4(head, s, hidden).picks
    assert set(base) == {min((HIGH, LOW), key=lambda i: (ms.pick_costs(s)[i], ids.index(i)))}
    cheap = base[0]
    other = LOW if cheap == HIGH else HIGH
    # The pick table says the cheap entry costs a dollar more a call on the
    # questions routers send it: the pick moves.
    dear = flat(1000, s.entry(cheap).mean_in,
                s.entry(cheap).mean_out + ms.PICO * ms.MILLI // s.entry(cheap).price_out)
    repriced = with_pick(s, cheap, dear)
    ms.validate(repriced)
    assert ms.costs(repriced) == ms.costs(s)
    assert set(head_odds.picks_and_odds_v4(head, repriced, hidden).picks) == {other}
    assert epoch_plan.explore_shares_v4(repriced) == epoch_plan.explore_shares_v4(s)
    # The call table moves the other way (the cheap entry's call means up
    # a dollar's worth): exploration's shares move, the picks do not.
    called = dataclasses.replace(s, entries=tuple(
        dataclasses.replace(e, mean_out=dear.mean_out, pick=flat(1000, e.mean_in, e.mean_out))
        if e.id == cheap else e for e in s.entries))
    pinned = dataclasses.replace(s, entries=tuple(
        dataclasses.replace(e, pick=flat(1000, e.mean_in, e.mean_out))
        if e.id == cheap else e for e in s.entries))
    assert ms.pick_costs(called) == ms.pick_costs(pinned) == ms.pick_costs(s)
    assert (head_odds.picks_and_odds_v4(head, called, hidden).picks
            == head_odds.picks_and_odds_v4(head, pinned, hidden).picks == base)
    assert epoch_plan.explore_shares_v4(called) != epoch_plan.explore_shares_v4(pinned)


def test_picks_are_priced_at_the_pick_cost_exactly():
    s = fx.state()
    priced = with_pick(s, HIGH, ms.pick_table(EXAMPLE, s.entry(HIGH), fx.PARAMS))
    head = _flat_head([HIGH, LOW, TWO], [0.3, 0.2, 0.1])
    out = head_odds.picks_and_odds_v4(head, priced, _hidden(2))
    costs = ms.pick_costs(priced)
    for q, pick in enumerate(out.picks):
        utils = {e: Fraction(int(out.odds[q, j]), head_odds.ODDS_SCALE) - costs[e]
                 for j, e in enumerate(priced.ids())}
        best = max(utils.values())
        assert pick == next(e for e in priced.ids() if utils[e] == best)
