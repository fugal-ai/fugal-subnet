"""Protocol v4: the pick-cost table's inputs, from verified proofs (W13;
fugal_subnet/menu_inputs.py).

A graded cell is one (question, entry) of an epoch's scored questions that at
least one verified proof answered without failing. Its tokens are a vote of
operators: each coldkey's answers to it are one vote (the mean over its
hotkeys), the cell's tokens the median across coldkeys, rounded half up to
whole tokens, and never more than the answer cap (completion tokens never
more than the entry's own answer limit). The epoch's summary keeps, per
(entry, revision, category), how many cells and their token sums; the step
reads them summed over the window, for each entry's current revision, as
`StepInputs.pick_stats`. No network.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from fugal_subnet import menu_inputs as mi
from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from tests import menu_fixtures as mf
from tests.test_menu_inputs import CATS, _r, _vp

HIGH, LOW, TWO = "lab-a/model-one@high", "lab-a/model-one@low", "lab-b/model-two"
STATE = mf.state()                         # the menu of epoch 101


def _graded(proofs, state=STATE):
    return {(c.entry_id, c.category): c for c in mi.graded_cells(proofs, state, CATS)}


def test_a_cells_tokens_are_the_median_of_its_coldkeys():
    # q000 is a code question. Three operators answered (HIGH, q000); one
    # of them from three hotkeys with different counts.
    proofs = [_vp("5Fa1", "ck-a", [_r("q000", HIGH, p_tok=100, c_tok=1_000)]),
              _vp("5Fa2", "ck-a", [_r("q000", HIGH, p_tok=100, c_tok=2_000)]),
              _vp("5Fa3", "ck-a", [_r("q000", HIGH, p_tok=100, c_tok=6_000)]),
              _vp("5Fb", "ck-b", [_r("q000", HIGH, p_tok=110, c_tok=500)]),
              _vp("5Fc", "ck-c", [_r("q000", HIGH, p_tok=121, c_tok=700)])]
    got = _graded(proofs)[(HIGH, "code")]
    # Votes: ck-a 3,000 (its three hotkeys' mean), ck-b 500, ck-c 700.
    assert got == mi.GradedCell(HIGH, 1, "code", 1, 110, 700)


def test_an_even_number_of_coldkeys_takes_the_midpoint_rounded_half_up():
    proofs = [_vp("5Fa", "ck-a", [_r("q000", HIGH, p_tok=100, c_tok=1_000)]),
              _vp("5Fb", "ck-b", [_r("q000", HIGH, p_tok=101, c_tok=1_001)])]
    assert _graded(proofs)[(HIGH, "code")] == mi.GradedCell(HIGH, 1, "code", 1, 101, 1_001)


def test_a_failed_call_is_not_a_cell_and_exploration_is_not_graded_here():
    proofs = [_vp("5Fa", "ck-a", [_r("q000", HIGH, failed=True, c_tok=99_999),
                                  _r("q001", HIGH, c_tok=300),
                                  _r("q002", HIGH, explore=True, c_tok=50_000)]),
              _vp("5Fb", "ck-b", [_r("q001", HIGH, failed=True, c_tok=99_999)])]
    got = _graded(proofs)
    assert set(got) == {(HIGH, "hard")}                         # q001 only
    assert got[(HIGH, "hard")] == mi.GradedCell(HIGH, 1, "hard", 1, 100, 300)


def test_cells_are_summed_per_entry_revision_and_category():
    proofs = [_vp("5Fa", "ck-a", [_r("q000", HIGH, c_tok=100), _r("q004", HIGH, c_tok=300),
                                  _r("q001", LOW, c_tok=50), _r("q002", TWO, c_tok=7)])]
    cells = mi.graded_cells(proofs, STATE, CATS)
    assert cells == tuple(sorted(cells, key=lambda c: (c.entry_id, c.rev, c.category)))
    assert _graded(proofs)[(HIGH, "code")] == mi.GradedCell(HIGH, 1, "code", 2, 200, 400)
    assert len(cells) == 3


def test_a_cells_tokens_are_bounded_by_the_answer_limit_and_the_cap():
    """A cell answered by one operator is that operator's alone. Whatever it
    reports, the cell counts no more completion tokens than the entry's own
    answer limit and no more prompt tokens than the answer cap, so one cell
    moves an entry's table by a bounded amount."""
    limited = dataclasses.replace(STATE, entries=tuple(
        dataclasses.replace(e, max_tokens=4_096) if e.id == HIGH else e for e in STATE.entries))
    proofs = [_vp("5Fa", "ck-a", [_r("q000", HIGH, p_tok=10_000_000, c_tok=10_000_000)])]
    assert _graded(proofs, limited)[(HIGH, "code")] == mi.GradedCell(
        HIGH, 1, "code", 1, mr.MAX_ANSWER_TOKENS, 4_096)


def test_a_scored_route_off_the_menu_is_refused():
    with pytest.raises(ValueError, match="not on the menu"):
        mi.graded_cells([_vp("5Fa", "ck-a", [_r("q000", "lab-z/elsewhere")])], STATE, CATS)
    with pytest.raises(ValueError, match="no category"):
        mi.graded_cells([_vp("5Fa", "ck-a", [_r("q999-unknown", HIGH)])], STATE, CATS)


def test_the_order_of_proofs_and_results_changes_nothing():
    import random

    results = [_r(f"q{i:03d}", (HIGH, LOW, TWO)[i % 3], c_tok=100 + 37 * i, p_tok=50 + i)
               for i in range(60)]
    proofs = [_vp(f"5F{j}", f"ck-{j % 4}", results[j::5]) for j in range(5)]
    want = mi.graded_cells(proofs, STATE, CATS)
    rnd = random.Random(5)
    for _ in range(5):
        shuffled = [mi.VerifiedProof(dataclasses.replace(vp.proof, results=rnd.sample(
            vp.proof.results, len(vp.proof.results))), vp.coldkey)
            for vp in rnd.sample(proofs, len(proofs))]
        assert mi.graded_cells(shuffled, STATE, CATS) == want


# --- the summary and the window -----------------------------------------------------------


def test_the_summary_carries_the_graded_cells_in_its_one_json_form():
    proofs = [_vp("5Fa", "ck-a", [_r("q000", HIGH, c_tok=1_000), _r("q001", LOW)])]
    s = mi.summarize_epoch(101, STATE, proofs, pooled=[], category_of=CATS, counted_heads=())
    assert s.graded == mi.graded_cells(proofs, STATE, CATS) and s.graded
    d = mi.summary_to_dict(s)
    assert d["graded"] == [dataclasses.asdict(c) for c in s.graded]
    assert mi.summary_from_dict(json.loads(json.dumps(d, sort_keys=True))) == s


def test_a_summarys_graded_rows_are_typed_and_bounded():
    good = {"epoch": 100, "votes": [], "calls": [], "pairs": [], "use": [],
            "graded": [{"entry_id": HIGH, "rev": 1, "category": "code", "cells": 2,
                        "tokens_in": 300, "tokens_out": 9_000}]}
    mi.summary_from_dict(good)
    row = good["graded"][0]
    cap = mr.MAX_ANSWER_TOKENS
    for key, bad in (("cells", "2"), ("cells", 0), ("cells", 1.0), ("tokens_in", -1),
                     ("tokens_out", True), ("rev", 0), ("category", ""), ("entry_id", 7),
                     ("tokens_out", 2 * cap + 1), ("tokens_in", 2 * cap + 1)):
        doc = json.loads(json.dumps(good))
        doc["graded"][0][key] = bad
        with pytest.raises(ValueError):
            mi.summary_from_dict(doc)
    for doc in ({k: v for k, v in good.items() if k != "graded"},
                dict(good, graded=[row, row]),
                dict(good, graded=[dict(row, category="math"), row]),
                dict(good, graded=[dict(row, extra=1)])):
        with pytest.raises(ValueError):
            mi.summary_from_dict(doc)


def _summary(epoch, *graded):
    return mi.EpochSummary(epoch, (), (), (), (), tuple(graded))


def test_pick_stats_sum_the_window_for_the_current_revision_from_its_join():
    state = dataclasses.replace(mf.state(), entries=tuple(
        dataclasses.replace(e, rev=2, joined_epoch=0) if e.id == LOW else
        dataclasses.replace(e, joined_epoch=97, scored_from=97) if e.id == TWO else e
        for e in mf.state().entries))
    window = [_summary(96, mi.GradedCell(HIGH, 1, "code", 2, 20, 200),
                       mi.GradedCell(TWO, 1, "code", 9, 9, 9)),            # before TWO joined
              _summary(98, mi.GradedCell(HIGH, 1, "code", 3, 30, 3_000),
                       mi.GradedCell(HIGH, 1, "math", 1, 10, 100),
                       mi.GradedCell(LOW, 1, "code", 5, 5, 5),             # LOW's old revision
                       mi.GradedCell(LOW, 2, "math", 4, 40, 400),
                       mi.GradedCell(TWO, 1, "hard", 1, 1, 1)),
              _summary(101, mi.GradedCell(HIGH, 1, "code", 100, 1, 1))]     # after `through`
    stats = {s.entry_id: s for s in mi.pick_stats(state, window, through=100)}
    assert stats[HIGH] == ms.PickStat(HIGH, 1, (ms.PickCell("code", 5, 50, 3_200),
                                                ms.PickCell("math", 1, 10, 100)))
    assert stats[LOW] == ms.PickStat(LOW, 2, (ms.PickCell("math", 4, 40, 400),))
    assert stats[TWO] == ms.PickStat(TWO, 1, (ms.PickCell("hard", 1, 1, 1),))
    assert mi.pick_stats(state, [], through=100) == ()


def test_the_step_inputs_carry_the_pick_stats():
    from tests.test_menu_inputs import _summary as summary_of

    start = mf.state(epoch=101)
    proofs = [_vp("5Fa", "ck-a", [_r("q000", HIGH, c_tok=1_000)])]
    this = mi.summarize_epoch(100, mf.state(epoch=100), proofs, pooled=[], category_of=CATS,
                              counted_heads=())
    window = [summary_of(99), this]
    inputs = mi.step_inputs(start, mf.state(epoch=100), window, proofs, epoch=100,
                            nonce=mf.NONCE, lab_list=mf.ALLOW, params=mf.PARAMS,
                            category_of=CATS)
    assert inputs.pick_stats == mi.pick_stats(start, window, through=100)
    assert inputs.pick_stats == (ms.PickStat(HIGH, 1, (ms.PickCell("code", 1, 100, 1_000),)),)
    out = ms.step(start, inputs, mf.PARAMS)
    assert out.entry(HIGH).pick == ms.pick_table(inputs.pick_stats[0].cells, out.entry(HIGH),
                                                 mf.PARAMS)


def test_one_operators_many_hotkeys_cannot_move_a_cell_three_others_answered():
    """Twenty hotkeys of one coldkey report the longest answers they can on
    every cell three other operators also answered: each cell's tokens are
    the honest median, so the entry's table does not move."""
    honest = [_vp(f"5Fh{c}", f"ck-{c}", [_r(f"q{i:03d}", HIGH, c_tok=800 + i) for i in range(8)])
              for c in "BCD"]
    attacker = [_vp(f"5Fa{j:02d}", "ck-A", [_r(f"q{i:03d}", HIGH, c_tok=131_072, p_tok=10**6)
                                            for i in range(8)]) for j in range(20)]
    alone = mi.graded_cells(honest, STATE, CATS)
    attacked = mi.graded_cells(honest + attacker, STATE, CATS)
    assert attacked == alone
