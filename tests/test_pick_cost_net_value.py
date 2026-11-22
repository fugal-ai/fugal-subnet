"""Protocol v4: the validator prices net value at the pick-cost table (W13).

Under v4 a call on the scored questions is priced, for net value, at the
epoch's pick cost of its entry in the question's category
(SEAT_MINIMAL_V4.md 3.4: "each call priced at the measured pick cost of its
entry and category"), not at the call's own attested tokens. The guard's
code path keeps its interface: the slice's spends are what change
(`v2_epoch.build_epoch_slice(pick_spend=...)`), built from the epoch's menu
by the validator (`_v4_tables`), and a rebuild from reveals prices a v4
reveal the same way. v1 to v3 are unchanged: without a pick table the
spends are attested tokens at the pinned prices, as before.
"""
from __future__ import annotations

import dataclasses
import math

from fugal_subnet import menu_state as ms
from fugal_subnet.v2_epoch import build_epoch_slice
from tests import menu_fixtures as fx

HIGH, LOW, TWO = "lab-a/model-one@high", "lab-a/model-one@low", "lab-b/model-two"
EXAMPLE = (ms.PickCell("code", 10, 3_000, 25_000), ms.PickCell("math", 20, 4_000, 100_000))


def _menu():
    s = fx.state()
    table = ms.pick_table(EXAMPLE, s.entry(HIGH), fx.PARAMS)
    return dataclasses.replace(s, entries=tuple(
        dataclasses.replace(e, pick=table) if e.id == HIGH else e for e in s.entries))


class _Menu:
    def __init__(self, state, ok=True):
        self.state, self.ok = state, ok


QUESTIONS = [{"question_id": "q-code", "benchmark": "code_leetcode", "prompt": "x" * 400},
             {"question_id": "q-math", "benchmark": "math_l35", "prompt": "y" * 40},
             {"question_id": "q-know", "benchmark": "mmlupro", "prompt": "z" * 4}]
CELLS = {("q-code", HIGH): 1.0, ("q-math", HIGH): 0.0, ("q-know", HIGH): 1.0,
         ("q-code", TWO): 1.0}
TOKENS = {("q-code", HIGH): 9_999.0, ("q-math", HIGH): 1.0, ("q-know", HIGH): 77.0,
          ("q-code", TWO): 5.0}


def test_the_v4_tables_carry_the_pick_cost_of_every_entry_in_every_category():
    from neurons.validator import _v4_tables

    menu = _menu()
    t = _v4_tables(_Menu(menu))
    cats = set(t.category_targets)
    assert set(t.pick_spend) == set(menu.ids())
    for eid in menu.ids():
        entry_value, by_cat = t.pick_spend[eid]
        assert entry_value == float(ms.pick_cost(menu, eid))
        assert set(by_cat) == cats
        for c in cats:
            assert by_cat[c] == float(ms.pick_cost(menu, eid, c))
    # HIGH's table lists code and math; its other categories are its entry value.
    assert t.pick_spend[HIGH][1]["code"] != t.pick_spend[HIGH][0]
    assert t.pick_spend[HIGH][1]["knowledge"] == t.pick_spend[HIGH][0]
    # An entry without graded cells: its call cost everywhere.
    assert t.pick_spend[TWO][0] == float(ms.costs(menu)[TWO])
    assert _v4_tables(_Menu(menu, ok=False)).pick_spend == {}


def test_a_v4_slice_prices_each_answered_cell_at_the_pick_cost_of_its_category():
    from neurons.validator import _v4_tables

    menu = _menu()
    t = _v4_tables(_Menu(menu))
    models = list(menu.ids())
    sl = build_epoch_slice(7, QUESTIONS, models, CELLS, TOKENS, t.prices, t.bench_category,
                           pick_spend=t.pick_spend)
    j = {m: i for i, m in enumerate(models)}
    assert sl.spends[0, j[HIGH]] == float(ms.pick_cost(menu, HIGH, "code"))
    assert sl.spends[1, j[HIGH]] == float(ms.pick_cost(menu, HIGH, "math"))
    assert sl.spends[2, j[HIGH]] == float(ms.pick_cost(menu, HIGH))          # knowledge: entry
    assert sl.spends[0, j[TWO]] == float(ms.costs(menu)[TWO])
    # The cell's own tokens no longer move its price; cells nobody answered
    # stay empty, as before.
    other = build_epoch_slice(7, QUESTIONS, models, CELLS, {k: 1.0 for k in TOKENS}, t.prices,
                              t.bench_category, pick_spend=t.pick_spend)
    assert (other.spends[~(other.spends != other.spends)]
            == sl.spends[~(sl.spends != sl.spends)]).all()
    assert math.isnan(sl.spends[1, j[TWO]]) and math.isnan(sl.grades[1, j[TWO]])


def test_without_a_pick_table_the_slice_is_priced_as_before():
    prices = {HIGH: (1e-6, 4e-6), TWO: (2e-6, 1e-5)}
    sl = build_epoch_slice(7, QUESTIONS, [HIGH, TWO], CELLS, TOKENS, prices)
    from fugal_subnet.pricing import question_input_tokens
    assert sl.spends[0, 0] == question_input_tokens("x" * 400) * 1e-6 + 9_999.0 * 4e-6


def test_a_rebuild_prices_a_v4_reveal_at_the_table_it_is_handed():
    from neurons.validator import _rebuild_tables, _v4_tables

    t = _v4_tables(_Menu(_menu()))
    tables = _rebuild_tables({}, object(), v4_tables=t)
    assert tables[4] == (t.prices, t.bench_category, t.pick_spend)


def test_the_validator_hands_the_epochs_pick_table_to_the_seat_epoch():
    """The live epoch and the rebuild price alike: the call that runs the
    v4 seat epoch passes the menu's pick spend, and v1 to v3 pass none."""
    import inspect

    import neurons.validator as v

    source = inspect.getsource(v)
    call = source[source.index("v2_result = run_v2_epoch("):]
    call = call[:call.index("\n                )")]
    assert "pick_spend=(v4_tables.pick_spend if v4_tables is not None else None)" in call
