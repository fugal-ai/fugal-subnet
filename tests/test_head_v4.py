"""Protocol v4 heads (docs/decisions/LIVE_MENU_V4.md, section 11; work package W7).

A v4 head is today's success-head artifact, a linear layer on the pinned
question embedding, whose rows are keyed by menu entry id. It need not carry
token means or a cost profile, and any it carries are never read: a pick is
the head's exact odds minus the benchmark lambda times the entry's pick cost on the
epoch's MenuState, in exact arithmetic, ties to the first entry in the state's
order. One function, `head_odds.picks_and_odds_v4`, turns (head, MenuState,
embeddings) into picks and odds for the TD's harness and for the validator.
No network, no spend, no backbone.
"""
from __future__ import annotations

import dataclasses
import io
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fugal_subnet import head_eval, head_odds
from fugal_subnet import menu_state as ms
from fugal_subnet import success_test_fixtures as fx
from fugal_subnet.head_eval import (
    HEAD_MAX_ROWS_V4,
    V4_IGNORED_FIELDS,
    V4_REQUIRED_FIELDS,
    load_head_from_npz,
    load_head_v4,
    v4_head_verdict,
)
from fugal_subnet.head_odds import ODDS_SCALE, MenuOdds, pair_odds_v4, picks_and_odds_v4
from fugal_subnet.vendor import success_contract as contract
from tests import menu_fixtures as mf

DIM = 1024
COST_FIELDS = ("mean_in_tokens", "mean_out_tokens", "cost_profile_id")

SETTINGS = settings(max_examples=40, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])


def _arrays(models, seed=0, scale=0.05):
    """Today's success-head fields for `models` (any number of rows: the
    contract's own builder stops at 64)."""
    rng = np.random.RandomState(seed)
    n = len(models)
    z = dict(fx.arrays(np.zeros((1, DIM), np.float32), np.zeros(1, np.float32), ["lab/x"]))
    z.update(W=(rng.randn(n, DIM) * scale).astype(np.float32),
             b=(rng.randn(n) * 0.5).astype(np.float32),
             models=np.asarray(list(models), dtype="U64"),
             mean_in_tokens=np.full(n, 500.0), mean_out_tokens=np.full(n, 300.0))
    return z


def _bytes(z) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **z)
    return buf.getvalue()


def _without(z, *keys):
    return {k: v for k, v in z.items() if k not in keys}


def _hidden(n, seed=1):
    h = np.random.RandomState(seed).randn(n, DIM).astype(np.float32)
    return h / np.linalg.norm(h, axis=1, keepdims=True)


def _menu(*entries, epoch=101):
    return mf.state(*entries, epoch=epoch)


MENU = _menu()
IDS = list(MENU.ids())  # lab-a/model-one@high, lab-a/model-one@low, lab-b/model-two


# --- what a v4 head is ----------------------------------------------------------


def test_the_fields_a_v4_head_needs_and_the_ones_it_may_omit():
    assert V4_REQUIRED_FIELDS == {"W", "b", "models", "contract", "profile_id",
                                  "backbone_revision", "provenance", "lam"}
    assert V4_IGNORED_FIELDS == set(COST_FIELDS)
    # Together they are exactly today's success-head fields.
    assert V4_REQUIRED_FIELDS | V4_IGNORED_FIELDS == set(_arrays(IDS))


def test_the_row_bound_is_the_menu_cap_plus_the_entries_that_may_be_provisional():
    assert HEAD_MAX_ROWS_V4 == ms.MENU_CAP + ms.MENU_MAX_PROVISIONAL == 72


def test_a_head_without_token_means_loads_under_v4_and_is_refused_under_v3():
    raw = _bytes(_without(_arrays(IDS), *COST_FIELDS))
    head = load_head_v4(raw)
    assert head.models == IDS and head.W.shape == (3, DIM)
    with pytest.raises(ValueError, match="unsupported head fields"):
        load_head_from_npz(raw)
    with pytest.raises(ValueError, match="unsupported head fields"):
        contract.load(raw)


@pytest.mark.parametrize("dropped", [(), ("mean_in_tokens",), ("mean_out_tokens",),
                                     ("cost_profile_id",), ("mean_in_tokens", "mean_out_tokens"),
                                     COST_FIELDS])
def test_any_subset_of_the_cost_fields_may_be_left_out(dropped):
    load_head_v4(_bytes(_without(_arrays(IDS), *dropped)))


def test_a_head_today_is_a_valid_v4_head_and_still_loads_under_v3():
    raw = _bytes(_arrays(IDS))
    v4, v3 = load_head_v4(raw), load_head_from_npz(raw)
    assert v4.models == v3.models
    assert np.array_equal(v4.W, v3.W) and np.array_equal(v4.b, v3.b)


@pytest.mark.parametrize("field", sorted({"W", "b", "models", "contract", "profile_id",
                                          "backbone_revision", "provenance", "lam"}))
def test_every_required_field_is_required(field):
    with pytest.raises(ValueError):
        load_head_v4(_bytes(_without(_arrays(IDS), field)))


def test_an_unknown_field_is_refused():
    z = _arrays(IDS)
    z["cost_head"] = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="unsupported head fields"):
        load_head_v4(_bytes(z))


@pytest.mark.parametrize("key,value", [
    ("mean_in_tokens", np.zeros(2)),                       # wrong length
    ("mean_out_tokens", np.array([1.0, np.nan, 2.0])),     # not finite
    ("mean_in_tokens", np.array([1.0, -1.0, 2.0])),        # negative
    ("cost_profile_id", np.asarray("not-a-hash")),
    ("cost_profile_id", np.asarray("")),
])
def test_a_cost_field_that_is_present_is_well_formed(key, value):
    """Ignored is not unread: an accepted head holds no unbounded array (I2)."""
    z = _arrays(IDS)
    z[key] = value
    with pytest.raises(ValueError):
        load_head_v4(_bytes(z))


@pytest.mark.parametrize("key,expected", [("contract", "x"), ("profile_id", "0" * 64),
                                          ("backbone_revision", "deadbeef")])
def test_the_contract_profile_and_backbone_are_todays(key, expected):
    z = _arrays(IDS)
    z[key] = np.asarray(expected)
    with pytest.raises(ValueError, match=f"unsupported {key}"):
        load_head_v4(_bytes(z))


def test_rows_up_to_the_bound_load_and_one_more_is_refused():
    ids = [f"lab/e{i:03d}" for i in range(HEAD_MAX_ROWS_V4 + 1)]
    z = _arrays(ids[:HEAD_MAX_ROWS_V4])
    assert len(load_head_v4(_bytes(z)).models) == HEAD_MAX_ROWS_V4
    # Over today's 64: a head can hold rows for entries that have not joined.
    assert HEAD_MAX_ROWS_V4 > 64
    with pytest.raises(ValueError, match="rows"):
        load_head_v4(_bytes(_arrays(ids)))


def test_duplicate_row_ids_are_refused():
    z = _arrays(IDS)
    z["models"] = np.asarray([IDS[0], IDS[1], IDS[0]])
    with pytest.raises(ValueError, match="unique"):
        load_head_v4(_bytes(z))


@pytest.mark.parametrize("where,value", [("W", np.nan), ("W", np.inf), ("b", -np.inf),
                                         ("W", 2e6), ("b", -1.5e6)])
def test_non_finite_and_out_of_bound_parameters_are_refused(where, value):
    z = _arrays(IDS)
    a = np.array(z[where], dtype=np.float32)
    a.flat[0] = value
    z[where] = a
    with pytest.raises(ValueError):
        load_head_v4(_bytes(z))


def test_a_parameter_at_the_bound_loads():
    z = _arrays(IDS)
    z["W"] = np.full((3, DIM), head_odds.HEAD_MAX_ABS_PARAM, dtype=np.float32)
    load_head_v4(_bytes(z))


def test_bytes_over_the_cap_are_refused_before_parsing():
    with pytest.raises(ValueError, match="too large"):
        load_head_v4(b"\0" * (head_eval.HEAD_MAX_BYTES + 1))


def test_a_head_needs_a_row_on_the_epochs_menu():
    assert v4_head_verdict(_bytes(_arrays(IDS)), MENU) is None
    assert v4_head_verdict(_bytes(_arrays(["lab/elsewhere"])), MENU) is not None
    assert v4_head_verdict(None, MENU) is not None
    assert v4_head_verdict(b"junk", MENU) is not None
    # One row on the menu is enough; the others are ignored.
    assert v4_head_verdict(_bytes(_arrays([IDS[2], "lab/later@high"])), MENU) is None


# --- rows keyed by id, aligned to the epoch's menu -----------------------------


def _head(models, seed=0):
    return load_head_v4(_bytes(_without(_arrays(models, seed), *COST_FIELDS)))


def _permuted(head, order):
    return dataclasses.replace(head, W=head.W[order], b=head.b[order],
                               models=[head.models[i] for i in order])


def test_odds_are_in_the_menus_order_and_are_the_exact_v3_odds():
    head = _head([IDS[2], IDS[0], IDS[1]])
    hidden = _hidden(5)
    out = picks_and_odds_v4(head, MENU, hidden)
    assert isinstance(out, MenuOdds)
    assert out.entries == MENU.ids() and out.odds.shape == (5, 3) and out.odds.dtype == np.int16
    for q in range(5):
        for e, eid in enumerate(IDS):
            r = head.models.index(eid)
            assert out.odds[q, e] == head_odds.odds(head.W[r], head.b[r], hidden[q])


def test_rows_in_any_order_give_the_same_picks_and_odds():
    head = _head(IDS, seed=3)
    hidden = _hidden(20)
    base = picks_and_odds_v4(head, MENU, hidden)
    for order in ([2, 1, 0], [1, 2, 0], [0, 2, 1]):
        got = picks_and_odds_v4(_permuted(head, order), MENU, hidden)
        assert got.picks == base.picks and np.array_equal(got.odds, base.odds)


def test_a_row_for_an_entry_not_on_the_menu_is_ignored():
    head = _head(IDS, seed=4)
    extra = _head(IDS + ["lab-z/not-yet@high"], seed=4)
    # The same rows for the menu's entries, plus one the menu does not list.
    extra = dataclasses.replace(extra, W=np.vstack([head.W, extra.W[3:]]),
                                b=np.concatenate([head.b, extra.b[3:]]))
    hidden = _hidden(20)
    a, b = picks_and_odds_v4(head, MENU, hidden), picks_and_odds_v4(extra, MENU, hidden)
    assert a.picks == b.picks and np.array_equal(a.odds, b.odds)


def test_an_entry_with_no_row_has_no_odds_and_is_never_picked():
    head = _head([IDS[0], IDS[2]], seed=5)
    hidden = _hidden(30)
    out = picks_and_odds_v4(head, MENU, hidden)
    assert (out.odds[:, 1] == -1).all() and (out.odds[:, [0, 2]] >= 0).all()
    assert IDS[1] not in out.picks and set(out.picks) <= {IDS[0], IDS[2]}


def test_a_head_with_no_row_on_the_menu_gives_no_odds_and_no_picks_without_raising():
    out = picks_and_odds_v4(_head(["lab/elsewhere"]), MENU, _hidden(4))
    assert (out.odds == -1).all() and out.picks == (None,) * 4


def test_a_request_update_keeps_the_row():
    head = _head(IDS, seed=6)
    hidden = _hidden(10)
    e = MENU.entry(IDS[2])
    updated = dataclasses.replace(e, rev=e.rev + 1, max_tokens=e.max_tokens // 2,
                                  explore_pairs=0)
    menu2 = _menu(*[updated if x.id == e.id else x for x in MENU.entries])
    a, b = picks_and_odds_v4(head, MENU, hidden), picks_and_odds_v4(head, menu2, hidden)
    assert np.array_equal(a.odds, b.odds) and a.picks == b.picks


def test_pair_odds_are_the_same_cells_as_the_menu_odds():
    head = _head([IDS[0], IDS[2], "lab/elsewhere"], seed=7)
    hidden = _hidden(6)
    full = picks_and_odds_v4(head, MENU, hidden)
    rows = [hidden[q] for q in (0, 1, 2, 3, 4, 5, 0)]
    ents = [IDS[0], IDS[1], IDS[2], IDS[2], IDS[1], IDS[0], IDS[2]]
    got = pair_odds_v4(head, MENU, rows, ents)
    want = [full.odds[q, IDS.index(e)] for q, e in zip((0, 1, 2, 3, 4, 5, 0), ents)]
    assert got.dtype == np.int16 and got.tolist() == want
    with pytest.raises(ValueError, match="not on"):
        pair_odds_v4(head, MENU, rows[:1], ["lab/elsewhere"])


# --- picks are priced on the menu ------------------------------------------------


def _flat_head(models, logits):
    """A head whose rows ignore the question: row i's logit is logits[i]."""
    z = _without(_arrays(models), *COST_FIELDS)
    z["W"] = np.zeros((len(models), DIM), dtype=np.float32)
    z["b"] = np.asarray(logits, dtype=np.float32)
    return load_head_v4(_bytes(z))


def _priced(menu, prices: dict):
    """`menu` with each named entry's cost set: price_out picodollars per token,
    over a mean of exactly one completion token and no prompt."""
    return _menu(*[dataclasses.replace(e, price_in=0, mean_in=0, price_out=prices[e.id],
                                       mean_out=ms.MILLI) if e.id in prices else e
                   for e in menu.entries])


def test_changing_a_price_on_the_menu_changes_the_pick():
    head = _flat_head(IDS, [2.0, 1.0, 0.0])          # odds: high > low > two
    cheap = {i: 0 for i in IDS}
    hidden = _hidden(3)
    assert set(picks_and_odds_v4(head, _priced(MENU, cheap), hidden).picks) == {IDS[0]}
    # Make the best-odds entry dear: one dollar a call is 10,000 odds units.
    dear = {**cheap, IDS[0]: ms.PICO}
    assert set(picks_and_odds_v4(head, _priced(MENU, dear), hidden).picks) == {IDS[1]}


def test_the_heads_own_token_means_do_not_change_the_pick():
    z = _arrays(IDS, seed=8)
    hidden = _hidden(25)
    a = picks_and_odds_v4(load_head_v4(_bytes(z)), MENU, hidden)
    stale = dict(z)
    stale["mean_in_tokens"] = np.asarray([1e6, 0.0, 3.0])
    stale["mean_out_tokens"] = np.asarray([0.0, 9e5, 1.0])
    stale["cost_profile_id"] = np.asarray("f" * 64)
    b = picks_and_odds_v4(load_head_v4(_bytes(stale)), MENU, hidden)
    c = picks_and_odds_v4(load_head_v4(_bytes(_without(z, *COST_FIELDS))), MENU, hidden)
    assert a.picks == b.picks == c.picks
    assert np.array_equal(a.odds, b.odds) and np.array_equal(a.odds, c.odds)


def test_utility_is_exact_and_a_tie_goes_to_the_first_entry_in_the_menus_order():
    head = _flat_head(IDS, [0.0, 0.0, 0.0])
    hidden = _hidden(1)
    base = picks_and_odds_v4(head, _priced(MENU, {i: 0 for i in IDS}), hidden)
    p = [int(x) for x in base.odds[0]]
    assert p[0] == p[1] == p[2]
    assert base.picks == (IDS[0],)                    # all equal: the first id
    # The first entry dearer by one picodollar a call: the second wins, by an
    # amount no utility on a 1e-4 grid would see.
    assert picks_and_odds_v4(head, _priced(MENU, {IDS[0]: 1, IDS[1]: 0, IDS[2]: 0}),
                             hidden).picks == (IDS[1],)
    # Row two has better odds, and costs exactly the difference more: a tie,
    # which goes to the first of the tied entries in the menu's order.
    head2 = _flat_head(IDS, [0.0, 0.001, -5.0])
    p0, p1, _ = (int(x) for x in picks_and_odds_v4(head2, MENU, hidden).odds[0])
    assert p1 > p0
    # One odds unit is 1/10,000 of a dollar at lambda 1: 10^8 picodollars here.
    gap = (p1 - p0) * (ms.PICO // ODDS_SCALE)
    for extra, want in ((0, IDS[0]), (-1, IDS[1]), (1, IDS[0])):
        menu = _priced(MENU, {IDS[0]: 0, IDS[1]: gap + extra, IDS[2]: 0})
        assert picks_and_odds_v4(head2, menu, hidden).picks == (want,), extra


def test_lambda_is_the_benchmark_setting_and_exact():
    head = _flat_head(IDS, [2.0, 1.0, 0.0])
    menu = _priced(MENU, {IDS[0]: ms.PICO // 10, IDS[1]: 0, IDS[2]: 0})
    hidden = _hidden(2)
    default = picks_and_odds_v4(head, menu, hidden)
    assert default.picks == picks_and_odds_v4(head, menu, hidden, lam=Fraction(1)).picks
    assert picks_and_odds_v4(head, menu, hidden, lam=0).picks == (IDS[0], IDS[0])
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            picks_and_odds_v4(head, menu, hidden, lam=bad)


def test_the_embeddings_must_match_the_heads_width():
    with pytest.raises(ValueError):
        picks_and_odds_v4(_head(IDS), MENU, np.zeros((2, DIM + 1), dtype=np.float32))
    with pytest.raises(ValueError):
        picks_and_odds_v4(_head(IDS), MENU, np.zeros(DIM, dtype=np.float32))


def test_the_state_must_be_a_menu_state():
    with pytest.raises(ValueError):
        picks_and_odds_v4(_head(IDS), {"entries": IDS}, _hidden(1))


# --- properties --------------------------------------------------------------------


def _brute(head, menu, hidden, lam=Fraction(1)):
    """The rule written out: for each question, the menu's entries with a row,
    utility odds / ODDS_SCALE - lam x pick cost, first maximum in the menu's
    order."""
    costs = ms.pick_costs(menu)
    picks, table = [], []
    for h in hidden:
        best, row = None, []
        for eid in menu.ids():
            if eid not in head.models:
                row.append(-1)
                continue
            r = head.models.index(eid)
            p = head_odds.odds(head.W[r], head.b[r], h)
            row.append(p)
            u = Fraction(p, ODDS_SCALE) - lam * costs[eid]
            if best is None or u > best[0]:
                best = (u, eid)
        picks.append(None if best is None else best[1])
        table.append(row)
    return tuple(picks), table


ENTRY_POOL = [f"lab-{c}/m{i}" for c in "ab" for i in range(4)]


@SETTINGS
@given(data=st.data())
def test_property_picks_and_odds_follow_the_rule_for_any_rows_and_menu(data):
    on_menu = data.draw(st.lists(st.sampled_from(ENTRY_POOL), min_size=1, max_size=6,
                                 unique=True), label="menu")
    rows = data.draw(st.lists(st.sampled_from(ENTRY_POOL + ["lab-z/off@high", "lab-z/x"]),
                              min_size=1, max_size=8, unique=True), label="rows")
    prices = data.draw(st.lists(st.integers(0, 3 * 10**8), min_size=len(on_menu),
                                max_size=len(on_menu)), label="prices")
    seed = data.draw(st.integers(0, 2**16), label="seed")
    menu = _menu(*[mf.entry(m, None, provider_order=("host-c",), price_in=0, mean_in=0,
                            price_out=p, mean_out=ms.MILLI)
                   for m, p in zip(on_menu, prices)])
    head = _head(rows, seed=seed % 997)
    hidden = _hidden(4, seed=seed)
    out = picks_and_odds_v4(head, menu, hidden)
    picks, table = _brute(head, menu, hidden)
    assert out.picks == picks and out.odds.tolist() == table
    order = data.draw(st.permutations(range(len(rows))), label="order")
    again = picks_and_odds_v4(_permuted(head, list(order)), menu, hidden)
    assert again.picks == out.picks and np.array_equal(again.odds, out.odds)


@SETTINGS
@given(payload=st.binary(min_size=0, max_size=4096))
def test_property_arbitrary_bytes_raise_only_value_error_under_v4(payload):
    try:
        load_head_v4(payload)
    except ValueError:
        pass


@SETTINGS
@given(n=st.integers(0, HEAD_MAX_ROWS_V4 + 3),
       fill=st.sampled_from([0.0, 1.0, -1.0, 1e30, np.nan, np.inf]),
       drop=st.sets(st.sampled_from(COST_FIELDS)))
def test_property_an_accepted_v4_head_is_bounded(n, fill, drop):
    ids = [f"lab/e{i:03d}" for i in range(n)]
    z = _arrays(ids)
    z["W"] = np.full((n, DIM), fill, dtype=np.float32)
    z["b"] = np.full(n, fill, dtype=np.float32)
    try:
        head = load_head_v4(_bytes(_without(z, *drop)))
    except ValueError:
        return
    assert 1 <= len(head.models) <= HEAD_MAX_ROWS_V4
    assert head.W.shape == (len(head.models), DIM) and head.b.shape == (len(head.models),)
    assert np.isfinite(head.W).all() and np.isfinite(head.b).all()
    assert np.abs(head.W).max() <= head_odds.HEAD_MAX_ABS_PARAM
