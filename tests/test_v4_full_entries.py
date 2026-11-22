"""Protocol v4: one definition of a FULL entry (menu_state.is_full), read by
the menu minimum (menu_state.full_count, MENU_MIN_ENTRIES) and by a
challenge's picks (picks_v4.challenge_picks), so that the menu step's rules
and the seat's challenge agree on which entries count:

- a provisional entry is never a challenge pick, on either side;
- an entry the step retires (here for the price ceiling, reason
  `price_ceiling`) has left the menu, and no challenge picks it;
- the entries a held-back signed list keeps (`lab_list_held`) stay full and
  are picked like any other full entry.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np

from fugal_subnet import menu_state as ms
from fugal_subnet import picks_v4
from fugal_subnet.head_odds import picks_and_odds_v4
from tests import menu_fixtures as fx

HIDDEN = np.eye(4, dtype=np.float32)[:3]


def _head(state, favour: str, dim: int = 4):
    """A head with a row for every entry of `state`, its odds high only for
    `favour`."""
    ids = list(state.ids())
    return SimpleNamespace(W=np.zeros((len(ids), dim), np.float32),
                           b=np.array([4.0 if m == favour else -4.0 for m in ids], np.float32),
                           models=ids)


def _full(state):
    return {e.id for e in state.entries if ms.is_full(e)}


def test_the_minimum_and_the_challenge_read_one_definition_of_full():
    state = fx.provisional(fx.state(epoch=200))
    new = next(e.id for e in state.entries if e.provisional)
    assert not ms.is_full(state.entry(new))
    assert ms.full_count(state.entries) == len(_full(state)) == len(state.entries) - 1
    for favour in state.ids():
        picks = picks_v4.challenge_picks(_head(state, favour), state, HIDDEN)
        assert set(picks) <= _full(state)
        assert picks == picks_and_odds_v4(_head(state, favour), state, HIDDEN,
                                          exclude=frozenset(state.ids()) - _full(state)).picks
        if favour != new:
            assert set(picks) == {favour}


def test_an_entry_retired_for_the_price_ceiling_is_no_challenge_pick():
    dear = "lab-b/model-dear"
    prev = fx.state(*fx.state(epoch=101).entries,
                    fx.entry(dear, None, provider_order=("host-c",), price_in=1_000_000,
                             price_out=30_000_000), epoch=101)
    assert picks_v4.challenge_picks(_head(prev, dear), prev, HIDDEN) == (dear,) * 3
    reads = (fx.read(dear, "host-c", price_in=1_000_000, price_out=120_000_000),)
    out = ms.step_detail(prev, fx.inputs(epoch=100, host_reads=reads), fx.PARAMS).state
    assert {r.id: r.reason for r in out.retired}[dear] == ms.REASON_PRICE_CEILING
    # The same head (its row for the retired entry is simply not on the menu)
    # picks a full entry of the new menu on every question.
    picks = picks_v4.challenge_picks(_head(prev, dear), out, HIDDEN)
    assert dear not in picks and None not in picks and set(picks) <= _full(out)
    doc = picks_v4.holder_picks_doc("ab" * 32, list(picks), list(out.ids()))
    assert list(picks_v4.picks_from_doc(doc, list(out.ids()))[1]) == list(picks)


def test_entries_a_held_back_list_keeps_are_full_challenge_picks():
    params = dataclasses.replace(fx.PARAMS, min_entries=ms.MENU_MIN_ENTRIES)
    prev = fx.wide_state(fillers=6)                 # six of the nine are lab-d's
    newer = fx.allowlist(3, labs=tuple(x for x in fx.LABS if x != "lab-d"))
    res = ms.step_detail(prev, fx.inputs(lab_list=newer), params)
    assert any(x.kind == "lab_list_held" for x in res.events)
    kept = [e.id for e in res.state.entries if e.model.startswith("lab-d/")]
    assert kept and all(ms.is_full(res.state.entry(k)) for k in kept)
    for k in kept:
        assert picks_v4.challenge_picks(_head(res.state, k), res.state, HIDDEN) == (k,) * 3
