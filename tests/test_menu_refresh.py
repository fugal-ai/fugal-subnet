"""Protocol v4: how often the published menu is refreshed
(docs/decisions/LIVE_MENU_V4.md, 4.4).

Prices, answer limits, token means and exploration-pair counts move in the
published state only in a step whose new epoch is a multiple of
`MENU_REFRESH_EPOCHS`, except on the entry that joins or is updated in that
very step, which takes its proposal's values. A structural change elsewhere
(an entry retires, a newer list is adopted, a proposal is filed, decided or
lapses) moves nothing else. So the menu hash changes rarely, a miner's box
needs new menu bytes rarely, and a root that makes one structural change
cannot reprice the whole menu with it. Every node checks the same rule from
the two menus alone (`check_transition`).

The other menu tests run with `fx.PARAMS`, which refreshes in every step so
that each of them states only the rule it is about. These use the compiled
cadence.
"""
from __future__ import annotations

import dataclasses

import pytest

from fugal_subnet import menu_state as ms
from tests import menu_fixtures as fx

HIGH, LOW, TWO = "lab-a/model-one@high", "lab-a/model-one@low", "lab-b/model-two"
CADENCE = dataclasses.replace(fx.PARAMS, refresh_epochs=ms.MENU_REFRESH_EPOCHS)


def run(state, params=CADENCE, **inputs) -> ms.StepResult:
    return ms.step_detail(state, fx.inputs(epoch=state.epoch - 1, **inputs), params)


def refreshes(state):
    """Inputs that would move a price, a token mean and a count of `state`."""
    return dict(
        host_reads=(fx.read("lab-a/model-one", "host-a/fp8", price_in=150_000),
                    fx.read("lab-a/model-one", "host-b/fp8", price_in=150_000)),
        token_stats=(ms.TokenStat(TWO, 1, (ms.TokenCell("code", 40, 40 * 300, 40 * 2000),)),),
        explore_counts=tuple(ms.ExploreCount(e.id, e.rev, 77) for e in state.entries))


def refreshed(before: ms.MenuState, after: ms.MenuState) -> bool:
    fields = ("price_in", "price_out", "max_tokens", "mean_in", "mean_out", "token_pairs",
              "explore_pairs")
    old = {e.id: e for e in before.entries}
    # An entry the gates make full in the step takes its own values: no refresh.
    return any(e.id in old and e.rev == old[e.id].rev
               and not (old[e.id].provisional and not e.provisional)
               and any(getattr(e, f) != getattr(old[e.id], f) for f in fields)
               for e in after.entries)


def test_the_compiled_cadence_is_twenty_epochs_and_the_reader_holds_it():
    assert ms.MENU_REFRESH_EPOCHS == 20
    assert ms.MenuParams(mix=fx.MIX, min_category_accuracy_bp=0).refresh_epochs == 20
    assert ms.READER_PARAMS.refresh_epochs == 20
    with pytest.raises(ValueError):
        dataclasses.replace(fx.PARAMS, refresh_epochs=0)
    assert ms.refresh_due(120, CADENCE) and not ms.refresh_due(121, CADENCE)


def test_outside_the_cadence_a_step_with_nothing_structural_refreshes_nothing():
    s = fx.state()                                   # epoch 101: the step derives 102
    out = run(s, **refreshes(s))
    assert out.state == ms.carry(s)                  # same serial: the carried menu
    assert out.events == ()


def test_at_the_cadence_a_step_refreshes_everything_it_is_handed():
    s = fx.state(epoch=119)                          # the step derives epoch 120
    out = run(s, **refreshes(s)).state
    assert out.serial == s.serial + 1
    assert out.entry(HIGH).price_in == 150_000
    assert out.entry(TWO).token_pairs == 40
    assert set(out.explore_pairs().values()) == {77}


@pytest.mark.parametrize("structural", ["gate", "lab_list", "device_list", "retirement"])
def test_a_structural_change_does_not_bring_a_refresh_with_it(structural):
    s = ms.carry(fx.provisional(fx.state(epoch=100))) if structural == "gate" else fx.state()
    extra = {
        "gate": dict(gates=(fx.gate("lab-c/model-new@high"),)),
        "lab_list": dict(lab_list=fx.allowlist(3)),
        "device_list": dict(device_list=fx.device_list(1)),
        "retirement": dict(health=(ms.HealthStat(LOW, 1, unlisted_epochs=6),)),
    }[structural]
    out = run(s, **refreshes(s), **extra).state
    assert ms.structural_change(s, out)
    assert out.entry(HIGH).price_in == s.entry(HIGH).price_in
    assert out.entry(HIGH).explore_pairs == s.entry(HIGH).explore_pairs
    assert not refreshed(s, out)
    assert ms.check_transition(s, out, CADENCE) is None


def test_a_provisional_entry_decided_outside_the_cadence_takes_its_values_and_nothing_else_moves():
    """The step that makes a provisional entry full sets its token means from
    its labels in any epoch (as a join set an entry's values); every other
    entry's values wait for the cadence."""
    s = ms.carry(fx.provisional(fx.state(epoch=100)))
    out = run(s, gates=(fx.gate("lab-c/model-new@high", tokens_in=300, tokens_out=1200),),
              **refreshes(s)).state
    new = out.entry("lab-c/model-new@high")
    assert not new.provisional and new.token_pairs == 240
    assert out.entry(HIGH).price_in == s.entry(HIGH).price_in
    assert ms.check_transition(s, out, CADENCE) is None


def test_the_reader_refuses_a_refresh_outside_the_cadence_with_nothing_structural():
    prev = fx.state()
    moved = dataclasses.replace(
        ms.carry(prev), serial=prev.serial + 1,
        entries=tuple(dataclasses.replace(e, explore_pairs=5) if e.id == HIGH else e
                      for e in prev.entries))
    why = ms.check_transition(prev, moved, CADENCE)
    assert why is not None and "refresh" in why
    with pytest.raises(ms.MenuRefused, match="refresh"):
        ms.menu_for_epoch(prev, ms.menu_hash(moved), ms.canonical_json(moved),
                          allowlist=fx.ALLOW, params=CADENCE)
    for field, value in (("price_in", 1), ("max_tokens", 4096), ("mean_out", 1), ("token_pairs", 9)):
        other = dataclasses.replace(moved, entries=tuple(
            dataclasses.replace(e, **{field: value}) if e.id == TWO else e for e in prev.entries))
        assert "refresh" in (ms.check_transition(prev, other, CADENCE) or ""), field
    # The same change at the cadence is a refresh; beside a structural change
    # outside it, it is still refused.
    at = fx.state(epoch=119)
    moved_at = dataclasses.replace(moved, epoch=120, entries=moved.entries)
    assert ms.check_transition(at, moved_at, CADENCE) is None
    with_list = dataclasses.replace(moved, labs=ms.AllowlistRef(3, fx.allowlist(3).sha256))
    assert "refresh" in (ms.check_transition(prev, with_list, CADENCE) or "")


def test_a_provisional_entrys_price_waits_for_the_cadence_too():
    held = ms.carry(fx.provisional(fx.state(epoch=101)))      # joined at 101; scored 101 next
    e = held.entry("lab-c/model-new@high")
    moved = dataclasses.replace(ms.carry(held), serial=held.serial + 1, entries=tuple(
        dataclasses.replace(x, price_in=x.price_in + 1) if x.id == e.id else x
        for x in held.entries))
    # (still provisional: it joined after the scored epoch of this step)
    prev = dataclasses.replace(held, epoch=101)
    moved = dataclasses.replace(moved, epoch=102)
    assert "refresh" in (ms.check_transition(prev, moved, CADENCE) or "")


def test_every_menu_an_honest_step_derives_under_the_cadence_passes_every_check():
    from tests.test_menu_reader_checks import _walk

    seen_due = seen_held = False
    for prev, nxt in _walk(1, params=CADENCE):
        if ms.refresh_due(nxt.epoch, CADENCE):
            seen_due = seen_due or refreshed(prev, nxt)
        else:
            assert not refreshed(prev, nxt), nxt.epoch
            seen_held = True
    assert seen_due and seen_held
