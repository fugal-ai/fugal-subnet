"""Fresh-data confirmation of a seat take.

docs/INVARIANTS.md, "I3 — a take is confirmed on fresh questions". Beating a
settled champion once records the challenger as pending; it takes the seat
when it passes the same test again on questions drawn after that first pass.
"""
import dataclasses

import numpy as np
import pytest

from fugal_subnet import config
from fugal_subnet.champion import (
    Ghost,
    HeadEpoch,
    Pending,
    SeatState,
    Window,
    seat_step,
)
from fugal_subnet.v2_epoch import seat_from_dict, seat_params_from_config, seat_to_dict
from tests.test_champion import PARAMS, Q, cand, constant, make_slice, noisy, oracle_router

# Two epochs of fresh questions confirm; a seat filled from empty is open for two.
CONFIRM = dataclasses.replace(PARAMS, confirm_questions=2 * Q, confirm_open_epochs=2)
CHAMP, GOOD, EQUAL = "champ", "good", "equal"
champion_policy = constant(1)
better = oracle_router                    # beats the constant policy by tens of points


def add(window, epoch, policies):
    sl, diff = make_slice(epoch)
    return window.add_epoch(sl, {
        h: HeadEpoch(routed=np.asarray(p(diff), dtype=np.int64), failed=np.zeros(Q, bool))
        for h, p in policies.items()})


def run(policies_by_epoch, state, params=CONFIRM, window=None, since=0, **kw):
    """Step the seat through the epochs given; returns (decisions by epoch, window)."""
    window = window or Window.empty(48)
    out = {}
    for e in sorted(policies_by_epoch):
        policies = policies_by_epoch[e]
        window = add(window, e, policies)
        cands = {h: cand(h, seniority=i, since=since) for i, h in enumerate(sorted(policies))}
        out[e] = seat_step(state, e, f"e{e:08d}", window, cands, params=params, **kw)
        state = out[e].state
    return out, window


def settled(h=CHAMP):
    # confirm_from None: a champion seated before the rule, settled.
    return SeatState(champion=h, champion_since=0)


def both(epochs, challenger=better):
    return {e: {CHAMP: champion_policy, GOOD: challenger} for e in epochs}


def why(decision, head):
    return [text for h, text in decision.detail if h == head]


# --- the rule off: nothing changes ---------------------------------------------


def test_with_the_rule_off_a_take_is_immediate_and_the_seat_is_published_as_before():
    out, _ = run(both(range(5)), settled(), params=PARAMS)
    assert out[0].event == "take" and out[0].champion == GOOD
    d = seat_to_dict(out[4].state)
    assert set(d) == {"champion", "champion_since", "history", "ghosts",
                      "high_water", "high_water_hard"}
    assert out[4].state.pending == () and out[4].state.confirm_from is None


def test_state_left_by_the_rule_is_dropped_when_it_is_not_in_force():
    state = dataclasses.replace(settled(), pending=(Pending(GOOD, 0, CHAMP),), confirm_from=9)
    out, _ = run({1: {CHAMP: champion_policy, EQUAL: champion_policy}}, state, params=PARAMS)
    assert out[1].state.pending == () and out[1].state.confirm_from is None


# --- pass, wait, confirm ------------------------------------------------------------


def test_the_first_pass_records_the_challenger_and_the_champion_holds():
    out, _ = run(both(range(1)), settled())
    d = out[0]
    assert d.event == "hold" and d.champion == CHAMP
    assert d.state.pending == (Pending(GOOD, 0, CHAMP),)
    assert any("passed the take test" in t for t in why(d, GOOD))


def test_it_takes_the_seat_once_it_passes_again_on_enough_fresh_questions():
    out, _ = run(both(range(4)), settled())
    assert [out[e].event for e in range(4)] == ["hold", "hold", "take", "hold"]
    # One fresh epoch is 300 questions; two are needed.
    assert any("only 300 fresh questions; 600 needed" in t for t in why(out[1], GOOD))
    take = out[2]
    assert take.champion == GOOD and take.bar == CHAMP
    assert any("confirmed on questions drawn since epoch 0" in t for t in why(take, GOOD))
    # Confirmed, so settled at once; every pending record is gone.
    assert take.state.confirm_from == 2 and take.state.pending == ()
    assert take.state.history == (CHAMP,)


def test_the_fresh_test_reads_only_epochs_after_the_first_pass():
    """Better on epochs 0-2, identical to the champion afterwards: the test on
    its whole run keeps passing, the test on the fresh questions never does."""
    plan = both(range(3))
    plan.update({e: {CHAMP: champion_policy, GOOD: champion_policy} for e in range(3, 9)})
    # Pass first at epoch 2, so that everything after it is the equal stretch.
    state = dataclasses.replace(settled(), confirm_from=2)
    out, window = run(plan, state)
    assert out[0].event == "take"             # open until epoch 2: the plain test
    # Same data, the champion settled from the start: it never confirms.
    out, _ = run(plan, settled())
    assert out[0].state.pending == (Pending(GOOD, 0, CHAMP),)
    assert out[2].event == "take"             # epochs 1-2 are fresh and it is better there
    plan2 = {e: {CHAMP: champion_policy, GOOD: (better if e == 0 else champion_policy)}
             for e in range(9)}
    out, _ = run(plan2, settled())
    assert all(out[e].event == "hold" for e in range(9))
    assert out[8].state.pending == (Pending(GOOD, 0, CHAMP),)
    fresh = [t for t in why(out[8], GOOD) if "fresh questions," in t]
    assert fresh and "NV=+0.000" in fresh[0]


def test_a_miss_drops_the_pending_record_and_the_next_pass_starts_again():
    plan = both([0, 1])
    plan[2] = {CHAMP: champion_policy}                    # the challenger misses
    plan.update(both([3, 4, 5, 6]))
    out, _ = run(plan, settled())
    assert out[1].state.pending == (Pending(GOOD, 0, CHAMP),)
    assert out[2].state.pending == ()
    assert any("not verified this epoch" in t for t in why(out[2], GOOD))
    # Its run restarts at 3: recorded again there, confirmed on epochs 4-5.
    assert out[3].state.pending == (Pending(GOOD, 3, CHAMP),)
    assert [out[e].event for e in (3, 4, 5)] == ["hold", "hold", "take"]


def test_a_record_lapses_when_its_first_pass_leaves_the_window():
    plan = {e: {CHAMP: champion_policy, GOOD: (better if e == 0 else champion_policy)}
            for e in range(6)}
    out, _ = run(plan, settled(), window=Window.empty(4))
    assert out[3].state.pending == (Pending(GOOD, 0, CHAMP),)
    assert out[4].state.pending == ()
    assert any("lapsed" in t for t in why(out[4], GOOD))


def test_a_validator_whose_window_is_not_trusted_keeps_the_pending_records():
    """A validator that lost its window (allow_take=False) makes no take and
    no first decision, and it must publish the pending records as its peers
    do. Dropping them as "run was broken" made a restarted root publish a seat
    with no records; followers re-anchor on the root, so every waiting
    challenger lost its progress across the subnet."""
    out, window = run(both(range(1)), settled())
    state = out[0].state
    assert state.pending == (Pending(GOOD, 0, CHAMP),)
    policies = {CHAMP: champion_policy, GOOD: better}
    cands = {h: cand(h, seniority=i, since=0) for i, h in enumerate(sorted(policies))}
    peer = seat_step(state, 1, "e00000001", add(window, 1, policies), cands, params=CONFIRM)
    lost = seat_step(state, 1, "e00000001", add(Window.empty(48), 1, policies), cands,
                     params=CONFIRM, allow_take=False)
    assert peer.event == lost.event == "hold"
    assert lost.state.pending == peer.state.pending == (Pending(GOOD, 0, CHAMP),)
    assert not any("dropped" in t for t in why(lost, GOOD))


def test_a_new_champion_clears_every_other_pending_record():
    other = noisy(oracle_router, 0.05, 7)          # also far better than the champion
    plan = {e: {CHAMP: champion_policy, GOOD: better, "other": other} for e in range(3)}
    out, _ = run(plan, settled())
    assert {p.head_hash for p in out[0].state.pending} == {GOOD, "other"}
    assert out[2].event == "take" and out[2].state.pending == ()


def test_the_decision_does_not_depend_on_candidate_order():
    window = Window.empty(48)
    for e in range(3):
        window = add(window, e, {CHAMP: champion_policy, GOOD: better, "z": better})
    state = dataclasses.replace(settled(), pending=(Pending(GOOD, 0, CHAMP),
                                                    Pending("z", 0, CHAMP)))
    cands = {h: cand(h, seniority=i) for i, h in enumerate([CHAMP, GOOD, "z"])}
    a = seat_step(state, 2, "e2", window, cands, params=CONFIRM)
    b = seat_step(state, 2, "e2", window, dict(reversed(list(cands.items()))), params=CONFIRM)
    assert a.state == b.state and a.champion == b.champion == GOOD   # tie: the senior


# --- the open period, and ghosts -------------------------------------------------


def test_a_seat_filled_from_empty_is_open_to_the_plain_test_for_the_open_period():
    plan = {0: {CHAMP: champion_policy}}
    plan.update(both([1, 2, 3]))
    out, _ = run(plan, SeatState())
    assert out[0].event == "first" and out[0].state.confirm_from == 2
    # Epoch 1 is inside the open period: taken at once, by the plain test.
    assert out[1].event == "take" and out[1].champion == GOOD
    # The taker won by the plain test too, so it gets its own open period.
    assert out[1].state.confirm_from == 3


def test_after_the_open_period_the_champion_is_settled():
    plan = {e: {CHAMP: champion_policy} for e in (0, 1)}
    plan.update(both([2, 3, 4]))
    out, _ = run(plan, SeatState())
    assert out[0].state.confirm_from == 2
    assert [out[e].event for e in (2, 3, 4)] == ["hold", "hold", "take"]


def test_a_champion_that_only_missed_is_still_confirmed_against():
    """A short outage must not reopen a settled seat to one lucky pass."""
    plan = {1: {GOOD: better}}                               # the champion misses
    out, _ = run(plan, settled(), window=add(Window.empty(48), 0,
                                             {CHAMP: champion_policy, GOOD: better}))
    d = out[1]
    assert d.event == "ghost" and d.champion is None
    assert d.state.ghosts == (Ghost(CHAMP, 3, "missed"),)
    assert d.state.pending == (Pending(GOOD, 1, CHAMP),)


def test_a_pending_record_survives_the_champions_miss_and_resume():
    plan = both([0])
    plan[1] = {GOOD: better}                                 # the champion misses
    plan.update(both([2, 3]))
    out, _ = run(plan, settled())
    assert out[1].event == "ghost" and out[1].state.pending == (Pending(GOOD, 0, CHAMP),)
    assert any("resumed" in t for t in why(out[2], CHAMP))
    # Fresh questions are epochs 2 and 3 (the champion answered none in 1).
    assert [out[e].event for e in (2, 3)] == ["hold", "take"]


def test_a_retired_ghost_is_beaten_by_the_plain_test():
    """It answers no new questions, so there is nothing fresh to pair with it."""
    window = Window.empty(48)
    for e in (0, 1):
        window = add(window, e, {CHAMP: champion_policy, GOOD: better})
    plan = {2: {GOOD: better}}
    out, _ = run(plan, settled(), window=window, retired={CHAMP})
    assert out[2].event == "take" and out[2].champion == GOOD
    assert out[2].state.confirm_from == 4            # an open period, not settled


def test_an_expired_ghost_clears_the_rules_state():
    state = SeatState(ghosts=(Ghost(CHAMP, 4, "retired"),),
                      pending=(Pending(GOOD, 1, CHAMP),), confirm_from=3)
    out, _ = run({5: {EQUAL: constant(0)}}, state)
    assert out[5].event == "first"
    assert out[5].state.pending == () and out[5].state.confirm_from == 7


# --- published state ------------------------------------------------------------


def test_the_seat_round_trips_through_the_reveal_with_its_pending_records():
    state = SeatState(champion="a" * 64, champion_since=5, history=("b" * 64,),
                      pending=(Pending("d" * 64, 7, "a" * 64), Pending("c" * 64, 6, "a" * 64)),
                      confirm_from=9)
    d = seat_to_dict(state)
    assert [p["head_hash"][0] for p in d["pending"]] == ["c", "d"]      # sorted by hash
    assert d["confirm_from"] == 9
    assert seat_from_dict(d) == state


@pytest.mark.parametrize("bad", [
    {"pending": "x"}, {"pending": [{"head_hash": "a"}]}, {"confirm_from": "7"},
    {"confirm_from": True}, {"pending": [{"head_hash": "a", "since_epoch": "x", "bar": "b"}]},
])
def test_a_malformed_published_seat_is_refused(bad):
    with pytest.raises((ValueError, KeyError, TypeError)):
        seat_from_dict({"champion": None, **bad})


# --- activation -----------------------------------------------------------------


def test_the_rule_applies_from_the_first_boundary_at_or_after_its_block():
    before = seat_params_from_config(999, 1000)
    at = seat_params_from_config(1000, 1000)
    assert before.confirm_questions == 0 and before.confirm_open_epochs == 0
    assert before.ghost_retire == config.CHAMPION_GHOST_RETIRE
    assert at.confirm_questions == config.CHAMPION_CONFIRM_QUESTIONS == 1500
    assert at.confirm_open_epochs == config.CHAMPION_CONFIRM_OPEN_EPOCHS == 5
    assert at.ghost_retire == config.CHAMPION_GHOST_RETIRE_CONFIRM == 8
    # No block, or no boundary: the rules before it.
    assert seat_params_from_config() == before == seat_params_from_config(10**9, None)


def test_finneys_block_is_compiled_in_and_the_variable_cannot_move_it(monkeypatch):
    finney, other = config.FINNEY_GENESIS_HASH, "0x" + "ab" * 32
    monkeypatch.setattr(config, "FINNEY_CONFIRM_ACTIVATION_BLOCK", 9_500_040)
    monkeypatch.setattr(config, "CONFIRM_ACTIVATION_BLOCK_ENV", None)
    assert config.confirm_activation_block(finney) == 9_500_040
    assert config.confirm_activation_block(finney.upper().replace("0X", "0x")) == 9_500_040
    assert config.confirm_activation_block(other) is None
    # An unreadable genesis cannot be shown not to be finney: the compiled value.
    assert config.confirm_activation_block(None) == 9_500_040

    monkeypatch.setattr(config, "CONFIRM_ACTIVATION_BLOCK_ENV", 123)
    assert config.confirm_activation_block(other) == 123
    for genesis in (finney, None):
        with pytest.raises(ValueError, match="fixed by the release"):
            config.confirm_activation_block(genesis)
    # Saying the same thing as the release is not an override.
    monkeypatch.setattr(config, "CONFIRM_ACTIVATION_BLOCK_ENV", 9_500_040)
    assert config.confirm_activation_block(finney) == 9_500_040
    monkeypatch.setattr(config, "CONFIRM_ACTIVATION_BLOCK_ENV", -1)
    with pytest.raises(ValueError, match="negative"):
        config.confirm_activation_block(other)


def test_the_new_constants_are_consensus_and_in_the_digest():
    from fugal_subnet import fingerprint

    for name in ("FUGAL_CHAMPION_CONFIRM_QUESTIONS", "FUGAL_CHAMPION_CONFIRM_OPEN_EPOCHS",
                 "FUGAL_CHAMPION_GHOST_RETIRE_CONFIRM"):
        assert name in config.CONSENSUS_ENV_VARS
    assert "FUGAL_CONFIRM_ACTIVATION_BLOCK" in config.CONSENSUS_ENV_INPUTS
    fp = fingerprint.environment_fingerprint()
    for name in ("CHAMPION_CONFIRM_QUESTIONS", "CHAMPION_CONFIRM_OPEN_EPOCHS",
                 "CHAMPION_GHOST_RETIRE_CONFIRM"):
        assert fp["scoring"][name] == getattr(config, name)
    assert fp["epoch"]["confirm_activation_block"]["finney"] == \
        config.FINNEY_CONFIRM_ACTIVATION_BLOCK
    moved = dict(fp, epoch=dict(fp["epoch"], confirm_activation_block={"finney": 1, "env": None}))
    assert fingerprint.consensus_digest(moved) != fingerprint.consensus_digest(fp)


# --- end to end: the epoch, the reveal, the rebuild -----------------------------


def _e2e_params():
    # Test slices are 60 questions: one epoch to pass, two fresh epochs to confirm.
    return dataclasses.replace(seat_params_from_config(), min_paired_questions=60,
                               confirm_questions=120, confirm_open_epochs=2)


def _e2e_schedule(e):
    from tests.test_v2_epoch import always, oracle

    pols = {5: always("m/cheap")}
    if e >= 3:
        pols[6] = oracle                  # the challenger arrives at chain epoch 4
    return pols


def _e2e_live(epochs):
    import json

    from tests.test_v2_epoch import run
    from tests.test_v2_rebuild import reveal_of

    window = seat = copy = None
    results, reveals = [], []
    for e in range(epochs):
        pols = _e2e_schedule(e)
        r = run(e, pols, window=window, seat=seat, copy=copy, params=_e2e_params())
        window, seat, copy = r.window, r.decision.state, r.copy_state
        results.append(r)
        reveals.append(json.loads(json.dumps(reveal_of(e, r, pols))))
    return results, reveals


def test_end_to_end_the_champion_is_paid_while_a_challenger_waits_then_the_seat_moves():
    from tests.test_v2_epoch import H, w

    results, reveals = _e2e_live(7)
    events = [r.decision.event for r in results]
    assert events == ["first", "hold", "hold", "hold", "hold", "take", "hold"]
    # Chain epoch 4: the challenger passes and is published as pending.
    seat = reveals[3]["v2_state"]["seat"]
    assert seat["pending"] == [{"head_hash": H(6), "since_epoch": 4, "bar": H(5)}]
    assert seat["champion"] == H(5) and seat["confirm_from"] == 3
    # The champion keeps the whole miner share while it waits.
    assert w(results[3]) == w(results[2]) == {5: 1.0}
    assert w(results[4]) == {5: 1.0}
    # Chain epoch 6: confirmed on epochs 5 and 6, seated, settled at once.
    seat = reveals[5]["v2_state"]["seat"]
    assert seat["champion"] == H(6) and "pending" not in seat and seat["confirm_from"] == 6
    assert w(results[5]) == {6: 0.9, 5: 0.1}


def test_end_to_end_a_validator_rebuilt_mid_confirmation_continues_identically(tmp_path):
    """The pending record is in the published seat, so a validator that
    rebuilds from reveals while a challenger waits confirms it at the same
    epoch as one that never stopped (I1)."""
    from fugal_subnet.copy_rule import from_rebuild
    from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals
    from tests.test_v2_epoch import CATEGORY, PRICES, run
    from tests.test_v2_rebuild import write

    results, reveals = _e2e_live(7)
    write(tmp_path, reveals[:5])                       # through chain epoch 5: still pending
    rb = rebuild_v2_from_reveals(tmp_path, window_size=48, prices=PRICES, bench_category=CATEGORY)
    assert rb.seat == results[4].decision.state and rb.seat.pending
    r = run(5, _e2e_schedule(5), window=rb.window, seat=rb.seat, copy=from_rebuild(rb),
            params=_e2e_params())
    assert r.decision.event == "take" and r.decision.state == results[5].decision.state
    assert r.reveal == results[5].reveal


def test_end_to_end_before_the_block_the_published_seat_has_no_new_field():
    from tests.test_v2_epoch import always, chain, oracle

    r = chain(range(3), {5: always("m/strong"), 6: oracle})
    assert set(r.reveal["v2_state"]["seat"]) == {
        "champion", "champion_since", "history", "ghosts", "high_water", "high_water_hard"}


def test_the_digest_is_the_same_whether_finneys_block_is_also_set_in_the_variable(monkeypatch):
    """On finney the variable is accepted only when it equals the compiled
    block, so setting it changes nothing; the published digest must not say
    otherwise."""
    from fugal_subnet import fingerprint

    monkeypatch.setattr(config, "FINNEY_CONFIRM_ACTIVATION_BLOCK", 9_600_000)
    monkeypatch.setattr(config, "CONFIRM_ACTIVATION_BLOCK_ENV", None)
    unset = fingerprint.consensus_digest()
    monkeypatch.setattr(config, "CONFIRM_ACTIVATION_BLOCK_ENV", 9_600_000)
    assert config.confirm_activation_block(config.FINNEY_GENESIS_HASH) == 9_600_000
    assert fingerprint.consensus_digest() == unset
    # A different block in the variable (another chain's rule) still shows.
    monkeypatch.setattr(config, "CONFIRM_ACTIVATION_BLOCK_ENV", 1_000)
    assert fingerprint.consensus_digest() != unset
