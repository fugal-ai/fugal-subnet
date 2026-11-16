"""The odds seat after the v3 review (2026-11-16), findings 3 and the first fill.

3. A head with no row for a menu entry was scored at the reference's odds
   there. R is fitted on the window's outcomes, so a copy of the champion with
   its weakest row removed beat the champion with no skill of its own. Now a
   missing row is scored as certainly wrong and the head is not eligible for
   the odds seat.
6. (INVARIANTS vs code) An empty seat was filled on odds by any head whose
   skill over R was >= 0 as a point estimate; now only by one whose skill is
   significantly positive, the test a challenger's odds pass against R.
"""
import dataclasses

import numpy as np
import pytest

import tests.test_v3_epoch as T
from fugal_subnet import odds_score
from fugal_subnet.champion import Window
from tests.test_v2_epoch import ACC, MODELS, oracle, run


@pytest.fixture
def heads(monkeypatch):
    saved = dict(T.HEADS)
    yield T.HEADS
    T.HEADS.clear()
    T.HEADS.update(saved)


def _champion_and_copy(delta):
    W, b, _ = T.head("sharp")
    W = W.copy()
    b = b.copy()
    # One row the champion never learned well: a constant, `delta` logits off
    # its model's base rate. The copy is the same head with that row dropped.
    W[0, :] = 0.0
    b[0] = np.log(ACC[MODELS[0]] / (1 - ACC[MODELS[0]])) + delta
    return (W, b, MODELS), (W[1:], b[1:], MODELS[1:])


def _play(epochs, pols, params, **odds_kw):
    window = seat = copy = None
    r = None
    for e in range(epochs):
        uids = tuple(sorted(pols(e)))
        r = run(e, pols(e), window=window, seat=seat, copy=copy, params=params,
                odds_input=T.odds_input(e, uids, **odds_kw))
        window, seat, copy = r.window, r.decision.state, r.copy_state
        yield e, r


@pytest.mark.parametrize("delta", [1.0, 0.5])
def test_a_copy_of_the_champion_with_a_row_removed_does_not_take_the_seat(heads, delta):
    champ, minus_row = _champion_and_copy(delta)
    T.HEADS.clear()
    T.HEADS.update({6: champ, 7: minus_row})
    params = dataclasses.replace(T.params(odds=True), confirm_questions=300, confirm_open_epochs=0)

    def pols(e):
        return {6: oracle, 7: oracle} if e >= 2 else {6: oracle}

    seen_ineligible = False
    for _, r in _play(30, pols, params):
        assert r.decision.champion in (None, T.H(6)), r.decision.detail
        assert not any(h == T.H(7) and "pending" in m for h, m in r.decision.detail)
        seen_ineligible |= any(h == T.H(7) and "not eligible for the odds seat" in m
                               for h, m in r.decision.detail)
    assert r.decision.champion == T.H(6) and seen_ineligible


def test_a_missing_row_is_scored_as_certainly_wrong():
    from fugal_subnet.champion import EpochPairs

    pairs = EpochPairs(question_ids=("a", "b"), models=("m1", "m1"), benchmarks=("x", "x"),
                       categories=("k", "k"), y=np.array([10_000, 0], dtype=np.int32),
                       prompt_tokens=np.zeros(2, dtype=np.int32),
                       completion_tokens=np.zeros(2, dtype=np.int32))
    w = Window.empty(4)
    w = dataclasses.replace(w, pairs={1: pairs}, odds={"h": {1: np.array([-1, -1], np.int16)}})
    assert odds_score.head_odds_on(w, None, "h", 1).tolist() == [0, 10_000]


def test_a_head_that_does_not_cover_the_menu_is_published_with_no_odds(heads):
    _, minus_row = _champion_and_copy(1.0)
    T.HEADS.clear()
    T.HEADS.update({6: T.head("sharp"), 7: minus_row})
    params = T.params(odds=True)
    for _, r in _play(2, lambda e: {6: oracle, 7: oracle}, params, menu=tuple(MODELS)):
        pass
    odds = r.window.odds
    assert (odds[T.H(7)][max(odds[T.H(7)])] == -1).all()
    assert (odds[T.H(6)][max(odds[T.H(6)])] >= 0).all()


def test_an_empty_seat_is_filled_on_odds_only_by_significant_skill(heads):
    # A head whose odds are its models' base rates has no skill over the
    # per-model, per-benchmark reference: it must not fill the seat on noise.
    T.HEADS.clear()
    T.HEADS.update({5: T.head("flat")})
    params = T.params(odds=True)
    flat_reasons = []
    for _, r in _play(12, lambda e: {5: oracle}, params):
        assert r.decision.champion is None, r.decision.detail
        flat_reasons += [m for h, m in r.decision.detail if h == T.H(5)]
    assert any("do not beat the reference" in m and "lower bound" in m for m in flat_reasons)

    # A head that reads the difficulty direction does fill it.
    T.HEADS.clear()
    T.HEADS.update({6: T.head("sharp")})
    events = [r.decision.event for _, r in _play(12, lambda e: {6: oracle}, params)]
    assert "first" in events
