"""Protocol v4: a provisional entry in a challenge (FAST_CROWN_V4.md 18(d), as
INVARIANTS "I3 - v4: the challenge" now reads it).

A provisional entry is never a challenge pick, on either side: the
challenger's TD, every validator and the published `holder_picks` compute a
challenge's picks over the menu's full entries only (picks_v4.
challenge_picks). A head may still route its users to a provisional entry
(head_odds.picks_and_odds_v4 with no exclusion), and such an entry is priced
at its call cost. In the published comparison with R (`routing_vs_r`) the
holder's pick is its challenge pick, no label of a provisional entry is read
on either side, and R never picks one (it has no rate the field measured
yet)."""
from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from fugal_subnet import labels_v4, picks_v4
from fugal_subnet import menu_state as ms
from fugal_subnet.champion import EpochPairs, Window
from fugal_subnet.head_odds import picks_and_odds_v4
from tests import menu_fixtures as fx

CAT = "math"


def _state():
    state = fx.provisional(fx.state(epoch=200))
    new = [e.id for e in state.entries if e.provisional]
    assert len(new) == 1
    return state, new[0]


def _head(state, favour: str, dim: int = 4):
    """A head with a row for every entry, its odds high only for `favour`."""
    ids = list(state.ids())
    W = np.zeros((len(ids), dim), np.float32)
    b = np.array([4.0 if m == favour else -4.0 for m in ids], np.float32)
    return SimpleNamespace(W=W, b=b, models=ids)


def test_a_provisional_entry_is_routable_priced_at_its_call_cost_and_never_a_challenge_pick():
    state, new = _state()
    hidden = np.eye(4, dtype=np.float32)[:3]
    picks = picks_and_odds_v4(_head(state, new), state, hidden).picks
    assert set(picks) == {new}
    challenge = picks_v4.challenge_picks(_head(state, new), state, hidden)
    assert new not in challenge and None not in challenge
    entry = state.entry(new)
    assert entry.provisional
    assert ms.pick_cost(state, new, CAT) == ms.entry_pick_cost(entry) > 0


def test_a_failed_call_to_a_provisional_entry_fails_closed_on_either_side():
    state, new = _state()
    other = next(m for m in state.ids() if m != new)
    A = picks_v4.Answer

    def item(own, holder_pick, held):
        return picks_v4.challenge_items(state, [CAT], [own], [holder_pick], {0: held})[0]

    # the challenger's own pick provisional: failed scores as answered wrong
    wrong = item(A(new, False, False), other, A(other, False, False))
    failed = item(A(new, True, True), other, A(other, False, False))
    right = item(A(new, True, False), other, A(other, False, False))
    assert failed == wrong < right
    v = picks_v4.call_value(Fraction(1), ms.pick_cost(state, new, CAT))
    assert right - wrong == v - picks_v4.call_value(Fraction(0), ms.pick_cost(state, new, CAT))
    # the holder's pick provisional: its failed call scores as right
    h_right = item(A(other, False, False), new, A(new, True, False))
    h_failed = item(A(other, False, False), new, A(new, False, True))
    h_wrong = item(A(other, False, False), new, A(new, False, False))
    assert h_failed == h_right < h_wrong


def test_the_holders_published_picks_are_its_challenge_picks():
    state, new = _state()
    ids = list(state.ids())
    hidden = np.eye(4, dtype=np.float32)[:5]
    picks = list(picks_v4.challenge_picks(_head(state, new), state, hidden))
    doc = picks_v4.holder_picks_doc("ab" * 32, picks, ids)
    head, got = picks_v4.picks_from_doc(doc, ids)
    assert head == "ab" * 32 and list(got) == picks and new not in picks


def test_routing_vs_r_reads_no_provisional_label_and_r_never_picks_one():
    state, new = _state()
    ids = list(state.ids())
    other = next(m for m in ids if m != new)
    n = 40
    models = tuple(new if i % 2 else other for i in range(n))

    def pairs(e, provisional):
        base = EpochPairs(question_ids=tuple(f"q{e}-{i}" for i in range(n)), models=models,
                          benchmarks=("b",) * n, categories=(CAT,) * n,
                          # both entries always right
                          y=np.full(n, 10_000, np.int32),
                          prompt_tokens=np.zeros(n, np.int32),
                          completion_tokens=np.zeros(n, np.int32))
        missing = np.array([labels_v4.PROVISIONAL_LABEL if (m == new and provisional) else -1
                            for m in models], np.int16)
        return labels_v4.label_pairs(base, [("ck",)] * n, missing)

    window = Window(size=48, slices={}, heads={}, pairs={9: pairs(9, True)}, odds={})
    now = pairs(10, True)
    hidden = {q: np.ones(4, np.float32) for q in now.question_ids}
    got = picks_v4.routing_vs_r(state, now, window, _head(state, new), hidden.__getitem__, now=10)
    # the holder favours the provisional entry, but its challenge pick is a
    # full entry, and no provisional label counts
    hp = set(picks_v4.challenge_picks(_head(state, new), state,
                                      np.ones((1, 4), np.float32)))
    assert new not in hp
    assert got["holder"][0] == (n // 2 if other in hp else 0)
    # R picks among full entries only (the other one, right every time): it
    # scores the other entry's labels, never the provisional one's
    assert got["r"][0] == n // 2
    assert not state.entry(other).provisional and state.entry(new).provisional
