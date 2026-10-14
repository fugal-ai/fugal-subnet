"""Diligence 2026-10, finding 5: weak dedup threshold and exposed heads.

Any registered hotkey can query a miner and receive its full router head,
because the minimum stake to query defaults to zero. A copier nudges one bias
term so the head routes a few questions differently and commits it. Dedup
compares routing choices at cosine >= 0.99; measured with the repo's
`find_duplicates`, copies differing on 0 to 2 of 300 questions are caught and
3 or more pass. Weight is proportional to score, so N near-copies collect close
to N shares.

Required behaviour, delivered by PR 5:
  * in live mode a caller needs a validator permit or real stake to pull a head;
  * dedup also compares heads in PARAMETER space, where a nudged copy sits at a
    tiny relative distance and independently trained heads do not;
  * a later head that is a near-duplicate of an earlier one in routing terms
    must beat it by a fixed margin or is disqualified (the epsilon rule).

Expected to fail until PR 5; strict xfail so the marker leaves with the fix.
"""
from __future__ import annotations

import numpy as np
import pytest

from fugal_subnet.dedup import find_duplicates

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="diligence 2026-10 finding 5: copies pass at 3 nudged decisions; heads are free",
    raises=(AssertionError, TypeError, AttributeError, ImportError),
)

MODELS = [f"m/{i}" for i in range(17)]


def _head(seed, W_base=None, nudge=0.0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((17, 1024)).astype(np.float32) if W_base is None else W_base.copy()
    if nudge:
        W = W + rng.standard_normal(W.shape).astype(np.float32) * nudge
    b = np.zeros(17, dtype=np.float32)
    return {"W": W, "b": b, "models": list(MODELS)}


def _decisions(seed, differ_from=None, n_diff=0):
    rng = np.random.default_rng(seed)
    if differ_from is None:
        return rng.integers(0, 17, size=300)
    d = differ_from.copy()
    for i in range(n_diff):
        d[i] = (d[i] + 1) % 17
    return d


def test_relative_distance_separates_copies_from_independent_heads():
    from fugal_subnet.head_distance import relative_distance

    leader = _head(0)
    copy = _head(1, W_base=leader["W"], nudge=1e-4)
    other = _head(2)
    assert relative_distance(leader, copy) < 0.05
    assert relative_distance(leader, other) > 0.5


@pytest.mark.parametrize("n_diff", [3, 5, 10])
def test_a_nudged_copy_is_disqualified_whatever_it_reroutes(n_diff):
    leader_d = _decisions(0)
    copy_d = _decisions(0, differ_from=leader_d, n_diff=n_diff)
    leader = _head(0)
    copy = _head(1, W_base=leader["W"], nudge=1e-4)
    out = find_duplicates({1: leader_d, 2: copy_d}, {1: 100, 2: 200},
                          head_params={1: leader, 2: copy})
    assert out == {2}, f"copy differing on {n_diff}/300 decisions survived: {out}"


def test_independent_heads_are_never_clustered_by_parameters():
    a, b = _head(0), _head(1)
    out = find_duplicates({1: _decisions(0), 2: _decisions(1)}, {1: 100, 2: 200},
                          head_params={1: a, 2: b})
    assert out == set()


def test_later_near_duplicate_must_beat_the_earlier_head_by_epsilon():
    from fugal_subnet.dedup import epsilon_disqualified

    leader_d = _decisions(0)
    near = _decisions(0, differ_from=leader_d, n_diff=12)       # 96% agreement
    scores_tie = {1: 0.100, 2: 0.1002}                          # +0.2%: not enough
    scores_win = {1: 0.100, 2: 0.102}                           # +2%: earned it
    assert epsilon_disqualified({1: leader_d, 2: near}, {1: 100, 2: 200}, scores_tie) == {2}
    assert epsilon_disqualified({1: leader_d, 2: near}, {1: 100, 2: 200}, scores_win) == set()


def test_live_miners_do_not_serve_heads_to_unstaked_callers():
    """The axon access rule, factored out of the closure so it can be tested."""
    from neurons.miner import caller_allowed

    class MG:
        hotkeys = ["5validator", "5miner"]
        validator_permit = [True, False]
        S = [10_000.0, 0.0]

    allowed, _ = caller_allowed(MG(), "5validator", live=True)
    blocked, reason = caller_allowed(MG(), "5miner", live=True)
    assert allowed
    assert not blocked and "stake" in reason.lower() or "permit" in reason.lower()
    # Mock keeps the open default for local testnets and the dress rehearsal.
    assert caller_allowed(MG(), "5miner", live=False)[0]
