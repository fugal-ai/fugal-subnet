"""Protocol v4: what the seat is scored on (docs/decisions/LIVE_MENU_V4.md, 8,
9.1, 11).

Under v4 the validator loads heads with the v4 loader (token means optional
and ignored), computes every head's pooled odds with head_odds.pair_odds_v4 on
the epoch's MenuState (rows by entry id; a missing row is scored at what the
entry's allowance gives, labels_v4), and pools the seat's pairs
as the menu step's are: each pair weighted by the set it was drawn in, under
the v4 coldkey cap.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io

import numpy as np
import pytest

from fugal_subnet import explore_pool, head_odds
from fugal_subnet import menu_state as ms
from fugal_subnet import success_test_fixtures as sfx
from fugal_subnet.explore_pool import ExploreAnswer
from fugal_subnet.v2_epoch import EpochPairs, OddsInput, _pool_and_odds  # noqa: F401
from tests import menu_fixtures as fx


def _head(rows, *, token_means=False, scale=0.05, seed=0):
    rng = np.random.RandomState(seed)
    n = len(rows)
    z = dict(sfx.arrays((rng.randn(n, 1024) * scale).astype(np.float32),
                        rng.randn(n).astype(np.float32), list(rows)))
    if not token_means:
        for key in ("mean_in_tokens", "mean_out_tokens", "cost_profile_id"):
            z.pop(key, None)
    buf = io.BytesIO()
    np.savez(buf, **z)
    return buf.getvalue()


def test_a_head_without_token_means_is_bounded_and_loaded_under_v4_only():
    from neurons import validator

    head = _head(fx.state().ids())
    assert validator._head_within_bounds(head, 4) is None
    assert validator._head_within_bounds(head, 3) is not None        # the v3 loader refuses it
    assert validator._head_within_bounds(_head(fx.state().ids(), token_means=True), 3) is None
    big = _head(fx.state().ids(), scale=1e6)
    assert validator._head_within_bounds(big, 4) is not None
    loaded = validator._odds_heads({7: head, 8: b"not a head"}, 4)
    assert sorted(loaded) == [7] and list(loaded[7][2]) == list(fx.state().ids())
    assert validator._odds_heads({7: head}, 3) == {}


def _answers(state, n=6):
    out = []
    for i in range(n):
        for j, eid in enumerate(state.ids()):
            out.append(ExploreAnswer(f"q{i}", eid, f"ck{(i + j) % 4}", (i + j) % 3 != 0,
                                     failed=False, private=i % 2 == 0, prompt_tokens=10,
                                     completion_tokens=20))
    return out


def _hidden(q):
    v = np.random.default_rng(int.from_bytes(hashlib.sha256(q.encode()).digest()[:8], "big")
                              ).standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def test_the_seats_odds_are_pair_odds_v4_on_the_menu_state():
    from fugal_subnet import champion
    from fugal_subnet.head_eval import load_head_v4

    state = fx.state()
    rows = list(state.ids())
    partial = rows[:2]                                   # no row for the third entry
    heads = {5: _head(rows, seed=1), 6: _head(partial, seed=2)}
    loaded = {u: load_head_v4(b) for u, b in heads.items()}
    oi = OddsInput(nonce=b"\x01" * 32, answers=_answers(state), benchmark_of={},
                   heads={u: (h.W, h.b, list(h.models)) for u, h in loaded.items()},
                   hidden_of=_hidden, coldkey_cap=explore_pool.coldkey_cap(4),
                   menu_state=state,
                   explore_weights={e: 2 for e in rows}, private_weights={e: 3 for e in rows})
    serving = {"h5": 5, "h6": 6}
    answered = {h: champion.HeadEpoch(np.zeros(1, np.int64), np.zeros(1, bool)) for h in serving}
    pairs, odds, _implausible, pooled = _pool_and_odds(champion.Window.empty(8), oi, serving,
                                                       answered, None)
    hidden = [_hidden(q) for q in pairs.question_ids]
    want = head_odds.pair_odds_v4(loaded[5], state, hidden, list(pairs.models))
    assert np.array_equal(odds["h5"], want)
    # Each pair weighs the set it was drawn in: 3 private, 2 shared.
    assert {p.weight for p in pooled} == {2, 3}
    # A head with no row for an entry has no odds on that entry's pairs:
    # on a full entry's they are stored -1 and scored at odds 0
    # (FAST_CROWN_V4.md 18); on the other entries it has its own odds.
    # Nothing blanks a head for a missing row.
    on_third = np.array([m == rows[2] for m in pairs.models])
    assert (odds["h6"][on_third] == -1).all() and (odds["h6"][~on_third] >= 0).all()
    assert (pairs.missing == -1).all()
    # The pairs of an entry that is provisional in the epoch are marked so no
    # comparison reads them; a missing row there is still stored -1.
    from fugal_subnet import labels_v4

    young = dataclasses.replace(state, entries=tuple(
        dataclasses.replace(e, joined_epoch=state.epoch, scored_from=ms.PROVISIONAL,
                            measured_rate=5000, proposer=fx.HOTKEY) if e.id == rows[2] else e
        for e in state.entries), provisional=(
            ms.Provisional(rows[2], "ab" * 32, fx.HOTKEY, fx.COLDKEY),))
    ms.validate(young)
    oi = dataclasses.replace(oi, menu_state=young)
    _p, odds, _i, _pooled = _pool_and_odds(champion.Window.empty(8), oi, serving, answered, None)
    assert (odds["h6"][on_third] == -1).all()
    assert (_p.missing[on_third] == labels_v4.PROVISIONAL_LABEL).all()
    assert (_p.missing[~on_third] == -1).all()
    assert np.array_equal(odds["h5"], want)


# That the loop's seat odds are pair_odds_v4 on the epoch's menu, every epoch
# on every validator, is shown by running the loop (tests/test_v4_loop.py),
# which replaced a search of the loop's source text.


def test_the_v4_odds_input_carries_the_v4_pooling():
    from neurons import validator

    state = fx.state()
    from fugal_subnet import epoch_plan
    shares = epoch_plan.explore_shares_v4(state)
    oi = validator._v4_odds_input(
        base=OddsInput(nonce=b"\x02" * 32, answers=[], benchmark_of={}, heads={},
                       hidden_of=_hidden),
        state=state, shares=shares, expected={}, verified_proofs={}, coldkey_of={},
        verified_heads={}, boundary_block=101 * 360, blocks_per_epoch=360)
    assert oi.menu_state is state and oi.coldkey_cap == explore_pool.coldkey_cap(4)
    assert oi.explore_weights == shares.shared_weights()
    assert oi.private_weights == shares.private_weights()
    # v4 has no grace and no full-menu rule: an entry in its allowance is
    # scored like any other (labels_v4), so neither is set.
    assert oi.menu is None and oi.unscored == frozenset()
    young = dataclasses.replace(state, entries=tuple(
        dataclasses.replace(e, joined_epoch=99, scored_from=119) if i == 0 else e
        for i, e in enumerate(state.entries)))
    oi = validator._v4_odds_input(
        base=oi, state=young, shares=shares, expected={}, verified_proofs={},
        coldkey_of={}, verified_heads={}, boundary_block=101 * 360, blocks_per_epoch=360)
    assert oi.menu is None and oi.unscored == frozenset() and oi.menu_state is young
    # The odds are computed on the epoch's own menu, and on no other.
    with pytest.raises(ValueError, match="own menu"):
        validator._v4_odds_input(
            base=oi, state=young, shares=shares, expected={}, verified_proofs={},
            coldkey_of={}, verified_heads={}, boundary_block=102 * 360, blocks_per_epoch=360)
