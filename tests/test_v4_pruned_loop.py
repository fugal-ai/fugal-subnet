"""Protocol v4 through the shipped loops on a chain endpoint that keeps only
recent state (tests/v4_world.Chain.keep): a pinned read of an older block
raises as a pruning node answers.

A follower that was down for ten epochs comes back, on an endpoint of its
own that keeps three epochs of state, to boundaries it no longer serves: its published set re-anchors on the root's
snapshot (memo_sync.ChainSet.advance_to) and its menu walk on the menu the
root's newest reveal names (menu_publish.MenuNode._anchor), each verified
against the root's commitment at a boundary the endpoint does serve. It
then agrees with the root, as a follower on a full-history endpoint does:
menu, published set (as published and as kept), memo, seat, digests and
weights, its re-anchor's replay finding every seat holder's head it seeks.
"""
from __future__ import annotations

import pytest

from tests import v4_world as w
from tests.test_menu_publish import LIST_BYTES, signer  # noqa: F401 - the fixture

DOWN = (w.SEAT_FIRST + 2, w.SEAT_FIRST + 11)


@pytest.fixture
def _signature(signer):  # noqa: F811
    return signer.sign(LIST_BYTES)


def test_a_follower_back_after_ten_epochs_on_a_pruning_endpoint_agrees_with_the_root(
        tmp_path, monkeypatch, _signature):
    import json

    from fugal_subnet import memo_sync, v4_replay
    from tests.test_v4_resync_replay import _consensus, _nonzero

    sought: list = []
    inner = v4_replay._holder_heads

    def spy(kw, doc, seat, epoch, store, holder_bytes):
        out = inner(kw, doc, seat, epoch, store, holder_bytes)
        from fugal_subnet import seat_v4
        if isinstance(seat, seat_v4.SeatStateV4) and kw.get("picks") is not None:
            want = {h for h in (seat_v4.challenge_holder(seat, epoch), seat.champion)
                    if h is not None}
            have = (set(str(x) for x in doc["odds"]["heads"].values())
                    | set(out["picks"].heads))
            sought.append((epoch, sorted(want - have)))
        return out
    monkeypatch.setattr(v4_replay, "_holder_heads", spy)

    a, b = DOWN
    keep = memo_sync.ANCHOR_HISTORY_EPOCHS * w.BPE + w.BPE // 2
    loop = w.seat_memo_scenario(tmp_path, monkeypatch, _signature, follower_down=DOWN,
                                last=b + 4, keep=keep, keep_threads=frozenset({"follower+"}))
    assert any(t == "follower+" for t, _b, _now in loop.chain.pruned_reads), \
        "the follower never met a pruned read"
    for k in range(b + 2, loop.last + 1):
        root, follower = loop.reveal(k), loop.reveal(k, "follower")
        assert follower is not None, k
        assert follower["menu"] == root["menu"], k
        assert follower["memo_v4"] == root["memo_v4"], k
        assert follower["seat_v4"] == root["seat_v4"], k
        for key in ("labels_digest", "odds_digest", "seat_digest", "picks_digest", "memo_digest"):
            assert follower["v2_state"][key] == root["v2_state"][key], (k, key)
        assert _consensus(follower["v2_state"]) == _consensus(root["v2_state"]), k
        assert (_nonzero(loop.weights[k][w.FOLLOWER[0]])
                == _nonzero(loop.weights[k][w.ROOT[0]])), k
    # It re-anchored and replayed, and every holder head a replayed epoch
    # sought was found (none left wanting).
    synced = [loop.reveal(k, "follower")["v2_sync"].get("reanchored") for k in
              range(b + 1, loop.last + 1) if loop.reveal(k, "follower") is not None]
    assert any(x and x.get("replayed") for x in synced), synced
    assert sought and all(not missing for _e, missing in sought), sought
    # The published set each keeps is the same one at the same boundary.
    held = {}
    for name in ("root", "follower"):
        with open(loop.dirs[name] / ("state.json" + memo_sync.STATE_SUFFIX), "rb") as f:
            held[name] = json.loads(f.read())
    assert held["follower"]["at"] == held["root"]["at"]
    assert held["follower"]["set"] == held["root"]["set"]
