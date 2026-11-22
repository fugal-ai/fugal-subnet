"""Protocol v4: a follower that misses epochs agrees with the root again, and
stays in agreement, through the shipped loops (tests/v4_world.py).

Under v4 an epoch's label pool depends on the window it is pooled on (the
plausibility check's reference R is fitted on the window's earlier pairs),
and so do its late odds and its seat step. A follower that missed an epoch
pools the next one on a window without it; verify-and-resync re-anchors it
on the root's state at an epoch k once the root's reveals pass the gap, and
under v4 it then replays its own epochs after k on that state from the
inputs it recorded (fugal_subnet/v4_replay.py), instead of keeping them as
it first ran them.

The bound these tests hold: a follower whose last missed epoch is b agrees
with the root from b + 2 on (label, odds, seat, picks and every other
consensus field, and the weights). Epoch b + 1 is run before the root's
reveal of b is public (it is published one epoch after it, and read at the
start of the next), so it is computed on the follower's own state and may
differ; at b + 2 the follower re-anchors on the root's state at b and
replays b + 1.

1. The flip world (the cheating TD of tests/test_v4_flip.py, whose label
   pool exclusions change from epoch to epoch): followers that miss the
   first v4 epoch, the second (in which the root's plausibility check
   leaves operators out of the pool), and five epochs in a row.
2. The seat world: a follower down for nine epochs while the label pool's
   plausibility check starts to judge a too-good TD, returning two epochs
   before the root's seat is filled.
"""
from __future__ import annotations

import json

import pytest

from fugal_subnet.benchmarks.slicer import epoch_id_for_block
from tests import v4_world as w
from tests.test_menu_publish import LIST_BYTES, signer  # noqa: F401 - the fixture
from tests.v4_world import BPE, MINERS, FlipWorld, MinerSpec, ValidatorSpec

FLIP = 12
FIRST, LAST = FLIP - 3, FLIP + 8
# name: (first missed epoch, last missed epoch)
DOWN = {"first": (FLIP, FLIP), "second": (FLIP + 1, FLIP + 1), "five": (FLIP + 1, FLIP + 5)}


@pytest.fixture(scope="module")
def flip_world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("resync")
    from fugal_subnet import menu_rules as mr
    from tests.test_menu_publish import Signer

    s = Signer(tmp, "release")
    mp.setattr(mr, "list_signers", lambda: s.signers_line().encode())
    specs = [ValidatorSpec("root", "root", "root", root=True)]
    for name, (a, b) in DOWN.items():
        specs += [ValidatorSpec(name, name, name, stop=a * BPE - 1),
                  ValidatorSpec(name + "+", name, name, start=(b + 1) * BPE)]
    miners = [MinerSpec(hk, ck, seed=i, scored="auto" if i == 0 else "off")
              for i, (hk, ck) in enumerate(MINERS)]
    world = FlipWorld(tmp, mp.setattr, mp.setenv, s, flip=FLIP, first=FIRST, last=LAST,
                      validators=specs, miners=miners, cheater=MINERS[2][0])
    try:
        world.setup()
        world.run()
        yield world
    finally:
        mp.undo()


def _consensus(state: dict) -> dict:
    return {k: v for k, v in state.items() if k != "sync"}


def _nonzero(v: dict) -> dict:
    return {u: x for u, x in v.items() if x != 0}


def test_the_flip_world_ran_every_epoch(flip_world):
    assert flip_world.clock.errors == {} and flip_world.errors == []
    ids = [epoch_id_for_block(k) for k in range(FIRST, LAST + 1)]
    for name, (a, b) in DOWN.items():
        ran = [x["epoch_id"] for x in flip_world.logged(name)]
        assert ran == [i for k, i in zip(range(FIRST, LAST + 1), ids) if not a <= k <= b], name


def test_the_epochs_missed_are_ones_the_root_left_operators_out_of(flip_world):
    """The plausibility check excludes operators on the root in the second
    v4 epoch and in each of the five, so a window without them judges
    differently (the root's first v4 epoch has no earlier pairs: none)."""
    assert not flip_world.reveal("root", FLIP)["exploration_excluded_coldkeys"]
    for k in range(FLIP + 1, FLIP + 6):
        assert flip_world.reveal("root", k)["exploration_excluded_coldkeys"], k


@pytest.mark.parametrize("name", sorted(DOWN))
def test_a_follower_that_missed_epochs_agrees_with_the_root_from_two_epochs_after(
        flip_world, name):
    a, b = DOWN[name]
    by = flip_world.weights()
    root_hk = w.ROOT[0]
    hk = w.validator_keys(name)[0]
    for k in range(b + 2, LAST + 1):
        st, base = flip_world.reveal(name, k)["v2_state"], flip_world.reveal("root", k)["v2_state"]
        for key in ("labels_digest", "odds_digest", "seat_digest", "picks_digest"):
            assert st[key] == base[key], (name, k, key)
        assert st["memo_digest"] == base["memo_digest"], (name, k)
        assert _consensus(st) == _consensus(base), (name, k)
        assert _nonzero(by[k][hk]) == _nonzero(by[k][root_hk]), (name, k)
    # The epoch it replayed ran with the memorisation memo the root's ran
    # with (the published set as of it, from the chain; its plan; its nonce).
    assert w.check_replayed_memos(flip_world.v2_calls, {name + "+"}) == {name + "+": [b + 1]}


@pytest.mark.parametrize("name", sorted(DOWN))
def test_the_state_is_replaced_once_and_the_published_reveals_stand(flip_world, name):
    """The follower re-anchors at b + 2 on the root's state at b and replays
    b + 1; its reveal of b + 2 says so (`v2_sync.reanchored`). The reveal it
    published for b + 1 is not rewritten, and it set weights once an epoch."""
    a, b = DOWN[name]
    sync = flip_world.reveal(name, b + 2)["v2_sync"]
    assert sync["reanchored"] == {"snapshot": epoch_id_for_block(b),
                                  "replayed": [epoch_id_for_block(b + 1)]}, sync
    for k in range(FIRST, LAST + 1):
        if k != b + 2 and flip_world.reveal(name, k) is not None:
            assert "reanchored" not in flip_world.reveal(name, k)["v2_sync"], (name, k)
    hk = w.validator_keys(name)[0]
    per_epoch: dict = {}
    for block, who, _u, _w in flip_world.chain.weights:
        if who == hk:
            per_epoch[block // BPE] = per_epoch.get(block // BPE, 0) + 1
    assert set(per_epoch.values()) == {1}, per_epoch
    if name == "first":
        # Its first v4 epoch ran on a window without the root's first: as
        # published, it still differs from the root's.
        assert (flip_world.reveal(name, FLIP + 1)["v2_state"]["labels_digest"]
                != flip_world.reveal("root", FLIP + 1)["v2_state"]["labels_digest"])


# --- the seat world -----------------------------------------------------------------------

SEAT_DOWN = (w.SEAT_FIRST + 1, w.SEAT_FIRST + 9)


@pytest.fixture
def _signature(signer):  # noqa: F811
    return signer.sign(LIST_BYTES)


def test_a_follower_back_while_the_seat_fills_sets_the_roots_weights(
        tmp_path, monkeypatch, _signature):
    """The root fills the seat two epochs after the follower returns. The
    follower's first epoch back pooled the too-good TD's labels (its window
    lacked the epochs that let R judge them); replayed on the root's state,
    that epoch pools as the root's did, so the follower fills when the root
    fills and sets the root's weights every epoch from b + 2."""
    a, b = SEAT_DOWN
    loop = w.seat_resync_scenario(tmp_path, monkeypatch, _signature, down=SEAT_DOWN,
                                  last=b + 4)
    holders = {k: loop.holder(k) for k in loop.epochs}
    fill = min(k for k in loop.epochs if holders[k] is not None)
    assert fill == b + 3, (fill, holders)
    assert all("5Fcold-x" in loop.reveal(k)["exploration_excluded_coldkeys"]
               for k in range(b + 1, loop.last + 1))
    for k in range(b + 2, loop.last + 1):
        root, follower = loop.reveal(k), loop.reveal(k, "follower")
        assert _nonzero(loop.weights[k][w.FOLLOWER[0]]) == _nonzero(loop.weights[k][w.ROOT[0]]), (
            k, loop.weights[k])
        assert _consensus(follower["v2_state"]) == _consensus(root["v2_state"]), k
        assert follower["seat_v4"] == root["seat_v4"], k
        assert follower["memo_v4"] == root["memo_v4"], k
    assert w.check_replayed_memos(loop.v2_calls, {"follower+"}) == {"follower+": [b + 1]}
    assert json.dumps(loop.reveal(b + 2, "follower")["v2_sync"]["reanchored"]) == json.dumps(
        {"snapshot": epoch_id_for_block(b), "replayed": [epoch_id_for_block(b + 1)]})


# --- the replay itself, on the flip world's records -------------------------------------------


def _replay_inputs(world):
    """(hidden_of, benchmark_of, tables, the epoch's _V4Tables) as the
    validator reads them."""
    import os

    from fugal_subnet import menu_publish as mp
    from fugal_subnet.embeddings_file import Embeddings
    from neurons import validator

    hidden = Embeddings.load(os.environ["FUGAL_EMBEDDINGS_FILE"], None)
    held = world.held("root").held(LAST)
    t = validator._v4_tables(mp.MenuRead(LAST, held, "carried"))
    return (hidden.of, {q["question_id"]: q["benchmark"] for q in world.pool},
            {4: (t.prices, t.bench_category, t.pick_spend)}, t)


def _state_at(world, name, k, tables):
    """The state `name` published at k, rebuilt from its own reveals."""
    from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals

    paths = [str(world.dirs[name] / "epochs" / epoch_id_for_block(j) / "reveal.json")
             for j in range(FLIP, k + 1)]
    t = tables[4]
    return rebuild_v2_from_reveals(paths, window_size=48, prices=t[0], bench_category=t[1],
                                   tables=tables)


def _digest(window, seat, copy_state) -> str:
    """One hash over the state's bytes: the window's arrays and metadata as
    saved, the seat and the copy rule as published."""
    import hashlib
    import tempfile

    import numpy as np

    from fugal_subnet.v2_epoch import save_window, seat_to_dict

    h = hashlib.sha256()
    with tempfile.TemporaryDirectory() as d:
        save_window(d + "/w.npz", window)
        with np.load(d + "/w.npz", allow_pickle=False) as z:
            for key in sorted(z.files):
                h.update(key.encode() + z[key].tobytes())
    h.update(json.dumps([seat_to_dict(seat), copy_state.to_reveal()], sort_keys=True).encode())
    return h.hexdigest()


def _store(world, name):
    from fugal_subnet import v4_replay

    return v4_replay.Store(str(world.dirs[name] / "state.json") + ".v4replay")


def _saved(world, name):
    from fugal_subnet import copy_rule
    from fugal_subnet.v2_epoch import load_window, seat_from_dict

    raw = json.loads((world.dirs[name] / "state.json").read_text())["v2"]
    return (load_window(str(world.dirs[name] / "state.json") + ".v2window.npz", 48),
            seat_from_dict(raw["seat"]), copy_rule.CopyState.from_reveal(raw[copy_rule.REVEAL_KEY]))


def _replay(world, store, k, *, max_span=48):
    from fugal_subnet import copy_rule, v4_replay

    hidden_of, benchmark_of, tables, _t = _replay_inputs(world)
    rebuilt = _state_at(world, "root", k, tables)
    return v4_replay.replay(store, window=rebuilt.window, seat=rebuilt.seat,
                            copy_state=copy_rule.from_rebuild(rebuilt),
                            rules=rebuilt.extra.get("rules"), epochs=range(k + 1, LAST + 1),
                            hidden_of=hidden_of, benchmark_of=benchmark_of,
                            memo_for=w.held_memo_for(world.dirs["root"], world.pool),
                            max_span=max_span)


def test_a_record_runs_the_epoch_it_was_kept_from_to_the_same_reveal(flip_world):
    """Each of the root's v4 epochs, run from its record on its state the
    epoch before, publishes what the root published: the record holds every
    input the epoch reads but its memo, which the validator's replay memo
    function gives from the published set the root keeps and the recorded
    nonce."""
    from fugal_subnet import copy_rule, seat_v4, v2_epoch, v4_replay

    hidden_of, benchmark_of, tables, _t = _replay_inputs(flip_world)
    store = _store(flip_world, "root")
    memo_for = w.held_memo_for(flip_world.dirs["root"], flip_world.pool)
    assert store.epochs() == list(range(FLIP, LAST + 1))
    for j in range(FLIP + 1, LAST + 1):
        before = _state_at(flip_world, "root", j - 1, tables)
        kw = v4_replay.inputs(store.load(j), hidden_of=hidden_of, benchmark_of=benchmark_of,
                              read_head=store.head_bytes)
        got = v2_epoch.run_v2_epoch(**kw, window=before.window, seat=before.seat,
                                    copy_state=copy_rule.from_rebuild(before),
                                    state_rules=before.extra.get("rules"), allow_take=True,
                                    memo=memo_for(j, kw["odds_input"].nonce))
        published = flip_world.reveal("root", j)
        for key in ("v2_state", "champion", seat_v4.REVEAL_KEY, "memo_v4", "labels_v4", "late_odds",
                    "odds_v4", "head_odds", "exploration_pool", "exploration_excluded_coldkeys",
                    "picks_v4", "copy_verdicts"):
            assert json.loads(json.dumps(got.reveal[key])) == published[key], (j, key)
        assert dict(zip(got.uids, got.weights)) == {
            int(u): x for u, x in published["weights"].items()}, j


def test_replaying_the_roots_epochs_on_its_own_state_reproduces_it(flip_world):
    got = _replay(flip_world, _store(flip_world, "root"), FLIP + 2)
    assert got.epochs == tuple(range(FLIP + 3, LAST + 1))
    assert _digest(got.window, got.seat, got.copy_state) == _digest(*_saved(flip_world, "root"))


def test_the_replay_does_not_depend_on_the_order_its_inputs_are_held_in(flip_world, tmp_path):
    """Every record rewritten with its answers and results shuffled and its
    maps and commitments in reverse order: the same state, byte for byte."""
    import gzip
    import random
    import shutil

    from fugal_subnet import v4_replay

    src = _store(flip_world, "root")
    shutil.copytree(src.directory, tmp_path / "shuffled")
    store = v4_replay.Store(tmp_path / "shuffled")
    rng = random.Random(7)
    for j in store.epochs():
        doc = store.load(j)
        rng.shuffle(doc["odds"]["answers"])
        doc["commitments"].reverse()
        for key in ("proofs", "head_hashes", "coldkey_of", "hotkey_of", "prices", "pick_spend"):
            doc[key] = dict(reversed(list(doc[key].items())))
        doc["odds"]["heads"] = dict(reversed(list(doc["odds"]["heads"].items())))
        for p in doc["proofs"].values():
            rng.shuffle(p["results"])
        with open(store._path(j), "wb") as f:
            f.write(gzip.compress(json.dumps(doc).encode()))
    a = _replay(flip_world, src, FLIP + 1)
    b = _replay(flip_world, store, FLIP + 1)
    assert _digest(a.window, a.seat, a.copy_state) == _digest(b.window, b.seat, b.copy_state)


def test_a_span_it_cannot_replay_replays_nothing(flip_world, tmp_path):
    import os
    import shutil

    from fugal_subnet import v4_replay

    root = _store(flip_world, "root")
    shutil.copytree(root.directory, tmp_path / "s")
    store = v4_replay.Store(tmp_path / "s")
    with pytest.raises(v4_replay.Missing) as e:          # longer than the bound
        _replay(flip_world, store, FLIP + 2, max_span=LAST - FLIP - 3)
    assert e.value.epochs == tuple(range(FLIP + 3, LAST + 1))
    os.remove(store._path(FLIP + 5))                       # a record missing
    with pytest.raises(v4_replay.Missing) as e:
        _replay(flip_world, store, FLIP + 2)
    assert e.value.epochs == (FLIP + 5,)
    shutil.copy(root._path(FLIP + 5), store._path(FLIP + 5))
    with open(store._path(FLIP + 4), "wb") as f:           # a record unreadable
        f.write(b"not gzip")
    with pytest.raises(v4_replay.Missing) as e:
        _replay(flip_world, store, FLIP + 2)
    assert e.value.epochs == (FLIP + 4,)
    shutil.copy(root._path(FLIP + 4), store._path(FLIP + 4))
    for sha in (tmp_path / "s" / "heads").iterdir():       # its heads not held
        sha.unlink()
    with pytest.raises(v4_replay.Missing) as e:
        _replay(flip_world, store, FLIP + 2)
    assert e.value.epochs == (FLIP + 3,)


def test_a_follower_that_cannot_replay_keeps_its_state_and_waits_for_a_later_root_epoch(
        flip_world, tmp_path, caplog):
    """The follower that missed five epochs, at b + 2, with the root's
    reveals through b and no records: nothing is replaced, it says so, and
    its gap moves to b + 1, so it re-anchors only on a root epoch from there."""
    import shutil

    from fugal_subnet import v4_replay
    from fugal_subnet.scoring import ScoringState
    from neurons import validator

    a, b = DOWN["five"]
    ref = tmp_path / "ref"
    for j in range(FIRST, b + 1):
        shutil.copytree(flip_world.dirs["root"] / "epochs" / epoch_id_for_block(j),
                        ref / epoch_id_for_block(j))
    hidden_of, benchmark_of, tables, t = _replay_inputs(flip_world)
    window, seat, copy_state = object(), object(), object()
    replay = v4_replay.Replayer(str(tmp_path / "none"), hidden_of=hidden_of,
                                benchmark_of=benchmark_of, max_span=48)

    def step(gap):
        return validator._v2_sync_step(
            epoch_index=b + 2, last_epoch_index=b + 1, first_v2_epoch=1, reference_dir=str(ref),
            scoring_state=ScoringState(records={}), window=window, seat=seat,
            copy_state=copy_state, window_degraded=False, gap_until=gap, anchor_at=None,
            prices=t.prices, bench_category=t.bench_category,
            epoch_dir=str(flip_world.dirs["five"] / "epochs"), window_epochs=48, tables=tables,
            replay=replay)

    with caplog.at_level("ERROR"):
        got = step(b)
    assert got.changed and not got.reanchored and got.audit is None
    assert got.state[1:4] == (window, seat, copy_state)
    assert got.state[5:] == (b + 1, None)
    assert "cannot replay" in caplog.text and epoch_id_for_block(b + 1) in caplog.text
    again = step(b + 1)
    assert not again.changed and again.state[1:4] == (window, seat, copy_state)
