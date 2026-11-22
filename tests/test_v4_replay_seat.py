"""Protocol v4's re-anchor replay (fugal_subnet/v4_replay.py) on the seat
loops' records (tests/v4_world.py SeatLoop: both validators run
neurons/validator.main, which keeps each v4 epoch's inputs beside its state).

1. The take scenario: the root's epochs after an anchor, replayed from its
   records on its own state at the anchor (rebuilt from its reveals), end in
   the state the root holds, byte for byte. The span carries the seat's fill
   and an odds-only miner's take, and menus carried from an earlier epoch
   (each reveal's `menu.source`).
2. The memorisation scenario: the epoch's memo is not in the record. The
   replay takes it from a function of the epoch and its recorded nonce
   (`v4_replay.Replayer.memo_for`, given through
   `neurons.validator._v4_replayer(..., memo_for=_v4_memo_for(...))`): with
   the validator's function on the published set it keeps, the replay
   reaches the root's seat (its holder refused, the seat open); without it,
   it does not.
3. The memorisation scenario with the follower down for five epochs while
   the holder is first refused: it agrees with the root from two epochs
   after its return, and the epoch it replays ran with the root's memo.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from fugal_subnet.benchmarks.slicer import epoch_id_for_block
from tests import v4_world as w
from tests.test_menu_publish import LIST_BYTES, signer  # noqa: F401 - the fixture
from tests.test_v4_resync_replay import _digest


@pytest.fixture
def _signature(signer):  # noqa: F811
    return signer.sign(LIST_BYTES)


def _run(scenario, tmp_path, monkeypatch, signature):
    """Run the loops, then give run_v2_epoch back: the loops wrap it (the
    follower's inputs reversed, both nodes' memo), and the replay must run
    the module's own."""
    from fugal_subnet import v2_epoch

    real = v2_epoch.run_v2_epoch
    loop = scenario(tmp_path, monkeypatch, signature)
    monkeypatch.setattr(v2_epoch, "run_v2_epoch", real)
    return loop


def _tables(loop):
    from fugal_subnet import menu_publish as mp
    from neurons import validator

    held = mp.MenuBook(str(loop.dirs["root"] / "state.json") + ".menu").held(loop.last)
    t = validator._v4_tables(mp.MenuRead(loop.last, held, "carried"))
    return t, {4: (t.prices, t.bench_category, t.pick_spend)}


def _root_at(loop, k):
    """The root's state at k, rebuilt from its reveals (what a re-anchor on
    it adopts)."""
    from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals

    t, tables = _tables(loop)
    paths = [str(loop.dirs["root"] / "epochs" / epoch_id_for_block(j) / "reveal.json")
             for j in range(w.SEAT_FIRST, k + 1)]
    return rebuild_v2_from_reveals(paths, window_size=48, prices=t.prices,
                                   bench_category=t.bench_category, tables=tables)


def _root_saved(loop):
    from fugal_subnet import copy_rule
    from fugal_subnet.v2_epoch import load_window, seat_from_dict

    path = loop.dirs["root"] / "state.json"
    raw = json.loads(path.read_text())["v2"]
    return (load_window(str(path) + ".v2window.npz", 48), seat_from_dict(raw["seat"]),
            copy_rule.CopyState.from_reveal(raw[copy_rule.REVEAL_KEY]))


def _replayer(loop, monkeypatch, memo_for=None):
    """The validator's own Replayer over the root's records, as main builds it."""
    import os

    from fugal_subnet.embeddings_file import Embeddings
    from neurons import validator

    monkeypatch.setattr(validator, "STATE_PATH", str(loop.dirs["root"] / "state.json"))
    gold = {q["question_id"]: {"benchmark": q["benchmark"]} for q in loop.pool}
    return validator._v4_replayer(gold, Embeddings.load(os.environ["FUGAL_EMBEDDINGS_FILE"], None),
                                  48, memo_for=memo_for)


def test_a_replay_over_a_take_and_carried_menus_reaches_the_roots_state(
        tmp_path, monkeypatch, _signature):
    from fugal_subnet import menu_publish as mp

    loop = _run(w.seat_take_scenario, tmp_path, monkeypatch, _signature)
    k = w.SEAT_FIRST + 1
    span = range(k + 1, loop.last + 1)
    events = {j: loop.reveal(j)["champion"]["event"] for j in span}
    assert {"fill", "take"} <= set(events.values()), events
    sources = {}
    for j in span:
        raw = (loop.dirs["root"] / "epochs" / epoch_id_for_block(j) / "reveal.json").read_bytes()
        got = mp.anchor_from_reveal(raw, epoch_id_for_block(j), hashlib.sha256(raw).hexdigest())
        assert got is not None and got[0] == j, j
        sources[j] = got[2]                                  # the epoch of `menu.source`
    assert any(src < j for j, src in sources.items()), sources       # carried menus

    memo_for = w.held_memo_for(loop.dirs["root"], loop.pool)
    got = _replayer(loop, monkeypatch, memo_for)(_root_at(loop, k), k, loop.last + 1)
    assert got.epochs == tuple(span)
    assert _digest(got.window, got.seat, got.copy_state) == _digest(*_root_saved(loop))


def _refused(loop) -> list[int]:
    a = loop.hash_of(w.SEAT_A)
    return [k for k in loop.epochs
            if loop.reveal(k)["memo_v4"]["evidence"].get(a, {}).get("refused")]


def test_a_replayed_epoch_takes_its_memo_from_the_published_set_the_validator_keeps(
        tmp_path, monkeypatch, _signature):
    """The memo of each replayed epoch comes from the validator's replay
    memo function (`_v4_memo_for`, given to `_v4_replayer`): on the
    published set the validator keeps (read as it is), the epoch's plan and
    its recorded nonce. Every replayed epoch then publishes the root's
    memorisation fields and decides the root's seat (the holder refused, the
    seat open, nobody paid). Without it, the same epochs keep paying the
    holder. (The seat state is the same either way here: a refusal opens
    the seat without changing its holder.)"""
    from fugal_subnet import seat_v4, v2_epoch

    loop = _run(w.seat_memo_scenario, tmp_path, monkeypatch, _signature)
    first = min(_refused(loop))
    memo_for = w.held_memo_for(loop.dirs["root"], loop.pool)
    k = first - 2
    asked, decided = [], {}

    def memo_of(epoch, nonce):
        asked.append(epoch)
        return memo_for(epoch, nonce)

    real = v2_epoch.run_v2_epoch

    def spy(**kw):
        r = real(**kw)
        decided[int(kw["epoch_index"])] = (kw["memo"], json.loads(json.dumps(r.reveal)))
        return r

    monkeypatch.setattr(v2_epoch, "run_v2_epoch", spy)
    span = list(range(k + 1, loop.last + 1))
    got = _replayer(loop, monkeypatch, memo_of)(_root_at(loop, k), k, loop.last + 1)
    assert asked == span and list(decided) == span
    assert _digest(got.window, got.seat, got.copy_state) == _digest(*_root_saved(loop))
    keys = ("champion", seat_v4.REVEAL_KEY, "memo_v4", "v2_state")
    for j in span:
        memo, reveal = decided[j]
        root = loop.reveal(j)
        assert memo is not None, j
        assert {x: reveal.get(x) for x in keys} == {x: root.get(x) for x in keys}, j
    assert root[seat_v4.REVEAL_KEY]["holder"]["refused"] is True and root[seat_v4.REVEAL_KEY]["open"]

    # Without the function no replayed epoch is handed a memo: the holder is
    # not refused, and the seat it decides is not the root's.
    decided.clear()
    _replayer(loop, monkeypatch)(_root_at(loop, k), k, loop.last + 1)
    for j in range(first, loop.last + 1):
        memo, reveal = decided[j]
        assert memo is None and "memo_v4" not in reveal, j
        assert not reveal[seat_v4.REVEAL_KEY]["open"], j
        assert reveal[seat_v4.REVEAL_KEY] != loop.reveal(j)[seat_v4.REVEAL_KEY], j


# The follower is down for five epochs around the first epoch the root's
# memorisation check refuses the holder.
MEMO_DOWN = (w.SEAT_FIRST + 5, w.SEAT_FIRST + 9)


def test_a_follower_down_five_epochs_while_the_holder_is_refused_agrees_from_two_after(
        tmp_path, monkeypatch, _signature):
    """The follower stops before MEMO_DOWN's first epoch and restarts after
    its last (b); the root's check refuses the seat holder inside the
    outage. At b + 2 the follower re-anchors on the root's state at b and
    replays b + 1 with the memo the root ran b + 1 with (the published set
    as of b + 1, from the chain; its plan; its nonce): from b + 2 it
    publishes the root's label, odds, seat, picks and memo digests, its
    memorisation fields and seat evidence, and sets the root's weights."""
    from tests.test_v4_resync_replay import _consensus, _nonzero

    a, b = MEMO_DOWN
    loop = w.seat_memo_scenario(tmp_path, monkeypatch, _signature, follower_down=MEMO_DOWN,
                                last=b + 4)
    refused = _refused(loop)
    assert a <= min(refused) <= b and refused == list(range(min(refused), loop.last + 1)), refused
    for k in range(a, b + 1):
        assert not (loop.dirs["follower"] / "epochs" / epoch_id_for_block(k)).exists(), k
    for k in range(b + 2, loop.last + 1):
        root, follower = loop.reveal(k), loop.reveal(k, "follower")
        st, base = follower["v2_state"], root["v2_state"]
        for key in ("labels_digest", "odds_digest", "seat_digest", "picks_digest", "memo_digest"):
            assert st[key] == base[key], (k, key)
        assert _consensus(st) == _consensus(base), k
        assert follower["memo_v4"] == root["memo_v4"] and follower["seat_v4"] == root["seat_v4"], k
        assert _nonzero(loop.weights[k][w.FOLLOWER[0]]) == _nonzero(loop.weights[k][w.ROOT[0]]), k
    assert loop.reveal(b + 2, "follower")["v2_sync"]["reanchored"] == {
        "snapshot": epoch_id_for_block(b), "replayed": [epoch_id_for_block(b + 1)]}
    assert w.check_replayed_memos(loop.v2_calls, {"follower+"}) == {"follower+": [b + 1]}


def test_a_re_anchor_replays_at_most_the_reveal_hold():
    """The span bound the validator gives its Replayer: three epochs (what
    fits beside the live epoch at the measured worst-case cost), or the
    reveal hold (the steady state's span) if longer; main passes it."""
    import ast
    import inspect

    from neurons import validator

    assert [validator._v4_replay_span(d) for d in (0, 1, 2, 3, 5)] == [3, 3, 3, 3, 5]
    tree = ast.parse(inspect.getsource(validator.main.callback))
    (call,) = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and ast.unparse(n.func) == "_v4_replayer"]
    assert ast.unparse(call.args[2]) == "_v4_replay_span(reveal_delay)"
