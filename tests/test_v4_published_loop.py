"""Protocol v4's published set through the shipped loops (tests/v4_world.py,
`FlipWorld`), with the root down three times: nothing waits on the root.

The set advances only by root reveals the chain names at a boundary
(fugal_subnet/memo_sync.py): while the root is down nothing is folded, every
TD still proves on the set it holds and every follower still checks every
proof. Over one run (F the first v4 epoch):

- the root is down for one epoch (DOWN_1), three (DOWN_3) and eight (DOWN_8),
  and on each return publishes and commits its backlog;
- a follower (`restart`) is stopped during the long outage and restarted
  from its saved state; another (`deleted`) is restarted during it with its
  published set deleted; a miner's TD restarts during it with nothing kept;
- a stray older commitment of the root's lands on a boundary (`STRAY`): it
  names a reveal older than one folded, and is ignored;
- the root's commitment fails once after its upload succeeded (`NO_COMMIT`):
  that reveal is never named at a boundary, and never folded;
- `retry` runs one epoch again with `--once`.

Every node's set is the one the chain gives (computed here from the fake
chain alone, `v4_world.chain_view`), every reveal of an epoch carries the
same `memo_v4` on every node that ran it, and no node raises an ALARM.
Followers run the epochs the root misses on their own state; once the root
is back they re-anchor on it (verify-and-resync) and replay their own epoch
since, with the memo the root ran it with, and agree with it again.
"""
from __future__ import annotations

import pytest

from fugal_subnet.benchmarks.slicer import epoch_id_for_block
from tests.test_menu_publish import signer  # noqa: F401 - the fixture
from tests.v4_world import BPE, MINERS, ROOT, FlipWorld, MinerSpec, ValidatorSpec

FLIP = 8
FIRST, LAST = FLIP - 2, FLIP + 23
DOWN_1 = (FLIP + 3,)
DOWN_3 = (FLIP + 6, FLIP + 7, FLIP + 8)
DOWN_8 = tuple(range(FLIP + 11, FLIP + 19))
STRAY = FLIP + 20                  # a stray older commitment at this boundary
NO_COMMIT = FLIP + 21              # the root's commitment fails in this epoch
RETRY = FLIP + 4
TD_RESTART = FLIP + 14
GONE = FLIP + 15                   # `deleted` restarts here with no published set


def _specs():
    root = [ValidatorSpec("root", "root", "root", root=True, stop=DOWN_1[0] * BPE - 1),
            ValidatorSpec("root+1", "root", "root", root=True, start=(DOWN_1[-1] + 1) * BPE,
                          stop=DOWN_3[0] * BPE - 1),
            ValidatorSpec("root+3", "root", "root", root=True, start=(DOWN_3[-1] + 1) * BPE,
                          stop=DOWN_8[0] * BPE - 1),
            ValidatorSpec("root+8", "root", "root", root=True, start=(DOWN_8[-1] + 1) * BPE)]
    return root + [
        ValidatorSpec("steady", "steady", "steady"),
        ValidatorSpec("restart", "restart", "restart", stop=(FLIP + 13) * BPE - 1),
        ValidatorSpec("restart+", "restart", "restart", start=(FLIP + 16) * BPE),
        ValidatorSpec("deleted", "deleted", "deleted", stop=GONE * BPE - 1),
        ValidatorSpec("deleted+", "deleted", "deleted", start=GONE * BPE),
        ValidatorSpec("retry", "retry", "retry", stop=RETRY * BPE + 8),
        ValidatorSpec("retry+once", "retry", "retry", start=RETRY * BPE + 9, once=True),
        ValidatorSpec("retry+", "retry", "retry", start=(RETRY + 1) * BPE),
    ]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("published")
    from fugal_subnet import memo_sync, reveal_publisher
    from fugal_subnet import menu_rules as mr
    from tests.test_menu_publish import Signer
    from tests.v4_world import Chain

    s = Signer(tmp, "release")
    mp.setattr(mr, "list_signers", lambda: s.signers_line().encode())
    seen = {"findings": [], "snapshots": []}
    real_check = memo_sync.check_named

    def check(published, reveal):
        got = real_check(published, reveal)
        if got is not None:
            seen["findings"].append(got)
        return got

    mp.setattr(memo_sync, "check_named", check)
    real_commit = Chain.commit

    def commit(chain, hotkey, data):
        # The root's upload went through, its commitment does not land.
        if hotkey == ROOT[0] and chain.block // BPE == NO_COMMIT:
            raise ConnectionError("the commitment extrinsic was not included")
        return real_commit(chain, hotkey, data)

    mp.setattr(Chain, "commit", commit)

    def on_epoch(w, epoch):
        if epoch == GONE:
            (w.dirs["deleted"] / ("state.json" + memo_sync.STATE_SUFFIX)).unlink()
        if epoch == TD_RESTART:
            spec = w.miner_specs[1]
            w.miners[spec.hotkey].session = w.session(spec)       # nothing kept
        if epoch == STRAY:
            # An older commitment of the root's, made at this boundary block.
            r = w.reveal("root", FLIP + 2)
            import hashlib
            import json
            raw = (w.dirs["root"] / "epochs" / epoch_id_for_block(FLIP + 2)
                   / "reveal.json").read_bytes()
            assert r is not None and json.loads(raw) == r
            w.chain.commits[ROOT[0]].append((epoch * BPE, reveal_publisher.commitment_for(
                epoch_id_for_block(FLIP + 2), hashlib.sha256(raw).hexdigest())))

    miners = [MinerSpec(hk, ck, seed=i, scored="auto" if i == 0 else "off")
              for i, (hk, ck) in enumerate(MINERS)]
    w = FlipWorld(tmp, mp.setattr, mp.setenv, s, flip=FLIP, first=FIRST, last=LAST,
                  validators=_specs(), miners=miners, on_epoch=on_epoch)
    w.seen = seen
    try:
        w.setup()
        w.run()
        yield w
    finally:
        mp.undo()


def _public(world):
    return {k: world.reveal("root", k) for k in range(FIRST, LAST + 1)
            if world.reveal("root", k) is not None}


def test_every_td_proves_and_every_follower_checks_every_epoch_through_the_outages(world):
    assert world.clock.errors == {}, world.clock.errors
    assert world.errors == [], world.errors
    assert world.chain.future_reads == [], world.chain.future_reads[:5]
    assert all(m.no_proof == {} for m in world.miners.values()), {
        m.hotkey: m.no_proof for m in world.miners.values()}
    ids = [epoch_id_for_block(k) for k in range(FIRST, LAST + 1)]
    ran = {s: [x["epoch_id"] for x in world.logged(s)] for s in world.dirs}
    down = {epoch_id_for_block(k) for k in DOWN_1 + DOWN_3 + DOWN_8}
    assert ran["root"] == [i for i in ids if i not in down]
    assert ran["steady"] == ids and ran["deleted"] == ids
    assert ran["restart"] == [i for i in ids if not FLIP + 13 <= int(i[1:]) < FLIP + 16]
    assert ran["retry"].count(epoch_id_for_block(RETRY)) == 2
    for s in world.dirs:
        assert all(x["n_heads_valid"] == len(MINERS) and x["n_heads_invalid"] == 0
                   and x["n_heads_unverifiable"] == 0 for x in world.logged(s)), s
    assert world.seen["findings"] == []


def test_every_node_holds_the_set_the_chain_gives_and_names_it_alike(world):
    from fugal_subnet import memo_sync
    from tests import v4_world

    public = _public(world)
    for k in range(FLIP, LAST + 1):
        expect = v4_world.chain_view(world.chain, world.published_seed, world.pool, public, k)
        on = {s: world.reveal(s, k) for s in world.dirs if world.reveal(s, k) is not None}
        assert on, k
        for s, r in on.items():
            assert r["memo_v4"]["published_digest"] == expect.view(k - 2).digest, (s, k)
        assert len({str(r["memo_v4"]) for r in on.values()}) == 1, k
    for s in world.dirs:
        v4_world.check_held(world.dirs[s] / ("state.json" + memo_sync.STATE_SUFFIX),
                            world.chain, world.published_seed, world.pool, public, LAST)


def test_late_reveals_are_folded_at_the_boundary_that_first_names_them(world):
    """The root's backlog after each outage is folded when the chain first
    names it; while the root is down the set does not move; a reveal never
    named at a boundary (a failed commitment, or behind a stray older
    commitment) is never folded."""
    from fugal_subnet import memo_v4
    from tests import v4_world

    public = _public(world)

    def view(k):
        return v4_world.chain_view(world.chain, world.published_seed, world.pool, public, k)

    # The set does not move through the long outage (after its last named
    # reveal is folded at the boundary following the root's last epoch).
    still = [view(k).view(k - 2).first for k in range(DOWN_8[0] + 2, DOWN_8[-1] + 2)]
    assert all(dict(x) == dict(still[0]) for x in still)
    # The reveal published during NO_COMMIT but never committed is named at
    # no boundary, so no question is first published at its epoch.
    from fugal_subnet import reveal_sync

    named = {memo_v4.epoch_index(got[0]) for b in range(FIRST, LAST + 1)
             if (got := reveal_sync.parse_commitment(
                 world.chain.at(b * BPE).get(ROOT[0], (0, ""))[1])) is not None}
    assert NO_COMMIT - 1 in public and NO_COMMIT - 1 not in named
    assert FLIP + 2 in named                       # the stray, older, named and ignored
    final = view(LAST)
    assert not any(e == NO_COMMIT - 1 for e in final.first.values())


def test_a_once_retry_leaves_the_published_set_as_it_was(world):
    public = _public(world)
    assert world.reveal("retry", RETRY)["memo_v4"] == public[RETRY]["memo_v4"]


def test_followers_re_anchor_after_each_outage_and_replay_with_the_roots_memo(world):
    """No follower waits on the root: each runs the epochs the root misses.
    Two epochs after each return the followers re-anchor on the root's
    state (in an epoch the root is down again, after DOWN_3) and replay the
    epoch since, which ran with the root's memo, the followers restarted
    during the long outage (with their set, and without it) included. From
    two epochs after the root's last return every follower publishes the
    root's whole consensus state and sets its weights."""
    from tests import v4_world

    back = DOWN_8[-1] + 1
    by = world.weights()
    followers = ("steady", "restart", "deleted", "retry")
    # A follower whose published state differs from the root's re-anchors;
    # with no challenge in the run, the seat's evidence reads no window
    # comparison, so a follower whose state matches has nothing to re-anchor
    for s in followers:
        sync = world.reveal(s, back + 2)["v2_sync"]
        if "reanchored" in sync:
            assert sync["reanchored"] == {
                "snapshot": epoch_id_for_block(back),
                "replayed": [epoch_id_for_block(back + 1)]}, s
        for k in range(back + 2, LAST + 1):
            st, base = world.reveal(s, k)["v2_state"], world.reveal("root", k)["v2_state"]
            assert {x: v for x, v in st.items() if x != "sync"} == {
                x: v for x, v in base.items() if x != "sync"}, (s, k)
            hk = world.validator_keys[s][0]
            assert {u: x for u, x in by[k][hk].items() if x} == {
                u: x for u, x in by[k][ROOT[0]].items() if x}, (s, k)
    before = DOWN_3[-1] + 2                 # re-anchored at DOWN_3[-1] + 3, the root down
    sync = world.reveal("steady", before + 1)["v2_sync"]
    assert "reanchored" not in sync or sync["reanchored"]["replayed"] == [
        epoch_id_for_block(before)]
    roots = {"root", "root+1", "root+3", "root+8"}
    # Every epoch a follower replayed ran with the root's memo; which
    # followers replay is which ones' published state differed (above).
    got = v4_world.check_replayed_memos(
        world.v2_calls, {"steady", "restart", "restart+", "deleted", "deleted+", "retry+"},
        root=roots)
    # The root's first reveal after an outage is a re-anchor point
    # (neurons/validator._root_came_back): after DOWN_1 the followers then
    # running re-anchor on its reveal of DOWN_1[-1] + 1 and replay the epoch after.
    first = DOWN_1[-1] + 2
    could = {"steady": [first, before, back + 1], "restart": [first, before],
             "restart+": [back + 1], "deleted": [first, before], "deleted+": [back + 1],
             "retry+": [first, before, back + 1]}
    assert all(set(v) <= set(could[s]) for s, v in got.items()), got
