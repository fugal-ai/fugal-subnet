"""Protocol v4's activation block, through the shipped loops (tests/v4_world.py).

Validators run `neurons/validator.main` from protocol v3 across the v4 block,
miners a v3 image's epochs before it and a v4 image's from it:

- the root and a follower cross the block running;
- a follower is stopped after the last v3 epoch and restarted exactly at
  the block, from its saved v3 state;
- a follower is stopped after the last v3 epoch and restarted one epoch
  after the block (it misses the first v4 epoch);
- a follower is stopped inside the last v3 epoch, after running it, and
  restarted a few blocks into the second v4 epoch (it misses the first);
- a follower is stopped after the first v4 epoch and restarted one epoch
  later (it misses the second);
- a follower with no state starts two epochs after the block and rebuilds
  from the root's published reveals;
- a control follower whose release does not schedule v4 runs the v3
  epochs: its v3 reveals are what the others publish before the block.

What carries and what resets at the block (docs/decisions/LIVE_MENU_V4.md 15,
SEAT_MINIMAL_V4.md 2.1 #24, docs/INVARIANTS.md "the v4 flip"): the window
(slices, pairs, odds, picks records) and the seat start empty (the `rules`
tag): the first v4 epoch publishes a protocol 4 seat state (`v2_state.seat`)
with no holder, no history and only the heads `seat_v4.observe` saw that
epoch; the menu is the pinned seed at the first v4 epoch and the menu step's
summaries start there; the copy rule's ownership state carries, and from the
block a retired head's record lapses after STANDING_EPOCHS_V4 epochs; nothing
the root writes that v3 code cannot parse (the four-field commitment, the v4
reveal fields, the v4 digest section) appears before the block, and no v4
reveal carries burn-in.

Every follower publishes the root's whole consensus state, and sets the
root's weights, from two epochs after the last epoch it did not run: the
one that missed the first v4 epoch included (verify-and-resync re-anchors
it on the root's state and replays its own epoch since,
fugal_subnet/v4_replay.py).
"""
from __future__ import annotations

import json

import pytest

from fugal_subnet import copy_rule, reveal_sync
from fugal_subnet import menu_state as ms
from fugal_subnet.benchmarks.slicer import epoch_id_for_block
from tests.test_menu_publish import signer  # noqa: F401 - the fixture
from tests.v4_world import (
    BPE,
    MINERS,
    ROOT,
    FlipWorld,
    MinerSpec,
    ValidatorSpec,
)

FLIP = 12                       # the first v4 epoch
FIRST, LAST = FLIP - 3, FLIP + 5
V4_FIELDS = ("menu", "menu_next", "menu_next_hash", "menu_summary", "menu_audit", "odds_v4",
             "labels_v4", "late_odds", "picks_v4")


@pytest.fixture(scope="module")
def world(tmp_path_factory, request):
    """One run shared by the tests below (it is the slow part)."""
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("flip")
    from fugal_subnet import menu_rules as mr
    from tests.test_menu_publish import Signer

    s = Signer(tmp, "release")
    mp.setattr(mr, "list_signers", lambda: s.signers_line().encode())
    specs = [
        ValidatorSpec("root", "root", "root", root=True),
        ValidatorSpec("crosser", "crosser", "crosser"),
        ValidatorSpec("at_block", "at_block", "at_block", stop=FLIP * BPE - 1),
        ValidatorSpec("at_block+", "at_block", "at_block", start=FLIP * BPE),
        ValidatorSpec("after", "after", "after", stop=FLIP * BPE - 1),
        ValidatorSpec("after+", "after", "after", start=(FLIP + 1) * BPE),
        ValidatorSpec("mid", "mid", "mid", stop=(FLIP - 1) * BPE + 6),
        ValidatorSpec("mid+", "mid", "mid", start=(FLIP + 1) * BPE + 4),
        ValidatorSpec("second", "second", "second", stop=(FLIP + 1) * BPE - 1),
        ValidatorSpec("second+", "second", "second", start=(FLIP + 2) * BPE),
        ValidatorSpec("fresh", "fresh", "fresh", start=(FLIP + 2) * BPE),
        ValidatorSpec("control", "control", "control", stop=FLIP * BPE - 1, v4=False),
    ]
    miners = [MinerSpec(hk, ck, seed=i, scored="auto" if i == 0 else "off")
              for i, (hk, ck) in enumerate(MINERS)]
    w = FlipWorld(tmp, mp.setattr, mp.setenv, s, flip=FLIP, first=FIRST, last=LAST,
                  validators=specs, miners=miners, cheater=MINERS[2][0])
    try:
        w.setup()
        w.run()
        yield w
    finally:
        mp.undo()


def test_every_validator_ran_its_epochs_without_error(world):
    assert world.clock.errors == {}, world.clock.errors
    assert world.errors == [], world.errors
    ran = {s: [x["epoch_id"] for x in world.logged(s)] for s in world.dirs}
    ids = [epoch_id_for_block(k) for k in range(FIRST, LAST + 1)]
    assert ran["root"] == ids and ran["crosser"] == ids
    assert ran["at_block"] == ids                        # stopped after FLIP - 1, back at FLIP
    assert ran["after"] == [i for i in ids if i != epoch_id_for_block(FLIP)]
    assert ran["mid"] == ran["after"]
    assert ran["second"] == [i for i in ids if i != epoch_id_for_block(FLIP + 1)]
    assert ran["fresh"] == ids[ids.index(epoch_id_for_block(FLIP + 2)):]
    assert ran["control"] == ids[:3]
    for s in world.dirs:
        assert all(x["n_heads_valid"] == len(MINERS) and x["n_heads_invalid"] == 0
                   for x in world.logged(s)), s


def test_no_node_reads_a_block_the_chain_has_not_produced(world):
    """Every pinned read is of a block that exists. On a real node a read of
    a future block answers nothing and the SDK caches the nothing for that
    block's hash, so the read of the next boundary fails on that connection
    for good: a follower filling its window from the root read the
    commitment of the epoch before at the next boundary, and stalled there
    (the local-chain v4 rehearsal, scripts/dress_rehearsal_v4.py)."""
    assert world.chain.future_reads == [], world.chain.future_reads[:5]


def test_the_protocol_and_the_rules_change_at_the_block_and_nowhere_else(world):
    for k in range(FIRST, LAST + 1):
        r = world.reveal("root", k)
        if k < FLIP:
            assert r["protocol_version"] == 3
            assert r["v2_state"]["rules"] == {"protocol": 3, "seat": "net_value"}
            assert not set(V4_FIELDS) & set(r), (k, sorted(set(V4_FIELDS) & set(r)))
            assert "v4" not in r["environment"]
        else:
            assert r["protocol_version"] == 4
            assert r["v2_state"]["rules"] == {"protocol": 4, "seat": "odds"}
            assert "v4" in r["environment"]



def test_the_window_and_the_seat_start_empty_at_the_block(world):
    from fugal_subnet.v2_epoch import load_window

    # Before the block the window holds the v3 epochs; from it, only v4 ones.
    v3 = world.reveal("root", FLIP - 1)["v2_state"]
    assert v3["rules"]["protocol"] == 3
    for s in ("root", "crosser", "at_block", "after", "second", "fresh"):
        saved = load_window(str(world.dirs[s] / "state.json") + ".v2window.npz", 48)
        assert sorted(saved.slices) and min(saved.slices) >= FLIP, (s, sorted(saved.slices))
        assert all(min(per) >= FLIP for per in saved.heads.values()), s
        assert all(min(per) >= FLIP for per in saved.odds.values()), s
        assert min(saved.pairs) >= FLIP, s
    seat = world.reveal("root", FLIP)["v2_state"]["seat"]
    assert seat["protocol"] == 4 and seat["epoch"] == FLIP
    assert seat["champion"] is None and not seat.get("history")
    # The heads the v4 seat holds at the block are the ones it saw there:
    # the v4 heads, first seen at the block; no v3 head is carried.
    v3_heads = set(world.reveal("root", FLIP - 1)["head_hashes"].values())
    v4_heads = set(world.reveal("root", FLIP)["head_hashes"].values())
    assert set(seat["heads"]) == v4_heads and not set(seat["heads"]) & v3_heads
    assert all(h["since"] == FLIP == h["seen"] and not h["predecessors"]
               for h in seat["heads"].values())
    for k in range(FLIP, LAST + 1):
        assert "burn_in" not in world.reveal("root", k)["v2_state"], k


def test_the_menu_is_the_seed_at_the_block_on_every_node(world):
    seed_at = ms.seed_state(FLIP)
    for s in ("root", "crosser", "at_block", "after"):
        held = world.held(s)
        assert held.held(FLIP - 1) is None, s
    for s in ("root", "crosser", "at_block"):
        assert ms.menu_hash(world.held(s).held(FLIP)) == ms.menu_hash(seed_at), s
    assert all(m.menus[FLIP] == ms.menu_hash(seed_at) for m in world.miners.values())
    for k in range(FLIP, LAST + 1):
        on = {ms.menu_hash(world.held(s).held(k)) for s in world.dirs
              if s != "control" and world.held(s).held(k) is not None}
        on |= {m.menus[k] for m in world.miners.values()}
        assert len(on) == 1, (k, on)


def test_nothing_v3_code_cannot_parse_is_written_before_the_block(world):
    rows = world.chain.commits[ROOT[0]]
    # A second field (the join pointer, chain_proposals) only from the block.
    assert all(b >= FLIP * BPE for b, d in rows if isinstance(d, tuple))
    rows = [(b, d[0] if isinstance(d, tuple) else d) for b, d in rows]
    parsed = [(b, reveal_sync.parse_commitment(d), d) for b, d in rows]
    four = [(b, c) for b, c, _d in parsed if c is not None and c[2] is not None]
    three = [(b, c) for b, c, _d in parsed if c is not None and c[2] is None]
    assert four and three
    assert min(b for b, _c in four) >= FLIP * BPE
    assert all(int(c[0].lstrip("e")) >= FLIP for _b, c in four)
    assert any(int(c[0].lstrip("e")) == FLIP - 1 for _b, c in three)
    for b, c, d in parsed:
        if c is not None and int(c[0].lstrip("e")) < FLIP:
            assert c[2] is None and d.count(":") == 2, d


def test_a_v3_epoch_is_what_a_release_without_v4_publishes(world):
    """The control (no v4 scheduled) and the crosser are both followers: their
    v3 reveals are the same bytes but for the environment block's record of
    the variable that schedules v4 off finney (on finney the compiled block
    is recorded only in a v4 epoch's section, so not even that)."""
    from fugal_subnet import fingerprint

    for k in range(FIRST, FLIP):
        a, b = world.reveal("crosser", k), world.reveal("control", k)
        env_a, env_b = a.pop("environment"), b.pop("environment")
        digest_a, digest_b = a.pop("consensus_digest"), b.pop("consensus_digest")
        assert a == b, k
        assert env_a["epoch"].pop("v4_activation_block") == FLIP * BPE
        assert "v4_activation_block" not in env_b["epoch"]
        assert {x: v for x, v in env_a.items() if x != "code"} == {
            x: v for x, v in env_b.items() if x != "code"}
        assert fingerprint.consensus_digest(env_a) == digest_b != digest_a



def test_the_copy_rules_ownership_carries_across_the_block(world):
    """Who owns a hash is not reset: a v3 head's owner still owns it at the
    block, with the block it was first committed at. From the block a
    retired head's record is kept for STANDING_EPOCHS_V4 epochs (the v4
    seat's G), not v1-v3's ghost window: the v3 heads, retired at the block
    when every miner moved to its v4 head, lapse G epochs after it."""
    from fugal_subnet import config

    g = config.STANDING_EPOCHS_V4
    before = world.reveal("root", FLIP - 1)["v2_state"][copy_rule.REVEAL_KEY]
    v3_heads = set(before["ownership"])
    assert v3_heads
    for k in range(FLIP, LAST + 1):
        after = world.reveal("root", k)["v2_state"][copy_rule.REVEAL_KEY]
        if k < FLIP + g:
            assert before["ownership"].items() <= after["ownership"].items(), k
            assert before["first_commit"].items() <= after["first_commit"].items(), k
            assert all(after["retired_at"][h] == FLIP for h in v3_heads), k
        else:
            assert not v3_heads & set(after["ownership"]), k
        v4_heads = set(world.reveal("root", k)["head_hashes"].values())
        assert v4_heads <= set(after["ownership"]), k



# Published fields a validator that ran every epoch and one that rebuilt or
# re-anchored are not expected to share: `sync` is per-validator. (A v4
# reveal carries no burn-in, so there is no burn-in difference to allow.)
_PER_VALIDATOR = {"sync"}
# The first epoch from which a validator that was not running publishes the
# root's state: two epochs after the last one it did not run (the root's
# reveal of an epoch is read from the start of the second epoch after it;
# the validator then re-anchors on it and replays its own epoch since,
# fugal_subnet/v4_replay.py). The one that missed the first v4 epoch is
# among them: under v4 an epoch's label pool depends on the window it is
# pooled on, and its first v4 epoch, pooled on a window without the root's
# first, is replayed on the root's, not kept.
_AGREES_FROM = {"after": FLIP + 2, "mid": FLIP + 2, "second": FLIP + 3, "fresh": FLIP + 3}


def test_every_validator_agrees_after_the_block(world):
    names = [s for s in world.dirs if s not in ("root", "control")]
    by = world.weights()
    for k in range(FIRST, LAST + 1):
        base = world.reveal("root", k)["v2_state"]
        for s in names:
            r = world.reveal(s, k)
            if r is None:
                continue
            if k >= FLIP:
                # The published set is the chain's (fugal_subnet/memo_sync.py),
                # not pooled: the memorisation fields agree from each
                # validator's first v4 epoch, before any re-anchor.
                assert r["v2_state"]["memo_digest"] == base["memo_digest"], (s, k)
                assert r["memo_v4"] == world.reveal("root", k)["memo_v4"], (s, k)
            if k < _AGREES_FROM.get(s, FIRST):
                continue
            st = r["v2_state"]
            assert {x: v for x, v in st.items() if x not in _PER_VALIDATOR} == {
                x: v for x, v in base.items() if x not in _PER_VALIDATOR}, (s, k)
            for key in ("labels_digest", "odds_digest", "seat_digest", "picks_digest",
                        "memo_digest"):
                assert st.get(key) == base.get(key), (s, k, key)
            hk = world.validator_keys[s][0]
            assert {u: x for u, x in by[k][hk].items() if x} == {
                u: x for u, x in by[k][ROOT[0]].items() if x}, (s, k)
    # Each re-anchored once, replaying its own epoch since, and says so.
    for s, k in _AGREES_FROM.items():
        assert world.reveal(s, k)["v2_sync"]["reanchored"] == {
            "snapshot": epoch_id_for_block(k - 2), "replayed": [epoch_id_for_block(k - 1)]}, s
    # ... and that epoch ran with the memorisation memo the root's ran with.
    from tests import v4_world
    threads = {"after": "after+", "mid": "mid+", "second": "second+", "fresh": "fresh"}
    assert v4_world.check_replayed_memos(world.v2_calls, set(threads.values())) == {
        threads[s]: [k - 1] for s, k in _AGREES_FROM.items()}
    # Validators that ran every epoch from before the block publish exactly
    # the root's state, burn-in (v3 epochs) included.
    for s in ("crosser", "at_block"):
        for k in range(FIRST, LAST + 1):
            st, base = world.reveal(s, k)["v2_state"], world.reveal("root", k)["v2_state"]
            assert {x: v for x, v in st.items() if x != "sync"} == {
                x: v for x, v in base.items() if x != "sync"}, (s, k)
    by = world.weights()
    for k in range(FIRST, LAST + 1):
        assert len({json.dumps(v, sort_keys=True) for v in by.get(k, {}).values()}) == 1, k


def test_the_menu_steps_summaries_start_at_the_block(world):
    for s in ("root", "crosser"):
        book = world.held(s)
        assert book.summaries(through=FLIP - 1, size=48) == []
        assert book.summaries(through=LAST, size=48)


def test_the_menu_field_and_its_source_agree_across_validators_from_the_block(world):
    """From the block every v4 reveal's `menu` field names its source; the
    validators that ran every epoch publish the root's consensus view (the
    field included), byte for byte, and their menu audit finds no
    difference; a miner reads the menu and its source from the root's bytes
    as the root's book records them."""
    import hashlib

    from fugal_subnet import menu_publish as mp
    from fugal_subnet import v2_sync

    for k in range(FLIP, LAST + 1):
        eid = epoch_id_for_block(k)
        root = world.reveal("root", k)
        raw = (world.dirs["root"] / "epochs" / eid / "reveal.json").read_bytes()
        got = mp.anchor_from_reveal(raw, eid, hashlib.sha256(raw).hexdigest())
        assert got is not None and got[:2] == (k, ms.menu_hash(world.held("root").held(k))), k
        assert got[2:] == world.held("root").source_of(k), k
        assert FLIP <= got[2] <= k, k                       # the seed at the block at the earliest
        for s in ("crosser", "at_block"):
            r = world.reveal(s, k)
            assert v2_sync.consensus_view(r) == v2_sync.consensus_view(root), (s, k)
            assert r[mp.REVEAL_AUDIT]["differences"] in ([], None), (s, k, r[mp.REVEAL_AUDIT])


def test_the_published_set_crosses_the_block_from_the_seed(world):
    """Every node and TD brings the published set from the release's seed
    across the block by folding the root's v3 reveals after it: the first
    v4 reveal names the view the seed and those reveals fold to (computed
    here from the root's public reveals alone), every v4 epoch's named view
    is that fold's, the root put each named view's snapshot in the store at
    its digest, every validator holds that fold beside its state, and the
    TDs drew the epoch's fresh questions (each proof verified on every node,
    test_every_validator_ran_its_epochs_without_error)."""
    import hashlib

    from fugal_subnet import memo_sync
    from tests import v4_world

    public = {k: world.reveal("root", k) for k in range(FIRST, LAST + 1)}
    assert all("memo_v4" not in public[k] for k in range(FIRST, FLIP))
    for k in range(FLIP, LAST + 1):
        named = public[k]["memo_v4"]
        expect = v4_world.chain_view(world.chain, world.published_seed, world.pool,
                                     public, k)
        assert named["through"] == k - 2
        assert named["published_digest"] == expect.view(k - 2).digest, k
        assert named["fresh"] and named["fresh_count"] == len(named["fresh"]), k
        if k < LAST:
            key = memo_sync.snapshot_object(v4_world.NETUID, named["published_digest"])
            raw = world.store.objects[key]
            assert hashlib.sha256(raw).hexdigest() == named["published_digest"]
    fresh = set(public[LAST]["memo_v4"]["fresh"])
    for m in world.miners.values():
        assert fresh <= {r.question_id for r in m.current["proof"].exploration_results}
    for s in ("root", "crosser", "at_block", "after", "second", "fresh"):
        v4_world.check_held(world.dirs[s] / ("state.json" + memo_sync.STATE_SUFFIX),
                            world.chain, world.published_seed, world.pool, public, LAST)
    # The control (no v4 scheduled) holds no published set.
    assert not (world.dirs["control"] / ("state.json" + memo_sync.STATE_SUFFIX)).exists()
