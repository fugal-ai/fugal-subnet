"""The size of a protocol v4 reveal at the subnet's 256 UIDs, and the miner's
holder read of it (SEAT_MINIMAL_V4.md 3.4; td_v4.seat_holder_from_reveal).

A miner running the scored questions on "auto" learns it holds the seat by
reading the root's newest reveal; a reveal larger than the read's limit
would leave it unread, and the holder would miss epochs it must answer. The
reveal is written by the real writer (commit_reveal.reveal_epoch, indented
JSON) with every field at its largest shape: 256 verified heads, each
hotkey's 112 private exploration calls pooled as 28,672 labels with a voter
each, every head's odds on them, the late odds one reveal may carry
(config.LATE_ODDS_SLOTS_V4 head-epochs), 64 menu entries and 8 provisional,
every head's challenge of 1,000 questions (FAST_CROWN_V4.md 3: int32 items,
the list as ids, per-entry means and summed cells, the holder's slice
picks, no per-UID failed lists), the seat state with every
hotkey's head standing for three epochs, the seat's evidence with every
qualifier tested against every earlier contender (rule C), and the
memorisation fields (memo_v4) with evidence for every head the seat keeps
and the fresh list at its largest F (the whole pinned pool unpublished). The
fields of the
earlier packages and of v1-v3 are built by their own encoders; questions and
the root's menu fields, whose sizes this package does not set, are given
generous allowances.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from fugal_subnet import (
    commit_reveal,
    config,
    explore_pool,
    labels_v4,
    memo_v4,
    picks_v4,
    reveal_sync,
    seat_v4,
)
from fugal_subnet.champion import EpochPairs
from fugal_subnet.picks_v4 import PicksEpoch, PicksEvidence
from fugal_subnet.seat_bound import Bound
from fugal_subnet.shared_grades import cells_for_reveal
from fugal_subnet.tee import td_v4
from fugal_subnet.v2_epoch import odds_b64, v2_state_for_reveal

UIDS = 256
PAIRS = UIDS * config.EXPLORE_PRIVATE_SIZE_V4         # 28,672 labels
SLICE = config.SLICE_SIZE_V4
ENTRIES = 64 + 8                                      # the menu's cap and 8 provisional entries
WINDOW = config.SEAT_WINDOW_EPOCHS_V4
EPOCH = 1000


def _hex(tag) -> str:
    return hashlib.sha256(str(tag).encode()).hexdigest()


def _address(i, kind) -> str:
    return f"5{kind}{i:046d}"[:48]


def _entry(i) -> str:
    return f"lab-{i:02d}/a-model-name-of-some-length-{i:03d}@reasoning-high"


def _qid(i) -> str:
    return f"livecodebench_v6-{i:08d}"


def _evidence_template():
    """One head's published seat evidence and a rule C row, from the real
    seat step (tests/seat_v4_world.py)."""
    from tests.seat_v4_world import World

    w = World()
    w.add("a", skill=0.3, block=3)
    w.add("b", skill=0.6, block=4)
    for e in range(1, 6):
        d = w.epoch(e, served=["a", "b"], answered=["a", "b"])
    doc = d.evidence["heads"][w.heads["b"].hash]
    assert doc["earlier"] and doc["picks"] and doc["vs_bar"]
    return d.evidence, doc


@pytest.fixture(scope="module")
def worst_reveal(tmp_path_factory):
    rng = np.random.default_rng(0)
    heads = [_hex(f"head-{i}") for i in range(UIDS)]
    coldkeys = [_address(i, "C") for i in range(UIDS)]
    hotkeys = [_address(i, "H") for i in range(UIDS)]
    entries = [_entry(i) for i in range(ENTRIES)]
    qids = [_qid(i) for i in range(PAIRS)]
    slice_q = [_qid(10_000_000 + i) for i in range(SLICE)]

    # --- labels and odds (package S2) ---
    pooled = [explore_pool.Pair(q, entries[i % ENTRIES], int(rng.integers(0, 10_001)), 1,
                                int(rng.integers(0, 20_000)), int(rng.integers(0, 40_000)),
                                int(rng.integers(1, 5_000_000)), int(rng.integers(0, 600_000)))
              for i, q in enumerate(qids)]
    bench = {q: "livecodebench_v6" for q in qids}
    pairs = labels_v4.label_pairs(
        EpochPairs(question_ids=tuple(qids), models=tuple(p.model for p in pooled),
                   benchmarks=("livecodebench_v6",) * PAIRS, categories=("hard",) * PAIRS,
                   y=np.zeros(PAIRS, np.int32), prompt_tokens=np.zeros(PAIRS, np.int32),
                   completion_tokens=np.zeros(PAIRS, np.int32)),
        [(coldkeys[i % UIDS],) for i in range(PAIRS)],
        rng.integers(-1, 10_001, PAIRS).astype(np.int16))
    odds = {h: rng.integers(0, 10_001, PAIRS).astype(np.int16) for h in heads}
    slots = config.LATE_ODDS_SLOTS_V4
    late = {}
    for k in range(slots):
        late.setdefault(heads[k % UIDS], {})[EPOCH - 1 - k // UIDS] = odds[heads[k % UIDS]]
    standing = {h: {"uid": i, "labels": PAIRS * WINDOW, "gain_points": -12.345678901234567,
                    "odds_epochs": list(range(EPOCH - WINDOW + 1, EPOCH + 1))}
                for i, h in enumerate(heads)}

    # --- the challenge (FAST_CROWN_V4.md 2, 3): every head a challenger of
    # 1,000 questions, each item at the int32 extremes ---
    size = max(config.CHALLENGE_SIZES_V4)
    cats = ("instruction", "easy_math", "knowledge", "everyday", "math", "code", "hard")
    listed = slice_q + [_qid(30_000_000 + i) for i in range(config.CHALLENGE_LIST_SIZE_V4 - SLICE)]
    rec = PicksEpoch(EPOCH, tuple(cats[i % 7] for i in range(len(listed))),
                     {h: tuple(int(x) for x in rng.integers(-2**31, 2**31, size)) for h in heads},
                     False, _hex("head-0"))
    ev = PicksEvidence(heads[0], heads[1], tuple(range(EPOCH - WINDOW + 1, EPOCH + 1)),
                       size * WINDOW, WINDOW, size * WINDOW,
                       Bound(size * WINDOW, -10**12, None, -10**9, True), True)
    picks_block = picks_v4.reveal_block(
        rec, {h: picks_v4.evidence_to_reveal(ev, answered_now=True) for h in heads},
        {hk: (112, 112) for hk in hotkeys}, questions=listed,
        means={e: (10**6, -10**15) for e in entries},
        cells={f"{e} {c}": [10**6, 10**6, 10**6] for e in entries for c in cats},
        holder_picks=picks_v4.holder_picks_doc(
            _hex("head-0"), [entries[i % ENTRIES] for i in range(len(listed))], entries),
        challenges_refused={},
        routing_vs_r={"holder": [10**6, -10**15, 10**21], "r": [10**6, -10**15, 10**21],
                      "lower": -10**9})

    # --- the seat (this package) ---
    menu_rows = {e: _hex(e) for e in entries[:64]}
    kept = []
    for i in range(UIDS):
        for age in range(seat_v4.STANDING_EPOCHS):        # a new head every epoch per hotkey
            h = heads[i] if age == 0 else _hex(f"old-{i}-{age}")
            kept.append(seat_v4.HeadState(
                head=h, owner=coldkeys[i], hotkey=hotkeys[i], block=10**9 - age,
                since=EPOCH - age, seen=EPOCH - age, origin=_hex(f"origin-{i}"),
                origin_block=10**8, origin_owner=coldkeys[i],
                rows_since={e: EPOCH - WINDOW for e in entries[:64]},
                predecessors=tuple(_hex(f"pred-{i}-{k}") for k in range(seat_v4.MAX_PREDECESSORS)),
                predecessor_since=(10**12 - 1,) * seat_v4.MAX_PREDECESSORS,
                rows=menu_rows if age == 0 else {}))
    state = seat_v4.SeatStateV4(champion=heads[0], champion_since=EPOCH - 5,
                                history=tuple(heads[1:5]), heads=tuple(kept), epoch=EPOCH,
                                holders=((EPOCH, heads[0]), (EPOCH - 1, heads[0])))
    evidence, doc = _evidence_template()
    order = sorted(heads)
    test = Bound(PAIRS * WINDOW, -10**15, None, -10**12, True)

    def tested(h):
        if h not in order:
            return []
        return ([seat_v4._earlier_item(seat_v4.R, test)]
                + [seat_v4._earlier_item(str(i), test) for i in range(order.index(h))])

    evidence = dict(evidence, order=order, qualifiers=order,
                    winner=order[-1], paid={h: True for h in heads[:3]},
                    heads={hs.head: dict(doc, uid=UIDS, precedence=[10**8, _hex("p")],
                                         origin=hs.origin, earlier=tested(hs.head))
                           for hs in kept})
    # --- the memorisation check (package S5, read by the seat) ---
    with open(os.path.join(os.path.dirname(memo_v4.__file__), "benchmarks",
                           "pool_manifest_v3.json"), encoding="utf-8") as f:
        pool_size = json.load(f)["n_questions"]
    fresh = memo_v4.fresh_count(pool_size)
    big = 10**15
    memo_block = {"through": EPOCH - 2, "published_digest": _hex("published"),
                  "reserve": pool_size, "fresh_count": fresh,
                  "fresh": sorted(_qid(20_000_000 + i) for i in range(fresh)),
                  "evidence": {hs.head: {"seen": PAIRS * WINDOW, "unseen": PAIRS * WINDOW,
                                         "benchmarks": 64, "gap_units": -big,
                                         "lower_units": -big, "refused": False}
                               for hs in kept}}
    from fugal_subnet import copy_rule
    copy_state = copy_rule.CopyState(epoch=EPOCH, heads=tuple(
        copy_rule.HeadRecord(hs.head, hs.block, hs.owner, hs.seen == EPOCH,
                             None if hs.seen == EPOCH else hs.seen + 1, hs.hotkey)
        for hs in kept))
    extra = {copy_rule.REVEAL_KEY: copy_state.to_reveal(), "rules": {"protocol": 4, "seat": "odds"},
             picks_v4.DIGEST_KEY: _hex(1), "odds_digest": _hex(2), "labels_digest": _hex(3),
             seat_v4.DIGEST_KEY: _hex(4), "memo_digest": _hex(5)}
    v2_fields = {
        "shared_grades": cells_for_reveal({(q, e): 2 / 3 for q in slice_q for e in entries},
                                          {(q, e): 12345.5 for q in slice_q for e in entries}),
        "copy_verdicts": {str(u): "owner" for u in range(UIDS)},
        "v2_seniority": {"floor_block": 10**8, "salt": "0x" + _hex("salt")},
        "champion": {"event": "take", "champion": heads[0], "bar": heads[1],
                     "detail": [[heads[0], "took the seat from x, the latest of 256 qualifier(s)"]]},
        "v2_state": v2_state_for_reveal(seat=state, owners={r.head_hash: r.first_hotkey
                                                            for r in copy_state.heads},
                                        serving={h: i for i, h in enumerate(heads)},
                                        burn_in=None, extra=extra),
        "labels_v4": labels_v4.labels_for_reveal(pairs),
        "late_odds": labels_v4.late_for_reveal(late),
        "odds_v4": standing,
        "memo_v4": memo_block,
        # chain proposals (FAST_CROWN_V4.md 18): every serving hotkey in the
        # served set, and a follower's join audit at its largest (64
        # differences and the line saying how many more)
        "served_v4": sorted(hotkeys),
        "join_audit": {"epoch": f"e{EPOCH:08d}", "block": EPOCH * 360 - 30, "root": _hex("delta"),
                       "differences": [f"proposal {_hex(i)} of {hotkeys[i]}: the root's verdict "
                                       f"joined, this validator's refused:gate_sibling ({i})"
                                       for i in range(64)] + ["... and 192 more"],
                       "own": hotkeys[:2]},
        "exploration_pool": explore_pool.pairs_for_reveal(pooled, bench),
        "head_odds": {h: odds_b64(o) for h, o in odds.items()},
        "exploration_excluded_coldkeys": {ck: "private answers 4.1 SD from the reference over "
                                              "112 pairs of 64 entries" for ck in coldkeys},
        "single_model_table": [{"model": e, "accuracy": 0.123456789, "spend_per_question": 0.01,
                                "pairs": 10**6, "by_category": {c: {"accuracy": 0.5, "pairs": 1}
                                                                for c in ("a", "b", "c", "d", "e")}}
                               for e in entries],
        "picks_v4": picks_block,
        seat_v4.REVEAL_KEY: evidence,
        "frame_mix_weights": {f"set-{i}": 0.1 for i in range(20)},
        "explore_benchmarks": dict(sorted(bench.items())),
        "v2_sync": {"unverified": False},
        # The root's menu fields (menu_publish): the menu it ran on, the menu
        # it derived (at most MENU_MAX_BYTES), its events and lists, and its
        # summary of the epoch (a call count per coldkey and entry, at most
        # 64 per coldkey here). Allowances, not built by this package.
        "menu": {"epoch": f"e{EPOCH:08d}", "hash": _hex("menu"),
                 "source": {"epoch": f"e{EPOCH - 19:08d}", "hash": _hex("menu-source")}},
        "menu_next": "x" * 1_048_576,
        "menu_summary": {"calls": [{"entry_id": entries[i % 64], "rev": 1, "coldkey": ck,
                                    "calls": 2, "failed": 0}
                                   for ck in coldkeys for i in range(64)],
                         "votes": [{"model": e, "host": "host-a/fp8", "voters": 256,
                                    "listed": True, "price_in": 10**9, "price_out": 10**9,
                                    "limit": 10**6} for e in entries for _h in range(2)],
                         "pairs": [{"entry_id": e, "rev": 1, "category": c, "measured": m,
                                    "pairs": 10**4, "y": 10**7, "tokens_in": 10**9,
                                    "tokens_out": 10**9} for e in entries
                                   for c in ("a", "b", "c", "d", "e") for m in (True, False)]},
    }
    # The commit-reveal fields: 200 questions with 20 KB prompts (an
    # allowance), the matrix, every head's routing, its exploration answers
    # with their voters, its v1 diagnostics and its weight.
    questions = [{"question_id": q, "benchmark": "livecodebench_v6", "prompt": "p" * 20_000,
                  "gold": "g" * 200, "grader_id": "code_tests", "metadata": {}} for q in slice_q]
    exploration = [{"question_id": q, "model": p.model, "correct": True, "prompt_tokens": 12345,
                    "completion_tokens": 23456, "coldkey": coldkeys[i % UIDS], "failed": False}
                   for i, (q, p) in enumerate(zip(qids, pooled))]
    root = tmp_path_factory.mktemp("reveals")
    old = commit_reveal.EPOCH_DIR
    commit_reveal.EPOCH_DIR = str(root)
    try:
        eid = f"e{EPOCH:08d}"
        commit_reveal.commit_epoch(eid, questions, "0x" + _hex("block"))
        ok = commit_reveal.reveal_epoch(
            eid, questions, np.full((SLICE, ENTRIES), -1, np.int32), entries,
            {e: 123.456789 for e in entries}, {u: heads[u] for u in range(UIDS)},
            {u: [int(x) for x in rng.integers(0, ENTRIES, SLICE)] for u in range(UIDS)},
            {u: {k: 0.123456789 for k in ("accuracy", "quality", "thrift", "headroom",
                                          "reference_accuracy", "cost_per_question", "score")}
             for u in range(UIDS)},
            {u: 1 / UIDS for u in range(UIDS)}, exploration=exploration, protocol_version=4,
            v2_fields=v2_fields)
    finally:
        commit_reveal.EPOCH_DIR = old
    assert ok
    path = root / eid / "reveal.json"
    sizes = {k: len(json.dumps({k: v}, indent=2)) for k, v in json.loads(path.read_text()).items()}
    return eid, path.read_bytes(), sizes


def test_every_reader_of_a_reveal_uses_one_limit():
    """One limit, defined once (reveal_sync.MAX_REVEAL_BYTES): the follower
    sync's fetch, the validator's read of its own reveal, the miner's holder
    read and the root summaries' read all take every reveal any of them
    takes."""
    import ast
    import inspect

    from fugal_subnet import menu_publish

    assert reveal_sync.MAX_REVEAL_BYTES == 96 * 2**20
    assert td_v4.MAX_HOLDER_REVEAL_BYTES == reveal_sync.MAX_REVEAL_BYTES
    assert menu_publish.MAX_REVEAL_READ_BYTES == reveal_sync.MAX_REVEAL_BYTES
    # Named, not repeated: neither reader binds a number of its own.
    for mod, name, value in ((td_v4, "MAX_HOLDER_REVEAL_BYTES", "MAX_REVEAL_BYTES"),
                             (menu_publish, "MAX_REVEAL_READ_BYTES",
                              "reveal_sync.MAX_REVEAL_BYTES")):
        tree = ast.parse(inspect.getsource(mod))
        values = [n.value for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        assert [ast.unparse(v) for v in values] == [value], (mod, values)
    assert td_v4.MAX_REVEAL_BYTES is reveal_sync.MAX_REVEAL_BYTES


def test_a_worst_case_v4_reveal_fits_and_the_miner_reads_its_holder(worst_reveal):
    eid, raw, sizes = worst_reveal
    big = sorted(sizes.items(), key=lambda kv: -kv[1])[:12]
    print("worst-case v4 reveal: %.1f MiB of %.0f MiB; largest fields (MiB): %s" % (
        len(raw) / 2**20, reveal_sync.MAX_REVEAL_BYTES / 2**20,
        ", ".join(f"{k} {v / 2**20:.1f}" for k, v in big)))
    # Within three quarters of the limit at the worst shape (72 MiB of 96), so
    # every node reads it with room to spare and growth is noticed here. Most
    # of the reveal is fields that predate the seat (every head's odds, the
    # pooled labels and the v1 exploration list): what the seat adds is its
    # state, its evidence and the bounded late odds.
    assert len(raw) <= reveal_sync.MAX_REVEAL_BYTES * 3 // 4 == 72 * 2**20
    # The seat state keeps every head with up to MAX_PREDECESSORS (the
    # 48-epoch window) predecessors, one "hash:since" line each.
    seat_part = sizes["seat_v4"] + sizes["late_odds"] + sizes["v2_state"]
    assert seat_part <= 11 * 2**20
    # The memorisation fields: one short row per head the seat keeps and F ids.
    assert sizes["memo_v4"] <= 2**20
    holder = td_v4.seat_holder_from_reveal(raw, eid, hashlib.sha256(raw).hexdigest())
    assert holder == _hex("head-0")
    # The same bytes give a TD its anchor: the menu field with its source.
    from fugal_subnet import menu_publish as mp

    assert mp.anchor_from_reveal(raw, eid, hashlib.sha256(raw).hexdigest()) == (
        EPOCH, _hex("menu"), EPOCH - 19, _hex("menu-source"))


def test_the_late_odds_a_reveal_carries_are_bounded():
    """Late odds (labels_v4) for heads that missed epochs are computed a
    bounded number at a time (v2_epoch.late_epochs: newest first, the heads
    in turn); the rest follow in the next epochs. Every head's newest missing
    epoch comes before any head's second."""
    from fugal_subnet.champion import Window
    from fugal_subnet.v2_epoch import late_epochs

    n = 5
    pairs = labels_v4.label_pairs(
        EpochPairs(question_ids=("q",), models=("m",), benchmarks=("b",), categories=("c",),
                   y=np.zeros(1, np.int32), prompt_tokens=np.zeros(1, np.int32),
                   completion_tokens=np.zeros(1, np.int32)), [("ck",)], [-1])
    w = Window(size=48, slices={}, heads={}, pairs={e: pairs for e in range(1, 11)})
    heads = {f"h{i}": None for i in range(n)}
    since = {h: 1 for h in heads}
    got = late_epochs(w, heads, since, 7)
    assert sum(map(len, got.values())) == 7 and set(got) == set(heads)
    assert all(got[h][0] == 10 for h in heads)               # each head's newest first
    assert sorted(len(v) for v in got.values()) == [1, 1, 1, 2, 2]
    assert all(v == [10, 9] for v in got.values() if len(v) == 2)
    # Which heads get the second round turns with the epoch, so no head is
    # always last.
    seconds = set()
    for top in range(10, 30):
        wt = Window(size=48, slices={}, heads={}, pairs={e: pairs for e in range(top - 9, top + 1)})
        seconds |= {h for h, v in late_epochs(wt, heads, {h: 1 for h in heads}, 7).items()
                    if len(v) == 2}
    assert seconds == set(heads)
    every = late_epochs(w, heads, since, None)
    assert all(v == list(range(10, 0, -1)) for v in every.values())
    assert config.LATE_ODDS_SLOTS_V4 == 32
