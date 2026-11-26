"""Protocol v5 on finney's own numbers, through the validator's glue.

v0.8.0's two v5 mistakes (fixed by v0.8.1) passed every test because the
tests used tiny synthetic blocks: the head book was fed the v1 seniority
key (registration x 10^10 + commit) as a commit block, so every head looked
committed in a far future epoch, every published exam question dropped out
and the seat could never fill; and a block was turned into its epoch by
dividing by EPOCH_INTERVAL (4320 seconds) instead of the 360 blocks of an
epoch. These tests drive the validator's own path (tests/v5_finney_world.py:
`_check_response`, the loop's bookkeeping, `_v5_commit_blocks`, `_v5_score`
on a real V5Node) at the real heights: 360-block epochs, the v5 block
9,585,720 (epoch 26627), the fix block 9,587,520 (epoch 26632),
registrations and commitments around 9.5M-9.59M, `arena_<Dataset>_<hex>`
ids, 200-question exams, N_MIN_V5 = 2000 where the window allows it.
"""
from __future__ import annotations

import ast
import json
import pathlib
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fugal_subnet import config, memo_v4, seat_v5, v5_epoch, v5_node
from fugal_subnet.benchmarks.slicer import (
    blocks_per_epoch,
    epoch_id_for_block,
    epoch_index_for_block,
)
from fugal_subnet.commitments import seniority_key
from neurons import validator
from tests import v5_finney_world as w

ROOT = pathlib.Path(__file__).resolve().parents[1]
EARLY_REG, EARLY_COMMIT = 9_576_077, 26613 * w.BPE + 77       # committed before v5
LATE_REG, LATE_COMMIT = 9_551_903, 26638 * w.BPE + 211        # committed mid-window


@pytest.fixture
def finney(monkeypatch):
    w.install(monkeypatch)
    return monkeypatch


def early():
    return w.Miner("early", 1, EARLY_REG, EARLY_COMMIT, w.GOOD)


def late():
    return w.Miner("late", 2, LATE_REG, LATE_COMMIT, w.BEST)


# --- the glue, one step at a time -----------------------------------------------------------


def test_finney_numbers_are_what_these_tests_assume():
    assert blocks_per_epoch(w.FINNEY_INTERVAL) == w.BPE == 360
    assert (w.V5_EPOCH, w.FIX_EPOCH, w.FIRST_EXAM) == (26627, 26632, 26629)
    assert config.v5_commit_fix_block(config.FINNEY_GENESIS_HASH) == w.FIX_BLOCK
    assert epoch_id_for_block(w.FIX_EPOCH) == "e00026632"
    assert int(config.N_MIN_V5) == 2000 and seat_v5.N_MIN == 2000


def test_check_response_hands_back_the_raw_block_beside_the_seniority_key(finney):
    m = early()
    world = w.World([m])
    proof = world.proof(w.FIX_EPOCH, m.hotkey, m.head)
    finney.setattr(validator, "_get_bundle_for_uid", lambda uid, resp: (proof, m.head_bytes))
    checks = validator._EpochChecks(
        boundary_block=w.FIX_BLOCK, require_commitment=True, approved_measurements=set(),
        questions_hash="", nonce_hex="00" * 32, slice_gold={}, question_ids=set(),
        explore_map={}, mock=False, collateral_budget=None, protocol_version=5)
    got = validator._check_response(1, SimpleNamespace(proof_hash="ab" * 32, weights_hash=m.head),
                                    m.hotkey, (m.head, EARLY_COMMIT), EARLY_REG, checks)
    assert got.status == "valid"
    assert got.commit_block == EARLY_COMMIT == 9_580_757
    assert got.seniority == seniority_key(EARLY_REG, EARLY_COMMIT) == 95_760_770_009_580_757
    # The two are told apart by size alone (v5_epoch.raw_commit_book).
    assert got.commit_block < v5_epoch.SENIORITY_BASE < got.seniority


@pytest.mark.parametrize("epoch,raw", [(w.V5_EPOCH, False), (w.FIX_EPOCH - 1, False),
                                       (w.FIX_EPOCH, True), (w.FIX_EPOCH + 1, True)])
def test_the_commit_blocks_v5_reads_switch_to_raw_at_the_fix_boundary(epoch, raw):
    keys, blocks = {1: seniority_key(EARLY_REG, EARLY_COMMIT)}, {1: EARLY_COMMIT}
    got, is_raw = validator._v5_commit_blocks(epoch * w.BPE, w.FIX_BLOCK, seniority=keys,
                                              raw=blocks)
    assert (got, is_raw) == ((blocks, True) if raw else (keys, False))
    # Another chain without a fix block keeps v0.8.0's (its v5 block is its fix).
    assert validator._v5_commit_blocks(epoch * w.BPE, None, seniority=keys,
                                       raw=blocks) == (keys, False)


def test_the_epoch_loop_feeds_v5_what_check_response_returned_through_the_switch():
    # The loop itself needs a chain; its v5 wiring is read from the source:
    # each uid's _Checked.seniority and .commit_block are kept apart, go
    # through _v5_commit_blocks, and its result is what _v5_score reads.
    tree = ast.parse((ROOT / "neurons" / "validator.py").read_text(encoding="utf-8"))
    src = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)}
    assert "commit_blocks[uid] = checked.seniority" in src
    assert "raw_commit_blocks[uid] = checked.commit_block" in src
    switch = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and isinstance(n.value, ast.Call)
              and getattr(n.value.func, "id", None) == "_v5_commit_blocks"]
    assert len(switch) == 1
    (sw,) = switch
    assert ast.unparse(sw.targets[0]) == "(v5_blocks, raw_blocks)"
    assert [ast.unparse(a) for a in sw.value.args] == ["boundary_block", "v5_fix_block"]
    assert {k.arg: ast.unparse(k.value) for k in sw.value.keywords} == {
        "seniority": "commit_blocks", "raw": "raw_commit_blocks"}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "_v5_score"]
    assert len(calls) == 1
    kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert (kw["commit_blocks"], kw["raw_commit_blocks"]) == ("v5_blocks", "raw_blocks")


# --- through _v5_score, at real sizes ---------------------------------------------------------


def test_from_the_fix_served_heads_carry_raw_blocks_and_the_seat_fills_at_n_min(finney):
    m = early()
    world = w.World([m]).run(w.V5_EPOCH, w.FIRST_EXAM + 11, finney)
    key = seniority_key(EARLY_REG, EARLY_COMMIT)
    for e, served in world.served.items():
        (s,) = served
        assert (s.head, s.uid, s.hotkey, s.coldkey) == (m.head, 1, m.hotkey, m.coldkey)
        assert s.commit_block == (EARLY_COMMIT if e >= w.FIX_EPOCH else key), e
        assert world.raw[e] == (e >= w.FIX_EPOCH)
    # The book holds the raw block from the fix on, in the state and in the reveal.
    assert world.node.heads[m.head].first_commit_block == EARLY_COMMIT
    for e in range(w.FIX_EPOCH, w.FIRST_EXAM + 12):
        assert world.results[e].reveal[v5_epoch.HEADS_KEY][m.head][2] == EARLY_COMMIT
    # Every exam since the first one counts: 200 questions an epoch add up to
    # N_MIN = 2000 at the tenth exam, and the seat fills there, not before.
    fill = w.FIRST_EXAM + w.n_min() // w.EXAM_N - 1
    assert fill == 26638
    for e in range(w.FIX_EPOCH, w.FIRST_EXAM + 12):
        res = world.results[e]
        n = w.bound_n(res, m.head, "vs_baseline")
        assert n == w.EXAM_N * (e - w.FIRST_EXAM + 1), e
        assert res.decision.holder == (m.head if e >= fill else None), e
    assert world.results[fill].decision.event == "fill"
    _scored, weights = world.epoch(w.FIRST_EXAM + 12, finney)
    assert weights[1] > 0 and weights[0] > 0              # the seat pays, the rest burns


def test_before_the_fix_v080_replays_as_published(finney):
    # The epochs before the fix block replay v0.8.0 byte for byte, and v0.8.0
    # kept on (no fix block) never fills: the key puts the commit in a far
    # future epoch, so only the two exams not yet published count.
    fixed = w.World([early()]).run(w.V5_EPOCH, w.FIRST_EXAM + 11, finney)
    v080 = w.World([early()], fix_block=None).run(w.V5_EPOCH, w.FIRST_EXAM + 11, finney)
    for e in range(w.V5_EPOCH, w.FIX_EPOCH):
        assert (json.dumps(fixed.results[e].reveal, sort_keys=True)
                == json.dumps(v080.results[e].reveal, sort_keys=True)), e
    head = early().head
    for e, res in v080.results.items():
        assert not v080.raw[e]
        assert res.decision.holder is None and res.decision.event == "empty", e
        assert (w.bound_n(res, head, "vs_baseline") or 0) <= 2 * w.EXAM_N, e
    assert v080.node.heads[head].first_commit_block == seniority_key(EARLY_REG, EARLY_COMMIT)


# --- a long window: a head committed before, one committed mid-window --------------------------


def _long_window(monkeypatch, *, stray_n=30):
    """26627 .. 26648 (exams 26629 .. 26648, twenty), N_MIN scaled to 1000 so a
    head committed mid-window can reach it inside the window. `stray_n`
    questions of the late head's first exam were published by another path
    in the epoch before its commit: they never count for it."""
    monkeypatch.setattr(seat_v5, "N_MIN", 1000)
    monkeypatch.setattr(seat_v5, "N0", 1000)
    e, x = early(), late()
    world = w.World([e, x])
    first_late = LATE_COMMIT // w.BPE + 1 + v5_epoch.SPEC_LAG      # 26641
    stray = set(world.exam_questions(first_late)[:stray_n])
    world.stray = {LATE_COMMIT // w.BPE - 1: stray}
    world.run(w.V5_EPOCH, w.FIRST_EXAM + 19, monkeypatch)
    return world, e, x, first_late, stray


def test_a_long_window_fills_on_the_bound_and_a_later_head_takes_on_its_own_questions(finney):
    world, e, x, first_late, stray = _long_window(finney)
    assert first_late == 26641 and LATE_COMMIT // w.BPE == 26638
    # The late head is served from the first boundary after its commitment,
    # and is in the exams built from then on.
    assert [s.head for s in world.served[26638]] == [e.head]
    assert sorted(s.head for s in world.served[26639]) == sorted([e.head, x.head])
    assert x.head not in world.specs[first_late - 1].picks and x.head in world.specs[first_late].picks
    events = {k: (r.decision.event, r.decision.holder) for k, r in world.results.items()}
    # The early head: nothing before the fix, fills at the first epoch its
    # paired questions reach N_MIN (five exams of 200) and the bound passes.
    for k in range(w.V5_EPOCH, 26633):
        assert events[k] == ("empty", None), k
    assert events[26633] == ("fill", e.head)
    assert w.bound_n(world.results[26633], e.head, "vs_baseline") == 1000
    # The late head: its stray questions never count (published before its
    # commit epoch); every other question of its exams does.
    for k in range(first_late, w.FIRST_EXAM + 20):
        want = w.EXAM_N * (k - first_late + 1) - len(stray)
        assert w.bound_n(world.results[k], x.head, "vs_baseline") == want, k
        if world.results[k].decision.holder == e.head:
            assert w.bound_n(world.results[k], x.head, "vs_holder") == want, k
    # So it reaches N_MIN one exam later than it would counting them, and takes there.
    take = first_late + -(-(1000 + len(stray)) // w.EXAM_N) - 1
    assert take == 26646
    for k in range(26634, take):
        assert events[k] == ("hold", e.head), k
    assert events[take] == ("take", x.head)
    assert world.results[take].decision.history == (e.head,)
    for k in range(take + 1, w.FIRST_EXAM + 20):
        assert events[k] == ("hold", x.head), k

    # B* has its labels: the one full entry, on every verified hotkey's
    # coverage of the window, one vote each, from the activation on.
    hotkeys = {kk: len(w.VOTERS) + len(world.served[kk]) for kk in world.served}
    for k, res in world.results.items():
        assert res.baseline == w.BASE, k
        table = res.reveal[v5_epoch.BASELINE_KEY]["table"]
        want = sum(hotkeys[kk] * w.COVERAGE_N for kk in hotkeys
                   if k - v5_epoch.WINDOW_EPOCHS < kk <= k)
        assert table[w.BASE]["n"] == want >= v5_epoch.BASELINE_MIN_PAIRS, k
    # Every exam question carries B*'s cell, and every one of them is labelled
    # by an operator other than the head's: the early head's paired count is
    # the window's whole exam.
    for k in range(26633, w.FIRST_EXAM + 20):
        assert w.bound_n(world.results[k], e.head, "vs_baseline") == w.EXAM_N * (k - w.FIRST_EXAM + 1)
    for spec in world.specs.values():
        assert set(spec.baseline_questions) == set(spec.questions)
        assert all(len(hks) == 3 for (q, ent), hks in spec.assignment.items() if ent == w.BASE)


def test_the_long_window_tells_both_v080_mistakes_apart(finney):
    # Seconds for blocks: the late head's commit epoch is 2220, so the stray
    # questions count and it takes one exam early.
    finney.setattr(v5_epoch, "commit_epoch", v5_epoch.commit_epoch_v080)
    world, _e, x, _first_late, _stray = _long_window(finney)
    takes = [k for k, r in world.results.items() if r.decision.event == "take"]
    assert takes == [26645] and world.results[26645].decision.holder == x.head
    finney.undo()
    # Seniority keys for blocks: nothing ever fills.
    w.install(finney)
    finney.setattr(seat_v5, "N_MIN", 1000)
    finney.setattr(seat_v5, "N0", 1000)
    keys = w.World([early(), late()], fix_block=None).run(w.V5_EPOCH, w.FIRST_EXAM + 19, finney)
    assert all(r.decision.holder is None for r in keys.results.values())


# --- every block -> epoch conversion agrees with slicer's ------------------------------------

REALISTIC = st.integers(min_value=9_400_000, max_value=9_800_000)
ANY_BLOCK = st.integers(min_value=0, max_value=2**40)


@settings(max_examples=300, deadline=None)
@given(REALISTIC | ANY_BLOCK)
def test_commit_epoch_is_slicers_epoch_index_on_finney_and_at_the_suite_default(block):
    for interval in (w.FINNEY_INTERVAL, int(config.EPOCH_INTERVAL)):
        bpe = blocks_per_epoch(interval)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(v5_epoch, "EPOCH_INTERVAL", interval)
            mp.setattr(v5_epoch, "BLOCKS_PER_EPOCH", bpe)
            e = v5_epoch.commit_epoch(block)
            assert e == epoch_index_for_block(block, bpe)
            # the same epoch the published set records (an epoch id round trip)
            assert memo_v4.epoch_index(epoch_id_for_block(e)) == e
            # the boundary of that epoch is at or before the block, the next after it
            assert e * bpe <= block < (e + 1) * bpe
    # the module's own constant is the suite's interval's
    assert v5_epoch.BLOCKS_PER_EPOCH == blocks_per_epoch(v5_epoch.EPOCH_INTERVAL)


@settings(max_examples=200, deadline=None)
@given(REALISTIC, REALISTIC)
def test_raw_and_keyed_commitments_name_the_same_epoch_after_the_fix(registered, block):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(v5_epoch, "BLOCKS_PER_EPOCH", w.BPE)
        key = seniority_key(registered, block)
        book = {"h": v5_epoch.HeadFacts("h", "ck", "hk", key, w.FIX_EPOCH)}
        raw = v5_epoch.raw_commit_book(book)["h"].first_commit_block
        assert raw == block
        assert v5_epoch.commit_epoch(raw) == epoch_index_for_block(block, w.BPE)
        # v0.8.0's arithmetic is wrong by a factor of 12 on every finney block
        assert v5_epoch.commit_epoch_v080(block) != v5_epoch.commit_epoch(block)


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=w.V5_EPOCH - 30, max_value=w.FIX_EPOCH + 300))
def test_epoch_to_block_reads_land_on_the_epochs_boundary(epoch):
    # The fix switch (validator._v5_commit_blocks) on a boundary block.
    _b, raw = validator._v5_commit_blocks(epoch * w.BPE, w.FIX_BLOCK, seniority={}, raw={})
    assert raw == (epoch >= w.FIX_EPOCH) == (epoch_index_for_block(epoch * w.BPE, w.BPE)
                                             >= epoch_index_for_block(w.FIX_BLOCK, w.BPE))
    # A follower's read of the root's exam spec (v5_node.published_spec_reader)
    # is pinned at the epoch's boundary block.
    from fugal_subnet import commitments

    asked = []

    def read(subtensor, netuid, block=None):
        asked.append(block)
        return {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(commitments, "get_commitments_with_blocks", read)
        reader = v5_node.published_spec_reader(None, netuid=70, root_hotkey="5Root",
                                               blocks_per_epoch=w.BPE,
                                               fetch_reveal=lambda *a: None)
        assert reader(epoch) == (None, {})
    assert asked == [epoch * w.BPE] and epoch_index_for_block(asked[0], w.BPE) == epoch


def test_a_miners_v5_epochs_start_at_the_v5_blocks_epoch():
    from fugal_subnet.tee import td_v5
    from neurons import miner

    session = td_v5.V5Session(v4_session=None, exam=None, first_epoch=w.V5_EPOCH,
                              protocol_of=miner._protocol_of_boundary(config.FINNEY_GENESIS_HASH),
                              blocks_per_epoch=w.BPE)
    for e in range(w.V5_EPOCH - 5, w.V5_EPOCH + 5):
        assert session.is_v5(e) == (e >= w.V5_EPOCH) == (e * w.BPE >= w.V5_BLOCK), e


# --- no v5 code divides a block by seconds ---------------------------------------------------

# Where dividing by EPOCH_INTERVAL is allowed: v0.8.0's arithmetic, kept only
# to replay the epochs before the fix block.
ALLOWED = {("fugal_subnet/v5_epoch.py", "commit_epoch_v080")}


def _divisions_by_the_interval(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.FloorDiv, ast.Div, ast.Mod)):
            right = node.right
        elif (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "divmod"
              and len(node.args) == 2):
            right = node.args[1]
        else:
            continue
        if isinstance(right, ast.Call) and any(
                k in ast.unparse(right.func) for k in ("blocks_per_epoch", "bpe")):
            continue                      # blocks_per_epoch(EPOCH_INTERVAL): blocks, correct
        if "EPOCH_INTERVAL" not in ast.unparse(right).upper():
            continue
        fn, up = None, parents.get(node)
        while up is not None and fn is None:
            if isinstance(up, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = up.name
            up = parents.get(up)
        yield fn, node.lineno, ast.unparse(node)


def test_no_v5_or_memo_code_divides_a_block_by_epoch_interval_seconds():
    files = sorted({*ROOT.glob("fugal_subnet/v5_*.py"), *ROOT.glob("fugal_subnet/*_v5.py"),
                    *ROOT.glob("fugal_subnet/memo_*.py"), *ROOT.glob("fugal_subnet/tee/*v5*.py"),
                    *ROOT.glob("fugal_subnet/**/*.py"), ROOT / "neurons" / "validator.py",
                    ROOT / "neurons" / "miner.py"})
    files = [f for f in files if "vendor" not in f.parts]
    assert {"v5_epoch.py", "seat_v5.py", "exam_v5.py", "memo_sync.py", "memo_v4.py",
            "v5_node.py", "td_v5.py"} <= {f.name for f in files}
    found = []
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        for fn, line, text in _divisions_by_the_interval(f):
            if (rel, fn) not in ALLOWED:
                found.append(f"{rel}:{line} in {fn}: {text}")
    assert not found, ("a block divided by EPOCH_INTERVAL (seconds) is not its epoch: use "
                       "slicer.epoch_index_for_block(block, blocks_per_epoch(EPOCH_INTERVAL))\n"
                       + "\n".join(found))
    # and the one allowed exception is still there, still only for the replay
    assert [fn for fn, _l, _t in _divisions_by_the_interval(ROOT / "fugal_subnet/v5_epoch.py")] \
        == ["commit_epoch_v080"]


def test_the_guard_sees_the_v080_mistake(tmp_path):
    bad = tmp_path / "v5_bad.py"
    bad.write_text("from fugal_subnet.config import EPOCH_INTERVAL\n"
                   "def commit_epoch(block):\n    return int(block) // EPOCH_INTERVAL\n"
                   "def other(b):\n    return divmod(b, config.EPOCH_INTERVAL)\n"
                   "def fine(b):\n    return b // blocks_per_epoch(EPOCH_INTERVAL)\n")
    assert [fn for fn, _l, _t in _divisions_by_the_interval(bad)] == ["commit_epoch", "other"]
