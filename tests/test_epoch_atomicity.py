"""An epoch is applied once, or not at all (I1, I3, I6).

Found 2026-11-02 by reading the loop. `update_scores` mutated the validator's
records in place and the reference frame was rebound as soon as it was
computed, both before the epoch was durable. Any exception between scoring
and `save_state` — a dropped websocket in `set_weights`, a failed owner-hotkey
read, a disk error writing the reveal — landed in the outer handler with the
loop-level state already advanced, and the retry of the SAME epoch (nothing
had bumped `last_epoch_index`) accumulated every miner's evidence a second
time, applied every miss a second time and decayed the frame a second time,
then persisted it. A single chain blip at the wrong moment made one
validator's evidence differ from every peer's until it decayed.

Separately, the on-chain commitments were read after the collection wait and
the 120 s query, with a failure swallowed into an empty map: under
REQUIRE_COMMITMENT that rejected every proof and marked the epoch processed
with no weights set, on exactly the validators whose endpoint had pruned the
boundary block (non-archive nodes keep ~300), while peers set theirs.

These tests hold the fixes' SHAPE: scoring is copy-on-write, the loop rebinds
its durable names only at the commit point, commitments are read at the
boundary and a required read never degrades to an empty map.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "neurons" / "validator.py"
DURABLE = {"scoring_state", "frame", "prev_uids", "prev_weights", "last_epoch_index"}


# ---------------------------------------------------------------- scoring --

def _state_with_one_record():
    from fugal_subnet.evidence import Evidence
    from fugal_subnet.scoring import MinerRecord, ScoringState

    rec = MinerRecord(uid=7, hotkey="hk7")
    rec.evidence = Evidence(weights_hash="h", n_total=300.0, n_correct=200.0)
    rec.epochs_seen = 3
    return ScoringState(records={7: rec}, epoch_count=3)


def _frontier():
    from fugal_subnet.frontier import build_frontier
    from fugal_subnet.reference_frame import ReferenceFrame, accumulate_exploration

    frame = ReferenceFrame()
    for _ in range(5):
        frame = accumulate_exploration(frame, [("m/a", True, 100.0, 50.0), ("m/b", False, 100.0, 50.0)])
    return build_frontier(frame, {"m/a": (1e-6, 2e-6), "m/b": (1e-7, 2e-7)}, 30_000, 300, 300.0)


def test_update_scores_leaves_its_input_untouched():
    """Copy-on-write: applying an epoch twice from the same input gives the
    same answer twice, not a doubled one."""
    import copy

    from fugal_subnet.scoring import update_scores

    before = _state_with_one_record()
    snapshot = copy.deepcopy(before)
    frontier = _frontier()

    # UID 7 is absent this epoch: a miss should be applied to the RESULT only.
    first = update_scores(before, {}, {}, frontier=frontier, hotkeys={}, n_questions=300, pool_size=20000)
    second = update_scores(before, {}, {}, frontier=frontier, hotkeys={}, n_questions=300, pool_size=20000)

    assert before == snapshot, "update_scores mutated the state it was given"
    assert first.records[7].epochs_missed == 1 == second.records[7].epochs_missed
    assert first.epoch_count == 4 == second.epoch_count
    assert first.records[7].evidence == second.records[7].evidence
    assert first.records[7].evidence != before.records[7].evidence


def test_accumulate_exploration_leaves_its_input_untouched():
    from fugal_subnet.reference_frame import ReferenceFrame, accumulate_exploration

    frame = ReferenceFrame(trials={"m/a": 10.0}, successes={"m/a": 7.0}, epochs=4)
    snapshot = frame.to_dict()
    out = accumulate_exploration(frame, [("m/a", True, 1.0, 1.0)])
    assert frame.to_dict() == snapshot
    assert out.epochs == 5 and out.trials["m/a"] > 10.0 * 0.99


# ------------------------------------------------------------ the loop --

def _main_while_body():
    tree = ast.parse(VALIDATOR.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loops = [n for n in ast.walk(main) if isinstance(n, ast.While)]
    assert len(loops) == 1, "main() has one epoch loop"
    return loops[0].body


def _assigned_names(stmt):
    if not isinstance(stmt, ast.Assign):
        return set()
    out = set()
    for t in stmt.targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                out.add(n.id)
    return out


def _calls_save_state(stmt):
    return any(
        isinstance(n, ast.Call) and getattr(n.func, "id", "") == "save_state"
        for n in ast.walk(stmt)
    )


def test_durable_state_is_rebound_only_at_a_commit_point():
    """Every assignment to a loop-level durable name inside the epoch loop is
    immediately followed (within the same statement list, at most four
    statements later) by save_state. Anything else is an epoch applied before
    it is durable, which is the double-apply on retry."""
    offenders = []

    def walk(body):
        for i, stmt in enumerate(body):
            names = _assigned_names(stmt) & DURABLE
            if names:
                window = body[i + 1:i + 5]
                if not any(_calls_save_state(s) for s in window):
                    offenders.append((stmt.lineno, sorted(names)))
            for field in ("body", "orelse", "handlers", "finalbody"):
                inner = getattr(stmt, field, None)
                if isinstance(inner, list):
                    for item in inner:
                        if isinstance(item, ast.AST) and hasattr(item, "body"):
                            walk(item.body)
                    if inner and isinstance(inner[0], ast.stmt):
                        walk(inner)

    walk(_main_while_body())
    assert not offenders, f"durable state rebound before save_state at {offenders}"


def test_commitments_are_read_before_the_collection_wait():
    """The pinned read is free to do early and ruinous to do late."""
    src = VALIDATOR.read_text(encoding="utf-8")
    main_src = src[src.index("def main("):src.index("def _get_bundle_for_uid")]
    read_at = main_src.index("read_commitments_or_fail(")
    wait_at = main_src.index("wait_for_block(subtensor, collect_block)")
    assert read_at < wait_at
    assert "get_commitments_with_blocks(" not in main_src, "only the helper reads commitments"


class _FailingSubtensor:
    def query_map(self, *a, **k):
        raise ConnectionError("Block 123 is not available (pruned)")


def test_a_required_commitments_read_never_degrades_to_an_empty_map():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("validator_under_test", VALIDATOR)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("validator_under_test", mod)
    spec.loader.exec_module(mod)

    with pytest.raises(RuntimeError, match="archive"):
        mod.read_commitments_or_fail(_FailingSubtensor(), 1, 123, required=True)
    assert mod.read_commitments_or_fail(_FailingSubtensor(), 1, 123, required=False) == {}
