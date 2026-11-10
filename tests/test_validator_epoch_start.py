"""The start of a validator epoch: which epoch runs, what it is anchored to,
what it asks for, and when it asks.

These rules lived inline in neurons/validator.py's epoch loop, where planted
bugs survived the whole suite (2026-11-08): a processed epoch run a second
time, the nonce taken from the current block's hash, the whole pool handed to
the verifier as gold, a wait that returned a block early. They are small
functions now, tested here. The wiring that is still inline (which block the
commitments and the metagraph are read at) is checked on the syntax tree.
"""
import ast
import pathlib

import pytest

from fugal_subnet.benchmarks.slicer import derive_nonce, epoch_id_for_block
from fugal_subnet.epoch_plan import exploration_plan, scored_slice
from fugal_subnet.tee.verify import compute_questions_hash
from neurons import validator
from neurons.validator import _epoch_boundary, _epoch_slice, _epoch_to_run

BPE = 360


# --------------------------------------------------------- which epoch runs --

def test_an_epoch_already_written_is_not_run_again():
    assert _epoch_to_run(10 * BPE + 5, BPE, last_epoch_index=10, once=False) is None


def test_the_next_epoch_runs():
    assert _epoch_to_run(11 * BPE, BPE, last_epoch_index=10, once=False) == 11


def test_once_reruns_the_current_epoch_on_purpose():
    assert _epoch_to_run(10 * BPE + 5, BPE, last_epoch_index=10, once=True) == 10


# ------------------------------------------------- what an epoch is anchored to --

class _Hashes:
    def __init__(self):
        self.asked = []

    def get_block_hash(self, block):
        self.asked.append(block)
        return f"0x{block:064x}"


def test_the_epoch_is_anchored_to_its_boundary_block():
    chain = _Hashes()
    boundary, block_hash, epoch_id, version = _epoch_boundary(chain, 12, BPE, None)
    assert boundary == 12 * BPE and chain.asked == [12 * BPE]
    assert block_hash == f"0x{12 * BPE:064x}"
    assert epoch_id == epoch_id_for_block(12)
    assert version == 1


def test_the_protocol_version_is_decided_by_the_boundary_block():
    assert _epoch_boundary(_Hashes(), 99, BPE, 100 * BPE)[3] == 1
    assert _epoch_boundary(_Hashes(), 100, BPE, 100 * BPE)[3] == 2


# ----------------------------------------------------- what an epoch asks for --

def _pool(version=1):
    if version == 2:
        from fugal_subnet.config import BENCHMARK_CATEGORY_V2
        names = sorted(BENCHMARK_CATEGORY_V2)
    else:
        names = ("mmlu", "math", "gsm8k")
    return [{"question_id": f"{b}-{i}", "benchmark": b, "prompt": "p"}
            for b in names for i in range(400)]


@pytest.mark.parametrize("version", [1, 2])
def test_the_slice_is_the_epoch_plan_for_the_boundary_nonce(version):
    pool = _pool(version)
    gold = {q["question_id"]: q for q in pool}
    models = ["m/a", "m/b", "m/c"]
    plan = _epoch_slice(version, "e00000012", "0xabc", pool, gold, models, 300, 6)

    nonce = derive_nonce("e00000012", "0xabc")
    assert plan.nonce == nonce
    assert plan.questions == scored_slice(version, nonce, pool, 300)
    assert plan.question_ids == [q["question_id"] for q in plan.questions]
    assert plan.questions_hash == compute_questions_hash(plan.question_ids)
    assert plan.explore_map == exploration_plan(version, nonce, pool, set(plan.question_ids),
                                                models, 6)


def test_the_verifier_gets_gold_for_the_assigned_questions_only():
    pool = _pool()
    gold = {q["question_id"]: q for q in pool}
    plan = _epoch_slice(1, "e00000012", "0xabc", pool, gold, ["m/a", "m/b"], 60, 6)
    assigned = set(plan.question_ids) | set(plan.explore_map)
    assert set(plan.slice_gold) == assigned
    assert len(plan.slice_gold) < len(gold)


# ------------------------------------------------------------- when it asks --

class _Chain:
    """One block per height read; the first `drops` reads fail like a dropped socket."""

    def __init__(self, start, drops=0):
        self.height, self.drops = start, drops

    def get_current_block(self):
        if self.drops:
            self.drops -= 1
            raise ConnectionError("socket closed")
        self.height += 1
        return self.height


def test_the_wait_returns_at_the_target_block_not_before(monkeypatch):
    monkeypatch.setattr(validator.time, "sleep", lambda s: None)
    chain = _Chain(start=990)
    validator.wait_for_block(chain, 1000)
    assert chain.height == 1000


def test_a_dropped_socket_does_not_end_the_wait(monkeypatch):
    monkeypatch.setattr(validator.time, "sleep", lambda s: None)
    chain = _Chain(start=990, drops=3)
    validator.wait_for_block(chain, 1000)
    assert chain.height == 1000


# ----------------------------------------- inline wiring, on the syntax tree --

def _calls_in_main(name):
    tree = ast.parse(pathlib.Path(validator.__file__).read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return [n for n in ast.walk(main)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == name]


def test_commitments_are_read_at_the_boundary_and_required_live():
    (call,) = _calls_in_main("read_commitments_or_fail")
    assert ast.unparse(call.args[2]) == "boundary_block"
    required = {k.arg: ast.unparse(k.value) for k in call.keywords}["required"]
    assert required == "live and REQUIRE_COMMITMENT"


def test_the_metagraph_is_reread_at_the_collection_block():
    (call,) = _calls_in_main("metagraph_at")
    assert ast.unparse(call.args[2]) == "collect_block"
    (wait,) = _calls_in_main("wait_for_block")
    assert ast.unparse(wait.args[1]) == "collect_block"
