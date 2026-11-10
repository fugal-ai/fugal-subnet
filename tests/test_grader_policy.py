"""Every checker the pool can reach must be able to score under the harness.

The harness calls grade(..., allow_exec=False) — a deliberate security choice,
because the sandbox in graders.py is process-level (rlimits, kill-tree) with no
filesystem or namespace isolation, and running model-influenced code inside the
enclave that produces the attested proof is a genuine risk.

Nothing reconciled that choice with what the pool contains. exec_io and
exec_unittest both begin `if not allow: return 0`, and they are the checkers for
humaneval and livecode. So those questions could never score above zero — and
because the slicer equalises benchmarks, that was not humaneval's 0.8% share of
the pool but a full SIXTH of every 300-question slice actually graded.

The harm compounds: miners pay real API cost on questions that cannot score, so
routing them to a capable model is punished on thrift with no possible quality
gain. The scoring was training heads to route code to the cheapest model, which
is the opposite of the routing policy this subnet exists to find.

This test fails if the two ever drift apart again, in either direction — a
checker added to the pool that the harness cannot run, or an execution policy
changed without revisiting the pool.
"""
import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

# Checkers that return 0 unless the caller passes allow_exec=True: one source,
# the config the loader filters by, never a copy here.
from fugal_subnet.config import EXECUTION_CHECKERS  # noqa: E402


def harness_allows_execution() -> bool:
    """Read the policy out of the harness rather than restating it here."""
    tree = ast.parse((REPO / "fugal_subnet" / "tee" / "harness.py").read_text(encoding="utf-8"))
    seen = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "grade":
            for kw in node.keywords:
                if kw.arg == "allow_exec":
                    seen.append(bool(getattr(kw.value, "value", False)))
    assert seen, "no grade(..., allow_exec=...) call found in the harness"
    assert len(set(seen)) == 1, (
        f"the harness passes inconsistent allow_exec values {set(seen)} — "
        "two questions in one epoch would be graded under different policies"
    )
    return seen[0]


def test_every_checker_that_needs_execution_is_one_the_loader_drops():
    """The drift that can actually happen: a checker that runs candidate code
    (it starts `if not allow: return 0`) but is missing from
    config.EXECUTION_CHECKERS, so the loader keeps its questions and they can
    never score; or a name in that set that no longer gates on execution, so
    scoreable questions are dropped. Read from graders.py itself, so it runs in
    CI with no pool. (It replaces a test that read a local-only pool file and
    skipped in every CI job.)"""
    tree = ast.parse((REPO / "fugal_subnet" / "graders.py").read_text(encoding="utf-8"))
    gated = set()
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        for node in ast.walk(fn):
            if (isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                    and isinstance(node.test.op, ast.Not)
                    and getattr(node.test.operand, "id", "") == "allow"):
                gated.add(fn.name)
    checkers = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", "") == "CHECKERS")
    needs_exec = {k.value for k, v in zip(checkers.keys, checkers.values) if v.id in gated}
    assert needs_exec, "no checker gates on allow: the premise of this file is gone"
    assert needs_exec == set(EXECUTION_CHECKERS), (
        f"checkers that need execution {sorted(needs_exec)} != config.EXECUTION_CHECKERS "
        f"{sorted(EXECUTION_CHECKERS)}: the loader would keep unscoreable questions or "
        "drop scoreable ones")


def test_the_exclusion_is_derived_from_the_harness_policy_not_hardcoded():
    """Turning execution on must re-include the benchmarks with no second edit.

    A hardcoded exclusion list is how the two drift apart again: someone enables
    execution, the pool stays truncated, and the benchmarks are silently gone
    rather than silently zero.
    """
    import json

    from fugal_subnet import config
    from fugal_subnet.benchmarks import loader

    raw = [{"question_id": "h1", "prompt": "p", "gold": [1],
            "grader_id": "exec_io", "benchmark": "humaneval"},
           {"question_id": "m1", "prompt": "p", "gold": "A",
            "grader_id": "letter_mcq", "benchmark": "mmlu"}]
    json  # noqa: B018 - imported for parity with the loader's own environment

    assert config.HARNESS_ALLOW_EXEC == harness_allows_execution(), (
        "config.HARNESS_ALLOW_EXEC and the value the harness actually passes to "
        "grade() disagree — the policy has two sources again"
    )

    original = config.HARNESS_ALLOW_EXEC
    try:
        config.HARNESS_ALLOW_EXEC = False
        assert [q["question_id"] for q in loader._drop_unscoreable(raw)] == ["m1"]
        config.HARNESS_ALLOW_EXEC = True
        assert [q["question_id"] for q in loader._drop_unscoreable(raw)] == ["h1", "m1"]
    finally:
        config.HARNESS_ALLOW_EXEC = original


def test_execution_checkers_really_do_return_zero_without_permission():
    """Pins the premise, so this suite cannot pass for the wrong reason."""
    from fugal_subnet.graders import CHECKERS

    task = {"gold": [1], "checker": {"id": "exec_io", "params": {"inputs": [[1]], "func": "f"}},
            "domain": "humaneval", "test": "", "entry": "", "stub": ""}
    for cid in EXECUTION_CHECKERS:
        assert CHECKERS[cid](task, "def f(x): return x", allow=False) == 0
