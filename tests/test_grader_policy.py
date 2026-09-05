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

# Checkers that return 0 unless the caller passes allow_exec=True.
EXECUTION_CHECKERS = {"exec_io", "exec_unittest"}


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


def test_the_pool_the_loader_returns_contains_no_unrunnable_checker():
    """The invariant on the pool the NEURONS actually see.

    Checked through the loader's own filter rather than against a materialised
    file: the exclusion happens at load time, so a raw pool JSON on disk still
    contains the code questions and testing it directly would assert the wrong
    artefact. This is the path load_all() takes.
    """
    import json
    from collections import Counter

    from fugal_subnet.benchmarks.loader import _drop_unscoreable

    pool_file = REPO / "data" / "rehearsal" / "pool_full.json"
    if not pool_file.exists():          # materialised pool is optional locally
        import pytest
        pytest.skip("no materialised pool to check")
    raw = json.loads(pool_file.read_text(encoding="utf-8"))

    served = _drop_unscoreable(raw)
    unscoreable = Counter(
        q["benchmark"] for q in served if q.get("grader_id") in EXECUTION_CHECKERS
    )
    assert not unscoreable, (
        f"the loader served {dict(unscoreable)} — {sum(unscoreable.values())} "
        "questions whose checkers run candidate code while the harness grades "
        "with allow_exec=False, so they can never score above zero. The slicer "
        "equalises benchmarks, so this is a sixth of every graded slice, not a "
        "rounding error. See docs/CODE_BENCHMARK_PLAN.md."
    )


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
