"""v3 review (2026-11-16), finding 7: a pool with code questions is served
only by a harness that grades them with graders_code and a working sandbox.

Before the gate, the harness graded every question with graders.grade, which
does not know exec_stdio / exec_func and returns 0 for them, so a v3 image
would have scored every code answer 0 and paid for it, with nothing to say so.
"""
import json
import sys
from pathlib import Path

import pytest

from fugal_subnet import code_gate, graders, graders_code
from fugal_subnet.tee import harness

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_safety_invariants as csi  # noqa: E402

CODE_POOL = [{"question_id": "q1", "grader_id": "exec_stdio"},
             {"question_id": "q2", "grader_id": "letter_mcq"}]


def _unrouted(q, reply):
    """A harness that grades every question with graders.grade (no code wiring)."""
    return bool(graders.grade(harness.build_grader_task(q), reply, allow_exec=False))


def _routed(q, reply):
    """A harness that routes the code checkers to graders_code (what the wiring does)."""
    task = harness.build_grader_task(q)
    if q.get("grader_id") in code_gate.CODE_CHECKERS:
        return bool(graders_code.grade(task, reply, allow_exec=True))
    return bool(graders.grade(task, reply, allow_exec=False))


def test_a_pool_without_code_needs_nothing(monkeypatch):
    monkeypatch.setattr(harness, "grade_question", _unrouted)
    assert code_gate.release_errors([{"question_id": "q", "grader_id": "letter_mcq"}]) == []


def test_an_unwired_harness_cannot_serve_code_questions(monkeypatch):
    monkeypatch.setattr(harness, "grade_question", _unrouted)
    errors = code_gate.release_errors(CODE_POOL)
    assert errors and "does not route code questions to graders_code" in errors[-1]


@pytest.mark.skipif(not graders_code.sandbox_selftest()[0], reason="sandbox unavailable here")
def test_a_wired_harness_with_a_working_sandbox_can(monkeypatch):
    monkeypatch.setattr(harness, "grade_question", _routed)
    assert code_gate.harness_grades_code() == (True, "")
    assert code_gate.release_errors(CODE_POOL) == []


def test_a_failing_sandbox_blocks_it(monkeypatch):
    monkeypatch.setattr(harness, "grade_question", _routed)
    monkeypatch.setattr(graders_code, "sandbox_selftest", lambda: (False, "no seccomp"))
    assert any("self-test failed: no seccomp" in e for e in code_gate.release_errors(CODE_POOL))


def test_a_harness_that_grades_everything_right_is_caught(monkeypatch):
    monkeypatch.setattr(harness, "grade_question", lambda *a, **k: True)
    ok, why = code_gate.harness_grades_code()
    assert not ok and "wrong one 1" in why


def test_the_safety_check_reads_a_v3_manifest_with_code(tmp_path, monkeypatch):
    bench = tmp_path / "fugal_subnet" / "benchmarks"
    bench.mkdir(parents=True)
    errors: list[str] = []
    csi.check_code_questions_are_graded(errors, root=tmp_path)       # no manifest: nothing
    assert errors == []
    (bench / "pool_manifest_v3.json").write_text(json.dumps({"benchmarks": ["mmlu"]}))
    csi.check_code_questions_are_graded(errors, root=tmp_path)       # no code: nothing
    assert errors == []
    (bench / "pool_manifest_v3.json").write_text(json.dumps({"graders": ["exec_stdio"]}))
    monkeypatch.setattr(harness, "grade_question", _unrouted)
    csi.check_code_questions_are_graded(errors, root=tmp_path)
    assert errors and "does not route code questions" in errors[-1]


def test_the_miner_runs_the_gate_on_its_pool():
    src = (ROOT / "neurons" / "miner.py").read_text()
    load = src.index("pool = load_pool(pool_protocol)")   # v4 runs on v3's pool
    gate = src.index("code_errors = release_errors(pool)")
    assert load < gate < src.index("Benchmark pool: %d questions")
    assert "if not mock:\n            raise click.ClickException(msg)" in src


@pytest.mark.skipif(not graders_code.sandbox_selftest()[0], reason="sandbox unavailable here")
def test_the_shipped_harness_passes_the_gate_unpatched():
    """The gate must test the call run_benchmark grades with
    (harness.grade_question). It tested harness.grade (graders.grade), so the
    wired harness failed its own gate and every v3 miner refused its code
    questions (found by the v3 local-chain rehearsal, 2026-11-16)."""
    assert code_gate.harness_grades_code() == (True, "")
