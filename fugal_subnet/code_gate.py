"""The release gate for v3 code questions (v3 review, 2026-11-16, finding 7).

A pool with `exec_stdio` / `exec_func` questions may be served only by an
image whose harness grades them with fugal_subnet/graders_code.py and whose
sandbox works. Without that every code answer grades 0 and is still paid
for, silently: graders.grade does not know the two checkers, and returns 0
for an unknown one.

The check goes through the harness's OWN path (tee.harness.grade_question,
the call run_benchmark grades every reply with), on one right and one wrong
answer per checker, so it holds whatever the wiring looks like and fails
closed if the harness grades code any other way.
"""
from __future__ import annotations

CODE_CHECKERS = frozenset({"exec_stdio", "exec_func"})

# (pool-format question, right answer, wrong answer) per checker.
_PROBES = (
    ({"question_id": "code-gate-stdio", "benchmark": "code_gate", "grader_id": "exec_stdio",
      "gold": "", "metadata": {"checker": {"id": "exec_stdio", "params": {
          "tests": [{"input": "3\n", "output": "6"}, {"input": "-4\n", "output": "-8"}]}}}},
     "```python\nprint(int(input()) * 2)\n```",
     "```python\nprint(int(input()) * 3)\n```"),
    ({"question_id": "code-gate-func", "benchmark": "code_gate", "grader_id": "exec_func",
      "gold": [3, 7], "metadata": {"checker": {"id": "exec_func", "params": {
          "func": "Solution().add", "inputs": [[1, 2], [3, 4]]}}}},
     "```python\nclass Solution:\n    def add(self, a: int, b: int) -> int:\n"
     "        return a + b\n```",
     "```python\nclass Solution:\n    def add(self, a: int, b: int) -> int:\n"
     "        return a * b\n```"),
)


def pool_has_code(pool) -> bool:
    return any(q.get("grader_id") in CODE_CHECKERS for q in pool)


def harness_grades_code() -> tuple[bool, str]:
    """(ok, why): the harness grades a right code answer 1 and a wrong one 0,
    for each code checker, through its own grading call."""
    from fugal_subnet.tee import harness

    for q, right, wrong in _PROBES:
        cid = q["grader_id"]
        try:
            got_right = int(harness.grade_question(q, right))
            got_wrong = int(harness.grade_question(q, wrong))
        except Exception as e:  # noqa: BLE001 - any failure is a failed gate
            return False, f"{cid}: the harness's grading call raised {type(e).__name__}: {e}"
        if got_right != 1 or got_wrong != 0:
            return False, (f"{cid}: the harness grades a right answer {got_right} and a wrong "
                           f"one {got_wrong}; it does not route code questions to graders_code")
    return True, ""


def release_errors(pool) -> list[str]:
    """Why this image cannot serve `pool` (empty when it can). Only a pool
    with code questions needs anything."""
    if not pool_has_code(pool):
        return []
    from fugal_subnet import graders_code

    errors = []
    ok, why = graders_code.sandbox_selftest()
    if not ok:
        errors.append(f"the code sandbox self-test failed: {why}")
    ok, why = harness_grades_code()
    if not ok:
        errors.append(why)
    return errors
