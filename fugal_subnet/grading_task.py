"""Build grader-compatible task dicts from benchmark questions.

`graders.grade()` does not consume the loader's question schema directly. A
loader question carries `gold`/`grader_id`/`benchmark`/`metadata`; the grader
resolves its checker from `task["checker"]["id"]` and falls back to
`DOMAIN_CHECKER[task["domain"]]`. Handing it a raw loader dict therefore
raises KeyError inside `grade()`, which swallows it and returns 0 — every
answer grades wrong, silently.

This translation lives here, in one place, because two callers need it: the
validator-side matrix builder and the TEE harness. They must agree exactly,
or a miner's self-graded proof would disagree with the validator's grading of
the same reply.
"""
from __future__ import annotations

# Benchmark name -> the domain string graders.DOMAIN_CHECKER is keyed by.
_BENCH_TO_DOMAIN = {
    "gsm8k": "gsm8k",
    "math": "math500",
    "mmlu": "mmlupro",
    "gpqa": "gpqa",
    "aime": "aime",
    "humaneval": "humaneval",
    "livecode": "codegen",
    "ifeval": "ifbench",
}


def build_grader_task(question: dict) -> dict:
    """Build a grader-compatible task dict from a benchmark question."""
    task: dict = {"gold": question["gold"]}

    meta = question.get("metadata", {})
    checker_meta = meta.get("checker")
    if checker_meta:
        task["checker"] = checker_meta
    else:
        task["checker"] = {"id": question["grader_id"]}

    task["domain"] = _BENCH_TO_DOMAIN.get(question["benchmark"], question["benchmark"])

    if question["grader_id"] == "exec_unittest":
        task["test"] = meta.get("test", "")
        task["entry"] = meta.get("entry_point", "")
        task["stub"] = meta.get("stub", "")

    return task
