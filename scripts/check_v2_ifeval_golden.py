#!/usr/bin/env python3
"""Evaluate every canonical IFEval instruction under the pinned runtime."""

from __future__ import annotations

import hashlib

from fugal_subnet.consensus_manifest import canonical_json
from fugal_subnet.v2.benchmarks import benchmark_sha256, load_benchmark, load_registry
from fugal_subnet.vendored.ifeval.evaluator import evaluate_strict

EXPECTED_COUNT = 541
EXPECTED_CHECKS = 834
EXPECTED_TRACE_SHA256 = "1236b7eb60e75979dc930334e57604149c2e95cf5b8e0cebada0d4451e61af4a"


def main() -> int:
    questions = load_benchmark("ifeval")
    trace = []
    for question in questions:
        metadata = question["metadata"]
        followed, parts = evaluate_strict(
            prompt=question["prompt"],
            response=question["prompt"],
            instruction_ids=metadata["instruction_ids"],
            kwargs=metadata["kwargs"],
        )
        trace.append([question["question_id"], followed, list(parts)])
    specification = load_registry()["benchmarks"]["ifeval"]
    digest = hashlib.sha256(canonical_json(trace)).hexdigest()
    checks = sum(len(item[2]) for item in trace)
    if (
        len(questions) != EXPECTED_COUNT
        or checks != EXPECTED_CHECKS
        or benchmark_sha256(questions) != specification["sha256"]
        or digest != EXPECTED_TRACE_SHA256
    ):
        raise RuntimeError(
            "complete IFEval golden differs: "
            f"questions={len(questions)}, checks={checks}, trace_sha256={digest}"
        )
    print(
        f"IFEval v2 golden passed: {len(questions)} rows, {checks} checks, {digest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
