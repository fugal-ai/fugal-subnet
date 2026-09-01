#!/usr/bin/env python3
"""Rebuild the curated, exact-I/O HumanEval v2 package resource."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

from datasets import load_dataset

from fugal_subnet.benchmarks.loader import DATASET_REVISIONS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "fugal_subnet" / "human-eval-cases-v2.json"
MIN_CASES = 8
SOURCE_DATASET = "openai/openai_humaneval"
SOURCE_REVISION = DATASET_REVISIONS[SOURCE_DATASET]


def _json_safe(value: object) -> bool:
    """Use only values whose Python type survives a JSON round trip exactly."""
    if value is None or type(value) in (bool, int, str):
        return True
    if isinstance(value, list):
        return all(_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_safe(item) for key, item in value.items())
    return False


def _extract_cases(test_source: str) -> list[dict]:
    tree = ast.parse(test_source)
    cases: list[dict] = []
    seen: set[bytes] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assert)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
        ):
            continue
        call = node.test.left
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "candidate"
            and not call.keywords
            and all(not isinstance(arg, ast.Starred) for arg in call.args)
        ):
            continue
        try:
            arguments = [ast.literal_eval(arg) for arg in call.args]
            expected = ast.literal_eval(node.test.comparators[0])
        except (ValueError, TypeError):
            continue
        if not _json_safe(arguments) or not _json_safe(expected):
            continue
        case = {"arguments": arguments, "expected": expected}
        encoded = json.dumps(
            case,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if encoded not in seen:
            seen.add(encoded)
            cases.append(case)
    return cases


def build() -> dict:
    dataset = load_dataset(
        SOURCE_DATASET,
        split="test",
        revision=SOURCE_REVISION,
    )
    tasks = []
    for row in dataset:
        cases = _extract_cases(row["test"])
        if len(cases) < MIN_CASES:
            continue
        task_id = row["task_id"]
        tasks.append({
            "task_id": task_id,
            "question_id": task_id.replace("/", "_"),
            "prompt": (
                "Write a Python function to solve the following problem.\n\n"
                + row["prompt"]
                + "\n\nComplete the function. Return only the code."
            ),
            "stub": row["prompt"],
            "function": row["entry_point"],
            "cases": cases,
            "source_test_sha256": hashlib.sha256(row["test"].encode("utf-8")).hexdigest(),
            "canonical_solution_sha256": hashlib.sha256(
                row["canonical_solution"].encode("utf-8")
            ).hexdigest(),
        })
    tasks.sort(key=lambda item: int(item["task_id"].split("/")[-1]))
    if not tasks or any(len(task["cases"]) < MIN_CASES for task in tasks):
        raise RuntimeError("HumanEval v2 curation invariant failed")
    return {
        "schema_version": 1,
        "license": "MIT",
        "minimum_cases_per_task": MIN_CASES,
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(
        f"Wrote {len(payload['tasks'])} tasks to {args.output} "
        f"(sha256={hashlib.sha256(encoded).hexdigest()})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
