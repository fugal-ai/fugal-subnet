"""Canonical, hash-verified benchmark pool for the inactive v2 protocol."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
from typing import Callable

from datasets import load_dataset

from fugal_subnet.benchmarks.loader import DATASET_REVISIONS
from fugal_subnet.consensus_manifest import canonical_json

REGISTRY_RESOURCE = "benchmark-registry-v2.json"
HUMANEVAL_RESOURCE = "human-eval-cases-v2.json"
CANONICAL_BENCHMARKS = ("aime", "gsm8k", "humaneval", "ifeval", "math", "mmlu")
_MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


class BenchmarkIntegrityError(RuntimeError):
    """Canonical benchmark material differs from its packaged commitment."""


def _resource(name: str) -> bytes:
    return importlib.resources.files("fugal_subnet").joinpath(name).read_bytes()


def _read_unique_json(data: bytes, label: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BenchmarkIntegrityError(f"duplicate {label} JSON key: {key}")
            result[key] = value
        return result

    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkIntegrityError(f"{label} is not canonical UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise BenchmarkIntegrityError(f"{label} root must be an object")
    return result


def load_registry() -> dict:
    registry = _read_unique_json(_resource(REGISTRY_RESOURCE), "benchmark registry")
    if set(registry) != {"schema_version", "pool_sha256", "benchmarks"}:
        raise BenchmarkIntegrityError("benchmark registry keys differ")
    if registry["schema_version"] != 1 or not isinstance(registry["benchmarks"], dict):
        raise BenchmarkIntegrityError("benchmark registry schema is unsupported")
    if tuple(sorted(registry["benchmarks"])) != CANONICAL_BENCHMARKS:
        raise BenchmarkIntegrityError("benchmark registry membership differs")
    return registry


def _last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    index = start + len(marker)
    depth = 1
    content_start = index
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    return text[content_start:index - 1] if depth == 0 else None


def _load_mmlu() -> list[dict]:
    dataset = load_dataset(
        "cais/mmlu",
        "all",
        split="test",
        revision=DATASET_REVISIONS["cais/mmlu"],
    )
    labels = ("A", "B", "C", "D")
    result = []
    for index, row in enumerate(dataset):
        choices = list(row["choices"])
        if len(choices) != 4 or not 0 <= int(row["answer"]) < 4:
            raise BenchmarkIntegrityError("MMLU row has invalid choices or answer")
        choice_text = "\n".join(f"{labels[i]}. {value}" for i, value in enumerate(choices))
        result.append({
            "prompt": f"{row['question']}\n\n{choice_text}\n\nAnswer with the letter only.",
            "gold": labels[int(row["answer"])],
            "grader_id": "letter_mcq_v2",
            "benchmark": "mmlu",
            "question_id": f"mmlu_{index:05d}",
            "metadata": {"subject": row.get("subject", "")},
        })
    return result


def _load_gsm8k() -> list[dict]:
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
        revision=DATASET_REVISIONS["openai/gsm8k"],
    )
    result = []
    for index, row in enumerate(dataset):
        match = re.search(r"####\s*(.+)", row["answer"])
        gold = (match.group(1) if match else row["answer"]).strip().replace(",", "")
        result.append({
            "prompt": row["question"],
            "gold": gold,
            "grader_id": "numeric_decimal_v2",
            "benchmark": "gsm8k",
            "question_id": f"gsm8k_{index:04d}",
            "metadata": {},
        })
    return result


def _load_math() -> list[dict]:
    result = []
    global_index = 0
    for config in _MATH_CONFIGS:
        dataset = load_dataset(
            "EleutherAI/hendrycks_math",
            config,
            split="test",
            revision=DATASET_REVISIONS["EleutherAI/hendrycks_math"],
        )
        for row in dataset:
            gold = _last_boxed(row["solution"])
            if gold:
                result.append({
                    "prompt": row["problem"],
                    "gold": gold,
                    "grader_id": "symbolic_math_v2",
                    "benchmark": "math",
                    "question_id": f"math_{config}_{global_index:04d}",
                    "metadata": {
                        "level": row.get("level", ""),
                        "type": row.get("type", config),
                    },
                })
            global_index += 1
    return result


def _load_aime() -> list[dict]:
    dataset = load_dataset(
        "qq8933/AIME_1983_2024",
        split="train",
        revision=DATASET_REVISIONS["qq8933/AIME_1983_2024"],
    )
    result = []
    for index, row in enumerate(dataset):
        question = row.get("Question", "")
        answer = str(row.get("Answer", ""))
        if not question or not answer:
            continue
        year = row.get("Year", "")
        part = row.get("Part", "I")
        try:
            number = int(row.get("Problem Number", 0) or 0)
        except (TypeError, ValueError):
            number = 0
        result.append({
            "prompt": question,
            "gold": answer,
            "grader_id": "integer_decimal_v2",
            "benchmark": "aime",
            "question_id": f"aime_{year}_{part}_{number:02d}_{index:04d}",
            "metadata": {"year": year, "part": row.get("Part", "")},
        })
    return result


def _load_ifeval() -> list[dict]:
    dataset = load_dataset(
        "google/IFEval",
        split="train",
        revision=DATASET_REVISIONS["google/IFEval"],
    )
    result = []
    for index, row in enumerate(dataset):
        instruction_ids = list(row["instruction_id_list"])
        raw_kwargs = list(row["kwargs"])
        if not instruction_ids or len(instruction_ids) != len(raw_kwargs):
            raise BenchmarkIntegrityError("IFEval instruction metadata is malformed")
        sparse_kwargs = [
            {key: value for key, value in kwargs.items() if value is not None}
            for kwargs in raw_kwargs
        ]
        key = row.get("key", index)
        result.append({
            "prompt": row["prompt"],
            "gold": None,
            "grader_id": "ifeval_strict_v2",
            "benchmark": "ifeval",
            "question_id": f"ifeval_{int(key):04d}",
            "metadata": {
                "instruction_ids": instruction_ids,
                "kwargs": sparse_kwargs,
            },
        })
    return result


def _load_humaneval() -> list[dict]:
    payload = _read_unique_json(_resource(HUMANEVAL_RESOURCE), "HumanEval cases")
    if payload.get("schema_version") != 1 or payload.get("minimum_cases_per_task") != 8:
        raise BenchmarkIntegrityError("HumanEval case resource schema differs")
    if payload.get("source_revision") != DATASET_REVISIONS["openai/openai_humaneval"]:
        raise BenchmarkIntegrityError("HumanEval case source revision differs")
    result = []
    for task in payload.get("tasks", []):
        cases = task.get("cases")
        if not isinstance(cases, list) or len(cases) < 8:
            raise BenchmarkIntegrityError("HumanEval task has fewer than eight cases")
        result.append({
            "prompt": task["prompt"],
            "gold": [case["expected"] for case in cases],
            "grader_id": "code_io_v2",
            "benchmark": "humaneval",
            "question_id": task["question_id"],
            "metadata": {
                "function": task["function"],
                "inputs": [case["arguments"] for case in cases],
                "source_test_sha256": task["source_test_sha256"],
            },
        })
    return result


_LOADERS: dict[str, Callable[[], list[dict]]] = {
    "aime": _load_aime,
    "gsm8k": _load_gsm8k,
    "humaneval": _load_humaneval,
    "ifeval": _load_ifeval,
    "math": _load_math,
    "mmlu": _load_mmlu,
}


def benchmark_sha256(questions: list[dict]) -> str:
    return hashlib.sha256(canonical_json(questions)).hexdigest()


def _validate_questions(name: str, questions: list[dict]) -> None:
    if not questions:
        raise BenchmarkIntegrityError(f"{name} canonical pool is empty")
    ids = set()
    expected_keys = {"prompt", "gold", "grader_id", "benchmark", "question_id", "metadata"}
    for question in questions:
        if not isinstance(question, dict) or set(question) != expected_keys:
            raise BenchmarkIntegrityError(f"{name} question schema differs")
        if question["benchmark"] != name:
            raise BenchmarkIntegrityError(f"{name} question has wrong benchmark label")
        if not isinstance(question["prompt"], str) or not question["prompt"]:
            raise BenchmarkIntegrityError(f"{name} question prompt is empty")
        question_id = question["question_id"]
        if not isinstance(question_id, str) or not question_id or question_id in ids:
            raise BenchmarkIntegrityError(f"{name} question IDs are invalid or duplicated")
        ids.add(question_id)
    canonical_json(questions)


def load_benchmark(name: str, *, verify: bool = True) -> list[dict]:
    if name not in _LOADERS:
        raise BenchmarkIntegrityError(f"benchmark is not canonical in v2: {name}")
    questions = _LOADERS[name]()
    _validate_questions(name, questions)
    if verify:
        specification = load_registry()["benchmarks"][name]
        actual_hash = benchmark_sha256(questions)
        if len(questions) != specification["count"] or actual_hash != specification["sha256"]:
            raise BenchmarkIntegrityError(
                f"{name} normalized pool differs: count={len(questions)}, sha256={actual_hash}"
            )
    return questions


def load_pool(*, names: tuple[str, ...] = CANONICAL_BENCHMARKS) -> list[dict]:
    if len(names) != len(set(names)) or any(name not in CANONICAL_BENCHMARKS for name in names):
        raise BenchmarkIntegrityError("requested benchmark subset is invalid")
    pool = [question for name in names for question in load_benchmark(name)]
    if names == CANONICAL_BENCHMARKS:
        actual = benchmark_sha256(pool)
        declared = load_registry()["pool_sha256"]
        if actual != declared:
            raise BenchmarkIntegrityError("combined canonical benchmark pool hash differs")
    return pool
