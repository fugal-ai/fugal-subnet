"""LiveCodeBench benchmark loader — competitive programming (fresh quarterly).

LiveCodeBench's HuggingFace dataset currently uses a legacy script format
that the datasets library no longer supports. This loader attempts to load
from a local JSON cache first, then falls back to HuggingFace. If neither
works, it returns an empty list with a warning — the benchmark pool still
has 17,000+ questions from the other 7 benchmarks.

To populate the local cache, download from the LiveCodeBench GitHub repo
and save to data/benchmarks/livecode.json as a list of dicts with keys:
    question_content, entry_point, test_list (list of {input, output} dicts)
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "benchmarks", "livecode.json",
)


def load() -> list[dict]:
    if os.path.exists(_CACHE_PATH):
        return _load_from_cache()
    return _load_from_hf()


def _load_from_cache() -> list[dict]:
    with open(_CACHE_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for i, row in enumerate(raw):
        question = row.get("question_content", "")
        if not question:
            continue
        func_name = row.get("entry_point", "")
        test_inputs, test_outputs = _parse_tests(row.get("test_list", []))
        qid = f"livecode_{i:04d}"
        if test_inputs and test_outputs and func_name:
            items.append({
                "prompt": question,
                "gold": test_outputs,
                "grader_id": "exec_io",
                "benchmark": "livecode",
                "question_id": qid,
                "metadata": {
                    "checker": {
                        "id": "exec_io",
                        "params": {"func": func_name, "inputs": test_inputs},
                    },
                },
            })
    return items


def _load_from_hf() -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset("livecodebench/code_generation_lite", split="test")
    except Exception as e:
        logger.warning(
            "LiveCodeBench not available (HF dataset uses unsupported script format). "
            "Download manually to %s. Error: %s", _CACHE_PATH, e
        )
        return []

    items = []
    for i, row in enumerate(ds):
        question = row.get("question_content", row.get("question", ""))
        if not question:
            continue
        func_name = row.get("entry_point", "")
        test_inputs, test_outputs = _parse_tests(row.get("test_list", []))
        qid = f"livecode_{i:04d}"
        if test_inputs and test_outputs and func_name:
            items.append({
                "prompt": question,
                "gold": test_outputs,
                "grader_id": "exec_io",
                "benchmark": "livecode",
                "question_id": qid,
                "metadata": {
                    "checker": {
                        "id": "exec_io",
                        "params": {"func": func_name, "inputs": test_inputs},
                    },
                },
            })
    return items


def _parse_tests(test_list) -> tuple[list, list]:
    if isinstance(test_list, str):
        try:
            test_list = json.loads(test_list)
        except (json.JSONDecodeError, TypeError):
            return [], []
    inputs, outputs = [], []
    for test in (test_list or []):
        if isinstance(test, dict):
            inp = test.get("input")
            out = test.get("output")
            if inp is not None and out is not None:
                inputs.append(inp if isinstance(inp, list) else [inp])
                outputs.append(out)
    return inputs, outputs
