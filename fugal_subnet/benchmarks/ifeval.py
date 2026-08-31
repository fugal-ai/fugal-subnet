"""IFEval benchmark loader — 541 instruction-following tasks."""
from __future__ import annotations
import json as _json
from datasets import load_dataset

from fugal_subnet.benchmarks.loader import DATASET_REVISIONS


def load() -> list[dict]:
    ds = load_dataset("google/IFEval", split="train",
                      revision=DATASET_REVISIONS["google/IFEval"])
    items = []
    for i, row in enumerate(ds):
        constraints = _build_constraints(row)
        if not constraints:
            continue
        items.append({
            "prompt": row["prompt"],
            "gold": "",
            "grader_id": "constraint_set",
            "benchmark": "ifeval",
            "question_id": f"ifeval_{i:04d}",
            "metadata": {
                "checker": {
                    "id": "constraint_set",
                    "params": {"constraints": constraints},
                },
            },
        })
    return items


def _build_constraints(row: dict) -> list[dict]:
    """Convert IFEval instruction_id_list + kwargs into constraint_set format."""
    ids = row.get("instruction_id_list", [])
    kwargs_list = row.get("kwargs", [])
    if isinstance(kwargs_list, str):
        try:
            kwargs_list = _json.loads(kwargs_list)
        except (_json.JSONDecodeError, TypeError):
            kwargs_list = [{}] * len(ids)

    constraints = []
    for inst_id, kwargs in zip(ids, kwargs_list):
        if kwargs is None:
            kwargs = {}
        c = _ifeval_to_constraint(inst_id, kwargs)
        if c:
            constraints.append(c)
    return constraints


def _ifeval_to_constraint(inst_id: str, kwargs: dict) -> dict | None:
    """Map IFEval instruction IDs to constraint_set constraint dicts.

    Only maps instructions that have direct equivalents in the grader's
    constraint_set checker. Unmappable instructions are skipped (returns None).
    """
    if "keywords:existence" in inst_id:
        kws = kwargs.get("keywords", [])
        if kws:
            return {"kind": "must_include", "s": kws[0], "k": 1}
    elif "keywords:forbidden_words" in inst_id:
        words = kwargs.get("forbidden_words", [])
        if words:
            return {"kind": "must_exclude", "s": words[0]}
    elif "length_constraints:number_words" in inst_id:
        rel = kwargs.get("relation", "")
        num = kwargs.get("num_words", 0)
        if rel == "at least":
            return {"kind": "word_count_range", "min": num, "max": 999999}
        elif rel == "at most":
            return {"kind": "word_count_range", "min": 0, "max": num}
    elif "change_case:english_lowercase" in inst_id:
        return {"kind": "lowercase_only"}
    elif "startend:end_checker" in inst_id:
        end_phrase = kwargs.get("end_phrase", "")
        if end_phrase:
            return {"kind": "ends_with", "s": end_phrase}
    elif "punctuation:no_comma" in inst_id:
        return {"kind": "must_exclude", "s": ","}
    elif "detectable_format:number_bullet_lists" in inst_id:
        return {"kind": "bullet_lines"}
    elif "detectable_format:json_format" in inst_id:
        return {"kind": "json_object", "required": {}}
    return None
