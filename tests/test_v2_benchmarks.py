"""Packaged v2 benchmark contract tests that do not require network access."""

from __future__ import annotations

import hashlib
import importlib.resources
import json

from fugal_subnet.v2.benchmarks import (
    CANONICAL_BENCHMARKS,
    benchmark_sha256,
    load_benchmark,
    load_registry,
)
from fugal_subnet.vendored.ifeval import instructions_registry


def test_registry_has_only_fixed_public_v2_membership():
    registry = load_registry()
    assert tuple(sorted(registry["benchmarks"])) == CANONICAL_BENCHMARKS
    assert "gpqa" not in registry["benchmarks"]
    assert "livecode" not in registry["benchmarks"]
    for spec in registry["benchmarks"].values():
        assert spec["count"] > 0
        assert len(spec["sha256"]) == 64
        assert len(spec["revision"]) == 40


def test_packaged_humaneval_has_eight_or_more_exact_json_cases_per_task():
    raw = importlib.resources.files("fugal_subnet").joinpath(
        "human-eval-cases-v2.json"
    ).read_bytes()
    payload = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == "fe780fed2362111388843a60b3fe46382c4c27154565bb5d433a9c7369d45a8e"
    assert len(payload["tasks"]) == 43
    assert all(len(item["cases"]) >= 8 for item in payload["tasks"])
    assert all("test" not in item and "canonical_solution" not in item for item in payload["tasks"])


def test_humaneval_normalization_matches_registry_without_network():
    questions = load_benchmark("humaneval")
    specification = load_registry()["benchmarks"]["humaneval"]
    assert len(questions) == specification["count"] == 43
    assert benchmark_sha256(questions) == specification["sha256"]
    assert all(question["grader_id"] == "code_io_v2" for question in questions)
    assert all(len(question["metadata"]["inputs"]) >= 8 for question in questions)


def test_vendored_ifeval_registry_covers_all_pinned_dataset_instruction_ids():
    expected = {
        "change_case:capital_word_frequency", "change_case:english_capital",
        "change_case:english_lowercase", "combination:repeat_prompt",
        "combination:two_responses", "detectable_content:number_placeholders",
        "detectable_content:postscript", "detectable_format:constrained_response",
        "detectable_format:json_format", "detectable_format:multiple_sections",
        "detectable_format:number_bullet_lists",
        "detectable_format:number_highlighted_sections", "detectable_format:title",
        "keywords:existence", "keywords:forbidden_words", "keywords:frequency",
        "keywords:letter_frequency", "language:response_language",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_paragraphs",
        "length_constraints:number_sentences", "length_constraints:number_words",
        "punctuation:no_comma", "startend:end_checker", "startend:quotation",
    }
    assert expected <= set(instructions_registry.INSTRUCTION_DICT)
