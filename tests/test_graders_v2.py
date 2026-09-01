"""Consensus regressions for corrected v2 graders."""

from __future__ import annotations

import pytest

from fugal_subnet import graders_v2
from fugal_subnet.graders_v2 import GraderContractError
from fugal_subnet.sandbox.client import GradingUnavailable


def task(grader_id: str, gold=None, **extra):
    return {
        "prompt": extra.pop("prompt", "prompt"),
        "gold": gold,
        "grader_id": grader_id,
        "benchmark": extra.pop("benchmark", "test"),
        "question_id": extra.pop("question_id", "test_1"),
        "metadata": extra.pop("metadata", {}),
        **extra,
    }


def test_exact_decimal_integer_accepts_integral_forms_only():
    item = task("integer_decimal_v2", "42")
    assert graders_v2.grade(item, "42") == 1
    assert graders_v2.grade(item, "The answer is 42.0") == 1
    assert graders_v2.grade(item, "4.2e1") == 1
    assert graders_v2.grade(item, "42.9") == 0
    assert graders_v2.grade(item, "41.999999999999999") == 0


def test_numeric_decimal_comparison_does_not_round_through_float():
    item = task("numeric_decimal_v2", "9007199254740993")
    assert graders_v2.grade(item, "9007199254740993") == 1
    assert graders_v2.grade(item, "9007199254740992") == 0


def test_official_ifeval_json_accepts_general_valid_json():
    item = task(
        "ifeval_strict_v2",
        metadata={"instruction_ids": ["detectable_format:json_format"], "kwargs": [{}]},
    )
    assert graders_v2.grade(item, '{"answer": [1, 2], "ok": true}') == 1
    assert graders_v2.grade(item, "```json\n[1, 2, 3]\n```") == 1
    assert graders_v2.grade(item, "not JSON") == 0


def test_official_ifeval_preserves_all_keywords_and_compound_rules():
    item = task(
        "ifeval_strict_v2",
        metadata={
            "instruction_ids": [
                "keywords:existence",
                "keywords:forbidden_words",
                "detectable_format:number_bullet_lists",
            ],
            "kwargs": [
                {"keywords": ["alpha", "beta"]},
                {"forbidden_words": ["gamma", "delta"]},
                {"num_bullets": 2},
            ],
        },
    )
    assert graders_v2.grade(item, "- alpha\n- beta") == 1
    assert graders_v2.grade(item, "- alpha\n- missing") == 0
    assert graders_v2.grade(item, "- alpha beta gamma\n- clean") == 0
    assert graders_v2.grade(item, "- alpha\n- beta\n- third") == 0


def test_ifeval_letter_frequency_preserves_literal_punctuation():
    item = task(
        "ifeval_strict_v2",
        metadata={
            "instruction_ids": ["keywords:letter_frequency"],
            "kwargs": [
                {"letter": "#", "let_frequency": 4, "let_relation": "at least"}
            ],
        },
    )
    assert graders_v2.grade(item, "#one #two #three #four") == 1
    assert graders_v2.grade(item, "#one #two #three") == 0


def test_ifeval_missing_deterministic_kwargs_aborts_contract():
    item = task(
        "ifeval_strict_v2",
        metadata={
            "instruction_ids": ["detectable_format:number_bullet_lists"],
            "kwargs": [{}],
        },
    )
    with pytest.raises(GraderContractError, match="IFEval"):
        graders_v2.grade(item, "- one")


class FakeClient:
    def __init__(self, value=True):
        self.value = value
        self.code = None

    def grade_code(self, **kwargs):
        self.code = kwargs
        return self.value

    def grade_symbolic_math(self, **kwargs):
        self.math = kwargs
        return self.value


def test_code_grader_strips_fence_and_delegates_without_same_process_exec():
    client = FakeClient()
    item = task(
        "code_io_v2",
        [3],
        metadata={"function": "add", "inputs": [[1, 2]]},
    )
    assert graders_v2.grade(item, "```python\ndef add(a, b): return a + b\n```", client) == 1
    assert client.code["code"] == "def add(a, b): return a + b"
    assert client.code["expected"] == [3]


def test_worker_unavailability_propagates_and_cannot_grade_zero_silently():
    class Missing(FakeClient):
        def grade_symbolic_math(self, **kwargs):
            raise GradingUnavailable("offline")

    item = task("symbolic_math_v2", "1")
    with pytest.raises(GradingUnavailable, match="offline"):
        graders_v2.grade(item, "\\boxed{1}", Missing())


def test_unknown_grader_aborts_contract():
    with pytest.raises(GraderContractError, match="unknown"):
        graders_v2.grade(task("made_up"), "anything")
