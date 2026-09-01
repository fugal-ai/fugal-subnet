"""Deterministic strict adapter around the pinned official IFEval classes."""

from __future__ import annotations

from langdetect import DetectorFactory

from fugal_subnet.vendored.ifeval import instructions_registry

# langdetect otherwise initializes its detector with random feature sampling.
DetectorFactory.seed = 0


class IFEvalError(ValueError):
    """The canonical IFEval task is malformed or unsupported."""


def evaluate_strict(
    *,
    prompt: str,
    response: str,
    instruction_ids: list[str],
    kwargs: list[dict],
) -> tuple[bool, tuple[bool, ...]]:
    """Apply every official checker using the paper's strict evaluation rule.

    Hugging Face materializes every possible kwarg as a nullable field, while
    the official JSONL contains sparse kwargs.  Filtering null fields restores
    the official input contract. Missing required values fail closed instead
    of allowing an evaluator class to choose a random default.
    """
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise IFEvalError("prompt and response must be strings")
    if not isinstance(instruction_ids, list) or not instruction_ids:
        raise IFEvalError("instruction_ids must be a non-empty list")
    if not isinstance(kwargs, list) or len(kwargs) != len(instruction_ids):
        raise IFEvalError("instruction kwargs do not align")

    followed: list[bool] = []
    for instruction_id, raw_kwargs in zip(instruction_ids, kwargs):
        if instruction_id not in instructions_registry.INSTRUCTION_DICT:
            raise IFEvalError(f"unsupported instruction id: {instruction_id}")
        if not isinstance(raw_kwargs, dict):
            raise IFEvalError("instruction kwargs must be objects")
        instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        expected_keys = set(instruction.get_instruction_args_keys())
        supplied = {
            key: value
            for key, value in raw_kwargs.items()
            if key in expected_keys and value is not None
        }
        if set(supplied) != expected_keys:
            missing = sorted(expected_keys - set(supplied))
            raise IFEvalError(f"missing deterministic kwargs for {instruction_id}: {missing}")
        instruction.build_description(**supplied)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)
        followed.append(bool(response.strip() and instruction.check_following(response)))
    return all(followed), tuple(followed)
