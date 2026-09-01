#!/usr/bin/env python3
"""Immutable-candidate v2 grading rules; semantic changes require a new version."""

from __future__ import annotations

import hashlib
import importlib.resources
import re
from decimal import Decimal, InvalidOperation

from fugal_subnet.sandbox.client import GradingClient
from fugal_subnet.vendored.ifeval.evaluator import IFEvalError, evaluate_strict

_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_GRADER_BUNDLE_RESOURCES = (
    "graders_v2.py",
    "sandbox/client.py",
    "sandbox/launcher.py",
    "sandbox/protocol.py",
    "vendored/ifeval/evaluator.py",
    "vendored/ifeval/instructions.py",
    "vendored/ifeval/instructions_registry.py",
    "vendored/ifeval/instructions_util.py",
    "vendored/ifeval/punkt-english.tar.gz.b64",
)


class GraderContractError(RuntimeError):
    """Canonical task material is malformed; the epoch must abort."""


def grader_hash() -> str:
    """Hash every packaged executable/resource byte that defines v2 grading."""
    package_root = importlib.resources.files("fugal_subnet")
    digest = hashlib.sha256(b"fugal-grader-bundle-v2\0")
    for name in _GRADER_BUNDLE_RESOURCES:
        payload = package_root.joinpath(name).read_bytes()
        if name.endswith((".py", ".b64")):
            payload = payload.replace(b"\r\n", b"\n")
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def _decimal_answer(text: str) -> Decimal | None:
    normalized = (text or "").replace(",", "").replace("−", "-")
    matches = _NUMBER.findall(normalized)
    if not matches:
        return None
    try:
        value = Decimal(matches[-1].rstrip("."))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def _gold_decimal(task: dict) -> Decimal:
    try:
        value = Decimal(str(task["gold"]))
    except (KeyError, InvalidOperation, ValueError) as exc:
        raise GraderContractError("numeric task gold is invalid") from exc
    if not value.is_finite():
        raise GraderContractError("numeric task gold must be finite")
    return value


def numeric_decimal(task: dict, reply: str, _client: GradingClient | None) -> int:
    answer = _decimal_answer(reply)
    return int(answer is not None and answer == _gold_decimal(task))


def integer_decimal(task: dict, reply: str, _client: GradingClient | None) -> int:
    answer = _decimal_answer(reply)
    gold = _gold_decimal(task)
    if gold != gold.to_integral_value():
        raise GraderContractError("integer task gold is not integral")
    return int(
        answer is not None
        and answer == answer.to_integral_value()
        and answer == gold
    )


def letter_mcq(task: dict, reply: str, _client: GradingClient | None) -> int:
    try:
        gold = task["gold"]
    except KeyError as exc:
        raise GraderContractError("multiple-choice task gold is missing") from exc
    if not isinstance(gold, str) or not re.fullmatch(r"[A-J]", gold):
        raise GraderContractError("multiple-choice task gold is invalid")
    matches = re.findall(
        r"answer is\s*:?\s*[\*_\(]*([A-J])[\*_\)\.]*",
        reply or "",
        re.IGNORECASE,
    )
    if not matches:
        matches = re.findall(r"\b([A-J])\b", (reply or "").strip()[-40:])
    return int(bool(matches) and matches[-1].upper() == gold)


def _require_client(client: GradingClient | None) -> GradingClient:
    if client is None:
        raise GraderContractError("v2 isolated grading client is required")
    return client


def symbolic_math(task: dict, reply: str, client: GradingClient | None) -> int:
    try:
        gold = task["gold"]
        question_id = task["question_id"]
    except KeyError as exc:
        raise GraderContractError("symbolic task material is incomplete") from exc
    if not isinstance(gold, str) or not isinstance(question_id, str):
        raise GraderContractError("symbolic task material is invalid")
    return int(
        _require_client(client).grade_symbolic_math(
            gold=gold,
            reply=reply,
            job_id=f"math-{hashlib.sha256(question_id.encode()).hexdigest()[:24]}",
        )
    )


def _extract_code(reply: str) -> str:
    blocks = _CODE_BLOCK.findall(reply or "")
    return max(blocks, key=len).strip() if blocks else (reply or "").strip()


def code_io(task: dict, reply: str, client: GradingClient | None) -> int:
    try:
        metadata = task["metadata"]
        function = metadata["function"]
        inputs = metadata["inputs"]
        expected = task["gold"]
        question_id = task["question_id"]
    except (KeyError, TypeError) as exc:
        raise GraderContractError("code task material is incomplete") from exc
    return int(
        _require_client(client).grade_code(
            code=_extract_code(reply),
            function=function,
            inputs=inputs,
            expected=expected,
            job_id=f"code-{hashlib.sha256(question_id.encode()).hexdigest()[:24]}",
        )
    )


def ifeval_strict(task: dict, reply: str, _client: GradingClient | None) -> int:
    try:
        metadata = task["metadata"]
        passed, _ = evaluate_strict(
            prompt=task["prompt"],
            response=reply,
            instruction_ids=metadata["instruction_ids"],
            kwargs=metadata["kwargs"],
        )
    except (KeyError, TypeError, IFEvalError) as exc:
        raise GraderContractError("IFEval task material is invalid") from exc
    return int(passed)


CHECKERS = {
    "numeric_decimal_v2": numeric_decimal,
    "integer_decimal_v2": integer_decimal,
    "letter_mcq_v2": letter_mcq,
    "symbolic_math_v2": symbolic_math,
    "code_io_v2": code_io,
    "ifeval_strict_v2": ifeval_strict,
}


def grade(task: dict, reply: str, client: GradingClient | None = None) -> int:
    try:
        checker_id = task["grader_id"]
    except (KeyError, TypeError) as exc:
        raise GraderContractError("task grader_id is missing") from exc
    checker = CHECKERS.get(checker_id)
    if checker is None:
        raise GraderContractError(f"unknown v2 grader: {checker_id}")
    return int(checker(task, reply or "", client))
