"""Journal-backed, fail-closed v2 matrix construction."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Mapping

import numpy as np

from fugal_subnet.api import PROMPT_TOKEN_OVERHEAD, SpendTracker, call_model
from fugal_subnet.graders_v2 import grade
from fugal_subnet.sandbox.client import GradingClient
from fugal_subnet.v2.journal import EpochJournal, JournalError, cell_key

MAX_QUESTIONS = 512
MAX_MODELS = 8
MAX_RESPONSE_BYTES = 4096
MAX_OUTPUT_TOKENS = 1024
MAX_RETRIES = 3
MAX_CONCURRENCY = 32


class MatrixBuildAborted(RuntimeError):
    pass


@dataclass(frozen=True)
class CellResult:
    question_id: str
    model_id: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    grade: int
    actual_cost_usd: str
    reserved_cost_usd: str


@dataclass(frozen=True)
class MatrixResult:
    question_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    matrix: np.ndarray
    cells: tuple[CellResult, ...]
    budget_usd: str
    reserved_usd: str
    actual_usd: str


ResponseFunction = Callable[[str, dict, Decimal], tuple[str, int, int, Decimal]]


def mock_response_function(
    model_id: str, question: dict, _reserved: Decimal,
) -> tuple[str, int, int, Decimal]:
    """Deterministic zero-spend response source for local/testnet exercises."""
    del model_id
    grader_id = question.get("grader_id")
    if grader_id in {
        "numeric_decimal_v2", "integer_decimal_v2", "letter_mcq_v2",
        "symbolic_math_v2",
    }:
        response = str(question.get("gold", ""))
    elif grader_id == "ifeval_strict_v2":
        response = str(question.get("prompt", ""))
    else:
        # Code tasks deliberately grade zero in mock mode; gold outputs never
        # need to be converted into or exposed as candidate source code.
        response = ""
    return response, 0, 0, Decimal(0)


def reservation_cost(
    prompt: str,
    model_id: str,
    spend_prices: Mapping[str, tuple[Decimal, Decimal]],
    *,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    retries: int = MAX_RETRIES,
) -> Decimal:
    if model_id not in spend_prices:
        raise MatrixBuildAborted(f"live spend price unavailable for {model_id}")
    input_price, output_price = spend_prices[model_id]
    if input_price < 0 or output_price < 0:
        raise MatrixBuildAborted(f"live spend price is invalid for {model_id}")
    one_attempt = (
        Decimal(len(prompt.encode("utf-8")) + PROMPT_TOKEN_OVERHEAD) * input_price
        + Decimal(max_output_tokens) * output_price
    )
    return one_attempt * retries


def openrouter_response_function(
    spend_prices: Mapping[str, tuple[Decimal, Decimal]],
) -> ResponseFunction:
    """Build the explicitly live paid cell caller used only by a v2 builder."""
    float_prices = {
        model: (float(prices[0]), float(prices[1]))
        for model, prices in spend_prices.items()
    }

    def response(model_id: str, question: dict, reserved: Decimal):
        tracker = SpendTracker(budget_cap_usd=float(reserved))
        # [PAID ~$0-$0.10/call] This function is reachable only from an
        # explicit --live v2 orchestrator after the journal reservation lands.
        text, prompt_tokens, completion_tokens = call_model(
            model_id,
            question["prompt"],
            max_tokens=MAX_OUTPUT_TOKENS,
            retries=MAX_RETRIES,
            tracker=tracker,
            prices=float_prices,
            live=True,
        )
        return text, prompt_tokens, completion_tokens, Decimal(str(tracker.total_cost_usd))

    return response


def _validate_inputs(questions: list[dict], model_ids: list[str]) -> None:
    if not 1 <= len(questions) <= MAX_QUESTIONS or not 1 <= len(model_ids) <= MAX_MODELS:
        raise MatrixBuildAborted("matrix dimensions exceed v2 bounds")
    question_ids = [item.get("question_id") if isinstance(item, dict) else None for item in questions]
    if any(not isinstance(value, str) or not value for value in question_ids):
        raise MatrixBuildAborted("matrix questions need non-empty IDs")
    if len(question_ids) != len(set(question_ids)) or len(model_ids) != len(set(model_ids)):
        raise MatrixBuildAborted("matrix question/model IDs must be unique")


def build_matrix(
    questions: list[dict],
    model_ids: list[str],
    *,
    journal: EpochJournal,
    response_function: ResponseFunction,
    spend_prices: Mapping[str, tuple[Decimal, Decimal]],
    grading_client: GradingClient | None,
    concurrency: int = 4,
) -> MatrixResult:
    """Reserve the entire matrix first; resume completed cells without repeats."""
    _validate_inputs(questions, model_ids)
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or not 1 <= concurrency <= MAX_CONCURRENCY:
        raise MatrixBuildAborted("matrix concurrency is outside the v2 bound")
    state = journal.read()
    if state["status"] != "active":
        raise MatrixBuildAborted(f"epoch journal is terminal: {state['status']}")
    inflight = [cell for cell in state["cells"].values() if cell["status"] == "reserved"]
    if inflight:
        for cell in inflight:
            journal.forfeit_cell(cell["question_id"], cell["model_id"])
        journal.abort("restart found ambiguous in-flight paid cells; reservations forfeited")
        raise MatrixBuildAborted("ambiguous in-flight cells were forfeited; epoch aborted")
    if any(cell["status"] == "forfeited" for cell in state["cells"].values()):
        journal.abort("forfeited paid cell prevents matrix consensus")
        raise MatrixBuildAborted("forfeited cell prevents matrix consensus")

    planned: list[tuple[dict, str, Decimal]] = []
    for question in questions:
        for model_id in model_ids:
            key = cell_key(question["question_id"], model_id)
            if key in state["cells"]:
                continue
            planned.append((
                question,
                model_id,
                reservation_cost(question["prompt"], model_id, spend_prices),
            ))
    available = Decimal(state["spend"]["budget_usd"]) - Decimal(state["spend"]["actual_usd"])
    if sum((item[2] for item in planned), Decimal(0)) > available:
        journal.abort("matrix worst-case reservations exceed the remaining epoch budget")
        raise MatrixBuildAborted("matrix worst-case reservations exceed epoch budget")

    try:
        journal.advance_phase("matrix_build")
        for question, model_id, reserved in planned:
            journal.reserve_cell(question["question_id"], model_id, reserved)
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending = {
                executor.submit(response_function, model_id, question, reserved): (
                    question, model_id,
                )
                for question, model_id, reserved in planned
            }
            for future in concurrent.futures.as_completed(pending):
                question, model_id = pending[future]
                try:
                    response, prompt_tokens, completion_tokens, actual = future.result()
                    if len(response.encode("utf-8")) > MAX_RESPONSE_BYTES:
                        raise MatrixBuildAborted(
                            "model response exceeds v2 published response bound"
                        )
                    journal.complete_cell(
                        question["question_id"],
                        model_id,
                        response,
                        prompt_tokens,
                        completion_tokens,
                        actual,
                    )
                except Exception as exc:
                    journal.forfeit_cell(question["question_id"], model_id)
                    failures.append(exc)
        if failures:
            journal.abort(
                f"matrix cell failed closed: {type(failures[0]).__name__}"
            )
            raise MatrixBuildAborted(
                "matrix response collection failed; epoch aborted"
            ) from failures[0]
    except JournalError as exc:
        try:
            if journal.read()["status"] == "active":
                journal.abort(f"journal failure during matrix build: {type(exc).__name__}")
        except JournalError:
            pass
        raise MatrixBuildAborted("epoch journal rejected matrix progress") from exc

    final = journal.read()
    cells = []
    rows = []
    try:
        for question in questions:
            row = []
            for model_id in model_ids:
                raw = final["cells"].get(cell_key(question["question_id"], model_id))
                if raw is None or raw["status"] != "complete":
                    raise MatrixBuildAborted("matrix journal is incomplete")
                value = grade(question, raw["response_text"], grading_client)
                row.append(value)
                cells.append(CellResult(
                    question_id=question["question_id"],
                    model_id=model_id,
                    response=raw["response_text"],
                    prompt_tokens=raw["prompt_tokens"],
                    completion_tokens=raw["completion_tokens"],
                    grade=value,
                    actual_cost_usd=raw["actual_cost_usd"],
                    reserved_cost_usd=raw["reserved_cost_usd"],
                ))
            rows.append(row)
    except Exception as exc:
        if journal.read()["status"] == "active":
            journal.abort(f"isolated grading failed closed: {type(exc).__name__}")
        if isinstance(exc, MatrixBuildAborted):
            raise
        raise MatrixBuildAborted("isolated grading failed; epoch aborted") from exc
    return MatrixResult(
        question_ids=tuple(question["question_id"] for question in questions),
        model_ids=tuple(model_ids),
        matrix=np.asarray(rows, dtype=np.int8),
        cells=tuple(cells),
        budget_usd=final["spend"]["budget_usd"],
        reserved_usd=final["spend"]["reserved_usd"],
        actual_usd=final["spend"]["actual_usd"],
    )
