from __future__ import annotations

import threading
import time
from decimal import Decimal

import pytest

from fugal_subnet.v2.journal import EpochJournal
from fugal_subnet.v2.matrix import (
    MatrixBuildAborted,
    build_matrix,
    mock_response_function,
)

QUESTIONS = [
    {"prompt": "one", "gold": "1", "grader_id": "numeric_decimal_v2", "benchmark": "test", "question_id": "q1", "metadata": {}},
    {"prompt": "two", "gold": "2", "grader_id": "numeric_decimal_v2", "benchmark": "test", "question_id": "q2", "metadata": {}},
]
MODELS = ["provider/a", "provider/b"]
PRICES = {model: (Decimal("0.000001"), Decimal("0.000002")) for model in MODELS}


def _journal(tmp_path, budget="100"):
    journal = EpochJournal(tmp_path, "v2-100")
    journal.initialize(
        manifest_hash="a" * 64,
        boundary_block=100,
        boundary_hash="b" * 64,
        budget_usd=budget,
    )
    return journal


def test_matrix_reserves_before_calls_and_grades_in_canonical_order(tmp_path):
    journal = _journal(tmp_path)
    calls = []

    def response(model, question, reserved):
        state = journal.read()
        assert sum(cell["status"] == "reserved" for cell in state["cells"].values()) >= 1
        calls.append((question["question_id"], model, reserved))
        return question["gold"], 3, 1, Decimal("0.00001")

    result = build_matrix(
        QUESTIONS, MODELS, journal=journal, response_function=response,
        spend_prices=PRICES, grading_client=None,
    )
    assert result.matrix.tolist() == [[1, 1], [1, 1]]
    assert [(cell.question_id, cell.model_id) for cell in result.cells] == [
        ("q1", "provider/a"), ("q1", "provider/b"),
        ("q2", "provider/a"), ("q2", "provider/b"),
    ]
    assert len(calls) == 4


def test_completed_cells_resume_without_repeating_calls(tmp_path):
    journal = _journal(tmp_path)
    calls = []

    def response(model, question, reserved):
        calls.append((question["question_id"], model))
        return question["gold"], 1, 1, Decimal("0")

    first = build_matrix(
        QUESTIONS, MODELS, journal=journal, response_function=response,
        spend_prices=PRICES, grading_client=None,
    )
    second = build_matrix(
        QUESTIONS, MODELS, journal=journal, response_function=response,
        spend_prices=PRICES, grading_client=None,
    )
    assert first.matrix.tolist() == second.matrix.tolist()
    assert len(calls) == 4


def test_mock_response_function_is_zero_spend_and_deterministic(tmp_path):
    journal = _journal(tmp_path, budget="0")
    zero_prices = {model: (Decimal(0), Decimal(0)) for model in MODELS}
    result = build_matrix(
        QUESTIONS,
        MODELS,
        journal=journal,
        response_function=mock_response_function,
        spend_prices=zero_prices,
        grading_client=None,
    )
    assert result.matrix.tolist() == [[1, 1], [1, 1]]
    assert result.actual_usd == "0"


def test_matrix_collection_uses_only_bounded_concurrency(tmp_path):
    journal = _journal(tmp_path)
    lock = threading.Lock()
    active = 0
    peak = 0

    def response(model, question, reserved):
        del model, reserved
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return question["gold"], 1, 1, Decimal("0")

    result = build_matrix(
        QUESTIONS,
        MODELS,
        journal=journal,
        response_function=response,
        spend_prices=PRICES,
        grading_client=None,
        concurrency=2,
    )
    assert result.matrix.tolist() == [[1, 1], [1, 1]]
    assert 1 < peak <= 2


def test_matrix_rejects_unbounded_concurrency_before_work(tmp_path):
    with pytest.raises(MatrixBuildAborted, match="concurrency"):
        build_matrix(
            QUESTIONS,
            MODELS,
            journal=_journal(tmp_path),
            response_function=mock_response_function,
            spend_prices=PRICES,
            grading_client=None,
            concurrency=0,
        )


def test_budget_preflight_aborts_before_any_call(tmp_path):
    journal = _journal(tmp_path, budget="0.000001")
    called = False

    def response(model, question, reserved):
        nonlocal called
        called = True
        raise AssertionError("must not be called")

    with pytest.raises(MatrixBuildAborted, match="budget"):
        build_matrix(
            QUESTIONS, MODELS, journal=journal, response_function=response,
            spend_prices=PRICES, grading_client=None,
        )
    assert called is False
    assert journal.read()["status"] == "aborted"


def test_ambiguous_restart_forfeits_and_never_repeats_cell(tmp_path):
    journal = _journal(tmp_path)
    journal.reserve_cell("q1", "provider/a", "0.01")
    with pytest.raises(MatrixBuildAborted, match="in-flight"):
        build_matrix(
            QUESTIONS, MODELS, journal=journal,
            response_function=lambda *_: (_ for _ in ()).throw(AssertionError()),
            spend_prices=PRICES,
            grading_client=None,
        )
    state = journal.read()
    assert state["status"] == "aborted"
    assert next(iter(state["cells"].values()))["status"] == "forfeited"


def test_call_failure_forfeits_reservation_and_aborts(tmp_path):
    journal = _journal(tmp_path)
    with pytest.raises(MatrixBuildAborted, match="response collection"):
        build_matrix(
            QUESTIONS, MODELS, journal=journal,
            response_function=lambda *_: (_ for _ in ()).throw(TimeoutError()),
            spend_prices=PRICES,
            grading_client=None,
        )
    state = journal.read()
    assert state["status"] == "aborted"
    assert any(cell["status"] == "forfeited" for cell in state["cells"].values())


def test_isolated_grader_unavailability_aborts_after_collection(tmp_path):
    journal = _journal(tmp_path)
    code_question = [{
        "prompt": "write f", "gold": [1], "grader_id": "code_io_v2",
        "benchmark": "humaneval", "question_id": "code1",
        "metadata": {"function": "f", "inputs": [[]]},
    }]
    with pytest.raises(MatrixBuildAborted, match="grading failed"):
        build_matrix(
            code_question, ["provider/a"], journal=journal,
            response_function=lambda *_: ("def f(): return 1", 1, 1, Decimal("0")),
            spend_prices=PRICES,
            grading_client=None,
        )
    assert journal.read()["status"] == "aborted"
