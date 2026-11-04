"""Shared grades: one coldkey-median grade per (question, model) cell (I4, I1).

docs/INVARIANTS.md, "I4 — shared grades: a cell is a vote of operators".
"""
import random

import pytest

from fugal_subnet.shared_grades import (
    ScoredAnswer,
    cells_for_reveal,
    head_grades,
    shared_cell_grades,
)


def A(q, m, ck, correct, failed=False):
    return ScoredAnswer(question_id=q, model=m, coldkey=ck, correct=correct, failed=failed)


def test_one_coldkey_is_its_own_grade():
    assert shared_cell_grades([A("q1", "m", "ck1", True)]) == {("q1", "m"): 1.0}
    assert shared_cell_grades([A("q1", "m", "ck1", False)]) == {("q1", "m"): 0.0}


def test_a_minority_of_coldkeys_cannot_move_a_cell():
    honest = [A("q1", "m", f"h{i}", True) for i in range(3)]
    dissent = [A("q1", "m", f"x{i}", False) for i in range(2)]
    assert shared_cell_grades(honest + dissent)[("q1", "m")] == 1.0


def test_exactly_half_moves_a_cell_halfway():
    cells = shared_cell_grades([A("q1", "m", "a", True), A("q1", "m", "b", False)])
    assert cells[("q1", "m")] == 0.5


def test_hotkeys_under_one_coldkey_are_one_vote():
    # Ten hotkeys of one operator against two other operators: the ten are one
    # vote (their mean, 0.0), so the cell is the median of {0, 1, 1} = 1.
    many = [A("q1", "m", "farm", False) for _ in range(10)]
    others = [A("q1", "m", "a", True), A("q1", "m", "b", True)]
    assert shared_cell_grades(many + others)[("q1", "m")] == 1.0


def test_failed_calls_do_not_vote():
    cells = shared_cell_grades([
        A("q1", "m", "a", True),
        A("q1", "m", "b", False, failed=True),
        A("q1", "m", "c", False, failed=True),
    ])
    assert cells[("q1", "m")] == 1.0


def test_a_cell_with_only_failed_calls_has_no_grade():
    assert shared_cell_grades([A("q1", "m", "a", False, failed=True)]) == {}


def test_cells_are_per_question_and_model():
    cells = shared_cell_grades([
        A("q1", "m1", "a", True), A("q1", "m2", "a", False), A("q2", "m1", "a", False),
    ])
    assert cells == {("q1", "m1"): 1.0, ("q1", "m2"): 0.0, ("q2", "m1"): 0.0}


def test_two_heads_on_the_same_cell_get_the_same_grade():
    # The point of the change: a provider flip between two answers to one
    # (question, model) no longer separates two heads that routed alike.
    cells = shared_cell_grades([
        A("q1", "m", "a", True), A("q1", "m", "b", False), A("q1", "m", "c", True),
    ])
    ga = head_grades([("q1", "m", False)], cells)
    gb = head_grades([("q1", "m", False)], cells)
    assert ga == gb == {"q1": 1.0}


def test_a_heads_own_failed_call_stays_wrong_whatever_the_cell_says():
    cells = shared_cell_grades([
        A("q1", "m", "a", True), A("q1", "m", "b", True), A("q1", "m", "me", False, failed=True),
    ])
    assert cells[("q1", "m")] == 1.0
    assert head_grades([("q1", "m", True)], cells) == {"q1": 0.0}


def test_a_missing_cell_for_an_answered_question_raises():
    with pytest.raises(KeyError):
        head_grades([("q1", "m", False)], {})


def test_order_independent():
    rng = random.Random(7)
    answers = [
        A(f"q{rng.randrange(20)}", f"m{rng.randrange(4)}", f"ck{rng.randrange(6)}",
          rng.random() < 0.6, rng.random() < 0.05)
        for _ in range(400)
    ]
    base = shared_cell_grades(answers)
    for _ in range(5):
        rng.shuffle(answers)
        assert shared_cell_grades(answers) == base
    assert cells_for_reveal(base) == cells_for_reveal(dict(reversed(list(base.items()))))


def test_reveal_form_is_sorted_and_complete():
    cells = {("q2", "m"): 0.5, ("q1", "z"): 1.0, ("q1", "a"): 0.0}
    out = cells_for_reveal(cells)
    assert [(c["question_id"], c["model"]) for c in out] == [("q1", "a"), ("q1", "z"), ("q2", "m")]
    assert {(c["question_id"], c["model"]): c["grade"] for c in out} == cells


def _reveal(tmp_path, monkeypatch, **kw):
    import json

    import numpy as np

    from fugal_subnet import commit_reveal

    monkeypatch.setattr(commit_reveal, "EPOCH_DIR", str(tmp_path))
    questions = [{"question_id": "q1", "prompt": "p", "gold": "1", "grader_id": "x"}]
    commit_reveal.commit_epoch("e1", questions, "0xabc")
    assert commit_reveal.reveal_epoch(
        "e1", questions, np.array([[1]]), ["a"], {"a": 0.1}, {}, {}, {}, {}, **kw,
    )
    return json.loads((tmp_path / "e1" / "reveal.json").read_text())


def test_a_v1_reveal_carries_no_v2_fields(tmp_path, monkeypatch):
    revealed = _reveal(tmp_path, monkeypatch)
    assert "protocol_version" not in revealed
    assert "shared_grades" not in revealed


def test_a_v2_reveal_publishes_the_cells(tmp_path, monkeypatch):
    cells = cells_for_reveal(shared_cell_grades([
        A("q1", "a", "ck1", True), A("q1", "a", "ck2", False),
    ]))
    revealed = _reveal(tmp_path, monkeypatch, protocol_version=2, shared_grades=cells)
    assert revealed["protocol_version"] == 2
    assert revealed["shared_grades"] == [{"question_id": "q1", "model": "a", "grade": 0.5}]


def test_the_validator_computes_cells_only_under_v2():
    # Structural: shared grades are built behind the boundary-keyed version
    # and handed to the reveal, so v1 epochs are untouched.
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1].joinpath("neurons", "validator.py").read_text()
    guard = src.index("if protocol_version >= 2:\n                shared_cells = shared_cell_grades(")
    assert src.index("shared_cells = None") < guard
    assert "shared_grades=(\n                    cells_for_reveal(shared_cells)" in src
