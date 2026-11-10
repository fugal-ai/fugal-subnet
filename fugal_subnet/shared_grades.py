"""Shared grades: one grade per (question, model) cell of an epoch (protocol v2).

WHY. From protocol v2 a challenger is compared with the champion question by
question (the paired net-value test). Two heads that route a question to the
same model are then asking the same thing of the same model, and any
difference between their grades on it is not routing: it is the provider
answering differently on the second call. Measured on mainnet reveals, two
independent answers to the same (question, model) disagree on about 8% of
cells. Grading both heads with one shared grade per cell removes that noise
from every question where they agree, which is most of them, and grading
every head on a cell with the median over operators removes most of the rest.
Measured on the same reveals, the paired comparison's variance falls to 0.48
of the unpaired one with each head's own answers, and to 0.23 with shared
cell grades.

WHAT. Per (question_id, model) in the scored slice:
  1. drop failed calls: a call that did not complete says nothing about the
     model (the same rule as the reference frame, I9);
  2. one vote per coldkey: the mean over that operator's hotkeys;
  3. the cell's grade is the MEDIAN across coldkeys, in [0, 1].
A head's grade on a question is the grade of the cell it routed to, EXCEPT
that its own failed call stays a wrong answer (0.0): a miner whose call
failed did not answer, and must not collect the answer other miners paid for.

THE ASSUMPTION (I4), stated plainly. A head's grade now depends on other
operators' answers to the same model on the same question. Those answers are
attested model output graded by the pinned grader, so what other operators
contribute is a measurement of the model, not a judgement of the head. A
cell's median moves only when at least half its voting coldkeys answer
differently; with exactly half it moves halfway. A cell answered by one
coldkey is that coldkey's own answer. Coldkeys, not hotkeys, are the unit,
so extra hotkeys under one operator are one vote.

Deterministic: every step is order-independent and the output is keyed, not
ordered, so validators that see proofs in any order agree (I1).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class ScoredAnswer:
    """One scored result from one verified proof, with the operator who sent it."""

    question_id: str
    model: str
    coldkey: str
    correct: bool
    failed: bool
    # Attested completion tokens of the call (hash-bound in the proof). Used
    # for the cell's spend, aggregated exactly like the grade.
    completion_tokens: int = 0


Cell = tuple[str, str]  # (question_id, model)


def shared_cell_grades(answers: Iterable[ScoredAnswer]) -> dict[Cell, float]:
    """The coldkey-median grade of every (question, model) cell with an answer.

    A cell whose every call failed has no grade and is absent from the result.
    """
    by_cell: dict[Cell, dict[str, list[bool]]] = {}
    for a in answers:
        if a.failed:
            continue
        by_cell.setdefault((a.question_id, a.model), {}).setdefault(a.coldkey, []).append(
            bool(a.correct)
        )
    return {
        cell: float(median(sum(v) / len(v) for _, v in sorted(votes.items())))
        for cell, votes in by_cell.items()
    }


def shared_cell_tokens(answers: Iterable[ScoredAnswer]) -> dict[Cell, float]:
    """The coldkey-median completion tokens of every (question, model) cell.

    Aggregated exactly like the grade: failed calls dropped, one vote per
    coldkey (the mean over its hotkeys), the median across coldkeys. From
    protocol v2 a head's spend on a question is the pool's prompt tokens and
    this, at the pinned prices: measured and shared, never read from the
    frame, and the same for every head that routed there (I4).
    """
    by_cell: dict[Cell, dict[str, list[int]]] = {}
    for a in answers:
        if a.failed:
            continue
        by_cell.setdefault((a.question_id, a.model), {}).setdefault(a.coldkey, []).append(
            int(a.completion_tokens)
        )
    return {
        cell: float(median(sum(v) / len(v) for _, v in sorted(votes.items())))
        for cell, votes in by_cell.items()
    }


def head_grades(
    results: Iterable[tuple[str, str, bool]],
    cells: Mapping[Cell, float],
) -> dict[str, float]:
    """A head's grade on each question it answered, from the shared cells.

    `results` are (question_id, routed_model, failed) for one head's scored
    answers. A failed call is 0.0. A non-failed answer always has a cell,
    because its own vote is in it; a missing cell is a caller error and
    raises rather than guessing a grade.
    """
    out: dict[str, float] = {}
    for qid, model, failed in results:
        if failed:
            out[qid] = 0.0
            continue
        try:
            out[qid] = cells[(qid, model)]
        except KeyError:
            raise KeyError(
                f"no shared grade for ({qid!r}, {model!r}); the head's own answer "
                "should be in it"
            ) from None
    return out


def cells_for_reveal(
    cells: Mapping[Cell, float], tokens: Mapping[Cell, float] | None = None,
) -> list[dict]:
    """The cells as a sorted list, for the published reveal.

    Published so anyone can recompute every head's v2 grades, and with the
    token medians every head's v2 spend, from the reveal alone, and so a
    rebuild from reveals reproduces them exactly.
    """
    out = []
    for (q, m), g in sorted(cells.items()):
        entry = {"question_id": q, "model": m, "grade": g}
        if tokens is not None:
            entry["completion_tokens"] = tokens.get((q, m), 0.0)
        out.append(entry)
    return out
