"""Diligence 2026-10, finding 2: a miner coalition drags down the reference frame.

Three facts combine. The harness runs exploration AFTER the scored pass, so a
miner who controls its VM's network can drop connections during exploration
only. A failed call is recorded exactly like a wrong answer. And every miner
answers the same 15 exploration questions, each copy counted as a separate
sample, so a coalition's pull on the frame equals its share of miners.
Reproduced with the repo's own `build_frontier` and `composite`:

    colluding share   frontier peak   constant policies paid   best score
             0%           0.898              0 of 17             0.000
            10%           0.809              9 of 17             0.086
            20%           0.719             12 of 17             0.175
            33%           0.603             15 of 17             0.292

The required behaviour, pinned here and delivered by PR 2:
  * a failed model call is RECORDED as failed in the attested proof, never as a
    wrong answer, and never enters the frame;
  * a proof whose exploration fails far more often than its scored pass is
    rejected, so blocking exploration is not free;
  * the frame aggregates one observation per (question, model) per epoch with
    one vote per coldkey and a median across coldkeys, so a minority coalition
    cannot move it.

Delivered by `reference_frame.aggregate_observations`, `QuestionResult.failed`
and the 9b/9c checks in `verify_proof`; see docs/INVARIANTS.md § I9.
"""
from __future__ import annotations

import hashlib

import pytest

from fugal_subnet.tee.proof import QuestionResult
from tests.frame_fixtures import MODELS, TOP, TRUE_ACC, frame_from_truth, frontier_of


def _obs(qid, model, correct, coldkey, failed=False):
    from fugal_subnet.reference_frame import Observation

    return Observation(question_id=qid, model=model, correct=correct, prompt_tokens=100,
                       completion_tokens=300, failed=failed, coldkey=coldkey)


def test_failed_calls_are_a_proof_field_and_never_a_wrong_answer():
    """`QuestionResult.failed` exists, defaults False, and is hash-bound."""
    base = QuestionResult("q", TOP, False, 0.0, hashlib.sha256(b"").hexdigest())
    failed = QuestionResult("q", TOP, False, 0.0, hashlib.sha256(b"").hexdigest(),
                            failed=True)
    assert base.failed is False
    assert failed.failed is True
    assert failed.to_dict()["failed"] is True
    assert QuestionResult.from_dict(failed.to_dict()).failed is True


def test_failed_calls_do_not_enter_the_frame():
    from fugal_subnet.reference_frame import aggregate_observations

    only_failures = [_obs("q1", TOP, False, f"ck{i}", failed=True) for i in range(10)]
    assert aggregate_observations(only_failures) == []

    mixed = [_obs("q1", TOP, True, "ck0"), _obs("q1", TOP, False, "ck1", failed=True)]
    (model, correct, _p, _c), = aggregate_observations(mixed)
    assert model == TOP and correct == pytest.approx(1.0), (
        "a failed call pulled the observation down as if it were a wrong answer"
    )


def test_a_minority_coalition_cannot_move_a_question_observation():
    """One vote per coldkey, median across coldkeys."""
    from fugal_subnet.reference_frame import aggregate_observations

    honest = [_obs("q1", TOP, True, f"honest{i}") for i in range(6)]
    # Four hotkeys under ONE coldkey all answering wrong: a sybil, one vote.
    sybil = [_obs("q1", TOP, False, "attacker") for _ in range(4)]
    (_m, correct, _p, _c), = aggregate_observations(honest + sybil)
    assert correct == pytest.approx(1.0)

    # Three independent colluding coldkeys against six honest: still a minority.
    coalition = [_obs("q1", TOP, False, f"bad{i}") for i in range(3)]
    (_m, correct, _p, _c), = aggregate_observations(honest + coalition)
    assert correct == pytest.approx(1.0)


def test_coalition_at_a_third_of_the_field_does_not_lower_the_frontier():
    """End to end: the reproduction table, after the fix, reads 0 of 17 paid."""
    from fugal_subnet.reference_frame import (
        ReferenceFrame,
        accumulate_exploration,
        aggregate_observations,
    )

    n_honest, n_bad, epochs = 6, 3, 400
    frame = ReferenceFrame()
    for e in range(epochs):
        obs = []
        for m in MODELS:
            qid = f"{m}:q{e}"
            # Honest truth on a deterministic schedule whose rate over any 100
            # epochs is exactly TRUE_ACC (37 is coprime to 100, so e*37 mod 100
            # visits every residue). No RNG, no PYTHONHASHSEED dependence.
            h = int(hashlib.sha256(m.encode()).hexdigest()[:4], 16)
            truth = (e * 37 + h) % 100 < round(TRUE_ACC[m] * 100)
            obs += [_obs(qid, m, truth, f"h{i}") for i in range(n_honest)]
            obs += [_obs(qid, m, False, f"b{i}") for i in range(n_bad)]
        frame = accumulate_exploration(frame, aggregate_observations(obs))

    reference = frontier_of(frame_from_truth(epochs))
    attacked = frontier_of(frame)
    assert abs(attacked.max_accuracy - reference.max_accuracy) < 0.03, (
        f"a 33% coalition moved the frontier peak from {reference.max_accuracy:.3f} "
        f"to {attacked.max_accuracy:.3f}"
    )
    # Whether constant policies are PAID against this frontier is finding 3's
    # property (the prior bias) and is pinned in test_finding3_prior_bias.py.


def test_a_proof_that_fails_exploration_but_not_scoring_is_rejected(monkeypatch):
    """Blocking the network during exploration only must cost the whole proof."""
    from fugal_subnet.attacks import run_tee_attacks as suite
    from fugal_subnet.tee import verify as verify_mod

    monkeypatch.setattr(verify_mod, "verify_dcap", lambda q, **_kw: True)
    explore = {f"q{i}": "deepseek/deepseek-v4-flash" for i in range(10, 25)}
    scored = [suite._qr(q) for q in suite.SLICE]
    failed = [
        QuestionResult(q, m, False, 0.0, hashlib.sha256(b"").hexdigest(),
                       prompt_tokens=0, completion_tokens=0, is_exploration=True,
                       failed=True)
        for q, m in explore.items()
    ]
    proof = suite._proof(scored + failed)
    gold = {k: suite.POOL_GOLD[k] for k in list(suite.SLICE) + list(explore)}
    result = suite._verify(proof, expected_exploration=explore, gold_answers=gold)
    assert not result.valid, "every exploration call failed and the proof still verified"
    assert "explor" in result.reason.lower()
