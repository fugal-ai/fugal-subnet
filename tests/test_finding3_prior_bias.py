"""Diligence 2026-10, finding 3: the frame prior pays constant policies at launch.

The frame starts every model at a neutral 50% worth 20 pseudo-samples. After 50
real samples a 90% model reads as 78.6%, which is exactly where the frontier's
confidence reaches 1.0 and headroom is paid in full. A head that always calls
that model is scored at its TRUE 90% against a frontier that believes the best
constant policy buys 78.6%, and collects the difference. Measured with the
repo's own code (see the table in docs/INVARIANTS.md once PR 3 lands):

    samples/model   top model frame   best constant-policy score
            50          0.786                 0.109
           200          0.864                 0.031
           650          0.888                 0.006
          2000          0.896                 0.000

The existing `test_every_constant_policy_scores_zero` scores each policy at the
frame's OWN estimate of its accuracy, so it can never see this. These tests
score policies at their true accuracy.

Expected to fail until PR 3; strict xfail so the marker leaves with the fix.
"""
from __future__ import annotations

import pytest

from tests.frame_fixtures import (
    MODELS,
    TOP,
    constant_policy_scores,
    frame_from_truth,
    frontier_of,
)

_PRIOR_BIAS = pytest.mark.xfail(
    strict=True,
    reason="diligence 2026-10 finding 3: a neutral prior lets constant policies score",
    raises=(AssertionError, TypeError, AttributeError, ImportError),
)


@_PRIOR_BIAS
@pytest.mark.parametrize("samples_per_model", [50, 200, 650])
def test_constant_policies_score_zero_at_true_accuracy(samples_per_model):
    """Whatever the frame has seen, a head that never reads the question earns 0.

    The frame here is built from exact-count samples equal to the truth, so
    any payout is an artefact of how the frontier VALUES those samples, not
    of noise.
    """
    frontier = frontier_of(frame_from_truth(samples_per_model))
    scores = constant_policy_scores(frontier)
    paid = {m: s for m, s in scores.items() if s > 0}
    assert not paid, (
        f"{len(paid)} of {len(MODELS)} constant policies are paid at "
        f"{samples_per_model} samples/model; best is {max(paid.values()):.3f} "
        f"(frontier top {frontier.max_accuracy:.3f} vs true {TOP} 0.900)"
    )


def test_duplicate_answers_to_one_question_are_one_distinct_trial():
    """Thirty miners answering the same exploration question is ONE observation.

    Finding 3 compounds with finding 2: every miner's copy of the shared
    exploration answer counts as a fresh sample today, so 30 miners reach
    `FRONTIER_MIN_TRIALS` in a couple of epochs on two distinct questions per
    model. The frame must count distinct (question, model) observations.
    """
    from fugal_subnet.reference_frame import (
        Observation,
        ReferenceFrame,
        accumulate_exploration,
        aggregate_observations,
    )

    obs = [
        Observation(question_id="q1", model=TOP, correct=True, prompt_tokens=100,
                    completion_tokens=300, failed=False, coldkey=f"ck{i}")
        for i in range(30)
    ]
    frame = accumulate_exploration(ReferenceFrame(), aggregate_observations(obs))
    assert frame.trials[TOP] == pytest.approx(1.0), (
        f"30 copies of one answer counted as {frame.trials[TOP]:.0f} trials"
    )
