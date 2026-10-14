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

Closed by valuing the frontier at each model's RAW sample mean
(`ReferenceFrame.frontier_accuracy`), with no prior: a constant policy at its
true accuracy sits on the frontier and its lower bound below it, at every
sample count. The distinct-trial count comes from finding 2's aggregation, and
the shipped bootstrap frame is what makes epoch 1 pay instead of epoch 340.
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


# TRUE_ACC steps by 9/320, so these sample counts make every acc * n an integer
# and the frame's raw mean equals the truth exactly: any payout is then an
# artefact of how the frontier VALUES the samples, never of rounding or noise.
@pytest.mark.parametrize("samples_per_model", [320, 640, 3200])
def test_constant_policies_score_zero_at_true_accuracy(samples_per_model):
    """Whatever the frame has seen, a head that never reads the question earns 0."""
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


def test_the_frontier_is_valued_at_the_raw_mean_with_no_prior():
    from fugal_subnet.reference_frame import ReferenceFrame

    frame = ReferenceFrame()
    assert frame.frontier_accuracy("never-seen") == 1.0
    for n in (50.0, 5000.0):
        frame.trials["m"], frame.successes["m"] = n, 0.9 * n
        assert frame.frontier_accuracy("m") == pytest.approx(0.9), n
        assert frame.accuracy("m") < 0.9, "the posterior mean still carries the prior"


def test_a_cold_frontier_pays_nobody_and_a_measured_one_pays_routing():
    from fugal_subnet.reference_frame import ReferenceFrame
    from fugal_subnet.scoring import composite
    from tests.frame_fixtures import PRICES, cost_per_q, evidence

    cold = frontier_of(ReferenceFrame())
    assert cold.max_accuracy == 1.0 and cold.confidence == 0.0
    router = evidence(0.95, cost_per_q(MODELS[3]))          # better than anything, cheaply
    assert composite(router, cold) == 0.0
    warm = frontier_of(frame_from_truth(5000))
    assert warm.confidence == 1.0
    assert composite(router, warm) > 0.0
    assert set(PRICES) == set(MODELS)


def test_live_validators_must_start_from_the_shipped_frame(tmp_path):
    from fugal_subnet.reference_frame import load_bootstrap

    missing = str(tmp_path / "reference_frame.json")
    assert load_bootstrap(missing).trials == {}
    with pytest.raises(FileNotFoundError, match="build_bootstrap_frame"):
        load_bootstrap(missing, require=True)
