"""Diligence 2026-10, finding 4: anomaly detection is one-sided and advisory.

`implausible_exploration` flags a miner only when its exploration accuracy
beats the frame's expectation by more than 35 points. There is no check for
accuracy that is too LOW, so the downward manipulation of finding 2 is
invisible. It compares against the frame the attack is moving. And its output
changes nothing: a log line and an anomaly entry.

Required behaviour, delivered by PR 4:
  * flags in both directions, against the frame snapshot from before the epoch;
  * cross-miner agreement: while exploration is shared, every coldkey answers
    the same question on the same model, so a miner that disagrees with the
    majority stands out without any baseline;
  * flagged miners' samples are excluded from the frame for that epoch. Scores
    are never touched (I4): the frame is already built from everyone.

Expected to fail until PR 4; strict xfail so the marker leaves with the fix.
"""
from __future__ import annotations

import pytest

from fugal_subnet.reference_frame import ReferenceFrame, implausible_exploration

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="diligence 2026-10 finding 4: detector is upward-only and advisory",
    raises=(AssertionError, TypeError, AttributeError, ImportError),
)


def _frame(acc=0.5, trials=200.0):
    frame = ReferenceFrame()
    frame.trials["cheap"] = trials
    frame.successes["cheap"] = trials * acc
    return frame


def test_too_low_is_flagged_like_too_high():
    frame = _frame(acc=0.5)
    honest = [("cheap", i % 2 == 0) for i in range(10)]      # 50%, as expected
    sandbagging = [("cheap", False) for _ in range(10)]       # 0%: 50 points low
    flagged = implausible_exploration(frame, {1: honest, 2: sandbagging})
    assert 2 in flagged, "a miner 50 points BELOW the frame's expectation was not flagged"
    assert 1 not in flagged


def test_disagreement_with_the_cross_miner_majority_is_flagged():
    from fugal_subnet.reference_frame import disagreeing_exploration

    # (question_id, model, correct) per miner; six coldkeys agree, one does not.
    questions = [(f"q{i}", "cheap") for i in range(10)]
    majority = {uid: [(q, m, True) for q, m in questions] for uid in range(6)}
    dissenter = {6: [(q, m, False) for q, m in questions]}
    coldkeys = {uid: f"ck{uid}" for uid in range(7)}
    flagged = disagreeing_exploration({**majority, **dissenter}, coldkeys)
    assert 6 in flagged, "a miner disagreeing with every other coldkey was not flagged"
    assert not any(uid in flagged for uid in range(6))


def test_disagreement_needs_enough_voters_and_samples():
    from fugal_subnet.reference_frame import disagreeing_exploration

    # Two voters: no majority to speak of, nobody is flagged.
    two = {0: [(f"q{i}", "cheap", True) for i in range(10)],
           1: [(f"q{i}", "cheap", False) for i in range(10)]}
    assert not disagreeing_exploration(two, {0: "a", 1: "b"})
    # Many voters but a miner that answered only three questions: too few samples.
    many = {uid: [(f"q{i}", "cheap", True) for i in range(10)] for uid in range(6)}
    many[6] = [(f"q{i}", "cheap", False) for i in range(3)]
    assert 6 not in disagreeing_exploration(many, {uid: f"ck{uid}" for uid in range(7)})


def test_flagged_miners_are_excluded_from_the_frame_not_from_scoring():
    from fugal_subnet.reference_frame import Observation, aggregate_observations

    obs = [Observation("q1", "cheap", True, 100, 300, False, f"ck{i}") for i in range(5)]
    obs += [Observation("q1", "cheap", False, 100, 300, False, "flagged")]
    with_all = aggregate_observations(obs)
    without = aggregate_observations(obs, exclude_coldkeys={"flagged"})
    assert len(with_all) == len(without) == 1
    # The honest majority already wins the median; exclusion makes the flagged
    # miner's vote vanish entirely rather than merely be outvoted.
    assert without[0][1] == pytest.approx(1.0)
