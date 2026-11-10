"""Protocol v2: exploration observations weighted by the scored mix in the frame.

docs/INVARIANTS.md, "I9 — v2: the frame is estimated on the scored mix".
"""
import pytest

from fugal_subnet.benchmarks.slicer import mix_weights
from fugal_subnet.frontier import build_frontier
from fugal_subnet.reference_frame import (
    Observation,
    ReferenceFrame,
    accumulate_exploration,
    aggregate_observations,
)

PRICES = {"m/a": (1e-6, 1e-6), "m/b": (1e-7, 1e-7)}


def test_v1_frames_serialise_exactly_as_before():
    f = accumulate_exploration(None, [("m/a", True, 10.0, 20.0)])
    assert not f.weighted
    assert set(f.to_dict()) == {"routing_protocol", "successes", "trials", "prompt_tokens",
                                "completion_tokens", "epochs"}
    assert ReferenceFrame.from_dict(f.to_dict()).to_dict() == f.to_dict()


def test_weights_change_accuracy_and_verbosity_but_not_the_trial_count():
    # Model a: right on an over-explored easy benchmark (weight 0.25), wrong on
    # an under-explored hard one (weight 4). Unweighted it looks 50% accurate;
    # on the scored mix it is 0.25/(0.25+4) = 5.9%.
    samples = [("m/a", True, 10.0, 50.0), ("m/a", False, 10.0, 1000.0)]
    raw = accumulate_exploration(None, samples)
    mixed = accumulate_exploration(None, samples, weights=[0.25, 4.0])
    assert raw.trials["m/a"] == mixed.trials["m/a"] == 2.0           # confidence unchanged
    assert raw.frontier_accuracy("m/a") == 0.5
    assert mixed.frontier_accuracy("m/a") == pytest.approx(0.25 / 4.25)
    assert mixed.avg_completion_tokens("m/a", 0) == pytest.approx((0.25 * 50 + 4 * 1000) / 4.25)
    assert raw.avg_completion_tokens("m/a", 0) == 525.0


def test_frontier_confidence_reads_the_raw_count():
    samples = [("m/a", True, 10.0, 10.0)] * 5 + [("m/b", False, 10.0, 10.0)] * 5
    raw = accumulate_exploration(None, samples)
    mixed = accumulate_exploration(None, samples, weights=[0.1] * 10)
    f_raw = build_frontier(raw, PRICES, 1000, 10, 10.0)
    f_mix = build_frontier(mixed, PRICES, 1000, 10, 10.0)
    assert f_raw.least_trials == f_mix.least_trials and f_raw.confidence == f_mix.confidence


def test_once_weighted_the_frame_stays_weighted():
    f = accumulate_exploration(None, [("m/a", True, 1.0, 1.0)], weights=[2.0])
    g = accumulate_exploration(f, [("m/a", False, 1.0, 1.0)])          # no weights given
    from fugal_subnet.reference_frame import decay_factor
    alpha = decay_factor(500)
    assert g.weighted and g.trials_w["m/a"] == pytest.approx(2.0 * alpha + 1.0)
    assert ReferenceFrame.from_dict(g.to_dict()).to_dict() == g.to_dict()


def test_weights_must_align_with_samples():
    with pytest.raises(ValueError):
        accumulate_exploration(None, [("m/a", True, 1.0, 1.0)], weights=[1.0, 2.0])


def test_keyed_aggregation_carries_the_question_and_matches_the_plain_rows():
    obs = [Observation("q1", "m/a", True, 10, 20, coldkey="c1"),
           Observation("q1", "m/a", False, 10, 40, coldkey="c2"),
           Observation("q2", "m/b", True, 5, 5, coldkey="c1")]
    plain = aggregate_observations(obs)
    keyed = aggregate_observations(obs, with_question=True)
    assert [k[0] for k in keyed] == ["q1", "q2"]
    assert [k[1:] for k in keyed] == plain


def test_mix_weights_restore_the_scored_share():
    scored = [{"benchmark": "mmlu"}] * 30 + [{"benchmark": "ifeval"}] * 5
    explored = [{"benchmark": "mmlu"}] * 2 + [{"benchmark": "ifeval"}] * 2
    w = mix_weights(scored, explored)
    # Weighted exploration shares equal scored shares.
    total = 2 * w["mmlu"] + 2 * w["ifeval"]
    assert 2 * w["ifeval"] / total == pytest.approx(5 / 35)


def test_the_first_weighted_epoch_keeps_every_estimate_continuous():
    """Before the fix a model with a long raw history was valued at its single
    newest sample the moment the frame became weighted (0.863 -> 0.000 on the
    shipped frame), and stayed there while the weighted sums refilled."""
    from fugal_subnet.reference_frame import decay_factor

    raw = accumulate_exploration(None, [("m/a", True, 10.0, 300.0)] * 9 + [("m/a", False, 10.0, 300.0)])
    assert raw.frontier_accuracy("m/a") == pytest.approx(0.9) and not raw.weighted
    first = accumulate_exploration(raw, [("m/a", False, 10.0, 2000.0)], weights=[1.0])
    alpha = decay_factor(500)
    # 10 decayed historical trials at weight 1, plus one new wrong answer.
    assert first.trials_w["m/a"] == pytest.approx(10 * alpha + 1)
    assert first.frontier_accuracy("m/a") == pytest.approx(9 * alpha / (10 * alpha + 1))
    assert 0.8 < first.frontier_accuracy("m/a") < 0.9
    assert first.avg_completion_tokens("m/a", 0) == pytest.approx(
        (3000 * alpha + 2000) / (10 * alpha + 1))
    # And it serialises / reloads as a weighted frame.
    assert ReferenceFrame.from_dict(first.to_dict()).to_dict() == first.to_dict()
