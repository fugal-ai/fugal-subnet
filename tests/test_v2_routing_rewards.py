"""Regression vectors for inactive v2 routing, dedup, and rewards."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from fugal_subnet.head_eval import HeadArtifact
from fugal_subnet.v2.dedup import find_duplicates
from fugal_subnet.v2.head_eval import evaluate_head, validate_head_models
from fugal_subnet.v2.rewards import ProjectionInfeasible, compute_bounded_weights
from fugal_subnet.v2.scoring import (
    MinerRecord,
    composite_score,
    reconcile_uid_ownership,
)
from fugal_subnet.v2.soft_targets import compute_soft_targets

ACTIVE = ["provider/a", "provider/b", "provider/c"]
COSTS = {"provider/a": 0.01, "provider/b": 0.02, "provider/c": 0.03}


def make_head(models: list[str], biases: list[float]) -> HeadArtifact:
    return HeadArtifact(
        W=np.zeros((len(models), 2), dtype=np.float32),
        b=np.asarray(biases, dtype=np.float32),
        models=models,
        commit_hash="",
    )


def evaluate(head: HeadArtifact):
    hidden = np.zeros((4, 2), dtype=np.float32)
    matrix = np.ones((4, len(ACTIVE)), dtype=np.int8)
    soft = np.full((4, len(ACTIVE)), 1 / len(ACTIVE), dtype=np.float64)
    return evaluate_head(
        head,
        hidden,
        matrix,
        ACTIVE,
        soft,
        COSTS,
        wire_model_pool=list(head.models),
        routing_lambda=0.0,
    )


def test_reordered_copied_head_has_identical_canonical_behavior():
    original = evaluate(make_head(["provider/a", "provider/b"], [2.0, 0.0]))
    reordered = evaluate(make_head(["provider/b", "provider/a"], [0.0, 2.0]))

    np.testing.assert_array_equal(original.routing_decisions, reordered.routing_decisions)
    np.testing.assert_allclose(
        original.routing_distributions,
        reordered.routing_distributions,
        atol=0,
        rtol=0,
    )
    assert original.routing_model_ids == ("provider/a",) * 4

    result = find_duplicates(
        {1: original.routing_decisions, 2: reordered.routing_decisions},
        {1: original.routing_distributions, 2: reordered.routing_distributions},
        {1: 100, 2: 200},
        {1: "hotkey-a", 2: "hotkey-b"},
    )
    assert result.disqualified == frozenset({2})
    assert result.clusters == ((1, 2),)


def test_same_local_row_number_for_different_models_is_not_duplicate():
    routes_a = evaluate(make_head(["provider/a"], [0.0]))
    routes_b = evaluate(make_head(["provider/b"], [0.0]))

    # Both v1 heads would have emitted local row 0. V2 stores registry indexes.
    assert np.all(routes_a.routing_decisions == 0)
    assert np.all(routes_b.routing_decisions == 1)
    result = find_duplicates(
        {1: routes_a.routing_decisions, 2: routes_b.routing_decisions},
        {1: routes_a.routing_distributions, 2: routes_b.routing_distributions},
        {1: 100, 2: 200},
        {1: "hotkey-a", 2: "hotkey-b"},
    )
    assert result.disqualified == frozenset()


def test_v2_head_model_validation_is_strict():
    duplicate = make_head(["provider/a", "provider/a"], [0.0, 0.0])
    with pytest.raises(ValueError, match="unique"):
        validate_head_models(duplicate, ACTIVE)

    inactive = make_head(["provider/not-active"], [0.0])
    with pytest.raises(ValueError, match="not active"):
        validate_head_models(inactive, ACTIVE)

    valid = make_head(["provider/a", "provider/b"], [0.0, 0.0])
    with pytest.raises(ValueError, match="exactly match"):
        validate_head_models(valid, ACTIVE, ["provider/b", "provider/a"])

    too_long = make_head(["p/" + "x" * 127], [0.0])
    with pytest.raises(ValueError, match="128"):
        validate_head_models(too_long, [too_long.models[0]])


def test_forced_zero_is_immediate_and_ordinary_deltas_are_exact():
    result = compute_bounded_weights(
        scores={1: 10, 2: 1, 3: 1},
        previous_weights={1: 0.5, 2: 0.5},
        eligible_uids={2, 3},
        forced_zero_uids={1},
        max_delta="0.3",
        precision=12,
    )
    assert result is not None
    assert result.weights[1] == Decimal(0)
    assert sum(result.weights.values()) == Decimal(1)
    assert abs(result.weights[2] - Decimal("0.5")) <= Decimal("0.3")
    assert abs(result.weights[3] - Decimal("0.0")) <= Decimal("0.3")


def test_projection_does_not_stretch_cap_during_normalization():
    result = compute_bounded_weights(
        scores={1: 0.9, 2: 0.1},
        previous_weights={1: 0.1, 2: 0.9},
        eligible_uids={1, 2},
        forced_zero_uids=set(),
        max_delta="0.2",
        precision=12,
    )
    assert result is not None
    assert result.weights == {
        1: Decimal("0.3"),
        2: Decimal("0.7"),
    }
    assert sum(result.weights.values()) == Decimal(1)


def test_uid_zero_is_not_a_burn_target():
    result = compute_bounded_weights(
        scores={1: 1, 2: 1, 3: 1, 4: 1},
        previous_weights={0: 1},
        eligible_uids={1, 2, 3, 4},
        forced_zero_uids={0},
        max_delta="0.3",
        precision=12,
    )
    assert result is not None
    assert result.weights[0] == Decimal(0)
    assert all(result.weights[uid] == Decimal("0.25") for uid in (1, 2, 3, 4))


def test_infeasible_cap_fails_closed_and_empty_scores_preserve_chain():
    with pytest.raises(ProjectionInfeasible):
        compute_bounded_weights(
            scores={2: 1},
            previous_weights={1: 1},
            eligible_uids={2},
            forced_zero_uids={1},
            max_delta="0.3",
        )

    assert compute_bounded_weights(
        scores={1: 0, 2: -1},
        previous_weights={1: 0.5, 2: 0.5},
        eligible_uids={1, 2},
        forced_zero_uids=set(),
    ) is None


def test_projection_serialization_is_exact_and_deterministic():
    result = compute_bounded_weights(
        scores={3: 1, 1: 1, 2: 1},
        previous_weights={},
        eligible_uids={1, 2, 3},
        forced_zero_uids=set(),
        precision=6,
    )
    assert result is not None
    assert result.serialized() == {
        "1": "0.333334",
        "2": "0.333333",
        "3": "0.333333",
    }
    assert sum(result.weights.values()) == Decimal(1)


def test_uid_ownership_change_resets_all_persisted_score_history():
    records = {
        1: MinerRecord(
            uid=1,
            hotkey="old-hotkey",
            epochs_seen=9,
            current_head_hash="copied-history",
            composite_score=0.99,
        ),
        2: MinerRecord(uid=2, hotkey="stable-hotkey", epochs_seen=4),
        9: MinerRecord(uid=9, hotkey="unregistered-hotkey", epochs_seen=5),
    }

    result = reconcile_uid_ownership(
        records,
        {1: "new-hotkey", 2: "stable-hotkey", 3: "brand-new-hotkey"},
    )

    assert result.reset_uids == frozenset({1})
    assert result.new_uids == frozenset({3})
    assert result.removed_uids == frozenset({9})
    assert result.records[1] == MinerRecord(uid=1, hotkey="new-hotkey")
    assert result.records[2].epochs_seen == 4
    assert result.records[3] == MinerRecord(uid=3, hotkey="brand-new-hotkey")


def test_v2_soft_targets_and_composite_scoring_are_rounded_and_bounded():
    matrix = np.asarray([[1, 0, 1], [0, 0, 0]], dtype=np.int8)
    targets = compute_soft_targets(matrix)
    np.testing.assert_allclose(targets.sum(axis=1), np.ones(2), atol=0, rtol=0)
    assert np.all(np.isfinite(targets))
    score = composite_score(accuracy=1.0, cost_efficiency=0.5, kl_score=-0.25)
    assert 0 < score < 1
    assert score == round(score, 12)

    with pytest.raises(ValueError):
        composite_score(1.1, 0.5, 0.0)
