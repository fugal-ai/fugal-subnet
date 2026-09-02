"""Cross-version deterministic golden vector for consensus-facing v2 math."""

from __future__ import annotations

import hashlib
from decimal import Decimal

import numpy as np

from fugal_subnet.consensus_manifest import canonical_json, load_consensus_manifest
from fugal_subnet.head_eval import HeadArtifact
from fugal_subnet.v2.committee import select_builders
from fugal_subnet.v2.dedup import find_duplicates
from fugal_subnet.v2.head_eval import evaluate_head
from fugal_subnet.v2.reveal import question_commitment_hash, registry_snapshot_hash
from fugal_subnet.v2.rewards import compute_bounded_weights
from fugal_subnet.v2.scoring import composite_score
from fugal_subnet.v2.soft_targets import compute_soft_targets

GOLDEN_SCHEMA_VERSION = 2
# Both pins are updated only by scripts/update_v2_golden.py, after an
# explicit review of the tests/fixtures/v2_golden.json diff.
EXPECTED_GOLDEN_SHA256 = "cd8a6976e93841f69b993f123413b68833ce9943bc9f07757af60e904c79a2d7"
# Covers the material-independent math only. This one must not move.
EXPECTED_MATH_SHA256 = "e15c8f129ebfe951685d97969729b984cd7da409b6e64a11bf32ed199b1a1a9d"


def _decimal(value: float) -> str:
    return format(Decimal(str(value)), ".12f")


def build_golden_vector() -> dict:
    """Exercise selection, slicing, matrices, routing, dedup, scores and weights."""
    manifest = load_consensus_manifest("local")
    boundary_hash = "0123456789abcdef" * 4
    hotkeys = [f"validator-{index}" for index in range(7)]
    permits = [True, False, True, True, True, True, False]
    committee = select_builders(boundary_hash, hotkeys, permits)

    questions = [
        {
            "benchmark": "golden",
            "gold": str(index),
            "grader_id": "integer_decimal_v2",
            "metadata": {},
            "prompt": f"golden question {index}",
            "question_id": f"golden-{index}",
        }
        for index in range(4)
    ]
    v2_spec = next(item for item in manifest.protocols if item.protocol_id == "v2")
    if v2_spec.consensus is None:
        raise RuntimeError("packaged v2 consensus material is missing")
    grader_hash = v2_spec.consensus["grader"]["sha256"]
    question_hash = question_commitment_hash(questions, grader_hash)
    model_ids = ["provider/alpha", "provider/beta", "provider/gamma"]
    costs_text = {
        "provider/alpha": "0.001",
        "provider/beta": "0.002",
        "provider/gamma": "0.003",
    }
    costs = {model: float(value) for model, value in costs_text.items()}
    registry_hash = registry_snapshot_hash(model_ids, costs_text)
    matrix: np.ndarray = np.asarray(
        [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=np.int8
    )
    hidden: np.ndarray = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    soft_targets = compute_soft_targets(matrix)
    gamma: np.ndarray = np.asarray([0.7, -0.3, 0.2, 0.1], dtype=np.float32)
    alpha: np.ndarray = np.asarray([-0.2, 0.8, 0.1, -0.1], dtype=np.float32)
    first = HeadArtifact(
        W=np.stack([gamma, alpha]),
        b=np.asarray([0.05, -0.02], dtype=np.float32),
        models=["provider/gamma", "provider/alpha"],
        commit_hash="1" * 64,
    )
    copied_reordered = HeadArtifact(
        W=np.stack([alpha, gamma]),
        b=np.asarray([-0.02, 0.05], dtype=np.float32),
        models=["provider/alpha", "provider/gamma"],
        commit_hash="2" * 64,
    )
    evaluations = {
        5: evaluate_head(first, hidden, matrix, model_ids, soft_targets, costs),
        9: evaluate_head(
            copied_reordered, hidden, matrix, model_ids, soft_targets, costs
        ),
    }
    decisions = {uid: score.routing_decisions for uid, score in evaluations.items()}
    distributions = {
        uid: score.routing_distributions for uid, score in evaluations.items()
    }
    dedup = find_duplicates(
        decisions,
        distributions,
        commit_blocks={5: 100, 9: 101},
        hotkeys={5: "miner-five", 9: "miner-nine"},
    )
    scores = {
        uid: composite_score(score.accuracy, score.cost_efficiency, score.kl_score)
        for uid, score in evaluations.items()
    }
    scores[12] = composite_score(0.4, 0.8, -0.5)
    weights = compute_bounded_weights(
        scores,
        previous_weights={5: "0.45", 9: "0.35", 12: "0.20"},
        eligible_uids={5, 9, 12},
        forced_zero_uids=set(dedup.disqualified),
        max_delta="0.3",
        precision=12,
    )
    if weights is None:
        raise RuntimeError("golden vector unexpectedly has no positive weight result")

    matrix_bytes = canonical_json(matrix.tolist())
    return {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        # Hashes of packaged consensus material. These legitimately change
        # whenever the manifest or grader bundle is rebuilt, so they are kept
        # apart from the math below and pinned separately.
        "material": {
            "manifest_sha256": manifest.sha256,
            "grader_sha256": grader_hash,
            "question_commitment": question_hash,
        },
        # Derived only from the fixed inputs above. Any change here is a real
        # consensus-math regression and must never be repinned without review.
        "math": {
            "boundary": {"block": 100, "hash": boundary_hash},
            "committee": [
                {"uid": builder.uid, "hotkey": builder.hotkey} for builder in committee
            ],
            "slice": {
                "question_ids": [question["question_id"] for question in questions],
            },
            "registry": {"model_ids": model_ids, "snapshot_hash": registry_hash},
            "matrix": matrix.tolist(),
            "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
            "soft_targets": [
                [_decimal(value) for value in row] for row in soft_targets
            ],
            "heads": {
                str(uid): {
                    "accuracy": _decimal(score.accuracy),
                    "cost_efficiency": _decimal(score.cost_efficiency),
                    "kl_score": _decimal(score.kl_score),
                    "decisions": score.routing_decisions.tolist(),
                    "model_ids": list(score.routing_model_ids),
                    "distributions": [
                        [_decimal(value) for value in row]
                        for row in score.routing_distributions
                    ],
                }
                for uid, score in sorted(evaluations.items())
            },
            "dedup": {
                "clusters": [list(cluster) for cluster in dedup.clusters],
                "disqualified": sorted(dedup.disqualified),
            },
            "scores": {
                str(uid): _decimal(score) for uid, score in sorted(scores.items())
            },
            "weights": weights.serialized(),
        },
    }


def golden_bytes() -> bytes:
    return canonical_json(build_golden_vector())


def golden_sha256() -> str:
    return hashlib.sha256(golden_bytes()).hexdigest()


def golden_math_bytes() -> bytes:
    """Serialize only the material-independent consensus math."""
    return canonical_json(build_golden_vector()["math"])


def golden_math_sha256() -> str:
    return hashlib.sha256(golden_math_bytes()).hexdigest()


def assert_golden() -> None:
    """Fail closed on any drift, naming the math pin first.

    A math mismatch is a consensus regression. A whole-vector mismatch with an
    intact math pin is ordinary packaged-material churn and is repinned with
    ``scripts/update_v2_golden.py`` after review.
    """
    actual_math = golden_math_sha256()
    if actual_math != EXPECTED_MATH_SHA256:
        raise RuntimeError(
            "v2 golden consensus math differs: expected "
            f"{EXPECTED_MATH_SHA256}, got {actual_math}"
        )
    actual = golden_sha256()
    if actual != EXPECTED_GOLDEN_SHA256:
        raise RuntimeError(
            f"v2 golden vector differs: expected {EXPECTED_GOLDEN_SHA256}, got {actual}"
        )
