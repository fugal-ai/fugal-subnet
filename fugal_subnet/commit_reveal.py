"""Commit-reveal for epoch integrity.

Before querying miners, the validator commits a hash of (tasks + golds + grader).
After grading, the full artifacts are revealed. Anyone can recompute the hash to
verify the validator didn't tamper with tasks or golds after seeing miner responses.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict

from fugal_subnet.graders import grader_hash

logger = logging.getLogger(__name__)

EPOCH_DIR = os.getenv("FUGAL_EPOCH_DIR", "results/epochs")


@dataclass
class EpochCommitment:
    epoch_id: str
    commit_hash: str
    grader_hash: str
    n_questions: int
    block_hash: str


def canonical_json(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def compute_commit_hash(questions: list[dict], grader_hash: str) -> str:
    payload = canonical_json(questions) + b"|" + grader_hash.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def commit_epoch(epoch_id: str, questions: list[dict], block_hash: str) -> EpochCommitment:
    gh = grader_hash()
    commit_hash = compute_commit_hash(questions, gh)

    commitment = EpochCommitment(
        epoch_id=epoch_id,
        commit_hash=commit_hash,
        grader_hash=gh,
        n_questions=len(questions),
        block_hash=block_hash,
    )

    epoch_dir = os.path.join(EPOCH_DIR, epoch_id)
    os.makedirs(epoch_dir, exist_ok=True)
    commit_path = os.path.join(epoch_dir, "commit.json")
    with open(commit_path, "w") as f:
        json.dump(asdict(commitment), f, indent=2)

    logger.info("Epoch %s committed: %s", epoch_id, commit_hash)
    return commitment


def reveal_epoch(
    epoch_id: str,
    questions: list[dict],
    matrix,
    models: list[str],
    model_costs: dict[str, float],
    head_hashes: dict[int, str],
    routing_decisions: dict[int, list[int]],
    scores: dict[int, dict],
    weights: dict[int, float],
) -> bool:
    """Verify the commitment and publish the full epoch artifact.

    The reveal IS the data flywheel: it must contain the ground truth matrix
    (per-question, per-model grades), the model list and cost snapshot —
    everything a miner needs to retrain — plus each miner's head hash,
    routing decisions, scores, and the resulting weights for auditability.

    Args:
        matrix: (N, M) binary ground truth (np.ndarray or nested lists).
    """
    epoch_dir = os.path.join(EPOCH_DIR, epoch_id)
    commit_path = os.path.join(epoch_dir, "commit.json")
    if not os.path.exists(commit_path):
        logger.error("No commitment found for epoch %s", epoch_id)
        return False

    with open(commit_path) as f:
        commitment = json.load(f)

    committed_grader_hash = commitment["grader_hash"]
    expected = commitment["commit_hash"]
    actual = compute_commit_hash(questions, committed_grader_hash)

    if actual != expected:
        logger.error("COMMIT-REVEAL MISMATCH for %s: expected=%s actual=%s",
                     epoch_id, expected, actual)
        return False

    matrix_rows = matrix.tolist() if hasattr(matrix, "tolist") else matrix

    reveal = {
        "epoch_id": epoch_id,
        "commit_hash": expected,
        "grader_hash": committed_grader_hash,
        "questions": questions,
        "matrix": matrix_rows,
        "models": models,
        "model_costs": model_costs,
        "head_hashes": {str(k): v for k, v in head_hashes.items()},
        "routing_decisions": {str(k): list(map(int, v)) for k, v in routing_decisions.items()},
        "scores": {str(k): v for k, v in scores.items()},
        "weights": {str(k): v for k, v in weights.items()},
    }

    reveal_path = os.path.join(epoch_dir, "reveal.json")
    with open(reveal_path, "w") as f:
        json.dump(reveal, f, indent=2)

    logger.info("Epoch %s revealed and verified: %s", epoch_id, expected)
    return True


def verify_epoch(epoch_dir: str) -> bool:
    """Independently verify a revealed epoch — can be run by anyone."""
    with open(os.path.join(epoch_dir, "commit.json")) as f:
        commitment = json.load(f)
    with open(os.path.join(epoch_dir, "reveal.json")) as f:
        reveal = json.load(f)

    actual = compute_commit_hash(reveal["questions"], commitment["grader_hash"])
    return actual == commitment["commit_hash"]
