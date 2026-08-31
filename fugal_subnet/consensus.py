"""Multi-validator consensus checks — an OFFLINE audit tool.

Validators publish their scores and weights in each epoch's reveal.json.
Anyone can collect the reveals from several validators, build ValidatorReport
objects, and run compute_consensus() to detect:
1. Outlier validators (possibly dishonest or buggy)
2. Score divergence that exceeds expected variance
3. Validators that disagree on the committed epoch tasks

On-chain, Yuma consensus (stake-weighted median of weights) is what actually
disciplines validators; this module is the off-chain magnifying glass for
investigating disagreements. The consensus here is a plain (unweighted)
median across reports.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

CONSENSUS_MIN_VALIDATORS = int(os.getenv("FUGAL_MIN_VALIDATORS", "2"))
SCORE_DIVERGENCE_THRESHOLD = float(os.getenv("FUGAL_SCORE_DIVERGENCE", "0.15"))
WEIGHT_DIVERGENCE_THRESHOLD = float(os.getenv("FUGAL_WEIGHT_DIVERGENCE", "0.20"))


@dataclass
class ValidatorReport:
    """A single validator's epoch report."""
    validator_uid: int
    epoch_id: str
    scores: dict[int, float]
    weights: dict[int, float]
    commit_hash: str = ""


@dataclass
class ConsensusResult:
    epoch_id: str
    n_validators: int
    consensus_scores: dict[int, float] = field(default_factory=dict)
    consensus_weights: dict[int, float] = field(default_factory=dict)
    outlier_validators: list[int] = field(default_factory=list)
    divergence_flags: list[str] = field(default_factory=list)
    is_valid: bool = True


def compute_consensus(reports: list[ValidatorReport]) -> ConsensusResult:
    """Compute median consensus from multiple validator reports.

    With deterministic graders and shared tasks, honest validators should
    produce near-identical scores. Divergence indicates bugs or dishonesty.
    """
    if not reports:
        return ConsensusResult(epoch_id="", n_validators=0, is_valid=False)

    epoch_id = reports[0].epoch_id
    result = ConsensusResult(epoch_id=epoch_id, n_validators=len(reports))

    if len(reports) < CONSENSUS_MIN_VALIDATORS:
        result.divergence_flags.append(
            "insufficient_validators: %d < %d minimum" % (len(reports), CONSENSUS_MIN_VALIDATORS)
        )
        if reports:
            result.consensus_scores = reports[0].scores.copy()
            result.consensus_weights = reports[0].weights.copy()
        return result

    all_uids = sorted(set(uid for r in reports for uid in r.scores))

    for uid in all_uids:
        uid_scores = [r.scores[uid] for r in reports if uid in r.scores]
        if uid_scores:
            result.consensus_scores[uid] = float(np.median(uid_scores))

    for uid in all_uids:
        uid_weights = [r.weights.get(uid, 0.0) for r in reports]
        result.consensus_weights[uid] = float(np.median(uid_weights))

    for report in reports:
        score_div = _score_divergence(report.scores, result.consensus_scores)
        weight_div = _weight_divergence(report.weights, result.consensus_weights)

        if score_div > SCORE_DIVERGENCE_THRESHOLD:
            result.outlier_validators.append(report.validator_uid)
            result.divergence_flags.append(
                "score_outlier: validator %d diverges %.3f from consensus (threshold %.3f)"
                % (report.validator_uid, score_div, SCORE_DIVERGENCE_THRESHOLD)
            )

        if weight_div > WEIGHT_DIVERGENCE_THRESHOLD:
            if report.validator_uid not in result.outlier_validators:
                result.outlier_validators.append(report.validator_uid)
            result.divergence_flags.append(
                "weight_outlier: validator %d weight divergence %.3f (threshold %.3f)"
                % (report.validator_uid, weight_div, WEIGHT_DIVERGENCE_THRESHOLD)
            )

    commit_hashes = [r.commit_hash for r in reports if r.commit_hash]
    if len(set(commit_hashes)) > 1:
        result.divergence_flags.append(
            "commit_hash_mismatch: validators disagree on epoch tasks"
        )
        result.is_valid = False

    return result


def check_self_consistency(
    my_scores: dict[int, float],
    consensus_scores: dict[int, float],
    my_uid: int,
) -> list[str]:
    """Check if this validator's scores are consistent with consensus.

    Called by a validator to self-check before submitting weights.
    Returns list of warning messages (empty = consistent).
    """
    warnings = []

    if not consensus_scores:
        return warnings

    div = _score_divergence(my_scores, consensus_scores)
    if div > SCORE_DIVERGENCE_THRESHOLD:
        warnings.append(
            "self_divergence: our scores diverge %.3f from consensus (threshold %.3f)"
            % (div, SCORE_DIVERGENCE_THRESHOLD)
        )

    for uid in my_scores:
        if uid in consensus_scores:
            diff = abs(my_scores[uid] - consensus_scores[uid])
            if diff > 0.3:
                warnings.append(
                    "large_score_gap: UID %d score %.3f vs consensus %.3f (delta %.3f)"
                    % (uid, my_scores[uid], consensus_scores[uid], diff)
                )

    rank_mine = _rank_order(my_scores)
    rank_consensus = _rank_order(consensus_scores)
    common = set(rank_mine.keys()) & set(rank_consensus.keys())
    if len(common) >= 3:
        tau = _kendall_tau(rank_mine, rank_consensus, common)
        if tau < 0.5:
            warnings.append(
                "rank_disagreement: Kendall tau=%.3f — our ranking diverges from consensus"
                % tau
            )

    return warnings


def _score_divergence(scores_a: dict[int, float], scores_b: dict[int, float]) -> float:
    """Mean absolute difference between two score dicts on shared UIDs."""
    common = set(scores_a.keys()) & set(scores_b.keys())
    if not common:
        return 0.0
    diffs = [abs(scores_a[uid] - scores_b[uid]) for uid in common]
    return sum(diffs) / len(diffs)


def _weight_divergence(weights_a: dict[int, float], weights_b: dict[int, float]) -> float:
    """L1 distance between two weight distributions."""
    all_uids = set(weights_a.keys()) | set(weights_b.keys())
    return sum(abs(weights_a.get(uid, 0) - weights_b.get(uid, 0)) for uid in all_uids)


def _rank_order(scores: dict[int, float]) -> dict[int, int]:
    """Convert scores to rank positions (0 = best)."""
    sorted_uids = sorted(scores, key=lambda u: scores[u], reverse=True)
    return {uid: rank for rank, uid in enumerate(sorted_uids)}


def _kendall_tau(rank_a: dict[int, int], rank_b: dict[int, int], common: set[int]) -> float:
    """Kendall rank correlation coefficient on shared UIDs."""
    uids = sorted(common)
    n = len(uids)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff_a = rank_a[uids[i]] - rank_a[uids[j]]
            diff_b = rank_b[uids[i]] - rank_b[uids[j]]
            if diff_a * diff_b > 0:
                concordant += 1
            elif diff_a * diff_b < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total
