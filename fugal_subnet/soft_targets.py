"""Soft target distribution computation.

Converts the binary ground truth matrix into soft probability distributions
via softmax with temperature. These distributions are the training signal
for router heads (KL divergence loss) and the oracle for scoring.
"""
from __future__ import annotations

import numpy as np

from fugal_subnet.config import SOFT_TARGET_TAU


def compute_soft_targets(
    matrix: np.ndarray,
    tau: float = SOFT_TARGET_TAU,
) -> np.ndarray:
    """Compute soft target distributions from the ground truth matrix.

    Args:
        matrix: (N, M) binary matrix — 1 if model got question right.
        tau: Temperature parameter. Lower = sharper distributions.
             tau=1.0 is the default. tau->0 approaches argmax.

    Returns:
        (N, M) array where each row is a probability distribution (sums to 1).
        Rows where no model succeeded get uniform distribution.
    """
    scores = matrix.astype(np.float64)
    if tau <= 0:
        raise ValueError(f"Temperature must be positive, got {tau}")

    scaled = scores / tau
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    row_sums = exp_scores.sum(axis=1, keepdims=True)

    zero_rows = (row_sums < 1e-10).flatten()
    row_sums[zero_rows.reshape(-1, 1)] = 1.0

    soft = exp_scores / row_sums
    if zero_rows.any():
        M = matrix.shape[1]
        soft[zero_rows] = 1.0 / M

    return soft


def oracle_routing(
    matrix: np.ndarray,
    costs: np.ndarray | list[float],
) -> np.ndarray:
    """Compute oracle routing decisions: cheapest correct model per question.

    Args:
        matrix: (N, M) binary matrix.
        costs: (M,) per-model cost.

    Returns:
        (N,) array of model indices — the oracle's routing choice per question.
        Questions where no model succeeded get -1.
    """
    costs = np.asarray(costs, dtype=np.float64)
    N, M = matrix.shape
    routing = np.full(N, -1, dtype=np.int32)

    for q in range(N):
        correct = np.where(matrix[q] == 1)[0]
        if len(correct) > 0:
            routing[q] = correct[np.argmin(costs[correct])]

    return routing
