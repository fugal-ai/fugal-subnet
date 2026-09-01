"""Rounded v2 matrix-to-training-target transformation."""

from __future__ import annotations

import math

import numpy as np

TAU = 1.0
ROUNDING_DECIMALS = 12


def compute_soft_targets(
    matrix: np.ndarray,
    *,
    tau: float = TAU,
    rounding_decimals: int = ROUNDING_DECIMALS,
) -> np.ndarray:
    grades = np.asarray(matrix)
    if grades.ndim != 2 or grades.shape[1] == 0:
        raise ValueError("matrix must be a non-empty two-dimensional model matrix")
    if not np.all((grades == 0) | (grades == 1)):
        raise ValueError("matrix values must be zero or one")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if not 0 <= rounding_decimals <= 15:
        raise ValueError("rounding_decimals must be between zero and fifteen")

    scaled = grades.astype(np.float64) / tau
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    targets = values / values.sum(axis=1, keepdims=True)
    rounded = np.round(targets, decimals=rounding_decimals)
    # Ensure exact row normalization after consensus rounding. The lowest
    # canonical model index receives a deterministic residual quantum.
    residual = 1.0 - rounded.sum(axis=1)
    rounded[:, 0] += residual
    return np.round(rounded, decimals=rounding_decimals)
