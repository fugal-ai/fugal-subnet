"""Artifact-keyed evidence accumulation for stable scoring.

Each miner's performance is accumulated into a decayed binomial keyed by
its head's weights hash.  Changing the head resets the accumulator.  Missing
an epoch counts as n_expected tasks scored 0.  The accumulated evidence
produces a Wilson LCB that converges as epochs pile up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from fugal_subnet.config import WILSON_CONFIDENCE


@dataclass
class Evidence:
    weights_hash: str
    n_correct: float = 0.0
    n_total: float = 0.0
    cost_sum: float = 0.0
    oracle_cost_sum: float = 0.0
    kl_sum: float = 0.0
    n_kl: float = 0.0
    coverage_sum: float = 0.0
    coverage_n: float = 0.0
    epochs_accumulated: int = 0
    epochs_missed: int = 0

    @property
    def wilson_lcb(self) -> float:
        if self.n_total < 1e-6:
            return 0.0
        p = self.n_correct / self.n_total
        return _wilson_lower_bound(p, self.n_total, WILSON_CONFIDENCE)

    @property
    def accuracy(self) -> float:
        if self.n_total < 1e-6:
            return 0.0
        return self.n_correct / self.n_total

    @property
    def cost_efficiency(self) -> float:
        if self.cost_sum < 1e-10:
            return 0.0
        return min(1.0, self.oracle_cost_sum / self.cost_sum)

    @property
    def avg_kl(self) -> float:
        if self.n_kl < 1e-6:
            return -100.0
        return -(self.kl_sum / self.n_kl)

    @property
    def avg_coverage(self) -> float:
        if self.coverage_n < 1e-6:
            return 0.0
        return self.coverage_sum / self.coverage_n


def decay_factor(half_life: int) -> float:
    return 2.0 ** (-1.0 / half_life)


def accumulate_epoch(
    evidence: Evidence | None,
    weights_hash: str,
    n_correct: int,
    n_total: int,
    head_cost: float,
    oracle_cost: float,
    kl_total: float,
    n_kl: int,
    coverage: float,
    half_life: int,
) -> Evidence:
    """Add one epoch's results to the accumulator.

    If weights_hash differs from existing evidence, the accumulator resets.
    """
    alpha = decay_factor(half_life)

    if evidence is None or evidence.weights_hash != weights_hash:
        return Evidence(
            weights_hash=weights_hash,
            n_correct=float(n_correct),
            n_total=float(n_total),
            cost_sum=head_cost,
            oracle_cost_sum=oracle_cost,
            kl_sum=kl_total,
            n_kl=float(n_kl),
            epochs_accumulated=1,
            epochs_missed=0,
            coverage_sum=coverage,
            coverage_n=1.0,
        )

    return Evidence(
        weights_hash=weights_hash,
        n_correct=evidence.n_correct * alpha + n_correct,
        n_total=evidence.n_total * alpha + n_total,
        cost_sum=evidence.cost_sum * alpha + head_cost,
        oracle_cost_sum=evidence.oracle_cost_sum * alpha + oracle_cost,
        kl_sum=evidence.kl_sum * alpha + kl_total,
        n_kl=evidence.n_kl * alpha + n_kl,
        epochs_accumulated=evidence.epochs_accumulated + 1,
        epochs_missed=0,
        coverage_sum=evidence.coverage_sum * alpha + coverage,
        coverage_n=evidence.coverage_n * alpha + 1.0,
    )


def apply_miss(
    evidence: Evidence,
    n_expected: int,
    half_life: int,
) -> Evidence:
    """Account for a missed epoch: n_expected tasks scored as 0 correct."""
    alpha = decay_factor(half_life)
    return Evidence(
        weights_hash=evidence.weights_hash,
        n_correct=evidence.n_correct * alpha,
        n_total=evidence.n_total * alpha + n_expected,
        cost_sum=evidence.cost_sum * alpha,
        oracle_cost_sum=evidence.oracle_cost_sum * alpha,
        kl_sum=evidence.kl_sum * alpha,
        n_kl=evidence.n_kl * alpha,
        epochs_accumulated=evidence.epochs_accumulated,
        epochs_missed=evidence.epochs_missed + 1,
        coverage_sum=evidence.coverage_sum * alpha,
        coverage_n=evidence.coverage_n * alpha,
    )


def _wilson_lower_bound(p: float, n: float, confidence: float) -> float:
    if n < 1e-6:
        return 0.0
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (center - spread) / denom)
