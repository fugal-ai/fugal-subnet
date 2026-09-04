"""Artifact-keyed evidence accumulation for stable scoring.

A single epoch is far too noisy to rank miners on: 300 questions gives a
standard error of ~3 accuracy points, so a lucky epoch dethrones a better
miner and two honest validators can disagree. Evidence accumulation pools each
artifact's results across epochs into a decayed binomial, so the noise cancels
while the ranking stays responsive to real change.

Two properties are load-bearing:

**Artifact keying.** Evidence is keyed by the head's weights hash, so changing
your head resets the accumulator. That makes dethroning cost real work rather
than a lucky epoch.

**Symmetric reset.** Reset clears accumulated penalties exactly as readily as
accumulated credit, so on its own it would let any miner wash a bad record by
flipping one weight bit. The burn-in ramp in `scoring.py` is what closes that:
climbing back after a reset costs precisely what earning the position cost the
first time. Reset stays cheap to *do* and expensive to *profit from*.

Missing an epoch counts as n_expected questions scored zero, which is what
stops a miner publishing only its lucky epochs.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from fugal_subnet.config import WILSON_CONFIDENCE

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    weights_hash: str
    n_correct: float = 0.0
    n_total: float = 0.0
    # Miner's attested cost on scored routes, and what the reference model
    # would have cost on the same questions. Both are dollars, so they need no
    # confidence interval — cost is arithmetic, not a sampled quantity.
    cost_sum: float = 0.0
    ref_cost_sum: float = 0.0
    epochs_accumulated: int = 0
    epochs_missed: int = 0
    # Distinct questions available to sample. Caps the effective sample size:
    # see effective_n.
    pool_size: float = 0.0

    @property
    def effective_n(self) -> float:
        """Sample size the confidence interval may actually assume.

        Wilson assumes independent Bernoulli trials. With a 200-epoch half-life
        and 300 questions an epoch, n_total settles near 86,550 — but the pool
        holds ~21,000 distinct questions, so those trials are roughly 4x reuse
        of the same questions, not 86,550 independent draws. Claiming the
        larger number overstates confidence. You cannot have more independent
        trials than there are distinct questions to draw.
        """
        if self.pool_size <= 0:
            return self.n_total
        return min(self.n_total, self.pool_size)

    @property
    def wilson_lcb(self) -> float:
        if self.n_total < 1e-6:
            return 0.0
        return _wilson_lower_bound(
            self.n_correct / self.n_total, self.effective_n, WILSON_CONFIDENCE,
        )

    @property
    def accuracy(self) -> float:
        if self.n_total < 1e-6:
            return 0.0
        return self.n_correct / self.n_total

    @property
    def thrift(self) -> float:
        """Reference cost over miner cost. >1 means cheaper than the reference."""
        if self.cost_sum < 1e-12:
            return 0.0
        return self.ref_cost_sum / self.cost_sum


def decay_factor(half_life: int) -> float:
    return 2.0 ** (-1.0 / max(1, half_life))


def accumulate_epoch(
    evidence: Evidence | None,
    weights_hash: str,
    n_correct: int,
    n_total: int,
    cost: float,
    ref_cost: float,
    half_life: int,
    pool_size: float = 0.0,
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
            cost_sum=cost,
            ref_cost_sum=ref_cost,
            epochs_accumulated=1,
            epochs_missed=0,
            pool_size=pool_size,
        )

    return Evidence(
        weights_hash=weights_hash,
        n_correct=evidence.n_correct * alpha + n_correct,
        n_total=evidence.n_total * alpha + n_total,
        cost_sum=evidence.cost_sum * alpha + cost,
        ref_cost_sum=evidence.ref_cost_sum * alpha + ref_cost,
        epochs_accumulated=evidence.epochs_accumulated + 1,
        epochs_missed=0,
        pool_size=pool_size or evidence.pool_size,
    )


def apply_miss(
    evidence: Evidence,
    n_expected: int,
    half_life: int,
) -> Evidence:
    """Account for a missed epoch: n_expected questions scored as 0 correct.

    Costs decay together, so a miss leaves thrift unchanged and moves only
    quality. That is the right shape: skipping an epoch says nothing about how
    cheaply the miner routes, but it does mean the questions went unanswered.
    """
    alpha = decay_factor(half_life)
    return Evidence(
        weights_hash=evidence.weights_hash,
        n_correct=evidence.n_correct * alpha,
        n_total=evidence.n_total * alpha + n_expected,
        cost_sum=evidence.cost_sum * alpha,
        ref_cost_sum=evidence.ref_cost_sum * alpha,
        epochs_accumulated=evidence.epochs_accumulated,
        epochs_missed=evidence.epochs_missed + 1,
        pool_size=evidence.pool_size,
    )


def _wilson_lower_bound(p: float, n: float, confidence: float) -> float:
    if n < 1e-6:
        return 0.0
    _z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = _z_table.get(confidence)
    if z is None:
        logger.warning("Wilson confidence %.2f not in lookup table, defaulting to z=1.96", confidence)
        z = 1.96
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(max(0.0, (p * (1 - p) + z * z / (4 * n)) / n))
    return max(0.0, (center - spread) / denom)
