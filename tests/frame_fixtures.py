"""Shared fixtures for the reference-frame and frontier property tests.

A 17-model pool shaped like the pinned price table: accuracy rises with price
from 45% to 90%, prices rise geometrically. Everything is deterministic (exact
success counts, no RNG) so a test that fails is a fact about the scoring code,
not about a seed.

Used by the diligence-finding tests (`tests/test_finding*.py`) and kept
separate so the fix PRs can reuse the same world when they flip those tests
from xfail to must-pass.
"""
from __future__ import annotations

from fugal_subnet.config import BURN_IN_QUESTIONS
from fugal_subnet.evidence import Evidence
from fugal_subnet.frontier import Frontier, build_frontier
from fugal_subnet.reference_frame import ReferenceFrame, accumulate_exploration
from fugal_subnet.scoring import composite

N_MODELS = 17
MODELS = [f"m/{i:02d}" for i in range(N_MODELS)]
TRUE_ACC = {m: 0.45 + 0.45 * i / (N_MODELS - 1) for i, m in enumerate(MODELS)}
PRICES = {m: (1e-7 * (1.3 ** i), 2e-7 * (1.3 ** i)) for i, m in enumerate(MODELS)}
N_Q = 300
PROMPT_TOKENS = N_Q * 100
COMPLETION = 300.0
TOP = MODELS[-1]


def cost_per_q(model: str) -> float:
    p_in, p_out = PRICES[model]
    return (p_in * PROMPT_TOKENS + p_out * COMPLETION * N_Q) / N_Q


def exact_samples(model: str, n: int, acc: float) -> list[tuple[str, bool, int, int]]:
    """Exactly round(acc * n) correct answers for `model`, legacy sample shape."""
    k = round(acc * n)
    return ([(model, True, 100, int(COMPLETION))] * k
            + [(model, False, 100, int(COMPLETION))] * (n - k))


def frame_from_truth(samples_per_model: int, acc_of=None) -> ReferenceFrame:
    """A frame fed `samples_per_model` exact-count samples per model in one epoch."""
    acc_of = acc_of or (lambda m: TRUE_ACC[m])
    samples: list[tuple[str, bool, int, int]] = []
    for m in MODELS:
        samples += exact_samples(m, samples_per_model, acc_of(m))
    return accumulate_exploration(ReferenceFrame(), samples)


def frontier_of(frame: ReferenceFrame) -> Frontier:
    return build_frontier(frame, PRICES, PROMPT_TOKENS, N_Q, COMPLETION)


def evidence(acc: float, cost: float, n: int = BURN_IN_QUESTIONS * 4) -> Evidence:
    """A miner's accumulated evidence: `acc` observed over `n` scored questions."""
    return Evidence(weights_hash="h", n_correct=acc * n, n_total=float(n),
                    cost_sum=cost * n, ref_cost_sum=0.0, pool_size=1e9,
                    n_priced=float(n))


def constant_policy_scores(frontier: Frontier) -> dict[str, float]:
    """Every 'always call model m' policy, scored at its TRUE accuracy.

    This is the number the subnet must never pay: a head that never reads the
    question. The old test scored each policy at the frame's own estimate of
    its accuracy, which is zero by construction and proves nothing.
    """
    return {m: composite(evidence(TRUE_ACC[m], cost_per_q(m)), frontier) for m in MODELS}
