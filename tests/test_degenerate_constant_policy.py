"""A constant policy — "always route to model k" — is a head that never reads
the question. On the previous score it OUTSCORED a real router; on this one it
scores zero, by construction.

Measured on 5,000 held-out questions with the shipped trainer and the shipped
routing rule, under the OLD score (quality^0.9 * thrift^0.1 against the best
single model):

    always gpt-4o-mini      1.183   <- W=0 and a one-hot bias
    trained head (SFT)      1.058
    trained head (+CMA)     1.060
    per-benchmark lookup    1.036
    best single model       0.985   <- the reference
    random head             0.868
    always cheapest         0.479

The trained head was not the problem: it genuinely routed and beat chance by
0.24. The problem was the reference. Comparing against ONE point of the
price/accuracy curve hands a cheaper point on the same curve a cost advantage
for free, whatever exponent trades quality for cost; the break-even was a
property of the constants alone (77.4% of the reference's accuracy at the
thrift cap). docs/design-decisions.md keeps that derivation as the record.

The score is now headroom above the constant-policy frontier
(fugal_subnet/frontier.py). This file pins the fact that made the change
necessary — a constant policy must never be the winning strategy — in the form
that can be asserted without any benchmark data: against a frontier built from
a model pool, every constant policy on that pool scores zero, and a head that
routes better than any of them scores more than all of them.
"""
import random

from fugal_subnet.config import BURN_IN_QUESTIONS
from fugal_subnet.evidence import Evidence
from fugal_subnet.frontier import build_frontier
from fugal_subnet.reference_frame import ReferenceFrame, accumulate_exploration
from fugal_subnet.scoring import composite

# A pool shaped like the one the finding came from: a mid-priced model that is
# 95% as accurate as the top model at ~1/15 the price. Under the old score this
# is exactly the model that won.
PRICES = {
    "cheapest": (2e-8, 5e-8),
    "mid":      (1.5e-7, 6e-7),     # the gpt-4o-mini shape
    "top":      (2.5e-6, 1e-5),
}
TRUE_ACC = {"cheapest": 0.26, "mid": 0.698, "top": 0.734}
N_Q, PROMPT_TOKENS, COMPLETION = 300, 300 * 100, 300.0


def _frame():
    rng = random.Random(0)
    return accumulate_exploration(ReferenceFrame(), [
        (m, rng.random() < TRUE_ACC[m], 100, 300) for m in PRICES for _ in range(500)
    ])


def _cost(model):
    p_in, p_out = PRICES[model]
    return (p_in * PROMPT_TOKENS + p_out * COMPLETION * N_Q) / N_Q


def _ev(acc, cpq, n=BURN_IN_QUESTIONS * 4):
    return Evidence("h", n_correct=acc * n, n_total=float(n), cost_sum=cpq * n,
                    ref_cost_sum=0.0, pool_size=1e9, n_priced=float(n))


def test_the_mid_priced_constant_policy_that_won_now_scores_zero():
    frame = _frame()
    f = build_frontier(frame, PRICES, PROMPT_TOKENS, N_Q, COMPLETION)
    always_mid = _ev(frame.accuracy("mid"), _cost("mid"))
    always_top = _ev(frame.accuracy("top"), _cost("top"))
    always_cheapest = _ev(frame.accuracy("cheapest"), _cost("cheapest"))
    assert composite(always_mid, f) == 0.0
    assert composite(always_top, f) == 0.0
    assert composite(always_cheapest, f) == 0.0


def test_a_trained_router_outscores_every_constant_policy():
    """The check HEAD_EFFICACY.md said would have caught the original defect:
    13 evaluations, assert the router beats all of them. Here the router is the
    measured one — 0.755 accuracy at roughly mid's price band — and the
    constant policies are every model in the pool."""
    frame = _frame()
    f = build_frontier(frame, PRICES, PROMPT_TOKENS, N_Q, COMPLETION)
    router = _ev(0.755, _cost("mid") * 1.3)
    constants = [_ev(frame.accuracy(m), _cost(m)) for m in PRICES]
    assert composite(router, f) > 0
    assert all(composite(router, f) > composite(c, f) for c in constants)


def test_the_finding_cannot_recur_by_moving_a_constant():
    """There is no exponent or cap to move: the reference is the whole curve,
    and any policy on the curve has zero headroom at every point of it."""
    frame = _frame()
    f = build_frontier(frame, PRICES, PROMPT_TOKENS, N_Q, COMPLETION)
    for c, a in f.points:
        assert composite(_ev(a, c), f) == 0.0
