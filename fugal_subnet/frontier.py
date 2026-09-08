"""The constant-policy frontier: what not routing at all can buy at each price.

A router is paid for **the value of reading the question**. Anything a policy
can achieve without reading it is free to build — pick one model and always
call it, or flip a coin between two — and the subnet must not pay for it. So
the reference a miner is scored against is not the best single model (one
point on the curve, which lets a cheap-and-decent model win on cost alone —
measured, INVARIANTS § I3) and not the other miners (which would let a miner's
score move because someone else came online — I4). It is the whole curve of
what constant policies can do, and the score is the accuracy a miner achieves
**above** that curve at the miner's own cost.

Why the curve is the upper convex hull, not just the best model at each price.
A policy that routes a fraction p of questions to model A and the rest to
model B *at random* reads nothing, and its (cost, accuracy) is the point p of
the way along the segment from A to B. Every point on every such segment is
therefore reachable by a non-routing policy, so the frontier is the upper
convex hull of the per-model points — together with the origin, since "answer
nothing" costs nothing and scores nothing and can be mixed in too. Past the
most accurate model the frontier is flat: spending more buys nothing a constant
policy could not have had for less.

Every input is consensus state: per-model accuracy from the reference frame
(pooled over time from nonce-assigned exploration, so no miner chooses the
questions, the models, or the outcomes), per-model cost from the pinned price
table applied to this epoch's slice and the frame's measured verbosity. Two
validators with the same frame and the same slice build the same frontier;
no miner's head enters it.

Valuation uses the frame's posterior mean, as `reference_frame.best_model`
does and for the same reason: a lower bound shrinks with sample count and so
would make the frontier a function of how many miners are online. The frame's
own warmth is instead expressed as a confidence factor on the score — see
`Frontier.confidence` — so that a frontier built from the prior alone pays
nobody much until it has been measured.
"""
from __future__ import annotations

from dataclasses import dataclass

from fugal_subnet.config import FRONTIER_MIN_TRIALS


@dataclass(frozen=True)
class Frontier:
    """Upper convex hull of (cost_per_question, accuracy) over constant policies.

    `points` are sorted by cost and start at the origin. `least_trials` is the
    decayed exploration-trial count of the least-observed model on the hull,
    which is how well the frontier itself is known.
    """

    points: tuple[tuple[float, float], ...]
    least_trials: float = 0.0
    models: tuple[str, ...] = ()

    @property
    def max_accuracy(self) -> float:
        return max((a for _, a in self.points), default=0.0)

    @property
    def confidence(self) -> float:
        """How much of a miner's headroom is paid, given how well the frontier
        is known. 0 with no exploration evidence at all, 1 once every model on
        the hull has FRONTIER_MIN_TRIALS decayed samples.

        With a cold frame every model sits at the prior, the hull is a flat
        line at the prior, and a constant policy on any decent model shows
        "headroom" above it. That headroom is an artifact of ignorance, not a
        fact about routing, and it must not be paid at face value.
        """
        if FRONTIER_MIN_TRIALS <= 0:
            return 1.0
        return max(0.0, min(1.0, self.least_trials / FRONTIER_MIN_TRIALS))

    def accuracy_at(self, cost_per_question: float) -> float:
        """Best accuracy a constant policy achieves at this cost or less."""
        pts = self.points
        if not pts:
            return 0.0
        c = max(0.0, float(cost_per_question))
        if c >= pts[-1][0]:
            return pts[-1][1]
        for (c0, a0), (c1, a1) in zip(pts, pts[1:]):
            if c0 <= c <= c1:
                if c1 - c0 <= 1e-18:
                    return max(a0, a1)
                t = (c - c0) / (c1 - c0)
                return a0 + t * (a1 - a0)
        return pts[0][1]


def build_frontier(
    frame,
    prices: dict[str, tuple[float, float]],
    prompt_tokens_total: float,
    n_questions: int,
    default_completion_tokens: float,
) -> Frontier:
    """Build the frontier for one epoch's slice.

    `prompt_tokens_total` is the slice's input tokens (a property of the
    questions, from the pool, identical for every miner); each model's cost is
    what routing the whole slice to it would cost under the pinned rates and the
    frame's measured completion length for that model, divided by the number of
    questions so the axis is cost per question.
    """
    from fugal_subnet.reference_frame import reference_cost

    n = max(1, int(n_questions))
    raw: list[tuple[float, float, str]] = []
    for model in sorted(prices):
        cost = reference_cost(
            frame, prices, model, prompt_tokens_total, n, default_completion_tokens,
        ) / n
        raw.append((max(0.0, cost), float(frame.accuracy(model)), model))
    return _hull(raw, frame)


def _hull(raw: list[tuple[float, float, str]], frame) -> Frontier:
    """Upper convex hull including the origin, made flat after its peak."""
    pts = sorted([(0.0, 0.0, "")] + raw, key=lambda p: (p[0], -p[1]))
    hull: list[tuple[float, float, str]] = []
    for p in pts:
        # Pop while the last two hull points and p make a non-concave turn.
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        # Same cost as the previous hull point: keep only the higher accuracy.
        if hull and abs(hull[-1][0] - p[0]) <= 1e-18:
            if p[1] > hull[-1][1]:
                hull[-1] = p
            continue
        hull.append(p)
    # Flat after the peak: more spend never lowers what a constant policy can do,
    # because it can always spend less.
    out: list[tuple[float, float, str]] = []
    best = -1.0
    for c, a, m in hull:
        if a < best:
            break
        best = a
        out.append((c, a, m))
    models = tuple(m for _, _, m in out if m)
    least = min((frame.trials.get(m, 0.0) for m in models), default=0.0)
    return Frontier(points=tuple((c, a) for c, a, _ in out), least_trials=least, models=models)


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def headroom(accuracy_lcb: float, cost_per_question: float, frontier: Frontier) -> float:
    """Accuracy above what any constant policy buys at this cost. May be negative."""
    return float(accuracy_lcb) - frontier.accuracy_at(cost_per_question)
