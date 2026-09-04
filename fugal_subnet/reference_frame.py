"""The reference frame a miner's score is measured against.

A score needs a reference: what does good look like, and what does it cost?
The critical design constraint is that the reference must be a fact about the
*world* — this question pool, this model pool — and never a function of the
miner population. If the frame moved with the field, a miner's score would
change because other miners came online or went dark, which is exactly the
non-interference property (I4) the TEE architecture exists to protect. It
would also make scores incomparable across time.

So the frame is accumulated over TIME, not over miners: exploration samples
from every epoch decay into per-model counts, and any single miner's
contribution to those counts is negligible.

**The ceiling is the best single model, not a per-question oracle.** Two
reasons. First, it is the actual product claim — "match frontier quality at a
fraction of the cost" — so a score of 1.0 means "matched the best model's
quality per dollar" and >1.0 means "beat it", which is a statement worth
making. Second, it needs only per-model marginals (~30 numbers) rather than a
dense question-by-model matrix (~21K x 30 cells), so it is well-estimated
within a few epochs instead of many hundreds, and it is stable at any field
size. A per-question oracle is the more precise measure of pure routing skill,
but it is unreachable in practice and compresses every real router into a
narrow low band.

Cold start is handled by a Beta prior worth K pseudo-observations, not by a
special case: with no evidence the frame reports the prior, and the prior
washes out as samples arrive. Nothing divides by zero and there is no epoch at
which the frame is undefined.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field

from fugal_subnet.config import (
    FRAME_HALF_LIFE,
    FRAME_PRIOR_ACCURACY,
    FRAME_PRIOR_STRENGTH,
    WILSON_CONFIDENCE,
)

logger = logging.getLogger(__name__)

_Z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}


@dataclass
class ReferenceFrame:
    """Decayed per-model evidence pooled from exploration samples."""

    successes: dict[str, float] = field(default_factory=dict)
    trials: dict[str, float] = field(default_factory=dict)
    prompt_tokens: dict[str, float] = field(default_factory=dict)
    completion_tokens: dict[str, float] = field(default_factory=dict)
    epochs: int = 0

    def to_dict(self) -> dict:
        return {
            "successes": self.successes,
            "trials": self.trials,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "epochs": self.epochs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ReferenceFrame:
        return cls(
            successes=dict(d.get("successes") or {}),
            trials=dict(d.get("trials") or {}),
            prompt_tokens=dict(d.get("prompt_tokens") or {}),
            completion_tokens=dict(d.get("completion_tokens") or {}),
            epochs=int(d.get("epochs", 0)),
        )

    def accuracy(self, model: str) -> float:
        """Posterior mean accuracy under the Beta prior."""
        s, n = self._posterior_counts(model)
        return s / n if n > 0 else FRAME_PRIOR_ACCURACY

    def accuracy_lcb(self, model: str) -> float:
        """Lower confidence bound on accuracy.

        The ceiling is selected on the LOWER bound so a model cannot become the
        reference on a lucky handful of samples. A model has to earn the
        position with evidence.
        """
        s, n = self._posterior_counts(model)
        if n <= 0:
            return 0.0
        return _wilson_lcb(s / n, n, WILSON_CONFIDENCE)

    def avg_completion_tokens(self, model: str, default: float) -> float:
        n = self.trials.get(model, 0.0)
        if n < 1e-9:
            return default
        return self.completion_tokens.get(model, 0.0) / n

    def _posterior_counts(self, model: str) -> tuple[float, float]:
        n = self.trials.get(model, 0.0)
        s = self.successes.get(model, 0.0)
        k = float(FRAME_PRIOR_STRENGTH)
        return s + k * FRAME_PRIOR_ACCURACY, n + k


def decay_factor(half_life: int) -> float:
    return 2.0 ** (-1.0 / max(1, half_life))


def accumulate_exploration(
    frame: ReferenceFrame | None,
    samples: list[tuple[str, bool, int, int]],
    half_life: int = FRAME_HALF_LIFE,
) -> ReferenceFrame:
    """Fold one epoch's exploration samples into the frame.

    `samples` are (model_id, correct, prompt_tokens, completion_tokens) tuples
    pooled from every verified proof in the epoch. Sample order does not affect
    the result — all terms are sums — so validators processing proofs in any
    order reach identical frames.
    """
    alpha = decay_factor(half_life)
    prev = frame or ReferenceFrame()

    out = ReferenceFrame(
        successes={m: v * alpha for m, v in prev.successes.items()},
        trials={m: v * alpha for m, v in prev.trials.items()},
        prompt_tokens={m: v * alpha for m, v in prev.prompt_tokens.items()},
        completion_tokens={m: v * alpha for m, v in prev.completion_tokens.items()},
        epochs=prev.epochs + 1,
    )
    for model, correct, p_tok, c_tok in samples:
        out.trials[model] = out.trials.get(model, 0.0) + 1.0
        out.successes[model] = out.successes.get(model, 0.0) + (1.0 if correct else 0.0)
        out.prompt_tokens[model] = out.prompt_tokens.get(model, 0.0) + float(p_tok)
        out.completion_tokens[model] = out.completion_tokens.get(model, 0.0) + float(c_tok)
    return out


def best_model(
    frame: ReferenceFrame,
    prices: dict[str, tuple[float, float]],
) -> tuple[str, float]:
    """The reference model, and the ceiling accuracy to score against.

    Selection and valuation deliberately use different statistics:

    - **Selected** on the accuracy lower bound, so a model cannot be crowned
      the reference on a lucky handful of samples. It has to earn the position.
    - **Valued** at the posterior mean, because the ceiling is an estimate of a
      fixed quantity — how good that model actually is.

    Using the lower bound for the value too was a real bug: an LCB is
    deliberately pessimistic, and its pessimism shrinks as evidence
    accumulates, so the ceiling became a function of how many samples existed
    and therefore of how many miners were online. Measured, that moved a fixed
    head's score by 0.14 between a 3-miner and a 50-miner field — exactly the
    non-interference property (I4) the frame exists to preserve. The posterior
    mean is a consistent estimator of the truth and carries no such systematic
    drift with sample count.

    Ties break toward the cheaper model, which is what makes cold start
    coherent: with no evidence every model shares the prior, and the right
    reference for "quality per dollar" among equals is the cheapest one.
    """
    if not prices:
        raise ValueError("cannot pick a reference model from an empty price table")

    def rank(model: str) -> tuple[float, float]:
        p_in, p_out = prices[model]
        return (-frame.accuracy_lcb(model), p_in + p_out)

    best = min(sorted(prices), key=rank)
    return best, frame.accuracy(best)


def reference_cost(
    frame: ReferenceFrame,
    prices: dict[str, tuple[float, float]],
    model: str,
    prompt_tokens: float,
    n_questions: int,
    default_completion_tokens: float,
) -> float:
    """What the reference model would have cost on this question set.

    Prompt tokens come from the miner's attested counts — a property of the
    questions, shared by any model answering them. Completion tokens come from
    the frame's observed average for the reference model, because a different
    model genuinely writes a different amount.
    """
    p_in, p_out = prices[model]
    c_tok = frame.avg_completion_tokens(model, default_completion_tokens)
    return p_in * prompt_tokens + p_out * c_tok * max(n_questions, 0)


def load_bootstrap(path: str | None = None) -> ReferenceFrame:
    """Load the shipped bootstrap frame, or an empty one if absent.

    The bootstrap only shifts where the prior sits; it is worth
    FRAME_PRIOR_STRENGTH pseudo-observations and is outweighed by real evidence
    within a few epochs.
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "reference_frame.json",
        )
    try:
        with open(path, encoding="utf-8") as f:
            return ReferenceFrame.from_dict(json.load(f))
    except FileNotFoundError:
        logger.info("No bootstrap reference frame at %s — starting from the prior", path)
        return ReferenceFrame()


def _wilson_lcb(p: float, n: float, confidence: float) -> float:
    if n < 1e-9:
        return 0.0
    z = _Z.get(confidence, 1.96)
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(max(0.0, (p * (1 - p) + z * z / (4 * n)) / n))
    return max(0.0, (center - spread) / denom)
