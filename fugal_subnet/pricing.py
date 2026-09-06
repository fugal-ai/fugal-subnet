"""Cost of a routing POLICY, computed from shared data alone.

`thrift = reference_cost / miner_cost`. The denominator was already built this
way — pinned rates plus the reference frame's measured per-model verbosity — and
the numerator came from whatever the miner reported it had spent. One side of
the ratio was unfakeable and the other was a self-report, which is the bug this
module closes.

WHAT IS BEING PRICED. The product is a routing policy, and its beneficiary is a
future user of the head, not the miner. A user cares what the policy costs
*them*; what one miner happened to spend on one benchmark run in one hour is an
artifact of that run. So the scored cost depends only on WHICH MODEL WAS CHOSEN
for each question — the thing actually being measured — and on nothing the miner
can vary.

Every input is already consensus state:

    tokens_in(question)          the question pool
    rate_in, rate_out(model)     the pinned price table
    typical completion(model)    the reference frame, pooled across miners and
                                 decayed over time

The miner's own reported costs remain in the proof and are still checked for
internal consistency, because a miner lying about them is worth knowing about.
They simply no longer decide anyone's score.

This is an estimate and not an invoice, and it is not trying to be one. The
pinned table already diverges from real billing on purpose, so that two miners
benchmarking hours apart are measured against the same denominator.
"""
from __future__ import annotations

# Characters per token. A rough, fixed, universal convention rather than a real
# tokenisation, chosen deliberately:
#
#   - every model tokenises differently, so there is no single true count and
#     precision here would be a fiction with extra steps;
#   - a validator must compute this without loading a tokeniser, since it never
#     loads the backbone and should not gain a dependency to price a proof;
#   - the figure is a RELATIVE signal. What matters is that a longer question
#     costs more and that every validator agrees on how much, both of which a
#     fixed divisor gives exactly.
#
# Changing it re-scores every miner, so it is consensus state and lives here
# rather than being passed in.
CHARS_PER_TOKEN = 4


def question_input_tokens(prompt: str) -> int:
    """Input tokens attributed to a question. Deterministic, tokeniser-free."""
    return max(1, round(len(prompt or "") / CHARS_PER_TOKEN))


def policy_cost(
    prompt: str,
    model: str,
    prices: dict[str, tuple[float, float]],
    completion_tokens: float,
) -> float:
    """Cost of routing one question to one model, under the published prices."""
    rate_in, rate_out = prices[model]
    return question_input_tokens(prompt) * rate_in + completion_tokens * rate_out


def policy_cost_total(
    routed: list[tuple[str, str]],
    gold_answers: dict[str, dict],
    prices: dict[str, tuple[float, float]],
    frame,
    default_completion_tokens: float,
) -> float:
    """Total policy cost for (question_id, model) pairs.

    Unknown questions and unpriced models contribute nothing rather than raising:
    the caller has already rejected a proof whose results are not the assigned
    slice, and an unpriced model is caught upstream by the harness, so reaching
    either here means something else already failed and this should not be the
    thing that reports it.
    """
    total = 0.0
    for qid, model in routed:
        q = gold_answers.get(qid)
        if q is None or model not in prices:
            continue
        completion = (
            frame.avg_completion_tokens(model, default_completion_tokens)
            if frame is not None else default_completion_tokens
        )
        total += policy_cost(q.get("prompt", ""), model, prices, completion)
    return total
