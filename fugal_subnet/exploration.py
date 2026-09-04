"""Nonce-seeded exploration: recovering the counterfactual a TEE destroys.

Under the TEE architecture a miner only ever calls the one model it routed to,
so nothing in its proof says what a *different* model would have done. That
missing counterfactual is why cost efficiency had no honest denominator: with
no observation of any other model, the only quantities available came from the
miner's own proof, and a self-referential denominator is trivially gamed.

The fix is to buy the counterfactual cheaply. Each epoch a miner additionally
answers a small set of questions using a model it does not choose — both the
questions and the models are derived from the epoch nonce. Those answers are
never scored against the miner (a forced random route must not be a penalty);
they exist only to sample the model pool.

Three properties make this safe:

- **Unpredictable.** Targets derive from the block-hash nonce, so a miner
  cannot prepare for them any more than it can prepare for the slice.
- **Not miner-chosen.** The target model comes from the globally agreed price
  table, not the miner's head, so no miner can steer the sample toward a model
  that flatters it.
- **Not optional.** Because the assignment is deterministic and attested, a
  proof with a missing or mis-targeted exploration set is rejected. Skipping
  the ~5% cost is not available.

Pooled across miners and across epochs, these samples are what the reference
frame is built from — see `fugal_subnet/reference_frame.py`.
"""
from __future__ import annotations

import hashlib
import hmac

from fugal_subnet.benchmarks.slicer import select_slice


def explore_nonce(nonce: bytes) -> bytes:
    """Domain-separated nonce for exploration question selection."""
    return hashlib.sha256(nonce + b":explore").digest()


def select_explore_set(
    nonce: bytes,
    pool: list[dict],
    scored_ids: set[str],
    size: int,
) -> list[dict]:
    """Choose exploration questions, disjoint from the scored slice.

    Disjoint so the scored slice stays a full SLICE_SIZE and a miner is never
    graded on a question it was forced to misroute. Selected with the same
    stratified sampler as the scored slice, so exploration samples come from
    the same distribution the miner is judged on — otherwise the reference
    frame would describe a different question population than the one being
    scored against it.
    """
    remaining = [q for q in pool if q["question_id"] not in scored_ids]
    if not remaining:
        return []
    return select_slice(explore_nonce(nonce), remaining, size)


def explore_target(nonce: bytes, question_id: str, models: list[str]) -> str:
    """The model this exploration question must be routed to.

    `models` must be the globally agreed model list (sorted price-table keys),
    never the miner's declared head models — otherwise a miner chooses its own
    exploration targets and the sample stops being unbiased.
    """
    if not models:
        raise ValueError("explore_target requires a non-empty model list")
    digest = hmac.new(
        nonce, f"target:{question_id}".encode("utf-8"), hashlib.sha256,
    ).digest()
    return models[int.from_bytes(digest[:8], "big") % len(models)]


def expected_exploration(
    nonce: bytes,
    pool: list[dict],
    scored_ids: set[str],
    models: list[str],
    size: int,
) -> dict[str, str]:
    """{question_id: required_model} the miner must have explored this epoch.

    Both sides compute this independently from public inputs, so the validator
    never has to trust the miner's account of what it was asked to do.
    """
    return {
        q["question_id"]: explore_target(nonce, q["question_id"], models)
        for q in select_explore_set(nonce, pool, scored_ids, size)
    }
