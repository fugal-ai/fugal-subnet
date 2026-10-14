"""Nonce-seeded, stratified benchmark slicer.

Deterministically selects a subset of questions from the benchmark pool.
Same nonce + same pool always produces the identical slice; different nonces
produce different slices. Selection is stratified by benchmark: slots are
filled round-robin across benchmarks so large pools (MMLU is 14k of ~21k
questions) cannot dominate the routing signal -- but a benchmark's share is
CAPPED at SLICE_SHARE_FACTOR times its share of the pool, so small pools
cannot dominate the score either. Equal quotas used to give three benchmarks
totalling 2,511 questions 60% of every slice, and 2,511 questions is about
what a linear head memorises outright (diligence 2026-10, finding 6). Within
each benchmark, questions are ranked by HMAC-SHA256 keyed on the nonce --
unpredictable without knowing the nonce.
"""
from __future__ import annotations

import hashlib
import hmac

from fugal_subnet.config import SLICE_SHARE_FACTOR


def select_slice(
    nonce: bytes,
    pool: list[dict],
    size: int = 300,
    share_factor: float = SLICE_SHARE_FACTOR,
) -> list[dict]:
    """Select a deterministic, nonce-dependent, benchmark-stratified slice.

    Args:
        nonce: Random bytes (typically derived from block hash + epoch_id).
        pool: Full benchmark pool — list of question dicts with 'question_id'
              and 'benchmark'.
        size: Number of questions to select. Clamped to len(pool).
        share_factor: A benchmark's quota is capped at
              share_factor * size * (its pool share), at least 1.

    Returns:
        List of question dicts, grouped by benchmark (sorted), HMAC-ranked
        within each benchmark. Quotas are filled round-robin under the caps,
        so a small benchmark that runs out, or hits its cap, cedes its
        remaining slots to the others. If every benchmark is capped before
        the slice is full (rounding), a second pass fills it ignoring caps.
    """
    if not pool:
        return []
    size = min(size, len(pool))

    by_bench: dict[str, list[dict]] = {}
    for item in pool:
        by_bench.setdefault(item.get("benchmark", ""), []).append(item)
    benches = sorted(by_bench)

    ranked: dict[str, list[dict]] = {}
    for bench in benches:
        scored = sorted(
            (
                (hmac.new(nonce, item["question_id"].encode("utf-8"),
                          hashlib.sha256).hexdigest(), item)
                for item in by_bench[bench]
            ),
            key=lambda x: x[0],
        )
        ranked[bench] = [item for _, item in scored]

    caps = {
        bench: max(1, int(share_factor * size * len(ranked[bench]) / len(pool)))
        for bench in benches
    }
    quotas = {bench: 0 for bench in benches}
    remaining = size
    for limit in (caps, {bench: len(ranked[bench]) for bench in benches}):
        while remaining > 0:
            progressed = False
            for bench in benches:
                if remaining == 0:
                    break
                if quotas[bench] < min(limit[bench], len(ranked[bench])):
                    quotas[bench] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break

    out: list[dict] = []
    for bench in benches:
        out.extend(ranked[bench][:quotas[bench]])
    return out


# Nominal chain block time. Epoch geometry is defined in BLOCKS, so both
# neurons agree regardless of how fast a given chain actually produces them —
# a devnet at 0.35s/block and mainnet at 12s/block both give the same epoch
# boundaries. What must never differ is this arithmetic, which is why it lives
# here rather than being restated in each neuron.
BLOCK_TIME_S = 12


def blocks_per_epoch(epoch_interval_s: int) -> int:
    """Blocks in one epoch. Consensus-critical: both neurons must agree."""
    return max(1, int(epoch_interval_s) // BLOCK_TIME_S)


def epoch_id_for_block(epoch_index: int) -> str:
    """Canonical epoch identifier for a block-derived epoch index.

    Consensus-critical and deliberately the only way to build an epoch_id.
    The nonce is sha256(f"{epoch_id}:{block_hash}"), so a miner and a validator
    that format the identifier differently derive different nonces, select
    different slices, and every proof fails the nonce check. That is exactly
    what happened when the miner used the block hash and the validator used the
    epoch index — the two never agreed on a single question.
    """
    return f"e{int(epoch_index):08d}"


def collect_block_for_epoch(
    epoch_index: int, blocks_per_epoch: int, fraction: float,
) -> int:
    """The block at which validators collect proofs for `epoch_index`.

    Consensus-critical and, like epoch_id_for_block, deliberately the only way
    to compute it. Every honest validator collects at the same block, so they
    see the same set of proofs; a validator using a different offset samples a
    different field and publishes different weights for honest reasons, which
    is exactly the class of divergence that is impossible to diagnose after the
    fact. Clamped into (boundary, boundary + blocks_per_epoch) so a misconfigured
    fraction cannot collect before the epoch exists or after it has ended.
    """
    bpe = max(1, int(blocks_per_epoch))
    boundary = int(epoch_index) * bpe
    offset = int(bpe * float(fraction))
    offset = max(1, min(offset, bpe - 1)) if bpe > 1 else 0
    return boundary + offset


def epoch_index_for_block(block: int, blocks_per_epoch: int) -> int:
    """Epoch index containing `block`. Shared so both neurons align epochs."""
    return int(block) // max(1, int(blocks_per_epoch))


def derive_nonce(epoch_id: str, block_hash: str) -> bytes:
    """Derive the epoch nonce from the epoch identifier and a block hash.

    The block hash makes the nonce unpredictable before the block lands.
    The epoch_id ensures different epochs get different nonces even if
    they happen to reference the same block.
    """
    payload = f"{epoch_id}:{block_hash}".encode("utf-8")
    return hashlib.sha256(payload).digest()
