"""Nonce-seeded, stratified benchmark slicer.

Deterministically selects a subset of questions from the benchmark pool.
Same nonce + same pool always produces the identical slice; different nonces
produce different slices. Selection is stratified by benchmark: each benchmark
gets an (as equal as possible) share of the slice, so large pools (MMLU is 14k
of ~21k questions) cannot dominate the routing signal. Within each benchmark,
questions are ranked by HMAC-SHA256 keyed on the nonce — unpredictable without
knowing the nonce.
"""
from __future__ import annotations

import hashlib
import hmac


def select_slice(
    nonce: bytes,
    pool: list[dict],
    size: int = 300,
) -> list[dict]:
    """Select a deterministic, nonce-dependent, benchmark-stratified slice.

    Args:
        nonce: Random bytes (typically derived from block hash + epoch_id).
        pool: Full benchmark pool — list of question dicts with 'question_id'
              and 'benchmark'.
        size: Number of questions to select. Clamped to len(pool).

    Returns:
        List of question dicts, grouped by benchmark (sorted), HMAC-ranked
        within each benchmark. Quotas are filled round-robin so a small
        benchmark that runs out cedes its remaining slots to the others.
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

    quotas = {bench: 0 for bench in benches}
    remaining = size
    while remaining > 0:
        progressed = False
        for bench in benches:
            if remaining == 0:
                break
            if quotas[bench] < len(ranked[bench]):
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
