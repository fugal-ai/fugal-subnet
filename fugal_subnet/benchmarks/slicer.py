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


def derive_nonce(epoch_id: str, block_hash: str) -> bytes:
    """Derive the epoch nonce from the epoch identifier and a block hash.

    The block hash makes the nonce unpredictable before the block lands.
    The epoch_id ensures different epochs get different nonces even if
    they happen to reference the same block.
    """
    payload = f"{epoch_id}:{block_hash}".encode("utf-8")
    return hashlib.sha256(payload).digest()
