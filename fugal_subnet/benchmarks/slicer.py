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


def validate_activation_block(activation_block: int | None, blocks_per_epoch: int) -> None:
    """Refuse a v2 activation block that is not an epoch boundary.

    Unset (None) is valid and means v2 never activates. A block inside an
    epoch would make that epoch's rules depend on which side of it a
    validator looked, which is the divergence a boundary-keyed version exists
    to prevent, so it is refused at startup rather than rounded.
    """
    if activation_block is None:
        return
    bpe = max(1, int(blocks_per_epoch))
    if isinstance(activation_block, bool) or not isinstance(activation_block, int):
        raise ValueError(f"v2 activation block must be an integer, got {activation_block!r}")
    if activation_block < 0:
        raise ValueError(f"v2 activation block must not be negative, got {activation_block}")
    if activation_block % bpe != 0:
        raise ValueError(
            f"v2 activation block {activation_block} is not an epoch boundary "
            f"(a multiple of {bpe} blocks); the nearest are "
            f"{activation_block // bpe * bpe} and {(activation_block // bpe + 1) * bpe}"
        )


def protocol_version_for_boundary(boundary_block: int, activation_block: int | None) -> int:
    """The protocol version an epoch is scored under: 1 or 2.

    Consensus-critical, and deliberately keyed on the epoch's BOUNDARY block:
    every block inside one epoch maps to the same boundary, so a retried
    epoch, or a validator that reaches it late, applies the same rules as
    every peer. v2 applies from the activation boundary itself onward.
    """
    if activation_block is None:
        return 1
    return 2 if int(boundary_block) >= int(activation_block) else 1


def derive_nonce(epoch_id: str, block_hash: str) -> bytes:
    """Derive the epoch nonce from the epoch identifier and a block hash.

    The block hash makes the nonce unpredictable before the block lands.
    The epoch_id ensures different epochs get different nonces even if
    they happen to reference the same block.
    """
    payload = f"{epoch_id}:{block_hash}".encode("utf-8")
    return hashlib.sha256(payload).digest()


# ---------------------------------------------------------------------------
# Protocol v2: category-first allocation, scored and exploration slices alike.
#
# v1 above caps each benchmark at max(1, int(share_factor * size * share)) and
# fills round-robin. At the scored size (300) that is fine; at the exploration
# size (15) the floor of 1 dominates. Measured on mainnet: five benchmarks got
# exploration 1:1:1:5:7, so a benchmark that was 1.7% of the scored slice was
# 6.7% of exploration and 4x over-represented in the reference frame. With
# more, smaller benchmarks the distortion grows: most of 15 slots would be
# floor-allocated whatever the scored mix.
#
# v2 allocates by CATEGORY first, at published targets, by largest remainder
# with no floor; then, inside a category:
#   * the scored slice splits across benchmarks by largest remainder on pool
#     share, under the same share_factor cap (diligence finding 6), now applied
#     within the category so a small pool still cannot dominate it;
#   * each exploration slot draws its benchmark by a nonce-keyed weighted draw
#     over the category's benchmarks, weighted by pool share within the
#     category, so a benchmark's EXPECTED exploration share equals its expected
#     scored share, with no floor at any size.
# Ties in largest remainder break on HMAC(nonce, key), so no category or
# benchmark is systematically favoured and the expected mix is exact over
# epochs. Exact per-epoch agreement between the two shares is impossible at 15
# slots; `mix_weights` gives the validator the per-benchmark correction factor
# (scored share / exploration share) so the frame can weight what it observed.
# ---------------------------------------------------------------------------


def _hmac_rank(nonce: bytes, key: str) -> str:
    return hmac.new(nonce, key.encode("utf-8"), hashlib.sha256).hexdigest()


def _largest_remainder(total: int, weights: dict[str, float], nonce: bytes, salt: str) -> dict[str, int]:
    """Integer quotas summing to `total`, proportional to `weights`, no floor.

    Floors first, then the leftover units go to the largest fractional parts;
    equal fractions are ordered by HMAC(nonce, salt:key) so ties rotate with the
    nonce instead of always favouring the same key. Deterministic.
    """
    keys = sorted(k for k, w in weights.items() if w > 0)
    if total <= 0 or not keys:
        return {k: 0 for k in weights}
    wsum = float(sum(weights[k] for k in keys))
    exact = {k: total * weights[k] / wsum for k in keys}
    quotas = {k: int(exact[k]) for k in keys}
    left = total - sum(quotas.values())
    order = sorted(keys, key=lambda k: (-(exact[k] - quotas[k]), _hmac_rank(nonce, f"{salt}:{k}")))
    for k in order[:left]:
        quotas[k] += 1
    return {k: quotas.get(k, 0) for k in weights}


def _ranked_by_bench(nonce: bytes, pool: list[dict]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for item in pool:
        by.setdefault(item.get("benchmark", ""), []).append(item)
    return {b: sorted(items, key=lambda it: _hmac_rank(nonce, it["question_id"])) for b, items in by.items()}


def _categories_of(ranked: dict[str, list[dict]], bench_category: dict[str, str]) -> dict[str, list[str]]:
    missing = sorted(b for b in ranked if b not in bench_category)
    if missing:
        raise ValueError(f"benchmarks with no category: {missing}")
    cats: dict[str, list[str]] = {}
    for b in sorted(ranked):
        cats.setdefault(bench_category[b], []).append(b)
    return cats


def _within_category_shares(benches: list[str], ranked: dict[str, list[dict]],
                            bench_weight: dict[str, float] | None) -> dict[str, float]:
    """Each benchmark's share of its category: pool size by default, or explicit
    weights (`bench_weight`) when the category is weighted. A weighted category
    must weight EVERY one of its benchmarks; a partial map would mix two
    meanings of "share" and is refused rather than guessed at."""
    given = [b for b in benches if bench_weight and b in bench_weight]
    if given and len(given) != len(benches):
        missing = sorted(set(benches) - set(given))
        raise ValueError(f"category weights must cover every benchmark in it; missing {missing}")
    raw = {b: (float(bench_weight[b]) if given else float(len(ranked[b]))) for b in benches}
    if any(w < 0 for w in raw.values()) or sum(raw.values()) <= 0:
        raise ValueError(f"category weights must be non-negative with a positive sum: {raw}")
    total = sum(raw.values())
    return {b: raw[b] / total for b in benches}


def category_quotas(size: int, targets: dict[str, float], present: set[str], nonce: bytes, salt: str) -> dict[str, int]:
    """Slots per category: largest remainder on the published targets, over the
    categories actually present in the pool (targets renormalised over them)."""
    return _largest_remainder(size, {c: targets.get(c, 0.0) for c in sorted(present)}, nonce, salt)


def select_slice_v2(
    nonce: bytes,
    pool: list[dict],
    size: int,
    targets: dict[str, float],
    bench_category: dict[str, str],
    share_factor: float = SLICE_SHARE_FACTOR,
    bench_weight: dict[str, float] | None = None,
) -> list[dict]:
    """The v2 scored slice: category quotas at `targets`, then benchmark quotas
    by share within each category (pool size, or `bench_weight` where a
    category is weighted) under the share_factor cap. A quota a
    benchmark cannot fill (pool exhausted, or capped) passes to the others in
    its category; a category that cannot fill passes its remainder to the
    other categories in proportion to their targets. Deterministic in
    (nonce, pool, size, targets, bench_category)."""
    if not pool:
        return []
    size = min(size, len(pool))
    ranked = _ranked_by_bench(nonce, pool)
    cats = _categories_of(ranked, bench_category)
    cq = category_quotas(size, targets, set(cats), nonce, "scored")

    take: dict[str, int] = {b: 0 for b in ranked}
    unfilled = 0
    for c in sorted(cats):
        benches = cats[c]
        csize = sum(len(ranked[b]) for b in benches)
        want = min(cq[c], csize)
        unfilled += cq[c] - want
        share = _within_category_shares(benches, ranked, bench_weight)
        caps = {b: min(len(ranked[b]), max(1, int(share_factor * want * share[b]))) for b in benches}
        q = _largest_remainder(want, share, nonce, f"scored:{c}")
        q = {b: min(q[b], caps[b]) for b in benches}
        # Redistribute what the caps cut, inside the category: first under the caps, then ignoring them.
        for limit in (caps, {b: len(ranked[b]) for b in benches}):
            short = want - sum(q.values())
            while short > 0:
                room = sorted((b for b in benches if q[b] < limit[b]),
                              key=lambda b: (-(share[b] * want - q[b]), _hmac_rank(nonce, f"fill:{c}:{b}")))
                if not room:
                    break
                q[room[0]] += 1
                short -= 1
        for b in benches:
            take[b] = q[b]
    # A category whose pool ran out: hand its slots to the others, in proportion to their targets.
    while unfilled > 0:
        room = {c: targets.get(c, 0.0) for c in cats
                if sum(take[b] for b in cats[c]) < sum(len(ranked[b]) for b in cats[c])}
        if not room:
            break
        extra = _largest_remainder(unfilled, room, nonce, "overflow")
        moved = 0
        for c in sorted(extra):
            for _ in range(extra[c]):
                free = [b for b in cats[c] if take[b] < len(ranked[b])]
                if not free:
                    break
                b = max(free, key=lambda b: (len(ranked[b]) - take[b], b))
                take[b] += 1
                moved += 1
        if moved == 0:
            break
        unfilled -= moved

    out: list[dict] = []
    for b in sorted(ranked):
        out.extend(ranked[b][:take[b]])
    return out


def _weighted_pick(nonce: bytes, key: str, weights: list[tuple[str, float]]) -> str:
    """One deterministic weighted draw: a uniform from HMAC(nonce, key) against
    the cumulative weights, in sorted key order."""
    total = sum(w for _, w in weights)
    u = int(_hmac_rank(nonce, key), 16) / float(1 << 256) * total
    acc = 0.0
    for name, w in weights:
        acc += w
        if u < acc:
            return name
    return weights[-1][0]


def select_explore_set_v2(
    nonce: bytes,
    pool: list[dict],
    scored_ids: set[str],
    size: int,
    targets: dict[str, float],
    bench_category: dict[str, str],
    bench_weight: dict[str, float] | None = None,
) -> list[dict]:
    """The v2 exploration set, disjoint from the scored slice: category quotas
    at `targets` by largest remainder (no floor), then each slot's benchmark by
    a nonce-keyed draw weighted by pool share within the category. Expected
    exploration share per benchmark therefore equals its expected scored share.
    `nonce` is the exploration nonce (exploration.explore_nonce)."""
    remaining = [q for q in pool if q["question_id"] not in scored_ids]
    if not remaining or size <= 0:
        return []
    ranked = _ranked_by_bench(nonce, remaining)
    full = _ranked_by_bench(nonce, pool)         # weights use the WHOLE pool's shares, like the scored slice
    cats = _categories_of(ranked, bench_category)
    cq = category_quotas(min(size, len(remaining)), targets, set(cats), nonce, "explore")
    used = {b: 0 for b in ranked}
    out: list[dict] = []
    for c in sorted(cats):
        # The scored slice's within-category shares (from the WHOLE pool), so the
        # expected exploration share equals the expected scored share.
        shares = _within_category_shares(sorted(cats[c]), full, bench_weight)
        weights = [(b, shares[b]) for b in cats[c]]
        for slot in range(cq[c]):
            live = [(b, w) for b, w in weights if used[b] < len(ranked[b])]
            if not live:
                break
            b = _weighted_pick(nonce, f"explore:{c}:{slot}", live)
            out.append(ranked[b][used[b]])
            used[b] += 1
    return out


def mix_weights(scored: list[dict], explored: list[dict]) -> dict[str, float]:
    """Per-benchmark correction for the reference frame: scored share ÷
    exploration share in this epoch. An exploration observation of benchmark b
    is weighted by mix_weights[b], so the frame estimates each model on the
    scored mix even when one epoch's 15 slots round away from it. Benchmarks
    not explored this epoch are absent (nothing to weight)."""
    if not scored or not explored:
        return {}
    sc: dict[str, int] = {}
    ex: dict[str, int] = {}
    for q in scored:
        sc[q.get("benchmark", "")] = sc.get(q.get("benchmark", ""), 0) + 1
    for q in explored:
        ex[q.get("benchmark", "")] = ex.get(q.get("benchmark", ""), 0) + 1
    return {b: (sc.get(b, 0) / len(scored)) / (n / len(explored)) for b, n in sorted(ex.items())}
