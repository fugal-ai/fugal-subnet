"""Diligence 2026-10, finding 6: the slicer makes 60% of every score memorisable.

The memorisation defence is head capacity against POOL size: a linear head can
fit about 16.9% of random labels over the full 21,553-question pool. But
`select_slice` gives each of the five benchmarks an equal 60 of 300 slots, so
AIME (933), GSM8K (1,319) and IFEval (259) -- 2,511 questions, a set the
team's own table says a linear head memorises completely -- fill 180 of 300
slots, 60% of every score. Exploration uses the same slicer.

Delivered by `SLICE_SHARE_FACTOR` in `select_slice`: a benchmark's quota is
capped at the factor times its share of the pool. The numbers behind the
factor are in docs/HEAD_EFFICACY.md, experiment 4.
"""
from __future__ import annotations

from fugal_subnet.benchmarks.slicer import select_slice

# benchmarks/pool_manifest.json per_benchmark, at the pinned pool.
POOL_SIZES = {"aime": 933, "gsm8k": 1319, "ifeval": 259, "math": 5000, "mmlu": 14042}
SMALL = {"aime", "gsm8k", "ifeval"}


def _pool():
    return [{"question_id": f"{b}:{i}", "benchmark": b}
            for b, n in POOL_SIZES.items() for i in range(n)]


def test_small_benchmarks_cannot_fill_most_of_the_slice():
    pool = _pool()
    for nonce in (b"\x00" * 32, b"\x01" * 32, b"nonce-three"):
        chosen = select_slice(nonce, pool, 300)
        small = sum(1 for q in chosen if q["benchmark"] in SMALL)
        assert small / len(chosen) <= 0.40, (
            f"{small} of {len(chosen)} slots come from {sum(POOL_SIZES[b] for b in SMALL)} "
            "memorisable questions"
        )


def test_each_benchmark_share_is_bounded_by_its_pool_share():
    from fugal_subnet.config import SLICE_SHARE_FACTOR

    pool = _pool()
    total = sum(POOL_SIZES.values())
    chosen = select_slice(b"\x02" * 32, pool, 300)
    for b, n in POOL_SIZES.items():
        got = sum(1 for q in chosen if q["benchmark"] == b) / len(chosen)
        assert got <= SLICE_SHARE_FACTOR * n / total + 1e-9, (
            f"{b}: {got:.2%} of the slice from {n / total:.2%} of the pool"
        )
