#!/usr/bin/env python3
"""What share of a score can a memoriser buy, for a given slice shape?

Diligence 2026-10, finding 6. A linear head over 1024-dim embeddings can
memorise roughly C questions outright (scripts/experiment_pool_memorization.py:
about 2,000 fully, saturating near 2,500 with random labels). A miner that
pays once to run every model on C questions and ships the lookup earns, on
every slice, the share of slots those questions fill. Which questions? The
ones worth the most per question: a slot drawn from a benchmark of N_b
questions with quota q_b is worth q_b / N_b per memorised question, so the
memoriser fills its capacity benchmark by benchmark in that order.

This script computes that share in closed form for the real pool sizes and a
range of SLICE_SHARE_FACTOR values, using the slicer itself for the quotas.
Fully offline, deterministic, no data needed.

    python scripts/experiment_slice_memorization.py --capacity 2500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fugal_subnet.benchmarks.slicer import select_slice  # noqa: E402

MANIFEST = os.path.join(ROOT, "fugal_subnet", "benchmarks", "pool_manifest.json")


def memoriser_share(quotas: dict[str, int], sizes: dict[str, int], capacity: int) -> float:
    """Share of slice slots a C-question lookup table covers, greedily."""
    order = sorted(quotas, key=lambda b: -quotas[b] / sizes[b])
    left, slots = capacity, 0.0
    for b in order:
        take = min(left, sizes[b])
        slots += quotas[b] * take / sizes[b]
        left -= take
        if left <= 0:
            break
    return slots / sum(quotas.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capacity", type=int, default=2500,
                    help="questions a linear head memorises outright (measured)")
    ap.add_argument("--slice", type=int, default=300)
    ap.add_argument("--factors", default="1,1.25,1.5,2,3,1000")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        sizes = {k: int(v) for k, v in json.load(f)["per_benchmark"].items()}
    pool = [{"question_id": f"{b}:{i}", "benchmark": b} for b, n in sizes.items() for i in range(n)]
    nonce = b"\x07" * 32

    print(f"pool {sum(sizes.values())} questions: {sizes}")
    print(f"memoriser capacity {args.capacity}, slice {args.slice}\n")
    print(f"{'factor':>7} {'quotas':<52} {'small':>6} {'mmlu':>6} {'memoriser':>10}")
    rows = []
    for factor in [float(x) for x in args.factors.split(",")]:
        chosen = select_slice(nonce, pool, args.slice, share_factor=factor)
        quotas = dict(sorted(Counter(q["benchmark"] for q in chosen).items()))
        small = sum(quotas.get(b, 0) for b in ("aime", "gsm8k", "ifeval")) / args.slice
        mmlu = quotas.get("mmlu", 0) / args.slice
        share = memoriser_share(quotas, sizes, args.capacity)
        label = "equal (old)" if factor >= 1000 else f"{factor:g}"
        print(f"{label:>7} {str(quotas):<52} {small:>6.0%} {mmlu:>6.0%} {share:>10.1%}")
        rows.append({"factor": factor, "quotas": quotas, "small_share": small,
                     "mmlu_share": mmlu, "memoriser_share": share})
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"capacity": args.capacity, "slice": args.slice, "rows": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
