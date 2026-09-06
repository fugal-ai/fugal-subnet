#!/usr/bin/env python3
"""Pin the benchmark pool's identity, including its CONTENT.

`pool_hash` covers sorted question ids. That is what the slice is drawn from,
so it is the right identity for "did we select the same questions" — but it says
nothing about what those questions SAY. Two operators whose ids agree and whose
prompts or gold answers differ select the same slice and then grade different
things, and every proof between them fails on a hash that names the symptom.

In practice `DATASET_REVISIONS` pins the content, so this cannot drift on its
own. It can still drift by hand: an edited local cache, a manually placed
`data/benchmarks/livecode.json`, or a benchmark loaded without a revision pin
(`livecode.py` is loaded without one today, which is also the one that silently
yields zero questions). This manifest makes any of those a named error at
startup instead of a mystery at verification time.

    python scripts/build_pool_manifest.py                  # write the manifest
    python scripts/build_pool_manifest.py --check          # verify, change nothing

The manifest holds hashes and counts, never question content — the datasets
carry their own licences and this repo does not redistribute them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MANIFEST_PATH = os.path.join("data", "pool_manifest.json")


def content_hash(pool: list[dict]) -> str:
    """Hash over everything a validator grades against, in id order.

    Deliberately covers the four fields that decide an outcome: which question
    it is, what the miner is asked, what counts as right, and which checker
    decides. A change in any of them changes a score, so a change in any of them
    must change this hash.
    """
    h = hashlib.sha256()
    for q in sorted(pool, key=lambda x: x["question_id"]):
        h.update(json.dumps([
            q.get("question_id", ""),
            q.get("prompt", ""),
            # Canonicalised the way the GRADER canonicalises it, not with str().
            # exec_io compares json.dumps(got) to json.dumps(gold), so a tuple
            # and a list of the same values grade identically — and eight
            # HumanEval golds are tuples in memory and lists after a JSON round
            # trip, which is what FUGAL_BENCHMARK_POOL does. Hashing str() made
            # this manifest reject the documented override for a difference no
            # score depends on. An identity hash must be sensitive to exactly
            # what changes an outcome: no less, and no more.
            json.dumps(q.get("gold", ""), sort_keys=True, default=str),
            q.get("grader_id", ""),
        ], separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def build(pool: list[dict], skip: list[str]) -> dict:
    from collections import Counter

    from fugal_subnet.benchmarks.loader import DATASET_REVISIONS, pool_hash

    return {
        "n_questions": len(pool),
        "pool_hash": pool_hash(pool),
        "content_hash": content_hash(pool),
        "per_benchmark": dict(sorted(Counter(
            q.get("benchmark", "") for q in pool).items())),
        # The configuration is part of the identity. A pool built with a
        # different skip list is a different pool, and recording it turns
        # "why is my hash different" into a one-line diff.
        "skip_benchmarks": sorted(skip),
        "dataset_revisions": dict(sorted(DATASET_REVISIONS.items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="Verify only; write nothing")
    ap.add_argument("--from-file", help="Use a materialised pool JSON instead of loading")
    ap.add_argument("--out", default=MANIFEST_PATH)
    args = ap.parse_args()

    from fugal_subnet.benchmarks.loader import load_all

    skip = sorted(s for s in os.getenv("FUGAL_SKIP_BENCHMARKS", "").split(",") if s)
    if args.from_file:
        with open(args.from_file, encoding="utf-8") as f:
            pool = json.load(f)
    else:
        # strict=False deliberately: this is the tool that RESOLVES a manifest
        # mismatch, so it must not be blocked by one. Requiring a valid manifest
        # to rebuild the manifest is a lock with the key inside.
        pool = load_all(strict=False)

    built = build(pool, skip)
    for k in ("n_questions", "pool_hash", "content_hash", "per_benchmark", "skip_benchmarks"):
        print(f"  {k:18s} {built[k]}")

    if args.check:
        if not os.path.exists(args.out):
            print(f"\nNo manifest at {args.out} — nothing to check against.")
            return 1
        with open(args.out, encoding="utf-8") as f:
            pinned = json.load(f)
        diffs = [k for k in ("pool_hash", "content_hash", "n_questions",
                             "skip_benchmarks", "per_benchmark")
                 if pinned.get(k) != built[k]]
        if diffs:
            print(f"\nMISMATCH in: {', '.join(diffs)}")
            for k in diffs:
                print(f"  {k}\n    pinned: {pinned.get(k)}\n    built : {built[k]}")
            return 1
        print("\nPool matches the pinned manifest.")
        return 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(built, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
