#!/usr/bin/env python3
"""Materialise the pinned benchmark pool as one file every neuron can load.

The pool is consensus state. Today each neuron rebuilds it from Hugging Face at
pinned dataset revisions and checks the result against the hash-pinned
manifest; that catches drift, but it leaves every new operator one upstream
removal away from being unable to build the pool at all (diligence 2026-10,
finding 8). This script builds the pool once, through the same `load_all()`
and the same manifest check, and writes it as a single gzipped JSON file that
`load_all()` prefers when it finds it at `data/benchmarks/pool.json.gz` (or
`FUGAL_POOL_FILE`). The file is verified against the manifest on every load,
so a wrong or edited file is a named startup error, not a silent divergence.

NOT a release asset. The datasets carry their own licences and this repo does
not redistribute them. Build it from a machine that can reach Hugging Face and
copy it to your neurons; the sha256 printed here is what to compare.

    python scripts/export_pool.py                       # writes data/benchmarks/pool.json.gz
    python scripts/export_pool.py --out /tmp/pool.json.gz

No API spend. Hugging Face access is needed to build, never to run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "benchmarks", "pool.json.gz"))
    args = ap.parse_args()

    # Build from the pinned revisions, never from an existing file: the point
    # is to produce the file, and reading one back would only copy it.
    for var in ("FUGAL_BENCHMARK_POOL", "FUGAL_POOL_FILE", "FUGAL_POOL_UNPINNED"):
        os.environ.pop(var, None)
    os.environ["FUGAL_POOL_FILE"] = os.devnull  # never exists as a pool file

    from fugal_subnet.benchmarks.loader import content_hash, load_all, pool_hash

    pool = load_all(strict=True)
    pool = sorted(pool, key=lambda q: q["question_id"])
    payload = json.dumps(pool, sort_keys=True, separators=(",", ":")).encode("utf-8")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with gzip.open(args.out, "wb", mtime=0) as f:   # mtime=0: byte-reproducible
        f.write(payload)
    with open(args.out, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"wrote {args.out}: {len(pool)} questions, {os.path.getsize(args.out)} bytes")
    print(f"pool_hash    {pool_hash(pool)}")
    print(f"content_hash {content_hash(pool)}")
    print(f"file sha256  {digest}")
    print("Copy the file to every neuron (same path or FUGAL_POOL_FILE); load_all() "
          "verifies it against the manifest at startup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
