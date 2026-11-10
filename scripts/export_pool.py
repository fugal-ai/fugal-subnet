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
    python scripts/export_pool.py --protocol 2          # writes data/benchmarks/pool_v2.json.gz
    python scripts/export_pool.py --out /tmp/pool.json.gz

Protocol v2 has its own pool (`load_all_v2`, verified against
pool_manifest_v2.json; FUGAL_POOL_FILE_V2 to place it elsewhere). A validator
with FUGAL_V2_ACTIVATION_BLOCK set loads it at startup, so place it before the
restart that sets the block. The output is byte-reproducible (sorted, gzip
mtime 0): two operators building the same pool get the same sha256.

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

# A path that can never be a regular file: a child of the null device.
NO_POOL_FILE = os.path.join(os.devnull, "no-pool-file")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--protocol", type=int, default=1, choices=(1, 2),
                    help="2: export the v2 pool (data/benchmarks/pool_v2.json.gz by default)")
    args = ap.parse_args()
    args.out = args.out or os.path.join(ROOT, "data", "benchmarks",
                                        "pool_v2.json.gz" if args.protocol == 2 else "pool.json.gz")

    # Build from the pinned revisions, never from an existing file: the point
    # is to produce the file, and reading one back would only copy it.
    for var in ("FUGAL_BENCHMARK_POOL", "FUGAL_POOL_FILE", "FUGAL_POOL_UNPINNED",
                "FUGAL_BENCHMARK_POOL_V2", "FUGAL_POOL_FILE_V2"):
        os.environ.pop(var, None)
    # A path that cannot name a file, so the loader falls through to the pinned
    # revisions. This used to be os.devnull, which EXISTS: the loader handed
    # /dev/null to the JSON parser and this script failed on every Linux host
    # with "Expecting value: line 1 column 1" (found 2026-11-02). The loader
    # now also ignores anything that is not a regular file.
    os.environ["FUGAL_POOL_FILE"] = NO_POOL_FILE
    os.environ["FUGAL_POOL_FILE_V2"] = NO_POOL_FILE

    from fugal_subnet.benchmarks.loader import content_hash, load_all, load_all_v2, pool_hash

    pool = load_all_v2(strict=True) if args.protocol == 2 else load_all(strict=True)
    pool = sorted(pool, key=lambda q: q["question_id"])
    payload = json.dumps(pool, sort_keys=True, separators=(",", ":")).encode("utf-8")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # GzipFile, not gzip.open: only the class takes mtime (gzip.open raised
    # TypeError here, so the script had never completed; found 2026-11-02).
    # mtime=0 and filename="" keep the header free of the build time and the
    # output path, so two operators building the same pool get the same bytes
    # and the sha256 printed below is comparable between them.
    with open(args.out, "wb") as raw, gzip.GzipFile(
        filename="", fileobj=raw, mode="wb", mtime=0,
    ) as f:
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
