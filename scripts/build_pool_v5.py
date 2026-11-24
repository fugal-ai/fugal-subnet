#!/usr/bin/env python3
"""Build the protocol v5 exam pool (fugal_subnet/benchmarks/arena_sets.py) from pinned inputs.

    uv run python scripts/build_pool_v5.py --v3-pool <pool_v3.json[.gz]> \
        --cache-dir <dir for downloads> --out <pool_v5.json.gz>

Writes the pool (gzip JSON list, not committed: it is large and rebuildable), and in the
package fugal_subnet/benchmarks/:
  - pool_manifest_v5.json      counts, RouterArena weights, content hash, every pin
  - exclusions_routerarena.json the candidate ids dropped as RouterArena questions
The v3 pool supplies the code questions; it is verified against pool_manifest_v3.json's
content hash first. Downloads are untrusted data: parquet/CSV/JSON only, never unpickled.
Needs Hugging Face access and ~3 GB of downloads (PubMedQA artificial, ChessInstruct train,
the geography set).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fugal_subnet import graders_arena  # noqa: E402
from fugal_subnet.benchmarks import arena_sets, loader  # noqa: E402

PKG = os.path.join(os.path.dirname(__file__), "..", "fugal_subnet", "benchmarks")


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _read_pool(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        p = json.load(f)
    return p["pool"] if isinstance(p, dict) else p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v3-pool", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    v3 = _read_pool(args.v3_pool)
    want = json.load(open(loader.manifest_path_v3()))["content_hash"]
    got = loader.content_hash_v3(v3)
    if got != want:
        raise SystemExit(f"v3 pool content hash {got[:16]} != manifest {want[:16]}")

    report: dict = {}
    pool, excluded, weights = arena_sets.build_pool(arena_sets.Fetcher(args.cache_dir), v3, report)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, separators=(",", ":"))

    excl_path = os.path.join(PKG, "exclusions_routerarena.json")
    with open(excl_path, "w", encoding="utf-8") as f:
        json.dump({"routerarena": {"repo": arena_sets.ROUTERARENA[0],
                                   "revision": arena_sets.ROUTERARENA[1],
                                   "rule": "same normalised question text, or (long questions) "
                                           f"a shared {arena_sets.SHINGLE_WORDS}-word shingle "
                                           "with a RouterArena question of the same dataset"},
                   "question_ids": excluded}, f, indent=1, sort_keys=True)
        f.write("\n")

    counts = Counter(q["metadata"]["ra_dataset"] for q in pool)
    manifest = {
        "protocol": 5,
        "n_questions": len(pool),
        "content_hash": arena_sets.content_hash(pool),
        "per_dataset": dict(sorted(counts.items())),
        "weights_ppm": weights,
        "not_in_pool": arena_sets.NOT_IN_POOL,
        "stand_ins": {"OpenTDB": "kth8/OpenTriviaQA (OpenTDB is not on Hugging Face)",
                      "LiveCodeBench": "the v3 calibrated code sets: " + ", ".join(arena_sets.CODE_BENCHMARKS)},
        "caps": {"total": arena_sets.CAP_TOTAL, "floor": arena_sets.CAP_FLOOR,
                 "order": f"sha256('{arena_sets.POOL_SALT}:' + question_id)"},
        "build_report": report,
        "dataset_revisions": arena_sets.SOURCES,
        "routerarena_revision": arena_sets.ROUTERARENA[1],
        "v3_pool_content_hash": want,
        "grader_hash": graders_arena.grader_hash(),
        "prompts_sha256": _sha(os.path.join(PKG, "arena_prompts.json")),
        "exclusions_sha256": _sha(excl_path),
        "licences_to_review": {"m-a-p/MusicTheoryBench": "cc (unspecified)",
                               "community-datasets/qanta": "unknown",
                               "wmt/wmt19": "unknown", "allenai/social_i_qa": "unknown",
                               "dreamerdeo/finqa": "unknown", "aps/super_glue": "other"},
    }
    with open(os.path.join(PKG, "pool_manifest_v5.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"pool {len(pool)} questions, content hash {manifest['content_hash'][:16]}")
    for k, v in sorted(counts.items()):
        print(f"  {k:28s} {v:6d}  weight {weights.get(k, 0) / 1e4:.2f}%  "
              f"loaded {report['loaded'].get(k, 0)}  excluded {report['excluded'].get(k, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
