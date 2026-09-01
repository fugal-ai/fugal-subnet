#!/usr/bin/env python3
"""Recompute v2 normalized benchmark counts and hashes from pinned sources."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fugal_subnet.benchmarks.loader import DATASET_REVISIONS
from fugal_subnet.v2.benchmarks import (
    _LOADERS,
    CANONICAL_BENCHMARKS,
    _validate_questions,
    benchmark_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "fugal_subnet" / "benchmark-registry-v2.json"


SOURCES = {
    "aime": {
        "dataset": "qq8933/AIME_1983_2024",
        "revision": DATASET_REVISIONS["qq8933/AIME_1983_2024"],
        "license": "unverified",
        "redistribution": "external-source-only",
    },
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "revision": DATASET_REVISIONS["openai/gsm8k"],
        "license": "MIT",
        "redistribution": "permitted",
    },
    "humaneval": {
        "dataset": "packaged:human-eval-cases-v2.json",
        "revision": DATASET_REVISIONS["openai/openai_humaneval"],
        "license": "MIT",
        "redistribution": "permitted",
    },
    "ifeval": {
        "dataset": "google/IFEval",
        "revision": DATASET_REVISIONS["google/IFEval"],
        "license": "Apache-2.0",
        "redistribution": "permitted",
    },
    "math": {
        "dataset": "EleutherAI/hendrycks_math",
        "revision": DATASET_REVISIONS["EleutherAI/hendrycks_math"],
        "license": "MIT-dataset-card",
        "redistribution": "external-source-only",
    },
    "mmlu": {
        "dataset": "cais/mmlu",
        "revision": DATASET_REVISIONS["cais/mmlu"],
        "license": "MIT",
        "redistribution": "permitted",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    benchmarks = {}
    pool = []
    for name in CANONICAL_BENCHMARKS:
        questions = _LOADERS[name]()
        _validate_questions(name, questions)
        benchmarks[name] = {
            **SOURCES[name],
            "count": len(questions),
            "sha256": benchmark_sha256(questions),
        }
        pool.extend(questions)
        print(
            f"{name}: count={len(questions)} sha256={benchmarks[name]['sha256']}",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "pool_sha256": benchmark_sha256(pool),
        "benchmarks": benchmarks,
    }
    encoded = (
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(f"pool_sha256={payload['pool_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
