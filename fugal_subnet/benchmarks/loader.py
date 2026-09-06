"""Unified benchmark loading interface.

Every loader returns list[dict] with schema:
    prompt:      str   — the question/task text sent to a model
    gold:        str   — the gold answer for grading
    grader_id:   str   — which grader to use (maps to graders.CHECKERS)
    benchmark:   str   — benchmark name (e.g. "mmlu", "math", "gsm8k")
    question_id: str   — unique stable identifier for this question
    metadata:    dict   — benchmark-specific fields (checker params, test code, etc.)

Dataset revisions are PINNED (DATASET_REVISIONS): the benchmark pool is
consensus-relevant state — two validators loading a dataset at different
times must build byte-identical pools, so loaders never track a moving
"main" branch. Bump these pins deliberately, as a coordinated release.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os

logger = logging.getLogger(__name__)

# HuggingFace dataset revision pins (commit SHAs, resolved 2026-08-24).
DATASET_REVISIONS: dict[str, str] = {
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "openai/openai_humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    "Idavidrein/gpqa": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
    "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    "qq8933/AIME_1983_2024": "3e2cc86390666c5c756622afc0eeb9e6194496bc",
}

_BENCHMARKS: dict[str, str] = {
    "mmlu":      "fugal_subnet.benchmarks.mmlu",
    "math":      "fugal_subnet.benchmarks.math_",
    "gsm8k":     "fugal_subnet.benchmarks.gsm8k",
    "humaneval": "fugal_subnet.benchmarks.humaneval",
    "livecode":  "fugal_subnet.benchmarks.livecode",
    "gpqa":      "fugal_subnet.benchmarks.gpqa",
    "ifeval":    "fugal_subnet.benchmarks.ifeval",
    "aime":      "fugal_subnet.benchmarks.aime",
}


def available_benchmarks() -> list[str]:
    return sorted(_BENCHMARKS.keys())


def load_benchmark(name: str) -> list[dict]:
    if name not in _BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name!r}. Available: {available_benchmarks()}")
    mod = importlib.import_module(_BENCHMARKS[name])
    return mod.load()


def pool_hash(pool: list[dict]) -> str:
    """Identity of a question pool.

    The pool is consensus state: the slice is drawn from it, so two neurons
    holding different pools derive different slices and every proof fails on
    questions_hash — correctly, but for a reason that looks nothing like the
    cause. This makes the real cause nameable in a log line.
    """
    ids = sorted(q["question_id"] for q in pool)
    canonical = json.dumps(ids, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _drop_unscoreable(pool: list[dict]) -> list[dict]:
    """Remove questions no checker can score under the current harness policy.

    Derived from config.HARNESS_ALLOW_EXEC rather than hardcoded, so enabling
    execution re-includes the benchmarks in exactly one place. A question the
    harness cannot grade is not neutral: the miner still pays to answer it, and
    since the slicer equalises benchmarks it was a full sixth of every graded
    slice, punishing anyone who routed code to a capable model while no quality
    was reachable. See docs/CODE_BENCHMARK_PLAN.md for putting them back.
    """
    from fugal_subnet.config import EXECUTION_CHECKERS, HARNESS_ALLOW_EXEC

    if HARNESS_ALLOW_EXEC:
        return pool
    kept = [q for q in pool if q.get("grader_id") not in EXECUTION_CHECKERS]
    dropped = len(pool) - len(kept)
    if dropped:
        from collections import Counter
        by_bench = Counter(q.get("benchmark", "") for q in pool
                           if q.get("grader_id") in EXECUTION_CHECKERS)
        logger.warning(
            "Excluded %d questions the harness cannot score (%s): their checkers "
            "run candidate code and FUGAL_HARNESS_ALLOW_EXEC is off. They would "
            "score zero for every miner while still costing them API spend.",
            dropped, dict(by_bench),
        )
    return kept


def load_all(strict: bool = True) -> list[dict]:
    """Load the full benchmark pool.

    Both neurons call this, so both derive the same pool from the same pinned
    dataset revisions. FUGAL_BENCHMARK_POOL overrides it with a local JSON file
    — for offline work and local testnets, where reaching HuggingFace is either
    impossible or pointlessly slow. The override applies to miner and validator
    alike, which is the point: a pool one side can load and the other cannot is
    how the two end up disagreeing.

    Benchmarks named in FUGAL_SKIP_BENCHMARKS (comma-separated) are skipped
    deliberately. Any OTHER load failure raises when strict=True: a validator
    running with a silently incomplete pool would select a different slice
    than its peers and diverge on every epoch. Fail loudly instead.
    """
    override = os.getenv("FUGAL_BENCHMARK_POOL", "").strip()
    if override:
        with open(override, encoding="utf-8") as f:
            pool = json.load(f)
        pool = _drop_unscoreable(pool)
        logger.info(
            "Benchmark pool loaded from FUGAL_BENCHMARK_POOL=%s "
            "(%d questions, pool_hash=%s)",
            override, len(pool), pool_hash(pool)[:16],
        )
        return pool

    skip = set(os.getenv("FUGAL_SKIP_BENCHMARKS", "").split(",")) - {""}
    pool = []
    for name in sorted(_BENCHMARKS):
        if name in skip:
            logger.info("Skipping benchmark %s (FUGAL_SKIP_BENCHMARKS)", name)
            continue
        try:
            items = load_benchmark(name)
        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"Benchmark {name!r} failed to load: {e}. "
                    f"A partial pool breaks cross-validator determinism — fix the "
                    f"load or add {name!r} to FUGAL_SKIP_BENCHMARKS explicitly."
                ) from e
            logger.warning("Skipping benchmark %s: %s", name, e)
            continue
        if not items:
            # Zero questions is a divergence, not a warning. It is the same
            # hazard as a load failure and gets the same treatment: a benchmark
            # that yields nothing here yields its full contents for anyone who
            # has a local copy of it, so two operators build different pools,
            # derive different slices, and every proof between them fails on
            # questions_hash — an error that names the symptom and never the
            # cause. LiveCodeBench does exactly this on current `datasets`
            # versions. Absence has to be declared to be reproducible.
            if strict:
                raise RuntimeError(
                    f"Benchmark {name!r} loaded 0 questions. An empty benchmark "
                    f"is a silent pool divergence: anyone holding a local copy "
                    f"of it builds a different pool and no proof between you "
                    f"will verify. Add {name!r} to FUGAL_SKIP_BENCHMARKS to "
                    f"declare the absence, or fix the load."
                )
            logger.warning("Benchmark %s loaded 0 questions", name)
        pool.extend(items)
    pool = _drop_unscoreable(pool)
    logger.info("Benchmark pool: %d questions, pool_hash=%s",
                len(pool), pool_hash(pool)[:16])
    _verify_against_manifest(pool, skip, strict)
    return pool


_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "pool_manifest.json",
)


def _verify_against_manifest(pool: list[dict], skip: set, strict: bool) -> None:
    """Check the loaded pool against the pinned manifest.

    pool_hash covers question ids, which is the right identity for slice
    selection but says nothing about what the questions SAY. Revisions are
    pinned so content cannot drift on its own — but it can drift by hand: an
    edited cache, a manually placed data/benchmarks/livecode.json, or a
    benchmark whose loader carries no revision pin. Any of those otherwise
    produces a pool that agrees on ids, selects the same slice, and grades
    different things, failing at verification with a hash that names the
    symptom rather than the cause.

    Absent manifest is not an error — a fresh checkout or a bespoke pool is a
    legitimate state. A manifest that DISAGREES is, under strict.
    """
    if not os.path.exists(_MANIFEST_PATH):
        return
    try:
        with open(_MANIFEST_PATH, encoding="utf-8") as f:
            pinned = json.load(f)
    except Exception as e:  # noqa: BLE001 - a broken manifest must not be fatal
        logger.warning("Could not read pool manifest %s: %s", _MANIFEST_PATH, e)
        return

    if sorted(pinned.get("skip_benchmarks", [])) != sorted(skip):
        logger.warning(
            "Pool manifest was built with FUGAL_SKIP_BENCHMARKS=%s but this "
            "process has %s — skipping content verification. The pools are "
            "different by construction and will not agree.",
            ",".join(pinned.get("skip_benchmarks", [])) or "(none)",
            ",".join(sorted(skip)) or "(none)",
        )
        return

    from scripts.build_pool_manifest import content_hash  # local: script-side helper

    actual_ids, actual_content = pool_hash(pool), content_hash(pool)
    if actual_ids == pinned.get("pool_hash") and actual_content == pinned.get("content_hash"):
        logger.info("Pool matches the pinned manifest (content_hash=%s)",
                    actual_content[:16])
        return

    detail = (
        f"pinned pool_hash={pinned.get('pool_hash','?')[:16]} "
        f"content_hash={pinned.get('content_hash','?')[:16]} "
        f"n={pinned.get('n_questions')}; "
        f"loaded pool_hash={actual_ids[:16]} content_hash={actual_content[:16]} "
        f"n={len(pool)}"
    )
    if strict:
        raise RuntimeError(
            "Benchmark pool does not match data/pool_manifest.json. The pool is "
            "consensus state: a pool that differs from every other operator's "
            "selects a different slice or grades different answers, and every "
            f"proof will fail on a hash that names none of this. {detail}. "
            "Rebuild with scripts/build_pool_manifest.py if the change is "
            "intended, and say why in the commit."
        )
    logger.warning("Benchmark pool does not match the pinned manifest: %s", detail)
