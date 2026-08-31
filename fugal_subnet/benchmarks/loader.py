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
import importlib
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


def load_all(strict: bool = True) -> list[dict]:
    """Load the full benchmark pool.

    Benchmarks named in FUGAL_SKIP_BENCHMARKS (comma-separated) are skipped
    deliberately. Any OTHER load failure raises when strict=True: a validator
    running with a silently incomplete pool would select a different slice
    than its peers and diverge on every epoch. Fail loudly instead.
    """
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
            logger.warning("Benchmark %s loaded 0 questions", name)
        pool.extend(items)
    return pool
