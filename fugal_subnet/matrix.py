"""Ground truth matrix construction.

Calls all models on all questions, grades responses against gold answers,
produces an N×M binary matrix. This is the core value the subnet produces.

Responses are collected concurrently (bounded by API_CONCURRENCY); grading is
done sequentially afterwards so grade order never depends on network timing.
"""
from __future__ import annotations
import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from fugal_subnet.api import BudgetExceeded, SpendTracker, call_model
from fugal_subnet.config import API_CONCURRENCY, CACHE_STALENESS_TTL
from fugal_subnet.graders import grade

logger = logging.getLogger(__name__)


@dataclass
class MatrixResult:
    matrix: np.ndarray          # (N, M) binary — 1 if model got question right
    questions: list[dict]       # length N, the benchmark slice
    models: list[str]           # length M, model IDs
    responses: dict             # {(q_idx, m_idx): response_text}
    costs: dict                 # {model_id: total_cost_usd}
    tracker: SpendTracker


def build_matrix(
    questions: list[dict],
    model_pool: list[str],
    *,
    tracker: SpendTracker | None = None,
    prices: dict[str, tuple[float, float]] | None = None,
    cache_dir: str | None = None,
    allow_exec: bool = False,
    concurrency: int = API_CONCURRENCY,
) -> MatrixResult:
    """Build the ground truth matrix by calling all models on all questions.

    Args:
        questions: List of question dicts from benchmark loaders.
        model_pool: List of model IDs to evaluate.
        tracker: SpendTracker for cost monitoring (thread-safe).
        prices: Price sheet for cost computation.
        cache_dir: Optional directory for response caching (TTL-expired).
        allow_exec: Whether to allow code execution graders.
        concurrency: Max in-flight API calls.

    Returns:
        MatrixResult with the N×M binary matrix and metadata.

    Raises:
        BudgetExceeded: if the tracker's budget cap is hit mid-build. In-flight
        calls finish, queued calls are cancelled, and the exception propagates —
        callers must treat the epoch as aborted.
    """
    if tracker is None:
        tracker = SpendTracker()

    N = len(questions)
    M = len(model_pool)
    matrix = np.zeros((N, M), dtype=np.int8)
    responses: dict[tuple[int, int], str] = {}
    costs: dict[str, float] = {m: 0.0 for m in model_pool}

    cache = _ResponseCache(cache_dir) if cache_dir else None

    # Phase 1 — collect responses (cache first, then concurrent API calls)
    tasks: list[tuple[int, int, str, dict, str]] = []
    for m_idx, model in enumerate(model_pool):
        for q_idx, question in enumerate(questions):
            key = _cache_key(model, question)
            cached = cache.get(key) if cache else None
            if cached is not None:
                responses[(q_idx, m_idx)] = cached
            else:
                tasks.append((q_idx, m_idx, model, question, key))

    logger.info("Matrix build: %d cells (%d cached, %d API calls)",
                N * M, N * M - len(tasks), len(tasks))

    def _fetch(task):
        q_idx, m_idx, model, question, key = task
        text, _, _ = call_model(
            model, question["prompt"], tracker=tracker, prices=prices,
        )
        return q_idx, m_idx, key, text

    budget_hit: BudgetExceeded | None = None
    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {pool.submit(_fetch, t): t for t in tasks}
            done = 0
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    q_idx, m_idx, key, text = fut.result()
                except BudgetExceeded as e:
                    budget_hit = e
                    for f in futures:
                        f.cancel()
                    break
                except Exception as e:
                    q_idx, m_idx, _, question, key = t
                    logger.warning("Model %s failed on %s: %s",
                                   t[2], question["question_id"], e)
                    text = ""
                responses[(q_idx, m_idx)] = text
                # Never cache failures — a transient outage must not poison
                # the cache with empty (auto-zero-graded) responses.
                if cache and text:
                    cache.put(key, text)
                done += 1
                if done % 200 == 0:
                    logger.info("  Responses: %d/%d ($%.2f spent)",
                                done, len(tasks), tracker.total_cost_usd)

    if budget_hit is not None:
        raise budget_hit

    # Phase 2 — grade sequentially (deterministic order, no network involved)
    for m_idx, model in enumerate(model_pool):
        for q_idx, question in enumerate(questions):
            task = _build_grader_task(question)
            matrix[q_idx, m_idx] = grade(
                task, responses.get((q_idx, m_idx), ""), allow_exec,
            )

        model_cost = sum(
            entry["cost_usd"]
            for entry in tracker.per_call_log
            if entry["model"] == model
        )
        costs[model] = model_cost
        logger.info("  %s: %d/%d correct, $%.4f",
                    model, int(matrix[:, m_idx].sum()), N, model_cost)

    return MatrixResult(
        matrix=matrix,
        questions=questions,
        models=model_pool,
        responses=responses,
        costs=costs,
        tracker=tracker,
    )


def build_matrix_mock(
    questions: list[dict],
    model_pool: list[str],
    mock_fn=None,
) -> MatrixResult:
    """Build a matrix with mock responses (no API calls).

    Args:
        questions: List of question dicts.
        model_pool: List of model IDs.
        mock_fn: Optional callable(model, question) -> (response_text, correct).
                 If None, uses random 0/1 assignment.
    """
    import random as rng
    N = len(questions)
    M = len(model_pool)
    matrix = np.zeros((N, M), dtype=np.int8)
    responses: dict[tuple[int, int], str] = {}
    tracker = SpendTracker()

    for m_idx, model in enumerate(model_pool):
        for q_idx, question in enumerate(questions):
            if mock_fn:
                text, correct = mock_fn(model, question)
            else:
                correct = rng.randint(0, 1)
                text = f"mock response for {question['question_id']} by {model}"
            matrix[q_idx, m_idx] = correct
            responses[(q_idx, m_idx)] = text

    return MatrixResult(
        matrix=matrix,
        questions=questions,
        models=model_pool,
        responses=responses,
        costs={m: 0.0 for m in model_pool},
        tracker=tracker,
    )


def _build_grader_task(question: dict) -> dict:
    """Build a grader-compatible task dict from a benchmark question."""
    task: dict = {"gold": question["gold"]}

    meta = question.get("metadata", {})
    checker_meta = meta.get("checker")
    if checker_meta:
        task["checker"] = checker_meta
    else:
        task["checker"] = {"id": question["grader_id"]}

    if "domain" not in task:
        bench_to_domain = {
            "gsm8k": "gsm8k", "math": "math500", "mmlu": "mmlupro",
            "gpqa": "gpqa", "aime": "aime", "humaneval": "humaneval",
            "livecode": "codegen", "ifeval": "ifbench",
        }
        task["domain"] = bench_to_domain.get(question["benchmark"], question["benchmark"])

    if question["grader_id"] == "exec_unittest":
        task["test"] = meta.get("test", "")
        task["entry"] = meta.get("entry_point", "")
        task["stub"] = meta.get("stub", "")

    return task


def _cache_key(model: str, question: dict) -> str:
    payload = f"{model}:{question['question_id']}:{question['prompt'][:200]}"
    return hashlib.sha256(payload.encode()).hexdigest()


class _ResponseCache:
    """File-per-response cache with staleness TTL (model behavior drifts, so
    cached responses expire after CACHE_STALENESS_TTL seconds)."""

    def __init__(self, cache_dir: str, ttl: int = CACHE_STALENESS_TTL):
        self.cache_dir = cache_dir
        self.ttl = ttl
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, key: str) -> str | None:
        path = os.path.join(self.cache_dir, f"{key}.txt")
        try:
            if time.time() - os.path.getmtime(path) > self.ttl:
                os.remove(path)
                return None
            with open(path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    def put(self, key: str, value: str):
        path = os.path.join(self.cache_dir, f"{key}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(value)
