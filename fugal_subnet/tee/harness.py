"""Routing benchmark harness — runs inside the TEE VM.

This is the miner-side code that executes the benchmark. It:
1. Derives the question slice from the nonce
2. Loads the miner's head
3. For each question: computes routing → calls the routed model → grades
4. Builds a BenchmarkProof with attested results

The harness imports the same graders.py (hash-pinned) and uses the same
ROUTING_DECISION_QUANTUM to ensure bit-identical routing decisions.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from urllib.request import Request, urlopen

import numpy as np

from fugal_subnet.benchmarks.slicer import select_slice
from fugal_subnet.config import ROUTING_LAMBDA
from fugal_subnet.graders import grade
from fugal_subnet.head_eval import HeadArtifact, load_head_from_npz, quantize_utility
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult
from fugal_subnet.tee.runtime import MeteringProxy

logger = logging.getLogger(__name__)


def run_benchmark(
    nonce: str,
    head_bytes: bytes,
    benchmark_pool: list[dict],
    proxy: MeteringProxy,
    hidden_states: np.ndarray,
    slice_size: int = 300,
    epoch_id: str = "",
    source_hash: str = "",
    model_costs: dict[str, float] | None = None,
) -> BenchmarkProof:
    """Execute the routing benchmark inside the TEE.

    Args:
        nonce: Block-derived nonce for this epoch.
        head_bytes: Raw .npz bytes of the miner's head.
        benchmark_pool: Full question pool with gold answers.
        proxy: MeteringProxy recording API calls.
        hidden_states: (N_pool, d) backbone hidden states for the full pool.
        slice_size: Number of questions to select.
        epoch_id: Epoch identifier.
        source_hash: SHA256 of the runtime image.
        model_costs: Per-model cost map {model_id: cost_per_token} for routing.

    Returns:
        BenchmarkProof with routing results and attestation.
    """
    if model_costs is None:
        model_costs = {}

    head = load_head_from_npz(head_bytes)
    weights_hash = hashlib.sha256(head_bytes).hexdigest()

    nonce_bytes = bytes.fromhex(nonce) if len(nonce) == 64 else nonce.encode()
    questions = select_slice(nonce_bytes, benchmark_pool, slice_size)

    question_ids = [q["question_id"] for q in questions]
    questions_hash = _compute_questions_hash(question_ids)

    pool_ids = [q["question_id"] for q in benchmark_pool]
    q_to_pool_idx = {qid: i for i, qid in enumerate(pool_ids)}

    proxy.clear()
    results: list[QuestionResult] = []

    for q in questions:
        q_idx = q_to_pool_idx.get(q["question_id"])
        if q_idx is None or q_idx >= hidden_states.shape[0]:
            logger.warning("Question %s not in hidden states, skipping", q["question_id"])
            continue

        h = hidden_states[q_idx]
        routing_decision = _route_question(head, h, model_costs)
        model_id = head.models[routing_decision]

        response_text = _call_model(proxy, model_id, q)
        response_hash = hashlib.sha256(response_text.encode()).hexdigest()

        correct = bool(grade(q, response_text, allow_exec=False))

        cost = proxy.records[-1].cost_usd if proxy.records else 0.0

        results.append(QuestionResult(
            question_id=q["question_id"],
            routed_model=model_id,
            correct=correct,
            cost_usd=cost,
            response_hash=response_hash,
        ))

    proof = BenchmarkProof(
        epoch_id=epoch_id,
        nonce=nonce,
        questions_hash=questions_hash,
        weights_hash=weights_hash,
        source_hash=source_hash,
        results=results,
        total_cost_usd=proxy.total_cost,
        per_model_costs=proxy.per_model_costs,
        attestation_quote=b"",  # filled by TEERuntime.generate_attestation
        timestamp=time.time(),
    )

    return proof


def _route_question(
    head: HeadArtifact,
    hidden_state: np.ndarray,
    model_costs: dict[str, float],
) -> int:
    """Route a single question through the head — same logic as head_eval.py."""
    logits = head.W @ hidden_state + head.b
    p = _softmax(logits)
    costs = np.array(
        [model_costs.get(m, 0.01) for m in head.models], dtype=np.float64,
    )
    utility = quantize_utility(p - ROUTING_LAMBDA * costs)
    return int(np.argmax(utility))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _call_model(
    proxy: MeteringProxy,
    model_id: str,
    question: dict,
) -> str:
    """Call a model via the MeteringProxy."""
    prompt = question.get("question", question.get("prompt", ""))
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
    }).encode()

    req = Request(
        f"http://127.0.0.1:{proxy.port}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""
    except Exception:
        logger.exception("Model call failed for %s", model_id)
        return ""


def _compute_questions_hash(question_ids: list[str]) -> str:
    canonical = json.dumps(sorted(question_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
