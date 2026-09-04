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
from fugal_subnet.exploration import expected_exploration
from fugal_subnet.graders import grade
from fugal_subnet.grading_task import build_grader_task
from fugal_subnet.head_eval import HeadArtifact, load_head_from_npz, quantize_utility
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult, compute_questions_hash
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
    explore_models: list[str] | None = None,
    explore_size: int = 0,
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
        explore_models: Globally agreed model list exploration draws from.
        explore_size: Number of extra nonce-forced questions to answer.

    Returns:
        BenchmarkProof with routing results and attestation.
    """
    head = load_head_from_npz(head_bytes)
    weights_hash = hashlib.sha256(head_bytes).hexdigest()

    nonce_bytes = bytes.fromhex(nonce) if len(nonce) == 64 else nonce.encode()
    questions = select_slice(nonce_bytes, benchmark_pool, slice_size)

    question_ids = [q["question_id"] for q in questions]
    questions_hash = compute_questions_hash(question_ids)

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
        routing_decision = _route_question(head, h)
        model_id = head.models[routing_decision]

        # Cost is attributed from the records this call actually appended.
        # Reading proxy.records[-1] unconditionally re-bills the PREVIOUS
        # question whenever a call fails and appends nothing, so the
        # per-question costs stop summing to the attested total.
        calls_before = len(proxy.records)
        response_text = _call_model(proxy, model_id, q)
        new_calls = proxy.records[calls_before:]
        cost = sum(r.cost_usd for r in new_calls)
        prompt_tokens = sum(r.prompt_tokens for r in new_calls)
        completion_tokens = sum(r.completion_tokens for r in new_calls)
        response_hash = hashlib.sha256(response_text.encode()).hexdigest()

        # grade() needs a grader task dict, not a raw loader question: it reads
        # task["checker"]["id"] / task["domain"], neither of which the loader
        # schema has. Passing the raw dict raises KeyError inside grade(), which
        # catches it and returns 0 — every answer would grade wrong, silently.
        # allow_exec=False: no code execution inside TEE (security boundary)
        correct = bool(grade(build_grader_task(q), response_text, allow_exec=False))

        results.append(QuestionResult(
            question_id=q["question_id"],
            routed_model=model_id,
            correct=correct,
            cost_usd=cost,
            response_hash=response_hash,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ))

    # --- Exploration pass ---
    # Extra questions routed to models the nonce picks, not the head. They are
    # the only unbiased observation of the model pool anyone gets, and because
    # the assignment is deterministic and attested, skipping them is a rejected
    # proof rather than a saving. See fugal_subnet/exploration.py.
    if explore_models and explore_size > 0:
        pool_by_id = {q["question_id"]: q for q in benchmark_pool}
        explore_map = expected_exploration(
            nonce_bytes, benchmark_pool, set(question_ids), explore_models, explore_size,
        )
        for qid in sorted(explore_map):
            q = pool_by_id.get(qid)
            if q is None:
                continue
            model_id = explore_map[qid]

            calls_before = len(proxy.records)
            response_text = _call_model(proxy, model_id, q)
            new_calls = proxy.records[calls_before:]
            correct = bool(grade(build_grader_task(q), response_text, allow_exec=False))

            results.append(QuestionResult(
                question_id=qid,
                routed_model=model_id,
                correct=correct,
                cost_usd=sum(r.cost_usd for r in new_calls),
                response_hash=hashlib.sha256(response_text.encode()).hexdigest(),
                prompt_tokens=sum(r.prompt_tokens for r in new_calls),
                completion_tokens=sum(r.completion_tokens for r in new_calls),
                is_exploration=True,
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


def _route_question(head: HeadArtifact, hidden_state: np.ndarray) -> int:
    """Route a single question through the head — same logic as head_eval.py.

    Must stay bit-identical to head_eval's rule: the validator re-runs this on
    held-out questions, and a divergence there would look like cheating.
    """
    logits = head.W @ hidden_state + head.b
    p = _softmax(logits)
    return int(np.argmax(quantize_utility(p)))


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
    prompt = question["prompt"]
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


