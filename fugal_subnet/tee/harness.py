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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

import numpy as np

from fugal_subnet.benchmarks.slicer import select_slice
from fugal_subnet.config import HARNESS_ALLOW_EXEC, HARNESS_CONCURRENCY
from fugal_subnet.exploration import expected_exploration
from fugal_subnet.graders import grade
from fugal_subnet.grading_task import build_grader_task
from fugal_subnet.head_eval import HeadArtifact, load_head_from_npz
from fugal_subnet.routing_protocol import BENCHMARK_LAMBDA, benchmark_costs, identity
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult, compute_questions_hash
from fugal_subnet.tee.runtime import MeteringProxy
from fugal_subnet.vendor import success_contract as contract

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
    hotkey: str = "",
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
        hotkey: SS58 address this proof is produced FOR. It goes into
            content_hash and therefore into the attestation's report_data, so
            the hardware attests whose proof this is and nobody else can
            present it. Public, so passing it in costs no confidentiality.

    Returns:
        BenchmarkProof with routing results and attestation.
    """
    head = load_head_from_npz(head_bytes)
    costs = benchmark_costs(head)
    weights_hash = hashlib.sha256(head_bytes).hexdigest()

    nonce_bytes = bytes.fromhex(nonce) if len(nonce) == 64 else nonce.encode()
    questions = select_slice(nonce_bytes, benchmark_pool, slice_size)

    question_ids = [q["question_id"] for q in questions]
    questions_hash = compute_questions_hash(question_ids)

    pool_ids = [q["question_id"] for q in benchmark_pool]
    q_to_pool_idx = {qid: i for i, qid in enumerate(pool_ids)}

    proxy.clear()
    results: list[QuestionResult] = []

    # Route every question first (pure numpy, deterministic), then make the
    # model calls several at a time, then grade in slice order on this thread.
    # The calls are the only part that waits on a network, so they are the only
    # part that runs concurrently; routing and grading stay sequential and
    # single-threaded, exactly as before.
    scheduled: list[tuple[dict, str, bool]] = []
    for q in questions:
        q_idx = q_to_pool_idx.get(q["question_id"])
        if q_idx is None or q_idx >= hidden_states.shape[0]:
            logger.warning("Question %s not in hidden states, skipping", q["question_id"])
            continue
        model_id = head.models[_route_question(head, hidden_states[q_idx], costs)]
        scheduled.append((q, model_id, False))

    for q, model_id, response_text, cost, prompt_tokens, completion_tokens in _call_all(
        proxy, scheduled, epoch_id,
    ):
        # grade() needs a grader task dict, not a raw loader question: it reads
        # task["checker"]["id"] / task["domain"], neither of which the loader
        # schema has. Passing the raw dict raises KeyError inside grade(), which
        # catches it and returns 0 — every answer would grade wrong, silently.
        # Execution policy is one constant, read by the harness AND by the
        # loader that decides what is in the pool — see config.HARNESS_ALLOW_EXEC.
        # They drifted apart once and a sixth of every slice scored zero.
        correct = bool(grade(build_grader_task(q), response_text,
                             allow_exec=HARNESS_ALLOW_EXEC))
        results.append(QuestionResult(
            question_id=q["question_id"],
            routed_model=model_id,
            correct=correct,
            cost_usd=cost,
            response_hash=hashlib.sha256(response_text.encode()).hexdigest(),
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
        explore_sched = [
            (pool_by_id[qid], explore_map[qid], True)
            for qid in sorted(explore_map) if qid in pool_by_id
        ]
        for q, model_id, response_text, cost, prompt_tokens, completion_tokens in _call_all(
            proxy, explore_sched, epoch_id,
        ):
            correct = bool(grade(build_grader_task(q), response_text,
                                 allow_exec=HARNESS_ALLOW_EXEC))
            results.append(QuestionResult(
                question_id=q["question_id"],
                routed_model=model_id,
                correct=correct,
                cost_usd=cost,
                response_hash=hashlib.sha256(response_text.encode()).hexdigest(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                is_exploration=True,
            ))

    proof = BenchmarkProof(
        hotkey=hotkey,
        epoch_id=epoch_id,
        nonce=nonce,
        questions_hash=questions_hash,
        weights_hash=weights_hash,
        source_hash=source_hash,
        routing_protocol=identity(),
        results=results,
        total_cost_usd=proxy.total_cost,
        per_model_costs=proxy.per_model_costs,
        attestation_quote=b"",  # filled by TEERuntime.generate_attestation
        timestamp=time.time(),
    )

    return proof


def _route_question(head: HeadArtifact, hidden_state: np.ndarray, costs=None) -> int:
    """Independent success minus estimated dollars; live lambda is always one."""
    if head.success is None:
        raise ValueError("success benchmark requires a versioned success head")
    if costs is None:
        costs = benchmark_costs(head)
    p = contract.predictions(head.W, head.b, hidden_state)
    return int(contract.rank(p, costs, BENCHMARK_LAMBDA)[0])


# The request id of the model call THIS thread is making, so that _call_model
# can label it and the proxy can attribute the record to the right question.
# Thread-local rather than a parameter: every stub that stands in for
# _call_model in tests and scripts keeps its (proxy, model_id, question)
# signature and reads current_request_id() when it appends its record.
_request_ctx = threading.local()


def current_request_id() -> str:
    return getattr(_request_ctx, "request_id", "")


def _call_all(proxy, scheduled, epoch_id):
    """Make every scheduled model call, HARNESS_CONCURRENCY at a time, and yield
    (question, model_id, text, cost, prompt_tokens, completion_tokens) in the
    ORDER SCHEDULED — never completion order. Cost is attributed by request id,
    so interleaved calls cannot bill one question for another's tokens, and a
    failed call that appended no record costs nothing (as before).
    """
    def one(item):
        q, model_id, _explore = item
        rid = f"{epoch_id}:{q['question_id']}:{model_id}"
        _request_ctx.request_id = rid
        try:
            # `or ""`: a reply that is not text is a wrong answer, never an
            # abort. One aborted question used to take the whole proof.
            text = _call_model(proxy, model_id, q) or ""
        finally:
            _request_ctx.request_id = ""
        return rid, text

    if not scheduled:
        return
    with ThreadPoolExecutor(max_workers=HARNESS_CONCURRENCY) as pool:
        outcomes = list(pool.map(one, scheduled))
    for (q, model_id, _explore), (rid, text) in zip(scheduled, outcomes):
        mine = [r for r in proxy.records if r.request_id == rid]
        yield (
            q, model_id, text,
            sum(r.cost_usd for r in mine),
            sum(r.prompt_tokens for r in mine),
            sum(r.completion_tokens for r in mine),
        )


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
        headers={"Content-Type": "application/json",
                 "X-Fugal-Request": current_request_id()},
        method="POST",
    )

    try:
        # Slightly longer than the proxy's own 180 s upstream timeout, so the
        # proxy's 502 reaches us before we give up on the proxy; otherwise the
        # proxy writes into a closed socket (BrokenPipe) and the failure is
        # logged twice as two different errors.
        with urlopen(req, timeout=200) as resp:
            data = json.loads(resp.read())
        return _text_of(data)
    except Exception:
        logger.exception("Model call failed for %s", model_id)
        return ""


def _text_of(data) -> str:
    """The reply text, or "" — never None, never a non-string.

    Found on real hardware with a real provider, 2026-09-07: a chat completion
    can carry `"content": null` (a refusal, a tool-only turn, an upstream error
    dressed as a completion), and `.get("content", "")` returns None for a key
    that is present with a null value. The harness then called `.encode()` on
    it and the whole epoch aborted — no proof, for one odd reply out of three
    hundred. Every stub in the test-suite returns a string, so nothing had ever
    exercised this. A reply that is not text is graded as a wrong answer, which
    is what it is; the epoch goes on.
    """
    try:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
    except AttributeError:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some providers return content parts: [{"type": "text", "text": ...}].
        return "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


