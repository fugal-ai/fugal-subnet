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
import hmac
import http.client
import json
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.request import HTTPHandler, Request, build_opener

import numpy as np

from fugal_subnet import epoch_plan, graders_code
from fugal_subnet.config import (
    HARNESS_ALLOW_EXEC,
    HARNESS_CALL_TIMEOUT_S,
    HARNESS_CONCURRENCY,
    HARNESS_MAX_TOKENS,
    HARNESS_SYSTEM_PROMPT_V3,
    SANDBOXED_CODE_CHECKERS,
)
from fugal_subnet.graders import grade
from fugal_subnet.grading_task import build_grader_task
from fugal_subnet.head_eval import HeadArtifact, load_head_from_npz
from fugal_subnet.routing_protocol import (
    BENCHMARK_LAMBDA,
    benchmark_costs,
    benchmark_costs_for,
    identity_for,
    reasoning_for,
    request_for,
)
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult, compute_questions_hash
from fugal_subnet.tee.runtime import MeteringProxy, reply_has_choices
from fugal_subnet.vendor import success_contract as contract

logger = logging.getLogger(__name__)

# Pause before the one reconnect to the proxy (see _call_model): long enough for
# the proxy's accept loop to take a queued burst, nothing next to a model call.
PROXY_RECONNECT_WAIT_S = 0.5


class ProxyNeverAnswered(ConnectionError):
    """The connection to the proxy ended before the first byte of a reply."""


class _ProxyConnection(http.client.HTTPConnection):
    """A connection that knows whether the proxy sent anything at all.

    http.client raises the same ConnectionResetError for a reset before the
    status line and for one in the middle of it, and that difference decides
    whether asking again is safe (see _call_model). So the first byte of the
    reply is waited for here, without taking it off the socket, under the
    connection's own timeout.
    """

    def getresponse(self):
        sock = self.sock
        if sock is not None:
            try:
                first = sock.recv(1, socket.MSG_PEEK)
            except ConnectionError as e:
                raise ProxyNeverAnswered(f"{type(e).__name__} before any reply") from e
            if not first:
                raise ProxyNeverAnswered("closed before any reply")
        return super().getresponse()


class _ProxyHTTPHandler(HTTPHandler):
    def http_open(self, req):
        return self.do_open(_ProxyConnection, req)


# urllib's urlopen with one difference: plain-HTTP calls (the proxy is on
# loopback) go over _ProxyConnection. Kept under this name because tests
# replace `harness.urlopen` with a stub taking (req, timeout).
urlopen = build_opener(_ProxyHTTPHandler).open


def _proxy_never_answered(e: BaseException) -> bool:
    """True when a call to the proxy failed before any byte of a reply arrived:
    refused or reset while connecting or sending (urllib wraps exactly those in
    URLError), or ended before the first reply byte (ProxyNeverAnswered)."""
    if isinstance(e, ProxyNeverAnswered):
        return True
    return (isinstance(e, URLError) and not isinstance(e, HTTPError)
            and isinstance(e.reason, ConnectionError))


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
    protocol_version: int = 1,
    menu: str | None = None,
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
    # v1 keeps calling benchmark_costs exactly as before; v2 prices on its own manifest and table.
    # `menu`: the epoch's v3 menu (routing_protocol.menu_for_epoch); the
    # validator verifies the head and prices the calls on the same one.
    costs = (benchmark_costs(head) if protocol_version == 1
             else benchmark_costs_for(head, protocol_version, menu=menu))
    weights_hash = hashlib.sha256(head_bytes).hexdigest()

    nonce_bytes = bytes.fromhex(nonce) if len(nonce) == 64 else nonce.encode()
    # One selection entry point for miner and validator (epoch_plan, INVARIANTS I1).
    questions = epoch_plan.scored_slice(protocol_version, nonce_bytes, benchmark_pool, slice_size)

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

    # --- Exploration, scheduled WITH the scored questions ---
    # Extra questions routed to models the nonce picks, not the head. They are
    # the only unbiased observation of the model pool anyone gets, and because
    # the assignment is deterministic and attested, skipping them is a rejected
    # proof rather than a saving. See fugal_subnet/exploration.py.
    #
    # They used to run as a second pass AFTER the scored questions. A miner
    # controls its own VM's network, so a distinct phase was a distinct thing
    # to block: drop connections for the last 5% of calls and every
    # exploration answer grades wrong, pulling the shared reference frame down
    # (diligence 2026-10, finding 2). Interleaving in nonce order means the
    # network sees one undifferentiated stream. Failed calls are also recorded
    # as failed and excluded from the frame, which is what actually closes the
    # attack; the interleave removes the cheap version of it.
    # v3 sizes are consensus constants (config.EXPLORE_SHARED_SIZE / _PRIVATE_SIZE),
    # and the private half is drawn for THIS proof's hotkey: the validator
    # derives the same set from the hotkey it queried.
    if explore_models and (explore_size > 0 or protocol_version >= 3):
        pool_by_id = {q["question_id"]: q for q in benchmark_pool}
        plan_kwargs = {"hotkey": hotkey or None} if protocol_version >= 3 else {}
        if protocol_version >= 3:
            # The same cost-aware shares the validator draws with (pinned menu
            # and token manifest), or every proof fails the exploration check.
            plan_kwargs["shares"] = epoch_plan.explore_shares(protocol_version, explore_models)
        explore_map = epoch_plan.exploration_plan(
            protocol_version, nonce_bytes, benchmark_pool, set(question_ids), explore_models,
            explore_size, **plan_kwargs,
        )
        scheduled += [
            (pool_by_id[qid], explore_map[qid], True)
            for qid in sorted(explore_map) if qid in pool_by_id
        ]

    # The CALLS go out in nonce order; the RESULTS are recorded in slice order
    # with exploration after, as they always were. What the network sees is
    # the call order; what the proof lays out is a stable, inspectable list.
    call_order = sorted(scheduled, key=lambda item: hmac.new(
        nonce_bytes, f"order:{item[0]['question_id']}".encode("utf-8"), hashlib.sha256,
    ).digest())
    # Code questions are graded by running the answer in the code sandbox
    # (graders_code). A TD that cannot build it must not grade them: every code
    # question would grade 0 and the proof would read as a router that cannot
    # code. Checked before any call is made, so nothing is spent on a proof
    # that will not be produced.
    if any(q.get("grader_id") in SANDBOXED_CODE_CHECKERS for q, _m, _e in scheduled):
        require_code_sandbox()

    outcomes = {
        (q["question_id"], is_explore): (response_text, failed, cost, p_tok, c_tok, lat)
        for q, _model, is_explore, response_text, failed, cost, p_tok, c_tok, lat
        in _call_all(proxy, call_order, epoch_id, protocol_version, menu=menu)
    }

    for q, model_id, is_explore in scheduled:
        response_text, failed, cost, prompt_tokens, completion_tokens, latency_ms = outcomes[
            (q["question_id"], is_explore)
        ]
        # grade() needs a grader task dict, not a raw loader question: it reads
        # task["checker"]["id"] / task["domain"], neither of which the loader
        # schema has. Passing the raw dict raises KeyError inside grade(), which
        # catches it and returns 0 — every answer would grade wrong, silently.
        # Execution policy is one constant, read by the harness AND by the
        # loader that decides what is in the pool — see config.HARNESS_ALLOW_EXEC.
        # They drifted apart once and a sixth of every slice scored zero.
        # A failed call is graded as what it is -- no answer -- and recorded
        # as failed; the grade of "" is wrong by construction, so `correct`
        # stays False and the proof says why.
        correct = (not failed) and grade_question(q, response_text)
        # Canonical number types at the source: a failed call sums an empty
        # record list to the int 0, and canonical JSON tells 0 from 0.0. The
        # validator keeps numbers as serialised since #120; writing them as
        # float/int here means both sides hash the same bytes by construction.
        results.append(QuestionResult(
            question_id=q["question_id"],
            routed_model=model_id,
            correct=correct,
            cost_usd=float(cost),
            response_hash=hashlib.sha256(response_text.encode()).hexdigest(),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            is_exploration=is_explore,
            failed=failed,
            latency_ms=int(latency_ms),
        ))

    proof = BenchmarkProof(
        hotkey=hotkey,
        epoch_id=epoch_id,
        nonce=nonce,
        questions_hash=questions_hash,
        weights_hash=weights_hash,
        source_hash=source_hash,
        routing_protocol=identity_for(protocol_version),
        results=results,
        total_cost_usd=proxy.total_cost,
        per_model_costs=proxy.per_model_costs,
        attestation_quote=b"",  # filled by TEERuntime.generate_attestation
        timestamp=time.time(),
    )

    return proof


class CodeSandboxUnavailable(RuntimeError):
    """The code sandbox's self-test failed on this machine, so code questions cannot be graded."""


_CODE_SANDBOX: tuple[bool, str] | None = None
_CODE_SANDBOX_LOCK = threading.Lock()


def code_sandbox_status() -> tuple[bool, str]:
    """graders_code.sandbox_selftest(), run once per process and logged."""
    global _CODE_SANDBOX
    with _CODE_SANDBOX_LOCK:
        if _CODE_SANDBOX is None:
            _CODE_SANDBOX = graders_code.sandbox_selftest()
            if _CODE_SANDBOX[0]:
                logger.info("Code sandbox self-test passed: code questions can be graded")
            else:
                logger.error("CODE SANDBOX UNAVAILABLE (%s): this machine refuses to grade code "
                             "questions, so it produces no proof for an epoch that has any",
                             _CODE_SANDBOX[1])
        return _CODE_SANDBOX


def require_code_sandbox() -> None:
    ok, why = code_sandbox_status()
    if not ok:
        raise CodeSandboxUnavailable(f"the code sandbox self-test failed: {why}")


def grade_question(q: dict, response_text: str) -> bool:
    """Grade one reply with the checker its question names: graders_code for the v3 code
    checkers (in the sandbox, which must have passed its self-test), graders.py otherwise
    (with execution as config.HARNESS_ALLOW_EXEC says, off in every shipped image)."""
    task = build_grader_task(q)
    if q.get("grader_id") in SANDBOXED_CODE_CHECKERS:
        require_code_sandbox()
        return bool(graders_code.grade(task, response_text, allow_exec=True))
    return bool(grade(task, response_text, allow_exec=HARNESS_ALLOW_EXEC))


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


def _call_all(proxy, scheduled, epoch_id, protocol_version: int = 1, menu: str | None = None):
    """Make every scheduled model call, HARNESS_CONCURRENCY at a time, and yield
    (question, model_id, is_exploration, text, failed, cost, prompt_tokens,
    completion_tokens) in the ORDER SCHEDULED — never completion order. Cost is
    attributed by request id, so interleaved calls cannot bill one question for
    another's tokens, and a failed call that appended no record costs nothing.

    `failed` is True when `_call_model` returned None: the call did not
    complete, or what came back carried no `choices` (an upstream error, not
    an answer). A reply that completed but was not text is "" and NOT failed
    -- that is a wrong answer, and it is billed.
    """
    def one(item):
        q, model_id, _explore = item
        rid = f"{epoch_id}:{q['question_id']}:{model_id}"
        _request_ctx.request_id = rid
        started = time.monotonic()
        try:
            if protocol_version >= 3:
                # v3: the menu entry's whole request (provider model id,
                # reasoning, answer budget, pinned providers), sent as pinned.
                text = _call_model(proxy, model_id, q, request=request_for(
                    protocol_version, model_id, HARNESS_MAX_TOKENS, menu=menu))
            else:
                # v2 sends each model's single pinned reasoning setting; v1 calls exactly as before.
                setting = reasoning_for(protocol_version, model_id) if protocol_version != 1 else None
                text = (_call_model(proxy, model_id, q, reasoning=setting) if setting
                        else _call_model(proxy, model_id, q))
        finally:
            _request_ctx.request_id = ""
        latency_ms = int(round((time.monotonic() - started) * 1000))
        # A reply that is not text is a wrong answer, never an abort. One
        # aborted question used to take the whole proof.
        return rid, (text is None), ("" if text is None else text), latency_ms

    if not scheduled:
        return
    with ThreadPoolExecutor(max_workers=HARNESS_CONCURRENCY) as pool:
        outcomes = list(pool.map(one, scheduled))
    for (q, model_id, explore), (rid, failed, text, latency_ms) in zip(scheduled, outcomes):
        mine = [r for r in proxy.records if r.request_id == rid]
        yield (
            q, model_id, explore, text, failed,
            sum(r.cost_usd for r in mine),
            sum(r.prompt_tokens for r in mine),
            sum(r.completion_tokens for r in mine),
            latency_ms,
        )


def _call_model(
    proxy: MeteringProxy,
    model_id: str,
    question: dict,
    reasoning: dict | None = None,
    request: dict | None = None,
) -> str | None:
    """Call a model via the MeteringProxy.

    Returns the reply text, "" for a reply that completed but carried no
    text, and None when there was no reply to grade: the call did not complete,
    or what came back has no `choices`. The distinction is attested downstream
    (`QuestionResult.failed`): "" is a wrong answer that measures the model,
    None measures the network and the provider.
    """
    prompt = question["prompt"]
    if request is not None:
        # Protocol v3: the entry's pinned request fields (routing_protocol.
        # request_for), and the shared system prompt when the image pins one,
        # so the harness and anything serving the head call a model the same way.
        messages = [{"role": "user", "content": prompt}]
        if HARNESS_SYSTEM_PROMPT_V3:
            messages.insert(0, {"role": "system", "content": HARNESS_SYSTEM_PROMPT_V3})
        payload = json.dumps({**request, "messages": messages}).encode()
    else:
        payload = json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": HARNESS_MAX_TOKENS,
            # Protocol v2 pins exactly one OpenRouter `reasoning` setting per model
            # (routing_protocol.reasoning_for). Sent only when given, so a v1 call is
            # byte-identical to before.
            **({"reasoning": reasoning} if reasoning else {}),
        }).encode()

    headers = {"Content-Type": "application/json", "X-Fugal-Request": current_request_id()}
    wait_s = HARNESS_CALL_TIMEOUT_S
    if request is not None:
        # v3: the proxy prices the call under the ENTRY id (the table's key) and
        # holds it to the entry's own deadline; we wait a margin past that.
        from fugal_subnet import config as _config
        headers["X-Fugal-Entry"] = model_id
        wait_s = _config.call_deadline_s(request) + _config.HARNESS_WAIT_MARGIN_S
    req = Request(
        f"http://127.0.0.1:{proxy.port}/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )

    reconnected = False
    while True:
        try:
            # Longer than the proxy's own per-call deadline (PROXY_CALL_DEADLINE_S),
            # so the proxy's 502 reaches us before we give up on the proxy;
            # otherwise the proxy writes into a closed socket (BrokenPipe) and the
            # failure is logged twice as two different errors.
            with urlopen(req, timeout=wait_s) as resp:
                data = json.loads(resp.read())
            break
        except Exception as e:
            # One reconnect, and only when not a byte of a reply arrived. The
            # proxy writes nothing until the upstream call is over and metered,
            # so a connection that ended before its first byte is one the proxy
            # did not serve (in the opening burst of an epoch, one that never
            # left its accept queue), and asking again cannot bill the question
            # twice. Once a reply has started the call was made and may be
            # billed: that is a failed call, never a second one.
            if not reconnected and _proxy_never_answered(e):
                reconnected = True
                logger.warning("Proxy connection for %s ended before any reply (%s); "
                               "connecting once more", model_id, type(e).__name__)
                time.sleep(PROXY_RECONNECT_WAIT_S)
                continue
            logger.exception("Model call failed for %s", model_id)
            return None
    # The proxy relays only replies that carry `choices`; checked here as well
    # because this is the last step before grading. A body without them is an
    # error, and grading its empty text would attest a wrong answer the model
    # never gave.
    if not reply_has_choices(data):
        logger.error("Reply for %s carried no choices; recorded as failed, not graded", model_id)
        return None
    return _text_of(data)


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


