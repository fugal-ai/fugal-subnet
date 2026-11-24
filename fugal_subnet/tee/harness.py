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
import math
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
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
from fugal_subnet.tee.proof import (
    BenchmarkProof,
    Challenge,
    QuestionResult,
    compute_questions_hash,
)
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
    v4=None,
    v5=None,
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

        v4: protocol v4 only, the epoch's `td_v4.V4EpochInputs` (its menu
            inputs, question plan, endpoint reader and proposal settings).
        v5: protocol v5 only, the epoch's `td_v5.V5EpochInputs` (its v4 menu
            inputs and its exam task, or None: coverage only).

    Returns:
        BenchmarkProof with routing results and attestation.
    """
    if protocol_version >= 5:
        return _run_benchmark_v5(
            nonce=nonce, head_bytes=head_bytes, benchmark_pool=benchmark_pool, proxy=proxy,
            epoch_id=epoch_id, source_hash=source_hash, hotkey=hotkey, v5=v5)
    if protocol_version >= 4:
        return _run_benchmark_v4(
            nonce=nonce, head_bytes=head_bytes, benchmark_pool=benchmark_pool, proxy=proxy,
            hidden_states=hidden_states, slice_size=slice_size, epoch_id=epoch_id,
            source_hash=source_hash, hotkey=hotkey, v4=v4)
    head = load_head_from_npz(head_bytes)
    # v1 keeps calling benchmark_costs exactly as before; v2 prices on its own manifest and table.
    # `menu`: the epoch's v3 menu (routing_protocol.menu_for_epoch); the
    # validator verifies the head and prices the calls on the same one.
    menu_kw = {"menu": menu} if menu else {}
    costs = (benchmark_costs(head) if protocol_version == 1
             else benchmark_costs_for(head, protocol_version, **menu_kw))
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
        in _call_all(proxy, call_order, epoch_id, protocol_version, **menu_kw)
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


# --- protocol v4 (docs/decisions/LIVE_MENU_V4.md; the TD side, W3) -------------

# A challenge's calls (FAST_CROWN_V4.md 2): the challenger's own pick
# ("scored") and the holder's where it differs ("holder").
CHALLENGE_KINDS = ("scored", "holder")
# The harness's clock for a challenge's time budget and a provisional
# entry's (a function, so a test can stand in for time).
_clock = time.monotonic
# A provisional entry's calls (FAST_CROWN_V4.md 18(d)): the proxy holds each
# to this deadline (shorter than any entry's own), a failed call is retried
# once, and none is started once PROVISIONAL_BUDGET_S has passed since the
# epoch's calls began: the TD's proof goes out on time whatever a new entry's
# host does. A call not made is reported failed; validators leave every
# provisional call out of the exploration failure bound, the outage rule and
# the health counts, so it moves only the entry's own answered share. A
# provisional entry is never a challenge pick (picks_v4.challenge_picks).
PROVISIONAL_CALL_DEADLINE_S = 300
PROVISIONAL_BUDGET_S = 1200.0
# What a provisional entry's EXPLORATION calls may cost one TD in one epoch,
# all provisional entries together, in dollars of REALISED cost (the
# exploration draw's caps are on expected cost, at the menu's median token
# means, and a new entry's answers may be far longer). Before each attempt
# the TD holds back the most the attempt can cost: PROXY_MAX_ATTEMPTS
# upstream completions, each `max_tokens` completion tokens and at most the
# prompt's UTF-8 bytes plus PROVISIONAL_PROMPT_OVERHEAD_TOKENS prompt
# tokens, at the dearer of the entry's menu price and this TD's own read of
# its listing (`_ProvisionalSpend`); after it, what it did cost. An attempt
# that cannot be held back within the cap waits for calls in flight, and is
# not made (reported failed, against the entry's answered share only) when
# none is in flight. The cap admits one attempt of an entry at the price
# ceiling (menu_rules.MAX_PRICE_OUT_PICO) with the largest answer limit:
# 3 x 131,072 x $100/M is $39.32 of completion tokens, with room for a
# prompt of about 9,000 tokens at $25/M. So a box's provisional exploration
# spends at most $40 an epoch, whatever the entry.
PROVISIONAL_SPEND_CAP_USD = "40"
PROVISIONAL_PROMPT_OVERHEAD_TOKENS = 512


@dataclass(frozen=True)
class _Outcome:
    """One v4 call as the proxy recorded it, attributed by request id."""

    rid: str
    text: str
    failed: bool
    cost: object                 # exact dollars at the epoch's prices (Fraction, or 0)
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    billed: int | None           # the provider's billed nanodollars, summed; None if none
    served_by: str
    not_started: bool = False    # a challenge call not started: a budget ran out first
    rids: tuple = ()             # every attempt's request id (a provisional call's retry too)


def _run_benchmark_v4(*, nonce, head_bytes, benchmark_pool, proxy, hidden_states, slice_size,
                      epoch_id, source_hash, hotkey, v4) -> BenchmarkProof:
    """One protocol v4 epoch inside the TD.

    In order, and nothing is called before the first two succeed:

    1. The menu (`td_v4.epoch_menu`): the epoch's MenuState from the bytes
       handed in, against the committed hash, with the compiled rules applied
       to every entry. `MenuUnavailable` or `MenuRejected` propagates: no
       calls, no proof.
    2. The endpoint reads of every model on the menu (no spend), provisional
       entries included; a provisional entry this read prices far above its
       menu price is not called (`td_v4.overpriced`).
    3. The challenge and exploration calls, each with the wire request built
       from its entry's named fields. A challenge is made only when
       `v4.challenge` names one (the operator's size, and the holder the
       root's reveal of E - 2 names, its bytes checked here:
       td_v4.ChallengePolicy; FAST_CROWN_V4.md 2): the first `size`
       questions of the epoch's challenge list (`plan.challenge_list`, the
       slice first), the head's own pick on each and, where the holder's
       computed pick differs, the holder's, both over the menu's full entries
       (picks_v4.challenge_picks). The challenger's own calls are started
       first, in list order; then the holder-side calls and exploration,
       interleaved in the nonce's order (picks_v4.call_order_key). No
       challenge is made whose estimated cost (td_v4.challenge_estimate, at
       the epoch's pick costs) exceeds the operator's dollar ceiling. No
       challenge call is started more than `v4.challenge_budget_s` after the
       first call, nor once the challenge's realised spend plus the estimate
       of its calls in flight reaches the ceiling (`_call_all_v4`); a call
       started keeps its outcome. The
       challenge is reported at the largest size whose own calls were all
       started (none below 500: no challenge, an unanswered attempt), and a
       holder-side call not started is in the proof marked `not_started`
       (no item; picks_v4.challenge_verdict decides whether the challenge
       counts). The slice is drawn either way (exploration is drawn outside
       it, and its hash is in the proof), and the proof's `scored` and
       `challenge` fields say whether a challenge was answered. A
       provisional entry's calls run on their own deadline with one retry,
       and stop at the TD's budget (PROVISIONAL_CALL_DEADLINE_S,
       PROVISIONAL_BUDGET_S).
    4. The proof: `routing_protocol` is identity_v4 of the menu's hash, and
       the v4 fields are set.
    """
    from fugal_subnet import menu_state as ms
    from fugal_subnet import picks_v4
    from fugal_subnet.head_eval import load_head_v4, v4_head_verdict
    from fugal_subnet.routing_protocol import identity_v4
    from fugal_subnet.tee import endpoint_reads as er
    from fugal_subnet.tee import td_v4

    if not isinstance(v4, td_v4.V4EpochInputs):
        raise ValueError("a protocol v4 epoch needs its td_v4.V4EpochInputs")
    if not hotkey:
        raise ValueError("a protocol v4 proof is produced for a hotkey")

    # 1. The menu first: without one there is nothing to call.
    state = v4.menu()
    menu_hash = ms.menu_hash(state)
    requests = {e.id: td_v4.entry_request(state, e.id) for e in state.entries}

    # The head (section 11): a v4 head, with a row for at least one entry of
    # this menu, or nothing is read or called.
    why = v4_head_verdict(head_bytes, state)
    if why:
        raise ValueError(why)
    head = load_head_v4(head_bytes)
    weights_hash = hashlib.sha256(head_bytes).hexdigest()
    nonce_bytes = bytes.fromhex(nonce) if len(nonce) == 64 else nonce.encode()
    questions = v4.plan.scored_slice(nonce_bytes, benchmark_pool, slice_size)
    question_ids = [q["question_id"] for q in questions]
    questions_hash = compute_questions_hash(question_ids)
    pool_by_id = {q["question_id"]: q for q in benchmark_pool}
    q_to_pool_idx = {q["question_id"]: i for i, q in enumerate(benchmark_pool)}

    # 2. Endpoint reads, before any model call.
    reads = er.read_endpoints(td_v4.wanted_reads(state), v4.fetch_listing)
    provisional = frozenset(e.id for e in state.entries if e.provisional)
    skipped = td_v4.overpriced(state, reads)

    # 3. The head's picks on the menu (head_odds.picks_and_odds_v4: exact odds
    #    minus lambda x the menu's cost, the function validators use), and
    #    exploration.
    scheduled: list[tuple[dict, str, str, dict, str]] = []   # (q, entry, kind, request, price key)
    routable = []
    for q in questions:
        idx = q_to_pool_idx.get(q["question_id"])
        if idx is None or idx >= hidden_states.shape[0]:
            logger.warning("Question %s not in hidden states, skipping", q["question_id"])
            continue
        routable.append((q, idx))
    challenge = v4.challenge
    if challenge is not None:
        listed = v4.plan.challenge_list(nonce_bytes, benchmark_pool, slice_size)[:challenge.size]
        idx = [q_to_pool_idx.get(q["question_id"]) for q in listed]
        if len(listed) < challenge.size or any(i is None or i >= hidden_states.shape[0]
                                                for i in idx):
            logger.error("Epoch %d: the challenge list does not hold %d questions this TD can "
                         "route: no challenge this epoch", v4.epoch, challenge.size)
            challenge = None
    if challenge is not None:
        hidden = hidden_states[idx]
        own = picks_v4.challenge_picks(head, state, hidden)
        theirs = (picks_v4.challenge_picks(challenge.holder_head, state, hidden)
                  if challenge.holder_head is not None else (None,) * len(listed))
        if any(mine is None for mine in own):
            logger.error("Epoch %d: the head has odds for no full entry on some challenge "
                         "question: no challenge this epoch", v4.epoch)
            challenge = None
    if challenge is not None:
        # The dollar ceiling, checked first on the estimate: no challenge
        # whose cost at the epoch's pick costs exceeds it.
        estimate = td_v4.challenge_estimate(state, own, theirs)
        if estimate > challenge.spend_usd:
            logger.error("Epoch %d: the challenge's estimated cost, $%.2f at the epoch's pick "
                         "costs, exceeds the operator's ceiling of $%d (FUGAL_V4_CHALLENGE_USD): "
                         "no challenge this epoch", v4.epoch, float(estimate),
                         challenge.spend_usd)
            challenge = None
    if challenge is not None:
        for q, mine, hp in zip(listed, own, theirs):
            scheduled.append((q, mine, "scored", requests[mine], mine))
            if hp is not None and hp != mine:
                scheduled.append((q, hp, "holder", requests[hp], hp))
    del routable
    explore_map = v4.plan.exploration(nonce_bytes, benchmark_pool, set(question_ids), state,
                                      hotkey)
    scored_set = set(question_ids)
    for qid in sorted(explore_map):
        eid = explore_map[qid]
        if eid not in requests:
            raise td_v4.MenuRejected(f"exploration names {eid!r}, which is not on the menu")
        if qid in pool_by_id and qid not in scored_set:
            scheduled.append((pool_by_id[qid], eid, "explore", requests[eid], eid))

    every_question = [s[0] for s in scheduled]
    if any(q.get("grader_id") in SANDBOXED_CODE_CHECKERS for q in every_question):
        require_code_sandbox()

    proxy.prices = td_v4.proxy_prices(state)
    proxy.clear()
    # The challenger's own calls are started first, in list order, so its
    # side is never starved by the holder's; then the holder-side calls and
    # exploration, interleaved in the nonce's order.
    call_order = [item for item in scheduled if item[2] == "scored"] + sorted(
        [item for item in scheduled if item[2] != "scored"],
        key=lambda item: picks_v4.call_order_key(nonce_bytes, item[0]["question_id"]))
    budget = v4.challenge_budget_s if challenge is not None else None
    spend_cap = challenge.spend_usd if challenge is not None else None
    outcomes = {(item[0]["question_id"], item[2]): o
                for item, o in zip(call_order, _call_all_v4(
                    proxy, call_order, epoch_id, budget_s=budget, provisional=provisional,
                    skipped=skipped, spend_cap=spend_cap,
                    call_estimate=ms.pick_costs(state) if spend_cap is not None else None,
                    spend=_ProvisionalSpend(provisional_prices(state, reads))))}
    if challenge is not None:
        size = reported_size(challenge.size, [
            not outcomes[(q["question_id"], "scored")].not_started for q in listed])
        if size is None:
            logger.error("Epoch %d: fewer than %d of the challenge's own calls were started "
                         "inside its budget: it is not reported (an unanswered attempt)",
                         v4.epoch, min(td_v4.CHALLENGE_CHOICES[1:]))
            challenge = None
            scheduled = [item for item in scheduled if item[2] not in CHALLENGE_KINDS]
        else:
            if size < challenge.size:
                logger.warning("Epoch %d: the challenge's own calls past question %d were not "
                               "all started inside its budget: it is reported at %d", v4.epoch,
                               size, size)
            kept_q = {q["question_id"] for q in listed[:size]}
            scheduled = [item for item in scheduled
                         if item[2] not in CHALLENGE_KINDS or item[0]["question_id"] in kept_q]
            import dataclasses

            challenge = dataclasses.replace(challenge, size=size)
            cut = sum(1 for item in scheduled if item[2] == "holder"
                      and outcomes[(item[0]["question_id"], "holder")].not_started)
            if cut:
                logger.warning("Epoch %d: %d holder-side challenge calls were not started "
                               "inside the budget (no items)", v4.epoch, cut)

    # 4. Grade, in the proof's order: scored, then exploration.
    results: list[QuestionResult] = []
    kept: set[str] = set()

    def add(q: dict, eid: str, o: _Outcome, *, explore: bool, holder: bool = False):
        kept.update(o.rids)
        results.append(QuestionResult(
            question_id=q["question_id"], routed_model=eid,
            correct=(not o.failed) and grade_question(q, o.text), cost_usd=float(o.cost),
            response_hash=hashlib.sha256(o.text.encode()).hexdigest(),
            prompt_tokens=int(o.prompt_tokens), completion_tokens=int(o.completion_tokens),
            is_exploration=explore, failed=o.failed, latency_ms=int(o.latency_ms),
            provider_cost=o.billed, served_by=o.served_by,
            holder_pick=True if holder else None,
            not_started=True if (holder and o.not_started) else None))

    for q, eid, kind, _req, _key in scheduled:
        add(q, eid, outcomes[(q["question_id"], kind)], explore=(kind == "explore"),
            holder=(kind == "holder"))

    records = [r for r in proxy.records if r.request_id in kept]
    by_key: dict[str, list] = {}
    for r in records:
        by_key.setdefault(r.model_id, []).append(r.cost_usd)
    return BenchmarkProof(
        hotkey=hotkey, epoch_id=epoch_id, nonce=nonce, questions_hash=questions_hash,
        weights_hash=weights_hash, source_hash=source_hash,
        routing_protocol=identity_v4(menu_hash), results=results,
        total_cost_usd=math.fsum(r.cost_usd for r in records),
        per_model_costs={k: math.fsum(v) for k, v in sorted(by_key.items())},
        attestation_quote=b"", timestamp=time.time(),
        menu_hash=menu_hash, endpoint_reads=reads,
        scored=challenge is not None,
        challenge=(Challenge(size=challenge.size, holder=challenge.holder)
                   if challenge is not None else None))


# --- protocol v5 (~/projects/fugal-v2-handoff/v5/DESIGN.md section 5; td_v5) -------------


def _run_benchmark_v5(*, nonce, head_bytes, benchmark_pool, proxy, epoch_id, source_hash,
                      hotkey, v5) -> BenchmarkProof:
    """One protocol v5 epoch inside the TD.

    1. The menu, exactly as under v4 (`v5.menu()`: td_v4.epoch_menu on the
       v4 session's inputs); the head is checked against it as under v4 and
       bound into the proof by its hash. The TD makes no picks.
    2. Endpoint reads before any call; an overpriced provisional entry is
       not called (as under v4).
    3. The calls: this hotkey's exam cells (`v5.exam.cells`, the root's spec
       checked in td_v5.ExamPolicy), then its coverage cells
       (td_v5.coverage_v5, disjoint from the spec's questions), started in
       that order. None is started more than `v5.budget_s` after the first
       (`not_started`) and every one is held to at most `v5.call_deadline_s`,
       so the proof is ready before the validators' collection whatever a
       host does. A provisional entry's coverage calls stay within v4's
       provisional spend cap.
    4. Each reply is labelled 0..Y_SCALE (td_v5.grade_y; a failed call is 0)
       and the proof carries the spec's hash (`spec_hash`, None for a
       coverage-only epoch). `scored` is False: v5 has no challenge.
    """
    from fugal_subnet import menu_state as ms
    from fugal_subnet.head_eval import v4_head_verdict
    from fugal_subnet.routing_protocol import identity_v4
    from fugal_subnet.tee import endpoint_reads as er
    from fugal_subnet.tee import td_v4, td_v5

    if not isinstance(v5, td_v5.V5EpochInputs):
        raise ValueError("a protocol v5 epoch needs its td_v5.V5EpochInputs")
    if not hotkey:
        raise ValueError("a protocol v5 proof is produced for a hotkey")

    # 1. The menu and the head.
    state = v5.menu()
    menu_hash = ms.menu_hash(state)
    requests = {e.id: td_v4.entry_request(state, e.id) for e in state.entries}
    why = v4_head_verdict(head_bytes, state)
    if why:
        raise ValueError(why)
    weights_hash = hashlib.sha256(head_bytes).hexdigest()
    nonce_bytes = bytes.fromhex(nonce) if len(nonce) == 64 else nonce.encode()
    # Every v5 call is on the v5 pool (arena_sets); `benchmark_pool` is the v3
    # pool the v4 menu plan draws on, used only when no v5 pool is given.
    v5_pool = list(v5.pool) or list(benchmark_pool)
    pool_by_id = {q["question_id"]: q for q in v5_pool}

    # 2. Endpoint reads, before any model call.
    reads = er.read_endpoints(td_v4.wanted_reads(state), v5.v4.fetch_listing)
    provisional = frozenset(e.id for e in state.entries if e.provisional)
    skipped = td_v4.overpriced(state, reads)

    # 3. The exam cells, then the coverage cells.
    scheduled: list[tuple[dict, str, str, dict, str]] = []   # (q, entry, kind, request, price key)
    exam = v5.exam
    exam_qids: list[str] = []
    spec_questions: set[str] = set()
    if exam is not None:
        spec_questions = {str(q) for q in getattr(exam.spec, "questions", ())}
        missing = 0
        for qid, eid in exam.cells:
            q = pool_by_id.get(qid)
            if q is None or eid not in requests:
                missing += 1
                continue
            scheduled.append((q, eid, "exam", requests[eid], eid))
            exam_qids.append(qid)
        if missing:
            logger.warning("Epoch %d: %d of this hotkey's %d exam cells name a question not in "
                           "this TD's pool or an entry not on the epoch's menu: not called",
                           v5.epoch, missing, len(exam.cells))
    coverage = td_v5.coverage_v5(nonce_bytes, v5_pool, spec_questions | set(exam_qids),
                                 state, hotkey)
    for qid in sorted(coverage):
        eid = coverage[qid]
        if eid not in requests:
            raise td_v4.MenuRejected(f"coverage names {eid!r}, which is not on the menu")
        if qid in pool_by_id:
            scheduled.append((pool_by_id[qid], eid, "coverage", requests[eid], eid))

    if any(s[0].get("grader_id") in SANDBOXED_CODE_CHECKERS
           or (s[0].get("metadata") or {}).get("checker") for s in scheduled):
        require_code_sandbox()

    proxy.prices = td_v4.proxy_prices(state)
    proxy.clear()
    outcomes = _call_all_v4(
        proxy, scheduled, epoch_id, budget_s=v5.budget_s, provisional=provisional,
        skipped=skipped, spend=_ProvisionalSpend(provisional_prices(state, reads)),
        cut_kinds=td_v5.EXAM_KINDS, call_deadline_s=v5.call_deadline_s)
    cut = sum(1 for o in outcomes if o.not_started)
    if cut:
        logger.warning("Epoch %d: %d of %d calls were not started inside the %ds budget "
                       "(reported not started)", v5.epoch, cut, len(outcomes), int(v5.budget_s))

    # 4. Label, in the proof's order: exam cells, then coverage.
    results: list[QuestionResult] = []
    kept: set[str] = set()
    for (q, eid, kind, _req, _key), o in zip(scheduled, outcomes):
        kept.update(o.rids)
        y = 0 if o.failed else td_v5.grade_y(q, o.text)
        results.append(QuestionResult(
            question_id=q["question_id"], routed_model=eid,
            correct=(y == td_v5.Y_SCALE), cost_usd=float(o.cost),
            response_hash=hashlib.sha256(o.text.encode()).hexdigest(),
            prompt_tokens=int(o.prompt_tokens), completion_tokens=int(o.completion_tokens),
            is_exploration=(kind == "coverage"), failed=o.failed, latency_ms=int(o.latency_ms),
            provider_cost=o.billed, served_by=o.served_by,
            not_started=True if o.not_started else None,
            exam=(kind == "exam"), y=y))

    records = [r for r in proxy.records if r.request_id in kept]
    by_key: dict[str, list] = {}
    for r in records:
        by_key.setdefault(r.model_id, []).append(r.cost_usd)
    return BenchmarkProof(
        hotkey=hotkey, epoch_id=epoch_id, nonce=nonce,
        questions_hash=compute_questions_hash(sorted(set(exam_qids))),
        weights_hash=weights_hash, source_hash=source_hash,
        routing_protocol=identity_v4(menu_hash), results=results,
        total_cost_usd=math.fsum(r.cost_usd for r in records),
        per_model_costs={k: math.fsum(v) for k, v in sorted(by_key.items())},
        attestation_quote=b"", timestamp=time.time(),
        menu_hash=menu_hash, endpoint_reads=reads, scored=False, challenge=None,
        spec_hash=exam.spec_hash if exam is not None else None)


class _ProvisionalSpend:
    """PROVISIONAL_SPEND_CAP_USD over one epoch's provisional exploration
    calls: `hold(worst)` before an attempt (False: not made), `settle(worst,
    cost)` after it. Exact dollars (Fractions)."""

    def __init__(self, prices, *, cap=None):
        from fractions import Fraction

        self.prices = dict(prices)          # entry id -> (pico in, pico out, max_tokens)
        self.cap = Fraction(PROVISIONAL_SPEND_CAP_USD if cap is None else cap)
        self.spent = self.held = Fraction(0)
        self._cond = threading.Condition()

    def worst(self, eid: str, question: dict):
        from fractions import Fraction

        from fugal_subnet.tee.runtime import PROXY_MAX_ATTEMPTS

        p_in, p_out, max_tokens = self.prices[eid]
        prompt = (len(str(question.get("prompt", "")).encode("utf-8"))
                  + len((HARNESS_SYSTEM_PROMPT_V3 or "").encode("utf-8"))
                  + PROVISIONAL_PROMPT_OVERHEAD_TOKENS)
        return Fraction(PROXY_MAX_ATTEMPTS * (p_in * prompt + p_out * max_tokens), 10**12)

    def cost(self, eid: str, records):
        from fractions import Fraction

        p_in, p_out, _ = self.prices[eid]
        return Fraction(sum(p_in * r.prompt_tokens + p_out * r.completion_tokens
                            for r in records), 10**12)

    def hold(self, worst, may_wait) -> bool:
        with self._cond:
            while self.spent + self.held + worst > self.cap:
                if self.held == 0 or not may_wait():
                    return False
                self._cond.wait(timeout=1.0)
            self.held += worst
            return True

    def settle(self, worst, cost) -> None:
        with self._cond:
            self.held -= worst
            self.spent += cost
            self._cond.notify_all()


def provisional_prices(state, reads) -> dict:
    """{entry id: (picodollars in, out, max_tokens)} of every provisional
    entry: the dearer of its menu price and this TD's read of its listing
    (the dearer listed pinned host), as `_ProvisionalSpend` prices it."""
    by_model = {r.model: r for r in reads}
    out = {}
    for e in state.entries:
        if not e.provisional:
            continue
        read = by_model.get(e.model)
        hosts = [h for h in (read.hosts if read is not None else ())
                 if h.listed and h.host in e.provider_order]
        p_in = max([e.price_in] + [h.price_in for h in hosts])
        p_out = max([e.price_out] + [h.price_out for h in hosts])
        out[e.id] = (int(p_in), int(p_out), int(e.max_tokens))
    return out


def reported_size(size: int, own_started) -> int | None:
    """The size a challenge of `size` questions is reported at, given
    whether each of its own calls (in list order) was started: the largest
    challenge size up to `size` whose own calls were all started, or None
    when that is less than the smallest (no challenge)."""
    from fugal_subnet.tee.proof import CHALLENGE_SIZES

    own_started = list(own_started)
    if all(own_started):
        return size
    for s in sorted((s for s in CHALLENGE_SIZES if s <= size), reverse=True):
        if len(own_started) >= s and all(own_started[:s]):
            return s
    return None


def _call_all_v4(proxy, items, epoch_id, *, budget_s=None, provisional=frozenset(),
                 skipped=frozenset(), clock=None, spend_cap=None, call_estimate=None,
                 spend=None, cut_kinds=CHALLENGE_KINDS, call_deadline_s=None) -> list[_Outcome]:
    """Make v4 calls, HARNESS_CONCURRENCY at a time, and return their outcomes
    in the order given. `items` are (question, entry id, kind, request, price
    key); kind is "scored" or "holder" (a challenge's calls) or "explore".
    Calls are started in the order given, each decided in that order: with
    `budget_s`, no challenge call is started more than `budget_s` after the
    first call, and once one is not started no later one is (`not_started`);
    a call started keeps its outcome whenever it ends. With `spend_cap`
    (dollars), likewise no challenge call is started once the challenge
    calls' realised cost plus the estimate of those in flight (each at
    `call_estimate[entry]`, the epoch's pick cost, until it is paid) has
    reached it. Other calls are never cut by either. Each request
    is sent as built (`td_v4.wire_request`), priced by the proxy under its
    price key, and attributed by request id: tokens and cost exactly, the
    provider's billed nanodollars summed over the call's upstream
    completions, and the provider that served the last of them.

    A call to an entry in `provisional` asks the proxy for
    PROVISIONAL_CALL_DEADLINE_S, is retried once when it fails (its outcome
    is the retry's; both attempts' records are kept), and is not started
    once PROVISIONAL_BUDGET_S has passed since the first call; with `spend`
    (a `_ProvisionalSpend`), each attempt of a provisional entry's
    exploration call is held within its cap. A call to an entry in
    `skipped` is not made. A call not made is failed.

    Protocol v5 (`_run_benchmark_v5`) passes `cut_kinds` ("exam",
    "coverage"): the budget then cuts those kinds instead of a challenge's,
    and `call_deadline_s` holds every call to at most that deadline (a
    provisional entry's to the shorter of it and its own)."""
    from fugal_subnet.tee import td_v4

    clock = _clock if clock is None else clock
    started_at = clock()
    # Start decisions are made one at a time in the items' order, so the
    # challenge calls not started are always the last of their order.
    gate = threading.Condition()
    turn = [0]
    cut = [False]
    spent = [Fraction(0)]                # challenge calls paid, realised
    in_flight = [Fraction(0)]            # challenge calls started, not yet paid, estimated
    spent_lock = threading.Lock()
    estimates = dict(call_estimate or {})

    def challenge_cut(kind: str, eid: str):
        """(not started, the estimate held for the call while in flight)."""
        if kind not in cut_kinds:
            return False, None
        if not cut[0] and budget_s is not None and clock() - started_at > budget_s:
            cut[0] = True
        held = None
        if not cut[0] and spend_cap is not None:
            with spent_lock:
                if spent[0] + in_flight[0] >= spend_cap:
                    cut[0] = True
                else:
                    held = Fraction(estimates.get(eid, 0))
                    in_flight[0] += held
        return cut[0], held

    def add_spend(rid: str, held) -> None:
        cost = sum((Fraction(r.cost_usd) for r in list(proxy.records) if r.request_id == rid),
                   Fraction(0))
        with spent_lock:
            in_flight[0] -= held
            spent[0] += cost

    def rid_of(item, attempt: int = 0) -> str:
        q, eid, kind = item[0], item[1], item[2]
        base = f"{epoch_id}:{q['question_id']}:{eid}"
        # A challenge question may also be one of this hotkey's exploration
        # questions: its calls carry ids of their own.
        if kind == "scored":
            base = f"{base}:c"
        elif kind == "holder":
            base = f"{base}:h"
        elif kind == "exam":
            base = f"{base}:x"
        return base if attempt == 0 else f"{base}:retry"

    def attempt(item, n: int, i: int | None = None):
        eid, kind = item[1], item[2]
        rid = rid_of(item, n)
        charged = None
        if i is not None:
            with gate:
                while turn[0] != i:
                    gate.wait()
                try:
                    stop, charged = challenge_cut(kind, eid)
                finally:
                    turn[0] += 1
                    gate.notify_all()
            if stop:
                return rid, None, 0, True        # past a budget: not called
        try:
            return _attempt(item, rid)
        finally:
            if charged is not None:
                add_spend(rid, charged)

    def _attempt(item, rid: str):
        q, eid, kind, request, price_key = item
        td_v4.check_wire_request(request)
        if eid in skipped or (eid in provisional and clock() - started_at > PROVISIONAL_BUDGET_S):
            return rid, None, 0, False
        held = None
        if spend is not None and kind in ("explore", "coverage") and eid in provisional:
            held = spend.worst(eid, q)
            if not spend.hold(held, lambda: clock() - started_at <= PROVISIONAL_BUDGET_S):
                logger.warning("A provisional call (%s) is not made: the epoch's provisional "
                               "spend cap of $%s is reached", eid, PROVISIONAL_SPEND_CAP_USD)
                return rid, None, 0, False
        _request_ctx.request_id = rid
        t0 = clock()
        try:
            # Only a provisional entry's call asks for a deadline of its own
            # under v4; under v5 every call is held to `call_deadline_s`.
            deadline = PROVISIONAL_CALL_DEADLINE_S if eid in provisional else None
            if call_deadline_s is not None:
                deadline = (int(call_deadline_s) if deadline is None
                            else min(deadline, int(call_deadline_s)))
            text = _call_model(proxy, price_key, q, request=request,
                               **({"deadline_s": deadline} if deadline is not None else {}))
        finally:
            _request_ctx.request_id = ""
            if held is not None:
                spend.settle(held, spend.cost(eid, [r for r in proxy.records
                                                    if r.request_id == rid]))
        return rid, text, int(round((clock() - t0) * 1000)), False

    def one(indexed):
        i, item = indexed
        rid, text, latency_ms, not_started = attempt(item, 0, i)
        rids = (rid,)
        if text is None and not not_started and item[1] in provisional and item[1] not in skipped:
            rid, text, latency_ms, not_started = attempt(item, 1)
            rids = rids + (rid,)
        return rids, text, latency_ms, not_started

    if not items:
        return []
    with ThreadPoolExecutor(max_workers=HARNESS_CONCURRENCY) as pool:
        done = list(pool.map(one, enumerate(items)))
    out = []
    for rids, text, latency_ms, not_started in done:
        last = [r for r in proxy.records if r.request_id == rids[-1]]
        billed = [r.billed_nano for r in last if r.billed_nano is not None]
        out.append(_Outcome(
            rid=rids[-1], text="" if text is None else text, failed=text is None,
            cost=sum((r.cost_usd for r in last), 0),
            prompt_tokens=sum(r.prompt_tokens for r in last),
            completion_tokens=sum(r.completion_tokens for r in last),
            latency_ms=latency_ms, billed=sum(billed) if billed else None,
            served_by=last[-1].served_by if last else "", not_started=bool(not_started),
            rids=rids))
    return out


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
                    protocol_version, model_id, HARNESS_MAX_TOKENS,
                    **({"menu": menu} if menu else {})))
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
    deadline_s: int | None = None,
) -> str | None:
    """Call a model via the MeteringProxy. `deadline_s` (protocol v4, a
    provisional entry's call) asks the proxy for a shorter deadline than the
    request's own (`X-Fugal-Deadline`); None sends no such header.

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
        if deadline_s is not None and 0 < int(deadline_s) < _config.call_deadline_s(request):
            headers["X-Fugal-Deadline"] = str(int(deadline_s))
            wait_s = int(deadline_s) + _config.HARNESS_WAIT_MARGIN_S
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


