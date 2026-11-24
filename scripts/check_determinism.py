#!/usr/bin/env python3
"""Differential determinism harness — invariant I1.

Consensus rests on one property: two honest validators, given the same epoch
inputs, produce identical scores. Nothing else in this repo tests that
directly. Reviews sample; this holds for every run.

The harness computes the full scoring pipeline twice in *separate processes*
and diffs every intermediate artifact. Separate processes matter — an in-process
repeat shares warmed caches, a resident BLAS handle, and an already-chosen
kernel, so it agrees for reasons that do not generalize to two machines.

    python scripts/check_determinism.py                  # same-host repeat
    python scripts/check_determinism.py --perturb        # simulate a second host
    python scripts/check_determinism.py --json           # machine-readable

--perturb is the interesting mode. It re-runs the second process with the CPU
dispatch pinning stripped, which is what a validator on different hardware
effectively looks like. If a stage diverges only under --perturb, that stage
depends on kernel dispatch and is a live consensus risk.

Exit code 0 means every stage matched. Non-zero names the first stage that did
not, so CI fails on the regression rather than waiting for someone to notice
weights drifting apart on mainnet.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stages are ordered so the first mismatch is the root cause: a differing slice
# explains a differing matrix, which explains differing scores. Reporting the
# earliest divergence points at the actual bug instead of its symptoms.
# These mirror the LIVE validator path, in the order it executes them. They
# used to mirror the pre-TEE path (build_matrix_mock / evaluate_head), which
# the TEE validator does not import at all — so I1's executable check was
# guarding code that no longer runs in production.
STAGES = (
    "slice",
    "epoch_plan_v2",
    "exploration",
    "synthetic_hidden",
    "routing_decisions",
    "proof",
    "verification",
    "copy_rule",
    "shared_grades",
    "frame",
    "scores",
    "weights",
    "v2_epoch",
    "seat_confirmation",
    "exploration_private",
    "head_odds",
    "explore_pool",
    "odds_seat",
    "explore_weights",
    "menu_derivation",
    "seat_v4",
    "explore_shares_v4",
    "endpoint_vote",
    "token_means",
    "provisional_gates",
    "retirement",
    "menu_publication",
    "head_odds_v4",
    "pool_mix_v4",
    "bound_v4",
    "pick_cost_v4",
    "picks_v4",
    "memo_v4",
    "v4_replay",
    "published_v4",
    "exam_spec_v5",
    "arena_items_v5",
    "seat_v5",
    "v5_epoch",
)


def _digest(obj) -> str:
    """Stable hash of a stage's output.

    float values are formatted with repr(), which round-trips exactly, so this
    catches divergence in the last bit rather than hiding it behind rounding.
    """
    def canon(x):
        if isinstance(x, float):
            return repr(x)
        if isinstance(x, dict):
            return {str(k): canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [canon(v) for v in x]
        if hasattr(x, "tolist"):
            return canon(x.tolist())
        return x

    payload = json.dumps(canon(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_pipeline(seed: int) -> dict:
    """Run the LIVE validator pipeline once and return a digest per stage.

    Mock attestation, stubbed model replies, no chain and no network — but
    every float and every hash after that point is the real one: the real
    harness routes, the real verifier checks, the real reference frame
    accumulates, the real scorer scores.

    The harness runs in protocol v1 pricing mode here, because v2 pricing
    needs a head built against the real pinned manifest. scripts/rehearsal_v2.py
    and tests/ cover that path; the v2 stages below (selection, copy rule,
    shared grades, the whole v2 epoch) are exercised on these proofs.
    """
    import numpy as np

    from fugal_subnet import success_test_fixtures as success_fixtures
    from fugal_subnet.benchmarks.slicer import (
        derive_nonce,
        epoch_id_for_block,
        select_slice,
    )
    from fugal_subnet.config import HEAD_HIDDEN_DIM
    from fugal_subnet.exploration import expected_exploration
    from fugal_subnet.head_eval import HeadScore
    from fugal_subnet.reference_frame import (
        Observation,
        ReferenceFrame,
        accumulate_exploration,
        aggregate_observations,
        best_model,
        reference_cost,
    )
    from fugal_subnet.rewards import compute_weights
    from fugal_subnet.scoring import ScoringState, update_scores
    from fugal_subnet.tee import harness as harness_mod
    from fugal_subnet.tee.proof import compute_questions_hash
    from fugal_subnet.tee.runtime import APICallRecord, MeteringProxy, TEERuntime
    from fugal_subnet.tee.verify import verify_proof

    models = [f"vendor/model-{i}" for i in range(8)]
    prices = {m: (1e-7 * (i + 1), 2e-7 * (i + 1)) for i, m in enumerate(models)}
    skill = {m: 0.2 + 0.08 * i for i, m in enumerate(models)}

    pool = [
        {
            "question_id": f"q{i}",
            "prompt": f"Question {i}: compute something deterministic.",
            "gold": str(i),
            "grader_id": "numeric_final",
            "benchmark": ["gsm8k", "math", "mmlu", "aime"][i % 4],
            "metadata": {},
        }
        for i in range(240)
    ]

    epoch_id = epoch_id_for_block(seed)
    nonce = derive_nonce(epoch_id, f"0x{seed:064x}")
    questions = select_slice(nonce, pool, size=64)
    question_ids = [q["question_id"] for q in questions]
    explore_map = expected_exploration(nonce, pool, set(question_ids), models, 8)

    # Synthetic, and named so. These stand in for the backbone's output; they
    # are NOT the backbone. numpy's RandomState is bit-identical everywhere by
    # specification, which is exactly why it makes a good fixed input: it holds
    # the embedding constant so any divergence downstream belongs to the
    # scoring path, which is what validators must agree on.
    #
    # The real Qwen3-0.6B forward pass is NOT covered here, and does not need to
    # be: validators never run it. It is a miner-side computation whose output
    # feeds only that miner's own routing, and under --live the architecture is
    # pinned by the approved TDX measurement. Measured directly, x86_64 and
    # aarch64 disagree in the last bits of float32 (~2e-9) — see
    # docs/INVARIANTS.md, I1.
    rng = np.random.RandomState(seed)
    hidden = rng.randn(len(pool), HEAD_HIDDEN_DIM).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)

    class _Proxy(MeteringProxy):
        def start(self):
            self.prices = prices

    def _stub_call(proxy, model_id, question):
        digest = hashlib.sha256(
            f"{question['question_id']}|{model_id}".encode()
        ).digest()
        correct = (digest[0] / 255.0) < skill[model_id]
        text = question["gold"] if correct else "0"
        proxy.records.append(APICallRecord(
            model_id=model_id, prompt_tokens=500, completion_tokens=300,
            cost_usd=proxy.price_call(model_id, 500, 300), timestamp=0.0,
            response_hash=hashlib.sha256(text.encode()).hexdigest(),
            request_id=harness_mod.current_request_id(),
        ))
        return text

    harness_mod.benchmark_costs = success_fixtures.costs
    harness_mod._call_model = _stub_call

    # Heads whose utilities land close together on purpose, so the argmax
    # quantization is genuinely exercised rather than trivially agreed.
    proof_objs, proofs, routing, epoch_scores, head_hashes, verdicts = (
        {}, {}, {}, {}, {}, {}
    )
    for uid in range(1, 5):
        hrng = np.random.RandomState(1000 + uid)
        W = (hrng.randn(len(models), HEAD_HIDDEN_DIM) * 0.01).astype(np.float32)
        b = (hrng.randn(len(models)) * 1e-6).astype(np.float32)
        buf = io.BytesIO()
        np.savez(buf, **success_fixtures.arrays(W, b, models))
        head_bytes = buf.getvalue()

        proxy = _Proxy(port=0)
        proxy.start()
        proof = harness_mod.run_benchmark(
            nonce=nonce.hex(), head_bytes=head_bytes, benchmark_pool=pool,
            proxy=proxy, hidden_states=hidden, slice_size=64,
            epoch_id=epoch_id, source_hash="det-image",
            explore_models=models, explore_size=8,
        )
        # Wall clock is a legitimate input that differs between runs, and it
        # is attested, so two honest miners DO produce different content
        # hashes. Pin it here so the check measures the routing, grading and
        # scoring arithmetic rather than the clock.
        proof.timestamp = 0.0
        # Per-call latency is wall clock too, and also attested (each result's
        # latency_ms is in content_hash). The stubbed calls take 0 ms on an
        # idle host and 1 ms under load, so unpinned it made the two children
        # disagree on `proof` whenever the machine was busy (reproduced under
        # CPU load, 2026-11-08). Pinned like the timestamp; it decides nothing
        # downstream.
        for r in proof.results:
            if r.latency_ms is not None:
                r.latency_ms = 0
        proof.attestation_quote = TEERuntime(mock=True).generate_attestation(
            bytes.fromhex(proof.content_hash())
        )

        result = verify_proof(
            proof, approved_measurements=set(),
            expected_questions_hash=compute_questions_hash(question_ids),
            expected_nonce=nonce.hex(),
            gold_answers={q["question_id"]: q for q in pool},
            expected_question_ids=set(question_ids),
            expected_exploration=explore_map,
            expected_weights_hash=proof.weights_hash,
            expected_proof_hash=proof.content_hash(),
            head_bytes=head_bytes, mock=True,
        )
        verdicts[uid] = (result.valid, result.reason)
        proof_objs[uid] = proof
        proofs[uid] = proof.content_hash()
        routing[uid] = [r.routed_model for r in proof.scored_results]
        head_hashes[uid] = proof.weights_hash

    # Protocol v2's copy rule: first-commit seniority and coldkey ownership,
    # folded over two boundary snapshots. Under --perturb the snapshot is fed
    # in reverse, so a state that depended on chain-read order would show.
    # Includes an exact copy under another coldkey, a same-block tie between
    # two coldkeys, and a retirement.
    from fugal_subnet import copy_rule

    def _snapshot(epoch_n):
        out = [
            copy_rule.Commitment(f"hk{uid}", f"coldkey-{uid % 3}", head_hashes[uid], 100 + uid)
            for uid in sorted(head_hashes)
            if not (epoch_n == 2 and uid == 2)
        ]
        out.append(copy_rule.Commitment("hk-copy", "coldkey-x", head_hashes[1], 150))
        out.append(copy_rule.Commitment("hk-tie", "coldkey-y", head_hashes[3], 103))
        if epoch_n == 2:
            out.append(copy_rule.Commitment(
                "hk2", "coldkey-2", hashlib.sha256(b"retrained").hexdigest(), 900))
        return list(reversed(out)) if os.environ.get("FUGAL_PERTURB") else out

    copy_state = copy_rule.CopyState()
    for epoch_n in (1, 2):
        copy_state = copy_rule.observe(copy_state, epoch_n, 1000 * epoch_n,
                                       _snapshot(epoch_n), ghost_retire=24)
    copy_out = {
        "state": copy_state.to_reveal(),
        "classify": copy_rule.classify(copy_state, {
            i: c for i, c in enumerate(sorted(_snapshot(2), key=lambda c: c.hotkey))
        }),
    }

    # Protocol v2's shared grades: one coldkey-median grade per (question,
    # model) cell of the scored slice. Fed in REVERSE uid order, the opposite
    # of the frame below, so a result that depended on proof order would show.
    from fugal_subnet.shared_grades import (
        ScoredAnswer,
        cells_for_reveal,
        shared_cell_grades,
    )
    shared = cells_for_reveal(shared_cell_grades(
        ScoredAnswer(
            question_id=r.question_id, model=r.routed_model,
            coldkey=f"coldkey-{uid % 3}", correct=bool(r.correct), failed=bool(r.failed),
        )
        for uid in sorted(proof_objs, reverse=True)
        for r in proof_objs[uid].scored_results
    ))

    # The frame pools every verified proof's exploration samples, one vote per
    # coldkey and one sample per (question, model), exactly as the validator
    # does. Every synthetic miner answers the same exploration questions, so
    # this stage exercises the aggregation and not just the sums. Iterated in
    # sorted uid order so order-independence holds by construction as well as
    # by arithmetic.
    frame = accumulate_exploration(ReferenceFrame(), aggregate_observations([
        Observation(
            question_id=r.question_id, model=r.routed_model, correct=r.correct,
            prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
            failed=r.failed, coldkey=f"coldkey-{uid}",
        )
        for uid in sorted(proof_objs)
        for r in proof_objs[uid].exploration_results
    ]))
    ref_model, acc_best = best_model(frame, prices)
    from fugal_subnet.frontier import build_frontier
    from fugal_subnet.pricing import question_input_tokens
    frontier = build_frontier(
        frame, prices, sum(question_input_tokens(q.get("prompt", "")) for q in questions),
        len(questions), 300.0,
    )

    for uid in sorted(proof_objs):
        proof = proof_objs[uid]
        scored = proof.scored_results
        ref = reference_cost(
            frame, prices, ref_model,
            prompt_tokens=sum(r.prompt_tokens for r in scored),
            n_questions=len(scored), default_completion_tokens=300.0,
        )
        epoch_scores[uid] = HeadScore(
            accuracy=proof.accuracy, cost_efficiency=0.0, kl_score=0.0,
            routing_decisions=np.array([], dtype=np.int32),
            correct_mask=np.array([r.correct for r in scored], dtype=bool),
            n_correct=proof.n_correct, n_scored=proof.n_total,
            total_head_cost=proof.scored_cost_usd, total_oracle_cost=ref,
        )

    state = update_scores(
        ScoringState(), epoch_scores, head_hashes, frontier=frontier,
        hotkeys={uid: f"hk{uid}" for uid in epoch_scores},
        n_questions=len(questions), pool_size=len(pool),
    )
    uids, weight_values = compute_weights(
        state.records, paid_fraction=frontier.confidence, burn_uid=0,
    )

    scores_out = {
        uid: {
            "composite": float(rec.composite_score),
            "quality": float(rec.quality),
            "thrift": float(rec.thrift),
            "wilson_lcb": float(rec.wilson_lcb),
        }
        for uid, rec in state.records.items()
    }

    # Protocol v2: the whole v2 epoch (shared cells and tokens, the window, the
    # seat and the 85/10/5 vector) on the same verified proofs, fed in
    # reverse uid order. Burn-in 0 so the seat is exercised on one epoch.
    from fugal_subnet.champion import SeatState, Window
    from fugal_subnet.v2_epoch import run_v2_epoch, seat_params_from_config
    v2_uids = sorted(proof_objs, reverse=True)
    from fugal_subnet.copy_rule import Commitment, CopyState
    v2 = run_v2_epoch(
        epoch_index=1, epoch_id=epoch_id_for_block(1), boundary_block=300, blocks_per_epoch=300,
        questions=questions, pool_models=sorted(prices), prices=prices,
        verified_proofs={u: proof_objs[u] for u in v2_uids},
        head_hashes={u: head_hashes[u] for u in v2_uids},
        coldkey_of={u: f"coldkey-{u % 3}" for u in v2_uids},
        hotkey_of={u: f"hk{u}" for u in v2_uids},
        records=state.records,
        commitments=[Commitment(f"hk{u}", f"coldkey-{u % 3}", head_hashes[u], u + 1)
                     for u in sorted(v2_uids)],
        window=Window.empty(48), seat=SeatState(), copy_state=CopyState(), burn_uid=0,
        params=seat_params_from_config(), shares=(0.85, 0.10, 0.05),
        burn_in_questions=0,
    )
    v2_out = {"uids": v2.uids, "weights": v2.weights, "reveal": v2.reveal}

    # Protocol v2 question selection (fugal_subnet/epoch_plan): the scored
    # slice by category and the exploration plan on a synthetic pool of the v2
    # benchmarks, so the miner and every validator select the same v2 questions.
    from fugal_subnet.config import BENCHMARK_CATEGORY_V2
    from fugal_subnet.epoch_plan import exploration_plan, scored_slice
    pool_v2 = [{"question_id": f"{b}-{i:04d}", "benchmark": b, "prompt": f"q {b} {i}"}
               for b in sorted(BENCHMARK_CATEGORY_V2) for i in range(300)]
    plan_nonce = hashlib.sha256(f"v2-plan-{seed}".encode()).digest()
    v2_slice = [q["question_id"] for q in scored_slice(2, plan_nonce, pool_v2, 300)]
    v2_plan = exploration_plan(2, plan_nonce, pool_v2, set(v2_slice),
                               [f"vendor/model-{i}" for i in range(19)], 15)

    return {
        "slice": _digest(question_ids),
        "epoch_plan_v2": _digest({"slice": v2_slice, "plan": sorted(v2_plan.items())}),
        "exploration": _digest(explore_map),
        "synthetic_hidden": _digest(hidden),
        "routing_decisions": _digest(routing),
        "proof": _digest(proofs),
        "verification": _digest(verdicts),
        "copy_rule": _digest(copy_out),
        "shared_grades": _digest(shared),
        "frame": _digest(frame.to_dict()),
        "scores": _digest(scores_out),
        "weights": _digest({"uids": uids, "weights": weight_values}),
        "v2_epoch": _digest(v2_out),
        "seat_confirmation": _digest(_seat_confirmation(seed)),
        **{stage: _digest(value) for stage, value in _protocol_v3(seed).items()},
        "menu_derivation": _digest(_menu_derivation(seed)),
        "seat_v4": _digest(_seat_v4(seed)),
        "explore_shares_v4": _digest(_explore_shares_v4(seed)),
        **{stage: _digest(value) for stage, value in _menu_inputs_v4(seed).items()},
        "menu_publication": _digest(_menu_publication(seed)),
        "head_odds_v4": _digest(_head_odds_v4(seed)),
        "pool_mix_v4": _digest(_pool_mix_v4(seed)),
        "bound_v4": _digest(_bound_v4(seed)),
        "pick_cost_v4": _digest(_pick_cost_v4(seed)),
        "picks_v4": _digest(_picks_v4(seed)),
        "memo_v4": _digest(_memo_v4(seed)),
        "v4_replay": _digest(_v4_replay(seed)),
        "published_v4": _digest(_published_v4(seed)),
        **{stage: _digest(value) for stage, value in _exam_v5(seed).items()},
        "seat_v5": _digest(_seat_v5(seed)),
        "v5_epoch": _digest(_v5_epoch(seed)),
    }


def _v5_epoch(seed: int) -> list:
    """Protocol v5's epoch end to end (fugal_subnet/v5_epoch.py) over six
    synthetic epochs: specs built by exam_v5 for three heads, votes cast by
    the assigned hotkeys (one operator lying), coverage votes, the seat, the
    weights, the reveal, and the rebuild of the last epoch's state from the
    reveals. Under --perturb the served heads, the votes and the window are
    fed in reverse. Digested per epoch: the reveal fields and the weights."""
    import hashlib as _h

    from fugal_subnet import exam_v5, seat_v5, v5_epoch
    from fugal_subnet.exam_v5 import Vote

    perturb = bool(os.environ.get("FUGAL_PERTURB"))

    def fed(xs):
        return list(reversed(list(xs))) if perturb else list(xs)

    entries = ("m/base", "m/cheap", "m/good", "m/best")
    acc = {"m/base": 50, "m/cheap": 60, "m/good": 90, "m/best": 97}
    cost = {"m/base": 1500, "m/cheap": 500, "m/good": 2000, "m/best": 2100}
    heads = {"a": ("ck-a", 1, 10, "m/good"), "b": ("ck-b", 2, 20, "m/cheap"),
             "c": ("ck-c", 3, 30, "m/best")}
    voters = [(f"hk-v{i}", f"ck-v{i}") for i in range(5)]

    def right(q, e):
        d = _h.sha256(f"{seed}|{q}|{e}".encode()).digest()
        return int.from_bytes(d[:2], "big") % 100 < acc[e]

    state, book, records, reveals, out = seat_v5.SeatStateV5(), {}, [], {}, []
    for i in range(6):
        epoch = 100 + i
        names = ["a", "b"] + (["c"] if i >= 3 else [])
        ring = voters + [(f"hk-{n}", heads[n][0]) for n in names]
        hk_ck = dict(ring)
        pool = [f"q{epoch}-{j}" for j in range(80)]
        spec = exam_v5.build_spec(epoch, bytes([seed % 256, epoch % 256]),
                                  {n: [heads[n][3]] * len(pool) for n in names}, entries, pool,
                                  ring, "m/base")
        votes = []
        for (q, e), hks in spec.assignment.items():
            for hk in hks:
                y = 10000 if right(q, e) else 0
                if hk_ck[hk] == "ck-v0":
                    y = 10000 - y
                votes.append(Vote(epoch, q, e, hk, hk_ck[hk], y, cost[e], False))
        cover = [Vote(epoch, f"c{epoch}-{k}-{e}", e, hk, ck,
                      10000 if right(f"c{epoch}-{k}", e) else 0, cost[e], False)
                 for e in entries for k in range(10) for hk, ck in voters[1:3]]
        votes = fed(votes)
        rec = v5_epoch.EpochRecord(epoch, spec, dict(cost), tuple(votes), tuple(fed(cover)),
                                   frozenset(v5_epoch.implausible(votes)))
        served = fed([v5_epoch.Served(n, heads[n][1], f"hk-{n}", heads[n][0], heads[n][2])
                      for n in names])
        old_n, old_n0, old_b = seat_v5.N_MIN, seat_v5.N0, v5_epoch.BASELINE_MIN_PAIRS
        seat_v5.N_MIN = seat_v5.N0 = 100
        v5_epoch.BASELINE_MIN_PAIRS = 10
        try:
            res = v5_epoch.score_epoch(epoch=epoch, state=state, heads=book, record=rec,
                                       window=fed(records), served=served,
                                       full_entries=entries, stratum_of=lambda q: q[-1],
                                       burn_uid=0)
        finally:
            seat_v5.N_MIN, seat_v5.N0, v5_epoch.BASELINE_MIN_PAIRS = old_n, old_n0, old_b
        state, book = res.decision.state, res.heads
        records.append(rec)
        reveals[epoch] = res.reveal
        reveals[epoch - v5_epoch.SPEC_LAG] = {**reveals.get(epoch - v5_epoch.SPEC_LAG, {}),
                                              **v5_epoch.spec_reveal(spec, cost)}
        out.append({"epoch": epoch, "reveal": res.reveal,
                    "weights": [list(res.uids), [repr(w) for w in res.weights]]})
    seat, rebuilt, recs = v5_epoch.rebuild(reveals, 105)
    out.append({"rebuilt": [seat.to_reveal(), {h: f.to_reveal() for h, f in rebuilt.items()},
                            [r.to_reveal() for r in recs]]})
    return out


def _seat_v5(seed: int) -> list:
    """Protocol v5's seat (fugal_subnet/seat_v5.py) over twelve epochs of its
    synthetic world (fugal_subnet/attacks/seat_v5_world.py): every head below
    the best single model at first (empty, all burned), a fill the first
    epoch a head beats it, a same-coldkey upgrade, a near-copy farm held off
    by the margin, a better head taking the seat, and a holder that stops
    standing. Under --perturb the contenders and the item strata are fed in
    reverse. Digested per epoch: the decision, the evidence and the weights."""
    from fugal_subnet import seat_v5
    from fugal_subnet.attacks.seat_v5_world import Head, World

    perturb = bool(os.environ.get("FUGAL_PERTURB"))

    def fed(items):
        return list(reversed(items)) if perturb else list(items)

    w = World(questions=3000, baseline_accuracy=0.84, seed=seed)
    plan = {1: [("low1", "ck-l1", 10, 0.80, 0.2, 0), ("low2", "ck-l2", 11, 0.82, 0.2, 1)],
            3: [("good", "ck-g", 20, 0.92, 0.0, 2)],
            5: [("good2", "ck-g", 25, 0.93, 0.0, 3)],
            6: [(f"farm{i}", f"ck-f{i}", 30 + i, 0.93, 0.0, 10 + i) for i in range(4)],
            8: [("best", "ck-b", 40, 0.99, 0.2, 4)]}
    state, out, events = seat_v5.SeatStateV5(), [], []
    for e in range(1, 13):
        for name, ck, block, acc, noise, sd in plan.get(e, []):
            w.add(Head(name, ck, block, acc, noise=noise, seed=sd))
        if e >= 11:
            w.heads[next(h for h in w.heads if w.heads[h].name == "best")].standing = False
        heads = fed([h.contender() for h in w.heads.values()])

        def items(x, y):
            return dict(fed(w.items(x, y).items()))

        def base(x):
            return dict(fed(w.baseline_items(x).items()))

        d = seat_v5.decide(state, e, heads, items, base)
        state = d.state
        uids, weights = seat_v5.weights(
            d, {h: i + 1 for i, h in enumerate(sorted(w.heads))}, burn_uid=0)
        names = {h: w.heads[h].name for h in w.heads}
        events.append((d.event, names.get(d.holder)))
        out.append({"event": d.event, "state": state.to_reveal(), "payable": list(d.payable),
                    "evidence": d.evidence,
                    "weights": [list(map(int, uids)), [repr(float(x)) for x in weights]]})
    want = [("empty", None), ("empty", None), ("fill", "good"), ("hold", "good"),
            ("upgrade", "good2"), ("hold", "good2"), ("hold", "good2"), ("take", "best"),
            ("hold", "best"), ("hold", "best")]
    if events[:10] != want or events[10][1] == "best":
        raise SystemExit(f"the seat_v5 stage no longer burns, fills, upgrades, holds off a "
                         f"farm, takes and lapses: {events}")
    return out


def _seat_confirmation(seed: int) -> list:
    """The seat over ten epochs with fresh-data confirmation in force: a
    challenger passes, waits as a pending record, and is confirmed on the
    questions drawn since (its own resampling stream), with candidates fed in
    reverse order. Every published seat and every decision line is digested.
    """
    import dataclasses

    import numpy as np

    from fugal_subnet.benchmarks.slicer import epoch_id_for_block
    from fugal_subnet.champion import (
        Candidate,
        EpochSlice,
        HeadEpoch,
        SeatState,
        Window,
        seat_step,
    )
    from fugal_subnet.v2_epoch import seat_params_from_config, seat_to_dict

    q, cats = 300, ["knowledge"] * 120 + ["math"] * 75 + ["hard"] * 90 + ["instruction"] * 15
    params = dataclasses.replace(seat_params_from_config(1, 0), min_paired_questions=300,
                                 confirm_questions=600, confirm_open_epochs=2)
    heads = {"champion": 1, "challenger": 2, "equal": 1}      # the model each always routes to
    rng = np.random.default_rng(seed)
    window, state, out = Window.empty(6), SeatState(champion="champion", champion_since=0), []
    for epoch in range(1, 11):
        difficulty = rng.random(q)
        grades = (difficulty[:, None] < np.array([0.55, 0.80, 0.84])[None, :]).astype(np.float64)
        sl = EpochSlice(epoch=epoch, question_ids=tuple(f"e{epoch}q{i}" for i in range(q)),
                        categories=tuple(cats), hard=tuple(c == "hard" for c in cats),
                        grades=grades, spends=np.tile(np.array([0.0001, 0.004, 0.005]), (q, 1)))
        window = window.add_epoch(sl, {h: HeadEpoch(np.full(q, m, dtype=np.int64),
                                                    np.zeros(q, dtype=bool))
                                       for h, m in heads.items()})
        cands = {h: Candidate(h, (i,), True, 1) for i, h in reversed(list(enumerate(sorted(heads))))}
        decision = seat_step(state, epoch, epoch_id_for_block(epoch), window, cands, params=params)
        state = decision.state
        out.append({"event": decision.event, "seat": seat_to_dict(state),
                    "detail": [list(d) for d in decision.detail]})
    return out


def _protocol_v3(seed: int) -> dict:
    """Protocol v3's consensus computations on synthetic inputs: each miner's
    private exploration set, exact odds, the pooled pairs with the reference,
    and the seat decided on odds over several epochs. Inputs are fed in a
    seed-shuffled order wherever order must not matter.
    """
    import dataclasses
    import random
    from fractions import Fraction

    import numpy as np

    from fugal_subnet import config, epoch_plan, explore_weights, head_odds, odds_score
    from fugal_subnet.champion import (
        Candidate,
        EpochPairs,
        EpochSlice,
        HeadEpoch,
        OddsParams,
        SeatState,
        Window,
        seat_step,
    )
    from fugal_subnet.explore_pool import ExploreAnswer, pool_pairs, private_implausible
    from fugal_subnet.v2_epoch import odds_b64, seat_params_from_config, seat_to_dict

    rnd = random.Random(seed)
    nonce = hashlib.sha256(f"v3-{seed}".encode()).digest()
    models = [f"vendor/model-{i:02d}@high" for i in range(9)]

    # Private exploration: the plan of several hotkeys on a synthetic v3 pool.
    pool = [{"question_id": f"{b}-{i:04d}", "benchmark": b, "prompt": f"q {b} {i}"}
            for b in sorted(config.BENCHMARK_CATEGORY_V3) for i in range(120)]
    rnd.shuffle(pool)
    scored = {q["question_id"] for q in epoch_plan.scored_slice(3, nonce, pool, 300)}
    # Cost-aware exploration: shares from exact costs, the exponent the spend
    # cap selects, and the weights every pooled pair carries.
    costs = {m: Fraction(3 ** i, 10_000) for i, m in enumerate(models)}
    halves = explore_weights.choose_exponent(costs, n_calls=32, cap_usd=Fraction(1, 2),
                                             floor=Fraction(1, 4), max_halves=4)
    shares = explore_weights.shares(costs, exponent_halves=halves, floor=Fraction(1, 4))
    weights = explore_weights.weights(shares)
    weights_out = {"halves": halves, "shares": sorted(shares.items()),
                   "weights": sorted(weights.items()),
                   "spend": str(explore_weights.expected_spend(shares, costs, 32))}
    plans = {hk: epoch_plan.exploration_plan(3, nonce, pool, scored, models, 0, hotkey=hk,
                                             shares=shares)
             for hk in ("5Fhotkey-a", "5Fhotkey-b", "5Fhotkey-c")}
    private_out = {hk: sorted(p.items()) for hk, p in sorted(plans.items())}

    # Exact odds: heads with large, cancelling weights on unit embeddings.
    rng = np.random.default_rng(seed)
    dim = 1024
    heads = {}
    for name in ("alpha", "beta", "gamma"):
        W = (rng.standard_normal((len(models), dim)) * rng.choice([0.5, 30.0, 4000.0])).astype(np.float32)
        b = rng.standard_normal(len(models)).astype(np.float32)
        heads[name] = (W, b, models if name != "gamma" else models[:4])
    hidden = rng.standard_normal((400, dim)).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    pair_models = [models[int(i)] for i in rng.integers(0, len(models), 400)]
    odds_out = {name: odds_b64(head_odds.head_odds(W, b, list(m), list(hidden), pair_models))
                for name, (W, b, m) in sorted(heads.items())}

    # The pool, the reference and the seat on odds, over twelve epochs: a pass,
    # a pending record, and the confirmation on fresh pairs.
    q, cats = 300, ["knowledge"] * 120 + ["math"] * 75 + ["hard"] * 90 + ["instruction"] * 15
    params = dataclasses.replace(
        seat_params_from_config(1, 0, protocol_version=3), min_paired_questions=300,
        resamples=400, confirm_questions=600, confirm_open_epochs=2,
        odds=OddsParams(margin_points=0.1, min_pairs=600, nv_noninferiority=0.5, ref_prior=20,
                        confirm_pairs=600))
    acc = [0.55, 0.7, 0.85, 0.6, 0.9, 0.5, 0.75, 0.8, 0.65]
    benches = sorted(set(cats))
    window, state = Window.empty(6), SeatState(champion="blurry", champion_since=0)
    pool_out, seat_out = [], []
    for epoch in range(1, 13):
        difficulty = rng.random(q)
        grades = (difficulty[:, None] < np.array(acc[:3])[None, :]).astype(np.float64)
        sl = EpochSlice(epoch=epoch, question_ids=tuple(f"e{epoch}q{i}" for i in range(q)),
                        categories=tuple(cats), hard=tuple(c == "hard" for c in cats),
                        grades=grades, spends=np.tile(np.array([0.0001, 0.004, 0.005]), (q, 1)))
        answers = []
        for ck in range(6):
            for i in range(60):
                qid, m = f"x{epoch}_{ck}_{i}", int(rng.integers(0, len(models)))
                d = float(rng.random())
                answers.append((ExploreAnswer(qid, models[m], f"ck{ck % 4}", d < acc[m],
                                              failed=(i % 17 == 0), private=True,
                                              prompt_tokens=30, completion_tokens=200 + m),
                                d, benches[i % len(benches)]))
        rnd.shuffle(answers)
        flagged = private_implausible([a for a, _, _ in answers],
                                      {(a.question_id, a.model): 6000 for a, _, _ in answers},
                                      z=4, min_samples=8)
        pooled = pool_pairs([a for a, _, _ in answers], nonce=nonce, coldkey_cap=36, flagged=flagged,
                            weights=weights)
        meta = {(a.question_id, a.model): (d, b) for a, d, b in answers}
        pairs = EpochPairs(
            question_ids=tuple(p.question_id for p in pooled), models=tuple(p.model for p in pooled),
            benchmarks=tuple(meta[(p.question_id, p.model)][1] for p in pooled),
            categories=tuple(meta[(p.question_id, p.model)][1] for p in pooled),
            y=np.array([p.y for p in pooled], dtype=np.int32),
            prompt_tokens=np.array([p.prompt_tokens for p in pooled], dtype=np.int32),
            completion_tokens=np.array([p.completion_tokens for p in pooled], dtype=np.int32),
            weight=np.array([p.weight for p in pooled], dtype=np.int64),
            # v3 menu B's grace (EpochPairs.scored): from epoch 7 the last
            # model is an added entry, unscored for three epochs.
            scored=(np.array([p.model != models[-1] for p in pooled], dtype=bool)
                    if 7 <= epoch < 10 else None))
        d_of =np.array([meta[(p.question_id, p.model)][0] for p in pooled])
        a_of = np.array([acc[models.index(p.model)] for p in pooled])
        predict = {noise_name: np.rint(10_000 / (1 + np.exp(-(a_of - d_of - shift) / width))
                                       ).astype(np.int16)
                   for noise_name, width, shift in (("sharp", 0.03, 0.0), ("blurry", 0.4, 0.1))}
        window = window.add_epoch(
            sl, {h: HeadEpoch(np.full(q, 2, dtype=np.int64), np.zeros(q, dtype=bool)) for h in predict},
            pairs, predict)
        cands = {h: Candidate(h, (i,), True, 1) for i, h in reversed(list(enumerate(sorted(predict))))}
        decision = seat_step(state, epoch, f"e{epoch:08d}", window, cands, params=params)
        state = decision.state
        ref = odds_score.reference(window.pairs, 20)
        pool_out.append({"pairs": [dataclasses.astuple(p) for p in pooled], "flagged": sorted(flagged),
                         "reference": ref.of(epoch).tolist()})
        seat_out.append({"event": decision.event, "seat": seat_to_dict(state),
                         "detail": [list(x) for x in decision.detail]})
    return {"exploration_private": private_out, "head_odds": odds_out,
            "explore_pool": pool_out, "odds_seat": seat_out, "explore_weights": weights_out}


def _menu_derivation(seed: int) -> dict:
    """Protocol v4's menu as published state (fugal_subnet/menu_state.py): the
    pinned seed, then ten epochs on synthetic inputs that touch every part of
    the lifecycle: each epoch's derivation step (the first signed list, then
    a newer one without a lab, which the step applies the compiled rules,
    menu_rules, to; voted prices and answer limits; token means; a failing
    entry and one whose hosts stay unlisted; the gates deciding provisional
    entries, which become full or fail a gate; a full menu that gives up its
    least-used entry), then the root's join step at the boundary
    (chain_proposals: proposals committed on chain, admitted in the order
    the block hash gives, the delta checked as every node checks it, and
    applied as provisional entries). Every step's base and joined state
    bytes, hash, proof namespace, decisions and verdicts are digested, with
    the exact cost of every entry, and every derived state must pass the
    compiled rules.

    Under --perturb every input collection is fed in reverse and the process
    has another hash seed, so a step that read arrival order or dict order
    would show here.
    """
    import dataclasses

    from fugal_subnet import chain_proposals as cp
    from fugal_subnet import menu_rules as mr
    from fugal_subnet import menu_state as ms
    from fugal_subnet import routing_protocol

    def draw(*parts) -> int:
        """A fixed pseudo-random integer for these parts: sha256, no RNG state."""
        text = "|".join(str(p) for p in (seed,) + parts)
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")

    def fed(items):
        items = list(items)
        return tuple(reversed(items)) if os.environ.get("FUGAL_PERTURB") else tuple(items)

    categories = ("code", "easy_math", "everyday", "hard", "instruction", "knowledge", "math")
    mix = tuple(zip(categories, (15, 5, 15, 20, 15, 20, 10)))
    seed_state = ms.load_seed()
    # Ten steps derive epochs 991..1000: epoch 1000 is a refresh epoch
    # (MENU_REFRESH_EPOCHS), the others refresh only with a structural change.
    state = ms.seed_state(990)
    # A cap two above the seed: two joins fill it, and from then on a full
    # menu gives up its least-used entry to keep a place for a join.
    params = ms.MenuParams(mix=mix, min_category_accuracy_bp=5000,
                           menu_cap=len(state.entries) + 2)
    unlisted_entry, failing_entry = state.entries[11].id, state.entries[23].id

    def signed_list(serial: int, without_lab: str | None = None) -> mr.MenuAllowlist:
        """A lab and host list as verify_signed_list returns one, covering the
        seed's entries (as scripts/signed_list.py derives a draft) and the
        lab and host the proposals below name. No key is involved: the step
        reads a list's content and hash, and the signature check has its own
        tests."""
        labs, hosts, first_party = {"lab-new"}, {"host-new"}, {}
        for e in seed_state.entries:
            lab = e.model.split("/", 1)[0]
            labs.add(lab)
            for tag in e.provider_order:
                hosts.add(mr.host_identity(tag))
                if e.quantization == mr.UNKNOWN_QUANTIZATION and mr.host_identity(tag) != lab:
                    first_party.setdefault(lab, set()).add(mr.host_identity(tag))
        labs.discard(without_lab)
        first_party.pop(without_lab, None)
        raw = mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, serial, labs=fed(sorted(labs)),
                                      hosts=fed(sorted(hosts)),
                                      first_party={lab: fed(sorted(first_party[lab]))
                                                   for lab in fed(sorted(first_party))})
        return mr.MenuAllowlist(serial=serial, labs=frozenset(labs), hosts=frozenset(hosts),
                                first_party=mr.first_party_map(first_party),
                                sha256=hashlib.sha256(raw).hexdigest(), signer="SHA256:stage")

    lists = {1: signed_list(1), 2: signed_list(2, without_lab="ibm-granite")}

    # The proposals on chain: each miner's commitment stays until it changes,
    # so a proposal that finds no place is considered again next epoch. Two
    # of one coldkey (one waits), one entry proposed by two miners, a lab the
    # list does not name and a host it does not name.
    def proposal(i: int, hotkey: str, coldkey: str, *, lab="lab-new", host="host-new/fp8"):
        raw = cp.document_bytes(cp.ProposalDoc(
            proposer=hotkey, model=f"{lab}/model-{i}", reasoning=("effort", "high"),
            provider_order=(host,), quantization="fp8"))
        listing = (cp.HostListing(host, True, 90_000 + 1000 * i,
                                  400_000 + draw("file", i) % 9999, 200_000, True),)
        return hotkey, coldkey, raw, listing

    filed = [proposal(i, f"5Fproposer{i}", f"5Fcoldkey{i % 3}") for i in range(5)] + [
        proposal(1, "5Fsecond-proposer", "5Fcoldkey9"),
        proposal(8, "5Fproposer8", "5Fcoldkey8", lab="lab-unlisted"),
        proposal(9, "5Fproposer9", "5Fcoldkey7", host="host-unlisted/fp8")]
    serving = frozenset([hk for hk, *_ in filed] + [f"5Fserving{i}" for i in range(6)])

    steps = []
    joined_order: list[str] = []
    for n in range(10):
        k = state.epoch - 1
        full = [e for e in state.entries if not e.provisional]
        reads = []
        for model, hosts in sorted({e.model: e.provider_order for e in state.entries}.items()):
            for host in hosts:
                listed = draw("listed", model, host) % 11 != 0
                reads.append(ms.HostRead(
                    model, host, voters=2 + draw("voters", k, model, host) % 4, listed=listed,
                    price_in=49_980 + draw("in", k // 3, model, host) % 9_000_000,
                    price_out=99_960 + draw("out", k // 3, model, host) % 40_000_000,
                    limit=(0, 65_536, 128_000, 393_216)[draw("limit", model, host) % 4]))
        token_stats = [
            ms.TokenStat(e.id, e.rev, tuple(
                ms.TokenCell(c, pairs, pairs * (150 + draw("tin", e.id, c) % 400) + draw("r", k, e.id, c) % 97,
                             pairs * (60 + draw("tout", e.id, c) % 9000) + draw("s", k, e.id, c) % 89)
                for c in categories for pairs in [draw("pairs", k, e.id, c) % 12]))
            for e in full]
        health = [ms.HealthStat(unlisted_entry, 1, unlisted_epochs=n + 1),
                  ms.HealthStat(failing_entry, 1, eligible=4 if n >= 4 else 3, failing=3)]
        health += [ms.HealthStat(e.id, e.rev, eligible=5, failing=draw("fail", k, e.id) % 3)
                   for e in full if e.id not in (unlisted_entry, failing_entry)]
        accuracy = [ms.AccuracyStat(e.id, e.rev, 40 + draw("apairs", e.id) % 200,
                                    500_000 + draw("rate", e.id) % 450_000,
                                    300_000 + draw("lower", e.id) % 400_000) for e in full]
        use = [ms.EntryUse(e.id, draw("use", e.id) % 40) for e in state.entries]
        # The gates' reads of each provisional entry that joined by k: the
        # second entry to join has no category at the floor, the third's
        # calls mostly fail for the median coldkey, the others pass.
        gates = []
        for e in state.entries:
            if not e.provisional or e.joined_epoch > k:
                continue
            weak, dead = joined_order.index(e.id) == 1, joined_order.index(e.id) == 2
            tallies = tuple(ms.CallTally(f"5Fvoter{j}", 40, 30 if dead else draw("f", e.id, j) % 3)
                            for j in range(4))
            cells = tuple(
                ms.MeasureCell(c, calls, calls, calls * (1 if weak else 3) // 4,
                               calls * (200 + j), calls * (900 + draw("mo", e.id) % 500))
                for j, (c, calls) in enumerate(zip(categories, (22, 8, 23, 30, 22, 30, 15))))
            gates.append(ms.GateStat(e.id, e.rev, tallies, cells,
                                     upper_ppm=700_000 + draw("upper", e.id) % 300_000))
        # The seed names no list: the first step adopts list 1. From the third
        # step the node holds list 2, which no longer names one lab; the step
        # works out which entries that retires (or holds the list back while
        # it would leave too few).
        lab_list = lists[1] if n < 2 else lists[2]
        # A device list as verify_signed_list returns one, adopted in steps 1 and 3.
        device_list = None
        if n in (1, 3):
            devices = fed(sorted(hashlib.sha256(f"device-{n}-{i}".encode()).hexdigest()
                                 for i in range(4)))
            raw_devices = mr.canonical_list_bytes(mr.KIND_DEVICE_LIST, n, devices=devices)
            device_list = mr.DeviceList(serial=n, devices=frozenset(devices),
                                        sha256=hashlib.sha256(raw_devices).hexdigest(),
                                        signer="SHA256:stage")
        inputs = ms.StepInputs(
            epoch=k, nonce=hashlib.sha256(f"menu-nonce-{seed}-{k}".encode()).digest(),
            lab_list=lab_list,
            lab_list_in_force=lists.get(state.labs.serial) if state.labs.serial else None,
            device_list=device_list,
            host_reads=fed(reads), token_stats=fed(token_stats), health=fed(health),
            explore_counts=fed(ms.ExploreCount(e.id, e.rev, draw("explore", k, e.id) % 300)
                               for e in state.entries),
            accuracy=fed(accuracy), use=fed(use), use_total=sum(u.picks for u in use) + 1,
            gates=fed(gates))
        previous = state
        result = ms.step_detail(state, inputs, params)
        base = result.state
        raw = ms.canonical_json(base)
        # Every node holding the previous menu reads the derived one back from
        # its bytes: the sequence, the caps, the transition and the rules. (The
        # cap here starts below the seed's size, which no release does; the
        # reader is held to the larger of the two.)
        reader = dataclasses.replace(params, menu_cap=max(params.menu_cap, len(base.entries)))
        allow = lists[base.labs.serial]
        if ms.from_json(raw) != base or ms.menu_for_epoch(
                previous, ms.menu_hash(base), raw, allowlist=allow,
                device_list=device_list, params=reader) != base:
            raise SystemExit("a derived menu does not read back from its published bytes")
        # The join step at the boundary of base.epoch: the root's pinned read,
        # its admission, and the delta every node checks before applying it.
        block_hash = hashlib.sha256(f"block-{seed}-{base.epoch}".encode()).hexdigest()
        read = cp.ChainRead(epoch=base.epoch, block=base.epoch * 360 - 30, block_hash=block_hash,
                            proposals=fed(sorted((hk, cp.proposal_hash(r)) for hk, _c, r, _l in filed)),
                            coldkey_of={hk: ck for hk, ck, _r, _l in filed}, serving=serving,
                            reveal=None)
        delta = cp.admit(base, read, {cp.proposal_hash(r): r for _h, _c, r, _l in fed(filed)},
                         {cp.proposal_hash(r): li for _h, _c, r, li in fed(filed)}, allow, params)
        delta_raw = cp.delta_bytes(delta)
        joins = cp.check_delta(base, delta_raw, allow, block=read.block, params=params)
        state = ms.apply_joins(base, joins, params)
        joined_order += [cp.parse_document(c.document).id for c in delta.joined()]
        steps.append({
            "epoch": state.epoch, "serial": state.serial, "base": ms.menu_hash(base),
            "hash": ms.menu_hash(state),
            "namespace": routing_protocol.identity_v4(ms.menu_hash(state)),
            "bytes": hashlib.sha256(ms.canonical_json(state)).hexdigest(),
            "entries": len(state.entries),
            "events": [dataclasses.astuple(e) for e in result.events],
            "delta": cp.delta_hash(delta_raw),
            "verdicts": [(c.hotkey, c.verdict) for c in delta.considered],
            "costs": [(i, str(c)) for i, c in sorted(ms.costs(state).items())],
            "explore_pairs": sorted(state.explore_pairs().items()),
            "due": ms.refresh_due(base.epoch, params),
            "structural": ms.structural_change(previous, base),
            "refreshed": any(e.kind in ("price", "token_means") for e in result.events
                             if not (e.kind == "token_means" and previous.entry(e.subject).provisional)) or any(
                previous.entry(e.id).explore_pairs != e.explore_pairs for e in base.entries
                if e.id in previous.ids() and previous.entry(e.id).rev == e.rev
                and not previous.entry(e.id).provisional),
        })
    kinds = {e[0] for s in steps for e in s["events"]}
    reasons = {e[2] for s in steps for e in s["events"] if e[0] in ("retired", "gate_failed")}
    verdicts = {v for s in steps for _h, v in s["verdicts"]}
    needed = {"lab_list", "device_list", "price", "token_means", "retired", "gate_failed",
              "full"}
    if not needed <= kinds or not {"lab_removed", "gone", "failing", "displaced",
                                   "gate_category", "gate_answered"} <= reasons:
        raise SystemExit(f"the menu_derivation stage no longer exercises "
                         f"{sorted(needed - kinds)} / its retirement reasons {sorted(reasons)}")
    if not {cp.JOINED, cp.COLDKEY_ONCE, cp.NO_PLACE, cp.REFUSED + "already_on_menu",
            cp.REFUSED + "lab_not_allowed"} <= verdicts or not any(
            v.startswith(cp.REFUSED) and "host" in v for v in verdicts):
        raise SystemExit(f"the menu_derivation stage no longer exercises the join verdicts "
                         f"{sorted(verdicts)}")
    if (any(s["refreshed"] and not (s["due"] or s["structural"]) for s in steps)
            or not any(s["refreshed"] and s["due"] and not s["structural"] for s in steps)
            or not any(not s["refreshed"] and not s["due"] and not s["structural"] for s in steps)):
        raise SystemExit("the menu_derivation stage no longer exercises the refresh cadence: "
                         f"{[(s['epoch'], s['due'], s['structural'], s['refreshed']) for s in steps]}")
    return {"seed": ms.menu_hash(seed_state), "seed_at_first_epoch": ms.menu_hash(ms.seed_state(1000)),
            "steps": steps, "final": ms.canonical_json(state).decode("ascii")}


def _seat_v4(seed: int) -> dict:
    """Protocol v4's labels and odds (docs/decisions/SEAT_MINIMAL_V4.md 3.2
    and 4; fugal_subnet/labels_v4.py) over twenty epochs of the live epoch,
    run_v2_epoch, on a changing menu: an entry on it from the start, one
    that joins (provisional for two epochs, then full: FAST_CROWN_V4.md 18)
    with heads with and without its row, one that joins and leaves inside
    the run; heads that miss epochs and get their odds
    for them when next verified (late odds); one that never answers the
    scored questions. Digested per epoch: the labels (voters, the odds a
    missing row is scored at), every head's odds, the late odds, the labels
    digest and each head's standing over R, and the seat (seat_v4) as the
    live epoch decides and publishes it: its reveal block, its evidence, the
    seat state and the weights, the heads answering the scored questions
    (picks_v4) every epoch but one that never does; at the end, the integer
    sums (labels_v4.gain_strata) of every ordered pair of heads and of each
    head over R.

    Then the seat step itself (seat_v4.seat_step) over twenty epochs of the
    seat's synthetic world (fugal_subnet/attacks/seat_v4_world.py), whose
    heads' odds and picks are set so that every clause decides something: an
    empty seat filled at activation (commits before the floor block tie at it
    by the salted digest), the holder adding a row by succession before an
    entry's allowance ends, a head missing an epoch and getting its odds for
    it when next verified, a near-copy of an earlier contender held by rule
    C, a head refused by the memorisation verdict, the holder missing three
    epochs and the seat going at once to the latest qualifier, and the
    former holder paid its history share. Digested per epoch: the event, the
    seat state as published, the evidence, the paid seats and the weights.

    Then the seat reading the memorisation verdict (`_seat_v4_memo`): v4
    epochs through run_v2_epoch with and without the epoch's memo, in which a
    refused holder opens the seat (and, not serving, loses it to an honest
    qualifier), a refused head drops out of rule C's order, and a successor
    is refused from its inherited rows.
    Digested: the champion block, the seat's evidence, v2_state (both
    digests), memo_v4 and the weights.

    Then a succession chain (`_seat_v4_chain`): on the seat's world with the
    live late-odds schedule, a holder whose hotkey commits a successor every
    epoch, past MAX_PREDECESSORS, once serving a predecessor's bytes again,
    and other hotkeys doing the same so that they share the late-odds bound,
    against a better challenger. Digested per epoch: the event, the seat
    state, the evidence and the epochs each head holds odds for.

    Under --perturb the answers, the heads, the menu plan, the commitments,
    the world's heads and every served, answered and refused list (and the
    memorisation part's scenarios, kept heads, proofs and answers) arrive in
    reverse order and the process has another hash seed.
    """
    from fugal_subnet import labels_v4

    run = _seat_v4_run(seed)
    window, out, heads, hash_of, uid_of, _rate, fed = (run.window, run.out, run.heads, run.hash_of,
                                                      run.uid_of, run.rate, run.fed)
    late = [o["late_odds"] for o in out]
    if not (late[8].get(hash_of["no-new"]) and late[12].get(hash_of["base-only"])
            and any(m == -1 for m in out[19]["labels_v4"]["missing_odds"])
            and any(m == labels_v4.PROVISIONAL_LABEL for o in out
                    for m in o["labels_v4"]["missing_odds"])):
        raise SystemExit("the seat_v4 stage no longer exercises late odds, a full entry's "
                         "missing row and a provisional entry")
    ref = labels_v4.reference(window, now=20)
    sides = {n: labels_v4.Side(hash_of[n], f"ck{uid_of[n]}", 1) for n in sorted(heads)}
    sides["R"] = labels_v4.REFERENCE
    sums = {}
    for x in sorted(heads):
        for y in sorted(sides):
            if x != y:
                sums[f"{x}>{y}"] = {c: [s.n, s.total, s.squares] for c, s in sorted(
                    labels_v4.gain_strata(window, ref, sides[x], sides[y], now=20).items())}
    if not any(o["picks_v4"]["items"] for o in out):
        raise SystemExit("the seat_v4 stage no longer has heads with a counted challenge")
    return {"epochs": out, "strata": sums, "seat": _seat_v4_step(fed), "memo": _seat_v4_memo(fed),
            "chain": _seat_v4_chain(fed)}


def _seat_v4_run(seed: int, *, dim: int = 32, spend: bool = False, memo_for=None, keep=None,
                 oracle: bool = False):
    """The twenty live epochs of the seat_v4 stage (its first part): the
    world and the epoch loop, returned with what the stage reads after it.
    The seat_v4 stage runs it as it always has (`dim` 32, no pick-cost
    table, no memo). The v4_replay stage runs it with heads that load as v4
    heads (`dim` 1024), the slice priced at each epoch's pick-cost table
    (`spend`), `memo_for(epoch)` as each epoch's memo, and `keep(epoch,
    kwargs, {uid: head bytes}, result)` called after each epoch; with
    `oracle` an exploration answer on an entry of the head "every-row" is
    right with that row's own probability, so that head beats R, and the
    slice is long enough for rule B's run: the seat has events."""
    import dataclasses
    from types import SimpleNamespace

    import numpy as np

    from fugal_subnet import epoch_plan, seat_v4
    from fugal_subnet import menu_state as ms
    from fugal_subnet.champion import OddsParams, SeatState, Window
    from fugal_subnet.copy_rule import Commitment, CopyState
    from fugal_subnet.explore_pool import ExploreAnswer
    from fugal_subnet.picks_v4 import challenge_picks
    from fugal_subnet.v2_epoch import (
        OddsInput,
        PicksInput,
        _artifact,
        run_v2_epoch,
        seat_params_from_config,
    )

    def fed(items):
        items = list(items)
        return list(reversed(items)) if os.environ.get("FUGAL_PERTURB") else items

    bpe = 360
    base = [f"lab-{c}/model@high" for c in "abcdef"]
    # id -> (joined, leaves): one on the menu from the start, one that joins
    # at 5 (provisional in 5 and 6, full from 7: FAST_CROWN_V4.md 18), one
    # that joins at 8 and leaves at 11.
    plan = {"lab-g/old@high": (0, None), "lab-h/new@low": (5, None),
            "lab-i/brief@high": (8, 11)}
    future = "lab-z/never@high"
    every = sorted(base + list(plan) + [future])
    acc = {m: 0.5 + 0.04 * i for i, m in enumerate(every)}
    rate = {m: 3000 + 500 * i for i, m in enumerate(sorted(plan))}

    def entry(eid, joined, scored_from, measured):
        return ms.Entry(id=eid, rev=1, model=eid.split("@")[0], reasoning=("effort", eid.split("@")[1]),
                        max_tokens=8192, provider_order=("host-a/fp8",), quantization="fp8",
                        price_in=1000, price_out=4000, mean_in=500_000, mean_out=900_000,
                        token_pairs=100, joined_epoch=joined, scored_from=scored_from,
                        measured_rate=measured,
                        proposer="5Fproposer" if scored_from == ms.PROVISIONAL else "")

    def menu_at(epoch):
        entries = [entry(m, 0, 0, 5000) for m in base]
        records = []
        for eid, (joined, leaves) in fed(plan.items()):
            if joined <= epoch and (leaves is None or epoch < leaves):
                full_from = joined + ms.MENU_LAG if joined else 0
                provisional = epoch < full_from
                entries.append(entry(eid, joined, ms.PROVISIONAL if provisional else full_from,
                                     5000 if provisional else rate[eid]))
                if provisional:
                    records.append(ms.Provisional(eid, "ab" * 32, "5Fproposer", f"ck-{eid}"))
        return ms.MenuState(epoch=epoch, serial=epoch, entries=tuple(sorted(entries, key=lambda e: e.id)),
                            retired=(), labs=ms.AllowlistRef(0, ""),
                            devices=ms.AllowlistRef(0, ""),
                            provisional=tuple(sorted(records, key=lambda r: r.id)))

    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((len(every), dim)) * 3.0).astype(np.float32)
    b = rng.standard_normal(len(every)).astype(np.float32)

    def head(rows):
        idx = [every.index(m) for m in rows]
        return W[idx], b[idx], list(rows)

    heads = {
        "every-row": head(base + list(plan)),
        "no-new": head(base + ["lab-g/old@high", "lab-i/brief@high"]),
        "base-only": head(base),
        "extra-row": head(base + list(plan) + [future]),
    }
    hash_of = {n: hashlib.sha256(f"seat-v4-{n}".encode()).hexdigest() for n in heads}

    def head_bytes(name):
        # A head's arrays as the v4 head file a TD serves (head_eval.load_head_v4).
        from fugal_subnet import success_test_fixtures as sfx

        z = dict(sfx.arrays(*heads[name]))
        z["provenance"] = np.array(f"seat-v4:{name}")
        for key in ("mean_in_tokens", "mean_out_tokens", "cost_profile_id"):
            z.pop(key, None)
        buf = io.BytesIO()
        np.savez(buf, **z)
        return buf.getvalue()

    uid_of = {n: uid for uid, n in enumerate(sorted(heads), start=5)}
    missed = {"no-new": {6, 7, 8}, "base-only": {12}}
    never_answers = {"extra-row"}
    hidden = {}

    def coin(tag):
        # A head's answers drawn from its own key, not from the shared
        # stream, so the order heads are fed in draws nothing different.
        return int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big") / 2**64

    def hidden_of(qid):
        if qid not in hidden:
            v = np.random.default_rng(int.from_bytes(hashlib.sha256(qid.encode()).digest()[:8], "big")
                                      ).standard_normal(dim).astype(np.float32)
            hidden[qid] = v / np.linalg.norm(v)
        return hidden[qid]

    params = dataclasses.replace(
        seat_params_from_config(1, 0, protocol_version=3), min_paired_questions=60, resamples=200,
        odds=OddsParams(margin_points=0.1, min_pairs=150, nv_noninferiority=0.5, ref_prior=20,
                        confirm_pairs=150))
    benches = ("mmlu", "math", "aime")
    category = {"mmlu": "knowledge", "math": "math", "aime": "hard"}
    q = 100 if oracle else 30         # with `oracle`, rule B's 500 items in five epochs

    def p_right(m, qid):
        rows = heads["every-row"][2]
        if not oracle or m not in rows:
            return acc[m]
        j = rows.index(m)
        z = float(heads["every-row"][0][j] @ hidden_of(qid)) + float(heads["every-row"][1][j])
        return 1.0 / (1.0 + float(np.exp(-z)))
    window, seat, copy, out = Window.empty(48), SeatState(), CopyState(), []
    for epoch in range(1, 21):
        menu = menu_at(epoch)
        ids = list(menu.ids())
        answers, bench_of = [], {}
        for ck in range(5):
            for i in range(48):
                qid = f"g{epoch}_{ck}_{i}"
                m = ids[int(rng.integers(0, len(ids)))]
                bench_of[qid] = benches[i % 3]
                answers.append(ExploreAnswer(qid, m, f"ck{ck + 4}", float(rng.random()) < p_right(m, qid),
                                             failed=(i % 19 == 0), private=True,
                                             prompt_tokens=30, completion_tokens=150))
        questions = [{"question_id": f"s{epoch}q{i}", "benchmark": benches[i % 3]} for i in range(q)]
        # The epoch's challenge list: the slice; each head but one challenges
        # on all of it, paired with the epoch's holder (seat_v4.challenge_holder),
        # its picks and the holder's those the heads give on this menu
        # (FAST_CROWN_V4.md 2). An epoch whose holder is not verified has no
        # challenge (its bytes are not in this stage's inputs).
        challenge = tuple((x["question_id"], category[x["benchmark"]]) for x in questions)
        hid = np.stack([hidden_of(x["question_id"]) for x in questions])
        name_of = {hash_of[n]: n for n in heads}
        holder = seat_v4.challenge_holder(seat, epoch) if isinstance(seat, seat_v4.SeatStateV4) \
            else None
        holder_name = name_of.get(holder)
        holder_serves = holder is None or epoch not in missed.get(holder_name, ())
        hps = ([None] * q if holder is None else
               list(challenge_picks(_artifact(heads[holder_name]), menu, hid)))
        verified, served = {}, {}
        for n in fed(sorted(heads)):
            if epoch in missed.get(n, ()):
                continue
            u = uid_of[n]
            served[u] = n
            challenges = n not in never_answers and holder_serves
            own = (list(challenge_picks(_artifact(heads[n]), menu, hid)) if challenges
                   else [])
            verified[u] = SimpleNamespace(
                scored=challenges,
                challenge=SimpleNamespace(size=q, holder=holder) if challenges else None,
                scored_results=[
                    SimpleNamespace(question_id=f"s{epoch}q{i}", routed_model=m,
                                    correct=coin(f"{seed}|{epoch}|{u}|{i}") < acc[m], failed=False,
                                    completion_tokens=100) for i, m in enumerate(own)],
                holder_results=[
                    SimpleNamespace(question_id=f"s{epoch}q{i}", routed_model=hp,
                                    correct=coin(f"h|{seed}|{epoch}|{u}|{i}") < acc[hp],
                                    failed=False, completion_tokens=100)
                    for i, (m, hp) in enumerate(zip(own, hps)) if hp is not None and hp != m])
        oi = OddsInput(nonce=hashlib.sha256(f"seat-v4-{seed}-{epoch}".encode()).digest(),
                       answers=fed(answers), benchmark_of=bench_of,
                       heads={uid_of[n]: heads[n] for n in fed(sorted(heads)) if uid_of[n] in served},
                       hidden_of=hidden_of, menu_state=menu)
        commits = [Commitment(f"hk{uid_of[n]}", f"ck{uid_of[n]}", hash_of[n], uid_of[n])
                   for n in fed(sorted(heads))]
        kw = dict(
            epoch_index=epoch, epoch_id=f"e{epoch:08d}", boundary_block=epoch * bpe,
            blocks_per_epoch=bpe, questions=questions, pool_models=ids,
            prices={m: (1e-6, 4e-6) for m in ids}, verified_proofs=verified,
            head_hashes={u: hash_of[n] for u, n in served.items()},
            coldkey_of={u: f"ck{u}" for u in served}, hotkey_of={u: f"hk{u}" for u in served},
            records={}, commitments=commits, window=window, seat=seat, copy_state=copy,
            burn_uid=0, params=params, shares=(0.85, 0.10, 0.05), burn_in_questions=0,
            bench_category=category, odds_input=oi,
            picks=PicksInput(state=menu, challenge=challenge),
            seniority_floor=bpe, seniority_salt="0x" + "5a" * 32)
        if spend:
            kw["pick_spend"] = {
                e.id: (float(ms.entry_pick_cost(e)),
                       {c: float(ms.entry_pick_cost(e, c)) for c in sorted(epoch_plan.mix_v4().targets)})
                for e in menu.entries}
        r = run_v2_epoch(**kw, memo=None if memo_for is None else memo_for(epoch))
        if keep is not None:
            keep(epoch, kw, {u: head_bytes(served[u]) for u in served}, r)
        window, seat, copy = r.window, r.decision.state, r.copy_state
        rv = r.reveal
        out.append({k: rv[k] for k in ("labels_v4", "head_odds", "late_odds", "odds_v4",
                                       "champion", "picks_v4", seat_v4.REVEAL_KEY)}
                   | {k: rv["v2_state"][k] for k in ("labels_digest", "seat", seat_v4.DIGEST_KEY)}
                   | {"weights": [list(map(int, r.uids)), [repr(float(x)) for x in r.weights]]})
    return SimpleNamespace(window=window, seat=seat, copy=copy, out=out, heads=heads,
                           hash_of=hash_of, uid_of=uid_of, rate=rate, fed=fed,
                           hidden_of=hidden_of)


def _seat_v4_memo(fed) -> list:
    """The seat reading the memorisation verdict (the third part of the
    seat_v4 stage): v4 epochs through run_v2_epoch with the epoch's memo on
    the constructed window of fugal_subnet/attacks/memo_seat_world.py, each
    scenario with and without the memo: a refused holder serving, a refused
    holder not serving beside an honest head that takes the opened seat, a
    refused head beside an earlier honest one, and a successor judged from
    its inherited rows."""
    from fugal_subnet import seat_v4
    from fugal_subnet.attacks import memo_seat_world as msw

    world = msw.World()
    out, events = [], []
    for sc in fed([dict(holder="mem", kept=["mem"], served=["mem"]),
                   dict(holder="mem", kept=["mem", "honest"], served=["honest"]),
                   dict(holder=None, kept=["mem", "honest"], served=["mem", "honest"]),
                   dict(holder="line1", kept=["line1"], served=["line2"])]):
        sc = dict(sc, kept=fed(sc["kept"]), served=fed(sc["served"]))
        for memo in (False, True):
            r = world.run(memo=memo, fed=fed, **sc)
            rv = r.reveal
            events.append((sc["holder"], memo, r.decision.event,
                           len([1 for ev in rv.get("memo_v4", {}).get("evidence", {}).values()
                                if ev["refused"]])))
            out.append({k: rv[k] for k in ("champion", seat_v4.REVEAL_KEY, "v2_state")}
                       | {"memo_v4": rv.get("memo_v4"), "event": r.decision.event,
                          "weights": [list(map(int, r.uids)), [repr(float(x)) for x in r.weights]]})
    events = sorted(events, key=repr)
    want = sorted([("mem", False, "hold", 0), ("mem", True, "open", 1),
                   ("mem", False, "hold", 0), ("mem", True, "take", 1),
                   (None, False, "fill", 0), (None, True, "fill", 1),
                   ("line1", False, "hold", 0), ("line1", True, "open", 1)], key=repr)
    if events != want:
        raise SystemExit(f"the seat_v4 stage no longer has the seat refuse a memorising head: {events}")
    return sorted(out, key=lambda o: json.dumps(o, sort_keys=True))


def _seat_v4_step(fed) -> list:
    """The seat step over twenty epochs of the seat's synthetic world (the
    second part of the seat_v4 stage)."""
    from fugal_subnet import payout, seat_v4
    from fugal_subnet.attacks.seat_v4_world import ENTRIES, World

    w = World(seed=11, floor_block=40, salt="0x" + "ab" * 32)
    w.missing_after[ENTRIES[1]] = 12            # the second entry's allowance ends at 12
    for name, kw in fed([
            ("a", dict(skill=0.6, block=10, rows=(ENTRIES[0],), hotkey="hk-a", owner="ck-a")),
            ("b", dict(skill=0.1, block=30)),
            ("c", dict(skill=0.5, block=60, noise=3)),
            ("d", dict(skill=0.5, block=70, noise=4, owner="ck-d")),   # c, lightly altered
            ("e", dict(skill=0.6, block=90))]):
        w.add(name, **kw)
    out, events = [], []
    for e in range(1, 21):
        if e == 8:                              # the holder adds the row by succession
            w.add("a2", skill=0.6, block=75, rows=ENTRIES, hotkey="hk-a", owner="ck-a")
        holder = "a2" if e >= 8 else "a"
        served, answered = [holder, "b"], [holder, "b"]
        if e >= 6 and e != 11:                  # c misses 11, challenges every other epoch
            served.append("c")                  # once the holder has the row (from 10)
            answered += ["c"] if e % 2 == 0 and e >= 10 else []
        if e >= 7:
            served.append("d")
            answered += ["d"] if e >= 9 else []
        if e >= 12:                             # e, no sharper than the holder, takes
            served.append("e")                  # only once the holder lapses
            answered.append("e")
        if e in (14, 15, 16):                   # the holder lapses
            served.remove(holder)
            answered.remove(holder)
        d = w.epoch(e, served=fed(served), answered=fed(answered),
                    memo=fed(["b"] if e == 9 else []))
        uids, weights = payout.seat_weights(d.champion, d.state.history, w.serving,
                                            set(d.payable), shares=(0.85, 0.10, 0.05),
                                            paid_fraction=1.0, burn_uid=0)
        events.append((d.event, w.holder()))
        out.append({"event": d.event, "state": seat_v4.to_dict(d.state),
                    "evidence": d.evidence, "decision": seat_v4.decision_for_reveal(d),
                    "payable": sorted(d.payable),
                    "weights": [list(map(int, uids)), [repr(float(x)) for x in weights]]})
    fill = next(i for i, ev in enumerate(events) if ev[0] != "empty")
    if not (events[fill] == ("fill", "a") and events[7] == ("hold", "a2")
            and events[15] == ("take", "e") and ("hold", "a2") in events[13:15]
            and all(h in ("a", "a2") for _ev, h in events[fill:15])):
        raise SystemExit(f"the seat_v4 stage no longer fills, succeeds and lapses: {events}")
    return out


def _seat_v4_chain(fed) -> list:
    """A succession chain on the seat's world with the live late-odds
    schedule (the fourth part of the seat_v4 stage): the holder's hotkey
    commits a successor every epoch from 6 (its rows byte for byte), serves
    an earlier one's bytes again at 15 and succeeds again from 16, until its
    chain is longer than MAX_PREDECESSORS (the window) and drops its oldest; three
    other hotkeys commit a successor every epoch, the four sharing a bound
    of 4 head-epochs; a better challenger answers from 12. A successor's
    odds before its commit are its predecessors' (labels_v4.predecessor_odds).
    A head that answers is down for 4..9 and back at 10 while two heads that
    never answer are first verified at 10 with nine epochs owed each: the
    seat's heads take their late-odds rounds first (seat_v4.late_odds_first),
    so the returning head's six epochs are filled at 10 and 11."""
    from fugal_subnet import seat_v4
    from fugal_subnet.attacks.seat_v4_world import World, hash_of

    w = World(seed=24, late_slots=4, labels=300)
    w.add("a0", skill=0.4, block=5, hotkey="hk-a", owner="ck-a")
    w.add("r", skill=0.1, block=5, hotkey="hk-r", owner="ck-r")
    for i in range(2):
        w.add(f"b{i}", skill=0.1, block=5, hotkey=f"hk-b{i}", owner="ck-b")
    out, events = [], []
    last = seat_v4.MAX_PREDECESSORS + 16           # a{last}'s chain: a{last-1}..a16, a7, a6, a0
    for e in range(1, last + 1):
        holder = "a0"
        if 6 <= e:
            holder = "a7" if e == 15 else f"a{e}"
            if e != 15:
                w.add(holder, skill=0.4, block=e * 10 - 1, hotkey="hk-a", owner="ck-a")
            else:
                w.commits["hk-a"] = ("a7", "ck-a", 69)
        others = []
        for i in fed(range(3)):
            if e >= 6:
                w.add(f"o{i}-{e}", skill=0.2, block=e * 10 - 1, hotkey=f"hk-o{i}", owner="ck-o")
                others.append(f"o{i}-{e}")
        if e == 11:
            w.add("c", skill=0.7, block=105)
        back = ([] if 4 <= e <= 9 else ["r"]) + (["b0", "b1"] if e >= 10 else [])
        served = [holder, *others, *back] + (["c"] if e >= 12 else [])
        d = w.epoch(e, served=fed(served),
                    answered=fed([holder] + (["c"] if e >= 12 else [])
                                 + ([] if 4 <= e <= 9 else ["r"])))
        events.append((d.event, w.holder()))
        out.append({"event": d.event, "state": seat_v4.to_dict(d.state), "evidence": d.evidence,
                    "odds_epochs": {h: sorted(per) for h, per in sorted(w.window.odds.items())}})
        if e == 11:                             # r's late odds came before b0's and b1's
            first = all(k in w.window.odds[hash_of("r")] for k in range(4, 10))
        if e == 16:                             # a7, served again at 15, is in a16's chain
            again = hash_of("a7") in w.state.head(hash_of("a16")).predecessors
            again = again and 15 in w.window.odds[hash_of("a7")]
    line = w.state.head(hash_of(f"a{last}"))
    if not (("take", "c") in events and again and first and line is not None
            and len(line.predecessors) == seat_v4.MAX_PREDECESSORS):
        raise SystemExit(f"the seat_v4 stage no longer runs a succession chain to a take: {events}")
    return out


def _explore_shares_v4(seed: int) -> dict:
    """Protocol v4 exploration (explore_weights.plan_shares_v4,
    epoch_plan.exploration_plan_v4): from the pinned seed menu's measured
    costs, the cost shares, the coverage shares, the damping f, the spend
    mix t, both pair weights and both expected spends, for pair counts that
    put the coverage part in each of its regimes (cold start, steady state,
    one entry short, a damped spike the cap binds on, a synthetic menu whose
    uniform coverage is over the cap), and three miners' plans on a synthetic
    pool. Fractions are digested as exact strings.

    Under --perturb the costs and pair counts are fed in reverse and the
    process has another hash seed, so a computation that read arrival order
    or dict order would show here.
    """
    import dataclasses
    from fractions import Fraction

    from fugal_subnet import config, epoch_plan, menu_state
    from fugal_subnet import explore_weights as xw

    def draw(*parts) -> int:
        text = "|".join(str(p) for p in (seed, "xv4") + parts)
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")

    def fed(mapping):
        items = sorted(mapping.items())
        return dict(reversed(items)) if os.environ.get("FUGAL_PERTURB") else dict(items)

    state = menu_state.seed_state(1000)
    ids = list(state.ids())
    costs = menu_state.costs(state)
    by_cost = sorted(ids, key=lambda e: (costs[e], e))
    dearest = by_cost[-1]
    cases = {
        "cold": {},
        "steady": {e: 100 + draw("steady", e) % 300 for e in ids},
        "window": {e: draw("window", e) % 260 for e in ids},
        "one_short": {**{e: 150 for e in ids}, dearest: draw("short") % 40},
        # The fourteen dearest entries short at once: a spike the coverage cap
        # binds on. At 44 coverage calls and $1.10 (SEAT_MINIMAL_V4.md 5.3) the
        # pinned menu's costs reach the cap only with a dozen or more of its
        # dearest entries short together.
        "spike": {**{e: 150 for e in ids}, **{e: draw("spike", e) % 20 for e in by_cost[-14:]}},
    }
    synthetic = {f"lab/e{i:02d}": Fraction(1 + draw("cost", i) % 9_000, 100_000) for i in range(12)}
    synthetic["lab/e00"] = Fraction(9, 2)                     # uniform coverage over the cap

    def summary(plan: xw.SharesV4, c) -> dict:
        return {"cost": sorted(plan.cost.items()), "coverage": sorted(plan.coverage.items()),
                "halves": plan.exponent_halves, "deficits": sorted(plan.deficits.items()),
                "f": str(plan.damping), "t": str(plan.toward_deficits),
                "shared_weights": sorted(plan.shared_weights().items()),
                "private_weights": sorted(plan.private_weights().items()),
                "spend": {k: str(v) for k, v in sorted(plan.expected_spend(c).items())}}

    out = {name: summary(xw.plan_shares_v4(fed(costs), fed(pairs)), costs)
           for name, pairs in cases.items()}
    out["synthetic_over_cap"] = summary(
        xw.plan_shares_v4(fed(synthetic), fed({"lab/e00": 0, "lab/e03": 50})), synthetic)
    if not (Fraction(out["spike"]["t"]) < 1 and out["synthetic_over_cap"]["t"] in ("0", "1")
            and out["cold"]["f"] == "1" and out["steady"]["f"] == "0"):
        raise SystemExit("the explore_shares_v4 stage no longer exercises every coverage regime")

    # The epoch's draw: the menu publishes each entry's count (Entry.explore_pairs)
    # and explore_shares_v4 reads it from there, with no other input.
    counted = dataclasses.replace(state, entries=tuple(
        dataclasses.replace(e, explore_pairs=cases["window"][e.id]) for e in state.entries))
    menu_state.validate(counted)
    plan = epoch_plan.explore_shares_v4(counted)
    if plan != xw.plan_shares_v4(fed(costs), fed(cases["window"])):
        raise SystemExit("explore_shares_v4 does not read the counts the menu publishes")
    pool = [{"question_id": f"{b}-{i:04d}", "benchmark": b, "prompt": f"q {b} {i}"}
            for b in sorted(config.BENCHMARK_CATEGORY_V3) for i in range(120)]
    if os.environ.get("FUGAL_PERTURB"):
        pool = list(reversed(pool))
    nonce = hashlib.sha256(f"v4-{seed}".encode()).digest()
    scored = {q["question_id"] for q in epoch_plan.scored_slice(3, nonce, pool, 300)}
    out["plans"] = {hk: [sorted(part.items()) for part in
                         epoch_plan.exploration_plan_v4(nonce, pool, scored, plan, hk)]
                    for hk in ("5Fhotkey-a", "5Fhotkey-b", "5Fhotkey-c")}
    return out


def _menu_inputs_v4(seed: int) -> dict:
    """Protocol v4's step inputs from verified proofs (fugal_subnet/menu_inputs.py),
    four stages on the pinned seed menu with two entries that joined as
    provisional (in 989 and 990):

    - `endpoint_vote`: fourteen epochs of reads by nine operators, several of
      them with many hotkeys, some reads failed, some hosts unlisted, one
      operator reading a price of its own;
    - `token_means`: the epochs' summaries (pooled pairs, and the
      provisional entries each names) and, through the window, each entry's
      token sums and their weighted means, exploration-pair counts and
      accuracy bounds;
    - `provisional_gates`: the validator's v4 expectation for four hotkeys
      in a join epoch (its coverage aimed first at the entry that joined),
      and what the gates read of each provisional entry (each coldkey's
      calls and failures, its labels by category, its accuracy bounds);
    - `retirement`: health over the window (a broken entry, one operator
      failing a healthy one, two outage epochs, an entry whose hosts left
      the listing), the step's verdict on each, and the StepInputs digest.

    Under --perturb the proofs, the pooled pairs, the window and every
    mapping arrive in reverse and the process has another hash seed, so an
    aggregation that read arrival order or dict order would show here.
    """
    import dataclasses

    from fugal_subnet import config, epoch_plan, menu_inputs, menu_state
    from fugal_subnet import explore_pool as xp
    from fugal_subnet.tee import verify as verify_mod
    from fugal_subnet.tee.endpoint_reads import EndpointRead, HostListing
    from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult

    perturb = bool(os.environ.get("FUGAL_PERTURB"))

    def draw(*parts) -> int:
        text = "|".join(str(p) for p in (seed, "inputs-v4") + parts)
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")

    def fed(items):
        items = list(items)
        return list(reversed(items)) if perturb else items

    def fed_map(mapping):
        items = sorted(mapping.items())
        return dict(reversed(items)) if perturb else dict(items)

    from fugal_subnet import chain_proposals as cp

    seed_menu = menu_state.seed_state(1000)
    entries = list(seed_menu.entries)
    proposers = ("5Fproposer-a", "5Fproposer-b")

    def joined(state, j: int):
        """`state` with lab-new/model-j joined as provisional at its epoch."""
        doc = cp.ProposalDoc(proposer=proposers[j], model=f"lab-new/model-{j}",
                             reasoning=("effort", "high"), provider_order=("deepinfra/fp8",),
                             quantization="fp8")
        return menu_state.apply_joins(state, [menu_state.NewEntry(
            request=doc.request(65536), price_in=150_000, price_out=600_000,
            proposal=cp.proposal_hash(cp.document_bytes(doc)), proposer=proposers[j],
            coldkey=f"ck-prop-{j}")])

    first = joined(menu_state.seed_state(989), 0)
    both = joined(dataclasses.replace(first, epoch=990), 1)
    new_ids = tuple(e.id for e in both.entries if e.provisional)
    pool = [{"question_id": f"{b}-{i:04d}", "benchmark": b}
            for b in sorted(config.BENCHMARK_CATEGORY_V3) for i in range(60)]
    plan = epoch_plan.plan_v4(pool)
    category_of = fed_map({q["question_id"]: epoch_plan.mix_v4().bench_category[q["benchmark"]]
                           for q in pool})
    by_cost = sorted(entries, key=lambda e: (menu_state.entry_cost(e), e.id))
    broken, attacked = by_cost[0], by_cost[1]
    gone = next(e for e in by_cost[2:] if len(e.provider_order) == 1
                and e.model not in (broken.model, attacked.model))
    operators = [(f"ck{c}", [f"5Fhk{c}-{h}" for h in range(1 + c % 3)]) for c in range(9)]
    operators += [(f"ck-prop-{j}", [hk]) for j, hk in enumerate(proposers)]

    def menu_of(epoch: int):
        held = seed_menu if epoch < 989 else first if epoch == 989 else both
        return dataclasses.replace(held, epoch=epoch)

    summaries, out_vote, out_expect = [], [], {}
    weighed_apart = 0                     # private pairs whose two tables differ
    for epoch in range(980, 994):
        state = menu_of(epoch)
        nonce = hashlib.sha256(f"v4-inputs-{seed}-{epoch}".encode()).digest()
        scored = [q["question_id"] for q in epoch_plan.scored_slice(3, nonce, pool, 20)]
        outage_epoch = epoch in (985, 986)
        proofs, shared_of = [], {}
        for ck, hotkeys in operators:
            for hk in hotkeys:
                x = verify_mod.expected_v4(state, epoch=epoch, nonce=nonce, pool=pool,
                                           scored_ids=scored, hotkey=hk, plan=plan)
                shared_of[hk] = set(x.shared)
                if epoch == 990 and hk in ("5Fhk0-0", "5Fhk4-0", "5Fproposer-a", "5Fproposer-b"):
                    out_expect[hk] = {"shared": sorted(x.shared.items()),
                                      "cost": sorted(x.cost_aware.items()),
                                      "coverage": sorted(x.coverage.items())}
                results = []
                for q in scored:
                    results.append(QuestionResult(q, entries[draw("pick", epoch, q) % 6].id,
                                                  draw("ok", epoch, q) % 4 != 0, 0.0, "ab" * 32,
                                                  100 + draw("pt", q) % 900, 50))
                for q, eid in sorted(x.exploration.items()):
                    failed = (outage_epoch or (eid == broken.id and epoch >= 984)
                              or (eid == attacked.id and ck == "ck2")
                              or (eid in new_ids and draw("pf", epoch, hk, q) % 4 == 0))
                    results.append(QuestionResult(
                        q, eid, not failed and draw("c", epoch, hk, q) % 5 != 0, 0.0,
                        "ab" * 32, 0 if failed else 80 + draw("i", q, eid) % 2000,
                        0 if failed else 20 + draw("o", q, eid) % 9000, is_exploration=True,
                        failed=failed))
                reads = []
                if draw("readfail", epoch, hk) % 7:
                    for model, hosts in sorted({e.model: e.provider_order for e in entries}
                                               .items()):
                        hl = []
                        for host in sorted(hosts):
                            listed = not (model == gone.model and epoch >= 986)
                            price = 50_000 + draw("price", epoch // 5, model, host) % 9_000_000
                            if ck == "ck5":
                                price *= 40                     # one operator's own price
                            hl.append(HostListing(host, True, price, 4 * price,
                                                  (65536, 131072)[draw("lim", model, host) % 2],
                                                  "fp8", 0) if listed else HostListing(host, False))
                        reads.append(EndpointRead(model, tuple(hl)))
                proofs.append(menu_inputs.VerifiedProof(BenchmarkProof(
                    epoch_id=f"e{epoch:08d}", nonce=nonce.hex(), questions_hash="",
                    weights_hash=hashlib.sha256(f"head-{ck}".encode()).hexdigest(),
                    source_hash="", results=results, total_cost_usd=0.0, per_model_costs={},
                    attestation_quote=b"", timestamp=0.0, hotkey=hk, routing_protocol="v4",
                    menu_hash=menu_state.menu_hash(state), endpoint_reads=tuple(reads)), ck))
        answers = [xp.ExploreAnswer(r.question_id, r.routed_model, vp.coldkey, r.correct, r.failed,
                                    private=r.question_id not in shared_of[vp.hotkey],
                                    prompt_tokens=r.prompt_tokens,
                                    completion_tokens=r.completion_tokens)
                   for vp in proofs for r in vp.proof.exploration_results]
        shares = epoch_plan.explore_shares_v4(state)
        pooled = xp.pool_pairs(fed(answers), nonce=nonce, coldkey_cap=xp.coldkey_cap(4),
                               weights=fed_map(shares.shared_weights()),
                               private_weights=fed_map(shares.private_weights()))
        # Each pair weighs the inverse of the probability its own set drew
        # its entry with: the private table for a question of the private
        # set, the shared table otherwise.
        private_cells = {(a.question_id, a.model) for a in answers if a.private}
        shared_w, private_w = shares.shared_weights(), shares.private_weights()
        for pair in pooled:
            mine = (private_w if (pair.question_id, pair.model) in private_cells else shared_w)
            if pair.weight != mine[pair.model]:
                raise SystemExit("explore_pool weighs a pooled pair by the other set's table")
            weighed_apart += ((pair.question_id, pair.model) in private_cells
                              and private_w[pair.model] != shared_w[pair.model])
        summary = menu_inputs.summarize_epoch(
            epoch, state, fed(proofs), pooled=fed(pooled), category_of=category_of,
            counted_heads={hashlib.sha256(b"head-ck0").hexdigest(),
                           hashlib.sha256(b"head-ck3").hexdigest()})
        summaries.append(menu_inputs.summary_to_dict(summary))
        out_vote.append([dataclasses.astuple(r) for r in menu_inputs.endpoint_vote(fed(proofs))])
        last = (state, nonce, proofs)

    window = [menu_inputs.summary_from_dict(d) for d in fed(summaries)]
    # The production step's parameters: token means and gates weighted to the
    # pinned v4 mix (epoch_plan.menu_step_params_v4).
    params = epoch_plan.menu_step_params_v4(5000)
    mix = params.mix
    k = 993
    start = menu_state.carry(last[0])
    stats = menu_inputs.token_stats(start, window, through=k)
    means = {s.entry_id: menu_state.weighted_token_means(s.cells, mix) for s in stats}
    health = menu_inputs.health_stats(start, window, through=k)
    verdicts = {h.entry_id: menu_state.health_retires(h, params) for h in health}
    gates = menu_inputs.gate_stats(start, window, through=k, mix=mix)
    first_pairs = sum(1 for hk in out_expect for part in ("shared", "cost", "coverage")
                      for _q, eid in out_expect[hk][part] if eid == new_ids[1])
    if not (verdicts[broken.id] == menu_state.REASON_FAILING and verdicts[attacked.id] is None
            and verdicts[gone.id] == menu_state.REASON_GONE
            and sum(menu_inputs.outage(s) for s in window) == 2
            and len(gates) == 2 and all(g.coldkeys and g.cells for g in gates)
            and any(c.failed for g in gates for c in g.coldkeys) and first_pairs
            and weighed_apart):
        raise SystemExit("the v4 input stages no longer exercise a broken entry, an attacked "
                         "one, a gone one, the outage guard, two provisional entries' gates "
                         "with failed calls, coverage aimed at a join and a private pair "
                         "weighed by its own table")
    lab_list = _stage_lab_list(seed_menu)
    inputs = menu_inputs.step_inputs(
        start, last[0], window, fed(last[2]), epoch=k, nonce=last[1], lab_list=lab_list,
        params=params, category_of=category_of)
    stepped = menu_state.step_detail(start, inputs, params)
    return {
        "endpoint_vote": out_vote,
        "token_means": {"stats": [dataclasses.astuple(s) for s in stats],
                        "means": sorted(means.items()),
                        "counts": [dataclasses.astuple(c) for c in
                                   menu_inputs.explore_counts(start, window, through=k)],
                        "accuracy": [dataclasses.astuple(a) for a in
                                     menu_inputs.accuracy_stats(start, window, through=k,
                                                                mix=mix)],
                        "summaries": summaries},
        "provisional_gates": {
            "expected": out_expect, "first_pairs": first_pairs,
            "gates": [dataclasses.astuple(g) for g in gates]},
        "retirement": {"health": [dataclasses.astuple(h) for h in health],
                       "verdicts": sorted((e, v or "") for e, v in verdicts.items()),
                       "events": [dataclasses.astuple(e) for e in stepped.events],
                       "menu": menu_state.menu_hash(stepped.state)},
    }


def _pick_cost_v4(seed: int) -> dict:
    """Protocol v4's pick-cost table (W13; menu_inputs.graded_cells and
    pick_stats, menu_state.pick_table and the step's refresh of it,
    pick_costs and pick_cost) over six epochs of synthetic verified proofs
    on the seed menu: five operators, two of them with three hotkeys each,
    failed calls, cells answered by an even number of operators, a cell past
    an entry's answer limit, a request update mid-window (its earlier
    revision's cells are not read) and an entry no head routes to (its call
    cost). Each epoch's summary goes through its JSON form; under
    --perturb every proof list, every proof's results, the category map
    and the window arrive reversed."""
    import dataclasses
    import random

    from fugal_subnet import epoch_plan, menu_inputs
    from fugal_subnet import menu_state as ms
    from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult

    perturb = bool(os.environ.get("FUGAL_PERTURB"))

    def fed(items):
        items = list(items)
        return list(reversed(items)) if perturb else items

    rnd = random.Random(seed * 7919 + 13)
    seed_menu = ms.seed_state(990)
    params = epoch_plan.menu_step_params_v4(5000)
    cats = [c for c, _ in params.mix]
    qids = [f"pc-q{i:04d}" for i in range(240)]
    category_of = {q: cats[int(hashlib.sha256(q.encode()).hexdigest(), 16) % len(cats)]
                   for q in qids}
    routed = list(seed_menu.ids())[:12]               # the rest: no graded cell
    updated = routed[3]
    tight = routed[5]
    menu = dataclasses.replace(seed_menu, entries=tuple(
        dataclasses.replace(e, max_tokens=2_000) if e.id == tight else e
        for e in seed_menu.entries))
    hotkeys = {"ck0": ["5Fpc0a", "5Fpc0b", "5Fpc0c"], "ck1": ["5Fpc1a"], "ck2": ["5Fpc2a"],
               "ck3": ["5Fpc3a", "5Fpc3b", "5Fpc3c"], "ck4": ["5Fpc4a"]}
    summaries, cells_out = [], []
    first, last = 993, 998
    for k in range(first, last + 1):
        state = dataclasses.replace(menu, epoch=k)
        if k >= 996:                                  # the request update joined at 996
            state = dataclasses.replace(state, entries=tuple(
                dataclasses.replace(e, rev=2, joined_epoch=0) if e.id == updated else e
                for e in state.entries))
        asked = rnd.sample(qids, 60)
        proofs = []
        for ck, hks in sorted(hotkeys.items()):
            for hk in hks:
                results = []
                for q in asked:
                    # Each question goes to one of two entries, so most cells
                    # are answered by several operators, some by two.
                    eid = routed[(int(hashlib.sha256(q.encode()).hexdigest(), 16)
                                  + rnd.randint(0, 1)) % len(routed)]
                    failed = rnd.random() < 0.05
                    results.append(QuestionResult(
                        q, eid, not failed and rnd.random() < 0.6, 0.0, "ab" * 32,
                        200 + rnd.randint(0, 400), rnd.choice([30, 900, 2_999, 15_000, 60_000]),
                        failed=failed))
                proofs.append(menu_inputs.VerifiedProof(BenchmarkProof(
                    epoch_id="pc", nonce="00" * 32, questions_hash="", weights_hash="1f" * 32,
                    source_hash="", results=fed(results), total_cost_usd=0.0,
                    per_model_costs={}, attestation_quote=b"", timestamp=0.0, hotkey=hk,
                    routing_protocol="v4", menu_hash="cd" * 32), ck))
        graded = menu_inputs.graded_cells(fed(proofs), state,
                                          dict(fed(category_of.items())))
        cells_out.append([dataclasses.astuple(c) for c in graded])
        summary = menu_inputs.EpochSummary(k, (), (), (), (), graded)
        summaries.append(json.loads(json.dumps(menu_inputs.summary_to_dict(summary))))
    window = [menu_inputs.summary_from_dict(d) for d in fed(summaries)]
    # The step for scored epoch 998 derives epoch 1000, a refresh epoch.
    start = dataclasses.replace(menu, epoch=last + params.menu_lag - 1, entries=tuple(
        dataclasses.replace(e, rev=2) if e.id == updated else e for e in menu.entries))
    stats = menu_inputs.pick_stats(start, window, through=last)
    stepped = ms.step_detail(start, ms.StepInputs(
        epoch=last, nonce=bytes(32), lab_list=_stage_lab_list(seed_menu),
        pick_stats=tuple(fed(stats))), params).state
    by_id = {e.id: e for e in stepped.entries}
    rows = [menu_inputs.GradedCell(*t) for g in cells_out for t in g]
    old_rev = sum(c.cells for c in rows if c.entry_id == updated and c.rev == 1)
    new_rev = sum(c.cells for c in rows if c.entry_id == updated and c.rev == 2)
    tight_rows = [c for c in rows if c.entry_id == tight]
    if not (old_rev and new_rev and by_id[updated].pick is not None
            and by_id[updated].pick.cells == new_rev
            and all(by_id[e].pick is None for e in stepped.ids() if e not in routed)
            and all(by_id[e].pick is not None for e in routed)
            and tight_rows and all(c.tokens_out <= 2_000 * c.cells for c in tight_rows)
            and stepped.epoch == 1000):
        raise SystemExit("the pick_cost_v4 stage no longer exercises an update mid-window, an "
                         "unrouted entry, a clamped cell and a refresh")
    costs = ms.pick_costs(stepped)
    return {
        "cells": cells_out,
        "stats": [dataclasses.astuple(s) for s in stats],
        "tables": [(e.id, dataclasses.astuple(e.pick) if e.pick else None)
                   for e in stepped.entries],
        "pick_costs": [(i, str(c)) for i, c in sorted(costs.items())],
        "by_category": [(i, c, str(ms.pick_cost(stepped, i, c)))
                        for i in sorted(routed) for c in cats],
        "menu": ms.menu_hash(stepped),
    }


def _stage_lab_list(seed_menu):
    """A lab and host list covering the seed menu and the proposals' lab,
    as verify_signed_list returns one (no key: the step reads its content
    and hash)."""
    from fugal_subnet import menu_rules as mr
    labs, hosts, first_party = {"lab-new"}, set(), {}
    for e in seed_menu.entries:
        lab = e.model.split("/", 1)[0]
        labs.add(lab)
        for tag in e.provider_order:
            hosts.add(mr.host_identity(tag))
            if e.quantization == mr.UNKNOWN_QUANTIZATION and mr.host_identity(tag) != lab:
                first_party.setdefault(lab, set()).add(mr.host_identity(tag))
    raw = mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, 1, labs=sorted(labs),
                                  hosts=sorted(hosts), first_party=first_party)
    return mr.MenuAllowlist(serial=1, labs=frozenset(labs), hosts=frozenset(hosts),
                            first_party=mr.first_party_map(first_party),
                            sha256=hashlib.sha256(raw).hexdigest(), signer="SHA256:stage")


def _menu_publication(seed: int) -> list:
    """Protocol v4's menu publication (fugal_subnet/menu_publish.py) over
    eight epochs that cross a refresh epoch, with an entry that retires and
    price reads: a root that steps, publishes the menu to a
    store, reads it back and commits its hash with the reveal; a second
    validator that derives the same step from the same proofs arriving in
    the reverse order; and a follower that reads each epoch's menu from the
    chain and audits the root's. Digested per epoch: the root's `menu_next`
    hash and bytes, its commitment, the menu each node ran on and the
    follower's audit. The stage stops unless the second validator's bytes
    equal the root's and every audit of the honest root is empty.

    The step inputs stand in for the validator-side aggregation over
    verified proofs (menu_inputs.step_inputs, W4a): each collection is fed
    in proof order to the root, reversed to the others, and reversed for
    the root too under --perturb.
    """
    import dataclasses
    import io
    import tempfile
    import urllib.error
    from pathlib import Path

    from fugal_subnet import menu_publish as mp
    from fugal_subnet import menu_rules as mr
    from fugal_subnet import menu_state as ms
    from fugal_subnet import reveal_publisher as rp
    from fugal_subnet import reveal_sync
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block
    from tests import menu_fixtures as fx
    from tests.signed_list_fixtures import Signer

    perturb = bool(os.environ.get("FUGAL_PERTURB"))
    bpe, netuid, bucket = 10, 7, "fugal-stage-reveals"
    params = dataclasses.replace(fx.PARAMS, refresh_epochs=ms.MENU_REFRESH_EPOCHS)

    def draw(*parts) -> int:
        text = "|".join(str(p) for p in (seed,) + parts)
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")

    def fed(items, flip: bool):
        items = list(items)
        return tuple(reversed(items)) if flip else tuple(items)

    class Store:
        def __init__(self):
            self.objects = {}

        def put(self, key, data):
            self.objects.setdefault(key, bytes(data))

        def get(self, key, limit):
            return self.objects[key]

        def opener(self, req, timeout=None):
            key = req.full_url.split("/", 4)[4]
            if key not in self.objects:
                raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)
            return io.BytesIO(self.objects[key])

    class Uploader:
        def __init__(self, store):
            self.store = store

        def upload(self, path, key):
            with open(path, "rb") as f:
                self.store.objects[key] = f.read()
            return f"gs://{bucket}/{key}"

    class Chain:
        def __init__(self):
            self.block, self.commits = 0, []

        def commit(self, data):
            self.commits.append((self.block, data))
            return True

        def read(self, epoch):
            older = [d for b, d in self.commits if b <= epoch * bpe]
            return mp.menu_hash_in_commitment(older[-1] if older else "", epoch)

    def inputs(k: int, state: ms.MenuState, flip: bool) -> ms.StepInputs:
        reads = [fx.read(m, h, price_in=100_000 + draw("in", k // 4, m, h) % 50_000,
                         price_out=400_000 + draw("out", k // 4, m, h) % 90_000,
                         voters=3 + draw("v", k, m, h) % 3)
                 for m, hosts in (("lab-a/model-one", ("host-a/fp8", "host-b/fp8")),
                                  ("lab-b/model-two", ("host-c",)))
                 for h in hosts]
        # One entry whose hosts left the listing retires (gone) in the step for 116.
        health = [ms.HealthStat(state.entries[-1].id, state.entries[-1].rev, unlisted_epochs=6)
                  ] if k == 116 else []
        return ms.StepInputs(
            epoch=k, nonce=hashlib.sha256(f"stage-{seed}-{k}".encode()).digest(),
            lab_list=fx.ALLOW, host_reads=fed(reads, flip),
            explore_counts=fed((ms.ExploreCount(e.id, e.rev, draw("x", k, e.id) % 200)
                                for e in state.entries), flip),
            token_stats=fed((ms.TokenStat(e.id, e.rev, (ms.TokenCell(
                "code", 31 + draw("t", e.id) % 9, 31 * 300, 31 * (900 + draw("o", k, e.id) % 99)),))
                for e in state.entries), flip),
            health=fed(health, flip))

    out = []
    with tempfile.TemporaryDirectory() as tmp:
        signer = Signer(Path(tmp), "stage")
        payload = mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, fx.ALLOW.serial, labs=fx.LABS,
                                          hosts=fx.HOSTS, first_party=fx.FIRST_PARTY)
        signature, signers = signer.sign(payload), signer.signers_line().encode()
        store, chain = Store(), Chain()
        base = fx.wide_state(epoch=115)            # above the compiled minimum

        def node(name):
            book = mp.MenuBook(os.path.join(tmp, name))
            book.hold(base)
            lists = mp.ListBook(None, shipped=False, signers=signers)
            if lists.add(mr.KIND_MENU_ALLOWLIST, payload, signature) is None:
                raise SystemExit("the menu_publication stage's signed list does not verify")
            return mp.MenuNode(
                first_epoch=100, book=book, lists=lists, read_commitment=chain.read,
                fetch_menu=lambda h: reveal_sync.fetch_menu_bytes(
                    h, bucket=bucket, netuid=netuid, opener=store.opener),
                step_params=params)

        root, second, follower = node("root"), node("second"), node("follower")
        epoch_dir = os.path.join(tmp, "epochs")
        os.makedirs(epoch_dir)
        publisher = mp.MenuPublisher(netuid=netuid, store=store, held=root.book.held,
                                     lists=root.lists)

        def committed(j):
            return mp.committed_menu(epoch_dir, j)

        changed = 0
        for k in range(116, 124):
            chain.block = k * bpe
            reads = {n: x.at_boundary(k) for n, x in (("root", root), ("second", second),
                                                       ("follower", follower))}
            chain.block = k * bpe + 5
            rp.publish_due_reveals(epoch_dir, k, 1, netuid, Uploader(store), commit=chain.commit,
                                   menu=publisher)
            fields = mp.epoch_reveal_fields(
                root, reads["root"], k=k, inputs=inputs(k, reads["root"].state, perturb),
                root=True, committed_menu_of=committed)
            folder = os.path.join(epoch_dir, epoch_id_for_block(k))
            os.makedirs(folder)
            with open(os.path.join(folder, "reveal.json"), "w", encoding="utf-8") as f:
                json.dump({"epoch_id": epoch_id_for_block(k), "protocol_version": 4, **fields}, f,
                          indent=2, sort_keys=True)
            other = second.root_step(k, inputs(k, reads["second"].state, not perturb), committed)
            if other is None or ms.canonical_json(other[0].state).decode() != fields[mp.REVEAL_NEXT]:
                raise SystemExit(f"menu_publication: two validators derived different menus at "
                                 f"{epoch_id_for_block(k)}")
            audit = mp.epoch_reveal_fields(
                follower, reads["follower"], k=k,
                inputs=inputs(k, reads["follower"].state, not perturb), root=False,
                committed_menu_of=lambda j: None).get(mp.REVEAL_AUDIT)
            if audit is not None and audit["differences"] not in (None, []):
                raise SystemExit(f"menu_publication: the audit of an honest root at "
                                 f"{epoch_id_for_block(k)} is {audit['differences']}")
            changed += fields[mp.REVEAL_NEXT_HASH] != ms.menu_hash(ms.carry(other[1]))
            out.append({
                "epoch": k, "next": fields[mp.REVEAL_NEXT_HASH],
                "next_bytes": hashlib.sha256(fields[mp.REVEAL_NEXT].encode()).hexdigest(),
                "events": fields[mp.REVEAL_EVENTS], "commitments": [d for _, d in chain.commits],
                "ran_on": {n: (r.status, r.menu_hash) for n, r in sorted(reads.items())},
                "audit": audit,
            })
        audited = [r["audit"] for r in out if r["audit"] and r["audit"]["differences"] == []]
        if changed < 2 or len(audited) < 4 or any(
                len({h for _, h in r["ran_on"].values()}) != 1 for r in out):
            raise SystemExit("the menu_publication stage no longer exercises a changed menu, "
                             "an audit and one menu on every node")
    return out


def _head_odds_v4(seed: int) -> dict:
    """Protocol v4 heads (head_eval.load_head_v4, head_odds.picks_and_odds_v4,
    pair_odds_v4): heads with no token means, loaded from bytes, on the pinned
    seed menu and on two menus derived from it (one entry's request updated
    and the most-picked entry made dear; every price zero, so equal odds tie).
    The heads have rows for every entry plus ids not on the menu, rows for
    some entries only, large cancelling weights, and one constant row
    repeated (ties). Odds are digested as published (int16, base64), picks as
    entry ids.

    Under --perturb each head's rows are stored in reverse order and the
    questions are fed in reverse (the results put back in question order),
    with another hash seed: a computation that read row order, question
    order or dict order would show here.
    """
    import dataclasses
    import io

    import numpy as np

    from fugal_subnet import head_eval, head_odds, menu_state
    from fugal_subnet.v2_epoch import odds_b64
    from fugal_subnet.vendor import success_contract as contract

    perturb = bool(os.environ.get("FUGAL_PERTURB"))
    rng = np.random.default_rng(seed + 4)
    state = menu_state.seed_state(1000)
    ids = list(state.ids())
    dim = 1024

    def head_bytes(models, W, b) -> bytes:
        order = list(range(len(models)))[::-1] if perturb else list(range(len(models)))
        z = dict(W=np.asarray(W, dtype=np.float32)[order],
                 b=np.asarray(b, dtype=np.float32)[order],
                 models=np.asarray([models[i] for i in order]),
                 contract=np.asarray(contract.CONTRACT), profile_id=np.asarray(contract.PROFILE_ID),
                 backbone_revision=np.asarray(contract.REVISION),
                 provenance=np.asarray("determinism-v4"), lam=np.asarray(1.0))
        buf = io.BytesIO()
        np.savez(buf, **z)
        return buf.getvalue()

    extra = [f"lab-x/not-on-menu-{i}" for i in range(4)]
    some = sorted(rng.choice(ids, size=40, replace=False).tolist())
    heads = {
        "full": head_bytes(ids + extra, rng.standard_normal((len(ids) + 4, dim)) * 1.5,
                           rng.standard_normal(len(ids) + 4)),
        "gappy": head_bytes(some, rng.standard_normal((len(some), dim)) * 0.8,
                            rng.standard_normal(len(some))),
        "loud": head_bytes(ids, rng.standard_normal((len(ids), dim)) * 3000.0,
                           rng.standard_normal(len(ids))),
        "flat": head_bytes(ids, np.zeros((len(ids), dim)), np.full(len(ids), 0.25)),
    }
    loaded = {n: head_eval.load_head_v4(raw) for n, raw in sorted(heads.items())}
    hidden = rng.standard_normal((48, dim)).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)

    def run(head, menu):
        order = list(range(len(hidden)))[::-1] if perturb else list(range(len(hidden)))
        got = head_odds.picks_and_odds_v4(head, menu, hidden[order])
        back = np.argsort(order)
        return {"odds": odds_b64(got.odds[back]), "picks": [got.picks[i] for i in back]}

    base = {n: run(h, state) for n, h in loaded.items()}
    counts: dict[str, int] = {}
    for p in base["full"]["picks"]:
        counts[p] = counts.get(p, 0) + 1
    top = min(counts, key=lambda e: (-counts[e], e))
    upd = ids[0] if ids[0] != top else ids[1]

    def changed(e):
        if e.id == top:
            return dataclasses.replace(e, price_out=e.price_out * 1000)
        if e.id == upd:
            return dataclasses.replace(e, rev=e.rev + 1, explore_pairs=0)
        return e

    moved = dataclasses.replace(state, entries=tuple(changed(e) for e in state.entries))
    free = dataclasses.replace(state, entries=tuple(
        dataclasses.replace(e, price_in=0, price_out=0) for e in state.entries))
    for menu in (moved, free):
        menu_state.validate(menu)
    out = {"base": base,
           "moved": {n: run(h, moved) for n, h in loaded.items()},
           "free": {n: run(h, free) for n, h in loaded.items()}}

    pair_q = rng.integers(0, len(hidden), 120)
    pair_e = [ids[int(i)] for i in rng.integers(0, len(ids), 120)]
    out["pairs"] = {n: odds_b64(head_odds.pair_odds_v4(h, state, [hidden[q] for q in pair_q],
                                                       pair_e))
                    for n, h in loaded.items()}

    gappy = head_odds.picks_and_odds_v4(loaded["gappy"], state, hidden[:1]).odds
    if not ((gappy == -1).any() and out["free"]["flat"]["picks"] == [ids[0]] * len(hidden)
            and out["moved"]["full"]["picks"] != base["full"]["picks"]
            and out["moved"]["full"]["odds"] == base["full"]["odds"]
            and top not in out["moved"]["full"]["picks"]):
        raise SystemExit("the head_odds_v4 stage no longer exercises a missing row, a tie, "
                         "a price that moves a pick and a request update that keeps a row")
    return out


def _pool_mix_v4(seed: int) -> dict:
    """Protocol v4's question draw on the pinned mix (epoch_plan.mix_v4,
    LIVE_MENU_V4.md section 10) through the one plan constructor
    (epoch_plan.plan_v4): on a pool of the v3 manifest's sets at their
    pinned sizes (so the cycle caps are the real ones), for three epochs,
    the scored slice and two miners' exploration on the pinned seed menu
    (no proof measures a proposal: FAST_CROWN_V4.md 18).

    Under --perturb the pool arrives reversed and the process has another
    hash seed, so a draw that read arrival order or dict order would show
    here."""
    from fugal_subnet import config, epoch_plan, menu_state

    manifest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fugal_subnet",
                            "benchmarks", "pool_manifest_v3.json")
    with open(manifest, encoding="utf-8") as f:
        per = json.load(f)["per_benchmark"]
    pool = [{"question_id": f"{b}_{i:05d}", "benchmark": b, "prompt": f"q {b} {i}"}
            for b in sorted(per) for i in range(per[b]["n"])]
    if os.environ.get("FUGAL_PERTURB"):
        pool = list(reversed(pool))
    mix = epoch_plan.mix_v4()
    out = {"targets": sorted(mix.targets.items()), "weights": sorted(mix.bench_weight.items())}
    plan = epoch_plan.plan_v4(pool)
    state = menu_state.seed_state(1000)
    for k in range(3):
        nonce = hashlib.sha256(f"v4-mix-{seed}-{k}".encode()).digest()
        scored = [q["question_id"] for q in plan.scored_slice(nonce, pool, config.SLICE_SIZE_V4)]
        explored = {hk: sorted(plan.exploration(nonce, pool, set(scored), state, hk).items())
                    for hk in ("5Fhotkey-a", "5Fhotkey-b")}
        out[f"epoch_{k}"] = {"scored": sorted(scored), "explored": explored}
    return out


def _bound_v4(seed: int) -> list:
    """Protocol v4's one bound (seat_bound.lower_bound, SEAT_MINIMAL_V4.md
    3.1) on adversarial integer items: seeded draws of gains in up to seven
    categories with margins on both sides of the bound itself (its rounded
    value and one unit above), a mean exactly at the margin, no spread,
    single-label categories, values far past 64 bits, all-negative gains
    under a negative margin, a count one short of the minimum, and the same
    items kept as per-epoch sums of a sliding window.

    Under --perturb every category arrives in reverse order with its items
    reversed and the process has another hash seed, so a bound that read
    either order would show here."""
    import random as _random

    from fugal_subnet.seat_bound import Stratum, bound_from_strata, lower_bound, strata

    perturb = bool(os.environ.get("FUGAL_PERTURB"))

    def arrange(items: dict) -> dict:
        if not perturb:
            return items
        return {k: list(reversed(v)) for k, v in reversed(list(items.items()))}

    def row(items: dict, margin: int, n_min: int, n0: int) -> list:
        r = lower_bound(arrange(items), margin_units=margin, n_min=n_min, n0=n0)
        return [r.n, r.total, str(r.variance), r.lower, r.passes]

    rng = _random.Random(f"bound-v4-{seed}")
    unit = 1_000_000
    out = []
    boundary = []
    for k in range(40):
        items = {f"cat{c}": [int(rng.gauss(rng.choice([0, unit, -unit // 2]), 8 * unit))
                             for _ in range(rng.randint(0, 400))]
                 for c in range(rng.randint(1, 7))}
        n0 = rng.choice([1, 100, 1000, 1500])
        first = lower_bound(items, margin_units=0, n_min=0, n0=n0)
        if first.lower is None:
            out.append(row(items, 0, 0, n0))
            continue
        at, above = row(items, first.lower, 0, n0), row(items, first.lower + 1, 0, n0)
        boundary.append((at[4], above[4]))
        out += [at, above, row(items, 100_000, 1500, n0), row(items, -1_500_000, 1000, n0)]
    n = 2501
    xs = [int(rng.gauss(0, 8 * unit)) + 100_000 for _ in range(n - 1)]
    xs.append(100_000 * n - sum(xs))
    at_margin = row({"a": xs[:1000], "b": xs[1000:]}, 100_000, 1500, 1500)
    out.append(at_margin)
    out.append(row({"a": [250_000] * 2000, "b": [-50_000] * 1000}, 100_000, 1500, 1500))
    out.append(row({f"s{i}": [rng.randint(-unit, unit)] for i in range(30)}, 0, 1, 1))
    big = 10**30
    out.append(row({"a": [big + rng.randint(-10**20, 10**20) for _ in range(50)]},
                   big - 10**19, 1, 50))
    out.append(row({"a": [-2 * unit + rng.randint(0, 9000) for _ in range(3000)]},
                   -3 * unit, 1500, 1500))
    short = row({"a": [50 * unit] * 1499}, 100_000, 1500, 1500)
    out.append(short)
    # a sliding window of per-epoch sums is the bound of its items
    epochs = [{c: [int(rng.gauss(unit // 4, 8 * unit)) for _ in range(300)]
               for c in ("hard", "code", "math")} for _ in range(8)]
    zero = Stratum(0, 0, 0)
    window: dict = {}
    for i, e in enumerate(epochs):
        for c, s in strata(e).items():
            window[c] = window.get(c, zero) + s
        if i >= 5:
            for c, s in strata(epochs[i - 5]).items():
                window[c] = window[c] - s
            held = {c: [x for ep in epochs[i - 4:i + 1] for x in ep[c]] for c in window}
            by_sums = bound_from_strata(dict(reversed(list(window.items()))) if perturb else window,
                                        margin_units=100_000, n_min=1500, n0=1500)
            if [by_sums.n, by_sums.total, str(by_sums.variance), by_sums.lower,
                    by_sums.passes] != row(held, 100_000, 1500, 1500):
                raise SystemExit("the bound_v4 stage: a window of sums disagrees with its items")
            out.append(row(held, 100_000, 1500, 1500))
    verdicts = {r[4] for r in out}
    if (verdicts != {True, False} or at_margin[4] or short[4]
            or not any(a and not b for a, b in boundary)):
        raise SystemExit("the bound_v4 stage no longer exercises both verdicts, a margin on the "
                         "bound itself, a mean at the margin and a count short of the minimum")
    return out


def _picks_v4(seed: int) -> dict:
    """Protocol v4's real-picks check as a one-epoch challenge (rule B,
    picks_v4; FAST_CROWN_V4.md 2) over twelve epochs on the seed menu: five
    operators, one with three hotkeys; a holder that changes mid-run; a
    challenger every epoch from the third; one in alternate epochs (its run
    restarts at each miss); challenges of 500 and 1,000 questions; failed
    calls on either side (failing closed), questions the holder has no pick
    for and picks that agree; junk exploration past the failure bound in a
    normal epoch and an outage epoch that waives it; and a pick-cost table
    that refreshes mid-run, so later epochs are priced at the new table.
    Every record, every head's evidence against the holder and against no
    bar, the per-entry means and every published block are digested; under
    --perturb every proof list, every proof's results and the records arrive
    in reverse order."""
    import dataclasses
    import random

    from fugal_subnet import menu_inputs
    from fugal_subnet import menu_state as ms
    from fugal_subnet import picks_v4 as pv
    from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult

    perturb = bool(os.environ.get("FUGAL_PERTURB"))

    def fed(items):
        items = list(items)
        return list(reversed(items)) if perturb else items

    rnd = random.Random(seed * 104729 + 7)
    seed_menu = ms.seed_state(2000)
    entries = list(seed_menu.ids())[:10]
    cats = ["hard", "code", "math", "knowledge"]
    refreshed = dataclasses.replace(seed_menu, entries=tuple(
        dataclasses.replace(e, pick=ms.pick_table(
            (ms.PickCell("hard", 40, 9_000, 150_000), ms.PickCell("code", 25, 2_000, 30_000)),
            e, ms.READER_PARAMS)) if e.id in entries[:4] else e
        for e in seed_menu.entries))
    hotkeys = {"ck-a": ["hold"], "ck-b": ["chal"], "ck-c": ["alt"], "ck-d": ["odds"],
               "ck-e": ["e1", "e2", "e3"]}
    heads = {hk: f"{i:02x}" * 32 for i, hk in enumerate(
        sorted(h for hs in hotkeys.values() for h in hs))}
    listed = 1000
    records, blocks, verdicts, prices, means = {}, [], [], [], []
    epochs = range(2001, 2013)
    for k in epochs:
        state = dataclasses.replace(refreshed if k >= 2007 else seed_menu, epoch=k)
        qids = [f"pv-{k}-{i:04d}" for i in range(listed)]
        categories = [cats[i % len(cats)] for i in range(listed)]
        outage_epoch = k == 2010
        holder = heads["hold"] if k < 2008 else heads["e1"]
        size = {"chal": 1000 if k % 3 else 500, "alt": 500, "e2": 500}
        challenging = {"chal": k >= 2003, "alt": k % 2 == 0, "e2": k != 2006}
        hps = [None if i % 11 == 0 else entries[(i * 7 + k) % len(entries)] for i in range(listed)]
        proofs, values, epoch_means = [], {}, {}
        for ck, hks in sorted(hotkeys.items()):
            for hk in hks:
                junk = hk == "e3" and k == 2004
                rate = 0.5 if junk or outage_epoch else 0.03
                results = [QuestionResult(f"pv-x-{k}-{hk}-{i}", entries[i % len(entries)], False,
                                          0.0, "ab" * 32, is_exploration=True,
                                          failed=rnd.random() < rate or (junk and i < 12))
                           for i in range(30)]
                proofs.append(menu_inputs.VerifiedProof(BenchmarkProof(
                    epoch_id=f"e{k:08d}", nonce="00" * 32, questions_hash="", source_hash="",
                    weights_hash=heads[hk], results=fed(results), total_cost_usd=0.0,
                    per_model_costs={}, attestation_quote=b"", timestamp=0.0, hotkey=hk,
                    routing_protocol="v4", menu_hash="cd" * 32, endpoint_reads=(),
                    scored=False), ck))
                if not challenging.get(hk):
                    continue
                n = size[hk]
                own = []
                held = {}
                for i in range(n):
                    mine = hps[i] if i % 5 == 0 and hps[i] else entries[(i * (3 + len(hk))) % 10]
                    f = rnd.random() < 0.04
                    own.append(pv.Answer(mine, not f and rnd.random() < 0.6, f))
                    if hps[i] is not None and hps[i] != mine:
                        g = rnd.random() < 0.04
                        held[i] = pv.Answer(hps[i], not g and rnd.random() < 0.6, g)
                items = pv.challenge_items(state, categories[:n], own,
                                           hps[:n], dict(fed(held.items())))
                values[heads[hk]] = items
                for e, (c, t) in pv.item_means(items, [a.entry for a in own], hps[:n]).items():
                    acc = epoch_means.setdefault(e, [0, 0])
                    acc[0] += c
                    acc[1] += t
        verdict = pv.failure_refusals(fed(proofs), state)
        verdicts.append([k, verdict.outage, sorted(verdict.refused.items())])
        rec = pv.PicksEpoch(k, tuple(categories), dict(fed(values.items())), verdict.outage, holder)
        records[k] = rec
        held_records = dict(fed(records.items()))
        evidence = {}
        for hk in sorted(heads):
            h = heads[hk]
            vs_holder = pv.picks_check(held_records, h, holder, through=k)
            alone = pv.picks_check(held_records, h, None, through=k)
            evidence[h] = pv.evidence_to_reveal(vs_holder, answered_now=pv.answered_in(
                held_records, h, k))
            verdicts.append([k, hk, vs_holder.run, vs_holder.paired,
                             str(vs_holder.bound.variance) if vs_holder.bound else None,
                             vs_holder.lower, vs_holder.passes, alone.passes])
        means.append([k, sorted(epoch_means.items())])
        block = pv.reveal_block(rec, evidence, verdict.refused, questions=qids,
                                means={e: tuple(v) for e, v in epoch_means.items()})
        blocks.append(pv.digest(block))
        back = pv.epoch_from_reveal(json.loads(json.dumps(block)), k)
        if back != rec:
            raise SystemExit("the picks_v4 stage: a published record does not read back")
        prices.append([k, str(ms.pick_cost(state, entries[0], "hard"))])
    refused_any = any(v[2] for v in verdicts if len(v) == 3)
    run_alt = pv.answered_run(records, heads["alt"], 2012)
    if not (refused_any and any(v[1] for v in verdicts if len(v) == 3)
            and run_alt == (2012,) and prices[0][1] != prices[-1][1]
            and heads["e2"] not in records[2006].values
            and len(records[2003].values[heads["chal"]]) == 1000
            and len(records[2004].values[heads["chal"]]) == 500):
        raise SystemExit("the picks_v4 stage no longer exercises a refused proof, an outage, a "
                         "restarted run, a refreshed table, both sizes and a missed challenge")
    return {"records": [[e, list(r.categories), sorted(r.values.items()), r.outage, r.holder]
                        for e, r in sorted(records.items())],
            "verdicts": verdicts, "blocks": blocks, "prices": prices, "means": means}
def _memo_v4(seed: int) -> dict:
    """Protocol v4's memorisation check and its reserve (memo_v4,
    SEAT_MINIMAL_V4.md 6): on a pool of the v3 manifest's sets at their
    pinned sizes, a published set folded from six reveals live and rebuilt
    from them (the two must agree), the digest of the view an epoch reads,
    the epoch's plan through plan_v4 with the published set (the reserve's
    size, F, the fresh questions of two epochs, the scored slice, two
    miners' exploration), and on a window of
    twenty epochs of published and fresh labels on two benchmarks the
    evidence of an honest head, a memoriser and a head skilled on one
    benchmark only (each part's counts, the exact gap, the bound's sums and
    the verdict).

    Under --perturb the pool, every reveal's rows, the reveals themselves
    and every epoch's labels arrive reversed, the window's epochs and heads
    in reverse insertion order, and the process has another hash seed."""
    import numpy as np

    from fugal_subnet import config, epoch_plan, labels_v4, memo_v4, menu_state
    from fugal_subnet.champion import EpochPairs, Window

    perturb = bool(os.environ.get("FUGAL_PERTURB"))

    def arrive(xs):
        return list(reversed(xs)) if perturb else list(xs)

    def u(*parts) -> int:
        return int.from_bytes(hashlib.sha256(":".join(map(str, (seed,) + parts)).encode())
                              .digest()[:8], "big")

    manifest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fugal_subnet",
                            "benchmarks", "pool_manifest_v3.json")
    with open(manifest, encoding="utf-8") as f:
        per = json.load(f)["per_benchmark"]
    pool = arrive([{"question_id": f"{b}_{i:05d}", "benchmark": b, "prompt": f"q {b} {i}"}
                   for b in sorted(per) for i in range(per[b]["n"])])
    epoch = 1000
    start = memo_v4.Published({q["question_id"]: 1 + u("p", q["question_id"]) % 990
                               for q in pool if u("s", q["question_id"]) % 3}, epoch - 4)
    reveals = []
    ids = sorted(q["question_id"] for q in pool)
    for e in range(epoch - 3, epoch + 3):
        picked = [ids[u("r", e, k) % len(ids)] for k in range(400)]
        reveals.append({"epoch_id": f"e{e:08d}",
                        "questions": arrive([{"question_id": q} for q in picked[:150]]),
                        "exploration": arrive([{"question_id": q, "failed": bool(k % 7 == 0)}
                                               for k, q in enumerate(picked[150:])])})
    live = start
    for r in reveals:
        live = memo_v4.fold_reveal(live, r)
    rebuilt = memo_v4.rebuild(start, arrive(reveals))
    if rebuilt != live:
        raise SystemExit("the memo_v4 stage: a rebuild from the reveals disagrees with the live fold")
    out: dict = {"views": {str(t): live.view(t).digest for t in range(epoch - 4, epoch + 3)},
                 "digest": live.digest}
    targets = epoch_plan.mix_v4().targets
    state = menu_state.seed_state(epoch + 1)
    for e in (epoch, epoch + 1):
        p = epoch_plan.plan_v4(pool, published=live, epoch=e)
        nonce = hashlib.sha256(f"memo-v4-{seed}-{e}".encode()).digest()
        fresh = sorted(q["question_id"] for q in memo_v4.fresh_questions(nonce, p.pool_index, targets))
        if e == epoch + 1:
            scored = [q["question_id"] for q in p.scored_slice(nonce, pool, config.SLICE_SIZE_V4)]
            explored = {hk: sorted(p.exploration(nonce, pool, set(scored), state, hk).items())
                        for hk in ("5Fhotkey-a", "5Fhotkey-b")}
            out["plan"] = {"scored": sorted(scored), "explored": explored}
        out[f"reserve_{e}"] = [p.pool_index.reserve_size, p.pool_index.fresh_count,
                               p.pool_index.digest, fresh]
    # the gap: published and fresh labels on two benchmarks
    entries = [f"m/{i}" for i in range(6)]
    operators = [f"ck{i}" for i in range(6)]
    pairs, first = {}, {}
    for e in range(200, 220):
        rows = []
        for i in range(90):
            qid = f"s{e}_{i}"
            first[qid] = 40 + u("f", qid) % 100
            rows.append((qid, entries[u("m", qid) % 6], "easy" if i % 3 else "hard",
                         operators[i % 6]))
        for j in range(5):
            qid = f"f{e}_{j}"
            first[qid] = e
            rows += [(qid, entries[k], "easy" if j % 2 else "hard", operators[k]) for k in range(6)]
        rows = arrive(rows)
        y = [10000 if u("y", q, m) % 100 < (75 if b == "easy" else 35) else 0 for q, m, b, _ in rows]
        n = len(rows)
        base = EpochPairs(question_ids=tuple(r[0] for r in rows), models=tuple(r[1] for r in rows),
                          benchmarks=tuple(r[2] for r in rows), categories=("knowledge",) * n,
                          y=np.array(y, np.int32), prompt_tokens=np.zeros(n, np.int32),
                          completion_tokens=np.zeros(n, np.int32),
                          weight=np.array([500_000 + u("w", r[0], r[1]) % 2_000_000 for r in rows],
                                          np.int64))
        pairs[e] = labels_v4.label_pairs(base, [(r[3],) for r in rows], np.full(n, -1, np.int16))
    published = memo_v4.Published(first, 219)
    knows = {"honest": lambda q, b: 35, "memoriser": lambda q, b: 60 if q.startswith("s") else 0,
             "by_benchmark": lambda q, b: 70 if b == "easy" else 0}
    odds = {}
    for h in arrive(sorted(knows)):
        odds[h] = {}
        for e in arrive(sorted(pairs)):
            p = pairs[e]
            odds[h][e] = np.array([(9000 if yy else 1000) if u("k", h, q, m) % 100 < knows[h](q, b)
                                   else (7500 if b == "easy" else 3500)
                                   for q, m, b, yy in zip(p.question_ids, p.models, p.benchmarks,
                                                          p.y.tolist())], np.int16)
    window = Window(size=48, slices={}, heads={},
                    pairs={e: pairs[e] for e in arrive(sorted(pairs))}, odds=odds)
    ref = labels_v4.reference(window, now=219)
    for h in sorted(knows):
        ev = memo_v4.memo_evidence(window, ref, labels_v4.Side(h, "ckH", 150), published, now=219)
        out[h] = {**memo_v4.evidence_for_reveal(ev), "gap": str(ev.gap),
                  "bound": [ev.bound.n, ev.bound.total, str(ev.bound.variance), ev.bound.lower,
                            ev.bound.passes],
                  "parts": {b: [s.labels, s.questions, s.total, v.labels, v.questions, v.total]
                            for b, (s, v) in sorted(ev.benchmarks.items())}}
    if not out["memoriser"]["refused"] or out["honest"]["refused"] or out["by_benchmark"]["refused"]:
        raise SystemExit("the memo_v4 stage no longer refuses its memoriser and only it")
    # The memoriser as a succession chain: its odds stored under five heads
    # of one line (each the line's head for four epochs, the newest from
    # 216), read through the newest's predecessors (labels_v4.predecessor_odds).
    line = [f"memoriser-{k}" for k in range(5)]                 # oldest first
    starts = [150, 204, 208, 212, 216]
    chain_odds = {}
    for k in arrive(range(5)):
        until = starts[k + 1] if k < 4 else 220
        chain_odds[line[k]] = {e: odds["memoriser"][e] for e in arrive(sorted(pairs))
                               if starts[k] <= e < until or (k == 0 and e < starts[1])}
    side = labels_v4.Side(line[4], "ckH", 216, {m: 150 for m in entries},
                          chain=tuple((line[k], starts[k + 1]) for k in (3, 2, 1, 0)))
    chained = Window(size=48, slices={}, heads={},
                     pairs={e: pairs[e] for e in arrive(sorted(pairs))}, odds=chain_odds)
    ev = memo_v4.memo_evidence(chained, labels_v4.reference(chained, now=219), side, published,
                               now=219)
    out["memoriser_line"] = memo_v4.evidence_for_reveal(ev)
    if out["memoriser_line"] != {k: v for k, v in out["memoriser"].items()
                                 if k not in ("gap", "bound", "parts")}:
        raise SystemExit("the memo_v4 stage: a successor read through its predecessors no longer "
                         "has its predecessors' evidence")
    return out


def _v4_replay(seed: int) -> dict:
    """Protocol v4's re-anchor replay (fugal_subnet/v4_replay.py;
    "Verify-and-resync" in docs/INVARIANTS.md): the seat_v4 stage's twenty
    live epochs (`_seat_v4_run`) with heads that load as v4 heads, the slice
    priced at each epoch's pick-cost table and, from epoch MEMO_FROM, each
    epoch's memorisation memo; each epoch's inputs kept as a validator keeps
    them (v4_replay.record, Store.keep). Then, from the live state after each
    anchor epoch k, the epochs k+1 .. 20 run again from the records
    (v4_replay.replay, each epoch's memo from the same function of the
    epoch): the state each replay ends in must be the live run's, byte for
    byte (SystemExit otherwise). The epochs carry an entry that joins, one
    whose allowance ends, one that leaves inside it, heads that miss epochs
    and get their late odds, a head that never answers the scored questions,
    the seat's events and the memorisation check over every kept head, so the
    replay runs the labels, odds, seat and memorisation code a live epoch
    runs. Digested: each record, each live epoch's consensus fields, and per
    anchor the replayed state (window, seat, copy rule) and its events.

    Under --perturb the live epochs' inputs arrive in reverse order (as in
    the seat_v4 stage) and the process has another hash seed.
    """
    import tempfile

    import numpy as np

    from fugal_subnet import epoch_plan, memo_v4, v4_replay
    from fugal_subnet.v2_epoch import load_window, save_window, seat_to_dict, window_arrays

    # The fill comes at e5 under the fast crown: the second anchor precedes it.
    memo_from, anchors, last = 12, (2, 3, 15), 20
    mix = epoch_plan.mix_v4()
    pool = [{"question_id": f"{s}_{i:03d}", "benchmark": s, "prompt": "x"}
            for s in sorted(mix.bench_category) for i in range(40)]
    first: dict[str, int] = {}
    bench_of: dict[str, str] = {}
    memos: dict = {}

    def memo_for(e):
        # The published set as of the epoch before: every even label
        # question long before the window, every odd one in its own epoch.
        if e < memo_from:
            return None
        if e not in memos:
            pub = memo_v4.Published({q: f for q, f in sorted(first.items()) if f < e}, e - 1)
            memos[e] = memo_v4.epoch_memo(
                pub, memo_v4.reserve_index(pub, pool, e),
                hashlib.sha256(f"v4-replay-memo-{seed}-{e}".encode()).digest(), mix.targets)
        return memos[e]

    def state_digest(window, seat, copy_state) -> str:
        # The window's logical contents (v2_epoch.window_arrays: the arrays
        # the v1-v3 file holds), whatever layout its file has; the file must
        # read back to exactly those contents.
        h = hashlib.sha256()
        logical = window_arrays(window)
        with tempfile.TemporaryDirectory() as d:
            save_window(os.path.join(d, "w.npz"), window)
            back = window_arrays(load_window(os.path.join(d, "w.npz"), window.size))
        if set(back) != set(logical) or any(
                back[k].dtype != v.dtype or back[k].shape != v.shape
                or back[k].tobytes() != v.tobytes() for k, v in logical.items()):
            raise SystemExit("v4_replay: the window file does not read back to the window")
        for key in sorted(logical):
            h.update(key.encode() + np.ascontiguousarray(logical[key]).tobytes())
        h.update(json.dumps([seat_to_dict(seat), copy_state.to_reveal()],
                            sort_keys=True).encode())
        return h.hexdigest()

    with tempfile.TemporaryDirectory() as d:
        store = v4_replay.Store(d)
        live, records, events = {}, [], []

        def keep(epoch, kw, head_bytes, result):
            doc, blobs = v4_replay.record(kw, head_bytes)
            store.keep(doc, blobs, keep_epochs=last + 2)
            records.append(hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest())
            oi = kw["odds_input"]
            for a in oi.answers:
                bench_of[a.question_id] = oi.benchmark_of[a.question_id]
                first.setdefault(a.question_id, 0 if int(a.question_id.rsplit("_", 1)[1]) % 2 == 0
                                 else epoch)
            live[epoch] = (result.window, result.decision.state, result.copy_state)
            events.append(result.decision.event)

        run = _seat_v4_run(seed, dim=1024, spend=True, memo_for=memo_for, keep=keep,
                           oracle=True)
        end = state_digest(*live[last])
        replayed = {}
        for k in anchors:
            window, seat, copy_state = live[k]
            got = v4_replay.replay(store, window=window, seat=seat, copy_state=copy_state,
                                   rules=None, epochs=range(k + 1, last + 1),
                                   hidden_of=run.hidden_of, benchmark_of=dict(bench_of),
                                   memo_for=lambda e, _nonce: memo_for(e), max_span=48)
            replayed[str(k)] = state_digest(got.window, got.seat, got.copy_state)
            if replayed[str(k)] != end or got.epochs != tuple(range(k + 1, last + 1)):
                raise SystemExit(f"the v4 replay from e{k} does not reach the live state at e{last}")
    late = [o["late_odds"] for o in run.out]
    if not (len(memos) == last - memo_from + 1 and events.index("fill") > anchors[1]
            and late[8].get(run.hash_of["no-new"]) and late[12].get(run.hash_of["base-only"])):
        raise SystemExit("the v4_replay stage no longer replays the memo, late odds and a seat "
                         f"event: {events}")
    return {"records": records, "epochs": run.out, "events": events, "state": end,
            "replayed": replayed}


def _published_v4(seed: int) -> dict:
    """Protocol v4's published set as every node advances it (memo_sync.
    ChainSet, SEAT_MINIMAL_V4.md 6): on a pool of the v3 manifest's sets
    (sixty questions each), a seed with a third of it unpublished and a root
    that commits on time, is down for four epochs, commits its backlog late,
    once commits an older reveal, and once fails to commit. A node that read
    every boundary, one restarted from its kept state halfway, and one with
    no state that anchors on the snapshot the newest reveal names must hold
    one set, as of every epoch; the stage publishes every epoch's view
    digest, the snapshot's sha256 and the last epoch's fresh draw on it.

    Under --perturb the pool and every reveal's rows arrive reversed, the
    reveals are offered in reverse order, and the process has another hash
    seed."""
    from fugal_subnet import epoch_plan, memo_sync, memo_v4
    from fugal_subnet.reveal_publisher import commitment_for

    perturb = bool(os.environ.get("FUGAL_PERTURB"))

    def arrive(xs):
        return list(reversed(xs)) if perturb else list(xs)

    def u(*parts) -> int:
        return int.from_bytes(hashlib.sha256(":".join(map(str, (seed,) + parts)).encode())
                              .digest()[:8], "big")

    manifest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fugal_subnet",
                            "benchmarks", "pool_manifest_v3.json")
    with open(manifest, encoding="utf-8") as f:
        sets = sorted(json.load(f)["per_benchmark"])
    pool = arrive([{"question_id": f"{b}_{i:03d}", "benchmark": b} for b in sets
                   for i in range(60)])
    s0 = 999
    unpublished = {}
    for q in pool:
        if u("s", q["question_id"]) % 3 == 0:
            unpublished.setdefault(q["benchmark"], []).append(q["question_id"])
    doc = {"format": memo_v4.SEED_FORMAT, "through": f"e{s0:08d}",
           "pool": {"pool_hash": "stage", "n_questions": len(pool),
                    "per_set": {b: 60 for b in sets}},
           "observations_sha256": "0" * 64,
           "unpublished": {b: sorted(v) for b, v in sorted(unpublished.items())}}
    ids = sorted(q["question_id"] for q in pool)
    # Boundary -> the reveal the root's commitment names there: on time, then
    # down for 1006..1009, its backlog (1005) named late, an older reveal
    # (1002) committed at 1013, and 1014's commitment never made.
    at = {b: b - 2 for b in range(1002, 1007)}
    at.update({1012: 1005, 1013: 1002, 1014: 1011, 1015: 1013, 1017: 1015})
    last = 1018
    raw, shas = {}, {}

    def read_at(b):
        named = [c for bb, c in sorted(at.items()) if bb <= b]
        if not named:
            return None
        return commitment_for(f"e{named[-1]:08d}", shas[named[-1]])

    def reveal_bytes(eid, sha):
        return {f"e{c:08d}": r for c, r in arrive(sorted(raw.items()))}.get(eid)

    reference = memo_sync.ChainSet(pool=pool, read_at=read_at, reveal_bytes=reveal_bytes,
                                   seed_doc=doc)
    for c in sorted({c for c in at.values()} | {1012, 1016}):
        if c >= last:
            continue
        reference.advance_to(c)
        picked = [ids[u("r", c, k) % len(ids)] for k in range(40)]
        view = reference.view(c).view(c - 2)
        r = {"epoch_id": f"e{c:08d}", "questions": arrive([{"question_id": q} for q in picked]),
             "memo_v4": {"through": view.through, "published_digest": view.digest}}
        raw[c] = json.dumps(r, sort_keys=True).encode()
        shas[c] = hashlib.sha256(raw[c]).hexdigest()
    views = {}
    whole = memo_sync.ChainSet(pool=pool, read_at=read_at, reveal_bytes=reveal_bytes,
                               seed_doc=doc)
    for e in range(s0 + 2, last + 1):
        views[e] = whole.view_for(e).digest
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "held.json")
        half = memo_sync.ChainSet(pool=pool, read_at=read_at, reveal_bytes=reveal_bytes,
                                  path=path, seed_doc=doc)
        half.advance_to(1010)
        half.save()
        again = memo_sync.ChainSet(pool=pool, read_at=read_at, reveal_bytes=reveal_bytes,
                                   path=path, seed_doc=doc)
        restarted = again.view_for(last).digest
    named = json.loads(raw[1015])["memo_v4"]
    snap = memo_v4.snapshot_bytes(whole.view(1015).view(1013))
    fresh = memo_sync.ChainSet(pool=pool, read_at=read_at, reveal_bytes=reveal_bytes,
                               snapshot=lambda d: snap if d == named["published_digest"] else None,
                               seed_doc=doc)
    if {restarted, fresh.view_for(last).digest} != {views[last]} or any(
            whole.findings + fresh.findings):
        raise SystemExit("the published_v4 stage: the nodes' sets disagree")
    p = epoch_plan.plan_v4(pool, published=whole.view(last), epoch=last)
    nonce = hashlib.sha256(f"published-v4-{seed}".encode()).digest()
    return {"views": {str(e): v for e, v in sorted(views.items())},
            "snapshot_sha256": hashlib.sha256(snap).hexdigest(),
            "reserve": [p.pool_index.reserve_size, p.pool_index.fresh_count],
            "fresh": sorted(q["question_id"] for q in memo_v4.fresh_questions(
                nonce, p.pool_index, epoch_plan.mix_v4().targets))}


def _v4_digest_isolation() -> dict:
    """The v4 part of the digest moves alone: a validator whose one v4
    constant differs (here menu_state.MENU_LAG, a config._V4 value, a seat
    constant and a memorisation one) publishes another v4 digest and the
    same v1-v3 digest. Raises SystemExit when it does not
    (fugal_subnet/consensus_v4.py)."""
    from fugal_subnet import config, memo_v4, seat_v4
    from fugal_subnet import menu_state as ms
    from fugal_subnet.fingerprint import consensus_digest

    v3, v4 = consensus_digest(), consensus_digest(protocol_version=4)
    seen = []
    for mod, name in ((ms, "MENU_LAG"), (config, "SLICE_SIZE_V4"),
                      (seat_v4, "MAX_PREDECESSORS"), (memo_v4, "READ_LAG")):
        kept = getattr(mod, name)
        setattr(mod, name, kept + 1)
        try:
            seen.append((consensus_digest(), consensus_digest(protocol_version=4)))
        finally:
            setattr(mod, name, kept)
    if any(a != v3 or b == v4 for a, b in seen) or v3 == v4:
        raise SystemExit(f"a differing v4 constant must move the v4 digest and not the v3 one: "
                         f"v3 {v3}, v4 {v4}, with one v4 constant changed {seen}")
    return {"v3": v3, "v4": v4}


def _child(seed: int) -> dict:
    """Entry point for the subprocess: emit stage digests as JSON on stdout."""
    from fugal_subnet.fingerprint import consensus_digest

    result = run_pipeline(seed)
    result["_consensus_digest"] = consensus_digest()
    # A v4 epoch's digest (the v1-v3 material and the v4 section), and the
    # check that a differing v4 constant moves only it.
    result["_consensus_digest_v4"] = _v4_digest_isolation()["v4"]
    return result


def _exam_v5(seed: int) -> dict:
    """Protocol v5's exam (exam_v5, arena_score) over three epochs: a draw on
    a four-dataset mix, six contenders' picks over five entries (two always
    equal, one with off-menu and no picks), a ring of 13 hotkeys on 7
    coldkeys, B* on the first questions, a budget that stops the exam early in
    one epoch; then votes from assigned and unassigned hotkeys, failures, a
    double vote and fractional medians, read as paired items for every pair
    of contenders (both operators excluded) and against B*; plus
    arena_score's log2, logarithmic mean, k and items on edge values.

    Under --perturb every input collection arrives reversed (the datasets and
    their questions, the contenders, the menu, the ring, the votes and the
    specs) and the process has another hash seed."""
    import random as _random

    from fugal_subnet import arena_score, exam_v5

    perturb = bool(os.environ.get("FUGAL_PERTURB"))

    def seq(xs):
        return list(reversed(list(xs))) if perturb else list(xs)

    def dct(d):
        return dict(reversed(list(d.items()))) if perturb else dict(d)

    rng = _random.Random(f"exam-v5-{seed}")
    menu = ["lab/a@low", "lab/a@high", "lab/b", "lab/c", "lab/d"]
    by = {f"set{k}": [f"set{k}-q{i}" for i in range(400)] for k in range(4)}
    weights = {"set0": 500_000, "set1": 250_000, "set2": 150_000, "set3": 100_000}
    ring = [(f"5Hk{i:02d}", f"5Ck{i % 7}") for i in range(13)]
    owner = {f"head{h}": f"5Ck{h}" for h in range(6)}
    specs, spec_docs = [], {}
    budget = exam_v5.EXAM_CALLS_PER_HOTKEY
    for e in range(3):
        nonce = hashlib.sha256(f"exam-v5-{seed}-{e}".encode()).digest()
        pool = exam_v5.draw_exam(nonce, dct({k: seq(v) for k, v in by.items()}), dct(weights), 600)
        picks = {}
        for h in range(6):
            if h == 1:
                picks[f"head{h}"] = list(picks["head0"])
            elif h == 5:
                picks[f"head{h}"] = [rng.choice(menu + ["off/x", ""]) for _ in pool]
            else:
                picks[f"head{h}"] = [rng.choice(menu[:3] if rng.random() < 0.7 else menu)
                                     for _ in pool]
        exam_v5.EXAM_CALLS_PER_HOTKEY = 12 if e == 1 else budget
        try:
            spec = exam_v5.build_spec(26_600 + e, nonce, dct({h: list(p) for h, p in picks.items()}),
                                      seq(menu), pool, seq(ring), "lab/b")
        finally:
            exam_v5.EXAM_CALLS_PER_HOTKEY = budget
        specs.append(spec)
        spec_docs[f"epoch_{e}"] = {"hash": spec.spec_hash(), "json": spec.to_json(),
                                   "assigned": {hk: exam_v5.assigned_cells(spec, hk)
                                                for hk, _ in ring}}
    ck = dict(ring)
    votes = []
    for spec in specs:
        for cell in spec.cells:
            for hk in spec.assignment[cell]:
                votes.append(exam_v5.Vote(spec.epoch, cell[0], cell[1], hk, ck[hk],
                                          rng.choice([0, exam_v5.Y_SCALE, 3_333]),
                                          rng.randint(0, 2_000_000), rng.random() < 0.08))
            if rng.random() < 0.05:
                votes.append(exam_v5.Vote(spec.epoch, cell[0], cell[1], "5Hstranger", "5Cx", 0, 1,
                                          False))
    if votes:
        votes.append(votes[0])                     # a double vote: both copies drop
    cost = {e: 1_000 * (i + 1) ** 3 for i, e in enumerate(menu)}
    heads = sorted(owner)
    mean = {h: exam_v5.mean_pick_cost_nd(specs, h, cost) for h in heads}
    mean.update(cost)
    items = {}
    for x in heads:
        for y in heads:
            if x < y:
                got = exam_v5.comparison_items(seq(specs), seq(votes), x, y,
                                               frozenset({owner[x], owner[y]}), mean,
                                               lambda q: q.split("-")[0])
                items[f"{x}~{y}"] = {k: sorted(v) for k, v in sorted(got.items())}
        got = exam_v5.baseline_items(seq(specs), seq(votes), x, frozenset({owner[x]}), mean,
                                     lambda q: q.split("-")[0])
        items[f"{x}~B*"] = {k: sorted(v) for k, v in sorted(got.items())}
    if not any(v for d in items.values() for v in d.values()) or "head0~head1" not in items:
        raise SystemExit("the exam_v5 stage no longer produces paired items")
    edges = [str(arena_score.log2_fixed(n, d)) for n, d in
             ((1, 1), (3, 1), (500_000, 11), (7, 1024), (10**30 + 7, 3))]
    edges += [arena_score.log_mean_nd(a, b) for a, b in ((0, 0), (1, 10**9), (440_000, 1_200_000))]
    edges += [str(arena_score.k_pair(a, b)) for a, b in ((1, 1), (620_000, 1_204_000),
                                                         (10**12, 10**12))]
    edges += [arena_score.item_units(exam_v5.Y_SCALE, 0, 5, 7, arena_score.value_scale_nd(a, b))
              for a, b in ((620_000, 1_204_000), (1, 1), (10**12, 10**12))]
    return {"exam_spec_v5": spec_docs, "arena_items_v5": {"items": items, "edges": edges}}


def run_in_subprocess(seed: int, perturb: bool) -> dict:
    env = os.environ.copy()
    if perturb:
        # Stand in for "a validator on different hardware". Two levers:
        #
        #  1. Drop kernel-dispatch pinning so each library selects from the
        #     host CPU. On a single machine this changes nothing (same CPU) —
        #     it only bites when CI runners differ, which is the point.
        #  2. Unpin thread counts. This *does* bite on one machine: parallel
        #     reductions sum in nondeterministic order, so it exercises the
        #     same class of float divergence locally that differing CPUs cause
        #     in production. Without it, --perturb passes vacuously on a
        #     single runner.
        for key in (
            "ATEN_CPU_CAPABILITY", "MKL_CBWR", "DNNL_MAX_CPU_ISA",
            "OPENBLAS_CORETYPE",
        ):
            env.pop(key, None)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[key] = "4"
        env["FUGAL_ALLOW_ENV_DRIFT"] = "1"
        env["FUGAL_PERTURB"] = "1"
    env["PYTHONHASHSEED"] = "0" if not perturb else "12345"

    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", str(seed)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"determinism child process failed:\n{proc.stdout}\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--child", type=int, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=7, help="Epoch seed to replay")
    p.add_argument("--perturb", action="store_true",
                   help="Run the second process without CPU dispatch pinning")
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    args = p.parse_args()

    if args.child is not None:
        print(json.dumps(_child(args.child)))
        return 0

    mode = "perturbed (simulated second host)" if args.perturb else "same-host repeat"
    if not args.json:
        print(f"Differential determinism check — {mode}, seed={args.seed}")

    a = run_in_subprocess(args.seed, perturb=False)
    b = run_in_subprocess(args.seed, perturb=args.perturb)

    mismatches = [s for s in STAGES if a[s] != b[s]]
    report = {
        "mode": "perturb" if args.perturb else "repeat",
        "seed": args.seed,
        "consensus_digest_a": a["_consensus_digest"],
        "consensus_digest_b": b["_consensus_digest"],
        "consensus_digest_v4_a": a["_consensus_digest_v4"],
        "consensus_digest_v4_b": b["_consensus_digest_v4"],
        "stages": {s: {"a": a[s], "b": b[s], "match": a[s] == b[s]} for s in STAGES},
        "mismatched_stages": mismatches,
        "ok": not mismatches,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if not mismatches else 1

    for stage in STAGES:
        mark = "OK  " if a[stage] == b[stage] else "DIFF"
        print(f"  {mark} {stage:<20} {a[stage][:16]}")
    print()

    print(f"  consensus digest   v1-v3 {a['_consensus_digest']}  v4 {a['_consensus_digest_v4']}"
          " (a differing v4 constant moves only the v4 digest)")
    if not mismatches:
        print(f"All {len(STAGES)} stages identical. I1 holds for this run.")
        return 0

    first = mismatches[0]
    print(f"DIVERGENCE at '{first}' (and {len(mismatches) - 1} later stage(s)).")
    print("Later stages diverge because this one did — fix the earliest.")
    if args.perturb:
        print(
            "\nThis run stripped CPU dispatch pinning, so this stage depends on\n"
            "kernel selection. Two validators on different CPUs will disagree,\n"
            "which splits weights. Pin the dispatch or quantize the decision."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
