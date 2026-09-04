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
    "exploration",
    "hidden_states",
    "routing_decisions",
    "proof",
    "verification",
    "frame",
    "scores",
    "weights",
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
    """
    import hashlib

    import numpy as np

    from fugal_subnet.benchmarks.slicer import (
        derive_nonce,
        epoch_id_for_block,
        select_slice,
    )
    from fugal_subnet.config import HEAD_HIDDEN_DIM
    from fugal_subnet.exploration import expected_exploration
    from fugal_subnet.head_eval import HeadScore
    from fugal_subnet.reference_frame import (
        ReferenceFrame,
        accumulate_exploration,
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
        ))
        return text

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
        np.savez(buf, W=W, b=b, models=np.array(models, dtype="U100"))
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

    # The frame pools every verified proof's exploration samples. Iterated in
    # sorted uid order so the accumulation is order-independent by construction
    # as well as by arithmetic.
    frame = accumulate_exploration(ReferenceFrame(), [
        (r.routed_model, r.correct, r.prompt_tokens, r.completion_tokens)
        for uid in sorted(proof_objs)
        for r in proof_objs[uid].exploration_results
    ])
    ref_model, acc_best = best_model(frame, prices)

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
        ScoringState(), epoch_scores, head_hashes, acc_best=acc_best,
        hotkeys={uid: f"hk{uid}" for uid in epoch_scores},
        n_questions=len(questions), pool_size=len(pool),
    )
    uids, weight_values = compute_weights(state.records)

    scores_out = {
        uid: {
            "composite": float(rec.composite_score),
            "quality": float(rec.quality),
            "thrift": float(rec.thrift),
            "wilson_lcb": float(rec.wilson_lcb),
        }
        for uid, rec in state.records.items()
    }

    return {
        "slice": _digest(question_ids),
        "exploration": _digest(explore_map),
        "hidden_states": _digest(hidden),
        "routing_decisions": _digest(routing),
        "proof": _digest(proofs),
        "verification": _digest(verdicts),
        "frame": _digest(frame.to_dict()),
        "scores": _digest(scores_out),
        "weights": _digest({"uids": uids, "weights": weight_values}),
    }


def _child(seed: int) -> dict:
    """Entry point for the subprocess: emit stage digests as JSON on stdout."""
    from fugal_subnet.fingerprint import consensus_digest

    result = run_pipeline(seed)
    result["_consensus_digest"] = consensus_digest()
    return result


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
