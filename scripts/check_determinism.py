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
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stages are ordered so the first mismatch is the root cause: a differing slice
# explains a differing matrix, which explains differing scores. Reporting the
# earliest divergence points at the actual bug instead of its symptoms.
STAGES = (
    "slice",
    "hidden_states",
    "matrix",
    "soft_targets",
    "routing_decisions",
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
    """Run the scoring pipeline once and return a digest per stage.

    Mock mode throughout: no API spend, no chain, no network. The point is the
    arithmetic, which is identical in mock and live — the matrix is synthesized
    rather than fetched, but every float path after it is the real one.
    """
    import numpy as np

    from fugal_subnet.benchmarks.slicer import derive_nonce, select_slice
    from fugal_subnet.config import HEAD_HIDDEN_DIM, ROUTING_LAMBDA
    from fugal_subnet.head_eval import HeadArtifact, evaluate_head
    from fugal_subnet.matrix import build_matrix_mock
    from fugal_subnet.rewards import compute_weights
    from fugal_subnet.scoring import ScoringState, update_scores
    from fugal_subnet.soft_targets import compute_soft_targets

    models = [f"vendor/model-{i}" for i in range(8)]

    # A synthetic pool large enough that near-ties in routing utility actually
    # occur — those are the cases where nondeterminism shows up.
    pool = [
        {
            "question_id": f"q{i}",
            "prompt": f"Question {i}: compute something deterministic.",
            "answer": str(i),
            "grader": "numeric_final",
            "benchmark": ["gsm8k", "math", "mmlu", "aime"][i % 4],
        }
        for i in range(240)
    ]

    nonce = derive_nonce(f"epoch-{seed}", f"0x{seed:064x}")
    questions = select_slice(nonce, pool, size=64)

    rng = np.random.RandomState(seed)
    hidden = rng.randn(len(questions), HEAD_HIDDEN_DIM).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)

    matrix_result = build_matrix_mock(questions, models)
    matrix = matrix_result.matrix
    model_costs = {m: 0.001 * (i + 1) for i, m in enumerate(models)}
    soft = compute_soft_targets(matrix)

    # Several heads whose utilities land close together on purpose, so the
    # argmax quantization is genuinely exercised rather than trivially agreed.
    routing, scores_out, epoch_scores, head_hashes = {}, {}, {}, {}
    for uid in range(1, 5):
        hrng = np.random.RandomState(1000 + uid)
        W = (hrng.randn(len(models), HEAD_HIDDEN_DIM) * 0.01).astype(np.float32)
        b = (hrng.randn(len(models)) * 1e-6).astype(np.float32)
        head = HeadArtifact(W=W, b=b, models=list(models), commit_hash=f"h{uid}")
        hs = evaluate_head(head, hidden, matrix, models, soft, model_costs,
                           lam=ROUTING_LAMBDA)
        routing[uid] = [int(x) for x in hs.routing_decisions]
        scores_out[uid] = {
            "accuracy": float(hs.accuracy),
            "cost_efficiency": float(hs.cost_efficiency),
            "kl_score": float(hs.kl_score),
        }
        epoch_scores[uid] = hs
        head_hashes[uid] = f"h{uid}"

    state = update_scores(ScoringState(), epoch_scores, head_hashes,
                          n_questions=len(questions))
    uids, weight_values = compute_weights(state.records)

    return {
        "slice": _digest([q["question_id"] for q in questions]),
        "hidden_states": _digest(hidden),
        "matrix": _digest(matrix),
        "soft_targets": _digest(soft),
        "routing_decisions": _digest(routing),
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
