"""Measure what a validator's proof-verification phase actually costs.

The epoch loop queries miners in PARALLEL (`dendrite.query` over every axon)
and then verifies them in a SERIAL `for uid, resp in enumerate(responses)`
loop. Each iteration reaches `verify_dcap`, which makes a blocking network
round trip for DCAP collateral. So verification cost is
`n_miners x (local_cpu + collateral_rtt)` with nothing overlapped.

This harness measures `local_cpu` for real -- real proofs, real heads, the
real verifier -- and composes it with a swept `collateral_rtt`, because the
RTT is a property of somebody else's service and cannot be measured into a
single honest number. The question is not "how slow is it" but "which miner
counts still fit inside the budget, under which RTT assumptions".

No network. No spend. Run:  uv run python scripts/bench_verify_throughput.py
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import statistics
import time

import numpy as np

from fugal_subnet import success_test_fixtures as success_fixtures
from fugal_subnet.benchmarks.slicer import (
    BLOCK_TIME_S,
    derive_nonce,
    epoch_id_for_block,
)
from fugal_subnet.config import (
    EPOCH_COLLECT_FRACTION,
    EPOCH_INTERVAL,
    HEAD_HIDDEN_DIM,
    SLICE_SIZE,
)
from fugal_subnet.exploration import expected_exploration
from fugal_subnet.tee import harness as harness_mod
from fugal_subnet.tee.proof import BenchmarkProof, compute_questions_hash
from fugal_subnet.tee.runtime import APICallRecord, MeteringProxy, TEERuntime
from fugal_subnet.tee.verify import verify_proof

MODELS = ["a/cheap", "b/mid", "c/expensive"]
PRICES = {"a/cheap": (1e-7, 2e-7), "b/mid": (1e-6, 2e-6), "c/expensive": (5e-6, 3e-5)}
SKILL = {"a/cheap": 0.3, "b/mid": 0.6, "c/expensive": 0.9}

# Sweep of plausible per-proof collateral round trips, seconds.
#   0.05  a warm local PCCS cache on the same host
#   0.25  a healthy remote PCCS, one region away
#   1.0   a loaded public endpoint
#   3.0   a degraded one
#   30.0  a black-holing service. NOT bounded by the `timeout=30` in
#         verify_dcap: that guard is on the ThreadPoolExecutor branch,
#         which a synchronous validator never takes. Measured -- on the
#         validator's path `loop.is_running()` is False, so it calls
#         `run_until_complete` with no timeout of its own, and the only
#         bound is whatever the Rust HTTP client applies internally.
RTT_SWEEP = [0.05, 0.25, 0.69, 1.0, 3.0, 30.0]

# The one row that is not a guess. Measured on a c3-standard-8 dstack CVM in
# GCP us-central1 against Phala's PCCS: 788 ms cold, then 688 and 699 ms warm
# (docs/INVARIANTS.md, "what a live dstack CVM actually reports"). Labelled with
# its network on purpose -- a validator on a home connection or another
# continent sees something else, so this is one point, not the constant.
MEASURED_RTT = 0.69
MEASURED_NOTE = "measured, GCP us-central1 -> Phala"
FIELD_SIZES = [16, 64, 128, 256]


def _pool(n: int) -> list[dict]:
    return [
        {
            "question_id": f"q{i}",
            "prompt": f"Question {i}?",
            "gold": str(i),
            "grader_id": "numeric_final",
            "benchmark": ["gsm8k", "math", "mmlu", "aime"][i % 4],
            "metadata": {},
        }
        for i in range(n)
    ]


def _head(models: list[str], seed: int = 0) -> bytes:
    from fugal_subnet import success_test_fixtures as success_fixtures
    rng = np.random.RandomState(seed)
    W = (rng.randn(len(models), HEAD_HIDDEN_DIM) * 0.01).astype(np.float32)
    b = rng.randn(len(models)).astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, **success_fixtures.arrays(W, b, models))
    return buf.getvalue()


class _StubProxy(MeteringProxy):
    def start(self):
        self.prices = PRICES


def _stub_call(proxy, model_id, question):
    qid = question["question_id"]
    digest = hashlib.sha256(f"{qid}|{model_id}".encode()).digest()
    correct = (digest[0] / 255.0) < SKILL[model_id]
    text = question["gold"] if correct else "0"
    p_tok, c_tok = 500, 300
    proxy.records.append(APICallRecord(
        model_id=model_id, prompt_tokens=p_tok, completion_tokens=c_tok,
        cost_usd=proxy.price_call(model_id, p_tok, c_tok),
        timestamp=0.0, response_hash=hashlib.sha256(text.encode()).hexdigest(),
        request_id=harness_mod.current_request_id(),
    ))
    return text


def build_field(n_miners: int, slice_size: int, pool_size: int):
    """Build `n_miners` real proofs, as the validator would receive them."""
    harness_mod.benchmark_costs = success_fixtures.costs
    harness_mod._call_model = _stub_call  # noqa: SLF001 - no network, no spend

    pool = _pool(pool_size)
    hidden = np.random.RandomState(0).randn(pool_size, HEAD_HIDDEN_DIM).astype(np.float32)

    block, blocks_per_epoch = 37_123, int(EPOCH_INTERVAL / BLOCK_TIME_S)
    epoch_id = epoch_id_for_block(block // blocks_per_epoch)
    nonce = derive_nonce(epoch_id, "0x" + "ab" * 32)

    field = []
    for i in range(n_miners):
        head_bytes = _head(MODELS, seed=i)
        proxy = _StubProxy(port=0)
        proxy.start()
        proof = harness_mod.run_benchmark(
            nonce=nonce.hex(), head_bytes=head_bytes, benchmark_pool=pool, proxy=proxy,
            hidden_states=hidden, slice_size=slice_size, epoch_id=epoch_id,
            source_hash="bench-image", explore_models=MODELS,
            explore_size=max(1, int(slice_size * 0.05)),
        )
        proof.hotkey = f"hk-{i}"
        proof.attestation_quote = TEERuntime(mock=True).generate_attestation(
            bytes.fromhex(proof.content_hash())
        )
        # The wire form: the validator receives JSON + base64, not objects.
        field.append((
            json.dumps(proof.to_dict()),
            base64.b64encode(head_bytes).decode(),
            proof,
            head_bytes,
        ))

    from fugal_subnet.benchmarks.slicer import select_slice
    questions = select_slice(nonce, pool, slice_size)
    scored_ids = {q["question_id"] for q in questions}
    explore_map = expected_exploration(
        nonce, pool, scored_ids, MODELS, max(1, int(slice_size * 0.05))
    )
    by_id = {q["question_id"]: q for q in pool}
    ctx = {
        "nonce_hex": nonce.hex(),
        # The validator hashes the ordered ids from select_slice, not a sorted set.
        "questions_hash": compute_questions_hash([q["question_id"] for q in questions]),
        "question_ids": scored_ids,
        # Gold for the assigned slice only -- scored plus exploration, exactly
        # as neurons/validator.py builds slice_gold.
        "gold": {qid: by_id[qid] for qid in list(scored_ids) + list(explore_map)},
        "explore": explore_map,
    }
    return field, ctx


def measure_local(field, ctx, repeats: int = 3) -> dict:
    """Per-proof CPU cost of the verify path, DCAP excluded."""
    parse_ts, verify_ts = [], []
    for _ in range(repeats):
        for proof_json, head_b64, proof, head_bytes in field:
            t0 = time.perf_counter()
            p = BenchmarkProof.from_dict(json.loads(proof_json))
            hb = base64.b64decode(head_b64, validate=True)
            t1 = time.perf_counter()
            r = verify_proof(
                p,
                approved_measurements="",
                expected_questions_hash=ctx["questions_hash"],
                expected_nonce=ctx["nonce_hex"],
                gold_answers=ctx["gold"],
                expected_question_ids=ctx["question_ids"],
                expected_exploration=ctx["explore"],
                expected_weights_hash=p.weights_hash,
                expected_hotkey=p.hotkey,
                expected_proof_hash=p.content_hash(),
                head_bytes=hb,
                mock=True,
            )
            t2 = time.perf_counter()
            if not r.valid:
                raise SystemExit(f"harness built an invalid proof: {r.reason}")
            parse_ts.append(t1 - t0)
            verify_ts.append(t2 - t1)
    return {
        "parse_mean": statistics.mean(parse_ts),
        "verify_mean": statistics.mean(verify_ts),
        "local_mean": statistics.mean(parse_ts) + statistics.mean(verify_ts),
        "n": len(parse_ts),
    }


def budget_seconds() -> float:
    """Seconds a validator has between collecting proofs and the epoch ending.

    Collection starts at EPOCH_COLLECT_FRACTION into the epoch, and everything
    -- verify, score, weights, reveal -- must land before the next boundary.
    """
    return EPOCH_INTERVAL * (1.0 - EPOCH_COLLECT_FRACTION)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-size", type=int, default=SLICE_SIZE)
    ap.add_argument("--pool-size", type=int, default=2000)
    ap.add_argument("--sample-miners", type=int, default=8,
                    help="proofs actually built; local cost is per-proof so "
                         "the field sizes below are extrapolated from it")
    args = ap.parse_args()

    budget = budget_seconds()
    print(f"EPOCH_INTERVAL      {EPOCH_INTERVAL}s")
    print(f"COLLECT_FRACTION    {EPOCH_COLLECT_FRACTION}  -> collection at "
          f"{EPOCH_INTERVAL * EPOCH_COLLECT_FRACTION:.0f}s")
    print(f"post-collect budget {budget:.0f}s  (verify + score + weights + reveal)")
    print(f"slice size          {args.slice_size} questions\n")

    print(f"Building {args.sample_miners} real proofs...")
    t0 = time.perf_counter()
    field, ctx = build_field(args.sample_miners, args.slice_size, args.pool_size)
    print(f"  built in {time.perf_counter() - t0:.1f}s\n")

    m = measure_local(field, ctx)
    print("--- measured local cost per proof (no DCAP, no network) ---")
    print(f"  parse+b64 decode  {m['parse_mean'] * 1000:.2f} ms")
    print(f"  verify_proof      {m['verify_mean'] * 1000:.2f} ms")
    print(f"  local total       {m['local_mean'] * 1000:.2f} ms   "
          f"(n={m['n']} observations)\n")

    local = m["local_mean"]
    print("--- serial epoch verify time = n x (local + collateral RTT) ---")
    hdr = "  RTT/proof   |" + "".join(f"{n:>12}" for n in FIELD_SIZES)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for rtt in RTT_SWEEP:
        mark = " *" if rtt == MEASURED_RTT else "  "
        row = f"  {rtt:>8.2f}s{mark}|"
        for n in FIELD_SIZES:
            total = n * (local + rtt)
            flag = "" if total < budget else " !"
            row += f"{total:>10.0f}s{flag:<2}"
        print(row)
    print(f"\n  * {MEASURED_NOTE}; every other row is swept, not observed")
    print(f"\n  ! = exceeds the {budget:.0f}s post-collection budget")
    print("  (and that budget must also cover scoring, weight-setting and the reveal)\n")

    print("--- local cost alone, if collateral were cached ---")
    for n in FIELD_SIZES:
        print(f"  {n:>4} miners  {n * local:>8.2f}s   "
              f"({100 * n * local / budget:.2f}% of budget)")

    print()
    abstention_table()


def abstention_table():
    """How often a consensus-safe validator would publish nothing at all.

    Once "unverifiable" is distinguished from "invalid", the validator has to
    choose a granularity for failing closed, and both choices are bad:

      per miner  skip the ones that could not be checked. Two validators hold
                 different sets, so they publish different weights. The fork is
                 politer, not smaller.
      per epoch  publish nothing unless every proof verified. No divergent
                 weights -- Yuma tolerates abstention -- but with one network
                 fetch per proof, the chance that ALL of them succeed falls off
                 a cliff as the field grows.

    This is the per-epoch column, and it is why error handling cannot fix this:
    the divergence is in the input, not in how the input is handled.

    MODELLED, not measured -- independent failures at a fixed per-fetch rate.
    Real failures cluster (one outage takes many fetches at once), so treat
    these as the optimistic end.
    """
    print("--- P(validator abstains for the whole epoch), per-epoch fail-closed ---")
    print("    modelled: 1-(1-p)^n, independent failures, one fetch per proof")
    hdr = "  p(fetch fails) |" + "".join(f"{n:>10}" for n in FIELD_SIZES)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pf in (0.0001, 0.001, 0.01, 0.05):
        row = f"  {pf:>14.4f} |"
        for n in FIELD_SIZES:
            row += f"{100 * (1 - (1 - pf) ** n):>9.1f}%"
        print(row)
    print()
    print("  Same rate, but collateral fetched ONCE per distinct platform (FMSPC)")
    print("  instead of once per proof -- caching, or shipping it with the proof:")
    hdr2 = "  p(fetch fails) |" + "".join(f"{f:>10}" for f in (1, 3, 5))
    print(hdr2 + "   <- distinct FMSPCs")
    print("  " + "-" * (len(hdr2) - 2))
    for pf in (0.0001, 0.001, 0.01, 0.05):
        row = f"  {pf:>14.4f} |"
        for f in (1, 3, 5):
            row += f"{100 * (1 - (1 - pf) ** f):>9.1f}%"
        print(row)
    print()
    print("  The field size stops appearing in the second table. That is the whole")
    print("  argument: removing the per-proof fetch decouples consensus safety from")
    print("  how many miners are on the subnet. It is not a latency optimisation.")


if __name__ == "__main__":
    main()
