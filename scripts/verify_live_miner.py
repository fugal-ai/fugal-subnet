#!/usr/bin/env python3
"""Query one miner and verify its proof the way a --live validator would.

Answers a question an operator actually has — *is this miner producing valid
attested proofs right now?* — without setting weights, touching validator state,
or interfering with a running subnet. The full live path in isolation:

    axon query -> DCAP signature -> measurement in approved set -> report_data
    binds the proof body -> slice, exploration, head and cost bindings

    python scripts/verify_live_miner.py --netuid 552 --uid 5 \\
        --coldkey fugal_owner2 --hotkey default \\
        --measurements a68d0ccd...  [--expect-reject]

`--expect-reject` inverts the assertion, which is how the negative control is
run: the same genuine proof against a DIFFERENT approved set must be refused.
A test that only ever confirms acceptance cannot tell a working check from one
that returns True.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="test")
    ap.add_argument("--netuid", type=int, required=True)
    ap.add_argument("--uid", type=int, required=True, help="Miner UID to query")
    ap.add_argument("--coldkey", required=True, help="Wallet to query as")
    ap.add_argument("--hotkey", default="default")
    ap.add_argument("--measurements", default="",
                    help="Comma-separated approved measurement_ids. Empty means "
                         "mock mode, which skips the hardware checks entirely.")
    ap.add_argument("--expect-reject", action="store_true",
                    help="Assert the proof is REFUSED (negative control)")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    import bittensor as bt

    from fugal_subnet.api import load_prices
    from fugal_subnet.benchmarks.loader import load_all, pool_hash
    from fugal_subnet.benchmarks.slicer import (
        blocks_per_epoch,
        derive_nonce,
        epoch_id_for_block,
        epoch_index_for_block,
        select_slice,
    )
    from fugal_subnet.config import EPOCH_INTERVAL, EXPLORE_FRACTION, SLICE_SIZE
    from fugal_subnet.exploration import expected_exploration
    from fugal_subnet.logging_setup import configure_logging
    from fugal_subnet.protocol import FugalProofSynapse
    from fugal_subnet.tee.proof import BenchmarkProof
    from fugal_subnet.tee.verify import compute_questions_hash, verify_proof

    configure_logging("INFO")

    wallet = bt.Wallet(name=args.coldkey, hotkey=args.hotkey)
    subtensor = bt.Subtensor(network=args.network)
    metagraph = subtensor.metagraph(args.netuid)

    bpe = blocks_per_epoch(EPOCH_INTERVAL)
    epoch_index = epoch_index_for_block(subtensor.get_current_block(), bpe)
    epoch_id = epoch_id_for_block(epoch_index)
    block_hash = subtensor.get_block_hash(epoch_index * bpe)
    nonce = derive_nonce(epoch_id, block_hash)

    pool = load_all()
    print(f"pool {len(pool)} questions, pool_hash={pool_hash(pool)[:16]}")
    questions = select_slice(nonce, pool, SLICE_SIZE)
    qids = [q["question_id"] for q in questions]
    prices = load_prices()
    explore_size = max(1, int(round(SLICE_SIZE * EXPLORE_FRACTION)))
    explore_map = expected_exploration(
        nonce, pool, set(qids), sorted(prices), explore_size,
    )
    gold = {q["question_id"]: q for q in pool}

    print(f"epoch {epoch_id} boundary {epoch_index * bpe} nonce {nonce.hex()[:16]}")
    axon = metagraph.axons[args.uid]
    print(f"querying uid {args.uid} at {axon.ip}:{axon.port} ...")

    dendrite = bt.Dendrite(wallet=wallet)
    try:
        responses = dendrite.query(
            [axon], FugalProofSynapse(epoch_id=epoch_id, nonce=nonce.hex()),
            timeout=args.timeout,
        )
    finally:
        try:
            dendrite.close_session()
        except Exception:  # noqa: BLE001
            pass

    resp = responses[0] if responses else None
    if resp is None or not getattr(resp, "proof_json", ""):
        print("FAIL: no proof returned — the miner is unreachable, still "
              "benchmarking, or answering for a different epoch")
        return 1

    proof = BenchmarkProof.from_dict(json.loads(resp.proof_json))
    approved = {m.strip() for m in args.measurements.split(",") if m.strip()}
    mock = not approved
    print(f"proof received: {len(proof.results)} results, "
          f"quote {len(proof.attestation_quote)} bytes, mode="
          f"{'MOCK (hardware checks skipped)' if mock else 'LIVE'}")

    result = verify_proof(
        proof,
        approved_measurements=approved,
        expected_questions_hash=compute_questions_hash(qids),
        expected_nonce=nonce.hex(),
        gold_answers=gold,
        expected_question_ids=set(qids),
        expected_exploration=explore_map,
        expected_proof_hash=getattr(resp, "proof_hash", ""),
        mock=mock,
    )

    print(f"\nvalid : {result.valid}")
    print(f"reason: {result.reason or '(none)'}")
    for w in getattr(result, "warnings", []) or []:
        print(f"  warning: {w}")

    if args.expect_reject:
        ok = not result.valid
        print(f"\n[{'PASS' if ok else 'FAIL'}] negative control: a genuine proof "
              f"from an unapproved measurement is refused")
        return 0 if ok else 1
    ok = result.valid
    print(f"\n[{'PASS' if ok else 'FAIL'}] the miner's proof verifies "
          f"{'against real attestation' if not mock else 'structurally'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
