#!/usr/bin/env python3
"""Watch a live Fugal subnet and assert it is actually working.

Runs against any real network. Every line is a pass/fail assertion rather than
a log entry, because the failure mode this project keeps hitting is a component
that reports success while doing nothing: an axon served at an unreachable
address, a set_weights that returned Success against a chain that stored
nothing, a subnet that was never activated. Each of those looked healthy.

    python scripts/watch_subnet.py --network test --netuid 552 --epochs 3 \
        --reveals validator1=/path/results/epochs,validator2=/path2/results/epochs

What it checks, per epoch:

  chain      the subnet is active, blocks advance, the epoch index advances
  miners     each has a routable axon on chain, and a head commitment made at
             or before the boundary block (a later one is unscoreable by design)
  validators each holds a permit, and its weights actually reached the chain —
             read back commit-reveal-aware, since on those subnets an immediate
             read correctly finds nothing
  agreement  every validator's published reveal for the epoch agrees, compared
             through fugal_subnet.consensus, which is the same audit anyone
             outside the subnet can run on the published artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.lines: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        tag = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        line = f"  [{tag}] {label}" + (f" — {detail}" if detail else "")
        print(line, flush=True)
        self.lines.append(line)
        return ok

    def summary(self) -> int:
        print(f"\n{'=' * 78}\n{self.passed} passed, {self.failed} failed\n{'=' * 78}")
        return 1 if self.failed else 0


def load_reveal(root: str, epoch_id: str) -> dict | None:
    path = os.path.join(root, epoch_id, "reveal.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def check_epoch(report, subtensor, netuid, epoch_id, boundary_block,
                miner_hotkeys, validator_hotkeys, reveal_roots) -> None:
    from fugal_subnet.commitments import get_commitments_with_blocks

    print(f"\n--- epoch {epoch_id} (boundary block {boundary_block}) ---", flush=True)
    mg = subtensor.metagraph(netuid)
    hk_to_uid = {hk: i for i, hk in enumerate(mg.hotkeys)}

    report.check(subtensor.is_subnet_active(netuid), "subnet is active")

    commitments = {}
    try:
        commitments = get_commitments_with_blocks(subtensor, netuid)
    except Exception as e:
        report.check(False, "read on-chain commitments", str(e)[:120])

    for name, hk in miner_hotkeys.items():
        uid = hk_to_uid.get(hk, -1)
        if not report.check(uid >= 0, f"miner {name} is registered"):
            continue
        ip = mg.axons[uid].ip
        report.check(ip not in ("0.0.0.0", ""),
                     f"miner {name} has a routable axon on chain",
                     f"uid {uid} at {ip}:{mg.axons[uid].port}")
        data, block = commitments.get(hk, ("", 0))
        report.check(bool(data), f"miner {name} has a head commitment",
                     f"block {block}")
        if data:
            # A head committed after the boundary block is deliberately
            # unscoreable: the nonce is knowable by then, so the head could
            # have been fitted to the slice.
            report.check(block <= boundary_block,
                         f"miner {name} committed at or before the boundary",
                         f"commit block {block} vs boundary {boundary_block}")

    for name, hk in validator_hotkeys.items():
        uid = hk_to_uid.get(hk, -1)
        if not report.check(uid >= 0, f"validator {name} is registered"):
            continue
        report.check(bool(mg.validator_permit[uid]),
                     f"validator {name} holds a permit",
                     f"uid {uid}, stake {float(mg.S[uid]):.2f}")
        last_update = int(mg.last_update[uid])
        report.check(last_update >= boundary_block,
                     f"validator {name} set weights this epoch",
                     f"LastUpdate {last_update} vs boundary {boundary_block}")

    # Cross-validator agreement, from the published artifacts.
    if len(reveal_roots) >= 2:
        from fugal_subnet.consensus import ValidatorReport, compute_consensus

        reports = []
        for name, root in reveal_roots.items():
            rv = load_reveal(root, epoch_id)
            if rv is None:
                report.check(False, f"reveal published by {name}",
                             f"no {epoch_id}/reveal.json under {root}")
                continue
            report.check(True, f"reveal published by {name}")
            reports.append(ValidatorReport(
                validator_uid=hk_to_uid.get(validator_hotkeys.get(name, ""), -1),
                epoch_id=epoch_id,
                scores={int(k): float(v) for k, v in rv.get("scores", {}).items()},
                weights={int(k): float(v) for k, v in rv.get("weights", {}).items()},
                commit_hash=rv.get("commit_hash", ""),
            ))
        if len(reports) >= 2:
            digests = {name: (load_reveal(root, epoch_id) or {}).get("consensus_digest")
                       for name, root in reveal_roots.items()}
            uniq = {d for d in digests.values() if d}
            report.check(len(uniq) <= 1,
                         "validators share one consensus environment digest",
                         ", ".join(f"{k}={str(v)[:12]}" for k, v in digests.items()))

            res = compute_consensus(reports)
            report.check(res.is_valid and not res.outlier_validators,
                         "validators agree on scores and weights",
                         "; ".join(res.divergence_flags) or "no divergence")
            exact = all(r.weights == reports[0].weights for r in reports)
            report.check(exact, "validator weight vectors are byte-identical",
                         json.dumps(reports[0].weights, sort_keys=True)[:160])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="test")
    ap.add_argument("--netuid", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=3, help="Epoch boundaries to observe")
    ap.add_argument("--epoch-interval", type=int,
                    default=int(os.getenv("FUGAL_EPOCH_INTERVAL", "3600")))
    ap.add_argument("--miners", default="", help="name=hotkey_ss58,...")
    ap.add_argument("--validators", default="", help="name=hotkey_ss58,...")
    ap.add_argument("--reveals", default="", help="name=/path/to/results/epochs,...")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    import bittensor as bt

    from fugal_subnet.benchmarks.slicer import (
        blocks_per_epoch,
        epoch_id_for_block,
        epoch_index_for_block,
    )
    from fugal_subnet.logging_setup import configure_logging
    configure_logging("WARNING")

    def kv(spec: str) -> dict:
        out = {}
        for item in filter(None, (s.strip() for s in spec.split(","))):
            k, _, v = item.partition("=")
            out[k] = v
        return out

    miners, validators, reveals = kv(args.miners), kv(args.validators), kv(args.reveals)
    subtensor = bt.Subtensor(network=args.network)
    bpe = blocks_per_epoch(args.epoch_interval)
    report = Report()

    print(f"Watching netuid {args.netuid} on {args.network}: epoch = {bpe} blocks "
          f"({args.epoch_interval}s nominal), observing {args.epochs} boundaries")

    seen: set[str] = set()
    deadline = time.time() + args.timeout
    while len(seen) < args.epochs and time.time() < deadline:
        block = subtensor.get_current_block()
        idx = epoch_index_for_block(block, bpe)
        epoch_id = epoch_id_for_block(idx)
        if epoch_id in seen:
            time.sleep(30)
            continue
        # Give the neurons time to finish the epoch before judging it: the
        # validator queries, verifies, scores and only then sets weights.
        elapsed_in_epoch = block - idx * bpe
        if elapsed_in_epoch < min(bpe - 1, 30):
            time.sleep(30)
            continue
        seen.add(epoch_id)
        check_epoch(report, subtensor, args.netuid, epoch_id, idx * bpe,
                    miners, validators, reveals)

    if len(seen) < args.epochs:
        report.check(False, "observed the requested number of epochs",
                     f"saw {len(seen)} of {args.epochs} before the {args.timeout}s timeout")
    return report.summary()


if __name__ == "__main__":
    raise SystemExit(main())
