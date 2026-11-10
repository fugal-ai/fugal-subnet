#!/usr/bin/env python3
"""Keep a miner's GCP firewall admitting only validators to its axon.

Reads the ACTIVE validators' published IPs from the metagraph (a validator
permit, stake at or above MIN_VALIDATOR_STAKE, weights set in the last
--active-blocks) and makes
one GCP firewall rule admit exactly those addresses to the axon port, for the
instances carrying the miner's network tag. Validators publish their egress
IP at startup (FUGAL_VALIDATOR_IP; fugal_subnet/axon_allowlist.py).

Why: a v2 champion kept offline past its 3-epoch ghost loses the seat, and
only validators ever need to reach a miner (INVARIANTS I3, "Accepted
residual: a champion kept offline"; docs/MINER_GUIDE.md, "Protect your axon").

    # Show what it would do (default: nothing is changed)
    python scripts/axon_firewall.py --network finney --netuid 70 --project <gcp-project>
    # Apply, then keep it current every 10 minutes
    python scripts/axon_firewall.py --network finney --netuid 70 --project <gcp-project> \\
        --apply --every 600

Safety: it never writes an empty allowlist (that would drop every validator),
refuses while any staked validator has not published an IP (that validator
would see this miner as offline while the others do not: --allow-partial
overrides), keeps the rule as it is when the chain cannot be read, and only REPORTS other
rules that still open the port to 0.0.0.0/0 (GCP allow rules add up, so such a
rule makes this one pointless): deleting a rule is yours to do.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fugal_subnet.axon_allowlist import (  # noqa: E402
    firewall_command,
    rule_is_current,
    validator_cidrs,
    world_open_rules,
)


def _gcloud_json(cmd: list[str]):
    out = subprocess.run(cmd + ["--format=json"], capture_output=True, text=True,
                         timeout=120, check=True).stdout
    return json.loads(out or "null")


def _instance_tags_and_vpc(args) -> tuple[set[str], str]:
    """The instance's real network tags and VPC (when --instance is given):
    a world-open rule on ANY tag the VM carries defeats the allowlist."""
    if not args.instance:
        return {args.target_tag}, args.vpc
    inst = _gcloud_json(["gcloud", "compute", "instances", "describe", args.instance,
                         f"--zone={args.zone}", f"--project={args.project}"])
    tags = set((inst.get("tags") or {}).get("items") or [])
    nics = inst.get("networkInterfaces") or [{}]
    vpc = str(nics[0].get("network", args.vpc)).rstrip("/").rsplit("/", 1)[-1]
    if args.target_tag not in tags:
        print(f"WARNING: instance {args.instance} does not carry tag {args.target_tag}: the "
              f"allowlist rule does not apply to it (gcloud compute instances add-tags "
              f"{args.instance} --tags {args.target_tag} --zone {args.zone})")
    return tags | {args.target_tag}, vpc


def run_once(args) -> int:
    import bittensor as bt

    from fugal_subnet.config import MIN_VALIDATOR_STAKE

    min_stake = MIN_VALIDATOR_STAKE if args.min_stake is None else args.min_stake
    try:
        subtensor = bt.Subtensor(network=args.network)
        metagraph = subtensor.metagraph(args.netuid)
    except Exception as e:  # noqa: BLE001 - keep the rule as it is
        print(f"cannot read the metagraph ({e}); the firewall rule is left as it is")
        return 1
    cidrs, notes = validator_cidrs(metagraph, min_stake, args.active_blocks)
    for n in notes:
        print(f"note: {n}")
    if notes and not args.allow_partial:
        print(f"refusing: {len(notes)} active validator(s) have not published an IP. "
              "Admitting the others would make this miner look offline to them, and "
              "validators would disagree about it. Rule left as it is (--allow-partial "
              "overrides).")
        return 1
    cidrs = sorted(set(cidrs) | set(args.extra_cidr or []))
    print(f"{len(cidrs)} address(es) to admit on tcp:{args.port}: {', '.join(cidrs) or '-'}")
    if not cidrs:
        print("refusing an empty allowlist: it would drop every validator; rule left as it is")
        return 1
    try:
        tags, vpc = _instance_tags_and_vpc(args)
        rules = _gcloud_json(["gcloud", "compute", "firewall-rules", "list",
                              f"--project={args.project}"]) or []
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        print(f"cannot read the GCP firewall ({e}); the rule is left as it is")
        return 1
    mine = next((r for r in rules if r.get("name") == args.rule), None)
    others = [r for r in rules if r.get("name") != args.rule]
    open_, manual = world_open_rules(others, args.port, tags, vpc)
    if open_:
        print(f"WARNING: rule(s) {', '.join(open_)} still open tcp:{args.port} broadly to this "
              f"instance; while they exist this allowlist changes nothing. Delete or narrow "
              f"them once the allowlist is in place, e.g. gcloud compute firewall-rules "
              f"delete {open_[0]} --project={args.project}")
    if manual:
        print(f"check by hand: rule(s) {', '.join(manual)} open tcp:{args.port} broadly to "
              "service accounts; if this instance runs as one of them they defeat the allowlist")
    if mine is not None and str(mine.get("network", "")).rstrip("/").rsplit("/", 1)[-1] \
            not in ("", vpc):
        print(f"rule {args.rule} is on another VPC than the instance ({vpc}): delete it and "
              "run again; it is left as it is")
        return 1
    if rule_is_current(mine, port=args.port, target_tag=args.target_tag, network=vpc,
                       cidrs=cidrs):
        print(f"rule {args.rule} is current")
        return 0
    cmd = firewall_command(rule=args.rule, project=args.project, port=args.port,
                           target_tag=args.target_tag, cidrs=cidrs, exists=mine is not None,
                           network=vpc)
    print(("applying: " if args.apply else "would run (pass --apply): ") + " ".join(cmd))
    if args.apply:
        try:
            subprocess.run(cmd, check=True, timeout=300)
        except (subprocess.SubprocessError, OSError) as e:
            print(f"gcloud failed ({e}); the rule may be unchanged, retried next run")
            return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", required=True, help="bittensor network (finney, test)")
    p.add_argument("--netuid", type=int, required=True)
    p.add_argument("--project", required=True, help="GCP project of the miner's instances")
    p.add_argument("--vpc", default="default", help="GCP VPC network of the instances")
    p.add_argument("--instance", help="the miner's VM: read its real tags and VPC")
    p.add_argument("--zone", default="us-central1-a")
    p.add_argument("--rule", default="fugal-axon-validators")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--target-tag", default="fugal-miner")
    p.add_argument("--extra-cidr", action="append",
                   help="also admit this range (e.g. your own monitoring); repeatable")
    p.add_argument("--min-stake", type=float, default=None,
                   help="default: MIN_VALIDATOR_STAKE, the axon's own rule")
    p.add_argument("--active-blocks", type=int, default=1080,
                   help="a validator counts if it set weights within this many blocks "
                        "(default 3 epochs)")
    p.add_argument("--allow-partial", action="store_true",
                   help="apply even if some active validators published no IP (they lose "
                        "access to this miner)")
    p.add_argument("--apply", action="store_true", help="change the rule (default: dry run)")
    p.add_argument("--every", type=int, default=0, help="repeat every N seconds")
    args = p.parse_args()
    while True:
        try:
            rc = run_once(args)
        except Exception as e:  # noqa: BLE001 - a loop that dies leaves the rule stale
            print(f"run failed ({e}); retried next run")
            rc = 1
        if not args.every:
            return rc
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
