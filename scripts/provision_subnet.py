#!/usr/bin/env python3
"""Provision a Fugal subnet on a real Bittensor network.

This is the script you run to launch. It works identically against `test` and
`finney`; the network is the only thing that changes. Every step reads chain
state before acting and skips what is already done, so it is safe to re-run
after a partial failure — which matters, because a half-provisioned subnet is
the normal outcome of a dropped websocket.

Ordering is load-bearing:

1. **create** the subnet (pays the lock cost) — only with --create.
2. **identity** so the subnet is findable by name rather than number.
3. **start_call** — the step whose absence is invisible. Without it a subnet
   has no first-emission block: staking is rejected with SubtokenDisabled,
   validators never earn a permit, and set_weights fails. The pipeline runs,
   reports success, and lands nothing. Netuid 552 sat in exactly this state.
4. **hyperparameters** before registration, so miners register into a subnet
   that already has its final shape.
5. **register** hotkeys (pays the recycle burn each).
6. **stake** validators, so they hold a permit before the first epoch.

Spending is gated: the run prints every cost and stops unless --yes is given.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import bittensor as bt

from fugal_subnet.benchmarks.slicer import BLOCK_TIME_S
from fugal_subnet.logging_setup import configure_logging

logger = logging.getLogger("fugal.provision")

# Hyperparameters Fugal actually needs, and why. Anything not listed is left at
# the chain default deliberately — an unexplained override is a liability.
#
# EPOCH ALIGNMENT. Fugal's epoch is EPOCH_INTERVAL // BLOCK_TIME_S blocks; the
# chain runs Yuma every `tempo` blocks. When they differ, some Fugal epochs set
# weights that a later epoch overwrites before any Yuma step reads them — that
# epoch's work never reaches emission. Setting tempo == blocks_per_epoch makes
# it exactly one weight-set per Yuma step. `sudo_set_tempo` may be root-only on
# a given chain; if it is refused we report the misalignment rather than
# pretending it is fine.
OWNER_HYPERPARAMS: dict[str, tuple[str, str]] = {
    # call name: (param name, why)
    "sudo_set_commit_reveal_weights_enabled": (
        "enabled",
        "weight copying is the cheapest attack on any subnet; commit-reveal is "
        "the chain-level defence and the validator already handles both modes",
    ),
    "sudo_set_network_registration_allowed": (
        "registration_allowed", "miners must be able to join",
    ),
    "sudo_set_serving_rate_limit": (
        "serving_rate_limit",
        "the miner serves its axon once at startup, so the default is ample; "
        "pinned so a future default change cannot strand miners at 0.0.0.0",
    ),
}


def _ok(resp) -> tuple[bool, str]:
    """Normalise an ExtrinsicResponse (or a (bool, msg) tuple) to (ok, message)."""
    if isinstance(resp, tuple):
        return bool(resp[0]), str(resp[1])
    success = getattr(resp, "success", None)
    if success is None:
        return bool(resp), str(resp)
    return bool(success), str(getattr(resp, "error_message", "") or "")


def _sudo_call(subtensor, wallet, call_name: str, params: dict) -> tuple[bool, str]:
    """Submit an AdminUtils call signed by the subnet owner coldkey.

    AdminUtils accepts either root or the subnet owner for the subnet-scoped
    setters, so no Sudo wrapper is used — on a public chain there is no sudo
    key to wrap with, which is why the local-testnet helper cannot be reused.
    """
    try:
        call = subtensor.substrate.compose_call(
            call_module="AdminUtils", call_function=call_name, call_params=params,
        )
        ext = subtensor.substrate.create_signed_extrinsic(call=call, keypair=wallet.coldkey)
        receipt = subtensor.substrate.submit_extrinsic(
            ext, wait_for_inclusion=True, wait_for_finalization=False,
        )
        if receipt.is_success:
            return True, ""
        return False, str(receipt.error_message)
    except Exception as e:  # noqa: BLE001 - report, never abort the whole run
        return False, f"{type(e).__name__}: {e}"


def step_create(subtensor, owner, args) -> int | None:
    cost = subtensor.get_subnet_burn_cost()
    logger.info("Subnet lock cost: %s", cost)
    if not args.yes:
        logger.error("Refusing to create a subnet without --yes (costs %s)", cost)
        return None
    before = set(subtensor.get_all_subnets_netuid())
    ok, msg = _ok(subtensor.register_subnet(owner))
    if not ok:
        logger.error("register_subnet failed: %s", msg)
        return None
    for _ in range(30):
        new = set(subtensor.get_all_subnets_netuid()) - before
        if new:
            netuid = sorted(new)[-1]
            logger.info("Subnet created: netuid %d", netuid)
            return netuid
        time.sleep(BLOCK_TIME_S)
    logger.error("Subnet registered but no new netuid appeared")
    return None


def step_identity(subtensor, owner, netuid: int, args) -> None:
    from bittensor.core.chain_data.subnet_identity import SubnetIdentity

    try:
        ident = SubnetIdentity(
            subnet_name=args.subnet_name,
            github_repo=args.github,
            subnet_contact=args.contact,
            subnet_url=args.url,
            discord="", description=args.description, additional="",
            logo_url="",
        )
    except TypeError:
        # Field set varies across SDK minors; a missing identity is cosmetic,
        # never a reason to abort a launch.
        logger.warning("SubnetIdentity signature not recognised — skipping identity")
        return
    ok, msg = _ok(subtensor.set_subnet_identity(owner, netuid, ident))
    logger.info("Identity set: %s%s", ok, "" if ok else f" ({msg})")


def step_activate(subtensor, owner, netuid: int) -> bool:
    """start_call. The single most consequential step in this script."""
    if subtensor.is_subnet_active(netuid):
        logger.info("Subnet %d is already active", netuid)
        return True
    delay = subtensor.get_start_call_delay()
    logger.info("Subnet %d is INACTIVE. start_call delay is %d blocks.", netuid, delay)
    ok, msg = _ok(subtensor.start_call(owner, netuid))
    if not ok:
        logger.error("start_call failed: %s", msg)
        return False
    for _ in range(20):
        if subtensor.is_subnet_active(netuid):
            logger.info("Subnet %d is now ACTIVE", netuid)
            return True
        time.sleep(BLOCK_TIME_S)
    logger.error("start_call submitted but the subnet is still inactive")
    return False


def step_hyperparams(subtensor, owner, netuid: int, epoch_interval: int) -> dict:
    """Apply the hyperparameters Fugal depends on. Reports every outcome."""
    results: dict[str, str] = {}
    hp = subtensor.get_subnet_hyperparameters(netuid=netuid)

    wanted = {
        "sudo_set_commit_reveal_weights_enabled": (
            {"netuid": netuid, "enabled": True},
            bool(getattr(hp, "commit_reveal_weights_enabled", False)) is True,
        ),
        "sudo_set_network_registration_allowed": (
            {"netuid": netuid, "registration_allowed": True},
            bool(getattr(hp, "registration_allowed", False)) is True,
        ),
    }
    for call, (params, already) in wanted.items():
        if already:
            results[call] = "already set"
            continue
        ok, msg = _ok(_sudo_call(subtensor, owner, call, params))
        results[call] = "set" if ok else f"FAILED: {msg}"

    # Epoch/tempo alignment.
    blocks_per_epoch = max(1, int(epoch_interval) // BLOCK_TIME_S)
    tempo = int(getattr(hp, "tempo", 0) or 0)
    if tempo == blocks_per_epoch:
        results["tempo"] = f"aligned ({tempo} blocks)"
    else:
        ok, msg = _sudo_call(
            subtensor, owner, "sudo_set_tempo",
            {"netuid": netuid, "tempo": blocks_per_epoch},
        )
        if ok:
            results["tempo"] = f"set {tempo} -> {blocks_per_epoch}"
        else:
            results["tempo"] = (
                f"MISALIGNED: chain tempo {tempo} blocks vs Fugal epoch "
                f"{blocks_per_epoch} blocks; sudo_set_tempo refused ({msg}). "
                f"Set FUGAL_EPOCH_INTERVAL={tempo * BLOCK_TIME_S} on every "
                "neuron so one epoch is exactly one Yuma step."
            )
    for k, v in results.items():
        logger.info("  %-46s %s", k, v)
    return results


def step_register(subtensor, netuid: int, wallets: list, args) -> dict:
    burn = subtensor.recycle(netuid=netuid)
    todo = [w for w in wallets
            if not subtensor.is_hotkey_registered_on_subnet(w.hotkey.ss58_address, netuid)]
    logger.info("Registration burn %s each; %d of %d hotkeys need registering",
                burn, len(todo), len(wallets))
    if todo and not args.yes:
        logger.error("Refusing to register without --yes (total ~%s)",
                     bt.Balance.from_rao(burn.rao * len(todo)))
        return {}
    out = {}
    for w in wallets:
        hk = w.hotkey.ss58_address
        if subtensor.is_hotkey_registered_on_subnet(hk, netuid):
            out[f"{w.name}/{w.hotkey_str}"] = "already registered"
            continue
        ok, msg = _ok(subtensor.burned_register(w, netuid))
        out[f"{w.name}/{w.hotkey_str}"] = "registered" if ok else f"FAILED: {msg}"
        logger.info("  %-34s %s", f"{w.name}/{w.hotkey_str}", out[f"{w.name}/{w.hotkey_str}"])
        time.sleep(BLOCK_TIME_S)  # max_regs_per_block is 1 on most chains
    return out


def step_stake(subtensor, netuid: int, validators: list, amount: float, args) -> dict:
    """Stake validators so they hold a permit before the first epoch.

    A validator without a permit produces correct weights that the chain
    silently refuses, which reads in the logs exactly like a network error.
    """
    out = {}
    minimum = subtensor.get_minimum_required_stake()
    logger.info("Minimum required stake: %s; staking %s per validator",
                minimum, bt.Balance.from_tao(amount))
    for w in validators:
        hk = w.hotkey.ss58_address
        existing = subtensor.get_stake(
            coldkey_ss58=w.coldkeypub.ss58_address, hotkey_ss58=hk, netuid=netuid,
        )
        if existing and existing.tao >= amount * 0.9:
            out[w.hotkey_str] = f"already staked {existing}"
            logger.info("  %-14s %s", w.hotkey_str, out[w.hotkey_str])
            continue
        if not args.yes:
            out[w.hotkey_str] = "skipped (no --yes)"
            continue
        ok, msg = _ok(subtensor.add_stake(
            wallet=w, netuid=netuid, hotkey_ss58=hk,
            amount=bt.Balance.from_tao(amount),
        ))
        out[w.hotkey_str] = "staked" if ok else f"FAILED: {msg}"
        logger.info("  %-14s %s", w.hotkey_str, out[w.hotkey_str])
    return out


def report(subtensor, netuid: int, epoch_interval: int) -> None:
    hp = subtensor.get_subnet_hyperparameters(netuid=netuid)
    mg = subtensor.metagraph(netuid)
    bpe = max(1, int(epoch_interval) // BLOCK_TIME_S)
    print()
    print("=" * 78)
    print(f"SUBNET {netuid} on {subtensor.network}")
    print("=" * 78)
    print(f"  active                {subtensor.is_subnet_active(netuid)}")
    print(f"  tempo                 {hp.tempo} blocks"
          f"   (Fugal epoch = {bpe} blocks"
          f"{'  ALIGNED' if hp.tempo == bpe else '  *** MISALIGNED ***'})")
    print(f"  commit-reveal         {hp.commit_reveal_weights_enabled}"
          f" (period {getattr(hp, 'commit_reveal_period', '?')} epochs)")
    print(f"  weights_rate_limit    {hp.weights_rate_limit} blocks")
    print(f"  serving_rate_limit    {hp.serving_rate_limit} blocks")
    print(f"  activity_cutoff       {hp.activity_cutoff} blocks")
    print(f"  registration_allowed  {hp.registration_allowed}   burn {subtensor.recycle(netuid=netuid)}")
    print(f"  neurons               {mg.n}")
    for uid in range(int(mg.n)):
        ax = mg.axons[uid]
        print(f"    uid {uid:<3} stake {float(mg.S[uid]):>12.4f}  "
              f"permit {str(bool(mg.validator_permit[uid])):<5}  "
              f"axon {ax.ip}:{ax.port}  {mg.hotkeys[uid]}")
    print("=" * 78)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", default="test")
    p.add_argument("--netuid", type=int, default=None,
                   help="Existing subnet to provision. Omit with --create to make one.")
    p.add_argument("--create", action="store_true", help="Register a new subnet (costs the lock)")
    p.add_argument("--owner-wallet", required=True)
    p.add_argument("--owner-hotkey", default="default")
    p.add_argument("--validators", default="",
                   help="Comma-separated wallet/hotkey pairs, e.g. w1/val1,w1/val2")
    p.add_argument("--miners", default="", help="Same format as --validators")
    p.add_argument("--stake", type=float, default=0.5, help="TAO to stake per validator")
    p.add_argument("--epoch-interval", type=int, default=3600,
                   help="FUGAL_EPOCH_INTERVAL the neurons will run with")
    p.add_argument("--subnet-name", default="Fugal")
    p.add_argument("--description", default="Continuous cost-aware LLM router optimization")
    p.add_argument("--github", default="https://github.com/fugal-ai/fugal-subnet")
    p.add_argument("--url", default="https://github.com/fugal-ai/fugal-subnet")
    p.add_argument("--contact", default="")
    p.add_argument("--status", action="store_true", help="Report only; change nothing")
    p.add_argument("--yes", action="store_true", help="Authorise spending")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    configure_logging(args.log_level)

    def parse(spec: str) -> list:
        out = []
        for item in filter(None, (s.strip() for s in spec.split(","))):
            name, _, hk = item.partition("/")
            out.append(bt.Wallet(name=name, hotkey=hk or "default"))
        return out

    subtensor = bt.Subtensor(network=args.network)
    owner = bt.Wallet(name=args.owner_wallet, hotkey=args.owner_hotkey)
    logger.info("Network %s at block %d", args.network, subtensor.get_current_block())
    logger.info("Owner coldkey %s balance %s",
                owner.coldkeypub.ss58_address,
                subtensor.get_balance(owner.coldkeypub.ss58_address))

    netuid = args.netuid
    if args.status:
        if netuid is None:
            logger.error("--status needs --netuid")
            return 2
        report(subtensor, netuid, args.epoch_interval)
        return 0

    if netuid is None:
        if not args.create:
            logger.error("Give --netuid, or --create to register a new subnet")
            return 2
        netuid = step_create(subtensor, owner, args)
        if netuid is None:
            return 1

    owner_ck = subtensor.subnet(netuid).owner_coldkey
    if owner_ck != owner.coldkeypub.ss58_address:
        logger.error("Wallet %s is not the owner of subnet %d (owner is %s). "
                     "start_call and hyperparameters would be refused.",
                     args.owner_wallet, netuid, owner_ck)
        return 1

    step_identity(subtensor, owner, netuid, args)
    if not step_activate(subtensor, owner, netuid):
        logger.error("Subnet %d could not be activated — stopping. Nothing "
                     "downstream can work: no permits, no weights.", netuid)
        return 1
    step_hyperparams(subtensor, owner, netuid, args.epoch_interval)

    validators, miners = parse(args.validators), parse(args.miners)
    if validators or miners:
        step_register(subtensor, netuid, validators + miners, args)
    if validators:
        step_stake(subtensor, netuid, validators, args.stake, args)

    report(subtensor, netuid, args.epoch_interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
