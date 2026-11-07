#!/usr/bin/env python3
"""Local-chain helpers for scripts/dress_rehearsal.py. Not run directly.

The primitives a rehearsal needs on the pinned local subtensor (Alice's
pre-funded devnet keys): create, activate and de-rate-limit a subnet, fund,
register and stake wallets, pick an axon IP the chain accepts, size an epoch
from the measured block rate, and a deterministic stub model upstream the
REAL metering proxy forwards to (no API key, no spend).

This file used to also run a one-epoch in-container demo
(docker-compose.localnet.yml) with its own copy of the validator epoch. It was
v1-only, re-implemented what dress_rehearsal runs for real, and stopped
working when the image stopped carrying scripts/; it was removed.
"""
import json
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("fugal.setup")


def get_alice_keypair():
    """Get Alice's well-known devnet keypair."""
    try:
        from substrateinterface import Keypair
    except ImportError:
        from bittensor_wallet import Keypair
    return Keypair.create_from_uri("//Alice")


def submit_extrinsic(subtensor, call, keypair):
    """Submit a signed extrinsic using substrate interface directly."""
    extrinsic = subtensor.substrate.create_signed_extrinsic(call=call, keypair=keypair)
    receipt = subtensor.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
    if receipt.is_success:
        return True
    logger.error("  Extrinsic failed: %s", getattr(receipt, "error_message", "unknown"))
    return False


def fund_wallet(subtensor, dest_wallet, amount_tao=10000):
    """Transfer TAO from Alice to a test wallet using substrate directly."""
    alice_kp = get_alice_keypair()
    dest_addr = dest_wallet.coldkeypub.ss58_address
    logger.info("Funding %s (%s) with %d TAO...", dest_wallet.name, dest_addr, amount_tao)

    call = subtensor.substrate.compose_call(
        call_module="Balances",
        call_function="transfer_keep_alive",
        call_params={
            "dest": dest_addr,
            "value": int(amount_tao * 1e9),
        },
    )
    success = submit_extrinsic(subtensor, call, alice_kp)
    if success:
        new_balance = subtensor.get_balance(dest_addr)
        logger.info("  Funded successfully, new balance: %s", new_balance)
    else:
        logger.error("  Funding FAILED")
    return success


def create_subnet(subtensor):
    """Create a subnet using Alice's pre-funded account via substrate directly."""
    alice_kp = get_alice_keypair()
    balance = subtensor.get_balance(alice_kp.ss58_address)
    logger.info("Alice balance: %s", balance)
    logger.info("Creating subnet...")

    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="register_network",
        call_params={"hotkey": alice_kp.ss58_address},
    )
    success = submit_extrinsic(subtensor, call, alice_kp)
    if success:
        logger.info("Subnet created successfully")
    else:
        logger.warning("Subnet creation failed — may already exist")

    netuids = subtensor.get_all_subnets_netuid()
    logger.info("Active netuids: %s", netuids)
    netuid = max(netuids) if netuids else 1
    logger.info("Using netuid %d", netuid)
    return netuid


def register_wallet(subtensor, wallet, netuid):
    """Register a wallet on the subnet."""
    logger.info("Registering %s on netuid %d...", wallet.name, netuid)
    try:
        success = subtensor.register(wallet=wallet, netuid=netuid)
        if success:
            logger.info("  Registered successfully")
        else:
            logger.info("  Registration returned False — may already be registered")
    except Exception as e:
        logger.warning("  Registration: %s", e)


def lan_ip() -> str:
    """A routable local address the chain will accept for serve_axon.

    Not 127.0.0.1: subtensor rejects loopback for serve_axon at pool validation
    (Custom error 11), bt.Axon.serve() swallows the rejection, and the only
    symptom is an axon stuck at 0.0.0.0 and a miner that silently earns nothing.
    """
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


class StubUpstream:
    """An OpenAI-compatible endpoint the REAL metering proxy forwards to.

    Deliberately not a monkeypatch. The miner runs its real harness, which
    calls its real MeteringProxy, which makes a real HTTP request — the only
    substitution is where that request lands. So token accounting, pricing
    against the pinned table, cost reconciliation and record-keeping are all
    the production code paths, and no API key exists anywhere in this run.

    Replies are a deterministic function of (question, model), so two runs and
    two validators see byte-identical results — which is what makes the
    agreement assertions meaningful rather than coincidental.
    """

    def __init__(self, gold_by_prompt: dict, skill: dict, port: int = 8799):
        self.gold = gold_by_prompt
        self.skill = skill
        self.port = port
        self._server = None
        self._thread = None
        self.calls = 0

    def start(self):
        import hashlib
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                req = json.loads(body)
                model = req.get("model", "")
                prompt = req["messages"][0]["content"]

                digest = hashlib.sha256(f"{prompt}|{model}".encode()).digest()
                correct = (digest[0] / 255.0) < outer.skill.get(model, 0.5)
                text = outer.gold.get(prompt, "0") if correct else "0"
                outer.calls += 1

                payload = json.dumps({
                    "choices": [{"message": {"content": text}}],
                    # Token counts vary a little per model so cost is not a
                    # constant, and thrift has something real to measure.
                    "usage": {
                        "prompt_tokens": 400 + (digest[1] % 200),
                        "completion_tokens": 100 + (digest[2] % 100),
                    },
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"  Stub upstream on 127.0.0.1:{self.port} (no API key, no spend)",
              flush=True)

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None


def calibrate_epoch_interval(subtensor, target_seconds: int = 90,
                             default: str = "3600") -> str:
    """Choose FUGAL_EPOCH_INTERVAL so one epoch lasts ~target_seconds.

    Measured, not assumed. An epoch is EPOCH_INTERVAL // BLOCK_TIME_S blocks,
    so how long it takes in wall-clock depends entirely on the chain — and a
    devnet producing a block every 0.35s runs through a nominal "1 hour" epoch
    in 100 seconds. With a hardcoded interval the miner is structurally unable
    to keep up: it polls every 30s, so it completed epoch e00000042 while the
    validator was already asking about e00000045, and no proof ever matched.
    """
    from fugal_subnet.benchmarks.slicer import BLOCK_TIME_S

    try:
        b0 = subtensor.get_current_block()
        t0 = time.time()
        time.sleep(8)
        rate = (subtensor.get_current_block() - b0) / max(time.time() - t0, 1e-6)
    except Exception as e:
        logger.warning("Could not measure block rate (%s); using default", e)
        return default
    if rate <= 0:
        logger.warning("Chain is not producing blocks; using default interval")
        return default

    want_blocks = max(2, int(target_seconds * rate))
    interval = str(want_blocks * BLOCK_TIME_S)
    logger.info(
        "Chain produces %.2f blocks/s; epoch = %d blocks (~%ds), "
        "FUGAL_EPOCH_INTERVAL=%s", rate, want_blocks, target_seconds, interval,
    )
    return interval


def activate_subnet(subtensor, netuid: int) -> None:
    """Start the subnet. Without this it has no first-emission block, staking is
    rejected with SubtokenDisabled, validators never earn a permit, and
    set_weights fails — so the pipeline runs and then quietly cannot land."""
    try:
        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule", call_function="start_call",
            call_params={"netuid": netuid},
        )
        submit_extrinsic(subtensor, call, get_alice_keypair())
        logger.info("Subnet %d activated (start_call)", netuid)
    except Exception as e:
        logger.warning("start_call failed: %s", e)


def relax_chain_limits(subtensor, netuid: int) -> None:
    """Drop the local chain's transaction rate limits via sudo (Alice).

    A fresh chain ships with TxRateLimit=1000 blocks and ServingRateLimit=50,
    so a miner that commits its head hash and then serves its axon has the
    second extrinsic rejected — `Custom error: 11` — and stays unreachable at
    0.0.0.0. Mainnet spaces these across hour-long epochs; a local run cannot.

    Test-chain configuration, not a subnet change.
    """
    alice = get_alice_keypair()
    for call_name, params in (
        ("sudo_set_tx_rate_limit", {"tx_rate_limit": 0}),
        ("sudo_set_serving_rate_limit", {"netuid": netuid, "serving_rate_limit": 0}),
        ("sudo_set_weights_set_rate_limit",
         {"netuid": netuid, "weights_set_rate_limit": 0}),
    ):
        try:
            inner = subtensor.substrate.compose_call(
                call_module="AdminUtils", call_function=call_name, call_params=params,
            )
            sudo = subtensor.substrate.compose_call(
                call_module="Sudo", call_function="sudo", call_params={"call": inner},
            )
            submit_extrinsic(subtensor, sudo, alice)
            logger.info("chain config: %s -> %s", call_name, list(params.values())[-1])
        except Exception as e:
            logger.warning("chain config %s failed: %s", call_name, e)


def stake_validator(subtensor, wallet, netuid: int, amount: float = 1000.0) -> None:
    """Stake a validator so it earns a permit; without one set_weights fails."""
    import bittensor as bt

    try:
        subtensor.add_stake(
            wallet=wallet, netuid=netuid,
            hotkey_ss58=wallet.hotkey.ss58_address,
            amount=bt.Balance.from_tao(amount), wait_for_inclusion=True,
        )
        logger.info("Staked %s TAO to %s", amount, wallet.name)
    except Exception as e:
        logger.warning("Staking %s failed: %s", wallet.name, e)
