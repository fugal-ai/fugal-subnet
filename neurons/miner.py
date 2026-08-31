#!/usr/bin/env python3
"""Fugal subnet miner — axon server for head submission.

Usage:
    python neurons/miner.py --netuid 1 --coldkey fugal_miner_1 --hotkey default --head-path data/my_head.npz

On startup the miner commits sha256(head bytes) on-chain (Commitments pallet).
Validators only score heads whose hash was committed at or before the epoch
boundary block, so a freshly (re)started miner becomes scoreable from the next
epoch after its commitment lands.
"""
import base64
import hashlib
import logging
import os
import threading
import time
from typing import Tuple

import click
import numpy as np

logger = logging.getLogger("fugal.miner")

METAGRAPH_REFRESH_S = 300


@click.command()
@click.option("--network", default=lambda: os.getenv("FUGAL_NETWORK", "test"),
              help="Network: finney, test, local")
@click.option("--netuid", type=int,
              default=lambda: int(os.getenv("FUGAL_NETUID", "1")),
              help="Subnet netuid")
@click.option("--coldkey", default=lambda: os.getenv("WALLET_NAME", "default"),
              help="Wallet coldkey name")
@click.option("--hotkey", default=lambda: os.getenv("HOTKEY_NAME", "default"),
              help="Hotkey name")
@click.option("--port", default=lambda: int(os.getenv("FUGAL_MINER_PORT", "8091")),
              type=int, help="Axon port")
@click.option("--head-path", required=True, type=click.Path(exists=True),
              help="Path to .npz head artifact")
@click.option("--log-level",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              default=lambda: os.getenv("LOG_LEVEL", "INFO"),
              help="Logging level")
def main(network, netuid, coldkey, hotkey, port, head_path, log_level):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    import bittensor as bt
    from fugal_subnet.protocol import FugalSynapse
    from fugal_subnet.config import MIN_VALIDATOR_STAKE
    from fugal_subnet.commitments import ensure_commitment

    head_data = _load_head_file(head_path)
    head_b64 = base64.b64encode(head_data).decode("ascii")
    head_hash = hashlib.sha256(head_data).hexdigest()

    with np.load(head_path, allow_pickle=False) as npz:
        model_pool = [str(m) for m in npz["models"]]

    logger.info("Head loaded: %s (%d bytes, %d models)",
                head_path, len(head_data), len(model_pool))
    logger.info("Head hash: %s", head_hash[:16])

    wallet = bt.Wallet(name=coldkey, hotkey=hotkey)
    subtensor = bt.Subtensor(network=network)
    metagraph = subtensor.metagraph(netuid)

    my_hotkey = wallet.hotkey.ss58_address
    if my_hotkey not in metagraph.hotkeys:
        logger.error("Hotkey %s not registered on netuid %d", my_hotkey, netuid)
        return
    my_uid = metagraph.hotkeys.index(my_hotkey)
    logger.info("Miner UID %d on %s netuid %d", my_uid, network, netuid)

    committed = ensure_commitment(subtensor, wallet, netuid, head_hash)
    if not committed:
        logger.warning("Head hash NOT committed on-chain yet — will retry. "
                       "Validators will not score this head until it lands.")

    # Shared metagraph snapshot for the blacklist (refreshed by the main loop).
    mg_lock = threading.Lock()
    mg_state = {"metagraph": metagraph}

    def blacklist(synapse: FugalSynapse) -> Tuple[bool, str]:
        """Only registered hotkeys may pull the head; with a configured stake
        floor, only validators (permit or stake) may. This stops competitors
        from copying the head with a plain unregistered dendrite query."""
        caller = getattr(synapse.dendrite, "hotkey", None)
        if not caller:
            return True, "no caller hotkey"
        with mg_lock:
            mg = mg_state["metagraph"]
        if caller not in mg.hotkeys:
            return True, f"unregistered hotkey {caller[:8]}..."
        if MIN_VALIDATOR_STAKE > 0:
            uid = mg.hotkeys.index(caller)
            permit = False
            try:
                permit = bool(mg.validator_permit[uid])
            except (AttributeError, IndexError):
                pass
            stake = 0.0
            try:
                stake = float(mg.S[uid])
            except (AttributeError, IndexError, TypeError):
                pass
            if not permit and stake < MIN_VALIDATOR_STAKE:
                return True, f"caller {caller[:8]}... lacks validator permit/stake"
        return False, "ok"

    def forward(synapse: FugalSynapse) -> FugalSynapse:
        logger.info("Epoch %s: serving head (hash=%s)", synapse.epoch_id, head_hash[:16])
        synapse.head_npz_b64 = head_b64
        synapse.model_pool = model_pool
        synapse.head_commit_hash = head_hash
        return synapse

    external_ip = os.getenv("FUGAL_AXON_IP")
    axon = bt.Axon(wallet=wallet, port=port,
                    external_ip=external_ip or None)
    axon.attach(forward_fn=forward, blacklist_fn=blacklist)

    for attempt in range(5):
        try:
            axon.serve(netuid=netuid, subtensor=subtensor)
        except Exception as e:
            logger.warning("axon.serve() attempt %d failed: %s", attempt + 1, e)
        metagraph = subtensor.metagraph(netuid)
        registered_ip = metagraph.axons[my_uid].ip
        if registered_ip not in ("0.0.0.0", ""):
            logger.info("Axon registered on chain: %s:%d", registered_ip, port)
            break
        if attempt < 4:
            logger.warning("Axon IP not registered yet, retrying in 15s...")
            time.sleep(15)
        else:
            logger.error("axon.serve() failed after 5 attempts — IP still unregistered")

    axon.start()
    logger.info("Miner axon serving on port %d", port)

    try:
        last_refresh = time.time()
        while True:
            time.sleep(30)
            if not committed:
                committed = ensure_commitment(subtensor, wallet, netuid, head_hash)
            if time.time() - last_refresh >= METAGRAPH_REFRESH_S:
                try:
                    fresh = subtensor.metagraph(netuid)
                    with mg_lock:
                        mg_state["metagraph"] = fresh
                except Exception as e:
                    logger.warning("Metagraph refresh failed: %s", e)
                last_refresh = time.time()
    except KeyboardInterrupt:
        logger.info("Miner stopped by user")
    finally:
        axon.stop()
        logger.info("Miner shutdown complete")


def _load_head_file(path: str) -> bytes:
    """Load and validate the head .npz file (same checks the validator runs)."""
    from fugal_subnet.head_eval import load_head_from_npz
    with open(path, "rb") as f:
        data = f.read()
    try:
        load_head_from_npz(data)
    except ValueError as e:
        raise click.ClickException(f"Head failed validation: {e}")
    return data


if __name__ == "__main__":
    main()
