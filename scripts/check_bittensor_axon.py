#!/usr/bin/env python3
"""No-chain smoke test: the synapse annotation survives into Axon registration.

Guards CLAUDE.md rule 1. A ``from __future__ import annotations`` in
neurons/miner.py or fugal_subnet/protocol.py turns the type hints into
strings, and the Axon silently registers nothing. The handlers below make
the same attach call the miner makes (neurons/miner.py), forward and
blacklist with the same signatures, so CI fails wherever the miner would.
"""

import tempfile
from typing import Tuple

import bittensor as bt

from fugal_subnet.protocol import FugalProofSynapse


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fugal-axon-smoke-") as wallet_path:
        wallet = bt.Wallet(name="smoke", hotkey="default", path=wallet_path)
        wallet.create_if_non_existent(
            coldkey_use_password=False,
            hotkey_use_password=False,
            suppress=True,
        )
        axon = bt.Axon(wallet=wallet, port=0)

        def blacklist(synapse: FugalProofSynapse) -> Tuple[bool, str]:
            return False, ""

        def forward(synapse: FugalProofSynapse) -> FugalProofSynapse:
            return synapse

        axon.attach(forward_fn=forward, blacklist_fn=blacklist)
        attached = axon.forward_class_types
        if attached.get("FugalProofSynapse") is not FugalProofSynapse:
            raise SystemExit(
                "FugalProofSynapse was not registered by bt.Axon.attach()"
            )
        print("Axon registered FugalProofSynapse with forward and blacklist.")


if __name__ == "__main__":
    main()
