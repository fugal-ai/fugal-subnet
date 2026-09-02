#!/usr/bin/env python3
"""No-chain smoke test for real Bittensor v10 Axon annotation inspection."""

import tempfile

import bittensor as bt

from fugal_subnet.protocol import FugalSynapse


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fugal-axon-smoke-") as wallet_path:
        wallet = bt.Wallet(name="smoke", hotkey="default", path=wallet_path)
        wallet.create_if_non_existent(
            coldkey_use_password=False,
            hotkey_use_password=False,
            suppress=True,
        )
        axon = bt.Axon(wallet=wallet, port=0)

        def miner_forward(synapse: FugalSynapse) -> FugalSynapse:
            return synapse

        axon.attach(miner_forward)
        attached = axon.forward_class_types
        if attached.get("FugalSynapse") is not FugalSynapse:
            raise SystemExit("FugalSynapse was not registered by bt.Axon.attach()")
        print("Bittensor v10 Axon attach smoke test passed.")


if __name__ == "__main__":
    main()
