"""Lightweight bittensor mock for Windows dev testing.

Bittensor's Rust-based packages (bittensor-core, bittensor-wallet) require
Perl + OpenSSL to compile on Windows. Since the subnet deploys on Linux,
this mock provides just enough API surface for import-testing and local
integration tests on Windows.

Usage:
    import tests.bt_mock  # patches sys.modules["bittensor"]
    import bittensor as bt  # now uses the mock
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pydantic
from bittensor_wallet import Keypair

# --- Synapse base ---

class Synapse(pydantic.BaseModel):
    """Pydantic-backed bt.Synapse stand-in with real field-bound validation."""

    model_config = pydantic.ConfigDict(validate_assignment=True)

    def deserialize(self) -> "Synapse":
        # Mirrors the real contract: deserialize() MUST return self —
        # returning a dict makes dendrite hand dicts to the validator.
        return self


# --- Wallet ---

class _Hotkey:
    ss58_address: str = "5MockHotkey000000000000000000000000000000000000"

    def __init__(self, ss58: str | None = None):
        if ss58:
            self.ss58_address = ss58


class Wallet:
    def __init__(self, name: str = "default", hotkey: str = "default", **kw):
        self.name = name
        self._hotkey_name = hotkey
        self.hotkey = _Hotkey()


# --- Subtensor ---

class Subtensor:
    def __init__(self, network: str = "test", **kw):
        self.network = network
        self._block = 1000

    def metagraph(self, netuid: int) -> "Metagraph":
        return Metagraph(netuid=netuid)

    def get_current_block(self) -> int:
        self._block += 1
        return self._block

    def set_weights(self, *, wallet, netuid, uids, weights,
                    wait_for_inclusion=True, wait_for_finalization=False):
        return (True, "mock success")


# --- Metagraph ---

@dataclass
class AxonInfo:
    ip: str = "127.0.0.1"
    port: int = 8091
    hotkey: str = ""


@dataclass
class Metagraph:
    netuid: int = 1
    n: int = 2
    hotkeys: list[str] = field(default_factory=lambda: [
        "5BurnUID0000000000000000000000000000000000000000",
        "5MockHotkey000000000000000000000000000000000000",
    ])
    axons: list[AxonInfo] = field(default_factory=lambda: [
        AxonInfo(hotkey="5BurnUID0000000000000000000000000000000000000000"),
        AxonInfo(hotkey="5MockHotkey000000000000000000000000000000000000"),
    ])


# --- Dendrite ---

class Dendrite:
    def __init__(self, wallet: Wallet | None = None, **kw):
        self.wallet = wallet
        self._mock_responses: list[Any] = []

    def set_mock_responses(self, responses: list):
        self._mock_responses = responses

    def query(self, axons, synapse, timeout=120, **kw) -> list:
        if self._mock_responses:
            return self._mock_responses
        return [None] * len(axons)


# --- Axon ---

class Axon:
    def __init__(self, wallet: Wallet | None = None, port: int = 8091, **kw):
        self.wallet = wallet
        self.port = port
        self._forward_fn = None

    def attach(self, forward_fn=None, **kw) -> "Axon":
        self._forward_fn = forward_fn
        return self

    def serve(self, netuid: int = 1, subtensor=None, **kw) -> "Axon":
        return self

    def start(self) -> "Axon":
        return self

    def stop(self):
        pass


# --- Patch sys.modules ---

_bt = types.ModuleType("bittensor")
_bt.Synapse = Synapse
_bt.Wallet = Wallet
_bt.Subtensor = Subtensor
_bt.Dendrite = Dendrite
_bt.Axon = Axon
_bt.Metagraph = Metagraph
_bt.Keypair = Keypair
_bt.__version__ = "mock-dev"

sys.modules["bittensor"] = _bt
