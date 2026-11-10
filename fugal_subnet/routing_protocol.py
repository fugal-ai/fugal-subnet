"""Fresh success-policy evidence namespace. No legacy proof/state admission."""
from __future__ import annotations

import json
from pathlib import Path

from fugal_subnet.vendor import success_contract as contract

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "routing_data" / "benchmark_tokens_v1.json"
PRICES_PATH = ROOT / "routing_data" / "models.json"
PRICES_PATH_V2 = ROOT / "routing_data" / "models_v2.json"
MANIFEST_PATH_V2 = ROOT / "routing_data" / "benchmark_tokens_protocol_v2.json"
BENCHMARK_LAMBDA = 1.0  # Consensus constant; never a miner environment setting.


# "fugal-testnet-success-v1" is a protocol name, not a network name. It is
# hashed into every reveal and into the bootstrap frame built from testnet
# reveals, so renaming it for mainnet, or changing models.json or the manifest
# between the warm-up and the launch, is a new identity that rejects that
# frame. The network is --network / FUGAL_NETWORK, never this string.
def identity():
    return contract.digest({"protocol": "fugal-testnet-success-v1", "contract": contract.CONTRACT,
                            "profile": contract.PROFILE_ID, "lambda": BENCHMARK_LAMBDA,
                            "tokens": contract.file_hash(MANIFEST_PATH),
                            "prices": contract.file_hash(PRICES_PATH)})


def benchmark_costs(head, require_reviewed=False):
    if head.success is None:
        raise ValueError("legacy preference heads cannot enter the success benchmark")
    manifest = json.loads(MANIFEST_PATH.read_text())
    contract.check_manifest(head.success, manifest, deployable=require_reviewed)
    rates = {r["id"]: (r["in"] / 1e6, r["out"] / 1e6) for r in json.loads(PRICES_PATH.read_text())}
    return contract.estimated_cost(head.success, rates)


def require_identity(value):
    if value != identity():
        raise ValueError("incompatible routing evidence namespace; archive old state and start fresh")


# --- Protocol v2: one versioned accessor for the model list -----------------
#
# A release/v2 validator scores v1 epochs until V2_ACTIVATION_BLOCK and v2
# epochs after it, so both price tables ship side by side and every caller asks
# for the table of the epoch it is scoring, keyed on the ONE version switch:
# slicer.protocol_version_for_boundary(boundary_block, V2_ACTIVATION_BLOCK).
# The global model index space (sorted ids) comes from here too, so a window
# never mixes the two index spaces (the v2 window starts empty at the flip).
# identity() above is deliberately unchanged: it is hashed into every v1 reveal
# and the v1 frame, and folding the v2 table into it would reject both
# mid-run. The v2 table's hash is in the fingerprint's epoch block instead.


def prices_path(version: int) -> Path:
    if version == 1:
        return PRICES_PATH
    if version == 2:
        return PRICES_PATH_V2
    raise ValueError(f"unknown protocol version {version!r}")


def _table(version: int) -> list[dict]:
    return json.loads(prices_path(version).read_text())


def price_table(version: int) -> dict[str, tuple[float, float]]:
    """{model_id: (price per input token, price per output token)} for this protocol."""
    return {r["id"]: (r["in"] / 1e6, r["out"] / 1e6) for r in _table(version)}


def models_for_protocol(version: int) -> list[str]:
    """The global model index space for this protocol: the sorted model ids."""
    return sorted(r["id"] for r in _table(version))


def reasoning_for(version: int, model_id: str) -> dict | None:
    """The single OpenRouter `reasoning` object pinned for this model (v2 only).

    The unified parameter accepts `effort` OR `max_tokens`, never both, and each
    provider honours one of them; the table pins exactly one per model. v1 sends
    none.
    """
    if version == 1:
        return None
    for r in _table(version):
        if r["id"] == model_id:
            return dict(r["reasoning"]) if r.get("reasoning") else None
    raise KeyError(f"{model_id!r} is not in the v{version} model list")


def manifest_path(version: int) -> Path:
    if version == 1:
        return MANIFEST_PATH
    if version == 2:
        return MANIFEST_PATH_V2
    raise ValueError(f"unknown protocol version {version!r}")


def identity_v2():
    """The v2 evidence namespace: the v2 token manifest and price table, under a new protocol
    name. A v2 reveal or frame cannot be mistaken for a v1 one, and v1's identity() is
    untouched."""
    return contract.digest({"protocol": "fugal-success-v2", "contract": contract.CONTRACT,
                            "profile": contract.PROFILE_ID, "lambda": BENCHMARK_LAMBDA,
                            "tokens": contract.file_hash(MANIFEST_PATH_V2),
                            "prices": contract.file_hash(PRICES_PATH_V2)})


def benchmark_costs_for(head, version: int, require_reviewed: bool = False):
    """benchmark_costs for the epoch's protocol: its token manifest and price table."""
    if version == 1:
        return benchmark_costs(head, require_reviewed=require_reviewed)
    if head.success is None:
        raise ValueError("legacy preference heads cannot enter the success benchmark")
    manifest = json.loads(manifest_path(version).read_text())
    contract.check_manifest(head.success, manifest, deployable=require_reviewed)
    return contract.estimated_cost(head.success, price_table(version))
