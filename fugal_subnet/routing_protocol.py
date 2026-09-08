"""Fresh success-policy evidence namespace. No legacy proof/state admission."""
from __future__ import annotations

import json
from pathlib import Path

from fugal_subnet.vendor import success_contract as contract

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "routing_data" / "benchmark_tokens_v1.json"
PRICES_PATH = ROOT / "routing_data" / "models.json"
BENCHMARK_LAMBDA = 1.0  # Consensus constant; never a miner environment setting.


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
