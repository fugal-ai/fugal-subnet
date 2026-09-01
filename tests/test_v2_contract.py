from __future__ import annotations

import copy

import pytest

from fugal_subnet.consensus_manifest import load_consensus_manifest
from fugal_subnet.v2.contract import ContractMismatch, verify_executable_contract


def _consensus():
    manifest = load_consensus_manifest("local")
    protocol = next(item for item in manifest.protocols if item.protocol_id == "v2")
    assert protocol.consensus is not None
    return copy.deepcopy(protocol.consensus)


def test_packaged_manifest_matches_every_executable_constant():
    verify_executable_contract(_consensus())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("committee", "maximum_response_bytes", 8192),
        ("committee", "api_concurrency", 100),
        ("rounding", "routing_decimals", 8),
    ],
)
def test_manifest_constant_drift_fails_closed(section, key, value):
    consensus = _consensus()
    consensus[section][key] = value
    with pytest.raises(ContractMismatch):
        verify_executable_contract(consensus)


def test_evaluation_parameter_drift_fails_closed():
    consensus = _consensus()
    consensus["rounding"]["evaluation"]["max_weight_delta"] = "0.4"
    with pytest.raises(ContractMismatch, match="evaluation parameters"):
        verify_executable_contract(consensus)
