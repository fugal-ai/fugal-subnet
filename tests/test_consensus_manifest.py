"""Consensus-manifest and immutable v1 grader regression tests."""

from __future__ import annotations

import hashlib
import importlib.resources
import json

import pytest

import fugal_subnet.consensus_manifest as manifest_module
from fugal_subnet.consensus_manifest import (
    V1_GRADER_SHA256,
    ManifestError,
    ProtocolUnavailable,
    canonical_json,
    load_consensus_manifest,
    select_protocol,
    verify_runtime_dependencies,
)
from fugal_subnet.grader_registry import get_grader, grade_for_protocol

V1_MODEL_SNAPSHOT_SHA256 = (
    "26b54ef396d5a92f3a03e6c1bb5a87011eb40ec007803addce0c65ac5bcb7e4a"
)


def test_packaged_manifest_preserves_v1_and_disables_v2():
    manifest = load_consensus_manifest("test")
    protocols = {protocol.protocol_id: protocol for protocol in manifest.protocols}

    assert protocols["v1"].consensus["grader"]["sha256"] == V1_GRADER_SHA256
    assert protocols["v1"].consensus["benchmarks"]["canonical"] is False
    assert protocols["v1"].status == "historical_experimental"
    assert protocols["v2"].status == "development_disabled"
    assert protocols["v2"].enabled is False
    assert protocols["v2"].complete is False
    assert all(block is None for block in protocols["v2"].activation_blocks.values())
    assert manifest.sha256 == hashlib.sha256(canonical_json(manifest.raw)).hexdigest()


def test_protocol_selection_is_block_and_network_scoped():
    assert select_protocol("local", 0).protocol_id == "v1"
    assert select_protocol("ws://127.0.0.1:9944", 123).protocol_id == "v1"
    assert select_protocol("test", 999).protocol_id == "v1"

    with pytest.raises(ProtocolUnavailable):
        select_protocol("finney", 999)


def test_manifest_override_is_local_only(tmp_path, monkeypatch):
    packaged = load_consensus_manifest("local")
    override = tmp_path / "manifest.json"
    override.write_text(json.dumps(packaged.raw), encoding="utf-8")
    monkeypatch.setenv("FUGAL_CONSENSUS_MANIFEST", str(override))

    assert load_consensus_manifest("local").sha256 == packaged.sha256
    with pytest.raises(ManifestError, match="local/mock-only"):
        load_consensus_manifest("test")


def test_incomplete_v2_cannot_be_enabled_or_activated(tmp_path):
    raw = load_consensus_manifest("local").raw
    raw = json.loads(json.dumps(raw))
    v2 = next(protocol for protocol in raw["protocols"] if protocol["id"] == "v2")
    v2["enabled"] = True
    v2["activation_blocks"]["local"] = 10
    override = tmp_path / "unsafe.json"
    override.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match="must be complete"):
        load_consensus_manifest("local", override)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda consensus: consensus["committee"].update(epoch_blocks=None), "schedule"),
        (
            lambda consensus: consensus["benchmarks"].update(
                rollout_blockers=["unresolved"]
            ),
            "benchmark rollout blockers",
        ),
        (
            lambda consensus: consensus["model_registry"].update(active_models=[]),
            "approved active models",
        ),
        (
            lambda consensus: consensus["worker"].update(image_digest="local-only"),
            "worker registry digest",
        ),
        (
            lambda consensus: consensus["committee"].update(minimum_reports=2),
            "five-builder/three-report quorum",
        ),
    ],
)
def test_complete_v2_requires_every_rollout_gate(tmp_path, mutation, message):
    raw = json.loads(json.dumps(load_consensus_manifest("local").raw))
    v2 = next(protocol for protocol in raw["protocols"] if protocol["id"] == "v2")
    v2["complete"] = True
    consensus = v2["consensus"]
    consensus["committee"].update({
        "epoch_blocks": 360,
        "precommit_deadline_offset_blocks": 60,
        "report_deadline_offset_blocks": 300,
    })
    consensus["benchmarks"]["rollout_blockers"] = []
    consensus["model_registry"]["active_models"] = ["provider/model"]
    consensus["worker"]["image_digest"] = "sha256:" + "a" * 64
    mutation(consensus)
    override = tmp_path / "incomplete-material.json"
    override.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match=message):
        load_consensus_manifest("local", override)


def test_duplicate_manifest_keys_are_rejected(tmp_path):
    override = tmp_path / "duplicate.json"
    override.write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="Duplicate JSON key"):
        load_consensus_manifest("local", override)


def test_v1_grader_bytes_and_legacy_semantics_remain_exact():
    grader = get_grader("v1")
    assert grader.grader_hash() == f"sha256:{V1_GRADER_SHA256}"

    # Historical verification must retain this known v1 false positive. The v2
    # integer grader will reject it under a separately activated contract.
    task = {"checker": {"id": "integer_exact"}, "gold": "42", "domain": "aime"}
    assert grade_for_protocol("v1", task, "42.9") == 1

    grader_v2 = get_grader("v2")
    assert grader_v2.__name__ == "fugal_subnet.graders_v2"
    manifest = load_consensus_manifest("local")
    protocol_v2 = next(item for item in manifest.protocols if item.protocol_id == "v2")
    assert not protocol_v2.enabled
    assert not protocol_v2.complete
    assert (
        grader_v2.grader_hash().removeprefix("sha256:")
        == protocol_v2.consensus["grader"]["sha256"]
    )


def test_v1_model_fallback_is_a_hashed_package_resource():
    resource = importlib.resources.files("fugal_subnet").joinpath(
        "model-registry-v1.json"
    )
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == V1_MODEL_SNAPSHOT_SHA256


def test_runtime_consensus_dependencies_are_exact(monkeypatch):
    manifest = load_consensus_manifest("local")
    protocol = next(item for item in manifest.protocols if item.protocol_id == "v2")
    assert protocol.consensus is not None
    verify_runtime_dependencies(protocol.consensus)

    real_version = manifest_module.version
    monkeypatch.setattr(
        manifest_module,
        "version",
        lambda name: "0.0.0" if name == "numpy" else real_version(name),
    )
    with pytest.raises(ManifestError, match="numpy differs"):
        verify_runtime_dependencies(protocol.consensus)
