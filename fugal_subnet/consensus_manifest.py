"""Strict loader and block-based selector for packaged consensus manifests."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import re
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

MANIFEST_RESOURCE = "consensus-manifest.json"
V1_GRADER_SHA256 = "895809dedf0d14c45d9ec046bcbec2f50a09fcf7d31d9996a178e35f3539c55f"
NETWORKS = ("local", "test", "finney")
SERIALIZATION_RULES = {
    "json_encoding": "utf-8",
    "json_key_order": "lexicographic",
    "json_separators": [",", ":"],
    "hash_algorithm": "sha256",
    "line_endings": "lf",
}
CONSENSUS_MATERIAL_KEYS = {
    "grader",
    "benchmarks",
    "model_registry",
    "prices",
    "backbone",
    "dependencies",
    "worker",
    "committee",
    "rounding",
}
RUNTIME_DISTRIBUTIONS = {
    "absl-py": "absl-py",
    "bittensor": "bittensor",
    "datasets": "datasets",
    "immutabledict": "immutabledict",
    "langdetect": "langdetect",
    "math-verify": "math-verify",
    "nltk": "nltk",
    "numpy": "numpy",
    "torch_linux": "torch",
    "transformers": "transformers",
}


class ManifestError(RuntimeError):
    """The packaged or overridden consensus manifest is unsafe or malformed."""


class ProtocolUnavailable(ManifestError):
    """No complete enabled protocol is active for a requested block."""


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: str
    protocol_version: int
    package_version: str
    status: str
    enabled: bool
    complete: bool
    activation_blocks: dict[str, int | None]
    consensus: dict[str, Any] | None


@dataclass(frozen=True)
class ConsensusManifest:
    schema_version: int
    manifest_id: str
    serialization: dict[str, Any]
    protocols: tuple[ProtocolSpec, ...]
    raw: dict[str, Any]
    sha256: str


def verify_runtime_dependencies(consensus: dict[str, Any]) -> None:
    """Fail closed when installed consensus libraries differ from the manifest."""
    dependencies = consensus.get("dependencies")
    expected_keys = {"uv_lock_sha256", "python", *RUNTIME_DISTRIBUTIONS}
    if not isinstance(dependencies, dict) or set(dependencies) != expected_keys:
        raise ManifestError("consensus dependency manifest is incomplete")
    if dependencies["python"] != ">=3.10,<3.13" or not (
        (3, 10) <= sys.version_info[:2] < (3, 13)
    ):
        raise ManifestError("runtime Python differs from the consensus manifest")
    for manifest_name, distribution_name in RUNTIME_DISTRIBUTIONS.items():
        expected = dependencies[manifest_name]
        if not isinstance(expected, str) or not expected:
            raise ManifestError(f"consensus dependency {manifest_name} is invalid")
        try:
            actual = version(distribution_name)
        except PackageNotFoundError as exc:
            raise ManifestError(
                f"consensus dependency {distribution_name} is not installed"
            ) from exc
        if actual != expected:
            raise ManifestError(
                f"consensus dependency {distribution_name} differs: "
                f"expected {expected}, found {actual}"
            )


def canonical_json(value: object) -> bytes:
    """Serialize manifest material according to the packaged v1 rules."""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_network(network: str) -> str:
    """Map SDK/local endpoint names to the three manifest profiles."""
    normalized = network.strip().lower()
    if normalized in {"local", "mock"}:
        return "local"
    if normalized.startswith(("ws://127.0.0.1", "ws://localhost")):
        return "local"
    if normalized in {"test", "testnet"}:
        return "test"
    if normalized in {"finney", "mainnet"}:
        return "finney"
    raise ManifestError(f"Unsupported consensus network profile: {network!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"Duplicate JSON key in consensus manifest: {key}")
        result[key] = value
    return result


def _read_json(data: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ManifestError(f"Consensus manifest is not valid UTF-8 JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ManifestError("Consensus manifest root must be an object")
    return parsed


def _expect_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ManifestError(f"{label} keys differ (missing={missing}, extra={extra})")


def _validate_protocol(raw: object) -> ProtocolSpec:
    if not isinstance(raw, dict):
        raise ManifestError("Each protocol entry must be an object")
    _expect_exact_keys(
        raw,
        {
            "id", "protocol_version", "package_version", "status", "enabled",
            "complete", "activation_blocks", "consensus",
        },
        "protocol",
    )

    protocol_id = raw["id"]
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ManifestError("Protocol id must be a non-empty string")
    if not isinstance(raw["protocol_version"], int) or raw["protocol_version"] <= 0:
        raise ManifestError(f"{protocol_id}: protocol_version must be positive")
    if not isinstance(raw["package_version"], str) or not raw["package_version"]:
        raise ManifestError(f"{protocol_id}: package_version must be a string")
    if not isinstance(raw["status"], str) or not raw["status"]:
        raise ManifestError(f"{protocol_id}: status must be a string")
    if type(raw["enabled"]) is not bool or type(raw["complete"]) is not bool:
        raise ManifestError(f"{protocol_id}: enabled/complete must be booleans")

    activation = raw["activation_blocks"]
    if not isinstance(activation, dict) or set(activation) != set(NETWORKS):
        raise ManifestError(f"{protocol_id}: activation_blocks must define {NETWORKS}")
    for network, block in activation.items():
        if block is not None and (not isinstance(block, int) or isinstance(block, bool) or block < 0):
            raise ManifestError(f"{protocol_id}: invalid {network} activation block")

    consensus = raw["consensus"]
    active_anywhere = raw["enabled"] or any(block is not None for block in activation.values())
    if active_anywhere and (not raw["complete"] or not isinstance(consensus, dict)):
        raise ManifestError(
            f"{protocol_id}: enabled/activated protocols must be complete with consensus material"
        )
    if not raw["enabled"] and any(block is not None for block in activation.values()):
        raise ManifestError(f"{protocol_id}: disabled protocol cannot have an activation block")
    if isinstance(consensus, dict) and set(consensus) != CONSENSUS_MATERIAL_KEYS:
        raise ManifestError(
            f"{protocol_id}: consensus material keys must be {sorted(CONSENSUS_MATERIAL_KEYS)}"
        )

    if protocol_id == "v1":
        try:
            declared = consensus["grader"]["sha256"]
        except (KeyError, TypeError) as e:
            raise ManifestError("v1 grader hash is missing") from e
        if declared != V1_GRADER_SHA256:
            raise ManifestError("v1 grader hash does not match the immutable historical value")
    elif protocol_id == "v2":
        try:
            declared = consensus["grader"]["sha256"]
        except (KeyError, TypeError) as e:
            raise ManifestError("v2 grader bundle hash is missing") from e
        if not isinstance(declared, str) or len(declared) != 64:
            raise ManifestError("v2 grader bundle hash is invalid")
        if raw["complete"]:
            committee = consensus["committee"]
            schedule_fields = (
                "epoch_blocks",
                "precommit_deadline_offset_blocks",
                "report_deadline_offset_blocks",
                "slice_size",
                "api_concurrency",
            )
            if any(
                not isinstance(committee.get(field), int)
                or isinstance(committee.get(field), bool)
                or committee[field] <= 0
                for field in schedule_fields
            ):
                raise ManifestError("complete v2 schedule fields must be positive integers")
            if not (
                committee["precommit_deadline_offset_blocks"]
                < committee["report_deadline_offset_blocks"]
                < committee["epoch_blocks"]
            ):
                raise ManifestError("complete v2 epoch deadlines are invalid")
            if (
                committee.get("maximum_builders") != 5
                or committee.get("minimum_reports") != 3
            ):
                raise ManifestError("complete v2 requires a five-builder/three-report quorum")
            if consensus["benchmarks"].get("rollout_blockers"):
                raise ManifestError("complete v2 cannot retain benchmark rollout blockers")
            if not consensus["model_registry"].get("active_models"):
                raise ManifestError("complete v2 requires approved active models")
            worker_digest = consensus["worker"].get("image_digest")
            if (
                not isinstance(worker_digest, str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", worker_digest)
            ):
                raise ManifestError("complete v2 requires a worker registry digest")

    return ProtocolSpec(
        protocol_id=protocol_id,
        protocol_version=raw["protocol_version"],
        package_version=raw["package_version"],
        status=raw["status"],
        enabled=raw["enabled"],
        complete=raw["complete"],
        activation_blocks=dict(activation),
        consensus=consensus,
    )


def _validate_manifest(raw: dict[str, Any]) -> ConsensusManifest:
    _expect_exact_keys(
        raw,
        {"schema_version", "manifest_id", "serialization", "protocols"},
        "manifest",
    )
    if raw["schema_version"] != 1:
        raise ManifestError("Unsupported consensus manifest schema_version")
    if raw["manifest_id"] != "fugal-consensus":
        raise ManifestError("Unexpected consensus manifest_id")
    if raw["serialization"] != SERIALIZATION_RULES:
        raise ManifestError("Unsupported or incomplete manifest serialization rules")
    if not isinstance(raw["protocols"], list) or not raw["protocols"]:
        raise ManifestError("protocols must be a non-empty list")

    protocols = tuple(_validate_protocol(item) for item in raw["protocols"])
    ids = [protocol.protocol_id for protocol in protocols]
    versions = [protocol.protocol_version for protocol in protocols]
    if len(ids) != len(set(ids)) or len(versions) != len(set(versions)):
        raise ManifestError("Protocol ids and numeric versions must be unique")
    if "v1" not in ids or "v2" not in ids:
        raise ManifestError("Manifest must preserve v1 and declare inactive v2")

    return ConsensusManifest(
        schema_version=raw["schema_version"],
        manifest_id=raw["manifest_id"],
        serialization=dict(raw["serialization"]),
        protocols=protocols,
        raw=raw,
        sha256=hashlib.sha256(canonical_json(raw)).hexdigest(),
    )


def load_consensus_manifest(
    network: str,
    override_path: str | os.PathLike[str] | None = None,
) -> ConsensusManifest:
    """Load the package manifest; overrides are restricted to local/mock use."""
    profile = canonical_network(network)
    configured_override = override_path or os.getenv("FUGAL_CONSENSUS_MANIFEST")
    if configured_override is not None:
        if profile != "local":
            raise ManifestError("Consensus manifest overrides are local/mock-only")
        data = Path(configured_override).read_bytes()
    else:
        resource = importlib.resources.files("fugal_subnet").joinpath(MANIFEST_RESOURCE)
        data = resource.read_bytes()
    return _validate_manifest(_read_json(data))


def select_protocol(
    network: str,
    block: int,
    manifest: ConsensusManifest | None = None,
) -> ProtocolSpec:
    """Select the highest-version complete protocol active at ``block``."""
    if not isinstance(block, int) or isinstance(block, bool) or block < 0:
        raise ManifestError("block must be a non-negative integer")
    profile = canonical_network(network)
    manifest = manifest or load_consensus_manifest(profile)
    eligible = []
    for protocol in manifest.protocols:
        activation_block = protocol.activation_blocks[profile]
        if (
            protocol.enabled
            and protocol.complete
            and activation_block is not None
            and activation_block <= block
        ):
            eligible.append(protocol)
    if not eligible:
        raise ProtocolUnavailable(
            f"No complete protocol is active on {profile} at block {block}"
        )
    return max(eligible, key=lambda protocol: protocol.protocol_version)
