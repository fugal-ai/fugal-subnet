"""Versioned grader dispatch without mutating the immutable v1 contract."""

from __future__ import annotations

from types import ModuleType

from fugal_subnet import graders as graders_v1
from fugal_subnet.consensus_manifest import (
    V1_GRADER_SHA256,
    ProtocolUnavailable,
    load_consensus_manifest,
    select_protocol,
)


def get_grader(protocol_id: str) -> ModuleType:
    """Return the exact grader module for a historical protocol id."""
    if protocol_id == "v1":
        actual = graders_v1.grader_hash().removeprefix("sha256:")
        if actual != V1_GRADER_SHA256:
            raise RuntimeError("Immutable v1 grader bytes have changed")
        return graders_v1
    if protocol_id == "v2":
        from fugal_subnet import graders_v2

        manifest = load_consensus_manifest("local")
        spec = next(item for item in manifest.protocols if item.protocol_id == "v2")
        if spec.consensus is None:
            raise RuntimeError("Packaged v2 consensus material is missing")
        declared = spec.consensus["grader"]["sha256"]
        actual = graders_v2.grader_hash().removeprefix("sha256:")
        if actual != declared:
            raise RuntimeError("Packaged v2 grader bundle differs from the manifest")
        return graders_v2
    raise ProtocolUnavailable(f"Grader for {protocol_id!r} is not packaged and active")


def grade_for_protocol(
    protocol_id: str,
    task: dict,
    reply: str,
    allow_exec: bool = False,
) -> int:
    if protocol_id == "v1":
        return int(get_grader(protocol_id).grade(task, reply, allow_exec))
    if allow_exec:
        raise ValueError("v2 execution requires a GradingClient, not allow_exec")
    return int(get_grader(protocol_id).grade(task, reply))


def grade_for_block(
    network: str,
    block: int,
    task: dict,
    reply: str,
    allow_exec: bool = False,
) -> int:
    protocol = select_protocol(network, block)
    return grade_for_protocol(protocol.protocol_id, task, reply, allow_exec)
