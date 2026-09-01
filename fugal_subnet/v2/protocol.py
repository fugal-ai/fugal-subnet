"""Bounded v2 validator-to-validator matrix report transport."""

# Do not add ``from __future__ import annotations``. Bittensor 10.x inspects
# forward-function annotations at Axon.attach() time.
import base64
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Sequence

import bittensor as bt
import pydantic

EPOCH_ID_MAX_LEN = 64
HASH_HEX_LEN = 64
HOTKEY_MAX_LEN = 128
SIGNATURE_MAX_LEN = 256
ERROR_MAX_LEN = 256
REPORT_CHUNK_BYTES = 256 * 1024
REPORT_CHUNK_B64_MAX_LEN = 350_000
REPORT_MAX_CHUNKS = 64
REPORT_MAX_BYTES = REPORT_CHUNK_BYTES * REPORT_MAX_CHUNKS
EPOCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class MatrixReportSynapse(bt.Synapse):
    """Request/response for one bounded chunk of a signed builder report."""

    epoch_id: pydantic.constr(max_length=EPOCH_ID_MAX_LEN) = ""  # type: ignore[valid-type]
    manifest_hash: pydantic.constr(max_length=HASH_HEX_LEN) = ""  # type: ignore[valid-type]
    artifact_hash: pydantic.constr(max_length=HASH_HEX_LEN) = ""  # type: ignore[valid-type]
    chunk_index: int = pydantic.Field(default=0, ge=0, lt=REPORT_MAX_CHUNKS)
    chunk_count: int = pydantic.Field(default=1, ge=1, le=REPORT_MAX_CHUNKS)
    payload_b64: pydantic.constr(max_length=REPORT_CHUNK_B64_MAX_LEN) = ""  # type: ignore[valid-type]
    builder_hotkey: pydantic.constr(max_length=HOTKEY_MAX_LEN) = ""  # type: ignore[valid-type]
    signature: pydantic.constr(max_length=SIGNATURE_MAX_LEN) = ""  # type: ignore[valid-type]
    error: pydantic.constr(max_length=ERROR_MAX_LEN) = ""  # type: ignore[valid-type]

    def deserialize(self) -> "MatrixReportSynapse":
        return self


@dataclass(frozen=True)
class ReportChunk:
    epoch_id: str
    manifest_hash: str
    artifact_hash: str
    chunk_index: int
    chunk_count: int
    payload_b64: str
    builder_hotkey: str
    signature: str


def signature_message(
    epoch_id: str,
    manifest_hash: str,
    artifact_hash: str,
    builder_hotkey: str,
) -> bytes:
    """Canonical domain-separated bytes signed once for the whole artifact."""
    value = {
        "artifact_hash": artifact_hash,
        "builder_hotkey": builder_hotkey,
        "epoch_id": epoch_id,
        "manifest_hash": manifest_hash,
        "purpose": "fugal-matrix-report-v2",
    }
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def chunk_report(
    payload: bytes,
    *,
    epoch_id: str,
    manifest_hash: str,
    builder_hotkey: str,
    signature: str,
) -> list[ReportChunk]:
    """Split one immutable report into independently bounded wire chunks."""
    if not isinstance(payload, bytes):
        raise ValueError("report payload must be bytes")
    if not payload:
        raise ValueError("report payload cannot be empty")
    if len(payload) > REPORT_MAX_BYTES:
        raise ValueError(f"report exceeds {REPORT_MAX_BYTES} bytes")
    if not isinstance(epoch_id, str) or not EPOCH_ID_RE.fullmatch(epoch_id):
        raise ValueError("invalid epoch_id")
    for label, value in (("manifest_hash", manifest_hash),):
        if len(value) != HASH_HEX_LEN or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{label} must be lowercase SHA256 hex")
    if not builder_hotkey or len(builder_hotkey) > HOTKEY_MAX_LEN:
        raise ValueError("invalid builder_hotkey")
    if not signature or len(signature) > SIGNATURE_MAX_LEN:
        raise ValueError("invalid signature")

    artifact_hash = hashlib.sha256(payload).hexdigest()
    raw_chunks = [
        payload[offset:offset + REPORT_CHUNK_BYTES]
        for offset in range(0, len(payload), REPORT_CHUNK_BYTES)
    ]
    count = len(raw_chunks)
    return [
        ReportChunk(
            epoch_id=epoch_id,
            manifest_hash=manifest_hash,
            artifact_hash=artifact_hash,
            chunk_index=index,
            chunk_count=count,
            payload_b64=base64.b64encode(raw).decode("ascii"),
            builder_hotkey=builder_hotkey,
            signature=signature,
        )
        for index, raw in enumerate(raw_chunks)
    ]


def assemble_report(chunks: Sequence[ReportChunk | MatrixReportSynapse]) -> bytes:
    """Validate metadata, bounds, completeness, base64, and artifact SHA256."""
    if not chunks:
        raise ValueError("no report chunks supplied")
    if len(chunks) > REPORT_MAX_CHUNKS:
        raise ValueError("too many report chunks")

    first = chunks[0]
    expected_metadata = (
        first.epoch_id,
        first.manifest_hash,
        first.artifact_hash,
        first.chunk_count,
        first.builder_hotkey,
        first.signature,
    )
    if first.chunk_count != len(chunks):
        raise ValueError("report chunk_count does not match supplied chunks")
    indexed: dict[int, bytes] = {}
    for chunk in chunks:
        metadata = (
            chunk.epoch_id,
            chunk.manifest_hash,
            chunk.artifact_hash,
            chunk.chunk_count,
            chunk.builder_hotkey,
            chunk.signature,
        )
        if metadata != expected_metadata:
            raise ValueError("report chunk metadata mismatch")
        if not 0 <= chunk.chunk_index < chunk.chunk_count <= REPORT_MAX_CHUNKS:
            raise ValueError("invalid report chunk index/count")
        if chunk.chunk_index in indexed:
            raise ValueError("duplicate report chunk index")
        try:
            raw = base64.b64decode(chunk.payload_b64, validate=True)
        except Exception as e:
            raise ValueError("invalid report chunk base64") from e
        if len(raw) > REPORT_CHUNK_BYTES:
            raise ValueError("decoded report chunk exceeds bound")
        indexed[chunk.chunk_index] = raw

    if set(indexed) != set(range(first.chunk_count)):
        raise ValueError("report chunks are incomplete")
    payload = b"".join(indexed[index] for index in range(first.chunk_count))
    if not payload or len(payload) > REPORT_MAX_BYTES:
        raise ValueError("assembled report size is invalid")
    if hashlib.sha256(payload).hexdigest() != first.artifact_hash:
        raise ValueError("assembled report hash mismatch")
    return payload
