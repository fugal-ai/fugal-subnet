"""Bounded chunk transport vectors for inactive v2 matrix reports."""

from __future__ import annotations

import base64
from dataclasses import replace

import pytest
from pydantic import ValidationError

from fugal_subnet.v2.protocol import (
    REPORT_CHUNK_B64_MAX_LEN,
    REPORT_CHUNK_BYTES,
    MatrixReportSynapse,
    assemble_report,
    chunk_report,
    signature_message,
)

MANIFEST_HASH = "a" * 64


def chunks(payload: bytes):
    return chunk_report(
        payload,
        epoch_id="e00000001",
        manifest_hash=MANIFEST_HASH,
        builder_hotkey="builder-hotkey",
        signature="signature",
    )


def test_report_roundtrip_is_bounded_order_independent_and_hash_verified():
    payload = b"a" * REPORT_CHUNK_BYTES + b"second chunk"
    report_chunks = chunks(payload)

    assert len(report_chunks) == 2
    assert all(len(chunk.payload_b64) <= REPORT_CHUNK_B64_MAX_LEN for chunk in report_chunks)
    assert assemble_report(list(reversed(report_chunks))) == payload

    synapses = [MatrixReportSynapse(**chunk.__dict__) for chunk in report_chunks]
    assert assemble_report(synapses) == payload
    assert synapses[0].deserialize() is synapses[0]


def test_signature_message_is_canonical_and_domain_separated():
    message = signature_message(
        "e00000001", MANIFEST_HASH, "b" * 64, "builder-hotkey"
    )
    assert message == (
        b'{"artifact_hash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        b'"builder_hotkey":"builder-hotkey","epoch_id":"e00000001",'
        b'"manifest_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"purpose":"fugal-matrix-report-v2"}'
    )


def test_missing_duplicate_and_mismatched_chunks_fail_closed():
    report_chunks = chunks(b"a" * (REPORT_CHUNK_BYTES + 1))

    with pytest.raises(ValueError, match="chunk_count"):
        assemble_report(report_chunks[:1])
    with pytest.raises(ValueError, match="duplicate"):
        assemble_report([report_chunks[0], report_chunks[0]])
    with pytest.raises(ValueError, match="metadata mismatch"):
        assemble_report([
            report_chunks[0],
            replace(report_chunks[1], builder_hotkey="different-builder"),
        ])


def test_payload_and_hash_tampering_fail_closed():
    report_chunk = chunks(b"original")[0]
    tampered_payload = replace(
        report_chunk,
        payload_b64=base64.b64encode(b"tampered").decode("ascii"),
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        assemble_report([tampered_payload])

    with pytest.raises(ValueError, match="base64"):
        assemble_report([replace(report_chunk, payload_b64="!!!")])


def test_wire_bounds_are_enforced_by_pydantic():
    with pytest.raises(ValidationError):
        MatrixReportSynapse(chunk_count=65)
    with pytest.raises(ValidationError):
        MatrixReportSynapse(chunk_index=-1)
    with pytest.raises(ValidationError):
        MatrixReportSynapse(payload_b64="x" * (REPORT_CHUNK_B64_MAX_LEN + 1))


def test_chunker_rejects_empty_and_oversized_metadata():
    with pytest.raises(ValueError, match="cannot be empty"):
        chunks(b"")
    with pytest.raises(ValueError, match="manifest_hash"):
        chunk_report(
            b"payload",
            epoch_id="e1",
            manifest_hash="not-a-hash",
            builder_hotkey="hotkey",
            signature="signature",
        )
