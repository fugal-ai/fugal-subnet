"""Signed builder reports and strict-majority matrix derivation for v2."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Mapping, Sequence

import bittensor as bt

from fugal_subnet.consensus_manifest import canonical_json
from fugal_subnet.v2.committee import MIN_REPORTS, Builder, QuorumUnavailable
from fugal_subnet.v2.matrix import MatrixResult
from fugal_subnet.v2.protocol import ReportChunk, chunk_report, signature_message

REPORT_SCHEMA_VERSION = 1
MAX_QUESTIONS = 512
MAX_MODELS = 8
MAX_CELLS = MAX_QUESTIONS * MAX_MODELS
MAX_RESPONSE_BYTES = 4096
MAX_REPORT_BYTES = 16 * 1024 * 1024
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")


class ReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConsensusMatrix:
    epoch_id: str
    builder_hotkeys: tuple[str, ...]
    question_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    matrix: tuple[tuple[int, ...], ...]
    report_hashes: tuple[str, ...]


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise ReportError(f"{label} must be lowercase SHA-256 hex")
    return value


def _decimal_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ReportError(f"{label} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ReportError(f"{label} is invalid") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ReportError(f"{label} is invalid")
    if str(parsed) != value:
        raise ReportError(f"{label} is not canonically formatted")
    return value


def report_signature_message(core: dict) -> bytes:
    return canonical_json({"purpose": "fugal-builder-report-v2", "core": core})


def _validate_cell(cell: object, question_id: str, model_id: str) -> dict:
    keys = {
        "question_id", "model_id", "response", "response_sha256",
        "prompt_tokens", "completion_tokens", "grade", "actual_cost_usd",
        "reserved_cost_usd",
    }
    if not isinstance(cell, dict) or set(cell) != keys:
        raise ReportError("report cell schema differs")
    if cell["question_id"] != question_id or cell["model_id"] != model_id:
        raise ReportError("report cell order or identity differs")
    response = cell["response"]
    if not isinstance(response, str) or len(response.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ReportError("report response is oversized or invalid")
    if hashlib.sha256(response.encode("utf-8")).hexdigest() != cell["response_sha256"]:
        raise ReportError("report response hash mismatch")
    for label in ("prompt_tokens", "completion_tokens"):
        value = cell[label]
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 10_000_000:
            raise ReportError(f"report {label} is invalid")
    if type(cell["grade"]) is not int or cell["grade"] not in (0, 1):
        raise ReportError("report grade must be zero or one")
    _decimal_string(cell["actual_cost_usd"], "actual_cost_usd")
    _decimal_string(cell["reserved_cost_usd"], "reserved_cost_usd")
    if Decimal(cell["actual_cost_usd"]) > Decimal(cell["reserved_cost_usd"]):
        raise ReportError("actual cell cost exceeds reservation")
    return cell


def validate_core(core: object) -> dict:
    keys = {
        "epoch_id", "boundary_block", "boundary_hash", "manifest_hash",
        "question_commitment", "grader_hash", "registry_hash", "builder_hotkey",
        "question_ids", "model_ids", "cells", "spend",
    }
    if not isinstance(core, dict) or set(core) != keys:
        raise ReportError("builder report core schema differs")
    if (
        not isinstance(core["epoch_id"], str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", core["epoch_id"])
    ):
        raise ReportError("report epoch_id is invalid")
    if not isinstance(core["boundary_block"], int) or isinstance(core["boundary_block"], bool) or core["boundary_block"] < 0:
        raise ReportError("report boundary block is invalid")
    for label in (
        "boundary_hash", "manifest_hash", "question_commitment", "grader_hash", "registry_hash"
    ):
        _hash(core[label], label)
    try:
        bt.Keypair(ss58_address=core["builder_hotkey"])
    except Exception as exc:
        raise ReportError("builder hotkey is invalid") from exc
    questions = core["question_ids"]
    models = core["model_ids"]
    if not isinstance(questions, list) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ReportError("report question_ids count is invalid")
    if not isinstance(models, list) or not 1 <= len(models) <= MAX_MODELS:
        raise ReportError("report model_ids count is invalid")
    if any(not isinstance(item, str) or not item for item in questions + models):
        raise ReportError("report question/model IDs are invalid")
    if len(set(questions)) != len(questions) or len(set(models)) != len(models):
        raise ReportError("report question/model IDs are duplicated")
    cells = core["cells"]
    if not isinstance(cells, list) or len(cells) != len(questions) * len(models):
        raise ReportError("report cell count differs")
    index = 0
    for question_id in questions:
        for model_id in models:
            _validate_cell(cells[index], question_id, model_id)
            index += 1
    spend = core["spend"]
    if not isinstance(spend, dict) or set(spend) != {
        "currency", "budget_usd", "reserved_usd", "actual_usd"
    } or spend["currency"] != "USD":
        raise ReportError("report spend schema differs")
    for label in ("budget_usd", "reserved_usd", "actual_usd"):
        _decimal_string(spend[label], label)
    if Decimal(spend["actual_usd"]) > Decimal(spend["reserved_usd"]):
        raise ReportError("report actual spend exceeds reserved spend")
    if Decimal(spend["reserved_usd"]) > Decimal(spend["budget_usd"]):
        raise ReportError("report reserved spend exceeds budget")
    return core


def sign_report(core: dict, keypair: bt.Keypair) -> bytes:
    validate_core(core)
    if core["builder_hotkey"] != keypair.ss58_address:
        raise ReportError("report builder does not match signing key")
    signature = bytes(keypair.sign(report_signature_message(core))).hex()
    payload = canonical_json({
        "schema_version": REPORT_SCHEMA_VERSION,
        "core": core,
        "signature": signature,
    })
    if len(payload) > MAX_REPORT_BYTES:
        raise ReportError("signed report exceeds artifact bound")
    return payload


def build_signed_report(
    matrix_result: MatrixResult,
    *,
    epoch_id: str,
    boundary_block: int,
    boundary_hash: str,
    manifest_hash: str,
    question_commitment: str,
    grader_hash: str,
    registry_hash: str,
    keypair: bt.Keypair,
) -> bytes:
    """Convert persisted matrix cells into the canonical signed report."""
    cells = [{
        "question_id": cell.question_id,
        "model_id": cell.model_id,
        "response": cell.response,
        "response_sha256": hashlib.sha256(cell.response.encode("utf-8")).hexdigest(),
        "prompt_tokens": cell.prompt_tokens,
        "completion_tokens": cell.completion_tokens,
        "grade": cell.grade,
        "actual_cost_usd": cell.actual_cost_usd,
        "reserved_cost_usd": cell.reserved_cost_usd,
    } for cell in matrix_result.cells]
    core = {
        "epoch_id": epoch_id,
        "boundary_block": boundary_block,
        "boundary_hash": boundary_hash.removeprefix("0x"),
        "manifest_hash": manifest_hash,
        "question_commitment": question_commitment,
        "grader_hash": grader_hash,
        "registry_hash": registry_hash,
        "builder_hotkey": keypair.ss58_address,
        "question_ids": list(matrix_result.question_ids),
        "model_ids": list(matrix_result.model_ids),
        "cells": cells,
        "spend": {
            "currency": "USD",
            "budget_usd": matrix_result.budget_usd,
            "reserved_usd": matrix_result.reserved_usd,
            "actual_usd": matrix_result.actual_usd,
        },
    }
    return sign_report(core, keypair)


def verify_report(payload: bytes, *, expected_hotkey: str | None = None) -> dict:
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_REPORT_BYTES:
        raise ReportError("report artifact size is invalid")
    try:
        report = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError("report artifact is not UTF-8 JSON") from exc
    if canonical_json(report) != payload:
        raise ReportError("report artifact is not canonical JSON")
    if not isinstance(report, dict) or set(report) != {"schema_version", "core", "signature"}:
        raise ReportError("signed report schema differs")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ReportError("signed report version differs")
    core = validate_core(report["core"])
    signature = report["signature"]
    if not isinstance(signature, str) or not SIGNATURE_RE.fullmatch(signature):
        raise ReportError("builder signature encoding is invalid")
    if expected_hotkey is not None and core["builder_hotkey"] != expected_hotkey:
        raise ReportError("report came from an unexpected builder")
    keypair = bt.Keypair(ss58_address=core["builder_hotkey"])
    if not keypair.verify(report_signature_message(core), bytes.fromhex(signature)):
        raise ReportError("builder report signature is invalid")
    return report


def chunk_signed_report(payload: bytes, keypair: bt.Keypair) -> list[ReportChunk]:
    report = verify_report(payload, expected_hotkey=keypair.ss58_address)
    core = report["core"]
    artifact_hash = hashlib.sha256(payload).hexdigest()
    chunk_signature = bytes(keypair.sign(signature_message(
        core["epoch_id"], core["manifest_hash"], artifact_hash, keypair.ss58_address
    ))).hex()
    return chunk_report(
        payload,
        epoch_id=core["epoch_id"],
        manifest_hash=core["manifest_hash"],
        builder_hotkey=keypair.ss58_address,
        signature=chunk_signature,
    )


def verify_chunk_signature(chunk: ReportChunk) -> bool:
    if not SIGNATURE_RE.fullmatch(chunk.signature):
        return False
    try:
        keypair = bt.Keypair(ss58_address=chunk.builder_hotkey)
        return bool(keypair.verify(
            signature_message(
                chunk.epoch_id,
                chunk.manifest_hash,
                chunk.artifact_hash,
                chunk.builder_hotkey,
            ),
            bytes.fromhex(chunk.signature),
        ))
    except Exception:
        return False


def derive_consensus_matrix(
    *,
    committee: Sequence[Builder],
    committed_hashes: Mapping[str, str],
    artifacts: Mapping[str, bytes],
    minimum_reports: int = MIN_REPORTS,
) -> ConsensusMatrix:
    committee_hotkeys = {builder.hotkey for builder in committee}
    if not set(committed_hashes) <= committee_hotkeys:
        raise ReportError("a non-committee report commitment was supplied")
    if len(committed_hashes) < minimum_reports:
        raise QuorumUnavailable("fewer than the required committed reports")
    if set(artifacts) != set(committed_hashes):
        raise QuorumUnavailable("a committed builder report is unavailable")
    verified = []
    report_hashes = []
    for hotkey in sorted(committed_hashes):
        expected_hash = _hash(committed_hashes[hotkey], "committed report hash")
        payload = artifacts[hotkey]
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != expected_hash:
            raise ReportError("committed report artifact hash mismatch")
        report = verify_report(payload, expected_hotkey=hotkey)
        verified.append(report)
        report_hashes.append(actual_hash)
    first = verified[0]["core"]
    shared_keys = (
        "epoch_id", "boundary_block", "boundary_hash", "manifest_hash",
        "question_commitment", "grader_hash", "registry_hash", "question_ids", "model_ids",
    )
    expected = tuple(first[key] for key in shared_keys)
    for report in verified[1:]:
        if tuple(report["core"][key] for key in shared_keys) != expected:
            raise ReportError("builder reports disagree on canonical epoch material")
    rows = []
    model_count = len(first["model_ids"])
    for question_index in range(len(first["question_ids"])):
        row = []
        for model_index in range(model_count):
            cell_index = question_index * model_count + model_index
            votes = sum(report["core"]["cells"][cell_index]["grade"] for report in verified)
            row.append(int(votes * 2 > len(verified)))
        rows.append(tuple(row))
    return ConsensusMatrix(
        epoch_id=first["epoch_id"],
        builder_hotkeys=tuple(sorted(committed_hashes)),
        question_ids=tuple(first["question_ids"]),
        model_ids=tuple(first["model_ids"]),
        matrix=tuple(rows),
        report_hashes=tuple(report_hashes),
    )
