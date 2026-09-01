"""Atomic, versioned v2 epoch journal for crash-safe paid-cell recovery."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1
MAX_RESPONSE_BYTES = 65_536
EPOCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
PHASES = (
    "boundary",
    "precommit",
    "miner_query",
    "matrix_build",
    "report_commit",
    "report_exchange",
    "matrix_consensus",
    "head_evaluation",
    "reveal",
    "set_weights",
    "complete",
    "aborted",
)
CELL_STATUSES = {"reserved", "complete", "forfeited"}


class JournalError(RuntimeError):
    """Journal data is malformed, conflicting, or unsafe to resume."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool):
        raise JournalError(f"{label} cannot be boolean")
    try:
        result = Decimal(str(value))
    except Exception as e:
        raise JournalError(f"{label} is not decimal") from e
    if not result.is_finite() or result < 0:
        raise JournalError(f"{label} must be finite and non-negative")
    return result


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def cell_key(question_id: str, model_id: str) -> str:
    if not isinstance(question_id, str) or not question_id:
        raise JournalError("question_id must be a non-empty string")
    if not isinstance(model_id, str) or not model_id:
        raise JournalError("model_id must be a non-empty string")
    return hashlib.sha256(_canonical_json([question_id, model_id])).hexdigest()


class EpochJournal:
    """Process/thread-safe JSON journal stored under one validated epoch id."""

    def __init__(self, root: str | os.PathLike[str], epoch_id: str):
        if not EPOCH_ID_RE.fullmatch(epoch_id) or epoch_id in {".", ".."}:
            raise JournalError("invalid epoch_id")
        self.epoch_id = epoch_id
        self.epoch_dir = Path(root) / epoch_id
        self.path = self.epoch_dir / "journal.json"
        self.lock_path = self.epoch_dir / ".journal.lock"
        self._thread_lock = threading.Lock()

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.epoch_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._thread_lock:
            with self.lock_path.open("a+b") as lock_file:
                os.chmod(self.lock_path, 0o600)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def initialize(
        self,
        *,
        manifest_hash: str,
        boundary_block: int,
        boundary_hash: str,
        budget_usd: int | float | str | Decimal,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{64}", manifest_hash):
            raise JournalError("manifest_hash must be lowercase SHA256 hex")
        if not isinstance(boundary_block, int) or isinstance(boundary_block, bool) or boundary_block < 0:
            raise JournalError("boundary_block must be a non-negative integer")
        if not isinstance(boundary_hash, str) or not boundary_hash or len(boundary_hash) > 128:
            raise JournalError("boundary_hash must be a bounded non-empty string")
        budget = _decimal(budget_usd, "budget_usd")

        with self._locked():
            if self.path.exists():
                existing = self._read_unlocked()
                expected = (
                    manifest_hash,
                    boundary_block,
                    boundary_hash,
                    budget,
                )
                actual = (
                    existing["manifest_hash"],
                    existing["boundary"]["block"],
                    existing["boundary"]["hash"],
                    _decimal(existing["spend"]["budget_usd"], "budget_usd"),
                )
                if actual != expected:
                    raise JournalError("existing journal does not match epoch boundary/manifest")
                return existing

            journal = {
                "schema_version": SCHEMA_VERSION,
                "sequence": 0,
                "epoch_id": self.epoch_id,
                "manifest_hash": manifest_hash,
                "boundary": {"block": boundary_block, "hash": boundary_hash},
                "status": "active",
                "phase": "boundary",
                "abort_reason": None,
                "commitments": [],
                "cells": {},
                "spend": {
                    "budget_usd": _decimal_text(budget),
                    "reserved_usd": "0",
                    "actual_usd": "0",
                },
                "report": {
                    "artifact_hash": None,
                    "committed_block": None,
                    "chunk_count": None,
                    "received_hotkeys": [],
                },
            }
            self._validate(journal)
            self._write_unlocked(journal)
            return journal

    def read(self) -> dict[str, Any]:
        with self._locked():
            return self._read_unlocked()

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            raise JournalError("epoch journal does not exist") from e
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise JournalError("epoch journal is not valid UTF-8 JSON") from e
        self._validate(raw)
        return raw

    def _write_unlocked(self, journal: dict[str, Any]) -> None:
        self._validate(journal)
        fd, temporary = tempfile.mkstemp(prefix=".journal.", dir=self.epoch_dir)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as output:
                output.write(_canonical_json(journal) + b"\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.epoch_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _mutate(self, change) -> tuple[dict[str, Any], Any]:
        with self._locked():
            journal = self._read_unlocked()
            result = change(journal)
            journal["sequence"] += 1
            self._recompute_spend(journal)
            self._write_unlocked(journal)
            return journal, result

    @staticmethod
    def _require_active(journal: dict[str, Any]) -> None:
        if journal["status"] != "active":
            raise JournalError(f"journal is terminal: {journal['status']}")

    def advance_phase(self, phase: str) -> dict[str, Any]:
        if phase not in PHASES:
            raise JournalError(f"unknown phase: {phase}")

        def change(journal):
            self._require_active(journal)
            current_index = PHASES.index(journal["phase"])
            requested_index = PHASES.index(phase)
            if requested_index < current_index:
                raise JournalError("journal phase cannot move backwards")
            if phase == "aborted":
                raise JournalError("use abort() to record a reason")
            journal["phase"] = phase
            if phase == "complete":
                if any(cell["status"] != "complete" for cell in journal["cells"].values()):
                    raise JournalError("cannot complete with reserved or forfeited cells")
                journal["status"] = "complete"

        return self._mutate(change)[0]

    def abort(self, reason: str) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 512:
            raise JournalError("abort reason must be a bounded non-empty string")

        def change(journal):
            self._require_active(journal)
            journal["status"] = "aborted"
            journal["phase"] = "aborted"
            journal["abort_reason"] = reason

        return self._mutate(change)[0]

    def record_finalized_commitment(
        self,
        namespace: str,
        commitment_hash: str,
        block: int,
    ) -> dict[str, Any]:
        if not isinstance(namespace, str) or not namespace or len(namespace) > 64:
            raise JournalError("invalid commitment namespace")
        if not re.fullmatch(r"[0-9a-f]{64}", commitment_hash):
            raise JournalError("commitment hash must be lowercase SHA256 hex")
        if not isinstance(block, int) or isinstance(block, bool) or block < 0:
            raise JournalError("commitment block must be non-negative")

        def change(journal):
            self._require_active(journal)
            entry = {"namespace": namespace, "hash": commitment_hash, "block": block}
            existing = [item for item in journal["commitments"] if item["namespace"] == namespace]
            if existing and existing[0] != entry:
                raise JournalError("conflicting finalized commitment")
            if not existing:
                journal["commitments"].append(entry)
                journal["commitments"].sort(key=lambda item: item["namespace"])

        return self._mutate(change)[0]

    def reserve_cell(
        self,
        question_id: str,
        model_id: str,
        reserved_cost_usd: int | float | str | Decimal,
    ) -> bool:
        key = cell_key(question_id, model_id)
        reserved = _decimal(reserved_cost_usd, "reserved_cost_usd")

        def change(journal):
            self._require_active(journal)
            existing = journal["cells"].get(key)
            if existing is not None:
                if existing["question_id"] != question_id or existing["model_id"] != model_id:
                    raise JournalError("cell-key collision")
                if _decimal(existing["reserved_cost_usd"], "reserved_cost_usd") != reserved:
                    raise JournalError("conflicting reservation for existing cell")
                return False
            spend = journal["spend"]
            committed = _decimal(spend["actual_usd"], "actual_usd") + _decimal(
                spend["reserved_usd"], "reserved_usd"
            )
            budget = _decimal(spend["budget_usd"], "budget_usd")
            if committed + reserved > budget:
                raise JournalError("cell reservation would exceed epoch budget")
            journal["cells"][key] = {
                "question_id": question_id,
                "model_id": model_id,
                "status": "reserved",
                "response_text": None,
                "response_sha256": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "reserved_cost_usd": _decimal_text(reserved),
                "actual_cost_usd": None,
            }
            return True

        return bool(self._mutate(change)[1])

    def complete_cell(
        self,
        question_id: str,
        model_id: str,
        response_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        actual_cost_usd: int | float | str | Decimal,
    ) -> bool:
        key = cell_key(question_id, model_id)
        if not isinstance(response_text, str):
            raise JournalError("response_text must be a string")
        if len(response_text.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise JournalError("response_text exceeds journal bound")
        for label, tokens in (("prompt_tokens", prompt_tokens), ("completion_tokens", completion_tokens)):
            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
                raise JournalError(f"{label} must be a non-negative integer")
        actual = _decimal(actual_cost_usd, "actual_cost_usd")
        response_hash = hashlib.sha256(response_text.encode("utf-8")).hexdigest()

        def change(journal):
            self._require_active(journal)
            cell = journal["cells"].get(key)
            if cell is None:
                raise JournalError("cell must be reserved before completion")
            if cell["question_id"] != question_id or cell["model_id"] != model_id:
                raise JournalError("cell-key collision")
            if cell["status"] == "complete":
                if (
                    cell["response_sha256"] != response_hash
                    or cell["prompt_tokens"] != prompt_tokens
                    or cell["completion_tokens"] != completion_tokens
                    or _decimal(cell["actual_cost_usd"], "actual_cost_usd") != actual
                ):
                    raise JournalError("conflicting response for completed cell")
                return False
            if cell["status"] != "reserved":
                raise JournalError("forfeited cell cannot be completed or repeated")
            reserved = _decimal(cell["reserved_cost_usd"], "reserved_cost_usd")
            if actual > reserved:
                raise JournalError("actual cell cost exceeds its reservation")
            cell.update({
                "status": "complete",
                "response_text": response_text,
                "response_sha256": response_hash,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "actual_cost_usd": _decimal_text(actual),
            })
            return True

        return bool(self._mutate(change)[1])

    def forfeit_cell(self, question_id: str, model_id: str) -> bool:
        key = cell_key(question_id, model_id)

        def change(journal):
            self._require_active(journal)
            cell = journal["cells"].get(key)
            if cell is None:
                raise JournalError("cell must be reserved before forfeiture")
            if cell["status"] == "forfeited":
                return False
            if cell["status"] != "reserved":
                raise JournalError("completed cell cannot be forfeited")
            cell["status"] = "forfeited"
            cell["actual_cost_usd"] = cell["reserved_cost_usd"]
            return True

        return bool(self._mutate(change)[1])

    def has_inflight_cells(self) -> bool:
        journal = self.read()
        return any(cell["status"] == "reserved" for cell in journal["cells"].values())

    def update_report(
        self,
        *,
        artifact_hash: str | None = None,
        committed_block: int | None = None,
        chunk_count: int | None = None,
        received_hotkeys: list[str] | None = None,
    ) -> dict[str, Any]:
        if artifact_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", artifact_hash):
            raise JournalError("report artifact_hash must be lowercase SHA256 hex")
        if committed_block is not None and (
            not isinstance(committed_block, int)
            or isinstance(committed_block, bool)
            or committed_block < 0
        ):
            raise JournalError("report committed_block must be non-negative")
        if chunk_count is not None and (
            not isinstance(chunk_count, int) or isinstance(chunk_count, bool) or chunk_count <= 0
        ):
            raise JournalError("report chunk_count must be positive")
        if received_hotkeys is not None:
            if any(not isinstance(hotkey, str) or not hotkey for hotkey in received_hotkeys):
                raise JournalError("received report hotkeys must be non-empty strings")
            if len(received_hotkeys) != len(set(received_hotkeys)):
                raise JournalError("received report hotkeys must be unique")

        def change(journal):
            self._require_active(journal)
            report = journal["report"]
            if artifact_hash is not None:
                report["artifact_hash"] = artifact_hash
            if committed_block is not None:
                report["committed_block"] = committed_block
            if chunk_count is not None:
                report["chunk_count"] = chunk_count
            if received_hotkeys is not None:
                report["received_hotkeys"] = sorted(received_hotkeys)

        return self._mutate(change)[0]

    @staticmethod
    def _recompute_spend(journal: dict[str, Any]) -> None:
        reserved = Decimal(0)
        actual = Decimal(0)
        for cell in journal["cells"].values():
            if cell["status"] == "reserved":
                reserved += _decimal(cell["reserved_cost_usd"], "reserved_cost_usd")
            elif cell["status"] in {"complete", "forfeited"}:
                actual += _decimal(cell["actual_cost_usd"], "actual_cost_usd")
        journal["spend"]["reserved_usd"] = _decimal_text(reserved)
        journal["spend"]["actual_usd"] = _decimal_text(actual)

    def _validate(self, journal: object) -> None:
        if not isinstance(journal, dict):
            raise JournalError("journal root must be an object")
        expected = {
            "schema_version", "sequence", "epoch_id", "manifest_hash", "boundary",
            "status", "phase", "abort_reason", "commitments", "cells", "spend", "report",
        }
        if set(journal) != expected:
            raise JournalError("journal root keys differ from schema")
        if journal["schema_version"] != SCHEMA_VERSION:
            raise JournalError("unsupported journal schema version")
        if journal["epoch_id"] != self.epoch_id:
            raise JournalError("journal epoch_id does not match its directory")
        if not isinstance(journal["sequence"], int) or journal["sequence"] < 0:
            raise JournalError("journal sequence must be non-negative")
        if not re.fullmatch(r"[0-9a-f]{64}", journal["manifest_hash"]):
            raise JournalError("invalid journal manifest hash")
        boundary = journal["boundary"]
        if not isinstance(boundary, dict) or set(boundary) != {"block", "hash"}:
            raise JournalError("invalid journal boundary")
        if (
            not isinstance(boundary["block"], int)
            or isinstance(boundary["block"], bool)
            or boundary["block"] < 0
        ):
            raise JournalError("invalid boundary block")
        if not isinstance(boundary["hash"], str) or not boundary["hash"]:
            raise JournalError("invalid boundary hash")
        if journal["status"] not in {"active", "complete", "aborted"}:
            raise JournalError("invalid journal status")
        if journal["phase"] not in PHASES:
            raise JournalError("invalid journal phase")
        if journal["status"] == "complete" and journal["phase"] != "complete":
            raise JournalError("complete journal has inconsistent phase")
        if journal["phase"] == "complete" and journal["status"] != "complete":
            raise JournalError("complete phase has inconsistent status")
        if journal["status"] == "aborted":
            if journal["phase"] != "aborted" or not journal["abort_reason"]:
                raise JournalError("aborted journal needs phase and reason")
        elif journal["abort_reason"] is not None:
            raise JournalError("non-aborted journal cannot have abort_reason")
        if journal["phase"] == "aborted" and journal["status"] != "aborted":
            raise JournalError("aborted phase has inconsistent status")

        if not isinstance(journal["commitments"], list):
            raise JournalError("commitments must be a list")
        namespaces: set[str] = set()
        for commitment in journal["commitments"]:
            if not isinstance(commitment, dict) or set(commitment) != {"namespace", "hash", "block"}:
                raise JournalError("invalid commitment record")
            if commitment["namespace"] in namespaces:
                raise JournalError("duplicate commitment namespace")
            namespaces.add(commitment["namespace"])
            if not re.fullmatch(r"[0-9a-f]{64}", commitment["hash"]):
                raise JournalError("invalid commitment hash")
            if not isinstance(commitment["block"], int) or commitment["block"] < 0:
                raise JournalError("invalid commitment block")

        if not isinstance(journal["cells"], dict):
            raise JournalError("cells must be an object")
        cell_fields = {
            "question_id", "model_id", "status", "response_text", "response_sha256",
            "prompt_tokens", "completion_tokens", "reserved_cost_usd", "actual_cost_usd",
        }
        for key, cell in journal["cells"].items():
            if not re.fullmatch(r"[0-9a-f]{64}", key):
                raise JournalError("invalid cell key")
            if not isinstance(cell, dict) or set(cell) != cell_fields:
                raise JournalError("invalid cell record")
            if key != cell_key(cell["question_id"], cell["model_id"]):
                raise JournalError("cell key does not match question/model")
            if cell["status"] not in CELL_STATUSES:
                raise JournalError("invalid cell status")
            reserved = _decimal(cell["reserved_cost_usd"], "reserved_cost_usd")
            if cell["status"] == "reserved":
                if any(cell[field] is not None for field in (
                    "response_text", "response_sha256", "prompt_tokens",
                    "completion_tokens", "actual_cost_usd",
                )):
                    raise JournalError("reserved cell contains completion data")
            elif cell["status"] == "complete":
                if not isinstance(cell["response_text"], str):
                    raise JournalError("complete cell is missing response text")
                if len(cell["response_text"].encode("utf-8")) > MAX_RESPONSE_BYTES:
                    raise JournalError("complete cell response exceeds bound")
                expected_hash = hashlib.sha256(cell["response_text"].encode("utf-8")).hexdigest()
                if cell["response_sha256"] != expected_hash:
                    raise JournalError("complete cell response hash mismatch")
                if any(
                    not isinstance(cell[field], int) or isinstance(cell[field], bool) or cell[field] < 0
                    for field in ("prompt_tokens", "completion_tokens")
                ):
                    raise JournalError("complete cell has invalid token usage")
                if _decimal(cell["actual_cost_usd"], "actual_cost_usd") > reserved:
                    raise JournalError("complete cell actual cost exceeds reservation")
            else:
                if cell["actual_cost_usd"] != cell["reserved_cost_usd"]:
                    raise JournalError("forfeited cell must charge its full reservation")
                if any(cell[field] is not None for field in (
                    "response_text", "response_sha256", "prompt_tokens", "completion_tokens",
                )):
                    raise JournalError("forfeited cell cannot contain response data")

        spend = journal["spend"]
        if not isinstance(spend, dict) or set(spend) != {
            "budget_usd", "reserved_usd", "actual_usd",
        }:
            raise JournalError("invalid spend record")
        budget = _decimal(spend["budget_usd"], "budget_usd")
        reserved = _decimal(spend["reserved_usd"], "reserved_usd")
        actual = _decimal(spend["actual_usd"], "actual_usd")
        expected_reserved = sum(
            _decimal(cell["reserved_cost_usd"], "reserved_cost_usd")
            for cell in journal["cells"].values()
            if cell["status"] == "reserved"
        )
        expected_actual = sum(
            _decimal(cell["actual_cost_usd"], "actual_cost_usd")
            for cell in journal["cells"].values()
            if cell["status"] in {"complete", "forfeited"}
        )
        if reserved != expected_reserved or actual != expected_actual:
            raise JournalError("journal spend totals do not match cells")
        if reserved + actual > budget:
            raise JournalError("journal committed spend exceeds budget")

        report = journal["report"]
        if not isinstance(report, dict) or set(report) != {
            "artifact_hash", "committed_block", "chunk_count", "received_hotkeys",
        }:
            raise JournalError("invalid report state")
        if report["artifact_hash"] is not None and not re.fullmatch(
            r"[0-9a-f]{64}", report["artifact_hash"]
        ):
            raise JournalError("invalid report artifact hash")
        if report["committed_block"] is not None and (
            not isinstance(report["committed_block"], int) or report["committed_block"] < 0
        ):
            raise JournalError("invalid report committed block")
        if report["chunk_count"] is not None and (
            not isinstance(report["chunk_count"], int) or report["chunk_count"] <= 0
        ):
            raise JournalError("invalid report chunk count")
        hotkeys = report["received_hotkeys"]
        if not isinstance(hotkeys, list) or any(
            not isinstance(hotkey, str) or not hotkey for hotkey in hotkeys
        ) or len(hotkeys) != len(set(hotkeys)):
            raise JournalError("invalid received report hotkeys")
