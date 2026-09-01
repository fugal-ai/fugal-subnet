"""Bounded, versioned messages for the local grading socket."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024
MAX_CODE_BYTES = 512 * 1024
MAX_REPLY_BYTES = 512 * 1024
MAX_CASES = 256
MAX_CASE_JSON_BYTES = 1024 * 1024
_ID_RE = re.compile(r"^[a-zA-Z0-9_.:-]{1,128}$")


class SandboxProtocolError(ValueError):
    """A launcher message violated the bounded grading protocol."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise SandboxProtocolError(f"{label} is invalid")
    return value


def _json_safe(value: object, *, depth: int = 0) -> bool:
    if depth > 20:
        return False
    if value is None or isinstance(value, (str, int, bool)):
        return True
    if isinstance(value, float):
        return value == value and value not in (float("inf"), float("-inf"))
    if isinstance(value, list):
        return all(_json_safe(item, depth=depth + 1) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _json_safe(item, depth=depth + 1)
            for key, item in value.items()
        )
    return False


def validate_request(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SandboxProtocolError("request must be an object")
    common = {"schema_version", "job_id", "operation"}
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise SandboxProtocolError("unsupported schema_version")
    job_id = _require_id(raw.get("job_id"), "job_id")
    operation = raw.get("operation")

    if operation == "health":
        if set(raw) != common:
            raise SandboxProtocolError("health request keys differ")
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "operation": operation,
        }

    if operation == "code_io":
        expected_keys = common | {"code", "function", "inputs", "expected"}
        if set(raw) != expected_keys:
            raise SandboxProtocolError("code_io request keys differ")
        code = raw.get("code")
        function = raw.get("function")
        inputs = raw.get("inputs")
        expected = raw.get("expected")
        if not isinstance(code, str) or len(code.encode("utf-8")) > MAX_CODE_BYTES:
            raise SandboxProtocolError("candidate code is missing or oversized")
        _require_id(function, "function")
        if not isinstance(inputs, list) or not 1 <= len(inputs) <= MAX_CASES:
            raise SandboxProtocolError("inputs must contain 1..256 cases")
        if not isinstance(expected, list) or len(expected) != len(inputs):
            raise SandboxProtocolError("expected outputs must align with inputs")
        if not all(isinstance(case, list) for case in inputs):
            raise SandboxProtocolError("each input case must be an argument list")
        if not _json_safe(inputs) or not _json_safe(expected):
            raise SandboxProtocolError("cases must be finite JSON-safe values")
        if len(canonical_json([inputs, expected])) > MAX_CASE_JSON_BYTES:
            raise SandboxProtocolError("test cases are oversized")
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "operation": operation,
            "code": code,
            "function": function,
            "inputs": inputs,
            "expected": expected,
        }

    if operation == "symbolic_math":
        expected_keys = common | {"gold", "reply"}
        if set(raw) != expected_keys:
            raise SandboxProtocolError("symbolic_math request keys differ")
        gold = raw.get("gold")
        reply = raw.get("reply")
        if not isinstance(gold, str) or not gold or len(gold.encode("utf-8")) > MAX_REPLY_BYTES:
            raise SandboxProtocolError("gold is missing or oversized")
        if not isinstance(reply, str) or len(reply.encode("utf-8")) > MAX_REPLY_BYTES:
            raise SandboxProtocolError("reply is missing or oversized")
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "operation": operation,
            "gold": gold,
            "reply": reply,
        }

    raise SandboxProtocolError("unsupported operation")


def worker_payload(request: dict[str, Any]) -> tuple[dict[str, Any], object | None]:
    """Remove trusted comparison material before entering the OCI worker."""
    if request["operation"] == "health":
        raise SandboxProtocolError("health requests do not enter the worker")
    if request["operation"] == "code_io":
        payload = {
            "schema_version": SCHEMA_VERSION,
            "operation": "code_io",
            "code": request["code"],
            "function": request["function"],
            "inputs": request["inputs"],
        }
        return payload, request["expected"]
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": "symbolic_math",
        "gold": request["gold"],
        "reply": request["reply"],
    }, None


def response(job_id: str, *, passed: bool, reason: str) -> dict[str, Any]:
    _require_id(job_id, "job_id")
    if not isinstance(reason, str) or len(reason) > 256:
        raise SandboxProtocolError("response reason is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "ok": True,
        "passed": bool(passed),
        "reason": reason,
    }


def error_response(job_id: str, reason: str) -> dict[str, Any]:
    safe_job_id = job_id if isinstance(job_id, str) and _ID_RE.fullmatch(job_id) else "invalid"
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": safe_job_id,
        "ok": False,
        "passed": False,
        "reason": reason[:256],
    }


def request_hash(request: dict[str, Any]) -> str:
    """Audit identifier; callers must not log request bodies or gold values."""
    return hashlib.sha256(canonical_json(request)).hexdigest()
