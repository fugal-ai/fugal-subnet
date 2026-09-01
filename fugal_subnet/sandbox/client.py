"""Permission-restricted Unix-socket client used by v2 graders."""

from __future__ import annotations

import json
import os
import socket
import stat
import uuid
from pathlib import Path

from fugal_subnet.sandbox.protocol import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    SCHEMA_VERSION,
    SandboxProtocolError,
    canonical_json,
    validate_request,
)


class GradingUnavailable(RuntimeError):
    """The required isolated grader is absent or returned an invalid result."""


class GradingClient:
    def __init__(self, socket_path: str | os.PathLike[str], timeout_seconds: float = 15.0):
        self.socket_path = Path(socket_path)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)

    def _call(self, request: dict) -> bool:
        request = validate_request(request)
        encoded = canonical_json(request) + b"\n"
        if len(encoded) > MAX_REQUEST_BYTES:
            raise SandboxProtocolError("encoded request is oversized")
        try:
            info = self.socket_path.lstat()
            if not stat.S_ISSOCK(info.st_mode):
                raise GradingUnavailable("grading endpoint is not a Unix socket")
            if info.st_mode & 0o007:
                raise GradingUnavailable("grading socket is accessible to other users")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout_seconds)
                sock.connect(str(self.socket_path))
                sock.sendall(encoded)
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = sock.recv(min(4096, MAX_RESPONSE_BYTES + 1 - total))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > MAX_RESPONSE_BYTES:
                        raise GradingUnavailable("grading response exceeded its bound")
                    if b"\n" in chunk:
                        break
        except GradingUnavailable:
            raise
        except (OSError, TimeoutError) as exc:
            raise GradingUnavailable(f"isolated grading service unavailable: {exc}") from exc

        payload = b"".join(chunks)
        if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
            raise GradingUnavailable("grading service returned a malformed frame")
        try:
            result = json.loads(payload[:-1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GradingUnavailable("grading service returned invalid JSON") from exc
        expected_keys = {"schema_version", "job_id", "ok", "passed", "reason"}
        if not isinstance(result, dict) or set(result) != expected_keys:
            raise GradingUnavailable("grading response schema differs")
        if result["schema_version"] != SCHEMA_VERSION or result["job_id"] != request["job_id"]:
            raise GradingUnavailable("grading response does not match the request")
        if type(result["ok"]) is not bool or type(result["passed"]) is not bool:
            raise GradingUnavailable("grading response booleans are invalid")
        if not isinstance(result["reason"], str) or len(result["reason"]) > 256:
            raise GradingUnavailable("grading response reason is invalid")
        if not result["ok"]:
            raise GradingUnavailable(f"isolated grading failed closed: {result['reason']}")
        return result["passed"]

    def grade_code(
        self,
        *,
        code: str,
        function: str,
        inputs: list[list],
        expected: list,
        job_id: str | None = None,
    ) -> bool:
        return self._call({
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id or f"code-{uuid.uuid4().hex}",
            "operation": "code_io",
            "code": code,
            "function": function,
            "inputs": inputs,
            "expected": expected,
        })

    def health(self) -> bool:
        """Prove the restricted launcher initialized its pinned worker image."""
        return self._call({
            "schema_version": SCHEMA_VERSION,
            "job_id": f"health-{uuid.uuid4().hex}",
            "operation": "health",
        })

    def grade_symbolic_math(
        self,
        *,
        gold: str,
        reply: str,
        job_id: str | None = None,
    ) -> bool:
        return self._call({
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id or f"math-{uuid.uuid4().hex}",
            "operation": "symbolic_math",
            "gold": gold,
            "reply": reply,
        })
