#!/usr/bin/env python3
"""Non-root OCI grading launcher exposed over a restricted Unix socket."""

from __future__ import annotations

import argparse
import grp
import json
import logging
import os
import pwd
import selectors
import shutil
import socket
import socketserver
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fugal_subnet.sandbox.protocol import (
    MAX_REQUEST_BYTES,
    SandboxProtocolError,
    canonical_json,
    error_response,
    request_hash,
    response,
    validate_request,
    worker_payload,
)

logger = logging.getLogger("fugal.sandbox_launcher")
MAX_WORKER_OUTPUT_BYTES = 1024 * 1024
IMAGE_ID_PREFIX = "sha256:"


class WorkerFailure(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


_WORKER_CANARY = {
    "schema_version": 1,
    "operation": "code_io",
    "code": "def fugal_worker_canary(value): return value",
    "function": "fugal_worker_canary",
    "inputs": [["ready"]],
}
_CANDIDATE_CONTROLLED_FAILURES = frozenset({
    "worker_timeout",
    "worker_output_limit",
    "worker_nonzero_exit",
    "worker_malformed_output",
    "worker_result_schema",
})


def _engine_environment() -> dict[str, str]:
    """Give the CLI only connection/config values, never validator secrets."""
    keep = {"PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "XDG_RUNTIME_DIR"}
    return {key: value for key, value in os.environ.items() if key in keep}


class OciRunner:
    def __init__(
        self,
        *,
        engine: str,
        image: str,
        timeout_seconds: float = 12.0,
        memory: str = "256m",
        cpus: str = "1.0",
        pids_limit: int = 1,
    ):
        if not image.startswith(IMAGE_ID_PREFIX) or len(image) != 71:
            raise ValueError("worker image must be pinned by sha256 image ID")
        if timeout_seconds <= 0 or pids_limit != 1:
            raise ValueError("worker timeout must be positive and pids_limit must equal 1")
        resolved = shutil.which(engine)
        if resolved is None:
            raise ValueError(f"container engine not found: {engine}")
        self.engine = resolved
        self.image = image
        self.timeout_seconds = float(timeout_seconds)
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.environment = _engine_environment()
        self._verify_image()
        self.verify_worker()

    def _engine(self, *args: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.engine, *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.environment,
            timeout=timeout,
            check=False,
        )

    def _verify_image(self) -> None:
        inspected = self._engine("image", "inspect", "--format", "{{.Id}}", self.image)
        actual = inspected.stdout.decode("ascii", "replace").strip()
        if inspected.returncode != 0 or actual != self.image:
            raise ValueError("pinned worker image is unavailable or hash-mismatched")

    def _cleanup(self, container_name: str) -> None:
        self._engine("rm", "--force", container_name, timeout=5.0)

    def verify_worker(self) -> None:
        """Execute a trusted canary so health means the worker really runs."""
        try:
            result = self.run(_WORKER_CANARY)
        except WorkerFailure as exc:
            raise WorkerFailure("worker_healthcheck_failed") from exc
        if result != {"outputs": ["ready"]}:
            raise WorkerFailure("worker_healthcheck_failed")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = canonical_json(payload)
        container_name = f"fugal-grade-{uuid.uuid4().hex}"
        command = [
            self.engine,
            "run",
            "--rm",
            "--pull",
            "never",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--user",
            "65532:65532",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--security-opt",
            "seccomp=builtin",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory,
            "--memory-swap",
            self.memory,
            "--cpus",
            self.cpus,
            "--ulimit",
            "nofile=64:64",
            "--ulimit",
            "core=0:0",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=16777216,mode=1777",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONHASHSEED=0",
            "--env",
            "PYTHONUNBUFFERED=1",
            "--attach",
            "stdin",
            "--interactive",
            "--attach",
            "stdout",
            "--attach",
            "stderr",
            self.image,
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.environment,
            start_new_session=True,
        )
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        try:
            process.stdin.write(encoded)
            process.stdin.close()
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            streams = {"stdout": bytearray(), "stderr": bytearray()}
            deadline = time.monotonic() + self.timeout_seconds
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WorkerFailure("worker_timeout")
                events = selector.select(min(remaining, 0.25))
                if not events and process.poll() is not None:
                    for key in list(selector.get_map().values()):
                        chunk = os.read(key.fd, 65536)
                        if chunk:
                            streams[key.data].extend(chunk)
                        selector.unregister(key.fileobj)
                    break
                for key, _ in events:
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    streams[key.data].extend(chunk)
                    if sum(len(value) for value in streams.values()) > MAX_WORKER_OUTPUT_BYTES:
                        raise WorkerFailure("worker_output_limit")
            remaining = max(0.01, deadline - time.monotonic())
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise WorkerFailure("worker_timeout") from exc
            if returncode != 0:
                raise WorkerFailure("worker_nonzero_exit")
            raw = bytes(streams["stdout"])
            if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
                raise WorkerFailure("worker_malformed_output")
            try:
                result = json.loads(raw[:-1].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WorkerFailure("worker_malformed_output") from exc
            if not isinstance(result, dict):
                raise WorkerFailure("worker_malformed_output")
            return result
        except WorkerFailure:
            self._cleanup(container_name)
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            raise
        finally:
            for stream in (process.stdout, process.stderr):
                stream.close()


class GradingService:
    def __init__(self, runner: OciRunner, *, allowed_uids: set[int], max_concurrency: int):
        if not allowed_uids or max_concurrency <= 0:
            raise ValueError("allowed_uids and max_concurrency must be non-empty/positive")
        self.runner = runner
        self.allowed_uids = frozenset(allowed_uids)
        self.capacity = threading.BoundedSemaphore(max_concurrency)

    def _peer_uid(self, connection: socket.socket) -> int:
        if not hasattr(socket, "SO_PEERCRED"):
            raise WorkerFailure("peer_credentials_unavailable")
        credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        return int.from_bytes(credentials[4:8], byteorder="little", signed=True)

    def _candidate_rejected(
        self,
        request: dict[str, Any],
        audit_hash: str,
        failure: WorkerFailure,
    ) -> dict[str, Any]:
        """Return a grade of zero only after a fresh trusted worker canary.

        Candidate code controls the worker process and can exit, hang, or write
        malformed/oversized stdout. Those are ordinary failed answers, not
        launcher outages. A fresh canary distinguishes them from a broken OCI
        boundary; if it cannot run, the request still fails closed.
        """
        self.runner.verify_worker()
        logger.info(
            "job=%s request_sha256=%s operation=code_io passed=False reason=%s",
            request["job_id"],
            audit_hash,
            failure.reason,
        )
        return response(request["job_id"], passed=False, reason="candidate_rejected")

    def handle(self, connection: socket.socket, raw: bytes) -> dict[str, Any]:
        if self._peer_uid(connection) not in self.allowed_uids:
            raise WorkerFailure("peer_uid_denied")
        request = validate_request(json.loads(raw.decode("utf-8")))
        audit_hash = request_hash(request)
        if request["operation"] == "health":
            logger.info("job=%s request_sha256=%s operation=health", request["job_id"], audit_hash)
            return response(request["job_id"], passed=True, reason="ready")
        acquired = self.capacity.acquire(timeout=1.0)
        if not acquired:
            raise WorkerFailure("launcher_busy")
        try:
            payload, expected = worker_payload(request)
            try:
                result = self.runner.run(payload)
            except WorkerFailure as exc:
                if (
                    request["operation"] == "code_io"
                    and exc.reason in _CANDIDATE_CONTROLLED_FAILURES
                ):
                    return self._candidate_rejected(request, audit_hash, exc)
                raise
            if request["operation"] == "code_io" and set(result) != {"outputs"}:
                return self._candidate_rejected(
                    request,
                    audit_hash,
                    WorkerFailure("worker_result_schema"),
                )
        finally:
            self.capacity.release()

        if request["operation"] == "code_io":
            passed = canonical_json(result["outputs"]) == canonical_json(expected)
        else:
            if set(result) != {"passed"} or type(result["passed"]) is not bool:
                raise WorkerFailure("worker_result_schema")
            passed = result["passed"]
        logger.info(
            "job=%s request_sha256=%s operation=%s passed=%s",
            request["job_id"],
            audit_hash,
            request["operation"],
            passed,
        )
        return response(request["job_id"], passed=passed, reason="graded")


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False
    service: GradingService
    request_timeout: float


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.connection.settimeout(self.server.request_timeout)  # type: ignore[attr-defined]
        job_id = "invalid"
        try:
            raw = self.rfile.readline(MAX_REQUEST_BYTES + 2)
            if not raw.endswith(b"\n") or len(raw) > MAX_REQUEST_BYTES + 1:
                raise SandboxProtocolError("request frame is missing or oversized")
            parsed = json.loads(raw[:-1].decode("utf-8"))
            if isinstance(parsed, dict) and isinstance(parsed.get("job_id"), str):
                job_id = parsed["job_id"]
            result = self.server.service.handle(self.connection, raw[:-1])  # type: ignore[attr-defined]
        except (SandboxProtocolError, WorkerFailure, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("job=%s failed_closed=%s", job_id, exc)
            result = error_response(job_id, str(exc))
        except Exception:
            logger.exception("job=%s failed_closed=internal_error", job_id)
            result = error_response(job_id, "internal_error")
        self.wfile.write(canonical_json(result) + b"\n")


def _resolve_identity(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return pwd.getpwnam(value).pw_uid


def serve(args: argparse.Namespace) -> None:
    if os.geteuid() == 0 and not args.allow_root_for_tests:
        raise SystemExit("sandbox launcher refuses to run as root")
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists() or socket_path.is_symlink():
        if not socket_path.is_socket():
            raise SystemExit("refusing to replace a non-socket grading endpoint")
        socket_path.unlink()

    allowed_uids = {_resolve_identity(value) for value in args.allowed_uid}
    runner = OciRunner(
        engine=args.engine,
        image=args.image,
        timeout_seconds=args.timeout,
        memory=args.memory,
        cpus=args.cpus,
    )
    service = GradingService(
        runner,
        allowed_uids=allowed_uids,
        max_concurrency=args.max_concurrency,
    )
    old_umask = os.umask(0o117)
    try:
        server = _UnixServer(str(socket_path), _Handler)
    finally:
        os.umask(old_umask)
    server.service = service
    server.request_timeout = args.timeout + 5.0
    os.chmod(socket_path, 0o660)
    if args.socket_group:
        os.chown(socket_path, -1, grp.getgrnam(args.socket_group).gr_gid)
    logger.info(
        "launcher ready socket=%s image=%s allowed_uids=%s",
        socket_path,
        args.image,
        sorted(allowed_uids),
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--socket", required=True)
    result.add_argument("--image", required=True, help="Exact sha256 OCI image ID")
    result.add_argument("--engine", default="docker", choices=("docker", "podman"))
    result.add_argument("--allowed-uid", action="append", required=True)
    result.add_argument("--socket-group")
    result.add_argument("--timeout", type=float, default=12.0)
    result.add_argument("--max-concurrency", type=int, default=4)
    result.add_argument("--memory", default="256m")
    result.add_argument("--cpus", default="1.0")
    result.add_argument("--allow-root-for-tests", action="store_true", help=argparse.SUPPRESS)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve(parser().parse_args())


if __name__ == "__main__":
    main()
