"""Client for the dstack guest agent's unix socket.

Under a dstack image this is the ONLY correct quote source, and reaching for
configfs-tsm instead produces a proof that is rejected. configfs returns a bare
TDX quote with no event log, and RTMR3 on a measured image is a chain that
includes per-deploy values, so without the log there is nothing to replay and
the approved-entry check has no choice but to refuse. The failure would look
like an unapproved image rather than a wrong API call.

The guest agent returns the whole envelope — TDX quote, event log, and on GCP a
TPM quote — which is the shape `verify.unwrap_attestation` expects.

Verified against a running deploy rather than documentation:

    POST http://localhost/v1/Attest   over /var/run/dstack.sock
    {"report_data": "<hex>"}
    -> {"attestation": "<hex>", "boottime_gpu_evidence": []}

The socket is internal to the VM and not reachable from outside it.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import socket

logger = logging.getLogger(__name__)

DSTACK_SOCKET = os.getenv("DSTACK_SOCKET", "/var/run/dstack.sock")

# A guest agent's reply is a hex string of a ~40KB blob, so ~80KB plus JSON.
# Capped anyway: an unbounded read from a socket is an unbounded read.
_MAX_RESPONSE = 8 * 1024 * 1024
_TIMEOUT = 30


class DstackError(RuntimeError):
    """The guest agent could not be reached, or did not answer usefully."""


class _UnixConnection(http.client.HTTPConnection):
    """HTTPConnection over a unix domain socket."""

    def __init__(self, path: str, timeout: int = _TIMEOUT):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._path)
        self.sock = sock


def available(path: str = "") -> bool:
    """Is a dstack guest agent socket present? Cheap enough to call on a path."""
    target = path or DSTACK_SOCKET
    try:
        import stat
        return stat.S_ISSOCK(os.stat(target).st_mode)
    except OSError:
        return False


def _call(method: str, payload: dict, path: str = "") -> dict:
    target = path or DSTACK_SOCKET
    conn = _UnixConnection(target)
    try:
        conn.request(
            "POST", f"/v1/{method}",
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        raw = resp.read(_MAX_RESPONSE + 1)
        if len(raw) > _MAX_RESPONSE:
            raise DstackError(f"/v1/{method} response exceeds {_MAX_RESPONSE} bytes")
        if resp.status != 200:
            raise DstackError(
                f"/v1/{method} returned HTTP {resp.status}: {raw[:200]!r}"
            )
    except OSError as e:
        raise DstackError(
            f"cannot reach the dstack guest agent at {target}: {e}. "
            "Under dstack this socket must be mounted into the container "
            "(volumes: - /var/run/dstack.sock:/var/run/dstack.sock)."
        ) from e
    finally:
        conn.close()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise DstackError(f"/v1/{method} did not return JSON: {e}") from e


def attest(report_data: bytes, path: str = "") -> bytes:
    """Ask the guest agent for a full attestation binding `report_data`.

    The 64 bytes are padded HERE, not left to the guest. The guest does
    right-zero-pad a short value — measured, it neither hashes nor rejects one —
    but that is behaviour we observed rather than behaviour we control, and the
    exact bytes in `report_data` are what binds a proof to its body. Padding
    locally means the value in the quote is decided by this function and a change
    in the agent's convention becomes a failed check rather than a silent
    re-binding.
    """
    if len(report_data) > 64:
        raise ValueError(f"report_data is {len(report_data)} bytes, maximum 64")
    padded = report_data.ljust(64, b"\x00")

    reply = _call("Attest", {"report_data": padded.hex()}, path)
    hex_blob = reply.get("attestation")
    if not isinstance(hex_blob, str) or not hex_blob:
        raise DstackError(
            f"/v1/Attest returned no attestation (keys: {sorted(reply)})"
        )
    try:
        blob = bytes.fromhex(hex_blob)
    except ValueError as e:
        raise DstackError(f"/v1/Attest returned malformed hex: {e}") from e

    logger.info("dstack attestation obtained (%d bytes)", len(blob))
    return blob


def info(path: str = "") -> dict:
    """Guest self-description: app_id, compose_hash, instance_id, app_compose...

    `app_compose` is the exact bytes the guest hashed, which is the only
    trustworthy source for a compose hash — a locally written file is only the
    same thing if nothing re-serialised it on the way in.

    Unlike `attest`, this call's verb has not been confirmed against a live
    agent by this code. A wrong verb surfaces as an HTTP status from `_call`,
    which is a loud failure and not a silent wrong answer.
    """
    return _call("Info", {}, path)
