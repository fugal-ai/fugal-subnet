"""Unit tests for the trusted/untrusted grading message boundary."""

from __future__ import annotations

import json
import os

import pytest

from fugal_subnet.sandbox.launcher import GradingService, WorkerFailure
from fugal_subnet.sandbox.protocol import (
    SCHEMA_VERSION,
    SandboxProtocolError,
    canonical_json,
    validate_request,
    worker_payload,
)


def code_request(**overrides):
    request = {
        "schema_version": SCHEMA_VERSION,
        "job_id": "unit-code-1",
        "operation": "code_io",
        "code": "def add(a, b): return a + b",
        "function": "add",
        "inputs": [[1, 2], [4, 9]],
        "expected": [3, 13],
    }
    request.update(overrides)
    return request


def test_worker_payload_never_contains_expected_outputs():
    request = validate_request(code_request())
    payload, expected = worker_payload(request)
    assert expected == [3, 13]
    assert "expected" not in payload
    assert b"13" not in canonical_json({**payload, "inputs": [[1, 2], [4, 8]]})


@pytest.mark.parametrize(
    "change",
    [
        {"extra": True},
        {"code": "x" * (512 * 1024 + 1)},
        {"inputs": []},
        {"inputs": [[float("nan")]], "expected": [0]},
        {"inputs": [[1]], "expected": [1, 2]},
        {"function": "bad function"},
    ],
)
def test_malformed_or_unbounded_code_requests_fail_closed(change):
    with pytest.raises(SandboxProtocolError):
        validate_request(code_request(**change))


def test_canonical_json_rejects_non_finite_values():
    with pytest.raises(ValueError):
        canonical_json({"value": float("inf")})


def test_worker_payload_is_valid_plain_json():
    payload, _ = worker_payload(validate_request(code_request()))
    assert json.loads(canonical_json(payload)) == payload


def test_health_request_is_bounded_and_never_enters_worker():
    request = validate_request({
        "schema_version": SCHEMA_VERSION,
        "job_id": "health-1",
        "operation": "health",
    })
    with pytest.raises(SandboxProtocolError, match="do not enter"):
        worker_payload(request)


class _Peer:
    def getsockopt(self, _level, _option, _length):
        return (
            (0).to_bytes(4, "little", signed=True)
            + os.getuid().to_bytes(4, "little", signed=True)
            + os.getgid().to_bytes(4, "little", signed=True)
        )


class _CandidateFailureRunner:
    def __init__(self, *, canary_works: bool = True):
        self.canary_works = canary_works
        self.canary_calls = 0

    def run(self, _payload):
        raise WorkerFailure("worker_nonzero_exit")

    def verify_worker(self):
        self.canary_calls += 1
        if not self.canary_works:
            raise WorkerFailure("worker_healthcheck_failed")


def test_candidate_controlled_worker_exit_grades_false_after_canary():
    runner = _CandidateFailureRunner()
    service = GradingService(runner, allowed_uids={os.getuid()}, max_concurrency=1)
    result = service.handle(_Peer(), canonical_json(code_request(code="raise SystemExit")))
    assert result["ok"] is True
    assert result["passed"] is False
    assert result["reason"] == "candidate_rejected"
    assert runner.canary_calls == 1


def test_candidate_exit_still_fails_closed_when_worker_canary_fails():
    runner = _CandidateFailureRunner(canary_works=False)
    service = GradingService(runner, allowed_uids={os.getuid()}, max_concurrency=1)
    with pytest.raises(WorkerFailure, match="worker_healthcheck_failed"):
        service.handle(_Peer(), canonical_json(code_request(code="raise SystemExit")))
