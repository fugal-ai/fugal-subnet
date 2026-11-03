"""The metering proxy retries transient upstream errors and never retries a payment problem.

Mainnet day one (2026-11-03): one provider's 429s failed ~13% of a seed's calls,
each scored as a wrong answer; an exhausted account balance (402) failed ~55% of
an epoch and read, at first, like a bad head. Both are operator-visible now.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from fugal_subnet.tee import runtime as rt


class _Upstream:
    """A scripted OpenAI-compatible upstream: a list of (status, retry_after) then 200s."""

    def __init__(self, script):
        self.script = list(script)
        self.hits = 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
                outer.hits += 1
                status, retry_after = outer.script.pop(0) if outer.script else (200, None)
                if status != 200:
                    self.send_response(status)
                    if retry_after is not None:
                        self.send_header("Retry-After", str(retry_after))
                    self.end_headers(); self.wfile.write(b"{}")
                    return
                body = json.dumps({"choices": [{"message": {"content": "42"}}],
                                   "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.0}}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def stop(self):
        self.server.shutdown()


def _proxy_call(proxy, model="openai/gpt-5.4-nano"):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).encode()
    req = Request(f"http://127.0.0.1:{proxy.port}/chat/completions", data=body,
                  headers={"Content-Type": "application/json", "X-Fugal-Request": "t"}, method="POST")
    with urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read())


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(rt.time, "sleep", lambda s: None)   # backoff is policy, not a test budget


def _run(monkeypatch, script):
    up = _Upstream(script)
    monkeypatch.setattr(rt, "_OPENROUTER_BASE", up.base)
    proxy = rt.MeteringProxy(port=0)
    proxy.api_key = "test"
    proxy.prices = {"openai/gpt-5.4-nano": (0.2e-6, 1.25e-6)}
    proxy.start()
    import socket
    import time as _t
    for _ in range(100):                       # the server thread may not be accepting yet
        try:
            socket.create_connection(("127.0.0.1", proxy.port), timeout=0.2).close(); break
        except OSError:
            _t.sleep(0.02)
    try:
        try:
            status, data = _proxy_call(proxy)
        except HTTPError as e:
            status, data = e.code, None
        return status, data, up.hits, list(proxy.records)
    finally:
        proxy.stop(); up.stop()


def test_a_429_then_200_is_one_answered_call(monkeypatch, fast):
    status, data, hits, records = _run(monkeypatch, [(429, "0")])
    assert status == 200 and data["choices"][0]["message"]["content"] == "42"
    assert hits == 2
    assert len(records) == 1 and records[0].completion_tokens == 2


def test_persistent_429_fails_after_the_attempt_cap(monkeypatch, fast):
    status, data, hits, records = _run(monkeypatch, [(429, None)] * 10)
    assert status == 502 and hits == rt.PROXY_MAX_ATTEMPTS and records == []


def test_402_is_never_retried_and_names_the_operator(monkeypatch, fast, caplog):
    import logging
    with caplog.at_level(logging.ERROR, logger="fugal_subnet.tee.runtime"):
        status, data, hits, records = _run(monkeypatch, [(402, None)])
    assert status == 502 and hits == 1 and records == []
    assert any("OPERATOR ACTION" in r.getMessage() and "402" in r.getMessage() for r in caplog.records)


def test_retry_wait_honours_retry_after_and_the_cap():
    assert 1.0 <= rt._retry_wait(1, None) <= 2.0
    assert rt._retry_wait(1, "7") >= 7.0
    assert rt._retry_wait(3, "999") == rt.PROXY_MAX_WAIT_S


def test_failed_results_carry_canonical_number_types(monkeypatch):
    """A failed call's cost is written as float 0.0 and its tokens as ints, so the
    JSON the validator hashes is the JSON the miner hashed (closes #120 from the
    miner side)."""
    e2e = pytest.importorskip("tests.test_tee_e2e")
    from fugal_subnet.tee import harness as hm
    monkeypatch.setattr(hm, "benchmark_costs", e2e.success_fixtures.costs)
    def failing_call(proxy, model_id, question):
        return None if model_id == e2e.MODELS[0] else e2e._stub_call(proxy, model_id, question)
    monkeypatch.setattr(hm, "_call_model", failing_call)
    import numpy as np
    pool = e2e._pool(); hidden = np.random.RandomState(0).randn(len(pool), e2e.HEAD_HIDDEN_DIM).astype(np.float32)
    proof = e2e._run_miner_epoch(e2e._head(e2e.MODELS), "ab" * 32, pool, hidden, "e1")
    failed = [r for r in proof.results if r.failed]
    assert failed, "the stub should have produced at least one failed call"
    for r in failed:
        assert type(r.cost_usd) is float and r.cost_usd == 0.0
        assert type(r.prompt_tokens) is int and type(r.completion_tokens) is int
    wire = json.loads(json.dumps(proof.to_dict()))
    from fugal_subnet.tee.proof import BenchmarkProof
    assert BenchmarkProof.from_dict(wire).content_hash() == proof.content_hash()
