"""The metering proxy's own rules, end to end through its HTTP handler.

Planted bugs (2026-11-08) showed 12 of 23 rules in tee/runtime.py untested:
the proxy listening on every interface (it holds the miner's API key),
dropping the request id the harness attributes cost by, failing a call over an
unreadable provider-cost field, totals that depend on the order calls complete,
no fallback when its port is taken, the retry budget, the watchdog that ends a
reply whose HEADERS drip, and the dstack guest agent being the first quote
source. Each case here fails if, and only if, its rule is removed.
"""
from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest

from fugal_subnet.tee import runtime as rt

MODEL = "openai/gpt-5.4-nano"
PRICES = {MODEL: (0.2e-6, 1.25e-6)}


def _upstream(body: bytes, status: int = 200):
    """A one-reply-fits-all OpenAI-compatible upstream on an ephemeral port."""
    class H(BaseHTTPRequestHandler):
        hits = 0

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            H.hits += 1
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, H


def _wait_listening(port):
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.02)


@pytest.fixture
def proxy_to(monkeypatch):
    started = []

    def make(body, status=200, port=0):
        srv, handler = _upstream(body, status)
        monkeypatch.setattr(rt, "_OPENROUTER_BASE", f"http://127.0.0.1:{srv.server_port}")
        monkeypatch.setattr(rt.time, "sleep", lambda s: None)
        proxy = rt.MeteringProxy(port=port, api_key="test", prices=dict(PRICES))
        proxy.start()
        _wait_listening(proxy.port)
        started.append((proxy, srv))
        return proxy, handler

    yield make
    for proxy, srv in started:
        proxy.stop()
        srv.shutdown()


def _call(proxy, request_id="e1:q1:" + MODEL):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}).encode()
    req = Request(f"http://127.0.0.1:{proxy.port}/chat/completions", data=body, method="POST",
                  headers={"Content-Type": "application/json", "X-Fugal-Request": request_id})
    with urlopen(req, timeout=30) as r:
        return r.status, r.read()


def _reply(usage):
    return json.dumps({"choices": [{"message": {"content": "42"}}], "usage": usage}).encode()


def test_a_call_is_recorded_whole(proxy_to):
    """Request id (the harness bills by it), tokens, the PINNED-table cost,
    the provider's own figure, and the hash of the exact body relayed."""
    body = _reply({"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.00042})
    proxy, _ = proxy_to(body)
    status, relayed = _call(proxy, request_id="e9:q7:" + MODEL)
    assert status == 200 and relayed == body
    (rec,) = proxy.records
    assert rec.request_id == "e9:q7:" + MODEL
    assert (rec.prompt_tokens, rec.completion_tokens) == (5, 2)
    assert rec.cost_usd == 5 * 0.2e-6 + 2 * 1.25e-6
    assert rec.provider_cost_usd == 0.00042
    assert rec.response_hash == hashlib.sha256(body).hexdigest()


@pytest.mark.parametrize("cost", ["n/a", None, [1]])
def test_an_unreadable_provider_cost_never_fails_the_call(proxy_to, cost):
    """The provider's figure is a cross-check only; the consensus cost comes
    from the pinned table, so a malformed one is 0, not a failed answer."""
    proxy, _ = proxy_to(_reply({"prompt_tokens": 5, "completion_tokens": 2, "cost": cost}))
    status, _ = _call(proxy)
    assert status == 200 and proxy.records[0].provider_cost_usd == 0.0


def test_the_proxy_listens_on_loopback_only(proxy_to):
    """It forwards with the miner's OpenRouter key: on any other interface,
    anyone who can reach the TD's port spends that key."""
    proxy, _ = proxy_to(_reply({"prompt_tokens": 1, "completion_tokens": 1}))
    assert proxy._server.server_address[0] == "127.0.0.1"


def test_a_taken_port_falls_back_to_an_ephemeral_one(proxy_to):
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    taken = holder.getsockname()[1]
    try:
        proxy, _ = proxy_to(_reply({"prompt_tokens": 1, "completion_tokens": 1}), port=taken)
        assert proxy.port != taken
        assert _call(proxy)[0] == 200
    finally:
        holder.close()


def test_totals_do_not_depend_on_the_order_calls_complete():
    """With calls in flight, records arrive in a different order each run. A
    left-to-right float sum rounds with the order; the attested totals must not."""
    def proxy_with(costs):
        p = rt.MeteringProxy()
        p.records = [rt.APICallRecord(m, 1, 1, c, 0.0, "") for m, c in costs]
        return p

    calls = [(MODEL, 0.1), (MODEL, 0.2), (MODEL, 0.3), ("other/m", 0.1), ("other/m", 0.2),
             ("other/m", 0.3)]
    a, b = proxy_with(calls), proxy_with(list(reversed(calls)))
    assert a.total_cost == b.total_cost
    assert a.per_model_costs == b.per_model_costs
    assert list(a.per_model_costs) == sorted(a.per_model_costs)


def test_no_retry_starts_after_the_retry_budget(proxy_to, monkeypatch):
    """A 429 is retried, but not once the call has used PROXY_RETRY_BUDGET_S."""
    clock = iter([0.0] + [rt.PROXY_RETRY_BUDGET_S + 1.0] * 50)
    monkeypatch.setattr(rt, "time", SimpleNamespace(
        time=lambda: next(clock), sleep=lambda s: None, monotonic=time.monotonic))
    proxy, handler = proxy_to(b'{"error":"busy"}', status=429)
    with pytest.raises(Exception):
        _call(proxy)
    assert handler.hits == 1


def test_a_reply_whose_headers_drip_ends_at_the_deadline(monkeypatch):
    """Every read returns a byte in time, so no per-read timeout fires while the
    status line and headers trickle in: only the watchdog ends it."""
    class Drip(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                for b in b"HTTP/1.1 200 OK\r\nX-Slow: " + b"a" * 400:
                    self.wfile.write(bytes([b]))
                    self.wfile.flush()
                    time.sleep(0.02)
            except OSError:
                pass

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Drip)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        req = Request(f"http://127.0.0.1:{srv.server_port}/v1/chat/completions", data=b"{}",
                      method="POST", headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with pytest.raises(rt.UpstreamDeadline):
            rt._post_with_deadline(req, time.monotonic() + 0.5)
        assert time.monotonic() - t0 < 3.0            # the drip alone would take ~8 s
    finally:
        srv.shutdown()


def test_under_dstack_the_guest_agent_is_the_only_quote_source(monkeypatch):
    """A bare configfs quote has no event log, and every validator refuses it as
    an unapproved image; under dstack the envelope is the only usable proof."""
    from fugal_subnet.tee import dstack_client

    monkeypatch.setattr(dstack_client, "available", lambda path="": True)
    monkeypatch.setattr(dstack_client, "attest", lambda report_data: b"envelope:" + report_data[:4])

    def configfs(_):
        raise AssertionError("configfs consulted although the guest agent is present")

    monkeypatch.setattr(rt, "_quote_via_configfs", configfs)
    assert rt._real_quote(b"abcd") == b"envelope:abcd"
