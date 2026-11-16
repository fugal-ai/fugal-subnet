"""Protocol v3 calls: each menu entry's own deadline, and pricing by entry id.

A v3 entry (`model@effort`) carries its own answer budget, up to ~128k tokens,
so one 420 s deadline for every call would cut off exactly the slow, careful
entries heads must learn about. And the proxy meters a call under the price
table of the epoch, which v3 keys on the ENTRY id, not the provider's model id.
"""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fugal_subnet import config
from fugal_subnet.tee import runtime as rt

V3 = {"model": "a/b", "max_tokens": 1000, "reasoning": {"effort": "max"},
      "provider": {"order": ["p"], "allow_fallbacks": False}}


@pytest.mark.parametrize("request_body, want", [
    ({"model": "a/b", "max_tokens": 16384}, config.PROXY_CALL_DEADLINE_S),                  # v1
    ({"model": "a/b", "reasoning": {"effort": "medium"}}, config.PROXY_CALL_DEADLINE_S),    # v2
    (V3, 1800.0),
    ({**V3, "reasoning": {"effort": "low"}}, 600.0),
    ({**V3, "reasoning": {"effort": "medium"}}, 900.0),
    ({**V3, "reasoning": {"enabled": True}}, config.PROXY_CALL_DEADLINE_V3_DEFAULT_S),
    ({k: v for k, v in V3.items() if k != "reasoning"}, config.PROXY_CALL_DEADLINE_V3_DEFAULT_S),
])
def test_a_v3_entry_has_its_own_deadline_and_older_calls_keep_theirs(request_body, want):
    assert config.call_deadline_s(request_body) == want


def test_every_v3_deadline_ends_before_the_collection_point():
    """Validators collect at EPOCH_COLLECT_FRACTION of a 72-minute epoch; a call
    that can outlive that is a proof that cannot exist in time."""
    collect_s = config.EPOCH_COLLECT_FRACTION * 72 * 60
    for d in (*config.PROXY_CALL_DEADLINE_V3_S.values(), config.PROXY_CALL_DEADLINE_V3_DEFAULT_S):
        assert d + config.HARNESS_WAIT_MARGIN_S + 300 <= collect_s


def _drip_server():
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                self.send_response(200)
                self.end_headers()
                t0 = time.monotonic()
                while time.monotonic() - t0 < 10:
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                pass

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _started_proxy(monkeypatch, upstream_port, prices):
    monkeypatch.setattr(rt, "_OPENROUTER_BASE", f"http://127.0.0.1:{upstream_port}")
    proxy = rt.MeteringProxy(port=0)
    proxy.api_key, proxy.prices = "test", prices
    proxy.start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", proxy.port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.02)
    return proxy


def test_the_proxy_holds_a_v3_call_to_its_entrys_deadline(monkeypatch):
    """The v2 deadline is short here; the v3 'max' deadline is longer. A v3 call
    must run to its own deadline, not the v2 one."""
    from fugal_subnet.tee import harness as h
    monkeypatch.setattr(rt, "PROXY_CALL_DEADLINE_S", 0.5)
    monkeypatch.setattr(config, "PROXY_CALL_DEADLINE_V3_S", {**config.PROXY_CALL_DEADLINE_V3_S, "max": 1.5})
    monkeypatch.setattr(config, "HARNESS_WAIT_MARGIN_S", 2.0)
    srv = _drip_server()
    proxy = _started_proxy(monkeypatch, srv.server_port, {"a/b@max": (1e-6, 1e-6)})
    try:
        t0 = time.monotonic()
        assert h._call_model(proxy, "a/b@max", {"prompt": "hi"}, request=dict(V3)) is None
        took = time.monotonic() - t0
        assert 1.4 <= took < 2.5, took
    finally:
        proxy.stop()
        srv.shutdown()


def test_the_harness_waits_past_a_v3_entrys_deadline(monkeypatch):
    from fugal_subnet.tee import harness as h
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["timeout"] = timeout
        raise OSError("stop here")

    monkeypatch.setattr(h, "urlopen", fake_urlopen)
    monkeypatch.setattr(h, "_proxy_never_answered", lambda e: False)

    class P:
        port = 1

    h._call_model(P(), "a/b@max", {"prompt": "hi"}, request=dict(V3))
    assert seen["timeout"] == 1800.0 + config.HARNESS_WAIT_MARGIN_S
    h._call_model(P(), "a/b", {"prompt": "hi"})
    assert seen["timeout"] == h.HARNESS_CALL_TIMEOUT_S


def test_the_proxy_prices_a_v3_call_by_its_entry_id(monkeypatch):
    """v3 prices are keyed on `model@effort`; the request carries the provider's
    model id. Pricing by the request's model would refuse every multi-effort
    entry as unpriced."""
    ok = json.dumps({"choices": [{"message": {"content": "42"}}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 20}}).encode()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(ok)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from fugal_subnet.tee import harness as h
    proxy = _started_proxy(monkeypatch, srv.server_port, {"a/b@max": (1e-6, 2e-6)})
    try:
        assert h._call_model(proxy, "a/b@max", {"prompt": "hi"}, request=dict(V3)) == "42"
        (rec,) = proxy.records
        assert rec.model_id == "a/b@max"
        assert rec.cost_usd == pytest.approx(10 * 1e-6 + 20 * 2e-6)
    finally:
        proxy.stop()
        srv.shutdown()
