"""PROXY_CALL_DEADLINE_S is wall-clock time for the whole upstream call.

It used to be a per-read socket timeout. OpenRouter keeps a non-streamed request
alive while the model reasons, so a call with a "420 s" timeout ran 917 s
(external measurement, 2026-11-05) and the proxy thread and upstream connection
outlived the harness. These upstreams drip, go silent, or answer normally.
"""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from fugal_subnet.tee import runtime as rt

DEADLINE = 1.0
OK = json.dumps({"choices": [{"message": {"content": "42"}}],
                 "usage": {"prompt_tokens": 5, "completion_tokens": 2}}).encode()


def _server(mode, seen):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1" if mode.startswith("chunked") else "HTTP/1.0"

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            t0 = time.monotonic()
            try:
                if mode == "silent":                       # accepted, never answers
                    time.sleep(10)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                if mode.startswith("chunked"):
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                if mode == "normal":
                    self.wfile.write(OK)
                    return
                if mode == "chunked-stall":                # a chunk header, then nothing
                    self.wfile.write(b"400\r\n" + b" " * 10)
                    self.wfile.flush()
                    time.sleep(10)
                    return
                while time.monotonic() - t0 < 10:          # keep-alive whitespace "while reasoning"
                    self.wfile.write(b"1\r\n \r\n" if mode == "chunked-drip" else b" ")
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                seen["closed_after"] = time.monotonic() - t0

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _req(srv):
    return Request(f"http://127.0.0.1:{srv.server_port}/chat/completions", data=b"{}",
                   headers={"Content-Type": "application/json"}, method="POST")


@pytest.fixture
def short_deadline(monkeypatch):
    monkeypatch.setattr(rt, "PROXY_CALL_DEADLINE_S", DEADLINE)


@pytest.mark.parametrize("mode", ["drip", "chunked-drip", "chunked-stall", "silent"])
def test_the_call_ends_at_the_wall_clock_deadline(short_deadline, mode):
    seen = {}
    srv = _server(mode, seen)
    try:
        t0 = time.monotonic()
        with pytest.raises(rt.UpstreamDeadline):
            rt._fetch_upstream(_req(srv), "m")
        took = time.monotonic() - t0
        assert DEADLINE - 0.05 <= took < DEADLINE + 0.5, took
        if mode in ("drip", "chunked-drip"):          # the connection is closed, not abandoned
            time.sleep(0.5)
            assert seen.get("closed_after", 99) < DEADLINE + 0.6
    finally:
        srv.shutdown()


def test_a_normal_reply_is_returned_whole(short_deadline):
    srv = _server("normal", {})
    try:
        assert json.loads(rt._fetch_upstream(_req(srv), "m"))["choices"][0]["message"]["content"] == "42"
    finally:
        srv.shutdown()


def test_an_error_status_still_raises_httperror_with_its_body(short_deadline):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(418); self.end_headers(); self.wfile.write(b'{"error":"teapot"}')

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(HTTPError) as e:
            rt._post_with_deadline(_req(srv), time.monotonic() + DEADLINE)
        assert e.value.code == 418 and b"teapot" in e.value.read()
    finally:
        srv.shutdown()


def test_the_proxy_answers_502_before_the_harness_gives_up(short_deadline, monkeypatch):
    """End to end: the harness gets the proxy's failure, not its own timeout, and no
    late record appears after the deadline."""
    from fugal_subnet.tee import harness as h
    monkeypatch.setattr(h, "HARNESS_CALL_TIMEOUT_S", DEADLINE + 2.0)
    srv = _server("drip", {})
    monkeypatch.setattr(rt, "_OPENROUTER_BASE", f"http://127.0.0.1:{srv.server_port}")
    proxy = rt.MeteringProxy(port=0)
    proxy.api_key, proxy.prices = "test", {"m": (1e-6, 1e-6)}
    proxy.start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", proxy.port), timeout=0.2).close(); break
        except OSError:
            time.sleep(0.02)
    try:
        t0 = time.monotonic()
        assert h._call_model(proxy, "m", {"prompt": "hi"}) is None
        assert time.monotonic() - t0 < DEADLINE + 1.0     # the proxy's 502, well before the harness timeout
        time.sleep(1.0)
        assert proxy.records == []
    finally:
        proxy.stop(); srv.shutdown()
