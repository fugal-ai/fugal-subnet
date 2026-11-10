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


def _closing_server(proto, close_header, drip=False):
    """A server that sends Content-Length and closes the connection after the
    reply (HTTP/1.0, or HTTP/1.1 with "Connection: close"). http.client then
    marks the proxy's socket closed while the reply is still being read."""
    class H(BaseHTTPRequestHandler):
        protocol_version = proto

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(OK) + (100 if drip else 0)))
            if close_header:
                self.send_header("Connection", "close")
            self.end_headers()
            if drip:                                   # promises more than it sends, slowly
                t0 = time.monotonic()
                while time.monotonic() - t0 < 10:
                    try:
                        self.wfile.write(b" ")
                        self.wfile.flush()
                    except OSError:
                        return
                    time.sleep(0.05)
                return
            self.wfile.write(OK)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.mark.parametrize("proto,close_header", [("HTTP/1.0", False), ("HTTP/1.1", True)])
def test_a_reply_that_closes_the_connection_is_returned_whole(short_deadline, proto, close_header):
    # Regression: getresponse() closes the connection object on these replies,
    # and the per-read settimeout raised EBADF, failing every call.
    srv = _closing_server(proto, close_header)
    try:
        assert json.loads(rt._post_with_deadline(_req(srv), time.monotonic() + DEADLINE)) \
            ["choices"][0]["message"]["content"] == "42"
    finally:
        srv.shutdown()


def test_a_closing_reply_that_drips_still_ends_at_the_deadline(short_deadline):
    srv = _closing_server("HTTP/1.1", True, drip=True)
    try:
        t0 = time.monotonic()
        with pytest.raises(rt.UpstreamDeadline):
            rt._post_with_deadline(_req(srv), time.monotonic() + DEADLINE)
        assert time.monotonic() - t0 < DEADLINE + 1.5
    finally:
        srv.shutdown()


def _tls_pair(tmp_path):
    """A self-signed cert for 127.0.0.1 and a client context that trusts it."""
    import datetime
    import ipaddress
    import ssl

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(hours=1))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                           critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    cert_pem, key_pem = tmp_path / "c.pem", tmp_path / "k.pem"
    cert_pem.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_pem.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                          serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(cert_pem, key_pem)
    client_ctx = ssl.create_default_context(cafile=str(cert_pem))
    return server_ctx, client_ctx


@pytest.mark.parametrize("proto,close_header,drip", [
    ("HTTP/1.1", False, False),      # keep-alive, the normal OpenRouter shape
    ("HTTP/1.1", True, False),       # closes after the reply
    ("HTTP/1.0", False, False),
    ("HTTP/1.1", True, True),        # closes, and drips past the deadline
])
def test_https_calls_work_and_still_end_at_the_deadline(short_deadline, monkeypatch, tmp_path,
                                                        proto, close_header, drip):
    # Regression: SSLSocket.dup() raises NotImplementedError, so a dup-based
    # watchdog broke every HTTPS call (every production call).
    import ssl
    server_ctx, client_ctx = _tls_pair(tmp_path)
    monkeypatch.setattr(ssl, "create_default_context", lambda *a, **k: client_ctx)
    srv = _closing_server(proto, close_header, drip=drip)
    srv.socket = server_ctx.wrap_socket(srv.socket, server_side=True)
    try:
        req = Request(f"https://127.0.0.1:{srv.server_address[1]}/v1/chat/completions",
                      data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        if drip:
            t0 = time.monotonic()
            with pytest.raises(rt.UpstreamDeadline):
                rt._post_with_deadline(req, time.monotonic() + DEADLINE)
            assert time.monotonic() - t0 < DEADLINE + 1.5
        else:
            body = rt._post_with_deadline(req, time.monotonic() + DEADLINE)
            assert json.loads(body)["choices"][0]["message"]["content"] == "42"
    finally:
        srv.shutdown()


def test_a_call_whose_deadline_has_passed_never_dials():
    """The deadline is checked before connecting: a call started with no time
    left raises UpstreamDeadline (what the proxy reports as a timeout) and the
    upstream never sees a request. (A planted no-op for that check survived the
    whole suite: the watchdog alone does not give this.)"""
    seen = {}
    hits = []
    srv = _server("normal", seen)
    orig = srv.RequestHandlerClass.do_POST
    srv.RequestHandlerClass.do_POST = lambda self: (hits.append(1), orig(self))
    try:
        with pytest.raises(rt.UpstreamDeadline):
            rt._post_with_deadline(_req(srv), time.monotonic() - 1.0)
        time.sleep(0.2)
        assert hits == []
    finally:
        srv.shutdown()
