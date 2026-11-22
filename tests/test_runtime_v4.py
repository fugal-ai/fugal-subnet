"""Protocol v4 in the TD's runtime: the endpoints read and what a reply billed.

Every upstream here is a local HTTP server on loopback: no network, no key,
no spend.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from fugal_subnet.tee import endpoint_reads as er
from fugal_subnet.tee import runtime as rt

MODEL = "lab-a/model-one"


def _server(handler_fn):
    """A loopback server; `handler_fn(handler)` answers each GET or POST."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.command, self.path, dict(self.headers)))
            handler_fn(self)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            seen.append((self.command, self.path, dict(self.headers), body))
            handler_fn(self)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


@pytest.fixture
def upstream(monkeypatch):
    servers = []

    def make(handler_fn):
        srv, seen = _server(handler_fn)
        monkeypatch.setattr(rt, "_OPENROUTER_BASE", f"http://127.0.0.1:{srv.server_port}/api/v1")
        servers.append(srv)
        return seen

    yield make
    for s in servers:
        s.shutdown()


def _reply(h, body: bytes, status: int = 200, headers=()):
    h.send_response(status)
    for k, v in headers:
        h.send_header(k, v)
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)


PAGE = json.dumps({"data": {"id": MODEL, "endpoints": [
    {"tag": "host-a/fp8", "pricing": {"prompt": "0.0000001", "completion": "0.0000004"},
     "max_completion_tokens": 32768, "quantization": "fp8", "status": 0}]}}).encode()


def test_a_read_is_a_keyless_get_under_the_one_base_url(upstream, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-not-a-real-key")
    seen = upstream(lambda h: _reply(h, PAGE))
    assert rt.fetch_endpoints_page(MODEL) == PAGE
    (method, path, headers), = seen
    assert method == "GET" and path == f"/api/v1/models/{MODEL}/endpoints"
    assert not any(k.lower() == "authorization" for k in headers)
    read = er.parse_listing(MODEL, rt.fetch_endpoints_page(MODEL), ["host-a/fp8"])
    assert read.hosts[0].price_out == 400_000


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_a_redirect_is_never_followed(upstream, status):
    seen = upstream(lambda h: _reply(h, b"", status,
                                     [("Location", "http://127.0.0.1:1/elsewhere")]))
    with pytest.raises(HTTPError):
        rt.fetch_endpoints_page(MODEL)
    assert len(seen) == 1


@pytest.mark.parametrize("status", [404, 429, 500])
def test_an_error_status_is_a_failed_read(upstream, status):
    upstream(lambda h: _reply(h, b'{"error": {}}', status))
    with pytest.raises(HTTPError):
        rt.fetch_endpoints_page(MODEL)


def test_an_oversized_reply_is_refused_by_its_declared_length(upstream):
    upstream(lambda h: _reply(h, b"x" * (er.MAX_LISTING_BYTES + 1)))
    with pytest.raises(rt.ReplyTooLarge):
        rt.fetch_endpoints_page(MODEL)


def test_an_oversized_reply_without_a_length_is_refused_while_read(upstream):
    def chunked(h):
        h.send_response(200)
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        piece = b"x" * 65536
        for _ in range(er.MAX_LISTING_BYTES // len(piece) + 2):
            h.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
        h.wfile.write(b"0\r\n\r\n")

    upstream(chunked)
    with pytest.raises(rt.ReplyTooLarge):
        rt.fetch_endpoints_page(MODEL)


def test_a_read_that_does_not_answer_ends_at_its_deadline(upstream):
    gate = threading.Event()
    upstream(lambda h: gate.wait(5))
    started = time.monotonic()
    with pytest.raises(rt.UpstreamDeadline):
        rt.fetch_endpoints_page(MODEL, deadline_s=0.5)
    gate.set()
    assert time.monotonic() - started < 3


@pytest.mark.parametrize("model", ["lab-a/../../keys", "lab-a/m?x", "lab-a/m#f", "LAB/m",
                                   "lab-a/m/endpoints", "", None])
def test_a_model_id_outside_the_listing_shape_is_never_put_in_a_path(upstream, model):
    seen = upstream(lambda h: _reply(h, PAGE))
    with pytest.raises(ValueError):
        rt.fetch_endpoints_page(model)
    assert seen == []


# --- billed cost and the serving host --------------------------------------------


def _body(cost_text: str | None, provider="DeepInfra") -> bytes:
    usage = '{"prompt_tokens":5,"completion_tokens":2' + (
        f',"cost":{cost_text}' if cost_text is not None else "") + "}"
    prov = f',"provider":{json.dumps(provider)}' if provider is not None else ""
    return ('{"choices":[{"message":{"content":"42"}}],"usage":' + usage + prov + "}").encode()


@pytest.mark.parametrize("text, nano", [
    ("0.00042", 420_000), ("0", 0), ("0.0", 0), ("1e-7", 100), ("1.5E-05", 15_000),
    ("0.0000000015", 2),           # 1.5 nanodollars rounds half up
    ("0.0000000014999", 1),
    ("0.1234567890123456789", 123_456_789),
    ("12.5", 12_500_000_000),
])
def test_billed_cost_is_read_from_the_reply_text_exactly(text, nano):
    assert rt.billed_nanodollars(_body(text)) == nano


@pytest.mark.parametrize("text", [None, '"0.00042"', "-0.00042", "true", "null", "1e99",
                                  "1000000", "[0.1]", '{"x":1}', "1e999"])
def test_billed_cost_that_is_absent_or_not_a_bounded_number_is_none(text):
    assert rt.billed_nanodollars(_body(text)) is None


def test_billed_cost_of_a_body_that_is_not_json_is_none():
    assert rt.billed_nanodollars(b"<html>") is None
    assert rt.billed_nanodollars(b"[" * 100000) is None


@pytest.mark.parametrize("provider, name", [
    ("DeepInfra", "DeepInfra"), ("Google AI Studio", "Google AI Studio"), (None, ""), (7, ""),
    ("x" * 65, ""), ("Deep\nInfra", ""), ("Dëep", ""),
])
def test_the_serving_host_is_the_name_the_reply_gives_when_it_is_one_a_proof_can_carry(
        provider, name):
    assert rt.served_by_of(json.loads(_body("0.1", provider=provider))) == name


def _wait_listening(port):
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.02)


def test_the_proxy_records_billed_cost_and_serving_host_per_call(upstream):
    body = _body("0.000123456", provider="Fireworks")
    upstream(lambda h: _reply(h, body))
    proxy = rt.MeteringProxy(port=0, api_key="test", prices={"e@high": (1e-7, 4e-7)})
    proxy.start()
    try:
        _wait_listening(proxy.port)
        req = Request(f"http://127.0.0.1:{proxy.port}/chat/completions",
                      data=json.dumps({"model": "lab-a/model-one", "messages": []}).encode(),
                      method="POST", headers={"Content-Type": "application/json",
                                              "X-Fugal-Request": "rid", "X-Fugal-Entry": "e@high"})
        with urlopen(req, timeout=30) as r:
            assert r.read() == body
    finally:
        proxy.stop()
    (rec,) = proxy.records
    assert (rec.billed_nano, rec.served_by) == (123_456, "Fireworks")
