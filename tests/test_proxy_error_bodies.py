"""A 200 is not always an answer, and a reset is not always a failed call.

Reported by a miner on mainnet (2026-11): three ways a question was lost or
mis-recorded with nothing wrong with the head.

- The proxy's accept queue held 5 connections while the harness opened 24 at
  the start of an epoch; the ones that did not fit were reset.
- A 200 whose body was not JSON, or was cut short, failed the call on the
  first try, although a 429 would have been retried.
- An upstream error inside a 200 body (`{"error": ...}`, no `choices`) was
  relayed, recorded with zero tokens, and graded as a wrong answer that
  completed (`failed=false`), so it voted in the grade every miner shares.

Every upstream here is a local stub on loopback. Nothing is spent.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from fugal_subnet import success_test_fixtures as success_fixtures
from fugal_subnet.config import HARNESS_CONCURRENCY, HEAD_HIDDEN_DIM
from fugal_subnet.tee import harness as hm
from fugal_subnet.tee import runtime as rt

MODEL = "openai/gpt-5.4-nano"
PRICES = {MODEL: (0.2e-6, 1.25e-6)}
GOOD = json.dumps({"choices": [{"message": {"content": "42"}}],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 2}}).encode()

UNAVAILABLE = {"error": {"code": 502, "message": "provider_unavailable"}}
RATE_LIMITED = {"error": {"code": 429, "message": "rate limited upstream"}}


class _BigQueue(ThreadingHTTPServer):
    request_queue_size = 128          # the stub must not be the thing that drops a burst


def _body(body: bytes, status: int = 200):
    def send(h):
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)
    return send


def _json(obj, status: int = 200):
    return _body(json.dumps(obj).encode(), status)


def _cut_short_chunked(h):
    """Announces a 64-byte chunk, sends 12 bytes of it, closes: a truncated
    reply as it arrives over chunked encoding."""
    h.send_response(200)
    h.send_header("Content-Type", "application/json")
    h.send_header("Transfer-Encoding", "chunked")
    h.end_headers()
    h.wfile.write(b"40\r\n" + b'{"choices": ')
    h.wfile.flush()
    h.close_connection = True


class _Upstream:
    """A scripted OpenAI-compatible upstream: one reply per request, then `default`."""

    def __init__(self, script=(), default=None, delay: float = 0.0):
        self.script = list(script)
        self.default = default or _body(GOOD)
        self.hits = 0
        outer = self
        lock = threading.Lock()

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                try:
                    self.request_json = json.loads(raw)     # for a reply that depends on the question
                except ValueError:
                    self.request_json = {}
                with lock:
                    outer.hits += 1
                    reply = outer.script.pop(0) if outer.script else outer.default
                if delay:
                    time.sleep(delay)
                reply(self)

            def log_message(self, *a):
                pass

        self.server = _BigQueue(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def _wait_listening(port):
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.02)


@pytest.fixture
def proxied(monkeypatch):
    """The REAL proxy in front of a scripted upstream, with the backoff skipped."""
    started = []

    def make(script=(), default=None, delay=0.0, api_key="test", prices=None, wait=True):
        up = _Upstream(script, default, delay)
        monkeypatch.setattr(rt, "_OPENROUTER_BASE", up.base)
        monkeypatch.setattr(rt.time, "sleep", lambda s: None)   # backoff is policy, not a test budget
        proxy = rt.MeteringProxy(port=0, api_key=api_key, prices=dict(prices or PRICES))
        proxy.start()
        started.append((proxy, up))
        if wait:
            _wait_listening(proxy.port)
        return proxy, up

    yield make
    for proxy, up in started:
        proxy.stop()
        up.stop()


def _post(proxy, request_id="e1:q1:" + MODEL):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}).encode()
    req = Request(f"http://127.0.0.1:{proxy.port}/chat/completions", data=body, method="POST",
                  headers={"Content-Type": "application/json", "X-Fugal-Request": request_id})
    try:
        with urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except HTTPError as e:
        return e.code, e.read()


# --- 1. the accept queue, and the harness's one reconnect --------------------


def test_the_accept_queue_holds_an_opening_burst_while_the_proxy_is_busy(monkeypatch):
    """With the accept loop not running at all, every connection of two full
    bursts still completes its handshake at once. On a backlog of 5 the seventh
    waits for a SYN retransmission (Linux drops it) and this times out."""
    assert rt._ProxyServer.request_queue_size == 128 >= 2 * HARNESS_CONCURRENCY
    release = threading.Event()
    real = rt._ProxyServer.serve_forever

    def stalled(self, *a, **k):
        release.wait(30)
        real(self, *a, **k)

    monkeypatch.setattr(rt._ProxyServer, "serve_forever", stalled)
    proxy = rt.MeteringProxy(port=0, api_key="test", prices=dict(PRICES))
    proxy.start()
    assert isinstance(proxy._server, rt._ProxyServer)
    held = []
    try:
        for _ in range(2 * HARNESS_CONCURRENCY):
            held.append(socket.create_connection(("127.0.0.1", proxy.port), timeout=0.5))
    finally:
        for s in held:
            s.close()
        release.set()
        proxy.stop()
    assert len(held) == 2 * HARNESS_CONCURRENCY


def test_a_burst_into_a_stalled_accept_loop_is_all_answered(proxied, monkeypatch, caplog):
    """Two bursts' worth of harness calls arrive before the proxy accepts any.
    Every one is answered and metered once, and none needed the reconnect.
    On a backlog of 5 this is the reported failure: ConnectionResetError
    (errno 104) on the calls that did not fit."""
    release = threading.Event()
    real = rt._ProxyServer.serve_forever

    def stalled(self, *a, **k):
        release.wait(30)
        real(self, *a, **k)

    monkeypatch.setattr(rt._ProxyServer, "serve_forever", stalled)
    monkeypatch.setattr(hm, "HARNESS_CALL_TIMEOUT_S", 30.0)
    proxy, up = proxied(wait=False)
    n = 2 * HARNESS_CONCURRENCY

    def call(i):
        hm._request_ctx.request_id = f"e1:q{i}:{MODEL}"
        return hm._call_model(proxy, MODEL, {"prompt": "hi"})

    with caplog.at_level(logging.WARNING, logger="fugal_subnet.tee.harness"):
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(call, i) for i in range(n)]
            threading.Timer(0.3, release.set).start()
            answers = [f.result() for f in futures]
    assert answers == ["42"] * n
    assert up.hits == n
    assert not [r for r in caplog.records if "before any reply" in r.getMessage()], "a call was reset"
    assert sorted(r.request_id for r in proxy.records) == sorted(f"e1:q{i}:{MODEL}" for i in range(n))


def _reset(conn):
    """Close with a TCP RST, as a kernel does for a connection it will not queue."""
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    conn.close()


class _RawProxy:
    """Stands where the proxy does, on a raw socket: one behaviour per connection."""

    REPLY = b'{"choices": [{"message": {"content": "42"}}]}'

    def __init__(self, script, listen=True):
        self.script = list(script)
        self.connections = 0
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        if listen:
            self.listen()

    def listen(self):
        self.sock.listen(16)
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            how = self.script.pop(0) if self.script else "answer"
            try:
                self._read_request(conn)
                head = b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n"
                if how == "answer":
                    conn.sendall(head + b"Content-Length: %d\r\n\r\n" % len(self.REPLY) + self.REPLY)
                    conn.close()
                elif how == "reset":                      # not a byte of a reply
                    _reset(conn)
                elif how == "close":                      # likewise, but a clean close
                    conn.close()
                elif how == "reset-in-status-line":
                    conn.sendall(b"HTTP/1.0 2")
                    time.sleep(0.3)
                    _reset(conn)
                elif how == "reset-in-body":
                    conn.sendall(head + b"Content-Length: 4096\r\n\r\n" + self.REPLY[:10])
                    time.sleep(0.3)
                    _reset(conn)
            except OSError:
                pass

    @staticmethod
    def _read_request(conn):
        conn.settimeout(5)
        data = b""
        while b"\r\n\r\n" not in data:
            data += conn.recv(65536)
        head, _, rest = data.partition(b"\r\n\r\n")
        length = next(int(line.split(b":")[1]) for line in head.split(b"\r\n")
                      if line.lower().startswith(b"content-length"))
        while len(rest) < length:
            rest += conn.recv(65536)

    def close(self):
        self.sock.close()


@pytest.fixture
def quick_harness(monkeypatch):
    monkeypatch.setattr(hm, "HARNESS_CALL_TIMEOUT_S", 10.0)
    monkeypatch.setattr(hm, "PROXY_RECONNECT_WAIT_S", 0.01)


@pytest.mark.parametrize("first", ["reset", "close"])
def test_a_connection_that_ends_before_any_reply_is_tried_once_more(quick_harness, first):
    stub = _RawProxy([first])
    try:
        assert hm._call_model(stub, MODEL, {"prompt": "hi"}) == "42"
        assert stub.connections == 2
    finally:
        stub.close()


def test_a_refused_connection_is_tried_once_more(monkeypatch, quick_harness):
    """Nothing is listening for the first connect; the stub starts listening
    during the harness's pause, so the second connect is answered."""
    stub = _RawProxy([], listen=False)
    pauses = []
    monkeypatch.setattr(hm.time, "sleep", lambda s: (pauses.append(s), stub.listen()))
    try:
        assert hm._call_model(stub, MODEL, {"prompt": "hi"}) == "42"
        assert pauses == [hm.PROXY_RECONNECT_WAIT_S] and stub.connections == 1
    finally:
        stub.close()


def test_only_one_reconnect_is_ever_made(quick_harness):
    stub = _RawProxy(["reset", "reset", "answer"])
    try:
        assert hm._call_model(stub, MODEL, {"prompt": "hi"}) is None
        assert stub.connections == 2
    finally:
        stub.close()


@pytest.mark.parametrize("how", ["reset-in-status-line", "reset-in-body"])
def test_a_reset_after_the_reply_has_started_is_a_failed_call_not_a_second_one(quick_harness, how):
    """The proxy answers only after the upstream call is made and metered, so
    once any byte of a reply has arrived a second call could bill the question
    twice. It fails instead."""
    stub = _RawProxy([how])
    try:
        assert hm._call_model(stub, MODEL, {"prompt": "hi"}) is None
        time.sleep(0.2)
        assert stub.connections == 1
    finally:
        stub.close()


def test_a_timeout_or_an_error_status_from_the_proxy_is_not_a_reconnect():
    assert not hm._proxy_never_answered(TimeoutError("timed out"))
    assert not hm._proxy_never_answered(ConnectionResetError("mid-reply"))
    assert not hm._proxy_never_answered(HTTPError("http://x", 502, "Bad Gateway", None, io.BytesIO(b"")))
    from urllib.error import URLError
    assert not hm._proxy_never_answered(URLError(TimeoutError("timed out")))
    assert hm._proxy_never_answered(URLError(ConnectionRefusedError()))
    assert hm._proxy_never_answered(hm.ProxyNeverAnswered("closed before any reply"))


# --- 2. a 200 whose body is not a reply --------------------------------------


@pytest.mark.parametrize("bad", [
    _body(b"<html><body>502 Bad Gateway</body></html>"),
    _body(GOOD[: len(GOOD) // 2]),            # valid JSON, cut short
    _body(b""),
    _body(b"[1, 2, 3]"),                      # JSON, but not a reply
    _cut_short_chunked,
], ids=["html", "truncated", "empty", "not-an-object", "chunked-cut-short"])
def test_a_200_that_is_not_json_is_retried_then_answered_and_metered_once(proxied, bad):
    proxy, up = proxied([bad])
    status, body = _post(proxy)
    assert status == 200 and body == GOOD
    assert up.hits == 2
    (rec,) = proxy.records
    assert (rec.prompt_tokens, rec.completion_tokens) == (5, 2)
    assert rec.response_hash == hashlib.sha256(GOOD).hexdigest()


def test_a_200_that_is_never_json_fails_after_the_attempt_cap(proxied):
    proxy, up = proxied(default=_body(b"upstream connect error"))
    status, _ = _post(proxy)
    assert status == 502 and up.hits == rt.PROXY_MAX_ATTEMPTS and proxy.records == []


# --- 3. an error inside a 200 body -------------------------------------------


@pytest.mark.parametrize("transient", [
    UNAVAILABLE, RATE_LIMITED,
    {"error": {"code": "429", "message": "slow down"}},
    {"error": "upstream request failed"},
    {"error": {"code": 402, "metadata": {"reason": rt.OPENROUTER_402_TRANSIENT}}},
    {"id": "gen-1", "choices": []},
    {},
], ids=["502", "429", "429-as-text", "error-string", "in-flight-402", "empty-choices", "empty-object"])
def test_a_transient_error_in_a_200_body_is_retried_then_answered(proxied, transient):
    proxy, up = proxied([_json(transient)])
    status, body = _post(proxy)
    assert status == 200 and body == GOOD and up.hits == 2
    assert len(proxy.records) == 1 and proxy.records[0].completion_tokens == 2


def test_a_persistent_error_in_a_200_body_fails_and_is_never_relayed_as_a_reply(proxied, caplog):
    """Before: relayed as a 200, recorded with zero tokens, graded wrong."""
    proxy, up = proxied(default=_json(UNAVAILABLE))
    with caplog.at_level(logging.WARNING, logger="fugal_subnet.tee.runtime"):
        status, body = _post(proxy)
    assert status == 502 and b"choices" not in body
    assert up.hits == rt.PROXY_MAX_ATTEMPTS and proxy.records == []
    assert any("no retry is left" in r.getMessage() and "502" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("refused", [
    {"error": {"code": 403, "message": "Your input was flagged"}},
    {"error": {"code": "403", "message": "forbidden"}},
    {"error": {"code": "content_policy_violation", "type": "invalid_request_error"}},
    {"error": {"code": 400, "type": "content-filter"}},
    {"error": {"type": "Moderation"}},
    {"error": {"type": "forbidden"}},
    {"error": {"code": 401, "message": "no auth"}},
    {"error": {"code": 402, "message": "insufficient credits"}},
    {"error": {"code": 402, "metadata": {"reason": rt.OPENROUTER_402_TOO_LARGE}}},
], ids=["403", "403-as-text", "content-policy-code", "content-filter-type", "moderation-type",
        "forbidden-type", "401", "402", "402-too-large"])
def test_an_error_no_retry_can_change_fails_at_once(proxied, caplog, refused):
    proxy, up = proxied(default=_json(refused))
    with caplog.at_level(logging.ERROR, logger="fugal_subnet.tee.runtime"):
        status, _ = _post(proxy)
    assert status == 502 and up.hits == 1 and proxy.records == []
    assert any("not retried" in r.getMessage() for r in caplog.records)


def test_an_operator_code_in_a_200_body_is_named_as_one(proxied, caplog):
    proxy, _ = proxied(default=_json({"error": {"code": 402, "message": "insufficient credits"}}))
    with caplog.at_level(logging.ERROR, logger="fugal_subnet.tee.runtime"):
        _post(proxy)
    assert any("OPERATOR ACTION" in r.getMessage() and "402" in r.getMessage() for r in caplog.records)


def test_what_the_upstream_wrote_in_an_error_never_reaches_the_log(proxied, caplog):
    """An error message is a third party's free text, and some providers quote
    the key they were called with. Only the numeric code and our own words are
    logged."""
    key = "sk-or-v1-0123456789abcdef0123456789abcdef"
    echo = {"error": {"code": 401, "type": f"bad key {key}", "message": f"Incorrect API key provided: {key}"}}
    echo_retried = {"error": {"code": 502, "type": key, "message": f"upstream said {key}",
                              "metadata": {"raw": key}}}
    with caplog.at_level(logging.DEBUG):
        proxy, up = proxied(default=_json(echo), api_key=key)
        assert _post(proxy)[0] == 502 and up.hits == 1
        proxy, up = proxied(default=_json(echo_retried), api_key=key)
        assert _post(proxy)[0] == 502 and up.hits == rt.PROXY_MAX_ATTEMPTS
    logged = caplog.text + "".join(str(r.args) for r in caplog.records)
    assert caplog.records
    assert key not in logged and key[-8:] not in logged and "Incorrect API key" not in logged


# --- metering: one record per upstream completion ----------------------------


def test_an_unanswered_completion_that_reports_tokens_is_metered_and_so_is_the_retry(proxied):
    """A 200 with no choices that still reports usage was billed by the
    provider. It is recorded under the same request id as the reply that
    follows, so the question carries what it cost; a body that reports no
    tokens leaves no record."""
    billed = {"error": {"code": 502, "message": "provider died mid-reply"},
              "usage": {"prompt_tokens": 5, "completion_tokens": 100, "cost": 0.001}}
    proxy, up = proxied([_json(UNAVAILABLE), _json(billed)])
    status, body = _post(proxy, request_id="e7:q3:" + MODEL)
    assert status == 200 and body == GOOD and up.hits == 3
    assert [(r.prompt_tokens, r.completion_tokens) for r in proxy.records] == [(5, 100), (5, 2)]
    assert {r.request_id for r in proxy.records} == {"e7:q3:" + MODEL}
    assert proxy.records[0].response_hash == hashlib.sha256(json.dumps(billed).encode()).hexdigest()
    assert proxy.total_cost == pytest.approx(10 * 0.2e-6 + 102 * 1.25e-6)


def test_a_call_that_never_answers_still_meters_what_was_billed(proxied):
    billed = {"error": {"code": 502}, "usage": {"prompt_tokens": 5, "completion_tokens": 7}}
    proxy, up = proxied(default=_json(billed))
    assert _post(proxy)[0] == 502
    assert up.hits == rt.PROXY_MAX_ATTEMPTS == len(proxy.records)


# --- the deadline and the retry budget still bound the whole call -------------


def test_retries_of_a_bad_200_end_at_the_wall_clock_deadline(monkeypatch):
    """Each reply takes 0.4 s and is an error; attempts are not the limit here.
    The third attempt is cut off by the 1 s deadline, like any slow call."""
    up = _Upstream(default=_json(UNAVAILABLE), delay=0.4)
    monkeypatch.setattr(rt, "PROXY_CALL_DEADLINE_S", 1.0)
    monkeypatch.setattr(rt, "PROXY_MAX_ATTEMPTS", 50)
    monkeypatch.setattr(rt, "_retry_wait", lambda attempt, retry_after: 0.0)
    req = Request(f"{up.base}/chat/completions", data=b"{}", method="POST",
                  headers={"Content-Type": "application/json"})
    try:
        t0 = time.monotonic()
        with pytest.raises(rt.UpstreamDeadline):
            rt._fetch_upstream(req, "m")
        took = time.monotonic() - t0
        assert 0.95 <= took < 1.5, took
        assert up.hits == 3
    finally:
        up.stop()


def test_no_retry_of_a_bad_200_starts_after_the_retry_budget(proxied, monkeypatch):
    proxy, up = proxied(default=_json(UNAVAILABLE))
    clock = iter([0.0] + [rt.PROXY_RETRY_BUDGET_S + 1.0] * 50)
    monkeypatch.setattr(rt, "time", SimpleNamespace(
        time=lambda: next(clock), sleep=lambda s: None, monotonic=time.monotonic))
    assert _post(proxy)[0] == 502 and up.hits == 1


def test_a_bad_200_waits_between_tries_like_a_429(proxied, monkeypatch):
    waits = []
    proxy, up = proxied(default=_json(UNAVAILABLE))
    monkeypatch.setattr(rt.time, "sleep", waits.append)
    _post(proxy)
    assert len(waits) == rt.PROXY_MAX_ATTEMPTS - 1
    assert all(0.0 < w <= rt.PROXY_MAX_WAIT_S for w in waits) and waits[1] > waits[0]


# --- the harness: no choices is never an answer ------------------------------


def test_reply_has_choices_is_the_line_between_an_answer_and_an_error():
    assert rt.reply_has_choices({"choices": [{"message": {"content": "42"}}]})
    assert rt.reply_has_choices({"choices": [{"message": {"content": None}}]})   # completed, no text
    assert rt.reply_has_choices({"choices": [{}]})
    for not_a_reply in ({"error": {"code": 429}}, {"choices": []}, {"choices": None},
                        {"choices": "x"}, {"choices": ["x"]}, {}, [], None, "ok", 7):
        assert not rt.reply_has_choices(not_a_reply)


@pytest.mark.parametrize("body", [UNAVAILABLE, {"choices": []}, {"error": "upstream request failed"}])
def test_the_harness_never_takes_a_reply_without_choices_as_an_answer(quick_harness, body):
    """Whatever stands in the proxy's place: a 200 with no choices is a failed
    call (None), not the empty answer ("")."""
    up = _Upstream(default=_json(body))
    try:
        assert hm._call_model(SimpleNamespace(port=up.server.server_port), MODEL, {"prompt": "hi"}) is None
    finally:
        up.stop()


def test_through_the_real_proxy_an_error_body_ends_failed_and_a_recovered_one_answered(proxied, quick_harness):
    proxy, up = proxied([_json(RATE_LIMITED)])            # then good replies
    hm._request_ctx.request_id = "e1:qa:" + MODEL
    try:
        assert hm._call_model(proxy, MODEL, {"prompt": "hi"}) == "42"
        assert up.hits == 2 and len(proxy.records) == 1
        up.default = _json(UNAVAILABLE)                    # from here on, only the error
        hm._request_ctx.request_id = "e1:qb:" + MODEL
        assert hm._call_model(proxy, MODEL, {"prompt": "hi"}) is None
        assert up.hits == 2 + rt.PROXY_MAX_ATTEMPTS and len(proxy.records) == 1
    finally:
        hm._request_ctx.request_id = ""


def test_an_epoch_records_an_error_body_as_failed_and_grades_the_rest(proxied, monkeypatch, quick_harness):
    """The reported case end to end: real harness, real proxy, a stub upstream
    that answers every question but one, for which it only ever returns
    {"error": ...} in a 200. That question is `failed`, carries no reply and no
    cost; it used to be `failed=False`, graded wrong."""
    models = ["m/a", "m/b"]
    pool = [{
        "question_id": f"q{i}", "prompt": f"What is {i}+{i}?", "gold": str(2 * i),
        "grader_id": "numeric_final", "benchmark": "gsm8k", "metadata": {},
    } for i in range(1, 9)]
    by_prompt = {q["prompt"]: q for q in pool}
    hits: list[str] = []

    def reply(h):
        q = by_prompt[h.request_json["messages"][0]["content"]]
        hits.append(q["question_id"])
        if q["question_id"] == "q3":
            _json(UNAVAILABLE)(h)
        else:
            _json({"choices": [{"message": {"content": q["gold"]}}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 3}})(h)

    proxy, _ = proxied(default=reply, prices={m: (1e-6, 2e-6) for m in models})

    rng = np.random.RandomState(3)
    buf = io.BytesIO()
    np.savez(buf, **success_fixtures.arrays(
        (rng.randn(len(models), HEAD_HIDDEN_DIM) * 0.02).astype(np.float32),
        np.zeros(len(models), np.float32), models))
    hidden = rng.randn(len(pool), HEAD_HIDDEN_DIM).astype(np.float32)
    monkeypatch.setattr(hm, "benchmark_costs", success_fixtures.costs)
    monkeypatch.setattr(hm.epoch_plan, "scored_slice", lambda version, nonce, pool_, n: pool_[:n])
    proof = hm.run_benchmark(
        nonce="ab" * 32, head_bytes=buf.getvalue(), benchmark_pool=pool, proxy=proxy,
        hidden_states=hidden, slice_size=len(pool), epoch_id="e1", explore_models=[],
        explore_size=0,
    )
    by_id = {r.question_id: r for r in proof.results}
    assert len(by_id) == len(pool)
    bad = by_id.pop("q3")
    assert bad.failed is True and bad.correct is False
    assert bad.response_hash == hashlib.sha256(b"").hexdigest()
    assert (bad.cost_usd, bad.prompt_tokens, bad.completion_tokens) == (0.0, 0, 0)
    assert hits.count("q3") == rt.PROXY_MAX_ATTEMPTS
    assert all(r.correct and not r.failed and r.completion_tokens == 3 for r in by_id.values())
    assert all(hits.count(qid) == 1 for qid in by_id)
    assert proof.total_cost_usd == pytest.approx(sum(r.cost_usd for r in by_id.values()))
