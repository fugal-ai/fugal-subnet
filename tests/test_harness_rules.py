"""The TEE harness's own rules, each held by a test.

Planted bugs (2026-11-08) left 9 of 18 rules in tee/harness.py untested by
the 19 test files that touch it: model calls going out in slice order instead
of the nonce's interleaved order (the cheap version of blocking exploration,
diligence 2026-10 finding 2), a failed call that could still grade correct,
latency not recorded, a v2 proof labelled v1, a v2 epoch drawing the v1
slice or pricing on v1 costs, max_tokens not sent, a failed call's cost not
a float, and the response hash not covering the reply.
"""
import hashlib
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import numpy as np

from fugal_subnet.config import HARNESS_MAX_TOKENS
from fugal_subnet.routing_protocol import identity, identity_v2
from fugal_subnet.tee import harness
from tests.test_tee_e2e import EXPLORE, MODELS, SLICE, _head, _pool, _stub_call, _StubProxy

NONCE = "ab" * 32


def _run(monkeypatch, call=_stub_call, version=1, **kw):
    pool = _pool()
    hidden = np.random.RandomState(0).randn(len(pool), 1024).astype(np.float32)
    monkeypatch.setattr(harness, "_call_model", call)
    if version == 1:
        from fugal_subnet import success_test_fixtures as fx
        monkeypatch.setattr(harness, "benchmark_costs", fx.costs)
    proxy = _StubProxy(port=0)
    proxy.start()
    return harness.run_benchmark(
        nonce=NONCE, head_bytes=_head(MODELS), benchmark_pool=pool, proxy=proxy,
        hidden_states=hidden, slice_size=SLICE, epoch_id="e00000001", source_hash="img",
        explore_models=MODELS, explore_size=EXPLORE, protocol_version=version, **kw)


def test_calls_go_out_in_the_nonces_order_with_exploration_interleaved(monkeypatch):
    seen = {}
    real = harness._call_all

    def spy(proxy, scheduled, epoch_id, protocol_version=1):
        seen["order"] = [(q["question_id"], explore) for q, _m, explore in scheduled]
        return real(proxy, scheduled, epoch_id, protocol_version)

    monkeypatch.setattr(harness, "_call_all", spy)
    proof = _run(monkeypatch)
    nonce = bytes.fromhex(NONCE)
    key = [hmac.new(nonce, f"order:{qid}".encode(), hashlib.sha256).digest() for qid, _ in seen["order"]]
    assert key == sorted(key)
    flags = [explore for _q, explore in seen["order"]]
    assert sum(flags) == EXPLORE and flags != sorted(flags), "exploration must not all trail the scored calls"
    # The PROOF still lays results out in slice order, exploration after.
    assert [r.is_exploration for r in proof.results] == sorted(r.is_exploration for r in proof.results)


def test_a_failed_call_never_grades_correct_even_if_the_grader_would_accept_nothing(monkeypatch):
    monkeypatch.setattr(harness, "grade", lambda task, text, allow_exec=False: 1)
    proof = _run(monkeypatch, call=lambda proxy, model_id, question: None)
    assert all(r.failed and not r.correct for r in proof.results)


def test_a_failed_calls_cost_and_tokens_are_canonical_numbers(monkeypatch):
    proof = _run(monkeypatch, call=lambda proxy, model_id, question: None)
    for r in proof.results:
        assert type(r.cost_usd) is float and r.cost_usd == 0.0
        assert type(r.prompt_tokens) is int and type(r.completion_tokens) is int


def test_the_response_hash_covers_the_reply_and_latency_is_measured(monkeypatch):
    def slow(proxy, model_id, question):
        time.sleep(0.03)
        return _stub_call(proxy, model_id, question)

    proof = _run(monkeypatch, call=slow)
    gold = {q["question_id"]: q["gold"] for q in _pool()}
    for r in proof.results:
        text = gold[r.question_id] if r.correct else "0"
        assert r.response_hash == hashlib.sha256(text.encode()).hexdigest()
        assert r.latency_ms >= 25


def test_a_v2_epoch_uses_the_v2_slice_costs_and_label(monkeypatch):
    calls = {}
    head = SimpleNamespace(models=MODELS, W=np.zeros((3, 1024), np.float32),
                           b=np.zeros(3, np.float32), success={})
    monkeypatch.setattr(harness, "load_head_from_npz", lambda data: head)

    def costs_for(h, version):
        calls["costs"] = version
        return np.zeros(3)

    def v1_costs(h):
        raise AssertionError("v1 benchmark_costs used in a v2 epoch")

    real_slice = harness.epoch_plan.scored_slice

    def slice_spy(version, nonce, pool, size):
        calls["slice"] = version
        return real_slice(1, nonce, pool, size)          # the synthetic pool has no v2 categories

    monkeypatch.setattr(harness, "benchmark_costs_for", costs_for)
    monkeypatch.setattr(harness, "benchmark_costs", v1_costs)
    monkeypatch.setattr(harness.epoch_plan, "scored_slice", slice_spy)
    monkeypatch.setattr(harness.epoch_plan, "exploration_plan", lambda *a: {})
    monkeypatch.setattr(harness, "_route_question", lambda head, h, costs=None: 0)
    monkeypatch.setattr(harness, "reasoning_for", lambda version, model: None)
    proof = _run(monkeypatch, version=2)
    assert calls == {"costs": 2, "slice": 2}
    assert proof.routing_protocol == identity_v2() != identity()


def test_a_v1_epoch_is_labelled_v1(monkeypatch):
    assert _run(monkeypatch).routing_protocol == identity()


def test_every_call_carries_the_answer_budget_and_v2_reasoning():
    bodies = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            out = json.dumps({"choices": [{"message": {"content": "7"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    proxy = SimpleNamespace(port=srv.server_port)
    try:
        assert harness._call_model(proxy, "m/x", {"prompt": "1+6?"}) == "7"
        assert harness._call_model(proxy, "m/x", {"prompt": "1+6?"}, reasoning={"effort": "low"}) == "7"
    finally:
        srv.shutdown()
    assert [b["max_tokens"] for b in bodies] == [HARNESS_MAX_TOKENS] * 2
    assert "reasoning" not in bodies[0] and bodies[1]["reasoning"] == {"effort": "low"}
    assert bodies[0]["messages"] == [{"role": "user", "content": "1+6?"}]


