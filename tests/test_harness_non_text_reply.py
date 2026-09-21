"""A reply that is not text is a wrong answer, never an aborted epoch.

Found 2026-09-07 on the first live epoch of the first real TDX miners: one
chat completion carried `"content": null`, `_call_model` handed `None` up,
`run_benchmark` called `.encode()` on it, and the whole epoch failed with no
proof — for one reply out of three hundred. Every stub in the suite returns a
string, so the path had never run. These tests make the failure path run.
"""
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from fugal_subnet import success_test_fixtures as success_fixtures
from fugal_subnet.tee import harness as harness_mod
from fugal_subnet.tee.harness import _text_of


def test_text_of_handles_every_shape_a_provider_has_sent():
    assert _text_of({"choices": [{"message": {"content": "42"}}]}) == "42"
    assert _text_of({"choices": [{"message": {"content": None}}]}) == ""
    assert _text_of({"choices": [{"message": {}}]}) == ""
    assert _text_of({"choices": [{}]}) == ""
    assert _text_of({"choices": []}) == ""
    assert _text_of({"error": {"message": "upstream"}}) == ""
    assert _text_of({"choices": [{"message": {"content": [
        {"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"},
    ]}}]}) == "ab"
    assert _text_of({"choices": [{"message": {"content": 7}}]}) == ""
    assert _text_of(None) == ""


def test_call_model_returns_empty_string_for_null_content():
    """Through the real HTTP path: a proxy that answers with content: null."""

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            raw = json.dumps({"choices": [{"message": {"role": "assistant", "content": None}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        class P:
            port = server.server_port
        out = harness_mod._call_model(P(), "some/model", {"prompt": "q"})
        assert out == "" and isinstance(out, str)
    finally:
        server.shutdown()


def test_run_benchmark_survives_a_none_reply(monkeypatch):
    """One question's model call yields None; the proof still has every
    question, that one graded wrong with the hash of the empty string."""
    from fugal_subnet.config import HEAD_HIDDEN_DIM

    models = ["m/a", "m/b"]
    pool = [{
        "question_id": f"q{i}", "prompt": f"What is {i}+{i}?", "gold": str(2 * i),
        "grader_id": "numeric_final", "benchmark": "gsm8k", "metadata": {},
    } for i in range(12)]
    rng = np.random.RandomState(3)
    W = (rng.randn(len(models), HEAD_HIDDEN_DIM) * 0.02).astype(np.float32)
    import io
    buf = io.BytesIO()
    np.savez(buf, **success_fixtures.arrays(W, np.zeros(len(models), np.float32), models))
    head_bytes = buf.getvalue()
    hidden = rng.randn(len(pool), HEAD_HIDDEN_DIM).astype(np.float32)

    seen = {}

    def stub(proxy, model_id, question):
        seen[question["question_id"]] = True
        return None if question["question_id"] == "q0" else question["gold"]

    from tests.test_tee_e2e import _StubProxy

    proxy = _StubProxy(port=0)
    proxy.start()
    monkeypatch.setattr(harness_mod, "benchmark_costs", success_fixtures.costs)
    monkeypatch.setattr(harness_mod, "_call_model", stub)
    monkeypatch.setattr(harness_mod, "select_slice", lambda nonce, pool_, n: pool_[:n])

    try:
        proof = harness_mod.run_benchmark(
            nonce="ab" * 32, head_bytes=head_bytes, benchmark_pool=pool, proxy=proxy,
            hidden_states=hidden, slice_size=6, epoch_id="e1", explore_models=[],
            explore_size=0,
        )
    finally:
        proxy.stop()
    by_id = {r.question_id: r for r in proof.results}
    assert len(by_id) == 6, "the epoch must not lose questions to a bad reply"
    assert by_id["q0"].correct is False
    assert by_id["q0"].response_hash == hashlib.sha256(b"").hexdigest()
    assert all(by_id[f"q{i}"].correct for i in range(1, 6))
