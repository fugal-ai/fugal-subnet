"""The miner makes model calls concurrently, and nothing about the proof
depends on the order they complete in.

Measured 2026-09-07 on the first live epoch on real TDX: 315 serial calls ran
past 49 minutes against a 36-minute collection window, with ~10 s replies and
180 s timeouts. Concurrency is the fix; these tests pin what it must not break:
results stay in slice order, every question is billed exactly its own calls
(attribution by request id, not list position), a stalled call costs one
worker rather than the epoch, and the attested cost totals are identical
whatever order the records landed in.
"""
import hashlib
import io
import threading
import time

import numpy as np
import pytest

from fugal_subnet.config import HEAD_HIDDEN_DIM
from fugal_subnet.tee import harness as harness_mod
from fugal_subnet.tee.runtime import APICallRecord
from tests.test_tee_e2e import _StubProxy

MODELS = ["m/a", "m/b", "m/c"]


def _pool(n):
    return [{
        "question_id": f"q{i:03d}", "prompt": f"What is {i}+{i}?", "gold": str(2 * i),
        "grader_id": "numeric_final", "benchmark": "gsm8k", "metadata": {},
    } for i in range(n)]


def _head(seed=1):
    rng = np.random.RandomState(seed)
    buf = io.BytesIO()
    np.savez(buf, W=(rng.randn(len(MODELS), HEAD_HIDDEN_DIM) * 0.02).astype(np.float32),
             b=np.zeros(len(MODELS), np.float32), models=np.array(MODELS, dtype="U100"))
    return buf.getvalue()


def _run(monkeypatch, stub, n=40, explore=False):
    pool = _pool(n)
    hidden = np.random.RandomState(7).randn(n, HEAD_HIDDEN_DIM).astype(np.float32)
    proxy = _StubProxy(port=0)
    proxy.start()
    proxy.prices = {m: (1e-6 * (i + 1), 2e-6 * (i + 1)) for i, m in enumerate(MODELS)}
    monkeypatch.setattr(harness_mod, "_call_model", stub)
    monkeypatch.setattr(harness_mod, "select_slice", lambda nonce, p, k: p[:k])
    try:
        t = time.time()
        proof = harness_mod.run_benchmark(
            nonce="ab" * 32, head_bytes=_head(), benchmark_pool=pool, proxy=proxy,
            hidden_states=hidden, slice_size=n // 2, epoch_id="e1",
            explore_models=MODELS if explore else [], explore_size=4 if explore else 0,
        )
        return proof, time.time() - t, proxy
    finally:
        proxy.stop()


def _recording_stub(delay=0.0, tokens_for=None):
    """Appends one record per call tagged with the current request id, the way
    the real proxy does, with a per-question token count so misattribution is
    visible in the numbers."""
    calls = []

    def stub(proxy, model_id, question):
        rid = harness_mod.current_request_id()
        assert rid, "the harness must set a request id before calling the model"
        if delay:
            time.sleep(delay)
        n = int(question["question_id"][1:])
        p_tok = 100 + n
        c_tok = 10 + n
        text = question["gold"]
        with proxy._lock:
            proxy.records.append(APICallRecord(
                model_id=model_id, prompt_tokens=p_tok, completion_tokens=c_tok,
                cost_usd=proxy.price_call(model_id, p_tok, c_tok), timestamp=time.time(),
                response_hash=hashlib.sha256(text.encode()).hexdigest(), request_id=rid,
            ))
        calls.append((threading.get_ident(), question["question_id"]))
        return text
    stub.calls = calls
    return stub


def test_results_stay_in_slice_order_and_each_question_bills_only_itself(monkeypatch):
    stub = _recording_stub(delay=0.05)
    proof, _, proxy = _run(monkeypatch, stub, n=40, explore=True)
    scored = [r for r in proof.results if not r.is_exploration]
    assert [r.question_id for r in scored] == [f"q{i:03d}" for i in range(20)], "slice order"
    for r in proof.results:
        n = int(r.question_id[1:])
        assert (r.prompt_tokens, r.completion_tokens) == (100 + n, 10 + n), \
            f"{r.question_id} billed someone else's tokens"
        assert r.correct
    assert len({tid for tid, _ in stub.calls}) > 1, "calls ran on more than one thread"
    # Every record is accounted for exactly once.
    assert abs(sum(r.cost_usd for r in proof.results) - proxy.total_cost) < 1e-12
    assert abs(sum(proof.per_model_costs.values()) - proof.total_cost_usd) < 1e-12


def test_concurrency_actually_overlaps_calls(monkeypatch):
    stub = _recording_stub(delay=0.25)
    _, elapsed, _ = _run(monkeypatch, stub, n=32)
    # 16 scored calls × 0.25 s = 4 s serial; 8 workers → ~0.5 s.
    assert elapsed < 2.0, f"took {elapsed:.2f}s — calls are not overlapping"


def test_a_stalled_call_costs_one_worker_not_the_epoch(monkeypatch):
    inner = _recording_stub()

    def stub(proxy, model_id, question):
        if question["question_id"] == "q003":
            time.sleep(1.0)
            return ""            # a timeout, as _call_model reports it
        return inner(proxy, model_id, question)

    proof, elapsed, _ = _run(monkeypatch, stub, n=32)
    by_id = {r.question_id: r for r in proof.results}
    assert len(by_id) == 16 and by_id["q003"].correct is False
    assert by_id["q003"].cost_usd == 0.0 and by_id["q003"].prompt_tokens == 0
    assert elapsed < 1.8, "one slow call must not serialise the rest"


def test_cost_totals_do_not_depend_on_record_order():
    """fsum makes the attested totals independent of scheduling."""
    proxy = _StubProxy(port=0)
    proxy.start()
    proxy.prices = {"m/a": (1e-7, 3e-7), "m/b": (7e-7, 1.1e-6)}
    rng = np.random.RandomState(0)
    recs = [APICallRecord(model_id=rng.choice(["m/a", "m/b"]), prompt_tokens=int(rng.randint(50, 900)),
                          completion_tokens=int(rng.randint(5, 700)), cost_usd=float(rng.random() * 1e-3),
                          timestamp=0.0, response_hash="", request_id=str(i)) for i in range(300)]
    totals = set()
    per_model = set()
    for _ in range(5):
        rng.shuffle(recs)
        proxy.records = list(recs)
        totals.add(proxy.total_cost)
        per_model.add(tuple(sorted(proxy.per_model_costs.items())))
    assert len(totals) == 1 and len(per_model) == 1, "cost sums varied with record order"
    proxy.stop()


@pytest.mark.parametrize("n", [1, 7])
def test_small_slices_still_work(monkeypatch, n):
    proof, _, _ = _run(monkeypatch, _recording_stub(), n=max(2, n * 2))
    assert len(proof.results) == max(1, n)
