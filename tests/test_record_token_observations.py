"""scripts/record_token_observations.py: the paid path is opt-in, the sample is
deterministic, re-runs resume, and the rows feed build_token_manifest."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "record_token_observations.py"


def _load():
    spec = importlib.util.spec_from_file_location("record_token_observations", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pool(n=60):
    benches = ["mmlu", "math", "gsm8k"]
    return [{"question_id": f"q{i:03d}", "prompt": f"question {i}?", "gold": str(i),
             "grader_id": "numeric_final", "benchmark": benches[i % 3], "metadata": {}}
            for i in range(n)]


def test_mock_refuses_the_real_upstream(monkeypatch, tmp_path):
    mod = _load()
    monkeypatch.delenv("FUGAL_OPENROUTER_BASE", raising=False)
    assert mod.main(["--per-model", "1", "--out", str(tmp_path / "o.jsonl")]) == 2
    monkeypatch.setenv("FUGAL_OPENROUTER_BASE", "https://openrouter.ai/api/v1")
    assert mod.main(["--per-model", "1", "--out", str(tmp_path / "o.jsonl")]) == 2
    assert not (tmp_path / "o.jsonl").exists()


def test_live_needs_a_key_file_never_argv(tmp_path):
    mod = _load()
    assert mod.main(["--live", "--per-model", "1", "--out", str(tmp_path / "o.jsonl")]) == 2
    assert not (tmp_path / "o.jsonl").exists()


def test_the_sample_is_deterministic_and_stratified():
    mod = _load()
    pool = _pool()
    a = mod.plan(pool, ["m/x", "m/y"], 6, "seed-1")
    b = mod.plan(pool, ["m/x", "m/y"], 6, "seed-1")
    c = mod.plan(pool, ["m/x", "m/y"], 6, "seed-2")
    assert a == b and a != c
    assert len(a) == 12
    x_ids = [q["question_id"] for m, q in a if m == "m/x"]
    y_ids = [q["question_id"] for m, q in a if m == "m/y"]
    assert x_ids != y_ids, "different models sample different questions, as the benchmark does"


def test_rows_feed_the_manifest_builder_and_reruns_resume(tmp_path, monkeypatch):
    mod = _load()
    from fugal_subnet.tee import harness
    from fugal_subnet.tee.runtime import APICallRecord, MeteringProxy

    def fake_call(proxy, model_id, question):
        rid = harness.current_request_id()
        proxy.records.append(APICallRecord(
            model_id=model_id, prompt_tokens=100 + len(question["prompt"]),
            completion_tokens=50, cost_usd=0.001, timestamp=0.0,
            response_hash="x" * 64, provider_cost_usd=0.0011, request_id=rid))
        return "42"

    monkeypatch.setattr(harness, "_call_model", fake_call)
    proxy = MeteringProxy()           # never started: no socket, no network
    pool = _pool()
    out = tmp_path / "obs.jsonl"
    scheduled = mod.plan(pool, ["m/x", "m/y"], 4, "s")
    totals = mod.record(proxy, scheduled, str(out), max_usd=10.0, concurrency=2,
                        provenance={"pool_hash": "p", "source_runtime": "s",
                                    "script_commit": "c", "upstream": "stub", "seed": "s"})
    assert totals["rows"] == 8 and totals["failed"] == 0

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(set(r) >= {"model", "prompt_tokens", "completion_tokens", "question_id",
                          "pool_hash", "source_runtime"} for r in rows)
    assert not any("text" in r or "reply" in r for r in rows), "no response text is kept"

    # The builder accepts the file as-is and emits a CANDIDATE, never reviewed.
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_token_manifest as b
    manifest = b.build([out], ["m/x", "m/y"], "test worker profile")
    assert manifest["status"] == "candidate"
    assert {m["id"]: m["samples"] for m in manifest["models"]} == {"m/x": 4, "m/y": 4}

    # Resume: the same plan again records nothing new.
    seen = mod.already_recorded(str(out))
    remaining = [(m, q) for m, q in scheduled if (m, q["question_id"]) not in seen]
    assert remaining == []


def test_the_budget_stops_scheduling(tmp_path, monkeypatch):
    mod = _load()
    from fugal_subnet.tee import harness
    from fugal_subnet.tee.runtime import APICallRecord, MeteringProxy

    def pricey(proxy, model_id, question):
        proxy.records.append(APICallRecord(
            model_id=model_id, prompt_tokens=1, completion_tokens=1, cost_usd=1.0,
            timestamp=0.0, response_hash="y" * 64, provider_cost_usd=1.0,
            request_id=harness.current_request_id()))
        return "ok"

    monkeypatch.setattr(harness, "_call_model", pricey)
    scheduled = mod.plan(_pool(), ["m/x"], 30, "s")
    totals = mod.record(MeteringProxy(), scheduled, str(tmp_path / "o.jsonl"), max_usd=3.0,
                        concurrency=1, provenance={})
    assert totals["rows"] == 3, totals   # $1 each, cap $3: the fourth call is never made
