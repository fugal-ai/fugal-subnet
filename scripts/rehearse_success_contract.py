#!/usr/bin/env python3
"""Mocked local-chain rehearsal: train -> commit -> proof -> weights -> restart.

The chain, embeddings and worker responses are explicit in-process fixtures.
No Docker, remote chain, TDX hardware or paid inference is used.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# isort: off
import fugal_subnet.determinism  # noqa: E402,F401
import numpy as np  # noqa: E402

# isort: on
from fugal_subnet.benchmarks.slicer import derive_nonce, select_slice  # noqa: E402
from fugal_subnet.exploration import expected_exploration  # noqa: E402
from fugal_subnet.frontier import build_frontier  # noqa: E402
from fugal_subnet.head_eval import HeadScore  # noqa: E402
from fugal_subnet.reference_frame import ReferenceFrame, accumulate_exploration  # noqa: E402
from fugal_subnet.rewards import compute_weights  # noqa: E402
from fugal_subnet.routing_protocol import MANIFEST_PATH, PRICES_PATH, identity  # noqa: E402
from fugal_subnet.scoring import ScoringState, update_scores  # noqa: E402
from fugal_subnet.success_training import fit, grouped_split  # noqa: E402
from fugal_subnet.tee import harness  # noqa: E402
from fugal_subnet.tee.proof import compute_questions_hash  # noqa: E402
from fugal_subnet.tee.runtime import APICallRecord, MeteringProxy, TEERuntime  # noqa: E402
from fugal_subnet.tee.verify import verify_proof  # noqa: E402
from fugal_subnet.vendor import success_contract as c  # noqa: E402


class MockChain:
    """A deterministic local commitment/weight journal, with no network access."""
    def __init__(self):
        self.commitment = None
        self.weights = []

    def commit(self, digest):
        self.commitment = digest

    def set_weights(self, uids, weights):
        assert abs(sum(weights) - 1) < 1e-9
        self.weights.append(dict(zip(uids, weights)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text())
    models = ["deepseek/deepseek-v4-flash", "openai/gpt-5.4-nano", "meta-llama/llama-4-maverick"]
    prices = {r["id"]: (r["in"] / 1e6, r["out"] / 1e6) for r in json.loads(PRICES_PATH.read_text())}
    rng = np.random.default_rng(42)
    questions = [f"What is {i}+1?" for i in range(100)]
    hidden = rng.normal(size=(100, 1024)).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    labels = rng.integers(0, 2, (100, len(models))).astype(float)
    labels[0] = 0
    labels[1, 1] = np.nan
    train, val, test = grouped_split(questions)
    W, b, epoch, history = fit(hidden, labels, train, val, epochs=5)
    buf = io.BytesIO()
    np.savez(buf, **c.make_head(W, b, models, manifest, "synthetic-test-only mocked local chain"))
    head = buf.getvalue()
    (out / "head.npz").write_bytes(head)
    chain = MockChain()
    chain.commit(hashlib.sha256(head).hexdigest())
    pool = [{"question_id": f"q{i}", "prompt": q, "gold": str(i+1),
             "grader_id": "numeric_final", "benchmark": "gsm8k", "metadata": {}} for i, q in enumerate(questions)]
    calls = []

    class Proxy(MeteringProxy):
        def start(self):
            self.prices = prices

    def worker(proxy, model_id, question):
        calls.append(question["question_id"])
        text = question["gold"] if len(calls) % 3 else "-1"
        proxy.records.append(APICallRecord(model_id=model_id, prompt_tokens=100, completion_tokens=30,
            cost_usd=proxy.price_call(model_id, 100, 30), timestamp=0.,
            response_hash=hashlib.sha256(text.encode()).hexdigest(), request_id=harness.current_request_id()))
        return text

    nonce = derive_nonce("e00000001", "0x" + "ab" * 32)
    scored = select_slice(nonce, pool, 20)
    ids = [q["question_id"] for q in scored]
    exploration = expected_exploration(nonce, pool, set(ids), models, 3)
    proxy = Proxy(port=0)
    proxy.start()
    with patch.object(harness, "_call_model", worker):
        proof = harness.run_benchmark(nonce.hex(), head, pool, proxy, hidden,
            slice_size=20, epoch_id="e00000001", source_hash="mocked-local-chain",
            explore_models=models, explore_size=3, hotkey="mock-hotkey")
    proof.timestamp = 0.
    proof.attestation_quote = TEERuntime(mock=True).generate_attestation(bytes.fromhex(proof.content_hash()))
    kwargs = dict(approved_measurements=set(), expected_questions_hash=compute_questions_hash(ids),
        expected_nonce=nonce.hex(), gold_answers={q["question_id"]: q for q in pool},
        expected_question_ids=set(ids), expected_exploration=exploration,
        expected_weights_hash=chain.commitment, expected_hotkey="mock-hotkey", head_bytes=head, mock=True)
    verdict = verify_proof(proof, **kwargs)
    assert verdict.valid, verdict.reason
    assert len(calls) == 23 and len(set(calls)) == 23
    old = verify_proof(replace(proof, routing_protocol="legacy"), **kwargs)
    assert not old.valid
    frame = accumulate_exploration(ReferenceFrame(), [(r.routed_model, r.correct, r.prompt_tokens, r.completion_tokens) for r in proof.exploration_results])
    frontier = build_frontier(frame, prices, 2000, 20, 30.)
    score = HeadScore(proof.accuracy, 0., 0., np.zeros(20, dtype=int),
                      np.array([r.correct for r in proof.scored_results]), n_correct=proof.n_correct,
                      n_scored=20, total_head_cost=proof.scored_cost_usd)
    state = update_scores(ScoringState(), {1: score}, {1: proof.weights_hash}, frontier=frontier,
                          hotkeys={1: "mock-hotkey"}, n_questions=20, pool_size=100)
    chain.set_weights(*compute_weights(state.records, paid_fraction=frontier.confidence))
    assert ReferenceFrame.from_dict(frame.to_dict()).to_dict() == frame.to_dict()
    report = {"mode": "mocked local-chain mechanism rehearsal; no paid inference or real chain writes",
              "routing_protocol": identity(), "head_sha256": c.file_hash(out / "head.npz"),
              "selected_epoch": epoch, "history": history, "test_rows": len(test),
              "scored_calls": proof.n_total, "exploration_calls": len(proof.exploration_results),
              "proof_valid": verdict.valid, "old_proof_rejected": not old.valid,
              "frame_restart_equal": True, "weight_writes": chain.weights}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "tokens.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "prices.json").write_bytes(PRICES_PATH.read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
