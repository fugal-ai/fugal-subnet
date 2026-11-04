"""Per-UID failed-call counts in the epoch log (mainnet, 2026-11-04).

A failed call is scored `correct=False`, so from the score alone a miner whose
upstream calls failed looks the same as one whose answers were wrong. The epoch
log now records, per UID, how many SCORED calls the proof marks `failed`.

Diagnostics only. These tests pin that: the count is right, exploration is left
out (as it is from the score), it lands in `epochs.jsonl`, and it stays out of
the score dicts that the reveal and consensus digest are built from.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from fugal_subnet.epoch_logger import EpochLog
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult
from neurons.validator import count_failed_scored

ROOT = Path(__file__).resolve().parents[1]
EMPTY = hashlib.sha256(b"").hexdigest()


def _proof(results):
    return BenchmarkProof(
        epoch_id="e1", nonce="n" * 64, questions_hash="q" * 64, weights_hash="w" * 64,
        source_hash="s" * 64, results=results, total_cost_usd=0.001, per_model_costs={"m": 0.001},
        attestation_quote=b"", timestamp=1.0, hotkey="5hot",
    )


def _r(qid, correct=False, failed=False, explore=False):
    return QuestionResult(
        qid, "m", correct, 0.0 if failed else 0.001, EMPTY if failed else "ab" * 32,
        0 if failed else 10, 0 if failed else 20, is_exploration=explore, failed=failed,
    )


def test_counts_failed_scored_calls_only():
    proof = _proof([
        _r("q1", failed=True),
        _r("q2", failed=True),
        _r("q3", correct=True),
        _r("q4"),                              # wrong answer, not failed
        _r("x1", failed=True, explore=True),   # exploration: not scored, not counted
    ])
    assert count_failed_scored(proof) == 2
    assert proof.n_total == 4                  # the score's denominator is unchanged


def test_no_failures_counts_zero():
    assert count_failed_scored(_proof([_r("q1", correct=True), _r("q2")])) == 0


def test_epoch_log_carries_n_failed_and_defaults_empty():
    base = dict(
        epoch_id="e-1", block_hash="0x" + "ab" * 32, timestamp=0.0,
        n_questions=300, n_miners_queried=256, n_heads_valid=2, n_heads_invalid=0,
    )
    assert EpochLog(**base).n_failed == {}
    row = json.loads(json.dumps(asdict(EpochLog(**base, n_failed={203: 7, 214: 0}))))
    assert row["n_failed"] == {"203": 7, "214": 0}


def test_n_failed_stays_out_of_the_reveal_score_dicts():
    src = (ROOT / "neurons" / "validator.py").read_text()
    m = re.search(r"epoch_score_dicts = \{(.*?)\n            \}\n", src, re.S)
    assert m, "epoch_score_dicts construction not found"
    assert "failed" not in m.group(1)
