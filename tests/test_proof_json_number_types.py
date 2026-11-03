"""A proof's numbers hash as the miner serialised them (mainnet day one, 2026-11-03).

Failed calls carry `cost_usd: 0` (int). The validator's `from_dict` used to coerce
it to float, so the canonical JSON it hashed said `0.0` where the miner's said `0`,
and `content_hash` no longer matched the attested `report_data` on every proof
that contained a failed call. These tests pin the round trip byte for byte and
the type check that replaced the coercion.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult

EMPTY = hashlib.sha256(b"").hexdigest()


def _proof(results):
    return BenchmarkProof(
        epoch_id="e1", nonce="n" * 64, questions_hash="q" * 64, weights_hash="w" * 64,
        source_hash="s" * 64, results=results, total_cost_usd=0.001, per_model_costs={"m": 0.001},
        attestation_quote=b"", timestamp=1.0, hotkey="5hot",
    )


def test_an_int_zero_cost_on_a_failed_result_round_trips_to_the_same_content_hash():
    failed = QuestionResult("q1", "m", False, 0, EMPTY, 0, 0, failed=True)   # int 0, as the harness records it
    ok = QuestionResult("q2", "m", True, 0.001, "ab" * 32, 10, 20)
    original = _proof([failed, ok])
    wire = json.loads(json.dumps(original.to_dict()))                        # what the validator receives
    parsed = BenchmarkProof.from_dict(wire)
    assert parsed.results[0].cost_usd == 0 and type(parsed.results[0].cost_usd) is int
    assert type(parsed.results[1].cost_usd) is float
    assert parsed.content_hash() == original.content_hash()
    # and the hash really does depend on the JSON number type
    coerced = _proof([QuestionResult("q1", "m", False, 0.0, EMPTY, 0, 0, failed=True), ok])
    assert coerced.content_hash() != original.content_hash()


def test_token_counts_keep_their_json_type_too():
    r = QuestionResult("q1", "m", True, 0.5, EMPTY, 7, 9)
    wire = json.loads(json.dumps(r.to_dict()))
    back = QuestionResult.from_dict(wire)
    assert (back.prompt_tokens, back.completion_tokens) == (7, 9)
    assert type(back.prompt_tokens) is int and type(back.completion_tokens) is int
    assert back.hash_payload() == r.hash_payload()


@pytest.mark.parametrize("bad", [True, False, "0", "0.0", None, [0]])
def test_non_numbers_are_rejected_not_reinterpreted(bad):
    d = QuestionResult("q1", "m", False, 0, EMPTY, failed=True).to_dict()
    d["cost_usd"] = bad
    with pytest.raises((ValueError, TypeError)):
        QuestionResult.from_dict(d)


def test_non_finite_numbers_are_rejected():
    d = QuestionResult("q1", "m", False, 0, EMPTY, failed=True).to_dict()
    for bad in (float("nan"), float("inf")):
        d["cost_usd"] = bad
        with pytest.raises(ValueError):
            QuestionResult.from_dict(d)
