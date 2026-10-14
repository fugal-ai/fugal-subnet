"""`QuestionResult.failed`: a call that did not complete, attested as such.

Diligence 2026-10, finding 2. Before this field a failed model call was
indistinguishable from a wrong answer, and the reference frame read it as the
model being wrong. The field is worth lying about in both directions once the
frame drops failed samples, so these tests pin what it may and may not mean,
and that adding it did not move the hash of any proof that has no failures.
"""
from __future__ import annotations

import hashlib

from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult

EMPTY = hashlib.sha256(b"").hexdigest()


def _proof(results):
    return BenchmarkProof(
        epoch_id="e1", nonce="n" * 64, questions_hash="q" * 64, weights_hash="w" * 64,
        source_hash="s" * 64, results=results, total_cost_usd=0.0, per_model_costs={},
        attestation_quote=b"", timestamp=1.0, hotkey="5hot",
    )


def test_a_proof_with_no_failures_hashes_exactly_as_before_the_field_existed():
    """Only a True flag enters the hash, so honest proofs are unchanged across
    the upgrade and miners and validators on either side of it agree."""
    r = QuestionResult("q1", "m", True, 0.001, "ab" * 32, 10, 20)
    assert "failed" not in r.hash_payload()
    legacy_payload = {k: v for k, v in r.to_dict().items() if k != "failed"}
    assert r.hash_payload() == legacy_payload


def test_a_failure_changes_the_hash():
    ok = QuestionResult("q1", "m", False, 0.0, EMPTY)
    failed = QuestionResult("q1", "m", False, 0.0, EMPTY, failed=True)
    assert _proof([ok]).content_hash() != _proof([failed]).content_hash()


def test_serialisation_round_trips_and_old_dicts_still_parse():
    failed = QuestionResult("q1", "m", False, 0.0, EMPTY, failed=True)
    assert QuestionResult.from_dict(failed.to_dict()) == failed
    old = {"question_id": "q1", "routed_model": "m", "correct": True, "cost_usd": 0.1,
           "response_hash": "ab" * 32, "prompt_tokens": 1, "completion_tokens": 2,
           "is_exploration": False}
    assert QuestionResult.from_dict(old).failed is False
    proof = _proof([failed])
    assert BenchmarkProof.from_dict(proof.to_dict()).content_hash() == proof.content_hash()


def test_the_harness_records_a_call_that_did_not_complete_as_failed(monkeypatch):
    """None from _call_model is failed; "" is a wrong answer that completed."""
    from fugal_subnet.tee import harness

    def fake_call(proxy, model_id, question):
        return None if question["question_id"] == "dead" else ""

    monkeypatch.setattr(harness, "_call_model", fake_call)

    class Proxy:
        records = []

    sched = [({"question_id": "dead"}, "m", False), ({"question_id": "mute"}, "m", False)]
    out = list(harness._call_all(Proxy(), sched, "e1"))
    (_q1, _m1, _x1, text1, failed1, *_rest1), (_q2, _m2, _x2, text2, failed2, *_rest2) = out
    assert (text1, failed1) == ("", True)
    assert (text2, failed2) == ("", False)
