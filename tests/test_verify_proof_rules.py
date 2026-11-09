"""verify_proof's local rules, each through the real entry point.

Planted bugs (2026-11-08) showed six of these rules untested by pytest: removing
any one of them passed the suite, and two were caught by nothing at all
(duplicate exploration entries, the prompt_tokens bound). Each case below
differs from an honest control by exactly one thing, so it fails if, and only
if, its own rule is removed.
"""
import hashlib

import pytest

from fugal_subnet.config import PROOF_MAX_TOKENS
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult
from fugal_subnet.tee.runtime import TEERuntime
from fugal_subnet.tee.verify import verify_proof

A, B = "model/a", "model/b"
EMPTY_REPLY = hashlib.sha256(b"").hexdigest()
GOLD = {q: {"question_id": q, "prompt": "?", "gold": "1", "grader_id": "numeric_final",
            "benchmark": "gsm8k", "metadata": {}} for q in ("q1", "q2", "x1", "x2")}
SLICE = {"q1", "q2"}
EXPLORE = {"x1": A, "x2": B}


def _r(qid, model=A, explore=False, **kw):
    fields = dict(correct=True, cost_usd=0.01, response_hash="f" * 64,
                  prompt_tokens=10, completion_tokens=5)
    fields.update(kw)
    return QuestionResult(question_id=qid, routed_model=model, is_exploration=explore, **fields)


def _honest():
    return [_r("q1"), _r("q2", B), _r("x1", A, explore=True), _r("x2", B, explore=True)]


def _verify(results, *, expected_question_ids=SLICE, expected_exploration=EXPLORE):
    """Mock mode: every local rule runs; only the hardware checks are skipped."""
    per_model: dict[str, float] = {}
    for r in results:
        per_model[r.routed_model] = per_model.get(r.routed_model, 0.0) + r.cost_usd
    p = BenchmarkProof(
        hotkey="5Miner", epoch_id="e1", nonce="ab" * 32, questions_hash="c" * 64,
        weights_hash="d" * 64, source_hash="e" * 64, results=results,
        total_cost_usd=sum(r.cost_usd for r in results), per_model_costs=per_model,
        attestation_quote=b"", timestamp=1.0,
    )
    p.attestation_quote = TEERuntime(mock=True).generate_attestation(bytes.fromhex(p.content_hash()))
    return verify_proof(p, set(), p.questions_hash, p.nonce, GOLD, mock=True,
                        expected_question_ids=expected_question_ids,
                        expected_exploration=expected_exploration)


def test_the_honest_control_passes():
    r = _verify(_honest())
    assert r.valid, r.reason


def test_a_duplicated_scored_result_is_refused():
    """Same question set, one answered twice: the set comparison alone passes it."""
    r = _verify(_honest() + [_r("q1")])
    assert not r.valid and r.reason.startswith("Duplicate results: 3 entries for 2")


def test_a_duplicated_exploration_result_is_refused():
    """Same assignment, same models, one question listed twice: only the count
    catches it (the dict built from the list keeps one)."""
    r = _verify(_honest() + [_r("x1", A, explore=True)])
    assert not r.valid and r.reason.startswith("Duplicate exploration results: 3 entries for 2")


def test_exploration_routed_to_another_model_is_refused():
    """The right questions, explored on a model the nonce did not assign: a
    miner sampling the model it prefers instead of the one the frame needs."""
    results = _honest()
    results[2] = _r("x1", B, explore=True)
    r = _verify(results)
    assert not r.valid and "routed to a model other than the one the nonce assigned" in r.reason
    assert "x1: got 'model/b', required 'model/a'" in r.reason


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
def test_each_token_count_is_bounded(field):
    """Token counts feed the frame and the v2 spend, and a JSON integer is
    unbounded (I2). Both fields, at the bound and one past it, and below 0."""
    at_bound = _honest()
    at_bound[0] = _r("q1", **{field: PROOF_MAX_TOKENS})
    assert _verify(at_bound).valid

    for bad in (PROOF_MAX_TOKENS + 1, -1):
        results = _honest()
        results[0] = _r("q1", **{field: bad})
        r = _verify(results)
        assert not r.valid and "token count outside" in r.reason, (field, bad)


def test_a_failed_call_is_a_wrong_answer_with_no_reply():
    """The flag is worth lying about in both directions, so it is held to its
    meaning: failed means not correct AND no reply."""
    honest_failure = _honest()
    honest_failure[0] = _r("q1", correct=False, failed=True, response_hash=EMPTY_REPLY)
    assert _verify(honest_failure).valid

    marked_correct = _honest()
    marked_correct[0] = _r("q1", correct=True, failed=True, response_hash=EMPTY_REPLY)
    with_reply = _honest()
    with_reply[0] = _r("q1", correct=False, failed=True, response_hash="f" * 64)
    for results in (marked_correct, with_reply):
        r = _verify(results)
        assert not r.valid and "marked failed but carry a grade or a reply" in r.reason


def test_a_result_with_no_gold_is_refused_even_without_a_slice():
    """With the slice given, an unknown question already fails the slice
    comparison. Without it (scripts, replays) this is the only backstop: a
    result nobody can have graded is refused."""
    results = _honest()
    results[1] = _r("q9", B)
    r = _verify(results, expected_question_ids=None)
    assert not r.valid and r.reason.startswith("1 results reference questions with no gold answer")
