"""Tests for TEE infrastructure: proof model, attestation, verification."""
from __future__ import annotations

import hashlib
import time

import pytest

from fugal_subnet.tee.attestation import (
    extract_report_data,
    parse_quote,
)
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult
from fugal_subnet.tee.runtime import MeteringProxy, TEERuntime, _mock_quote
from fugal_subnet.tee.verify import compute_questions_hash, verify_proof


def _make_results(n=5, accuracy=0.8):
    results = []
    for i in range(n):
        results.append(QuestionResult(
            question_id=f"q{i}",
            routed_model="model-a" if i % 2 == 0 else "model-b",
            correct=i < int(n * accuracy),
            cost_usd=0.001,
            response_hash=hashlib.sha256(f"resp{i}".encode()).hexdigest(),
        ))
    return results


def _make_proof(results=None, nonce="abc123", epoch_id="epoch_1") -> BenchmarkProof:
    if results is None:
        results = _make_results()
    per_model = {}
    for r in results:
        per_model[r.routed_model] = per_model.get(r.routed_model, 0.0) + r.cost_usd
    total = sum(r.cost_usd for r in results)
    proof = BenchmarkProof(
        epoch_id=epoch_id,
        nonce=nonce,
        questions_hash=compute_questions_hash([r.question_id for r in results]),
        weights_hash="deadbeef" * 8,
        source_hash="approved_measurement_1",
        results=results,
        total_cost_usd=total,
        per_model_costs=per_model,
        attestation_quote=b"",
        timestamp=time.time(),
    )
    runtime = TEERuntime(mock=True)
    content_hash_bytes = bytes.fromhex(proof.content_hash())
    proof.attestation_quote = runtime.generate_attestation(content_hash_bytes)
    return proof


def test_proof_construction():
    proof = _make_proof()
    assert proof.n_total == 5
    assert proof.n_correct == 4
    assert proof.accuracy == 0.8


def test_proof_content_hash_deterministic():
    results = _make_results()
    ts = 1000000.0
    p1 = _make_proof(results)
    p1.timestamp = ts
    p2 = _make_proof(results)
    p2.timestamp = ts
    assert p1.content_hash() == p2.content_hash()


def test_proof_serialization_roundtrip():
    proof = _make_proof()
    d = proof.to_dict()
    restored = BenchmarkProof.from_dict(d)
    assert restored.epoch_id == proof.epoch_id
    assert restored.nonce == proof.nonce
    assert restored.n_total == proof.n_total
    assert restored.n_correct == proof.n_correct
    assert abs(restored.total_cost_usd - proof.total_cost_usd) < 1e-10
    assert restored.attestation_quote == proof.attestation_quote
    assert restored.content_hash() == proof.content_hash()


def test_mock_quote_structure():
    data = b"test_report_data_32bytes_padding!"
    quote = _mock_quote(data)
    assert len(quote) >= 632
    parsed = parse_quote(quote)
    assert parsed.version == 4
    assert parsed.tee_type == 0x81
    rd = extract_report_data(quote)
    assert rd[:len(data)] == data


def test_parse_quote_too_short():
    try:
        parse_quote(b"short")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "bytes" in str(e)


def test_extract_report_data():
    report_data = b"A" * 64
    quote = _mock_quote(report_data)
    extracted = extract_report_data(quote)
    assert extracted == report_data


def test_verify_proof_mock_valid():
    proof = _make_proof()
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers=gold,
        mock=True,
    )
    assert result.valid, result.reason


def test_verify_proof_wrong_nonce():
    proof = _make_proof(nonce="correct_nonce")
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce="wrong_nonce",
        gold_answers=gold,
        mock=True,
    )
    assert not result.valid
    assert "Nonce mismatch" in result.reason


def test_verify_proof_wrong_questions():
    proof = _make_proof()
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash="wrong_hash",
        expected_nonce=proof.nonce,
        gold_answers=gold,
        mock=True,
    )
    assert not result.valid
    assert "Questions hash mismatch" in result.reason


def test_verify_proof_mock_skips_measurement_check():
    """In mock mode, DCAP and measurement checks are skipped."""
    proof = _make_proof()
    proof.source_hash = "unapproved_hash"
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    result = verify_proof(
        proof,
        approved_measurements={"some_other_measurement"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers=gold,
        mock=True,
    )
    assert result.valid  # mock=True skips DCAP + measurement check


def test_verify_proof_nonmock_order():
    """Non-mock mode raises ImportError when dcap_qvl is not installed."""
    proof = _make_proof()
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    with pytest.raises(ImportError, match="dcap_qvl"):
        verify_proof(
            proof,
            approved_measurements={"some_other_measurement"},
            expected_questions_hash=proof.questions_hash,
            expected_nonce=proof.nonce,
            gold_answers=gold,
            mock=False,
        )


def test_verify_proof_cost_inconsistency():
    proof = _make_proof()
    proof.total_cost_usd = 999.0  # Tamper with total
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers=gold,
        mock=True,
    )
    assert result.valid  # Cost inconsistency is a warning, not a failure
    assert result.warnings


def test_compute_questions_hash_deterministic():
    ids = ["q3", "q1", "q2"]
    h1 = compute_questions_hash(ids)
    h2 = compute_questions_hash(["q2", "q3", "q1"])
    assert h1 == h2  # Order-independent (sorted internally)


def test_tee_runtime_mock_attestation():
    runtime = TEERuntime(mock=True)
    data = hashlib.sha256(b"test").digest()
    quote = runtime.generate_attestation(data)
    assert len(quote) >= 632
    extracted = extract_report_data(quote)
    assert extracted[:32] == data


def test_metering_proxy_lifecycle():
    proxy = MeteringProxy(port=18299)
    proxy.start()
    assert proxy._server is not None
    assert proxy.total_cost == 0.0
    assert proxy.per_model_costs == {}
    proxy.stop()
    assert proxy._server is None


# --- TEE Attack Tests ---

def test_attack_tampered_proof_after_attestation():
    """A miner that modifies proof fields after attestation should be detected.

    The content_hash in the report_data won't match the tampered proof.
    """
    proof = _make_proof()
    pre_tamper_hash = proof.content_hash()

    # Tamper with a result after attestation was generated
    proof.results[0].correct = not proof.results[0].correct
    post_tamper_hash = proof.content_hash()

    # The content hash must change after tampering
    assert pre_tamper_hash != post_tamper_hash

    # The report_data in the quote was bound to the pre-tamper hash
    report_data = extract_report_data(proof.attestation_quote)
    post_tamper_expected = bytes.fromhex(post_tamper_hash).ljust(64, b"\x00")[:64]

    # report_data should NOT match the tampered proof's content hash
    assert report_data != post_tamper_expected


def test_attack_replayed_old_proof():
    """A miner replaying a proof from a previous epoch (wrong nonce)."""
    proof = _make_proof(nonce="old_epoch_nonce")
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce="current_epoch_nonce",
        gold_answers=gold,
        mock=True,
    )
    assert not result.valid
    assert "Nonce mismatch" in result.reason


def test_attack_wrong_question_set():
    """A miner answering a different set of questions than requested."""
    proof = _make_proof()
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash="completely_different_hash",
        expected_nonce=proof.nonce,
        gold_answers=gold,
        mock=True,
    )
    assert not result.valid
    assert "Questions hash mismatch" in result.reason


def test_attack_fabricated_attestation():
    """Without dcap_qvl, non-mock verification raises ImportError."""
    proof = _make_proof()
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    with pytest.raises(ImportError, match="dcap_qvl"):
        verify_proof(
            proof,
            approved_measurements={"approved_measurement_1"},
            expected_questions_hash=proof.questions_hash,
            expected_nonce=proof.nonce,
            gold_answers=gold,
            mock=False,
        )


def test_proof_content_hash_changes_on_any_tamper():
    """Any change to proof content must change the content hash."""
    proof = _make_proof()
    original = proof.content_hash()

    # Tamper: change a result
    proof.results[0].correct = not proof.results[0].correct
    assert proof.content_hash() != original

    # Restore and tamper differently
    proof.results[0].correct = not proof.results[0].correct
    assert proof.content_hash() == original  # restored

    proof.total_cost_usd += 0.001
    assert proof.content_hash() != original


# --- Harness unit tests (M3) ---

def test_compute_questions_hash_sorted_order():
    """compute_questions_hash is order-independent (sorted internally)."""
    from fugal_subnet.tee.proof import compute_questions_hash as cqh
    h1 = cqh(["q3", "q1", "q2"])
    h2 = cqh(["q1", "q2", "q3"])
    assert h1 == h2


def test_compute_questions_hash_unique():
    """Different question sets produce different hashes."""
    from fugal_subnet.tee.proof import compute_questions_hash as cqh
    h1 = cqh(["q1", "q2"])
    h2 = cqh(["q1", "q3"])
    assert h1 != h2


def test_cost_consistency_tolerance():
    """Verify M7 fix: relative + absolute cost tolerance."""
    from fugal_subnet.tee.verify import _costs_consistent
    # Small costs: 5% relative tolerance
    assert _costs_consistent(0.005, 0.005)
    assert _costs_consistent(0.005, 0.0055)
    # Large costs: 5% relative tolerance
    assert _costs_consistent(1.0, 1.04)
    assert not _costs_consistent(1.0, 1.06)
    # Absolute floor: $0.001
    assert _costs_consistent(0.0, 0.0005)
    assert not _costs_consistent(0.0, 0.002)


# --- TEE pipeline integration test (M2) ---

def test_tee_verify_then_score_pipeline():
    """End-to-end: make proof → verify → convert to HeadScore."""
    import numpy as np

    from fugal_subnet.tee.proof import compute_questions_hash as cqh

    proof = _make_proof()
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    expected_qhash = cqh([r.question_id for r in proof.results])

    # Step 1: verify
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash=expected_qhash,
        expected_nonce=proof.nonce,
        gold_answers=gold,
        mock=True,
    )
    assert result.valid, f"verify failed: {result.reason}"

    # Step 2: convert to HeadScore (same as validator._proof_to_head_score)
    from fugal_subnet.head_eval import HeadScore
    n_correct = proof.n_correct
    n_scored = proof.n_total
    total_head_cost = proof.total_cost_usd
    if proof.per_model_costs:
        cheapest = min(proof.per_model_costs.values())
        total_oracle_cost = cheapest * max(len(proof.results), 1)
    else:
        total_oracle_cost = total_head_cost
    cost_efficiency = min(1.0, total_oracle_cost / max(total_head_cost, 1e-10))

    score = HeadScore(
        accuracy=proof.accuracy,
        cost_efficiency=cost_efficiency,
        kl_score=0.0,
        routing_decisions=np.array([0, 1, 0, 1, 0], dtype=np.int32),
        correct_mask=np.array([r.correct for r in proof.results], dtype=bool),
        coverage=1.0,
        n_correct=n_correct,
        n_scored=n_scored,
        total_head_cost=total_head_cost,
        total_oracle_cost=total_oracle_cost,
        total_kl=0.0,
    )
    assert score.accuracy == proof.accuracy
    assert 0.0 <= score.cost_efficiency <= 1.0
    assert score.n_correct == 4
    assert score.n_scored == 5
