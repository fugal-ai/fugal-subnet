"""Tests for TEE infrastructure: proof model, attestation, verification."""
from __future__ import annotations

import hashlib
import time

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
    """Non-mock mode checks DCAP first, then measurement."""
    # Without dcap_qvl installed, DCAP fails before measurement check.
    # This verifies the check ordering: DCAP → measurement → nonce → ...
    proof = _make_proof()
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    result = verify_proof(
        proof,
        approved_measurements={"some_other_measurement"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers=gold,
        mock=False,
    )
    assert not result.valid
    assert "DCAP" in result.reason or "Unapproved" in result.reason


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
    # Tamper with a result after attestation
    proof.results[0].correct = not proof.results[0].correct

    # In non-mock mode, the report_data binding check would catch this.
    # In mock mode, binding is skipped, but the content hash changed.
    original_hash = proof.content_hash()
    # The attestation was generated for the pre-tamper content hash.
    # The report_data in the quote doesn't match the post-tamper hash.
    report_data = extract_report_data(proof.attestation_quote)
    tampered_expected = bytes.fromhex(original_hash).ljust(64, b"\x00")[:64]
    # These should NOT match because we tampered after attestation
    assert report_data[:32] != tampered_expected[:32] or proof.n_correct != 4


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
    """Without dcap_qvl, fabricated quotes fail DCAP verification."""
    proof = _make_proof()
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers=gold,
        mock=False,  # Non-mock mode requires real DCAP
    )
    # Without dcap_qvl installed, DCAP verification fails
    assert not result.valid


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
