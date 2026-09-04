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


def _make_proof(
    results=None,
    nonce="abc123",
    epoch_id="epoch_1",
    source_hash="approved_measurement_1",
    total_cost_usd=None,
    per_model_costs=None,
) -> BenchmarkProof:
    """Build an internally consistent, correctly attested proof.

    Overrides are applied BEFORE attestation, so a proof built here is always
    self-consistent — a test that wants to model tampering must mutate the
    returned object, which is exactly what report_data binding then catches.
    """
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
        source_hash=source_hash,
        results=results,
        total_cost_usd=total if total_cost_usd is None else total_cost_usd,
        per_model_costs=per_model if per_model_costs is None else per_model_costs,
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


def test_verify_proof_mock_skips_hardware_trust_checks():
    """Mock skips DCAP and approved-image matching — there is no real hardware.

    Everything else still applies: mock is not a bypass, it is the absence of a
    machine to attest. source_hash is set before attestation here, so the proof
    is internally consistent; it is simply not an approved image.
    """
    proof = _make_proof(source_hash="an_unapproved_image")
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    result = verify_proof(
        proof,
        approved_measurements={"some_other_measurement"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers=gold,
        mock=True,
    )
    assert result.valid, result.reason


def test_post_attestation_tamper_caught_even_in_mock():
    """Editing any attested field after the fact breaks the report_data binding.

    This is enforced in mock too: the structural hash chain costs nothing to
    check and is what makes a local testnet a meaningful rehearsal.
    """
    proof = _make_proof()
    proof.source_hash = "swapped_after_attestation"
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers=gold,
        mock=True,
    )
    assert not result.valid
    assert "report_data mismatch" in result.reason


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


def test_verify_proof_cost_inconsistency_is_rejected():
    """An internally inconsistent cost report is a rejection, not a warning.

    Understating total_cost_usd raises a miner's thrift score, so treating the
    inconsistency as advisory paid the attacker. Under real attestation every
    figure comes from the same metered image, so this can never legitimately
    fail — if it does, the proof is not what it claims to be.
    """
    results = _make_results()
    proof = _make_proof(results=results, total_cost_usd=0.0001)
    gold = {f"q{i}": {"question_id": f"q{i}"} for i in range(5)}
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers=gold,
        mock=True,
    )
    assert not result.valid
    assert "Cost inconsistency" in result.reason


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


# --- Binding tests: each covers a hole that was open before hardening ---

def _gold(n=5):
    return {f"q{i}": {"question_id": f"q{i}"} for i in range(n)}


def test_results_must_match_the_assigned_slice():
    """questions_hash is public, so matching it proves nothing on its own.

    A miner can copy the expected hash while grading an easier set of its own
    choosing. Only comparing the result ids against the assigned slice closes it.
    """
    assigned = {f"q{i}" for i in range(5)}
    easy = [
        QuestionResult(f"easy{i}", "model-a", True, 0.001,
                       hashlib.sha256(b"x").hexdigest())
        for i in range(5)
    ]
    proof = _make_proof(results=easy)
    proof.questions_hash = compute_questions_hash(sorted(assigned))
    # Re-attest so only the slice binding, not report_data, is under test.
    proof.attestation_quote = TEERuntime(mock=True).generate_attestation(
        bytes.fromhex(proof.content_hash())
    )
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers={**_gold(), **{f"easy{i}": {} for i in range(5)}},
        expected_question_ids=assigned,
        mock=True,
    )
    assert not result.valid
    assert "does not match the assigned slice" in result.reason


def test_proof_must_run_the_head_committed_on_chain():
    """Commit head A, run head B: the evidence key would otherwise be free."""
    proof = _make_proof()
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers=_gold(),
        expected_weights_hash="cafe" * 16,  # what the chain says
        mock=True,
    )
    assert not result.valid
    assert "Head mismatch" in result.reason


def test_bundled_head_must_hash_to_the_attested_weights_hash():
    proof = _make_proof()
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers=_gold(),
        head_bytes=b"not the head that ran",
        mock=True,
    )
    assert not result.valid
    assert "Bundled head does not match" in result.reason


def test_downloaded_bundle_must_match_the_advertised_hash():
    proof = _make_proof()
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers=_gold(),
        expected_proof_hash="f" * 64,
        mock=True,
    )
    assert not result.valid
    assert "Bundle mismatch" in result.reason


def test_all_bindings_satisfied_verifies():
    """The honest path: every expectation supplied and every one satisfied."""
    head_bytes = b"a real head artifact"
    results = _make_results()
    proof = _make_proof(results=results)
    proof.weights_hash = hashlib.sha256(head_bytes).hexdigest()
    proof.attestation_quote = TEERuntime(mock=True).generate_attestation(
        bytes.fromhex(proof.content_hash())
    )
    result = verify_proof(
        proof,
        approved_measurements={"approved_measurement_1"},
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers=_gold(),
        expected_question_ids={r.question_id for r in results},
        expected_weights_hash=proof.weights_hash,
        expected_proof_hash=proof.content_hash(),
        head_bytes=head_bytes,
        mock=True,
    )
    assert result.valid, result.reason


def test_measurement_comes_from_the_quote_not_the_proof():
    """The approved-image check must read hardware registers, not a claim.

    An attacker with real TDX hardware produces a genuine Intel-signed quote
    while running modified code. What separates them from an honest miner is
    the measurement register the CPU filled in — so measurement_id must be a
    function of the quote alone.
    """
    from fugal_subnet.tee.attestation import measurement_id

    proof = _make_proof(source_hash="i-claim-to-be-the-approved-image")
    quote = parse_quote(proof.attestation_quote)
    assert measurement_id(quote) != proof.source_hash
    # It is derived purely from the measurement registers.
    assert measurement_id(quote) == measurement_id(parse_quote(proof.attestation_quote))


def test_parse_quote_rejects_non_tdx():
    import struct
    bad = bytearray(_mock_quote(b"x" * 64))
    struct.pack_into("<I", bad, 4, 0x0)  # tee_type: not TDX
    with pytest.raises(ValueError, match="not TDX"):
        parse_quote(bytes(bad))
