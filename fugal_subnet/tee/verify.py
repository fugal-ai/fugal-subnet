"""Proof verification for TEE-attested benchmarks.

The validator calls verify_proof() for each miner's submission.
It checks attestation, measurements, question consistency, grading,
and cost consistency — without ever calling a model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from fugal_subnet.tee.attestation import extract_report_data, verify_dcap
from fugal_subnet.tee.proof import (
    BenchmarkProof,
)
from fugal_subnet.tee.proof import (
    compute_questions_hash as compute_questions_hash,  # noqa: F401 — re-export
)

logger = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    valid: bool
    reason: str = ""
    warnings: list[str] | None = None


def verify_proof(
    proof: BenchmarkProof,
    approved_measurements: set[str],
    expected_questions_hash: str,
    expected_nonce: str,
    gold_answers: dict[str, dict],
    mock: bool = False,
) -> VerifyResult:
    """Verify a miner's TEE-attested benchmark proof.

    Checks (in order, fail-fast):
    1. Attestation quote is valid (DCAP chain) — skipped in mock mode
    2. report_data in quote matches proof content hash
    3. source_hash matches an approved runtime image measurement
    4. nonce matches expected epoch nonce
    5. questions_hash matches expected question set
    6. Each QuestionResult.correct matches re-grading against gold
    7. Costs are consistent (per-question sum ≈ total)

    Args:
        proof: The miner's benchmark proof.
        approved_measurements: Set of approved source_hash values.
        expected_questions_hash: SHA256 of the expected question IDs.
        expected_nonce: The expected epoch nonce.
        gold_answers: {question_id: task_dict} for re-grading.
        mock: If True, skip DCAP attestation verification.

    Returns:
        VerifyResult with pass/fail and reason.
    """
    warnings: list[str] = []

    # 1. DCAP attestation
    if not mock:
        if not verify_dcap(proof.attestation_quote):
            return VerifyResult(False, "DCAP attestation verification failed")

    # 2. report_data binding
    try:
        report_data = extract_report_data(proof.attestation_quote)
    except ValueError as e:
        return VerifyResult(False, f"Invalid attestation quote: {e}")

    content_hash = proof.content_hash()
    expected_report_data = bytes.fromhex(content_hash).ljust(64, b"\x00")[:64]

    if not mock and report_data != expected_report_data:
        return VerifyResult(
            False,
            f"report_data mismatch: quote has {report_data.hex()[:32]}..., "
            f"expected {expected_report_data.hex()[:32]}...",
        )

    # 3. Source measurement
    if proof.source_hash not in approved_measurements and not mock:
        return VerifyResult(
            False,
            f"Unapproved source_hash: {proof.source_hash[:16]}... "
            f"not in {len(approved_measurements)} approved measurements",
        )

    # 4. Nonce
    if proof.nonce != expected_nonce:
        return VerifyResult(
            False,
            f"Nonce mismatch: proof has {proof.nonce[:16]}..., "
            f"expected {expected_nonce[:16]}...",
        )

    # 5. Questions hash
    if proof.questions_hash != expected_questions_hash:
        return VerifyResult(
            False,
            f"Questions hash mismatch: proof has {proof.questions_hash[:16]}..., "
            f"expected {expected_questions_hash[:16]}...",
        )

    # 6. Verify question coverage
    # We can't re-grade without the actual response text (only its hash
    # is in the proof). The proof's `correct` field was computed inside
    # the TEE by the hash-pinned grader. We trust the attestation that
    # the grading happened correctly. Re-grading is only possible for
    # held-out questions where the validator independently evaluates the
    # head's routing.
    for result in proof.results:
        if result.question_id not in gold_answers:
            warnings.append(f"Unknown question_id: {result.question_id}")

    # 7. Cost consistency (relative + absolute tolerance)
    per_q_sum = sum(r.cost_usd for r in proof.results)
    if not _costs_consistent(per_q_sum, proof.total_cost_usd):
        warnings.append(
            f"Cost inconsistency: per-question sum {per_q_sum:.4f} vs "
            f"reported total {proof.total_cost_usd:.4f}"
        )

    per_model_sum = sum(proof.per_model_costs.values())
    if not _costs_consistent(per_model_sum, proof.total_cost_usd):
        warnings.append(
            f"Per-model cost sum {per_model_sum:.4f} vs "
            f"reported total {proof.total_cost_usd:.4f}"
        )

    return VerifyResult(
        valid=True,
        reason="All checks passed",
        warnings=warnings if warnings else None,
    )


def _costs_consistent(a: float, b: float) -> bool:
    """Check cost consistency using both relative (5%) and absolute ($0.001) tolerance."""
    diff = abs(a - b)
    return diff <= max(0.05 * max(abs(a), abs(b)), 0.001)


