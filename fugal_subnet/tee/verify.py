"""Proof verification for TEE-attested benchmarks.

The validator calls verify_proof() for each miner's submission. It checks
attestation, measurements, question consistency, and cost consistency —
without ever calling a model.

The security model in one line: *nothing a miner asserts about itself is
trusted; only what the hardware measured, or what a hash chain forces to be
true, is trusted.*

That distinction is the reason this module rejects `proof.source_hash` as an
image identity. A workload running inside a genuine TDX VM produces a genuine
Intel-signed quote, so DCAP verification passes for an attacker who simply
runs *modified* code on real hardware. What separates honest from modified is
the measurement register the CPU filled in, not a string the workload wrote
about itself.

The chain that makes a proof trustworthy:

    Intel DCAP signature
      -> quote is genuine, from real TDX hardware
    measurement_id(quote) in approved_measurements
      -> the image that ran is the published one
    report_data == proof.content_hash()
      -> the proof body is exactly what that image produced
    proof.weights_hash == on-chain commitment, == sha256(head bytes)
      -> the head that ran is the head that was committed, before the nonce
    result question ids == the assigned slice
      -> those answers are to the questions we asked
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from fugal_subnet.tee.attestation import (
    extract_report_data,
    measurement_id,
    parse_quote,
    verify_dcap,
)
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
    *,
    expected_question_ids: set[str] | None = None,
    expected_exploration: dict[str, str] | None = None,
    expected_weights_hash: str = "",
    expected_proof_hash: str = "",
    head_bytes: bytes | None = None,
    mock: bool = False,
) -> VerifyResult:
    """Verify a miner's TEE-attested benchmark proof.

    Structural and hash-chain checks run in every mode — they cost nothing and
    they are what makes a local testnet meaningful. Only the checks that need
    real hardware (the DCAP signature chain and approved-image matching) are
    skipped under `mock`.

    Args:
        proof: The miner's benchmark proof.
        approved_measurements: Approved `measurement_id()` values.
        expected_questions_hash: SHA256 of the expected question IDs.
        expected_nonce: The expected epoch nonce.
        gold_answers: {question_id: task_dict} for the assigned slice.
        expected_question_ids: The exact question set the miner was assigned.
        expected_exploration: {question_id: required_model} for this epoch.
        expected_weights_hash: Head hash committed on-chain before the boundary.
        expected_proof_hash: content_hash the miner advertised over the axon.
        head_bytes: The head artifact shipped in the bundle.
        mock: If True, skip hardware-trust checks (DCAP, measurement).

    Returns:
        VerifyResult with pass/fail and reason.
    """
    warnings: list[str] = []

    # 1. DCAP attestation — proves the quote is genuine Intel-signed hardware.
    #    It does NOT prove the code was unmodified; check 3 does that.
    if not mock and not verify_dcap(proof.attestation_quote):
        return VerifyResult(False, "DCAP attestation verification failed")

    # 2. Quote parses, and report_data binds the proof body to the hardware.
    #    Enforced in every mode: the mock quote generator embeds report_data
    #    correctly, so tamper detection works on a local testnet too.
    try:
        quote = parse_quote(proof.attestation_quote)
        report_data = extract_report_data(proof.attestation_quote)
    except ValueError as e:
        return VerifyResult(False, f"Invalid attestation quote: {e}")

    content_hash = proof.content_hash()
    expected_report_data = bytes.fromhex(content_hash).ljust(64, b"\x00")[:64]
    if report_data != expected_report_data:
        return VerifyResult(
            False,
            f"report_data mismatch: quote has {report_data.hex()[:32]}..., "
            f"expected {expected_report_data.hex()[:32]}... — the proof body "
            "was altered after it was attested",
        )

    # 3. Approved runtime image, from the hardware's own measurement registers.
    if not mock:
        measured = measurement_id(quote)
        if measured not in approved_measurements:
            return VerifyResult(
                False,
                f"Unapproved runtime image: measurement {measured[:16]}... "
                f"not among {len(approved_measurements)} approved measurements",
            )

    # 4. Nonce — ties the proof to this epoch's unpredictable block hash.
    if proof.nonce != expected_nonce:
        return VerifyResult(
            False,
            f"Nonce mismatch: proof has {proof.nonce[:16]}..., "
            f"expected {expected_nonce[:16]}...",
        )

    # 5. Questions hash.
    if proof.questions_hash != expected_questions_hash:
        return VerifyResult(
            False,
            f"Questions hash mismatch: proof has {proof.questions_hash[:16]}..., "
            f"expected {expected_questions_hash[:16]}...",
        )

    # 6. The answers must be to the questions we actually assigned.
    #    questions_hash alone proves nothing here: it is a public value, so a
    #    miner can copy it while grading an easier set entirely of its own
    #    choosing. Only comparing the result ids against the slice closes that.
    if expected_question_ids is not None:
        actual_ids = {r.question_id for r in proof.scored_results}
        if actual_ids != expected_question_ids:
            missing = expected_question_ids - actual_ids
            extra = actual_ids - expected_question_ids
            return VerifyResult(
                False,
                f"Result set does not match the assigned slice: "
                f"{len(missing)} missing, {len(extra)} unassigned "
                f"(e.g. {sorted(extra)[:3] or sorted(missing)[:3]})",
            )
        if len(proof.scored_results) != len(expected_question_ids):
            return VerifyResult(
                False,
                f"Duplicate results: {len(proof.scored_results)} entries for "
                f"{len(expected_question_ids)} questions",
            )

    # 6b. The exploration set must be exactly what the nonce assigned.
    #     Exploration costs the miner money and earns it nothing directly, so
    #     the only thing making it happen is that an incomplete or re-targeted
    #     set is a rejected proof. Both sides derive the assignment from public
    #     inputs, so this never relies on the miner's account of it.
    if expected_exploration is not None:
        explored = {r.question_id: r.routed_model for r in proof.exploration_results}
        if set(explored) != set(expected_exploration):
            return VerifyResult(
                False,
                f"Exploration set mismatch: {len(expected_exploration)} questions "
                f"assigned, {len(explored)} answered",
            )
        wrong = [
            qid for qid, model in expected_exploration.items()
            if explored.get(qid) != model
        ]
        if wrong:
            return VerifyResult(
                False,
                f"{len(wrong)} exploration questions routed to a model other than "
                f"the one the nonce assigned (e.g. {wrong[0]}: got "
                f"{explored.get(wrong[0])!r}, required "
                f"{expected_exploration[wrong[0]]!r})",
            )

    # 7. The head that ran must be the head committed on-chain before the nonce
    #    was knowable — otherwise a miner commits one head and runs another,
    #    and the evidence accumulator is keyed on a value it controls freely.
    if expected_weights_hash and proof.weights_hash != expected_weights_hash:
        return VerifyResult(
            False,
            f"Head mismatch: proof ran weights {proof.weights_hash[:16]}..., "
            f"on-chain commitment is {expected_weights_hash[:16]}...",
        )
    if head_bytes is not None:
        actual = hashlib.sha256(head_bytes).hexdigest()
        if actual != proof.weights_hash:
            return VerifyResult(
                False,
                f"Bundled head does not match the attested weights_hash: "
                f"head is {actual[:16]}..., proof claims {proof.weights_hash[:16]}...",
            )

    # 8. The bundle we downloaded must be the one the miner advertised.
    if expected_proof_hash and content_hash != expected_proof_hash:
        return VerifyResult(
            False,
            f"Bundle mismatch: downloaded proof hashes to {content_hash[:16]}..., "
            f"axon advertised {expected_proof_hash[:16]}...",
        )

    # 9. Every graded question must be one we have gold for, or it cannot have
    #    been graded against anything.
    unknown = [r.question_id for r in proof.results if r.question_id not in gold_answers]
    if unknown:
        return VerifyResult(
            False,
            f"{len(unknown)} results reference questions with no gold answer "
            f"(e.g. {unknown[:3]})",
        )

    # 10. Costs must be internally consistent. Under real attestation this can
    #     never legitimately fail — the metering proxy produced every figure
    #     inside the same measured image — so an inconsistency means the proof
    #     is not what it claims to be, not that a number drifted.
    per_q_sum = sum(r.cost_usd for r in proof.results)
    if not _costs_consistent(per_q_sum, proof.total_cost_usd):
        return VerifyResult(
            False,
            f"Cost inconsistency: per-question sum ${per_q_sum:.4f} vs "
            f"attested total ${proof.total_cost_usd:.4f}",
        )

    per_model_sum = sum(proof.per_model_costs.values())
    if not _costs_consistent(per_model_sum, proof.total_cost_usd):
        return VerifyResult(
            False,
            f"Cost inconsistency: per-model sum ${per_model_sum:.4f} vs "
            f"attested total ${proof.total_cost_usd:.4f}",
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
