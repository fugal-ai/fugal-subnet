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
    CollateralUnavailable,
    expected_rtmr3,
    extract_report_data,
    measurement_id,
    parse_quote,
    replay_event_log,
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
    # "This validator could not check the proof", as distinct from "the proof
    # is bad". Defaults False so every existing construction keeps the meaning
    # it has today and only a deliberate one opts in — a third truthy state
    # someone can forget to branch on would be worse than no distinction.
    #
    # NEVER set this for anything a miner's bytes could have caused. A miner who
    # can reach this branch becomes UNSCOREABLE AT WILL: neither rewarded nor
    # punished, absent from the invalid count, and invisible to the attack
    # suite. That is an I4 problem wearing an I6 costume, and it is why the only
    # things that set it are the operator's own infrastructure.
    unverifiable: bool = False


def parse_approved(entries) -> dict[str, set[str]]:
    """Turn the approved list into {base_measurement: {app_identity, ...}}.

    An entry is either

        <base>                 the image is approved, nothing is required of RTMR3
        <base>:<app_identity>  and the runtime identity must also match

    Two halves rather than one hash, because they have different lifecycles and
    different approvers. The base rotates when the image or kernel changes,
    which is the cloud provider's schedule; the app identity rotates when our
    code, pool or grader changes, which is ours. Hashing them together would
    force a single rotation for either event and make "what code is approved"
    unreadable. Kept apart, the app identity is computable off-hardware from the
    repo, so it is reviewable in a pull request instead of requiring someone to
    hold a quote.

    A bare base is the pre-existing behaviour and stays valid: on an unlocked
    image an RTMR3 match proves nothing anyway, since an attacker running
    modified code extends whatever value is expected. Requiring it becomes
    meaningful when the extend is performed from a measured initrd.

    THE APP IDENTITY IS NEVER AN RTMR3 VALUE, and storing one here would be a
    subtle, expensive mistake. A measured image extends RTMR3 several times
    during boot and one of those events is `instance-id`, which changes on every
    deploy — so two honest miners running the identical approved application
    produce different RTMR3 values, as does the same miner after a redeploy.
    Measured on two real dstack deploys of the same compose.

    What is approved is the compose hash carried in an event PAYLOAD. The replay
    authenticates the log against the register the CPU signed; the payload is the
    thing compared. An approved list holding RTMR3 values would pass its first
    test and reject every honest miner from their second deploy onward, and the
    symptom would look like an attack rather than a design error.
    """
    out: dict[str, set[str]] = {}
    for raw in entries:
        entry = str(raw).strip()
        if not entry:
            continue
        base, sep, app = entry.partition(":")
        base, app = base.strip(), app.strip()
        # A colon with nothing after it is a typo, not an intention, and
        # interpreting it as the weaker form silently disables the app binding
        # while the operator believes they configured one. That is the exact
        # failure this codebase keeps finding: a security check that reports
        # success and does nothing. An unexpanded shell variable or a trailing
        # copy-paste is all it takes, so it fails loudly instead.
        if sep and not app:
            raise ValueError(
                f"Approved measurement {entry!r} ends in a colon with no runtime "
                "identity. Write '<base>' for an image-only entry, or "
                "'<base>:<app_identity>' to require one — never a bare colon, "
                "which would silently accept any runtime."
            )
        if not base:
            raise ValueError(f"Approved measurement {entry!r} has no base measurement")
        out.setdefault(base, set())
        if app:
            out[base].add(app)
    return out


# A bare TDX quote's first field is a u16 little-endian version. Anything else
# at offset 0 is an envelope, not a quote.
_TDX_QUOTE_VERSIONS = (4, 5)


def unwrap_attestation(blob: bytes) -> tuple[bytes, list | None]:
    """Return (tdx_quote, event_log) for either a bare quote or a dstack blob.

    A miner running under dstack sends the whole attestation — TDX quote, event
    log, and on GCP a TPM quote — not a bare quote. Both shapes have to verify,
    and which one arrived must be decided from the bytes rather than from
    configuration, because two validators configured differently would disagree
    about the same proof.

    **The TPM half is verified here, not optionally later.** If the envelope
    carries one and it does not verify, the proof is rejected: a validator that
    checked only the TDX half would accept proofs another validator rejects,
    which is a fork with no bug behind it. That is also why an absent
    `cryptography` raises rather than returning invalid — a missing library is
    the operator's misconfiguration, and downgrading it to "this proof is bad"
    would let a --live validator reject the whole field while looking healthy.
    Identical reasoning to the ImportError handling around `verify_dcap`.
    """
    if len(blob) >= 2 and int.from_bytes(blob[:2], "little") in _TDX_QUOTE_VERSIONS:
        return blob, None

    from fugal_subnet.scale import ScaleError, decode_dstack_attestation

    try:
        att = decode_dstack_attestation(blob)
    except ScaleError as e:
        raise ValueError(f"attestation is neither a TDX quote nor a dstack blob: {e}") from e

    from fugal_subnet.tee.tpm import TpmError, verify_tpm_quote

    try:
        ok = verify_tpm_quote(att["tpm"])
    except TpmError as e:
        raise ValueError(f"TPM quote rejected: {e}") from e
    if not ok:
        raise ValueError(
            "TPM quote verification failed — the attestation key certificate "
            "does not chain to Google's pinned EK/AK root, or the quote was not "
            "signed by it"
        )

    return att["quote"], att["events"]


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
    expected_hotkey: str = "",
    expected_proof_hash: str = "",
    head_bytes: bytes | None = None,
    event_log: list | None = None,
    mock: bool = False,
    collateral_budget=None,
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
    from fugal_subnet.routing_protocol import identity
    if proof.routing_protocol != identity():
        return VerifyResult(False, "Incompatible routing protocol evidence namespace")

    # 1. DCAP attestation — proves the quote is genuine Intel-signed hardware.
    #    It does NOT prove the code was unmodified; check 3 does that.
    #
    #    VERIFICATION OF UNTRUSTED INPUT RETURNS A VERDICT, IT DOES NOT THROW.
    #    verify_dcap raises on a quote it cannot parse or whose collateral it
    #    cannot fetch, and those bytes come from a miner. Unguarded, the
    #    exception left verify_proof, left the validator's per-miner loop, and
    #    was caught by the epoch loop's catch-all — abandoning the WHOLE epoch.
    #    Any registered hotkey could therefore halt every validator on the
    #    subnet, every epoch, for the price of one registration, by returning
    #    forty bytes of garbage. That is I6, broken by miner input, and it only
    #    appears under --live, which is why it was never seen.
    #
    #    ImportError is deliberately NOT caught: a missing dcap-qvl is the
    #    operator's misconfiguration, not a miner's doing, and silently
    #    downgrading it to "this proof is invalid" would let a --live validator
    #    reject the entire field while looking like it was working.
    # 0a. WHOSE proof this is. First, because it is the cheapest check that can
    #     reject, it needs no hardware, and a relayed proof should not cost a
    #     DCAP verification to refuse.
    #
    #     Enforced in EVERY mode, like the report_data binding below and for the
    #     same reason: this is a property of the proof's content, not of the
    #     silicon, so a local testnet must catch relay too. Mock mode is the
    #     absence of hardware checks, not a weaker set of rules.
    #
    #     A missing expected_hotkey is a caller error rather than a licence to
    #     skip: verifying without it silently restores the property this exists
    #     to remove. Outside mock that is fatal. Inside mock — used by attack
    #     fixtures and determinism runs that have no metagraph — the check is
    #     simply not requested.
    if not expected_hotkey and not mock:
        raise ValueError(
            "verify_proof called without expected_hotkey outside mock mode. A "
            "proof that is not bound to a miner can be relayed by anyone who "
            "can read it off that miner's axon."
        )
    if expected_hotkey and proof.hotkey != expected_hotkey:
        return VerifyResult(
            False,
            f"Proof is bound to hotkey {proof.hotkey or '(none)'}, but this uid's "
            f"hotkey is {expected_hotkey}. A proof names the miner it was "
            "produced for; this one names somebody else.",
        )

    # 0. Unwrap. Under dstack the miner sends an envelope, not a bare quote, and
    #    the TPM half inside it is verified as part of unwrapping — see
    #    unwrap_attestation for why that is not optional.
    try:
        attestation_quote, embedded_events = unwrap_attestation(proof.attestation_quote)
    except ValueError as e:
        return VerifyResult(False, str(e))
    if event_log is None:
        event_log = embedded_events

    if not mock:
        try:
            dcap_ok = verify_dcap(
                attestation_quote, budget=collateral_budget)
        except ImportError:
            raise
        except CollateralUnavailable as e:
            # The operator's infrastructure, not the miner's proof. Sits here
            # beside the ImportError re-raise because it is the same category:
            # in both, nothing was learned about this miner, and reporting
            # "invalid" would be an accusation the evidence does not support.
            #
            # It does NOT return valid=True. Nothing was verified, so nothing is
            # accepted — this only changes what the failure is called, and what
            # the caller is permitted to conclude from it.
            return VerifyResult(
                False,
                f"DCAP collateral unavailable, so this proof was never judged: {e}",
                unverifiable=True,
            )
        except Exception as e:  # noqa: BLE001 - miner-controlled bytes
            return VerifyResult(
                False, f"DCAP attestation verification failed: {type(e).__name__}: {e}",
            )
        if not dcap_ok:
            return VerifyResult(False, "DCAP attestation verification failed")

    # 2. Quote parses, and report_data binds the proof body to the hardware.
    #    Enforced in every mode: the mock quote generator embeds report_data
    #    correctly, so tamper detection works on a local testnet too.
    try:
        quote = parse_quote(attestation_quote)
        report_data = extract_report_data(attestation_quote)
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
        approved = parse_approved(approved_measurements)
        measured = measurement_id(quote)
        if measured not in approved:
            return VerifyResult(
                False,
                f"Unapproved runtime image: measurement {measured[:16]}... "
                f"not among {len(approved)} approved measurements",
            )

        # 3b. Runtime identity, when the approved entry names one.
        #     RTMR3 is an EXTEND, not a set — the register holds
        #     SHA384(SHA384(...zeros || first) || second)..., never the value
        #     written. So this replays from zero and compares the result to what
        #     the CPU signed, which is also what makes an untrusted extend log
        #     safe to read later: a log that does not reproduce the quote's
        #     register is discarded before any field of it is believed.
        required_apps = approved[measured]
        if required_apps:
            if event_log:
                # A measured image extends RTMR3 several times during boot, so
                # the register is a chain and the log is the only description of
                # it. Replay first, compare to the quote, and only then read a
                # field: a log that does not reproduce what the CPU signed is
                # discarded whole rather than partially believed.
                try:
                    replayed, events = replay_event_log(event_log, imr=3)
                except ValueError as e:
                    return VerifyResult(False, f"Malformed TDX event log: {e}")
                if replayed != quote.rtmr3:
                    return VerifyResult(
                        False,
                        f"Event log does not reproduce RTMR3: replayed "
                        f"{replayed[:16]}... but the quote says "
                        f"{quote.rtmr3[:16]}...",
                    )
                app_seen = events.get("compose-hash", b"").hex()
                if app_seen not in required_apps:
                    return VerifyResult(
                        False,
                        f"Unapproved application: compose-hash {app_seen[:16]}... "
                        f"not among {len(required_apps)} approved for this image",
                    )
            elif any(len(a) == 64 for a in required_apps):
                # A 64-hex identity is a dstack compose hash, and a dstack TD
                # extends RTMR3 NINE times during boot — including `instance-id`,
                # WHICH CHANGES ON EVERY DEPLOY. Two honest miners running the
                # identical approved compose therefore produce DIFFERENT RTMR3
                # values, and so does one miner after a redeploy. Measured on two
                # real deploys: same compose, different RTMR3.
                #
                # So RTMR3 can never be compared to a fixed value for this shape,
                # and without the log there is nothing to replay. Refusing is the
                # only correct answer; falling through to the single-extend
                # comparison below would reject an honest miner on their second
                # deploy and look exactly like an attack.
                return VerifyResult(
                    False,
                    "Approved entry names a compose hash but the proof carries no "
                    "TDX event log. RTMR3 on a measured image is a chain that "
                    "includes per-deploy values, so it can only be checked by "
                    "replaying the log — never by comparison.",
                )
            else:
                # No log: the single-extend case a Fugal miner produces today,
                # where the whole chain is one runtime_identity from a zeroed
                # register. A fresh TD really does start from zero — a GCP
                # confidential VM stop/start yields a new TD with RTMR3 cleared,
                # measured twice — so a miner cannot accumulate extends across
                # restarts to reach a chosen value.
                candidates = {expected_rtmr3(app) for app in required_apps}
                if quote.rtmr3 not in candidates:
                    return VerifyResult(
                        False,
                        f"Runtime identity mismatch: RTMR3 {quote.rtmr3[:16]}... "
                        f"replays from none of the {len(required_apps)} approved "
                        f"runtime identities for this image",
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

    if not mock:
        from fugal_subnet.head_eval import load_head_from_npz
        from fugal_subnet.routing_protocol import benchmark_costs
        try:
            if head_bytes is None:
                raise ValueError("missing success head bytes")
            benchmark_costs(load_head_from_npz(head_bytes))
        except (ValueError, KeyError, TypeError) as e:
            return VerifyResult(False, f"Incompatible success head: {e}")

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
