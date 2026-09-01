"""Canonical v2 reveal artifact and independent deterministic verifier."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from fugal_subnet.consensus_manifest import canonical_json
from fugal_subnet.graders_v2 import grade
from fugal_subnet.head_eval import load_head_from_npz
from fugal_subnet.sandbox.client import GradingClient
from fugal_subnet.v2.commitments import (
    HistoricalCommitmentReceipt,
    verify_historical_receipt,
)
from fugal_subnet.v2.committee import Builder
from fugal_subnet.v2.dedup import find_duplicates
from fugal_subnet.v2.head_eval import evaluate_head
from fugal_subnet.v2.reports import derive_consensus_matrix, verify_report
from fugal_subnet.v2.rewards import (
    MAX_WEIGHT_DELTA,
    WEIGHT_PRECISION,
    compute_bounded_weights,
)
from fugal_subnet.v2.scoring import composite_score
from fugal_subnet.v2.soft_targets import compute_soft_targets

SCHEMA_VERSION = 2
MAX_REVEAL_BYTES = 32 * 1024 * 1024
MAX_HEADS = 4096
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class RevealVerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedReveal:
    epoch_id: str
    question_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    matrix: np.ndarray
    canonical_costs: dict[str, float]
    scores: dict[int, float]
    weights: dict[str, str] | None
    chain_receipts_verified: bool


@dataclass(frozen=True)
class HeadSubmission:
    uid: int
    hotkey: str
    artifact: bytes
    wire_model_pool: tuple[str, ...]
    commitment_receipt: HistoricalCommitmentReceipt


def question_commitment_hash(questions: list[dict], grader_hash: str) -> str:
    if not HASH_RE.fullmatch(grader_hash):
        raise RevealVerificationError("grader hash must be lowercase SHA-256 hex")
    return hashlib.sha256(canonical_json({
        "grader_hash": grader_hash,
        "questions": questions,
        "purpose": "fugal-questions-v2",
    })).hexdigest()


def registry_snapshot_hash(model_ids: list[str], route_costs_usd: Mapping[str, str]) -> str:
    return hashlib.sha256(canonical_json({
        "model_ids": model_ids,
        "route_costs_usd": dict(route_costs_usd),
        "purpose": "fugal-model-registry-snapshot-v2",
    })).hexdigest()


def encode_reveal(value: dict) -> bytes:
    payload = canonical_json(value)
    if len(payload) > MAX_REVEAL_BYTES:
        raise RevealVerificationError("reveal exceeds artifact size bound")
    return payload


def _decode_reveal(payload: bytes) -> dict:
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_REVEAL_BYTES:
        raise RevealVerificationError("reveal artifact size is invalid")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RevealVerificationError("reveal is not UTF-8 JSON") from exc
    if canonical_json(value) != payload:
        raise RevealVerificationError("reveal is not canonical JSON")
    if not isinstance(value, dict):
        raise RevealVerificationError("reveal root must be an object")
    return value


def _decimal(value: object, label: str, *, nonnegative: bool = True) -> Decimal:
    if not isinstance(value, str):
        raise RevealVerificationError(f"{label} must be a decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise RevealVerificationError(f"{label} is invalid") from exc
    if not result.is_finite() or (nonnegative and result < 0) or str(result) != value:
        raise RevealVerificationError(f"{label} is not a canonical finite decimal")
    return result


def _b64(value: object, label: str, maximum: int, *, allow_empty: bool = False) -> bytes:
    if not isinstance(value, str):
        raise RevealVerificationError(f"{label} must be base64 text")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RevealVerificationError(f"{label} is invalid base64") from exc
    if (not raw and not allow_empty) or len(raw) > maximum:
        raise RevealVerificationError(f"{label} decoded size is invalid")
    return raw


def _receipt(
    raw: object,
    *,
    namespace: str,
    epoch_id: str | None,
    artifact_hash: str,
    hotkey: str,
    chain_resolver: Callable[[HistoricalCommitmentReceipt], str] | None,
) -> HistoricalCommitmentReceipt:
    try:
        receipt = HistoricalCommitmentReceipt.from_dict(raw)
    except Exception as exc:
        raise RevealVerificationError(f"invalid {namespace} commitment receipt") from exc
    if (
        receipt.namespace != namespace
        or (epoch_id is not None and receipt.epoch_id != epoch_id)
        or receipt.artifact_hash != artifact_hash
        or receipt.hotkey != hotkey
    ):
        raise RevealVerificationError(f"{namespace} receipt does not bind expected material")
    if chain_resolver is not None and chain_resolver(receipt) != receipt.payload:
        raise RevealVerificationError(f"{namespace} receipt differs from historical chain state")
    return receipt


def _parse_registry(raw: object) -> tuple[list[str], dict[str, float], str]:
    if not isinstance(raw, dict) or set(raw) != {
        "model_ids", "route_costs_usd", "snapshot_hash"
    }:
        raise RevealVerificationError("reveal registry schema differs")
    model_ids = raw["model_ids"]
    costs = raw["route_costs_usd"]
    if (
        not isinstance(model_ids, list)
        or not 1 <= len(model_ids) <= 8
        or len(model_ids) != len(set(model_ids))
        or any(not isinstance(model, str) or not model for model in model_ids)
    ):
        raise RevealVerificationError("reveal model order is invalid")
    if not isinstance(costs, dict) or set(costs) != set(model_ids):
        raise RevealVerificationError("reveal route costs do not match model order")
    parsed = {model: float(_decimal(costs[model], f"cost[{model}]")) for model in model_ids}
    expected_hash = registry_snapshot_hash(model_ids, costs)
    if raw["snapshot_hash"] != expected_hash:
        raise RevealVerificationError("registry snapshot hash mismatch")
    return model_ids, parsed, expected_hash


def _evaluation_evidence(score) -> dict:
    return {
        "accuracy": str(Decimal(str(score.accuracy))),
        "cost_efficiency": str(Decimal(str(score.cost_efficiency))),
        "kl_score": str(Decimal(str(score.kl_score))),
        "routing_decisions": [int(value) for value in score.routing_decisions],
        "routing_model_ids": list(score.routing_model_ids),
        "routing_distributions": [
            [str(Decimal(str(value))) for value in row]
            for row in score.routing_distributions.tolist()
        ],
        "correct_mask": [bool(value) for value in score.correct_mask],
    }


def _canonical_decimal(value: int | float | str | Decimal, label: str) -> str:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RevealVerificationError(f"{label} is invalid") from exc
    if not parsed.is_finite() or parsed < 0:
        raise RevealVerificationError(f"{label} is invalid")
    return str(parsed)


def build_reveal(
    *,
    epoch: Mapping[str, object],
    committee: Sequence[Builder],
    questions: list[dict],
    route_costs_usd: Mapping[str, int | float | str | Decimal],
    question_receipts: Sequence[HistoricalCommitmentReceipt],
    report_receipts: Sequence[HistoricalCommitmentReceipt],
    builder_reports: Mapping[str, bytes],
    head_submissions: Sequence[HeadSubmission],
    grading_client: GradingClient | None,
    embedder: Callable[[list[str]], np.ndarray],
    previous_weights: Mapping[int, int | float | str | Decimal],
    eligible_uids: set[int],
    forced_zero_uids: set[int],
    max_delta: int | float | str | Decimal = MAX_WEIGHT_DELTA,
    precision: int = WEIGHT_PRECISION,
) -> tuple[bytes, VerifiedReveal]:
    """Build all public v2 evidence, then independently verify it before use."""
    epoch_value = dict(epoch)
    expected_epoch_keys = {
        "epoch_id", "boundary_block", "boundary_hash",
        "precommit_deadline_block", "report_deadline_block", "manifest_hash",
        "grader_hash",
    }
    if set(epoch_value) != expected_epoch_keys:
        raise RevealVerificationError("reveal builder epoch schema differs")
    epoch_id = epoch_value["epoch_id"]
    boundary_block = epoch_value["boundary_block"]
    precommit_deadline = epoch_value["precommit_deadline_block"]
    deadline = epoch_value["report_deadline_block"]
    if (
        not isinstance(epoch_id, str)
        or not isinstance(boundary_block, int)
        or not isinstance(precommit_deadline, int)
        or not isinstance(deadline, int)
        or not boundary_block < precommit_deadline < deadline
    ):
        raise RevealVerificationError("reveal builder epoch identity is invalid")

    model_ids = list(route_costs_usd)
    costs_text = {
        model: _canonical_decimal(route_costs_usd[model], f"cost[{model}]")
        for model in model_ids
    }
    registry_hash = registry_snapshot_hash(model_ids, costs_text)
    question_hash = question_commitment_hash(questions, str(epoch_value["grader_hash"]))

    committee_hotkeys = {builder.hotkey for builder in committee}
    question_blocks: dict[str, int] = {}
    for receipt in question_receipts:
        verify_historical_receipt(receipt)
        if (
            receipt.epoch_id != epoch_id
            or receipt.namespace != "questions"
            or receipt.artifact_hash != question_hash
            or receipt.hotkey not in committee_hotkeys
            or receipt.hotkey in question_blocks
            or not boundary_block <= receipt.block <= precommit_deadline
        ):
            raise RevealVerificationError("reveal builder question receipts differ")
        question_blocks[receipt.hotkey] = receipt.block
    report_hashes: dict[str, str] = {}
    for receipt in report_receipts:
        verify_historical_receipt(receipt)
        if (
            receipt.epoch_id != epoch_id
            or receipt.namespace != "report"
            or receipt.hotkey not in question_blocks
            or receipt.hotkey in report_hashes
            or not question_blocks[receipt.hotkey] < receipt.block <= deadline
        ):
            raise RevealVerificationError("reveal builder report receipts differ")
        report_hashes[receipt.hotkey] = receipt.artifact_hash
    consensus = derive_consensus_matrix(
        committee=committee,
        committed_hashes=report_hashes,
        artifacts=builder_reports,
    )
    if list(consensus.question_ids) != [item.get("question_id") for item in questions]:
        raise RevealVerificationError("report question order differs from canonical slice")
    if list(consensus.model_ids) != model_ids:
        raise RevealVerificationError("report model order differs from canonical registry")

    prompts = [item.get("prompt") for item in questions]
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise RevealVerificationError("every canonical question needs a prompt")
    hidden = np.asarray(embedder([str(prompt) for prompt in prompts]), dtype=np.float32)
    matrix = np.asarray(consensus.matrix, dtype=np.int8)
    soft_targets = compute_soft_targets(matrix)
    costs_float = {model: float(Decimal(costs_text[model])) for model in model_ids}

    accepted = []
    rejected = []
    decisions: dict[int, np.ndarray] = {}
    distributions: dict[int, np.ndarray] = {}
    hotkeys: dict[int, str] = {}
    commit_blocks: dict[int, int | float] = {}
    scores: dict[int, float] = {}
    rejected_submission_uids: set[int] = set()
    seen_uids: set[int] = set()
    seen_hotkeys: set[str] = set()
    for submission in sorted(head_submissions, key=lambda item: item.uid):
        uid, hotkey = submission.uid, submission.hotkey
        if (
            not isinstance(uid, int)
            or isinstance(uid, bool)
            or uid < 0
            or uid in seen_uids
            or not isinstance(hotkey, str)
            or not hotkey
            or hotkey in seen_hotkeys
        ):
            raise RevealVerificationError("head submission identities are invalid or duplicated")
        seen_uids.add(uid)
        seen_hotkeys.add(hotkey)
        receipt = submission.commitment_receipt
        verify_historical_receipt(receipt)
        artifact_hash = hashlib.sha256(submission.artifact).hexdigest()
        if (
            receipt.namespace != "head"
            or receipt.uid != uid
            or receipt.hotkey != hotkey
            or receipt.artifact_hash != artifact_hash
        ):
            raise RevealVerificationError("head receipt does not bind submitted bytes/identity")
        evidence = {
            "uid": uid,
            "hotkey": hotkey,
            "artifact_b64": base64.b64encode(submission.artifact).decode("ascii"),
            "artifact_sha256": artifact_hash,
            "wire_model_pool": list(submission.wire_model_pool),
            "commitment_receipt": asdict(receipt),
        }
        if receipt.block > boundary_block:
            evidence["rejection_code"] = "commitment_after_boundary"
            rejected.append(evidence)
            rejected_submission_uids.add(uid)
            continue
        try:
            head = load_head_from_npz(submission.artifact)
        except Exception:
            evidence["rejection_code"] = "artifact_invalid"
            rejected.append(evidence)
            rejected_submission_uids.add(uid)
            continue
        head.commit_hash = artifact_hash
        try:
            evaluation = evaluate_head(
                head,
                hidden,
                matrix,
                model_ids,
                soft_targets,
                costs_float,
                wire_model_pool=list(submission.wire_model_pool),
            )
        except ValueError:
            evidence["rejection_code"] = "models_invalid"
            rejected.append(evidence)
            rejected_submission_uids.add(uid)
            continue
        evidence["evaluation"] = _evaluation_evidence(evaluation)
        accepted.append(evidence)
        decisions[uid] = evaluation.routing_decisions
        distributions[uid] = evaluation.routing_distributions
        hotkeys[uid] = hotkey
        commit_blocks[uid] = receipt.block
        scores[uid] = composite_score(
            evaluation.accuracy, evaluation.cost_efficiency, evaluation.kl_score
        )

    dedup = find_duplicates(decisions, distributions, commit_blocks, hotkeys)
    forced = (
        set(forced_zero_uids)
        | set(dedup.disqualified)
        | rejected_submission_uids
    )
    max_delta_text = _canonical_decimal(max_delta, "max_delta")
    previous_text = {
        str(uid): _canonical_decimal(value, f"previous_weights[{uid}]")
        for uid, value in sorted(previous_weights.items())
    }
    weight_result = compute_bounded_weights(
        scores,
        previous_weights,
        set(eligible_uids),
        forced,
        max_delta=Decimal(max_delta_text),
        precision=precision,
    )
    reveal = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "v2",
        "epoch": epoch_value,
        "committee": [asdict(builder) for builder in committee],
        "questions": questions,
        "registry": {
            "model_ids": model_ids,
            "route_costs_usd": costs_text,
            "snapshot_hash": registry_hash,
        },
        "question_receipts": [asdict(item) for item in question_receipts],
        "report_receipts": [asdict(item) for item in report_receipts],
        "builder_reports": {
            hotkey: base64.b64encode(payload).decode("ascii")
            for hotkey, payload in sorted(builder_reports.items())
        },
        "consensus": {
            "question_ids": list(consensus.question_ids),
            "model_ids": list(consensus.model_ids),
            "matrix": [list(row) for row in consensus.matrix],
            "builder_hotkeys": list(consensus.builder_hotkeys),
            "report_hashes": list(consensus.report_hashes),
        },
        "heads": accepted,
        "rejected_heads": rejected,
        "dedup": {
            "clusters": [list(cluster) for cluster in dedup.clusters],
            "disqualified": sorted(dedup.disqualified),
        },
        "scores": {
            str(uid): str(Decimal(str(scores[uid]))) for uid in sorted(scores)
        },
        "weight_inputs": {
            "previous_weights": previous_text,
            "eligible_uids": sorted(eligible_uids),
            "forced_zero_uids": sorted(forced),
            "max_delta": max_delta_text,
            "precision": precision,
        },
        "weights": None if weight_result is None else weight_result.serialized(),
        "set_weights": weight_result is not None,
    }
    payload = encode_reveal(reveal)
    verified = verify_reveal(
        payload,
        grading_client=grading_client,
        embedder=lambda _prompts: hidden,
    )
    if verified.weights != reveal["weights"]:
        raise RevealVerificationError("reveal builder self-verification differs")
    return payload, verified


def _pending_reveal_path(destination: str | Path) -> Path:
    path = Path(destination)
    return path.with_name(f"{path.name}.pending")


def stage_reveal(payload: bytes, destination: str | Path) -> Path:
    """Durably stage verified bytes before an externally visible weight change."""
    _decode_reveal(payload)
    path = Path(destination)
    pending = _pending_reveal_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or pending.is_symlink():
        raise RevealVerificationError("reveal paths cannot be symbolic links")
    if path.exists():
        if path.read_bytes() != payload:
            raise RevealVerificationError("published reveal path already has other bytes")
        return path
    if pending.exists():
        if pending.read_bytes() != payload:
            raise RevealVerificationError("staged reveal path already has other bytes")
        return pending
    temporary = pending.with_name(f".{pending.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o600)
    temporary.replace(pending)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return pending


def finalize_reveal(payload: bytes, destination: str | Path) -> Path:
    """Publish only the exact reveal staged before weight submission."""
    _decode_reveal(payload)
    path = Path(destination)
    pending = _pending_reveal_path(path)
    if path.is_symlink() or pending.is_symlink():
        raise RevealVerificationError("reveal paths cannot be symbolic links")
    if path.exists():
        if path.read_bytes() != payload:
            raise RevealVerificationError("published reveal path already has other bytes")
        return path
    if not pending.exists() or pending.read_bytes() != payload:
        raise RevealVerificationError("exact reveal bytes were not staged")
    pending.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


def publish_reveal(payload: bytes, destination: str | Path) -> Path:
    """Atomically stage and publish immutable reveal bytes."""
    stage_reveal(payload, destination)
    return finalize_reveal(payload, destination)


def verify_reveal(
    payload: bytes,
    *,
    grading_client: GradingClient | None,
    embedder: Callable[[list[str]], np.ndarray],
    chain_resolver: Callable[[HistoricalCommitmentReceipt], str] | None = None,
) -> VerifiedReveal:
    """Recompute all reveal-derived consensus values from published artifacts.

    ``chain_resolver`` must query or independently prove state at the receipt's
    exact block. Without it, receipt payloads are checked internally but the
    result explicitly reports ``chain_receipts_verified=False``.
    """
    reveal = _decode_reveal(payload)
    root_keys = {
        "schema_version", "protocol", "epoch", "committee", "questions",
        "registry", "question_receipts", "report_receipts", "builder_reports",
        "consensus", "heads", "rejected_heads", "dedup", "scores",
        "weight_inputs", "weights", "set_weights",
    }
    if set(reveal) != root_keys or reveal["schema_version"] != SCHEMA_VERSION or reveal["protocol"] != "v2":
        raise RevealVerificationError("reveal root schema or protocol differs")

    epoch = reveal["epoch"]
    if not isinstance(epoch, dict) or set(epoch) != {
        "epoch_id", "boundary_block", "boundary_hash",
        "precommit_deadline_block", "report_deadline_block", "manifest_hash",
        "grader_hash",
    }:
        raise RevealVerificationError("reveal epoch schema differs")
    epoch_id = epoch["epoch_id"]
    if not isinstance(epoch_id, str) or not 1 <= len(epoch_id) <= 64:
        raise RevealVerificationError("epoch id is invalid")
    for label in ("boundary_hash", "manifest_hash", "grader_hash"):
        if not isinstance(epoch[label], str) or not HASH_RE.fullmatch(epoch[label]):
            raise RevealVerificationError(f"epoch {label} is invalid")
    boundary_block = epoch["boundary_block"]
    precommit_deadline = epoch["precommit_deadline_block"]
    deadline = epoch["report_deadline_block"]
    if (
        not isinstance(boundary_block, int)
        or isinstance(boundary_block, bool)
        or not isinstance(precommit_deadline, int)
        or isinstance(precommit_deadline, bool)
        or not isinstance(deadline, int)
        or isinstance(deadline, bool)
        or not 0 <= boundary_block < precommit_deadline < deadline
    ):
        raise RevealVerificationError("epoch block bounds are invalid")

    committee_raw = reveal["committee"]
    if not isinstance(committee_raw, list) or not 3 <= len(committee_raw) <= 5:
        raise RevealVerificationError("reveal committee size is invalid")
    committee = []
    for item in committee_raw:
        if not isinstance(item, dict) or set(item) != {"uid", "hotkey"}:
            raise RevealVerificationError("committee entry schema differs")
        committee.append(Builder(uid=item["uid"], hotkey=item["hotkey"]))
    hotkeys = [builder.hotkey for builder in committee]
    if len(hotkeys) != len(set(hotkeys)):
        raise RevealVerificationError("committee hotkeys are duplicated")

    questions = reveal["questions"]
    if not isinstance(questions, list) or not questions:
        raise RevealVerificationError("reveal questions are invalid")
    raw_question_ids = [
        question.get("question_id") if isinstance(question, dict) else None
        for question in questions
    ]
    if any(not isinstance(value, str) or not value for value in raw_question_ids):
        raise RevealVerificationError("reveal question IDs are invalid or duplicated")
    question_ids: list[str] = [value for value in raw_question_ids if isinstance(value, str)]
    if len(question_ids) != len(set(question_ids)):
        raise RevealVerificationError("reveal question IDs are invalid or duplicated")
    question_hash = question_commitment_hash(questions, epoch["grader_hash"])

    model_ids, costs, registry_hash = _parse_registry(reveal["registry"])

    question_receipts = reveal["question_receipts"]
    if not isinstance(question_receipts, list) or not 3 <= len(question_receipts) <= len(committee):
        raise RevealVerificationError("three to five committee question receipts are required")
    question_blocks: dict[str, int] = {}
    for raw in question_receipts:
        if not isinstance(raw, dict):
            raise RevealVerificationError("question receipt is invalid")
        hotkey = raw.get("hotkey")
        if not isinstance(hotkey, str):
            raise RevealVerificationError("question receipt hotkey is invalid")
        receipt = _receipt(
            raw, namespace="questions", epoch_id=epoch_id,
            artifact_hash=question_hash, hotkey=hotkey, chain_resolver=chain_resolver,
        )
        if (
            hotkey not in hotkeys
            or not boundary_block <= receipt.block <= precommit_deadline
        ):
            raise RevealVerificationError("question receipt builder/block is invalid")
        if hotkey in question_blocks:
            raise RevealVerificationError("question receipt builder is duplicated")
        question_blocks[hotkey] = receipt.block

    reports_raw = reveal["builder_reports"]
    receipts_raw = reveal["report_receipts"]
    if not isinstance(reports_raw, dict) or not isinstance(receipts_raw, list):
        raise RevealVerificationError("report artifact/receipt schema differs")
    report_artifacts = {
        hotkey: _b64(encoded, f"builder report {hotkey}", 16 * 1024 * 1024)
        for hotkey, encoded in reports_raw.items()
    }
    committed_hashes: dict[str, str] = {}
    for raw in receipts_raw:
        if not isinstance(raw, dict):
            raise RevealVerificationError("report receipt is invalid")
        hotkey = raw.get("hotkey")
        artifact_hash = raw.get("artifact_hash")
        if not isinstance(hotkey, str) or not isinstance(artifact_hash, str):
            raise RevealVerificationError("report receipt identity/hash is invalid")
        receipt = _receipt(
            raw, namespace="report", epoch_id=epoch_id,
            artifact_hash=artifact_hash, hotkey=hotkey, chain_resolver=chain_resolver,
        )
        if (
            hotkey not in question_blocks
            or not question_blocks[hotkey] < receipt.block <= deadline
        ):
            raise RevealVerificationError("report receipt builder/block is invalid")
        if hotkey in committed_hashes:
            raise RevealVerificationError("report receipt builder is duplicated")
        committed_hashes[hotkey] = artifact_hash
    consensus = derive_consensus_matrix(
        committee=committee,
        committed_hashes=committed_hashes,
        artifacts=report_artifacts,
    )
    consensus_raw = reveal["consensus"]
    expected_consensus = {
        "question_ids": list(consensus.question_ids),
        "model_ids": list(consensus.model_ids),
        "matrix": [list(row) for row in consensus.matrix],
        "builder_hotkeys": list(consensus.builder_hotkeys),
        "report_hashes": list(consensus.report_hashes),
    }
    if consensus_raw != expected_consensus:
        raise RevealVerificationError("published consensus matrix/evidence differs")
    if list(consensus.question_ids) != question_ids or list(consensus.model_ids) != model_ids:
        raise RevealVerificationError("consensus order differs from questions/registry")

    question_by_id = dict(zip(question_ids, questions))
    for hotkey, report_payload in report_artifacts.items():
        core = verify_report(report_payload, expected_hotkey=hotkey)["core"]
        if core["question_commitment"] != question_hash or core["registry_hash"] != registry_hash:
            raise RevealVerificationError("builder report commitment differs from reveal")
        for cell in core["cells"]:
            expected_grade = grade(
                question_by_id[cell["question_id"]], cell["response"], grading_client
            )
            if expected_grade != cell["grade"]:
                raise RevealVerificationError("builder response regrading differs")

    matrix: np.ndarray = np.asarray(consensus.matrix, dtype=np.int8)
    soft_targets = compute_soft_targets(matrix)
    raw_prompts = [question.get("prompt") for question in questions]
    if any(not isinstance(prompt, str) for prompt in raw_prompts):
        raise RevealVerificationError("every revealed question needs a prompt")
    prompts: list[str] = [prompt for prompt in raw_prompts if isinstance(prompt, str)]
    hidden = np.asarray(embedder(prompts), dtype=np.float32)

    heads = reveal["heads"]
    rejected_heads = reveal["rejected_heads"]
    if (
        not isinstance(heads, list)
        or not isinstance(rejected_heads, list)
        or len(heads) + len(rejected_heads) > MAX_HEADS
    ):
        raise RevealVerificationError("reveal heads are invalid or oversized")
    decisions: dict[int, np.ndarray] = {}
    distributions: dict[int, np.ndarray] = {}
    hotkey_by_uid: dict[int, str] = {}
    commit_blocks: dict[int, int | float] = {}
    computed_scores: dict[int, float] = {}
    rejected_uids: set[int] = set()
    rejected_hotkeys: set[str] = set()
    for item in rejected_heads:
        keys = {
            "uid", "hotkey", "artifact_b64", "artifact_sha256", "wire_model_pool",
            "commitment_receipt", "rejection_code",
        }
        if not isinstance(item, dict) or set(item) != keys:
            raise RevealVerificationError("rejected head entry schema differs")
        uid, hotkey = item["uid"], item["hotkey"]
        if (
            not isinstance(uid, int)
            or isinstance(uid, bool)
            or uid < 0
            or uid in rejected_uids
            or not isinstance(hotkey, str)
            or not hotkey
            or hotkey in rejected_hotkeys
        ):
            raise RevealVerificationError("rejected head identity is invalid or duplicated")
        wire_pool = item["wire_model_pool"]
        if (
            not isinstance(wire_pool, list)
            or len(wire_pool) > 8
            or any(not isinstance(model, str) for model in wire_pool)
        ):
            raise RevealVerificationError("rejected head wire model pool is invalid")
        artifact = _b64(
            item["artifact_b64"], f"rejected head {uid}", 1024 * 1024,
            allow_empty=True,
        )
        artifact_hash = hashlib.sha256(artifact).hexdigest()
        if item["artifact_sha256"] != artifact_hash:
            raise RevealVerificationError("rejected head artifact hash mismatch")
        receipt = _receipt(
            item["commitment_receipt"], namespace="head", epoch_id=None,
            artifact_hash=artifact_hash, hotkey=hotkey, chain_resolver=chain_resolver,
        )
        code = item["rejection_code"]
        if code == "commitment_after_boundary":
            if receipt.block <= boundary_block:
                raise RevealVerificationError("rejected head commitment was not late")
        elif code == "artifact_invalid":
            if receipt.block > boundary_block:
                raise RevealVerificationError("rejected head has multiple rejection causes")
            try:
                load_head_from_npz(artifact)
            except Exception:
                pass
            else:
                raise RevealVerificationError("rejected head artifact is actually valid")
        elif code == "models_invalid":
            if receipt.block > boundary_block:
                raise RevealVerificationError("rejected head has multiple rejection causes")
            try:
                rejected = load_head_from_npz(artifact)
            except Exception as exc:
                raise RevealVerificationError(
                    "models-invalid rejection contains an invalid artifact"
                ) from exc
            try:
                evaluate_head(
                    rejected, hidden, matrix, model_ids, soft_targets, costs,
                    wire_model_pool=wire_pool,
                )
            except ValueError:
                pass
            else:
                raise RevealVerificationError("rejected head models are actually valid")
        else:
            raise RevealVerificationError("rejected head reason is unsupported")
        rejected_uids.add(uid)
        rejected_hotkeys.add(hotkey)

    for item in heads:
        keys = {
            "uid", "hotkey", "artifact_b64", "artifact_sha256", "wire_model_pool",
            "commitment_receipt", "evaluation",
        }
        if not isinstance(item, dict) or set(item) != keys:
            raise RevealVerificationError("head entry schema differs")
        uid, hotkey = item["uid"], item["hotkey"]
        if (
            not isinstance(uid, int)
            or isinstance(uid, bool)
            or uid < 0
            or uid in decisions
            or uid in rejected_uids
        ):
            raise RevealVerificationError("head UID is invalid or duplicated")
        if (
            not isinstance(hotkey, str)
            or not hotkey
            or hotkey in hotkey_by_uid.values()
            or hotkey in rejected_hotkeys
        ):
            raise RevealVerificationError("head hotkey is invalid or duplicated")
        artifact = _b64(item["artifact_b64"], f"head {uid}", 1024 * 1024)
        artifact_hash = hashlib.sha256(artifact).hexdigest()
        if item["artifact_sha256"] != artifact_hash:
            raise RevealVerificationError("head artifact hash mismatch")
        receipt = _receipt(
            item["commitment_receipt"], namespace="head", epoch_id=None,
            artifact_hash=artifact_hash, hotkey=hotkey, chain_resolver=chain_resolver,
        )
        if receipt.block > boundary_block:
            raise RevealVerificationError("head was committed after the epoch boundary")
        head = load_head_from_npz(artifact)
        head.commit_hash = artifact_hash
        score = evaluate_head(
            head, hidden, matrix, model_ids, soft_targets, costs,
            wire_model_pool=item["wire_model_pool"],
        )
        if item["evaluation"] != _evaluation_evidence(score):
            raise RevealVerificationError("published head evaluation differs")
        decisions[uid] = score.routing_decisions
        distributions[uid] = score.routing_distributions
        hotkey_by_uid[uid] = hotkey
        commit_blocks[uid] = receipt.block
        computed_scores[uid] = composite_score(
            score.accuracy, score.cost_efficiency, score.kl_score
        )

    dedup = find_duplicates(decisions, distributions, commit_blocks, hotkey_by_uid)
    expected_dedup = {
        "clusters": [list(cluster) for cluster in dedup.clusters],
        "disqualified": sorted(dedup.disqualified),
    }
    if reveal["dedup"] != expected_dedup:
        raise RevealVerificationError("published dedup result differs")
    expected_scores = {str(uid): str(Decimal(str(computed_scores[uid]))) for uid in sorted(computed_scores)}
    if reveal["scores"] != expected_scores:
        raise RevealVerificationError("published composite scores differ")

    inputs = reveal["weight_inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {
        "previous_weights", "eligible_uids", "forced_zero_uids", "max_delta", "precision"
    }:
        raise RevealVerificationError("weight input schema differs")
    previous = {
        int(uid): _decimal(value, f"previous weight {uid}")
        for uid, value in inputs["previous_weights"].items()
    }
    eligible = set(inputs["eligible_uids"])
    forced = set(inputs["forced_zero_uids"])
    if dedup.disqualified - forced:
        raise RevealVerificationError("duplicate UIDs are not forced to zero")
    weight_result = compute_bounded_weights(
        computed_scores,
        previous,
        eligible,
        forced,
        max_delta=_decimal(inputs["max_delta"], "max_delta"),
        precision=inputs["precision"],
    )
    expected_weights = None if weight_result is None else weight_result.serialized()
    if reveal["weights"] != expected_weights:
        raise RevealVerificationError("published weights differ")
    if reveal["set_weights"] is not (expected_weights is not None):
        raise RevealVerificationError("set_weights decision differs")

    return VerifiedReveal(
        epoch_id=epoch_id,
        question_ids=tuple(question_ids),
        model_ids=tuple(model_ids),
        matrix=matrix,
        canonical_costs=costs,
        scores=computed_scores,
        weights=expected_weights,
        chain_receipts_verified=chain_resolver is not None,
    )
