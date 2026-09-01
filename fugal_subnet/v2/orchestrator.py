"""Fail-closed v2 epoch state machine with injectable chain/network operations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from fugal_subnet.v2.commitments import (
    HistoricalCommitmentReceipt,
    verify_historical_receipt,
)
from fugal_subnet.v2.committee import MIN_REPORTS, Builder
from fugal_subnet.v2.journal import PHASES, EpochJournal
from fugal_subnet.v2.reports import (
    ConsensusMatrix,
    derive_consensus_matrix,
    verify_report,
)
from fugal_subnet.v2.reveal import question_commitment_hash


class EpochAborted(RuntimeError):
    pass


class EpochAlreadyComplete(RuntimeError):
    pass


@dataclass(frozen=True)
class EpochDefinition:
    epoch_id: str
    boundary_block: int
    boundary_hash: str
    precommit_deadline_block: int
    report_deadline_block: int
    manifest_hash: str
    grader_hash: str
    questions: list[dict]
    committee: tuple[Builder, ...]
    budget_usd: str


@dataclass(frozen=True)
class EpochHooks:
    commit_questions: Callable[[str], HistoricalCommitmentReceipt | None]
    collect_question_receipts: Callable[[str], Sequence[HistoricalCommitmentReceipt]]
    query_heads: Callable[[], object]
    build_signed_report: Callable[[object, str], bytes | None]
    commit_report: Callable[[str], HistoricalCommitmentReceipt | None]
    wait_until_report_deadline: Callable[[int], int]
    collect_report_receipts: Callable[[], Sequence[HistoricalCommitmentReceipt]]
    fetch_report: Callable[[str, str], bytes]
    evaluate_reveal_and_set_weights: Callable[
        [object, ConsensusMatrix, Mapping[str, bytes], Sequence[HistoricalCommitmentReceipt]],
        bool,
    ]


def _journal_commitments(journal: EpochJournal) -> dict[str, dict]:
    return {
        item["namespace"]: item
        for item in journal.read()["commitments"]
    }


def _advance(journal: EpochJournal, phase: str) -> None:
    """Advance only when needed so supervisor restarts remain resumable."""
    state = journal.read()
    if state["status"] == "complete":
        raise EpochAlreadyComplete("epoch journal is already complete")
    if state["status"] == "aborted":
        raise EpochAborted("epoch journal is already aborted")
    if PHASES.index(state["phase"]) < PHASES.index(phase):
        journal.advance_phase(phase)


def _validate_receipts(
    receipts: Sequence[HistoricalCommitmentReceipt],
    *,
    namespace: str,
    epoch: EpochDefinition,
    artifact_hash: str | None = None,
) -> dict[str, HistoricalCommitmentReceipt]:
    committee = {builder.hotkey for builder in epoch.committee}
    result = {}
    for receipt in receipts:
        verify_historical_receipt(receipt)
        if (
            receipt.namespace != namespace
            or receipt.epoch_id != epoch.epoch_id
            or receipt.hotkey not in committee
            or receipt.hotkey in result
            or (artifact_hash is not None and receipt.artifact_hash != artifact_hash)
        ):
            raise EpochAborted(f"{namespace} commitment receipt is invalid or duplicated")
        if namespace == "questions":
            if not epoch.boundary_block <= receipt.block <= epoch.precommit_deadline_block:
                raise EpochAborted("question commitment block is outside the epoch window")
        elif not epoch.boundary_block < receipt.block <= epoch.report_deadline_block:
            raise EpochAborted("report commitment block is outside the epoch window")
        result[receipt.hotkey] = receipt
    return result


def execute_epoch(
    epoch: EpochDefinition,
    *,
    self_hotkey: str,
    journal: EpochJournal,
    hooks: EpochHooks,
) -> ConsensusMatrix:
    """Execute one v2 epoch; every failure propagates and preserves weights."""
    if not 3 <= len(epoch.committee) <= 5:
        raise EpochAborted("v2 committee must contain three to five builders")
    if len({builder.hotkey for builder in epoch.committee}) != len(epoch.committee):
        raise EpochAborted("v2 committee contains duplicate hotkeys")
    if not (
        epoch.boundary_block
        < epoch.precommit_deadline_block
        < epoch.report_deadline_block
    ):
        raise EpochAborted("epoch deadlines must be strictly ordered")
    initial = journal.initialize(
        manifest_hash=epoch.manifest_hash,
        boundary_block=epoch.boundary_block,
        boundary_hash=epoch.boundary_hash,
        budget_usd=epoch.budget_usd,
    )
    if initial["status"] == "complete":
        raise EpochAlreadyComplete("epoch journal is already complete")
    if initial["status"] == "aborted":
        raise EpochAborted("epoch journal is already aborted")
    question_hash = question_commitment_hash(epoch.questions, epoch.grader_hash)
    is_builder = self_hotkey in {builder.hotkey for builder in epoch.committee}

    _advance(journal, "precommit")
    if is_builder:
        namespace = f"questions:{self_hotkey}"
        if namespace not in _journal_commitments(journal):
            receipt = hooks.commit_questions(question_hash)
            if receipt is None:
                raise EpochAborted("local builder question commitment did not finalize")
            _validate_receipts(
                [receipt], namespace="questions", epoch=epoch, artifact_hash=question_hash,
            )
            journal.record_finalized_commitment(namespace, question_hash, receipt.block)
    question_receipts = _validate_receipts(
        hooks.collect_question_receipts(question_hash),
        namespace="questions",
        epoch=epoch,
        artifact_hash=question_hash,
    )
    if len(question_receipts) < MIN_REPORTS:
        raise EpochAborted("fewer than three finalized builder question commitments")

    _advance(journal, "miner_query")
    heads = hooks.query_heads()
    if heads is None:
        raise EpochAborted("miner head query failed")

    local_report = None
    if is_builder:
        namespace = f"report:{self_hotkey}"
        recorded_report = _journal_commitments(journal).get(namespace)
        if recorded_report is None:
            _advance(journal, "matrix_build")
            local_report = hooks.build_signed_report(heads, question_hash)
            if local_report is None:
                raise EpochAborted("local builder matrix/report failed")
        else:
            local_report = hooks.fetch_report(self_hotkey, recorded_report["hash"])
        report = verify_report(local_report, expected_hotkey=self_hotkey)
        core = report["core"]
        if (
            core["epoch_id"] != epoch.epoch_id
            or core["manifest_hash"] != epoch.manifest_hash
            or core["question_commitment"] != question_hash
        ):
            raise EpochAborted("local signed report differs from epoch definition")
        local_hash = hashlib.sha256(local_report).hexdigest()
        if recorded_report is not None:
            if recorded_report["hash"] != local_hash:
                raise EpochAborted("persisted local report differs from journal commitment")
        else:
            receipt = hooks.commit_report(local_hash)
            if receipt is None:
                raise EpochAborted("local builder report commitment did not finalize")
            _validate_receipts([receipt], namespace="report", epoch=epoch)
            if receipt.artifact_hash != local_hash:
                raise EpochAborted("local report receipt hash differs")
            journal.record_finalized_commitment(namespace, local_hash, receipt.block)
            journal.update_report(
                artifact_hash=local_hash,
                committed_block=receipt.block,
            )
        _advance(journal, "report_commit")

    finalized_block = hooks.wait_until_report_deadline(epoch.report_deadline_block)
    if finalized_block < epoch.report_deadline_block:
        raise EpochAborted("report deadline was not finalized before exchange")
    _advance(journal, "report_exchange")
    report_receipts = _validate_receipts(
        hooks.collect_report_receipts(), namespace="report", epoch=epoch,
    )
    if len(report_receipts) < MIN_REPORTS:
        raise EpochAborted("fewer than three finalized builder report commitments")
    if not set(report_receipts) <= set(question_receipts):
        raise EpochAborted("report builder lacks a finalized question commitment")
    if any(
        receipt.block <= question_receipts[hotkey].block
        for hotkey, receipt in report_receipts.items()
    ):
        raise EpochAborted("report commitment did not follow its question commitment")
    if local_report is not None and self_hotkey not in report_receipts:
        raise EpochAborted("local finalized report commitment is absent from historical query")

    artifacts = {}
    committed_hashes = {}
    for hotkey in sorted(report_receipts):
        receipt = report_receipts[hotkey]
        committed_hashes[hotkey] = receipt.artifact_hash
        if hotkey == self_hotkey and local_report is not None:
            artifact = local_report
        else:
            artifact = hooks.fetch_report(hotkey, receipt.artifact_hash)
        if hashlib.sha256(artifact).hexdigest() != receipt.artifact_hash:
            raise EpochAborted("fetched builder report differs from commitment")
        artifacts[hotkey] = artifact
    journal.update_report(received_hotkeys=list(artifacts))

    _advance(journal, "matrix_consensus")
    consensus = derive_consensus_matrix(
        committee=epoch.committee,
        committed_hashes=committed_hashes,
        artifacts=artifacts,
    )
    _advance(journal, "head_evaluation")
    _advance(journal, "reveal")
    if not hooks.evaluate_reveal_and_set_weights(
        heads, consensus, artifacts, tuple(report_receipts.values())
    ):
        raise EpochAborted("reveal verification or weight submission failed")
    _advance(journal, "set_weights")
    _advance(journal, "complete")
    return consensus


def run_once(
    epoch: EpochDefinition,
    *,
    self_hotkey: str,
    journal: EpochJournal,
    hooks: EpochHooks,
) -> int:
    """Return nonzero for every aborted --once epoch and never mask failure."""
    try:
        execute_epoch(epoch, self_hotkey=self_hotkey, journal=journal, hooks=hooks)
        return 0
    except EpochAlreadyComplete:
        return 0
    except Exception as exc:
        try:
            state = journal.read()
            if state["status"] == "active":
                journal.abort(f"{type(exc).__name__}: {str(exc)[:400]}")
        except Exception:
            pass
        return 1
