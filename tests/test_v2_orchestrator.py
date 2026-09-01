from __future__ import annotations

import hashlib

import bittensor as bt

from fugal_subnet.v2.commitments import HistoricalCommitmentReceipt, commitment_payload
from fugal_subnet.v2.committee import Builder
from fugal_subnet.v2.journal import EpochJournal
from fugal_subnet.v2.orchestrator import (
    EpochDefinition,
    EpochHooks,
    execute_epoch,
    run_once,
)
from fugal_subnet.v2.reports import sign_report
from fugal_subnet.v2.reveal import question_commitment_hash


def _receipt(namespace, epoch_id, artifact_hash, hotkey, uid, block):
    return HistoricalCommitmentReceipt(
        network="test", netuid=11, hotkey=hotkey, uid=uid, block=block,
        block_hash=f"{block:064x}", namespace=namespace, epoch_id=epoch_id,
        artifact_hash=artifact_hash,
        payload=commitment_payload(namespace, epoch_id, artifact_hash),
    )


def _fixture(tmp_path):
    keys = [bt.Keypair.create_from_uri(f"//OrchestratorBuilder{i}") for i in range(3)]
    committee = tuple(Builder(index, key.ss58_address) for index, key in enumerate(keys))
    questions = [{
        "prompt": "one", "gold": "1", "grader_id": "numeric_decimal_v2",
        "benchmark": "test", "question_id": "q1", "metadata": {},
    }]
    epoch = EpochDefinition(
        epoch_id="v2-100", boundary_block=100, boundary_hash="1" * 64,
        precommit_deadline_block=120, report_deadline_block=200,
        manifest_hash="2" * 64,
        grader_hash="4" * 64, questions=questions, committee=committee,
        budget_usd="1",
    )
    q_hash = question_commitment_hash(questions, epoch.grader_hash)
    question_receipts = [
        _receipt("questions", epoch.epoch_id, q_hash, key.ss58_address, index, 101)
        for index, key in enumerate(keys)
    ]
    reports = {}
    for index, key in enumerate(keys):
        response = "1" if index < 2 else "999"
        grade = int(index < 2)
        core = {
            "epoch_id": epoch.epoch_id,
            "boundary_block": epoch.boundary_block,
            "boundary_hash": epoch.boundary_hash,
            "manifest_hash": epoch.manifest_hash,
            "question_commitment": q_hash,
            "grader_hash": epoch.grader_hash,
            "registry_hash": "5" * 64,
            "builder_hotkey": key.ss58_address,
            "question_ids": ["q1"],
            "model_ids": ["provider/a"],
            "cells": [{
                "question_id": "q1", "model_id": "provider/a",
                "response": response,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "prompt_tokens": 1, "completion_tokens": 1, "grade": grade,
                "actual_cost_usd": "0", "reserved_cost_usd": "0",
            }],
            "spend": {"currency": "USD", "budget_usd": "1", "reserved_usd": "0", "actual_usd": "0"},
        }
        reports[key.ss58_address] = sign_report(core, key)
    report_receipts = [
        _receipt(
            "report", epoch.epoch_id, hashlib.sha256(reports[key.ss58_address]).hexdigest(),
            key.ss58_address, index, 150,
        )
        for index, key in enumerate(keys)
    ]
    return epoch, keys, q_hash, question_receipts, reports, report_receipts, EpochJournal(tmp_path, epoch.epoch_id)


def test_epoch_state_machine_completes_strict_quorum_and_weights(tmp_path):
    epoch, keys, q_hash, q_receipts, reports, r_receipts, journal = _fixture(tmp_path)
    events = []
    hooks = EpochHooks(
        commit_questions=lambda artifact: events.append("commit_questions") or q_receipts[0],
        collect_question_receipts=lambda artifact: events.append("collect_questions") or q_receipts,
        query_heads=lambda: events.append("query_heads") or {7: b"head"},
        build_signed_report=lambda heads, artifact: events.append("build_report") or reports[keys[0].ss58_address],
        commit_report=lambda artifact: events.append("commit_report") or r_receipts[0],
        wait_until_report_deadline=lambda block: events.append("deadline") or block,
        collect_report_receipts=lambda: events.append("collect_reports") or r_receipts,
        fetch_report=lambda hotkey, artifact: events.append(f"fetch:{hotkey}") or reports[hotkey],
        evaluate_reveal_and_set_weights=lambda heads, consensus, artifacts, receipts: (
            events.append("weights") or consensus.matrix == ((1,),)
        ),
    )
    assert run_once(
        epoch, self_hotkey=keys[0].ss58_address, journal=journal, hooks=hooks,
    ) == 0
    assert journal.read()["status"] == "complete"
    assert events[:5] == [
        "commit_questions", "collect_questions", "query_heads", "build_report", "commit_report",
    ]
    assert events[-1] == "weights"


def test_quorum_loss_returns_nonzero_and_persists_abort(tmp_path):
    epoch, keys, q_hash, q_receipts, reports, r_receipts, journal = _fixture(tmp_path)
    hooks = EpochHooks(
        commit_questions=lambda _: q_receipts[0],
        collect_question_receipts=lambda _: q_receipts[:2],
        query_heads=lambda: (_ for _ in ()).throw(AssertionError("must not query")),
        build_signed_report=lambda *_: None,
        commit_report=lambda _: None,
        wait_until_report_deadline=lambda block: block,
        collect_report_receipts=lambda: [],
        fetch_report=lambda *_: b"",
        evaluate_reveal_and_set_weights=lambda *_: False,
    )
    assert run_once(
        epoch, self_hotkey=keys[0].ss58_address, journal=journal, hooks=hooks,
    ) == 1
    state = journal.read()
    assert state["status"] == "aborted"
    assert "fewer than three" in state["abort_reason"]


def test_committed_artifact_refusal_aborts_before_weights(tmp_path):
    epoch, keys, q_hash, q_receipts, reports, r_receipts, journal = _fixture(tmp_path)
    called_weights = False

    def fetch(hotkey, artifact):
        if hotkey == keys[1].ss58_address:
            raise RuntimeError("selective refusal")
        return reports[hotkey]

    def weights(*args):
        nonlocal called_weights
        called_weights = True
        return True

    hooks = EpochHooks(
        commit_questions=lambda _: q_receipts[0],
        collect_question_receipts=lambda _: q_receipts,
        query_heads=lambda: {7: b"head"},
        build_signed_report=lambda *_: reports[keys[0].ss58_address],
        commit_report=lambda _: r_receipts[0],
        wait_until_report_deadline=lambda block: block,
        collect_report_receipts=lambda: r_receipts,
        fetch_report=fetch,
        evaluate_reveal_and_set_weights=weights,
    )
    assert run_once(
        epoch, self_hotkey=keys[0].ss58_address, journal=journal, hooks=hooks,
    ) == 1
    assert called_weights is False
    assert journal.read()["status"] == "aborted"


def test_restart_after_report_commit_reuses_report_and_completed_paid_cells(tmp_path):
    epoch, keys, q_hash, q_receipts, reports, r_receipts, journal = _fixture(tmp_path)
    calls = {"questions": 0, "build": 0, "report": 0, "weights": 0}
    crash_once = True

    def commit_questions(_):
        calls["questions"] += 1
        return q_receipts[0]

    def build_report(*_):
        calls["build"] += 1
        return reports[keys[0].ss58_address]

    def commit_report(_):
        calls["report"] += 1
        return r_receipts[0]

    def weights(*_):
        nonlocal crash_once
        calls["weights"] += 1
        if crash_once:
            crash_once = False
            raise KeyboardInterrupt("simulated supervisor crash after reveal")
        return True

    hooks = EpochHooks(
        commit_questions=commit_questions,
        collect_question_receipts=lambda _: q_receipts,
        query_heads=lambda: {7: b"committed-head"},
        build_signed_report=build_report,
        commit_report=commit_report,
        wait_until_report_deadline=lambda block: block,
        collect_report_receipts=lambda: r_receipts,
        fetch_report=lambda hotkey, _: reports[hotkey],
        evaluate_reveal_and_set_weights=weights,
    )
    try:
        execute_epoch(
            epoch, self_hotkey=keys[0].ss58_address, journal=journal, hooks=hooks,
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("simulated crash did not occur")

    assert journal.read()["status"] == "active"
    assert run_once(
        epoch, self_hotkey=keys[0].ss58_address, journal=journal, hooks=hooks,
    ) == 0
    assert calls == {"questions": 1, "build": 1, "report": 1, "weights": 2}
    assert journal.read()["status"] == "complete"

    hooks_after_complete = EpochHooks(
        commit_questions=lambda *_: (_ for _ in ()).throw(AssertionError()),
        collect_question_receipts=lambda *_: (_ for _ in ()).throw(AssertionError()),
        query_heads=lambda *_: (_ for _ in ()).throw(AssertionError()),
        build_signed_report=lambda *_: (_ for _ in ()).throw(AssertionError()),
        commit_report=lambda *_: (_ for _ in ()).throw(AssertionError()),
        wait_until_report_deadline=lambda *_: (_ for _ in ()).throw(AssertionError()),
        collect_report_receipts=lambda *_: (_ for _ in ()).throw(AssertionError()),
        fetch_report=lambda *_: (_ for _ in ()).throw(AssertionError()),
        evaluate_reveal_and_set_weights=lambda *_: (_ for _ in ()).throw(AssertionError()),
    )
    assert run_once(
        epoch,
        self_hotkey=keys[0].ss58_address,
        journal=journal,
        hooks=hooks_after_complete,
    ) == 0
