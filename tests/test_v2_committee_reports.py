from __future__ import annotations

import copy
import hashlib
import importlib.resources
import json

import bittensor as bt
import numpy as np
import pytest

from fugal_subnet.consensus_manifest import canonical_json
from fugal_subnet.v2.committee import Builder, QuorumUnavailable, select_builders
from fugal_subnet.v2.matrix import CellResult, MatrixResult
from fugal_subnet.v2.model_registry import ModelRegistryError, load_model_registry
from fugal_subnet.v2.protocol import assemble_report
from fugal_subnet.v2.reports import (
    ReportError,
    build_signed_report,
    chunk_signed_report,
    derive_consensus_matrix,
    sign_report,
    verify_chunk_signature,
    verify_report,
)


def _keypairs(count: int = 5) -> list[bt.Keypair]:
    return [bt.Keypair.create_from_uri(f"//FugalV2Builder{index}") for index in range(count)]


def _core(keypair: bt.Keypair, grades: tuple[int, int, int, int]) -> dict:
    question_ids = ["gsm8k:1", "ifeval:2"]
    model_ids = ["provider/a", "provider/b"]
    cells = []
    for index, (question_id, model_id) in enumerate(
        (q_m for q in question_ids for q_m in ((q, model_ids[0]), (q, model_ids[1])))
    ):
        response = f"answer-{index}"
        cells.append({
            "question_id": question_id,
            "model_id": model_id,
            "response": response,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "grade": grades[index],
            "actual_cost_usd": "0.001",
            "reserved_cost_usd": "0.002",
        })
    return {
        "epoch_id": "v2-100",
        "boundary_block": 100,
        "boundary_hash": "1" * 64,
        "manifest_hash": "2" * 64,
        "question_commitment": "3" * 64,
        "grader_hash": "4" * 64,
        "registry_hash": "5" * 64,
        "builder_hotkey": keypair.ss58_address,
        "question_ids": question_ids,
        "model_ids": model_ids,
        "cells": cells,
        "spend": {
            "currency": "USD",
            "budget_usd": "1",
            "reserved_usd": "0.008",
            "actual_usd": "0.004",
        },
    }


def _signed_reports(votes: list[tuple[int, int, int, int]]):
    keypairs = _keypairs(len(votes))
    payloads = [sign_report(_core(keypair, grades), keypair) for keypair, grades in zip(keypairs, votes)]
    committee = tuple(Builder(uid=index, hotkey=keypair.ss58_address) for index, keypair in enumerate(keypairs))
    commits = {
        keypair.ss58_address: hashlib.sha256(payload).hexdigest()
        for keypair, payload in zip(keypairs, payloads)
    }
    artifacts = {keypair.ss58_address: payload for keypair, payload in zip(keypairs, payloads)}
    return keypairs, committee, commits, artifacts


def test_committee_selection_is_boundary_deterministic_and_permit_only():
    keypairs = _keypairs(7)
    hotkeys = [keypair.ss58_address for keypair in keypairs]
    permits = [True, True, False, True, True, True, True]
    selected = select_builders("ab" * 32, hotkeys, permits)
    assert len(selected) == 5
    assert all(permits[builder.uid] for builder in selected)
    assert selected == select_builders("ab" * 32, hotkeys, permits)
    assert selected != select_builders("cd" * 32, hotkeys, permits)


def test_committee_aborts_without_minimum_permitted_validators():
    hotkeys = [keypair.ss58_address for keypair in _keypairs(3)]
    with pytest.raises(QuorumUnavailable):
        select_builders("ab" * 32, hotkeys, [True, False, True])


def test_candidate_registry_is_valid_but_fails_closed_until_approved():
    registry = load_model_registry()
    assert len(registry.models) == 6
    assert registry.model_ids == ()
    assert all(not model.enabled for model in registry.models)
    with pytest.raises(ModelRegistryError, match="no approved active models"):
        load_model_registry(require_active=True)


def test_model_registry_override_is_local_only(tmp_path, monkeypatch):
    source = json.loads(
        importlib.resources.files("fugal_subnet")
        .joinpath("model-registry-v2.json")
        .read_text(encoding="utf-8")
    )
    source["models"][0]["enabled"] = True
    source["models"][0]["review_status"] = "approved"
    override = tmp_path / "local-models.json"
    override.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setenv("FUGAL_MODEL_REGISTRY", str(override))

    assert load_model_registry(require_active=True, network="local").model_ids == (
        source["models"][0]["id"],
    )
    with pytest.raises(ModelRegistryError, match="local/mock-only"):
        load_model_registry(require_active=True, network="test")


def test_signed_report_and_chunks_round_trip():
    keypair = _keypairs(1)[0]
    payload = sign_report(_core(keypair, (1, 0, 1, 1)), keypair)
    report = verify_report(payload, expected_hotkey=keypair.ss58_address)
    assert report["core"]["cells"][0]["grade"] == 1
    chunks = chunk_signed_report(payload, keypair)
    assert all(verify_chunk_signature(chunk) for chunk in chunks)
    assert assemble_report(list(reversed(chunks))) == payload


def test_matrix_result_builds_canonical_signed_report():
    keypair = _keypairs(1)[0]
    cells = tuple(
        CellResult(
            question_id=question_id,
            model_id=model_id,
            response=f"answer-{index}",
            prompt_tokens=10,
            completion_tokens=2,
            grade=grade,
            actual_cost_usd="0.001",
            reserved_cost_usd="0.002",
        )
        for index, (question_id, model_id, grade) in enumerate([
            ("gsm8k:1", "provider/a", 1),
            ("gsm8k:1", "provider/b", 0),
            ("ifeval:2", "provider/a", 1),
            ("ifeval:2", "provider/b", 1),
        ])
    )
    matrix = MatrixResult(
        question_ids=("gsm8k:1", "ifeval:2"),
        model_ids=("provider/a", "provider/b"),
        matrix=np.asarray([[1, 0], [1, 1]], dtype=np.int8),
        cells=cells,
        budget_usd="1",
        reserved_usd="0.008",
        actual_usd="0.004",
    )
    payload = build_signed_report(
        matrix,
        epoch_id="v2-100",
        boundary_block=100,
        boundary_hash="0x" + "1" * 64,
        manifest_hash="2" * 64,
        question_commitment="3" * 64,
        grader_hash="4" * 64,
        registry_hash="5" * 64,
        keypair=keypair,
    )
    report = verify_report(payload, expected_hotkey=keypair.ss58_address)
    assert [cell["grade"] for cell in report["core"]["cells"]] == [1, 0, 1, 1]


def test_report_tampering_and_noncanonical_json_are_rejected():
    keypair = _keypairs(1)[0]
    payload = sign_report(_core(keypair, (1, 0, 1, 1)), keypair)
    raw = json.loads(payload)
    raw["core"]["cells"][0]["grade"] = 0
    with pytest.raises(ReportError, match="signature"):
        verify_report(canonical_json(raw))
    with pytest.raises(ReportError, match="canonical"):
        verify_report(json.dumps(json.loads(payload), indent=2).encode())


def test_report_epoch_id_rejects_path_traversal():
    keypair = _keypairs(1)[0]
    core = _core(keypair, (1, 0, 1, 0))
    core["epoch_id"] = "../escape"
    with pytest.raises(ReportError, match="epoch_id"):
        sign_report(core, keypair)


def test_consensus_uses_strict_majority_and_ties_grade_zero():
    _, committee, commits, artifacts = _signed_reports([
        (1, 1, 1, 0),
        (1, 0, 1, 0),
        (0, 0, 1, 1),
        (0, 1, 1, 1),
    ])
    result = derive_consensus_matrix(
        committee=committee,
        committed_hashes=commits,
        artifacts=artifacts,
    )
    assert result.matrix == ((0, 0), (1, 0))


def test_every_committed_report_must_be_available_and_match_hash():
    _, committee, commits, artifacts = _signed_reports([
        (1, 1, 1, 0), (1, 0, 1, 0), (0, 0, 1, 1),
    ])
    missing = dict(artifacts)
    missing.pop(next(iter(missing)))
    with pytest.raises(QuorumUnavailable, match="unavailable"):
        derive_consensus_matrix(
            committee=committee, committed_hashes=commits, artifacts=missing,
        )

    damaged = dict(artifacts)
    hotkey = next(iter(damaged))
    damaged[hotkey] += b" "
    with pytest.raises(ReportError, match="hash mismatch"):
        derive_consensus_matrix(
            committee=committee, committed_hashes=commits, artifacts=damaged,
        )


def test_reports_must_share_canonical_epoch_material():
    keypairs, committee, commits, artifacts = _signed_reports([
        (1, 1, 1, 0), (1, 0, 1, 0), (0, 0, 1, 1),
    ])
    replacement_core = copy.deepcopy(_core(keypairs[2], (0, 0, 1, 1)))
    replacement_core["boundary_block"] = 101
    replacement = sign_report(replacement_core, keypairs[2])
    hotkey = keypairs[2].ss58_address
    artifacts[hotkey] = replacement
    commits[hotkey] = hashlib.sha256(replacement).hexdigest()
    with pytest.raises(ReportError, match="canonical epoch material"):
        derive_consensus_matrix(
            committee=committee, committed_hashes=commits, artifacts=artifacts,
        )
