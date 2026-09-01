from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import asdict
from decimal import Decimal

import bittensor as bt
import numpy as np
import pytest

from fugal_subnet.graders_v2 import grader_hash
from fugal_subnet.head_eval import load_head_from_npz
from fugal_subnet.v2.commitments import HistoricalCommitmentReceipt, commitment_payload
from fugal_subnet.v2.committee import Builder
from fugal_subnet.v2.head_eval import evaluate_head
from fugal_subnet.v2.reports import derive_consensus_matrix, sign_report
from fugal_subnet.v2.reveal import (
    HeadSubmission,
    RevealVerificationError,
    _evaluation_evidence,
    build_reveal,
    encode_reveal,
    finalize_reveal,
    publish_reveal,
    question_commitment_hash,
    registry_snapshot_hash,
    stage_reveal,
    verify_reveal,
)
from fugal_subnet.v2.scoring import composite_score
from fugal_subnet.v2.soft_targets import compute_soft_targets


def _receipt(namespace, epoch_id, artifact_hash, hotkey, uid, block):
    return HistoricalCommitmentReceipt(
        network="test",
        netuid=11,
        hotkey=hotkey,
        uid=uid,
        block=block,
        block_hash=f"{block:064x}",
        namespace=namespace,
        epoch_id=epoch_id,
        artifact_hash=artifact_hash,
        payload=commitment_payload(namespace, epoch_id, artifact_hash),
    )


def _head_bytes() -> bytes:
    output = io.BytesIO()
    np.savez(
        output,
        W=np.zeros((2, 1024), dtype=np.float32),
        b=np.asarray([1.0, 0.0], dtype=np.float32),
        models=np.asarray(["provider/a", "provider/b"]),
    )
    return output.getvalue()


def _make_reveal() -> dict:
    keys = [bt.Keypair.create_from_uri(f"//RevealBuilder{i}") for i in range(3)]
    committee = [Builder(uid=index, hotkey=key.ss58_address) for index, key in enumerate(keys)]
    questions = [
        {"prompt": "one", "gold": "1", "grader_id": "numeric_decimal_v2", "benchmark": "test", "question_id": "q1", "metadata": {}},
        {"prompt": "two", "gold": "2", "grader_id": "numeric_decimal_v2", "benchmark": "test", "question_id": "q2", "metadata": {}},
    ]
    raw_grader_hash = grader_hash().removeprefix("sha256:")
    question_hash = question_commitment_hash(questions, raw_grader_hash)
    model_ids = ["provider/a", "provider/b"]
    costs = {"provider/a": "0.001", "provider/b": "0.002"}
    registry_hash = registry_snapshot_hash(model_ids, costs)
    votes = [
        (1, 0, 1, 0),
        (1, 1, 0, 0),
        (0, 1, 1, 0),
    ]
    payloads = {}
    for key, grades in zip(keys, votes):
        cells = []
        index = 0
        for question in questions:
            for model_id in model_ids:
                response = question["gold"] if grades[index] else "999"
                cells.append({
                    "question_id": question["question_id"],
                    "model_id": model_id,
                    "response": response,
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "grade": grades[index],
                    "actual_cost_usd": "0",
                    "reserved_cost_usd": "0",
                })
                index += 1
        core = {
            "epoch_id": "v2-100",
            "boundary_block": 100,
            "boundary_hash": "1" * 64,
            "manifest_hash": "2" * 64,
            "question_commitment": question_hash,
            "grader_hash": raw_grader_hash,
            "registry_hash": registry_hash,
            "builder_hotkey": key.ss58_address,
            "question_ids": ["q1", "q2"],
            "model_ids": model_ids,
            "cells": cells,
            "spend": {"currency": "USD", "budget_usd": "0", "reserved_usd": "0", "actual_usd": "0"},
        }
        payloads[key.ss58_address] = sign_report(core, key)
    commits = {hotkey: hashlib.sha256(payload).hexdigest() for hotkey, payload in payloads.items()}
    consensus = derive_consensus_matrix(
        committee=committee, committed_hashes=commits, artifacts=payloads,
    )

    matrix = np.asarray(consensus.matrix, dtype=np.int8)
    hidden = np.zeros((2, 1024), dtype=np.float32)
    head_raw = _head_bytes()
    head_hash = hashlib.sha256(head_raw).hexdigest()
    head = load_head_from_npz(head_raw)
    evaluation = evaluate_head(
        head,
        hidden,
        matrix,
        model_ids,
        compute_soft_targets(matrix),
        {"provider/a": 0.001, "provider/b": 0.002},
        wire_model_pool=model_ids,
    )
    score = composite_score(
        evaluation.accuracy, evaluation.cost_efficiency, evaluation.kl_score
    )
    head_key = bt.Keypair.create_from_uri("//RevealMiner")
    return {
        "schema_version": 2,
        "protocol": "v2",
        "epoch": {
            "epoch_id": "v2-100",
            "boundary_block": 100,
            "boundary_hash": "1" * 64,
            "precommit_deadline_block": 120,
            "report_deadline_block": 200,
            "manifest_hash": "2" * 64,
            "grader_hash": raw_grader_hash,
        },
        "committee": [asdict(builder) for builder in committee],
        "questions": questions,
        "registry": {
            "model_ids": model_ids,
            "route_costs_usd": costs,
            "snapshot_hash": registry_hash,
        },
        "question_receipts": [
            asdict(_receipt("questions", "v2-100", question_hash, key.ss58_address, index, 101))
            for index, key in enumerate(keys)
        ],
        "report_receipts": [
            asdict(_receipt("report", "v2-100", commits[key.ss58_address], key.ss58_address, index, 150))
            for index, key in enumerate(keys)
        ],
        "builder_reports": {
            hotkey: base64.b64encode(payload).decode() for hotkey, payload in payloads.items()
        },
        "consensus": {
            "question_ids": list(consensus.question_ids),
            "model_ids": list(consensus.model_ids),
            "matrix": [list(row) for row in consensus.matrix],
            "builder_hotkeys": list(consensus.builder_hotkeys),
            "report_hashes": list(consensus.report_hashes),
        },
        "heads": [{
            "uid": 7,
            "hotkey": head_key.ss58_address,
            "artifact_b64": base64.b64encode(head_raw).decode(),
            "artifact_sha256": head_hash,
            "wire_model_pool": model_ids,
            "commitment_receipt": asdict(
                _receipt("head", "head-v2", head_hash, head_key.ss58_address, 7, 99)
            ),
            "evaluation": _evaluation_evidence(evaluation),
        }],
        "rejected_heads": [],
        "dedup": {"clusters": [], "disqualified": []},
        "scores": {"7": str(Decimal(str(score)))},
        "weight_inputs": {
            "previous_weights": {},
            "eligible_uids": [7],
            "forced_zero_uids": [],
            "max_delta": "0.3",
            "precision": 12,
        },
        "weights": {"7": "1.000000000000"},
        "set_weights": True,
    }


def _embed(prompts):
    assert prompts == ["one", "two"]
    return np.zeros((2, 1024), dtype=np.float32)


def test_complete_reveal_recomputes_matrix_heads_scores_and_weights():
    reveal = _make_reveal()
    result = verify_reveal(
        encode_reveal(reveal),
        grading_client=None,
        embedder=_embed,
        chain_resolver=lambda receipt: receipt.payload,
    )
    assert result.matrix.tolist() == [[1, 1], [1, 0]]
    assert result.model_ids == ("provider/a", "provider/b")
    assert result.weights == {"7": "1.000000000000"}
    assert result.chain_receipts_verified is True


def test_reveal_builder_produces_self_verified_complete_artifact(tmp_path):
    source = _make_reveal()
    reports = {
        hotkey: base64.b64decode(payload)
        for hotkey, payload in source["builder_reports"].items()
    }
    head = source["heads"][0]
    payload, verified = build_reveal(
        epoch=source["epoch"],
        committee=[Builder(**item) for item in source["committee"]],
        questions=source["questions"],
        route_costs_usd=source["registry"]["route_costs_usd"],
        question_receipts=[
            HistoricalCommitmentReceipt.from_dict(item)
            for item in source["question_receipts"]
        ],
        report_receipts=[
            HistoricalCommitmentReceipt.from_dict(item)
            for item in source["report_receipts"]
        ],
        builder_reports=reports,
        head_submissions=[HeadSubmission(
            uid=head["uid"],
            hotkey=head["hotkey"],
            artifact=base64.b64decode(head["artifact_b64"]),
            wire_model_pool=tuple(head["wire_model_pool"]),
            commitment_receipt=HistoricalCommitmentReceipt.from_dict(
                head["commitment_receipt"]
            ),
        )],
        grading_client=None,
        embedder=_embed,
        previous_weights={},
        eligible_uids={7},
        forced_zero_uids=set(),
    )
    assert verified.weights == {"7": "1.000000000000"}
    destination = tmp_path / "epoch" / "reveal.json"
    assert publish_reveal(payload, destination) == destination
    assert publish_reveal(payload, destination) == destination
    assert destination.read_bytes() == payload
    with pytest.raises(RevealVerificationError, match="other bytes"):
        publish_reveal(encode_reveal({**source, "set_weights": False}), destination)


def test_reveal_without_chain_resolver_is_honest_about_receipt_status():
    result = verify_reveal(
        encode_reveal(_make_reveal()), grading_client=None, embedder=_embed,
    )
    assert result.chain_receipts_verified is False


def test_reveal_is_durably_staged_before_publication(tmp_path):
    payload = encode_reveal(_make_reveal())
    destination = tmp_path / "epoch" / "reveal.json"

    pending = stage_reveal(payload, destination)

    assert pending == destination.with_name("reveal.json.pending")
    assert pending.read_bytes() == payload
    assert not destination.exists()
    assert finalize_reveal(payload, destination) == destination
    assert destination.read_bytes() == payload
    assert not pending.exists()


def test_conflicting_staged_reveal_fails_closed(tmp_path):
    destination = tmp_path / "epoch" / "reveal.json"
    payload = encode_reveal(_make_reveal())
    stage_reveal(payload, destination)

    with pytest.raises(RevealVerificationError, match="staged reveal"):
        stage_reveal(
            encode_reveal({**_make_reveal(), "set_weights": True}), destination
        )


def test_five_builder_committee_accepts_three_builder_quorum():
    reveal = _make_reveal()
    extra_keys = [bt.Keypair.create_from_uri(f"//IdleBuilder{i}") for i in range(2)]
    reveal["committee"].extend(
        {"uid": index + 3, "hotkey": key.ss58_address}
        for index, key in enumerate(extra_keys)
    )
    result = verify_reveal(
        encode_reveal(reveal), grading_client=None, embedder=_embed,
    )
    assert result.matrix.tolist() == [[1, 1], [1, 0]]


def test_duplicate_receipt_builder_fails_closed():
    reveal = _make_reveal()
    reveal["question_receipts"][1] = reveal["question_receipts"][0]
    with pytest.raises(RevealVerificationError, match="duplicated"):
        verify_reveal(encode_reveal(reveal), grading_client=None, embedder=_embed)


def test_question_commitment_after_precommit_deadline_fails_closed():
    reveal = _make_reveal()
    receipt = reveal["question_receipts"][0]
    receipt["block"] = 121
    receipt["block_hash"] = f"{121:064x}"
    with pytest.raises(RevealVerificationError, match="builder/block"):
        verify_reveal(encode_reveal(reveal), grading_client=None, embedder=_embed)


def test_rejected_head_bytes_hash_and_reason_are_verified():
    reveal = _make_reveal()
    rejected_key = bt.Keypair.create_from_uri("//RejectedMiner")
    artifact = b"not an npz"
    artifact_hash = hashlib.sha256(artifact).hexdigest()
    reveal["rejected_heads"] = [{
        "uid": 8,
        "hotkey": rejected_key.ss58_address,
        "artifact_b64": base64.b64encode(artifact).decode(),
        "artifact_sha256": artifact_hash,
        "wire_model_pool": [],
        "commitment_receipt": asdict(
            _receipt("head", "head-v2", artifact_hash, rejected_key.ss58_address, 8, 99)
        ),
        "rejection_code": "artifact_invalid",
    }]
    result = verify_reveal(
        encode_reveal(reveal), grading_client=None, embedder=_embed,
    )
    assert result.weights == {"7": "1.000000000000"}

    reveal["rejected_heads"][0]["rejection_code"] = "models_invalid"
    with pytest.raises(RevealVerificationError, match="invalid artifact"):
        verify_reveal(encode_reveal(reveal), grading_client=None, embedder=_embed)


def test_reveal_tampering_and_historical_chain_mismatch_fail_closed():
    reveal = _make_reveal()
    reveal["weights"]["7"] = "0.999999999999"
    with pytest.raises(RevealVerificationError, match="weights differ"):
        verify_reveal(encode_reveal(reveal), grading_client=None, embedder=_embed)

    valid = encode_reveal(_make_reveal())
    with pytest.raises(RevealVerificationError, match="historical chain"):
        verify_reveal(
            valid,
            grading_client=None,
            embedder=_embed,
            chain_resolver=lambda receipt: "overwritten-latest-value",
        )


def test_noncanonical_reveal_json_is_rejected():
    payload = encode_reveal(_make_reveal())
    with pytest.raises(RevealVerificationError, match="canonical"):
        verify_reveal(payload + b" ", grading_client=None, embedder=_embed)
