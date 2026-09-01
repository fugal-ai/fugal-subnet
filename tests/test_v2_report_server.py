from __future__ import annotations

import asyncio
import hashlib

import bittensor as bt
import pytest

from fugal_subnet.v2.report_server import (
    ReportFetchError,
    ReportStore,
    fetch_report,
    make_report_forward,
)
from fugal_subnet.v2.reports import sign_report


def _large_report(keypair):
    questions = [f"q{index:03d}" for index in range(70)]
    response = "x" * 4000
    cells = [{
        "question_id": question,
        "model_id": "provider/a",
        "response": response,
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "grade": 0,
        "actual_cost_usd": "0",
        "reserved_cost_usd": "0",
    } for question in questions]
    core = {
        "epoch_id": "v2-100",
        "boundary_block": 100,
        "boundary_hash": "1" * 64,
        "manifest_hash": "2" * 64,
        "question_commitment": "3" * 64,
        "grader_hash": "4" * 64,
        "registry_hash": "5" * 64,
        "builder_hotkey": keypair.ss58_address,
        "question_ids": questions,
        "model_ids": ["provider/a"],
        "cells": cells,
        "spend": {"currency": "USD", "budget_usd": "0", "reserved_usd": "0", "actual_usd": "0"},
    }
    return sign_report(core, keypair)


class LocalDendrite:
    def __init__(self, forward, refuse_index=None):
        self.forward = forward
        self.refuse_index = refuse_index

    def query(self, axons, synapse, timeout=30):
        if synapse.chunk_index == self.refuse_index:
            synapse.error = "selective refusal"
            return [synapse]
        return [asyncio.run(self.forward(synapse))]


def test_report_store_and_dendrite_fetch_all_signed_chunks(tmp_path):
    keypair = bt.Keypair.create_from_uri("//ReportStoreBuilder")
    payload = _large_report(keypair)
    store = ReportStore(tmp_path)
    chunks = store.publish(payload, keypair)
    assert len(chunks) == 2
    assert all(path.stat().st_mode & 0o077 == 0 for path in tmp_path.iterdir())
    fetched = fetch_report(
        LocalDendrite(make_report_forward(store)),
        object(),
        epoch_id="v2-100",
        manifest_hash="2" * 64,
        artifact_hash=hashlib.sha256(payload).hexdigest(),
        builder_hotkey=keypair.ss58_address,
    )
    assert fetched == payload


def test_selective_chunk_refusal_fails_entire_committed_report(tmp_path):
    keypair = bt.Keypair.create_from_uri("//ReportStoreBuilder")
    payload = _large_report(keypair)
    store = ReportStore(tmp_path)
    store.publish(payload, keypair)
    with pytest.raises(ReportFetchError, match="refused"):
        fetch_report(
            LocalDendrite(make_report_forward(store), refuse_index=1),
            object(),
            epoch_id="v2-100",
            manifest_hash="2" * 64,
            artifact_hash=hashlib.sha256(payload).hexdigest(),
            builder_hotkey=keypair.ss58_address,
        )


def test_wrong_expected_builder_is_rejected(tmp_path):
    keypair = bt.Keypair.create_from_uri("//ReportStoreBuilder")
    payload = _large_report(keypair)
    store = ReportStore(tmp_path)
    store.publish(payload, keypair)
    with pytest.raises(ReportFetchError, match="identity or signature"):
        fetch_report(
            LocalDendrite(make_report_forward(store)),
            object(),
            epoch_id="v2-100",
            manifest_hash="2" * 64,
            artifact_hash=hashlib.sha256(payload).hexdigest(),
            builder_hotkey=bt.Keypair.create_from_uri("//OtherBuilder").ss58_address,
        )


def test_report_store_restores_committed_bytes_after_restart(tmp_path):
    keypair = bt.Keypair.create_from_uri("//ReportStoreBuilder")
    payload = _large_report(keypair)
    first = ReportStore(tmp_path)
    first.publish(payload, keypair)

    restarted = ReportStore(tmp_path)
    assert restarted.restore(keypair) == 1
    fetched = fetch_report(
        LocalDendrite(make_report_forward(restarted)),
        object(),
        epoch_id="v2-100",
        manifest_hash="2" * 64,
        artifact_hash=hashlib.sha256(payload).hexdigest(),
        builder_hotkey=keypair.ss58_address,
    )
    assert fetched == payload


def test_report_stays_sealed_until_finalized_deadline_and_restores_gate(tmp_path):
    keypair = bt.Keypair.create_from_uri("//ReportStoreBuilder")
    payload = _large_report(keypair)
    block = [149]
    first = ReportStore(tmp_path, current_block=lambda: block[0])
    first.publish(payload, keypair, release_block=150)
    artifact_hash = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ReportFetchError, match="sealed"):
        first.read_payload("v2-100", artifact_hash)

    block[0] = 150
    assert first.read_payload("v2-100", artifact_hash) == payload
    restarted = ReportStore(tmp_path, current_block=lambda: block[0])
    assert restarted.restore(keypair) == 1
    assert restarted.read_payload("v2-100", artifact_hash) == payload


def test_report_store_rejects_symlink_root(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "store"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ReportFetchError, match="symlink"):
        ReportStore(link)
