"""Exports bind the reviewed inputs and fail closed on incomplete observations."""
import json
from pathlib import Path

import numpy as np
import pytest

from fugal_subnet import success_test_fixtures as fixtures
from fugal_subnet.vendor import success_contract as c
from scripts.build_token_manifest import build
from scripts.export_success_head import export, verify


def test_offline_bundle_integrity_and_deployable_guard(tmp_path):
    head = tmp_path / "h.npz"
    head.write_bytes(fixtures.head_bytes(np.zeros((1, 1024)), np.zeros(1), ["a"]))
    manifest, prices, report = [tmp_path / n for n in ("tokens.json", "prices.json", "report.json")]
    manifest.write_text(json.dumps(fixtures.manifest(["a"])))
    prices.write_text(json.dumps([{"id": "a", "in": 1, "out": 2}]))
    report.write_text(json.dumps({"head_sha256": c.file_hash(head), "test_only": True}))
    output = tmp_path / "bundle"
    metadata = export(head, manifest, prices, report, output)
    assert metadata["deployable"] is False
    assert verify(output) == metadata
    altered = dict(metadata, default_lambda=2)
    (output / "bundle.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="metadata disagrees"):
        verify(output)
    (output / "bundle.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="reviewed"):
        export(head, manifest, prices, report, tmp_path / "live", deployable=True)
    report.write_text('{"head_sha256": "different-head"}')
    with pytest.raises(ValueError, match="bind"):
        export(head, manifest, prices, report, tmp_path / "unbound")
    (output / "head.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        verify(output)


def test_missing_token_observations_never_get_fabricated(tmp_path):
    source = tmp_path / "observations.jsonl"
    source.write_text(json.dumps({"model": "a", "prompt_tokens": 10, "completion_tokens": 2}) + "\n")
    with pytest.raises(ValueError, match="missing recorded"):
        build([source], ["a", "b"], "test")
    source.write_text(json.dumps({"model": "a", "prompt_tokens": 0, "completion_tokens": 0}) + "\n")
    with pytest.raises(ValueError, match="missing recorded"):
        build([source], ["a"], "test")


def test_checked_in_observations_reproduce_candidate_manifest():
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "data/benchmark_tokens_v1.json").read_text())
    actual = build([root / "data/observations/benchmark_tokens_v1.jsonl"],
                   [m["id"] for m in expected["models"]], expected["worker_profile"])
    assert actual == expected
