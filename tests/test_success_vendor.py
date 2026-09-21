"""The offline projection must be proven by the actual pinned Git object graph."""
import json
from pathlib import Path

import pytest

from scripts.check_success_vendor import verify_source


def test_source_proof_rejects_modified_code_even_if_sha256_list_is_changed():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "fugal_subnet/vendor/SOURCE.json").read_text())
    proof = json.loads((root / "fugal_subnet/vendor/source_proof.json").read_text())
    data = {name: (root / "tests/core_snapshot" / name).read_bytes() for name in source["test_snapshot"]}
    verify_source(source, proof, data)
    data["fugal/router.py"] += b"\n# modified\n"
    with pytest.raises(AssertionError, match="blob"):
        verify_source(source, proof, data)


def test_source_proof_rejects_a_different_commit_pin():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "fugal_subnet/vendor/SOURCE.json").read_text())
    proof = json.loads((root / "fugal_subnet/vendor/source_proof.json").read_text())
    source["commit"] = "0" * 40
    with pytest.raises(AssertionError, match="commit"):
        verify_source(source, proof, {})
