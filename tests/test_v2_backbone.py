from __future__ import annotations

import hashlib
import json
import platform

import numpy as np
import pytest

from fugal_subnet.v2 import backbone


def test_backbone_spec_is_pinned_and_hashable():
    spec = backbone.BackboneSpec()
    assert spec.model_revision == spec.tokenizer_revision
    assert len(spec.model_revision) == 40
    assert spec.device == "cpu"
    assert spec.dtype == "float32"
    assert len(spec.sha256) == 64


def test_unsupported_platform_fails_closed(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    with pytest.raises(backbone.BackboneEnvironmentError, match="Linux x86-64"):
        backbone.require_supported_host()


def test_empty_prompt_list_does_not_download_model():
    result = backbone.compute_hidden_states([])
    assert result.shape == (0, backbone.HIDDEN_DIM)
    assert result.dtype == np.float32


def test_prompt_schema_is_strict():
    with pytest.raises(ValueError, match="list of strings"):
        backbone.compute_hidden_states(("not", "a", "list"))  # type: ignore[arg-type]


def test_backbone_golden_checks_prompt_and_embedding_hashes(monkeypatch):
    values = np.arange(4 * backbone.HIDDEN_DIM, dtype=np.float32).reshape(
        4, backbone.HIDDEN_DIM
    )
    monkeypatch.setattr(backbone, "compute_hidden_states", lambda _prompts: values)
    expected = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    prompt_bytes = json.dumps(
        list(backbone.GOLDEN_PROMPTS),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()

    assert backbone.verify_backbone_golden(
        expected_prompts_sha256=prompt_hash,
        expected_embeddings_sha256=expected,
    ) == expected
    with pytest.raises(backbone.BackboneEnvironmentError, match="prompts"):
        backbone.verify_backbone_golden(
            expected_prompts_sha256="0" * 64,
            expected_embeddings_sha256=expected,
        )


def test_cpu_capability_is_pinned_before_torch_dispatch():
    """AVX-512 and AVX2 hosts otherwise compute different embedding bits."""
    import os

    import torch

    assert backbone.CPU_CAPABILITY == "avx2"
    assert os.environ["ATEN_CPU_CAPABILITY"] == "avx2"
    assert str(torch.backends.cpu.get_cpu_capability()).upper() == "AVX2"
    assert backbone.BackboneSpec().cpu_capability == "avx2"


def test_capability_mismatch_fails_closed(monkeypatch):
    """A wider kernel set must abort rather than silently diverge."""
    import torch

    monkeypatch.setattr(backbone, "_configured", False)
    monkeypatch.setattr(
        torch.backends.cpu, "get_cpu_capability", lambda: "AVX512"
    )
    try:
        backbone.configure_determinism()
    except backbone.BackboneEnvironmentError as exc:
        assert "AVX512" in str(exc)
    else:
        raise AssertionError("capability mismatch must fail closed")
    finally:
        backbone._configured = False
