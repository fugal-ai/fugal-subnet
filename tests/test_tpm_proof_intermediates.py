"""Proof-carried TPM intermediates (decision doc item 3, 2026-11-04).

`verify_ak_chain` always accepted `extra_certs`, and INVARIANTS said a proof
could carry its own intermediates, but nothing passed them: only the vendored
us-central1 intermediate could ever chain. A Google rotation, or a miner in
another region, would have failed every proof at once.

`BenchmarkProof.tpm_intermediates` now carries them to `unwrap_attestation`.
These tests pin that:
- the field changes no hash, and old proofs serialise byte-identically;
- it is bounded (I2);
- with the vendored intermediate removed, a real attestation fails without the
  field and verifies with it (the rotation case);
- a supplied certificate that does not chain to the pinned root is refused.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from fugal_subnet.tee import tpm as tpm_mod
from fugal_subnet.tee import verify as verify_mod
from fugal_subnet.tee.proof import (
    TPM_INTERMEDIATE_MAX_BYTES,
    TPM_INTERMEDIATES_MAX,
    BenchmarkProof,
    QuestionResult,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
ROOTS = pathlib.Path(tpm_mod.__file__).parent / "roots"
VENDORED = (ROOTS / "google_ek_ak_intermediate_us-central1.der").read_bytes()
EMPTY = hashlib.sha256(b"").hexdigest()


def _proof(intermediates=None):
    p = BenchmarkProof(
        epoch_id="e1", nonce="n" * 64, questions_hash="q" * 64, weights_hash="w" * 64,
        source_hash="s" * 64, results=[QuestionResult("q1", "m", True, 0.001, "ab" * 32, 10, 20)],
        total_cost_usd=0.001, per_model_costs={"m": 0.001},
        attestation_quote=b"\x01\x02", timestamp=1.0, hotkey="5hot",
    )
    if intermediates is not None:
        p.tpm_intermediates = intermediates
    return p


def test_intermediates_change_no_hash():
    assert _proof([VENDORED]).content_hash() == _proof().content_hash()


def test_an_old_proof_serialises_byte_identically():
    d = _proof().to_dict()
    assert "tpm_intermediates" not in d
    assert BenchmarkProof.from_dict(d).tpm_intermediates == []


def test_intermediates_round_trip_through_json():
    wire = json.loads(json.dumps(_proof([VENDORED, b"\x30\x00"]).to_dict()))
    assert BenchmarkProof.from_dict(wire).tpm_intermediates == [VENDORED, b"\x30\x00"]


@pytest.mark.parametrize("bad", [
    "not a list",
    ["00"] * (TPM_INTERMEDIATES_MAX + 1),
    ["00" * (TPM_INTERMEDIATE_MAX_BYTES + 1)],
    [123],
    ["zz"],
])
def test_out_of_bounds_intermediates_are_a_malformed_proof(bad):
    d = _proof().to_dict()
    d["tpm_intermediates"] = bad
    with pytest.raises(ValueError):
        BenchmarkProof.from_dict(d)


def _rotated(monkeypatch):
    """Google has rotated: the vendored intermediate no longer matches anything."""
    real = tpm_mod._load_pinned
    monkeypatch.setattr(tpm_mod, "_load_pinned", lambda *a, **k: (real(*a, **k)[0], []))


def _blob(name="A"):
    p = FIXTURES / f"attestation_{name}.bin"
    if not p.exists():
        pytest.skip(f"fixture {p.name} not present")
    return p.read_bytes()


def test_after_a_rotation_a_real_attestation_needs_its_intermediate(monkeypatch):
    blob = _blob()
    _rotated(monkeypatch)
    with pytest.raises(ValueError, match="TPM quote"):
        verify_mod.unwrap_attestation(blob)
    quote, _events = verify_mod.unwrap_attestation(blob, extra_certs=[VENDORED])
    assert quote


def test_after_a_rotation_a_rogue_intermediate_does_not_help(monkeypatch):
    from fugal_subnet.attacks.run_tee_attacks import _rogue_chain

    blob = _blob()
    _rotated(monkeypatch)
    _leaf, rogue_ca = _rogue_chain(tpm_mod._load_pinned()[0].subject)
    with pytest.raises(ValueError, match="TPM quote"):
        verify_mod.unwrap_attestation(blob, extra_certs=[rogue_ca])


def test_verify_proof_passes_the_proofs_intermediates(monkeypatch):
    seen = {}

    def fake_unwrap(blob, extra_certs=None):
        seen["extra"] = extra_certs
        raise ValueError("stop here")

    monkeypatch.setattr(verify_mod, "unwrap_attestation", fake_unwrap)
    r = verify_mod.verify_proof(
        _proof([VENDORED]), approved_measurements=set(), expected_questions_hash="q" * 64,
        expected_nonce="n" * 64, gold_answers={}, expected_hotkey="5hot", mock=True,
    )
    assert not r.valid and seen["extra"] == [VENDORED]


def test_an_unparseable_intermediate_is_a_rejected_attestation():
    """A miner sends junk in tpm_intermediates. The TPM check raises (the
    certificate does not parse), and that must refuse the envelope, never
    count as a TPM half that verified."""
    with pytest.raises(ValueError, match="TPM quote rejected"):
        verify_mod.unwrap_attestation(_blob(), extra_certs=[b"not a certificate"])
    # The same envelope with nothing extra still verifies (vendored intermediate).
    quote, _events = verify_mod.unwrap_attestation(_blob())
    assert quote
