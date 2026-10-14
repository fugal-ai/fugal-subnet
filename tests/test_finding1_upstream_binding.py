"""Diligence 2026-10, finding 1: the model upstream is miner-controlled.

The enclave reads its model endpoint from FUGAL_OPENROUTER_BASE and holds every
gold answer, so a miner pointing the variable at their own server reports
perfect accuracy at two tokens of cost with every signature intact. The binding
that closes this already exists: the dstack compose hash (which fixes the
environment) is replayed from the RTMR3 event log and compared against
`<base>:<compose_hash>` entries. What is missing is making that binding
MANDATORY. A bare `<base>` entry skips the application check entirely, and a
live validator merely logs a warning when one is configured.

These tests pin the required behaviour: `parse_approved(require_app=True)`
refuses a bare base, the live validator passes it at startup, and
`verify_proof` refuses a bare-base match outside mock whatever the caller did.
"""
from __future__ import annotations

import pytest

from fugal_subnet.tee import verify as verify_mod


def test_parse_approved_can_demand_an_application_identity():
    """`parse_approved(entries, require_app=True)` must refuse a bare base.

    A bare base means "any application on this image", which after finding 1
    is never a production configuration. The refusal must be loud, like the
    existing trailing-colon case, not a warning.
    """
    base = "ab" * 32
    with pytest.raises(ValueError, match="app"):
        verify_mod.parse_approved([base], require_app=True)
    # The explicit form still parses, and mock-style callers keep the old default.
    assert verify_mod.parse_approved([f"{base}:{'cd' * 32}"], require_app=True)
    assert verify_mod.parse_approved([base]) == {base: set()}


def test_live_verification_rejects_a_proof_under_a_bare_base_entry(monkeypatch):
    """Outside mock, a proof that matches only a bare base must not verify.

    Modelled with the attack suite's own builders: genuine hardware (DCAP
    stubbed to pass), honest MRTD, arbitrary RTMR3. Today this proof is VALID
    because the approved entry names no application.
    """
    from fugal_subnet.attacks import run_tee_attacks as suite

    monkeypatch.setattr(verify_mod, "verify_dcap", lambda q, **_kw: True)
    proof = suite._proof([suite._qr(q) for q in suite.SLICE])
    bare = {suite.BASE}
    result = suite._verify(proof, approved_measurements=bare)
    assert not result.valid, "a bare base entry approved an arbitrary application"
    assert "app" in result.reason.lower() or "bare" in result.reason.lower()
