"""The I8 app-identity chain, through `verify_proof` itself (live path).

    approved entry  <base>:<compose_hash>, 64 hex each
    measurement_id(quote) == base
    the RTMR3 event log replays to the quote's RTMR3
    the log's compose-hash payload == compose_hash

Built on the attack suite's builders (genuine-looking quote, DCAP stubbed to
pass) so each case differs from the honest control by one thing.
"""
import hashlib

import pytest

from fugal_subnet.attacks import run_tee_attacks as suite
from fugal_subnet.tee import verify as verify_mod
from fugal_subnet.tee.attestation import replay_rtmr


@pytest.fixture(autouse=True)
def _dcap_passes(monkeypatch):
    monkeypatch.setattr(verify_mod, "verify_dcap", lambda q, **_kw: True)


def _honest_proof(**kw):
    return suite._proof([suite._qr(q) for q in suite.SLICE], **kw)


def test_the_honest_control_verifies_through_the_log():
    r = suite._verify(_honest_proof())
    assert r.valid, r.reason


# --- the approved entry: one live shape -------------------------------------

BASE = "ab" * 32
COMPOSE = "cd" * 32


@pytest.mark.parametrize("entry, why", [
    (f"{BASE}:{'ef' * 48}", "a 96-hex runtime identity"),
    (f"{BASE}:{COMPOSE[:-2]}", "a short compose hash"),
    (f"{BASE}:{'zz' * 32}", "a non-hex compose hash"),
    (f"{BASE[:-2]}:{COMPOSE}", "a short base"),
    (f"{BASE}:{COMPOSE}:{COMPOSE}", "a third half"),
])
def test_live_parse_accepts_only_two_64_hex_halves(entry, why):
    with pytest.raises(ValueError, match="64 hex"):
        verify_mod.parse_approved([entry], require_app=True)


def test_live_parse_normalises_case():
    parsed = verify_mod.parse_approved([f"{BASE.upper()}:{COMPOSE.upper()}"],
                                       require_app=True)
    assert parsed == {BASE: {COMPOSE}}


def test_mock_parse_stays_permissive():
    """Mock binds no application; local testnets use free-form entries."""
    assert verify_mod.parse_approved(["base1:app_a"]) == {"base1": {"app_a"}}


# --- verify_proof: only a replayed compose hash is evidence ------------------

def test_a_runtime_identity_entry_is_refused_even_when_the_register_matches():
    """The miner's own code can extend any value into RTMR3, so a register one
    extend away from an approved runtime identity proves nothing."""
    identity = hashlib.sha384(b"src|pool|grader|upstream").hexdigest()
    proof = _honest_proof(rtmr3=bytes.fromhex(replay_rtmr([identity])))
    r = suite._verify(proof, approved_measurements={f"{suite.BASE}:{identity}"},
                      event_log=None)
    assert not r.valid
    assert "names no compose hash" in r.reason


def test_a_proof_without_an_event_log_is_refused():
    r = suite._verify(_honest_proof(), event_log=None)
    assert not r.valid
    assert "no TDX event log" in r.reason


def test_a_compose_hash_entry_beside_a_runtime_identity_still_needs_the_log():
    """A transition list may hold both; only the compose hash is checkable."""
    identity = "ef" * 48
    approved = {f"{suite.BASE}:{identity}", f"{suite.BASE}:{suite.COMPOSE}"}
    assert suite._verify(_honest_proof(), approved_measurements=approved).valid
    proof = _honest_proof(rtmr3=bytes.fromhex(replay_rtmr([identity])))
    r = suite._verify(proof, approved_measurements=approved, event_log=None)
    assert not r.valid
