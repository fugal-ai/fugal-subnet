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


# --- each link of the chain, through verify_proof ----------------------------

def _log_verify(log, approved=None, rtmr3=None):
    """Verify an honest proof whose quote carries `rtmr3` (default: what `log`
    honestly replays to) against `log`."""
    if rtmr3 is None:
        rtmr3 = bytes.fromhex(replay_rtmr([e["digest"] for e in log]))
    return suite._verify(_honest_proof(rtmr3=rtmr3), event_log=log,
                         approved_measurements=approved or suite.APPROVED)


def test_an_unapproved_image_is_refused():
    other_base = "00" * 32
    r = suite._verify(_honest_proof(),
                      approved_measurements={f"{other_base}:{suite.COMPOSE}"})
    assert not r.valid and "Unapproved runtime image" in r.reason


def test_a_bare_base_is_refused():
    r = suite._verify(_honest_proof(), approved_measurements={suite.BASE})
    assert not r.valid and "bare base" in r.reason


def test_a_log_that_does_not_reproduce_the_quote_is_refused():
    """The claim is approved, the chain is not the one the CPU signed."""
    forged = [dict(suite._HONEST_LOG[0]), suite._ev("instance-id", "0bad")]
    r = _log_verify(forged, rtmr3=suite.HONEST_RTMR3)
    assert not r.valid and "does not reproduce RTMR3" in r.reason


@pytest.mark.parametrize("shape", ["reordered", "truncated", "extended"])
def test_a_reshaped_log_is_refused(shape):
    log = list(suite._HONEST_LOG)
    log = {"reordered": log[::-1], "truncated": log[:1],
           "extended": log + [suite._ev("extra", "00")]}[shape]
    r = _log_verify(log, rtmr3=suite.HONEST_RTMR3)
    assert not r.valid and "does not reproduce RTMR3" in r.reason


def test_a_valid_log_for_an_unapproved_compose_is_refused():
    log = [suite._ev("compose-hash", suite.OTHER_COMPOSE), suite._ev("instance-id", "beef")]
    r = _log_verify(log)
    assert not r.valid and "Unapproved application" in r.reason


def test_a_valid_log_with_no_compose_hash_is_refused():
    r = _log_verify([suite._ev("instance-id", "beef")])
    assert not r.valid and "Unapproved application" in r.reason


@pytest.mark.parametrize("version", [1, 2])
def test_a_relabelled_compose_payload_is_refused(version):
    """Genuine digests for the miner's compose, payload rewritten to the
    approved one: same register, so only the digest binding refuses it."""
    genuine = [suite._ev("compose-hash", suite.OTHER_COMPOSE, version),
               suite._ev("instance-id", "beef", version)]
    rtmr3 = bytes.fromhex(replay_rtmr([e["digest"] for e in genuine]))
    forged = [dict(genuine[0], event_payload=suite.COMPOSE), genuine[1]]
    r = _log_verify(forged, rtmr3=rtmr3)
    assert not r.valid and "does not bind" in r.reason


def test_a_second_compose_hash_is_refused():
    log = [suite._ev("compose-hash", suite.OTHER_COMPOSE), suite._ev("instance-id", "beef"),
           suite._ev("compose-hash", suite.COMPOSE)]
    r = _log_verify(log)
    assert not r.valid and "repeats the name" in r.reason


def test_either_compose_in_a_transition_window_verifies():
    approved = {f"{suite.BASE}:{suite.COMPOSE}", f"{suite.BASE}:{suite.OTHER_COMPOSE}"}
    for compose in (suite.COMPOSE, suite.OTHER_COMPOSE):
        log = [suite._ev("compose-hash", compose), suite._ev("instance-id", "beef")]
        assert _log_verify(log, approved=approved).valid


def test_the_log_inside_the_attestation_is_the_one_checked(monkeypatch):
    """The live validator passes no event_log: the log comes out of the dstack
    blob the miner sent. Pin that wiring, honest and forged."""
    real_unwrap = verify_mod.unwrap_attestation
    for log, valid in ((suite._HONEST_LOG, True),
                       ([dict(suite._HONEST_LOG[0]), suite._ev("instance-id", "0bad")], False)):
        monkeypatch.setattr(verify_mod, "unwrap_attestation",
                            lambda blob, extra_certs=None, _log=log: (real_unwrap(blob)[0], _log))
        r = suite._verify(_honest_proof(), event_log=None)
        assert r.valid is valid, r.reason


def test_a_resplit_compose_hash_name_is_refused():
    """The first encoding joins type, name and payload with ':'. The miner's own
    compose hash starts with ':', so its genuine digest can be presented as the
    name 'compose-hash:', leaving a later approved extend as the only one."""
    i = 0
    while (mine := hashlib.sha256(b'{"compose": %d}' % i).digest())[0] != ord(":"):
        i += 1
    genuine = [suite._ev("compose-hash", mine.hex(), 1), suite._ev("instance-id", "beef", 1),
               suite._ev("compose-hash", suite.COMPOSE, 1)]
    forged = [dict(genuine[0], event="compose-hash:", event_payload=mine[1:].hex()),
              genuine[1], genuine[2]]
    r = _log_verify(forged, rtmr3=bytes.fromhex(replay_rtmr([e["digest"] for e in genuine])))
    assert not r.valid and "contains ':'" in r.reason


def test_events_for_other_registers_are_not_replayed_into_rtmr3():
    """A real dstack log carries the boot-time IMR 0-2 events beside IMR 3's;
    only IMR 3's are extended into RTMR3. An honest log must still verify."""
    log = [dict(suite._ev("boot-event", "11"), imr=0), *suite._HONEST_LOG,
           dict(suite._ev("other", "22"), imr=1)]
    r = _log_verify(log, rtmr3=suite.HONEST_RTMR3)
    assert r.valid, r.reason
