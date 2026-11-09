"""The provisioning channel's own rules, each held by a test.

Planted bugs (2026-11-08) showed 9 of 22 rules in tee/provision.py untested,
among them the provisioned VALUES (the operator's OpenRouter key, the hotkey
keyfile) reaching the log, a nonce staying usable after its push was
refused, a malformed envelope crashing the handler instead of answering 400,
and the bound on remembered nonces. Built on the real receiver and the fake
TD of test_provision_transport.
"""
import hashlib
import logging
import urllib.error

import pytest

from fugal_subnet.tee import provision
from fugal_subnet.tee.provision import ProvisionError, ProvisionStore, report_data_for, seal
from tests.test_provision_transport import (
    PAYLOAD,
    _Fake,
    _free_port,
    _post,
    _push,
    _real_receiver,
)

SECRET_KEYFILE = "c2VjcmV0LWtleWZpbGUtYnl0ZXM="          # base64 of a fake keyfile


@pytest.fixture
def receiver(monkeypatch):
    port = _free_port()
    store, server, _calls = _real_receiver(monkeypatch, port)
    yield store, f"http://127.0.0.1:{port}"
    server.shutdown()


def _attest(base):
    nonce = provision.new_nonce()
    ans = _post(f"{base}/provision/attest", {"nonce": nonce})
    return nonce, bytes.fromhex(ans["pubkey"])


def _status(base, body):
    try:
        return 200, _post(f"{base}/provision", body)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_the_pushed_values_never_reach_any_log(receiver, caplog):
    """CLAUDE.md: never log any part of OPENROUTER_API_KEY. The receiver logs
    field NAMES; the values are what the whole design protects."""
    store, base = receiver
    payload = {**PAYLOAD, "hotkey_keyfile_b64": SECRET_KEYFILE}
    with caplog.at_level(logging.DEBUG):
        nonce, pub = _attest(base)
        env, _ = seal(payload, pub, nonce)
        assert _status(base, env)[0] == 200
    assert store.ready
    text = caplog.text + "".join(str(r.args) for r in caplog.records)
    assert "openrouter_api_key" in text                    # the name is logged
    for secret in (PAYLOAD["openrouter_api_key"], SECRET_KEYFILE):
        assert secret not in text


def test_a_nonce_is_single_use_even_when_its_push_is_refused(receiver):
    store, base = receiver
    nonce, pub = _attest(base)
    assert _status(base, {**PAYLOAD, "nonce": nonce})[0] == 400          # plaintext: refused
    env, _ = seal(PAYLOAD, pub, nonce)                                   # now properly sealed
    code, body = _status(base, env)
    assert code == 400 and "already used" in body and not store.ready


@pytest.mark.parametrize("epk_len,iv_len", [(31, 12), (33, 12), (32, 11), (32, 13)])
def test_an_envelope_of_the_wrong_shape_is_a_400_not_a_crash(receiver, epk_len, iv_len):
    store, base = receiver
    nonce, _pub = _attest(base)
    code, body = _status(base, {"nonce": nonce, "epk": "00" * epk_len,
                                "iv": "00" * iv_len, "ct": "00" * 32})
    assert code == 400 and "wrong length" in body and not store.ready


def test_remembered_nonces_are_bounded_oldest_first(receiver):
    store, base = receiver
    attested = [_attest(base) for _ in range(provision._MAX_PENDING_NONCES + 3)]
    (old_nonce, pub), (new_nonce, _) = attested[0], attested[-1]
    env, _ = seal(PAYLOAD, pub, old_nonce)
    assert _status(base, env)[0] == 400                    # forgotten
    env, _ = seal(PAYLOAD, pub, new_nonce)
    assert _status(base, env)[0] == 200 and store.ready


def test_report_data_is_the_nonce_then_the_hash_of_the_key():
    nonce, key = "ab" * 32, bytes(range(32))
    assert report_data_for(nonce, key) == bytes.fromhex(nonce) + hashlib.sha256(key).digest()
    for bad in ("ab" * 31, "ab" * 33):
        with pytest.raises(ProvisionError, match="nonce must be"):
            report_data_for(bad, key)
    with pytest.raises(ProvisionError, match="32 bytes"):
        report_data_for(nonce, key[:31])


def test_push_refuses_a_td_that_offers_no_key(monkeypatch):
    fake = _Fake(monkeypatch, offered_pubkey=b"")
    try:
        with pytest.raises(ProvisionError, match="no X25519 public key"):
            _push(fake)
        assert fake.raw_pushed is None
    finally:
        fake.stop()


def test_push_refuses_an_attestation_with_no_event_log_when_identity_is_expected(monkeypatch):
    fake = _Fake(monkeypatch)
    monkeypatch.setattr("fugal_subnet.tee.verify.unwrap_attestation", lambda blob: (b"quote", None))
    try:
        with pytest.raises(ProvisionError, match="no event log"):
            _push(fake)
        assert fake.raw_pushed is None
    finally:
        fake.stop()


def test_the_store_is_one_push_and_in_memory():
    s = ProvisionStore()
    s.accept(dict(PAYLOAD), b"k")
    with pytest.raises(ProvisionError, match="already provisioned"):
        s.accept(dict(PAYLOAD), b"other")
    assert s.session_key == b"k"
