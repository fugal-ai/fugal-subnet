"""Each verification step in `push` must be individually load-bearing — and so
must the seal.

A miner releases its API key and its hotkey on the strength of these checks. If
any one of them can be removed without a test failing, it was decoration. So
each test here defeats exactly one and requires the push to be refused.

The order matters as much as the checks: the event log arrives from the thing
being authenticated, so its payloads must not be read until the replay has
reproduced the RTMR3 the hardware signed. `test_log_is_not_read_before_it_is_
authenticated` is the one that pins that, by giving a log with a PERFECT
compose-hash and instance-id but a broken chain.

The seal tests exist because an earlier version of this channel verified the
quote and then POSTed the secrets in plaintext. The quote now covers the TD's
ephemeral public key as well as the nonce; the payload is encrypted to it; a
plaintext push, a push to a key the hardware did not sign for, and a replayed
envelope are each refused by the TD.
"""
import base64
import hashlib
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from cryptography.hazmat.primitives.asymmetric import x25519

from fugal_subnet.tee import provision
from fugal_subnet.tee.provision import (
    ProvisionError,
    ProvisionStore,
    RingLogHandler,
    log_token,
    pull_logs,
    push,
    report_data_for,
    seal,
    unseal,
    validate_payload,
)

APPROVED = "a" * 64
APP = "b" * 64
INSTANCE = "c" * 64
PAYLOAD = {"openrouter_api_key": "sk-or-v1-secret", "hotkey_ss58": "5Fk"}


def _events(compose=APP, instance=INSTANCE):
    """An RTMR3 log whose digests are real sha384s, so replay is meaningful."""
    out = []
    for name, payload in (("compose-hash", compose), ("instance-id", instance)):
        raw = bytes.fromhex(payload)
        out.append({
            "imr": 3, "event_type": 0x08000001,
            "digest": hashlib.sha384(name.encode() + raw).hexdigest(),
            "event": name, "event_payload": raw.hex(),
        })
    return out


class _Fake:
    """A TD that answers /provision/attest with whatever we tell it to, holds a
    real X25519 key, and decrypts what it is pushed — so `pushed` is the
    PLAINTEXT the TD recovered, or None."""

    def __init__(self, monkeypatch, *, report_data=None, measurement=APPROVED,
                 rtmr3=None, events=None, offered_pubkey=None):
        self.pushed = None
        self.raw_pushed = None
        self.key = x25519.X25519PrivateKey.generate()
        self.pub = self.key.public_key().public_bytes_raw()
        # What the TD OFFERS may differ from what the quote covers (attack case).
        self.offered = offered_pubkey if offered_pubkey is not None else self.pub
        log = _events() if events is None else events
        reg = rtmr3
        if reg is None:
            reg = b"\x00" * 48
            for ev in log:
                reg = hashlib.sha384(reg + bytes.fromhex(ev["digest"])).digest()
            reg = reg.hex()
        self._log, self._reg, self._meas, self._rd = log, reg, measurement, report_data

        parent = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/provision/attest":
                    # The quote covers nonce || sha256(the REAL key) unless the
                    # test forces a specific report_data.
                    rd = parent._rd or report_data_for(body["nonce"], parent.pub).hex()
                    out = {"attestation": "00", "pubkey": parent.offered.hex()}
                    parent._rd_used = rd
                else:
                    parent.raw_pushed = body
                    try:
                        parent.pushed, _ = unseal(parent.key, body)
                    except ProvisionError:
                        parent.pushed = None
                    out = {"provisioned": parent.pushed is not None}
                raw = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(raw.__len__()))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):
                pass

        self.http = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.address = f"http://127.0.0.1:{self.http.server_port}"

        class _Q:
            report_data = property(lambda s: parent._rd_used)
            rtmr3 = property(lambda s: parent._reg)

        monkeypatch.setattr("fugal_subnet.tee.verify.unwrap_attestation",
                            lambda blob: (b"quote", parent._log))
        monkeypatch.setattr("fugal_subnet.tee.attestation.parse_quote", lambda b: _Q())
        monkeypatch.setattr("fugal_subnet.tee.attestation.measurement_id",
                            lambda q: parent._meas)

    def stop(self):
        self.http.shutdown()


def _push(fake, **kw):
    kw.setdefault("approved_measurements", {APPROVED})
    kw.setdefault("expected_app_identity", APP)
    kw.setdefault("expected_instance_id", INSTANCE)
    return push(fake.address, PAYLOAD, **kw)


def test_happy_path_pushes_the_payload_sealed(monkeypatch):
    f = _Fake(monkeypatch)
    try:
        key = _push(f)
        assert f.pushed == PAYLOAD
        # What crossed the wire was an envelope, and it carried no secret.
        assert set(f.raw_pushed) == {"nonce", "epk", "iv", "ct"}
        assert "sk-or-v1-secret" not in json.dumps(f.raw_pushed)
        assert isinstance(key, bytes) and len(key) == 32
    finally:
        f.stop()


def test_refuses_a_quote_that_is_not_over_our_nonce(monkeypatch):
    """A captured attestation replayed later carries the wrong nonce."""
    f = _Fake(monkeypatch, report_data=("de" * 32 + "00" * 32))
    try:
        with pytest.raises(ProvisionError, match="not over our nonce"):
            _push(f)
        assert f.raw_pushed is None
    finally:
        f.stop()


def test_refuses_a_key_the_hardware_did_not_sign_for(monkeypatch):
    """The seal's own canary. The quote is over our nonce and the TD's REAL key,
    but the TD offers a DIFFERENT public key to encrypt to — the shape of a
    man-in-the-middle substituting their key after a genuine attestation. If
    this ever passes, the seal encrypts to whoever asks."""
    mitm = x25519.X25519PrivateKey.generate().public_key().public_bytes_raw()
    f = _Fake(monkeypatch, offered_pubkey=mitm)
    try:
        with pytest.raises(ProvisionError, match="not over our nonce"):
            _push(f)
        assert f.raw_pushed is None
    finally:
        f.stop()


def test_refuses_an_unapproved_measurement(monkeypatch):
    f = _Fake(monkeypatch, measurement="f" * 64)
    try:
        with pytest.raises(ProvisionError, match="not approved"):
            _push(f)
        assert f.raw_pushed is None
    finally:
        f.stop()


def test_log_is_not_read_before_it_is_authenticated(monkeypatch):
    """The canary. The log names the RIGHT app and the RIGHT instance, and is
    still refused, because it does not replay to the signed register. If this
    ever passes, the verifier is reading an attacker's claims."""
    f = _Fake(monkeypatch, rtmr3="9" * 96)
    try:
        with pytest.raises(ProvisionError, match="does not replay"):
            _push(f)
        assert f.raw_pushed is None
    finally:
        f.stop()


def test_refuses_a_different_instance(monkeypatch):
    """A cloud provider booting the approved image gets a different instance-id."""
    f = _Fake(monkeypatch, events=_events(instance="d" * 64))
    try:
        with pytest.raises(ProvisionError, match="instance"):
            _push(f)
        assert f.raw_pushed is None
    finally:
        f.stop()


def test_refuses_a_different_app(monkeypatch):
    f = _Fake(monkeypatch, events=_events(compose="e" * 64))
    try:
        with pytest.raises(ProvisionError, match="running app"):
            _push(f)
        assert f.raw_pushed is None
    finally:
        f.stop()


def test_bad_payload_fails_before_contacting_anything(monkeypatch):
    f = _Fake(monkeypatch)
    try:
        with pytest.raises(ProvisionError, match="unknown provisioning field"):
            push(f.address, {"FUGAL_OPENROUTER_BASE": "http://evil"},
                 approved_measurements={APPROVED})
        assert f.raw_pushed is None
    finally:
        f.stop()


def test_store_holds_in_memory_and_reports_readiness():
    s = ProvisionStore()
    assert not s.ready
    s.accept(dict(PAYLOAD), b"k" * 32)
    assert s.ready
    assert s.session_key == b"k" * 32
    assert s.get("openrouter_api_key") == "sk-or-v1-secret"
    assert s.get("head_b64") == ""
    with pytest.raises(ProvisionError):
        s.get("FUGAL_OPENROUTER_BASE")


def test_store_rejects_a_bad_payload_and_stays_unprovisioned():
    s = ProvisionStore()
    with pytest.raises(ProvisionError):
        s.accept({"pool": "x"})
    assert not s.ready
    assert validate_payload(dict(PAYLOAD)) == PAYLOAD


# --- the seal itself, and the real TD-side receiver -------------------------

def test_seal_round_trips_and_binds_the_nonce():
    td = x25519.X25519PrivateKey.generate()
    pub = td.public_key().public_bytes_raw()
    nonce = "ab" * 32
    env, key = seal(PAYLOAD, pub, nonce)
    got, key2 = unseal(td, env)
    assert got == PAYLOAD and key == key2
    # The nonce is associated data: change it and the AEAD tag fails.
    with pytest.raises(ProvisionError, match="does not decrypt"):
        unseal(td, {**env, "nonce": "cd" * 32})
    # A different TD key cannot open it.
    with pytest.raises(ProvisionError, match="does not decrypt"):
        unseal(x25519.X25519PrivateKey.generate(), env)


def test_unseal_refuses_plaintext_and_malformed_envelopes():
    td = x25519.X25519PrivateKey.generate()
    with pytest.raises(ProvisionError, match="plaintext is refused"):
        unseal(td, dict(PAYLOAD))
    with pytest.raises(ProvisionError, match="not hex"):
        unseal(td, {"nonce": "zz", "epk": "00", "iv": "00", "ct": "00"})


def _real_receiver(monkeypatch, port):
    """Run the actual `serve()` with the attest call stubbed: the quote is not
    what these tests are about, the receiver's own refusals are."""
    store = ProvisionStore()
    calls = {}

    def fake_attest(report_data, path=""):
        calls["rd"] = report_data
        return b"\x00" * 8

    monkeypatch.setattr("fugal_subnet.tee.dstack_client.attest", fake_attest)
    server = provision.serve(store, port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return store, server, calls


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def test_real_receiver_attests_over_nonce_and_key_then_accepts_one_sealed_push(monkeypatch):
    store, server, calls = _real_receiver(monkeypatch, 18093)
    try:
        nonce = provision.new_nonce()
        ans = _post("http://127.0.0.1:18093/provision/attest", {"nonce": nonce})
        pub = bytes.fromhex(ans["pubkey"])
        assert calls["rd"] == report_data_for(nonce, pub), "quote must cover nonce || sha256(pubkey)"

        # Plaintext is refused even with a good nonce.
        with pytest.raises(urllib.error.HTTPError) as e:
            _post("http://127.0.0.1:18093/provision", {**PAYLOAD, "nonce": nonce})
        assert e.value.code == 400
        assert not store.ready

        # The nonce was consumed by that attempt; a fresh attest is required.
        nonce = provision.new_nonce()
        ans = _post("http://127.0.0.1:18093/provision/attest", {"nonce": nonce})
        env, key = seal(PAYLOAD, bytes.fromhex(ans["pubkey"]), nonce)
        assert _post("http://127.0.0.1:18093/provision", env) == {"provisioned": True}
        assert store.ready and store.session_key == key

        # Replaying the same envelope is refused: the nonce is spent.
        with pytest.raises(urllib.error.HTTPError) as e:
            _post("http://127.0.0.1:18093/provision", env)
        assert e.value.code == 400 and "already used" in e.value.read().decode()

        # A never-attested nonce is refused before any decryption is attempted.
        env2, _ = seal(PAYLOAD, bytes.fromhex(ans["pubkey"]), provision.new_nonce())
        with pytest.raises(urllib.error.HTTPError) as e:
            _post("http://127.0.0.1:18093/provision", env2)
        assert e.value.code == 400 and "not attested" in e.value.read().decode()
    finally:
        server.shutdown()


def test_logs_are_operator_only_and_encrypted(monkeypatch):
    import logging

    store, server, _ = _real_receiver(monkeypatch, 18094)
    try:
        # Nobody without the session key gets anything, not even a 200.
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen("http://127.0.0.1:18094/provision/logs", timeout=5)
        assert e.value.code == 403

        nonce = provision.new_nonce()
        ans = _post("http://127.0.0.1:18094/provision/attest", {"nonce": nonce})
        env, key = seal(PAYLOAD, bytes.fromhex(ans["pubkey"]), nonce)
        _post("http://127.0.0.1:18094/provision", env)
        logging.getLogger("fugal.test").warning("a line the operator should see: %d", 7)

        seq, lines = pull_logs("http://127.0.0.1:18094", key)
        assert any("a line the operator should see: 7" in ln for ln in lines)
        assert seq >= 1
        # Incremental: nothing new since seq.
        seq2, more = pull_logs("http://127.0.0.1:18094", key, since=seq)
        assert more == [] and seq2 == seq

        # A wrong key is refused at the token, before any bytes are sealed.
        with pytest.raises(urllib.error.HTTPError) as e:
            req = urllib.request.Request(
                "http://127.0.0.1:18094/provision/logs",
                headers={"Authorization": f"Bearer {log_token(b'x' * 32)}"})
            urllib.request.urlopen(req, timeout=5)
        assert e.value.code == 403
    finally:
        server.shutdown()


def test_ring_log_handler_is_bounded_and_incremental():
    h = RingLogHandler(capacity=3)
    import logging
    for i in range(5):
        h.emit(logging.LogRecord("t", logging.INFO, "f", 1, "line %d", (i,), None))
    seq, lines = h.since(0)
    assert seq == 5 and [ln.split()[-1] for ln in lines] == ["2", "3", "4"]
    assert h.since(4) == (5, [ln for ln in lines if ln.endswith("4")])


def test_miner_blocks_until_provisioned_then_returns_the_head(monkeypatch):
    """The whole point of the wait: no head, no serving.

    Runs the miner's own `_await_provisioning` in a thread, confirms it is still
    blocked with nothing pushed, then pushes (sealed, through the real
    receiver) and confirms it returns the exact head bytes and puts the key
    where MeteringProxy reads it.
    """
    import importlib
    import os
    import time

    miner = importlib.import_module("neurons.miner")
    head = b"\x93NUMPY-not-really-but-bytes"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("fugal_subnet.tee.dstack_client.attest", lambda rd, path="": b"\x00" * 8)

    result = {}

    def run():
        result["head"], result["wallet_path"] = miner._await_provisioning(
            coldkey="c", hotkey="h", wallet_path="/original/wallets",
        )

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.6)
    assert "head" not in result, "returned before anything was pushed"

    from fugal_subnet.tee.provision import PROVISION_PORT
    base = f"http://127.0.0.1:{PROVISION_PORT}"
    nonce = provision.new_nonce()
    ans = _post(f"{base}/provision/attest", {"nonce": nonce})
    env, _ = seal({
        "head_b64": base64.b64encode(head).decode(),
        "openrouter_api_key": "sk-or-v1-pushed",
    }, bytes.fromhex(ans["pubkey"]), nonce)
    _post(f"{base}/provision", env)

    t.join(timeout=10)
    assert result.get("head") == head
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-v1-pushed"
    # No keyfile pushed: the caller's own wallet path is untouched.
    assert result.get("wallet_path") == "/original/wallets"


def test_a_pushed_wallet_lands_in_tmpfs_with_private_modes(monkeypatch, tmp_path):
    """The TD signs serve_axon and the head commitment itself, so the keyfile
    must arrive — and it must arrive somewhere private and volatile."""
    import importlib
    import os
    import stat
    import time

    miner = importlib.import_module("neurons.miner")
    monkeypatch.setenv("FUGAL_PROVISIONED_WALLET_DIR", str(tmp_path / "shm"))
    monkeypatch.setattr("fugal_subnet.tee.provision.PROVISION_PORT", 18092)
    monkeypatch.setattr("fugal_subnet.tee.dstack_client.attest", lambda rd, path="": b"\x00" * 8)

    keyfile = b'{"accountId":"0x00","ss58Address":"5Fk","secretPhrase":"not real"}'
    coldpub = b'{"ss58Address":"5Cold"}'
    result = {}

    def run():
        result["head"], result["wallet_path"] = miner._await_provisioning(
            coldkey="fugal", hotkey="td1", wallet_path=None,
        )

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.6)
    nonce = provision.new_nonce()
    ans = _post("http://127.0.0.1:18092/provision/attest", {"nonce": nonce})
    env, _ = seal({
        "head_b64": base64.b64encode(b"head").decode(),
        "hotkey_keyfile_b64": base64.b64encode(keyfile).decode(),
        "coldkeypub_b64": base64.b64encode(coldpub).decode(),
    }, bytes.fromhex(ans["pubkey"]), nonce)
    _post("http://127.0.0.1:18092/provision", env)
    t.join(timeout=10)

    root = result.get("wallet_path")
    assert root == str(tmp_path / "shm")
    hot = os.path.join(root, "fugal", "hotkeys", "td1")
    pub = os.path.join(root, "fugal", "coldkeypub.txt")
    assert open(hot, "rb").read() == keyfile
    assert open(pub, "rb").read() == coldpub
    for p in (hot, pub):
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600, f"{p} is not 0600"
    for d in (root, os.path.join(root, "fugal")):
        assert stat.S_IMODE(os.stat(d).st_mode) == 0o700, f"{d} is not 0700"
