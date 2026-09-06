"""Each verification step in `push` must be individually load-bearing.

A miner releases its API key on the strength of these checks. If any one of
them can be removed without a test failing, it was decoration. So each test
here defeats exactly one and requires the push to be refused.

The order matters as much as the checks: the event log arrives from the thing
being authenticated, so its payloads must not be read until the replay has
reproduced the RTMR3 the hardware signed. `test_log_is_not_read_before_it_is_
authenticated` is the one that pins that, by giving a log with a PERFECT
compose-hash and instance-id but a broken chain.
"""
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from fugal_subnet.tee.provision import ProvisionError, ProvisionStore, push, validate_payload

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
    """A TD that answers /provision/attest with whatever we tell it to."""

    def __init__(self, monkeypatch, *, report_data=None, measurement=APPROVED,
                 rtmr3=None, events=None):
        self.pushed = None
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
                    rd = parent._rd or bytes.fromhex(body["nonce"]).ljust(64, b"\0").hex()
                    out = {"attestation": "00"}
                    parent._nonce_seen = body["nonce"]
                    parent._rd_used = rd
                else:
                    parent.pushed = body
                    out = {"provisioned": True}
                raw = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
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


def test_happy_path_pushes_the_payload(monkeypatch):
    f = _Fake(monkeypatch)
    try:
        _push(f)
        assert f.pushed == PAYLOAD
    finally:
        f.stop()


def test_refuses_a_quote_that_is_not_over_our_nonce(monkeypatch):
    """A captured attestation replayed later carries the wrong nonce."""
    f = _Fake(monkeypatch, report_data=("de" * 32 + "00" * 32))
    try:
        with pytest.raises(ProvisionError, match="not over our nonce"):
            _push(f)
        assert f.pushed is None
    finally:
        f.stop()


def test_refuses_an_unapproved_measurement(monkeypatch):
    f = _Fake(monkeypatch, measurement="f" * 64)
    try:
        with pytest.raises(ProvisionError, match="not approved"):
            _push(f)
        assert f.pushed is None
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
        assert f.pushed is None
    finally:
        f.stop()


def test_refuses_a_different_instance(monkeypatch):
    """A cloud provider booting the approved image gets a different instance-id."""
    f = _Fake(monkeypatch, events=_events(instance="d" * 64))
    try:
        with pytest.raises(ProvisionError, match="instance"):
            _push(f)
        assert f.pushed is None
    finally:
        f.stop()


def test_refuses_a_different_app(monkeypatch):
    f = _Fake(monkeypatch, events=_events(compose="e" * 64))
    try:
        with pytest.raises(ProvisionError, match="running app"):
            _push(f)
        assert f.pushed is None
    finally:
        f.stop()


def test_bad_payload_fails_before_contacting_anything(monkeypatch):
    f = _Fake(monkeypatch)
    try:
        with pytest.raises(ProvisionError, match="unknown provisioning field"):
            push(f.address, {"FUGAL_OPENROUTER_BASE": "http://evil"},
                 approved_measurements={APPROVED})
        assert f.pushed is None
    finally:
        f.stop()


def test_store_holds_in_memory_and_reports_readiness():
    s = ProvisionStore()
    assert not s.ready
    s.accept(dict(PAYLOAD))
    assert s.ready
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
