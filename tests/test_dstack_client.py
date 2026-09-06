"""The dstack guest agent client, against a fake agent serving the real blob.

The socket cannot be reached from outside the VM, so the transport is tested
against a stand-in. What makes that worth doing is that the stand-in replies
with the ACTUAL 39,471-byte attestation from a real deploy: the bytes that come
back are the bytes a live agent returns, so everything downstream of the
transport is exercised for real.
"""
import http.server
import json
import pathlib
import socket
import threading

import pytest

from fugal_subnet.tee import dstack_client
from fugal_subnet.tee.dstack_client import DstackError, attest, available

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


class _Agent(http.server.BaseHTTPRequestHandler):
    """Stand-in guest agent. Set _Agent.reply / _Agent.status per test."""

    reply: object = None
    status: int = 200
    seen: dict = {}

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's interface
        n = int(self.headers.get("Content-Length", 0))
        _Agent.seen = {"path": self.path, "body": json.loads(self.rfile.read(n) or b"{}")}
        body = (_Agent.reply if isinstance(_Agent.reply, bytes)
                else json.dumps(_Agent.reply).encode())
        self.send_response(_Agent.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a):
        pass


class _UnixServer(http.server.HTTPServer):
    address_family = socket.AF_UNIX

    def server_bind(self):
        self.socket.bind(self.server_address)
        self.server_name, self.server_port = "localhost", 0


@pytest.fixture
def agent(tmp_path):
    path = str(tmp_path / "dstack.sock")
    srv = _UnixServer(path, _Agent)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield path
    srv.shutdown()
    srv.server_close()


def _blob():
    p = FIXTURES / "attestation_A.bin"
    if not p.exists():
        pytest.skip("fixture not present")
    return p.read_bytes()


def test_a_real_attestation_round_trips_through_the_socket(agent):
    blob = _blob()
    _Agent.reply = {"attestation": blob.hex(), "boottime_gpu_evidence": []}
    assert attest(b"\xab" * 32, path=agent) == blob
    assert _Agent.seen["path"] == "/v1/Attest"


def test_report_data_is_padded_here_rather_than_by_the_agent(agent):
    """The guest does pad a short value, but that is behaviour we observed and
    do not control, and these exact bytes are what bind a proof to its body."""
    _Agent.reply = {"attestation": _blob().hex()}
    attest(b"fugal", path=agent)
    assert _Agent.seen["body"]["report_data"] == b"fugal".hex() + "00" * 59


def test_the_result_is_the_shape_the_verifier_expects(agent):
    """The point of using the agent at all: an envelope with an event log, not
    a bare quote. A bare quote cannot satisfy an approved entry naming a
    compose hash, and the rejection would read as 'unapproved image'."""
    from fugal_subnet.tee.verify import unwrap_attestation

    _Agent.reply = {"attestation": _blob().hex()}
    quote, events = unwrap_attestation(attest(b"\x00" * 32, path=agent))
    assert int.from_bytes(quote[:2], "little") in (4, 5)
    assert events


def test_oversized_report_data_is_refused_before_the_call(agent):
    _Agent.reply = {"attestation": ""}
    with pytest.raises(ValueError, match="maximum 64"):
        attest(b"\x00" * 65, path=agent)


@pytest.mark.parametrize("reply,status,match", [
    ({"boottime_gpu_evidence": []}, 200, "no attestation"),
    ({"attestation": "not hex"}, 200, "malformed hex"),
    (b"<html>oops</html>", 200, "did not return JSON"),
    ({"attestation": "00"}, 500, "HTTP 500"),
])
def test_an_agent_that_answers_badly_raises_rather_than_returning_junk(
    agent, reply, status, match,
):
    _Agent.reply, _Agent.status = reply, status
    try:
        with pytest.raises(DstackError, match=match):
            attest(b"\x00" * 32, path=agent)
    finally:
        _Agent.status = 200


def test_a_missing_socket_is_reported_as_such(tmp_path):
    missing = str(tmp_path / "absent.sock")
    assert not available(missing)
    with pytest.raises(DstackError, match="cannot reach the dstack guest agent"):
        attest(b"\x00" * 32, path=missing)


def test_available_distinguishes_a_socket_from_a_regular_file(tmp_path, agent):
    plain = tmp_path / "not-a-socket"
    plain.write_text("")
    assert available(agent)
    assert not available(str(plain))
    assert not available(str(tmp_path / "nothing"))


def test_the_socket_path_is_overridable_by_env(monkeypatch):
    """Miners bind-mount the socket; a hard-coded path would make that a fork."""
    monkeypatch.setenv("DSTACK_SOCKET", "/tmp/elsewhere.sock")
    import importlib
    reloaded = importlib.reload(dstack_client)
    try:
        assert reloaded.DSTACK_SOCKET == "/tmp/elsewhere.sock"
    finally:
        monkeypatch.delenv("DSTACK_SOCKET")
        importlib.reload(dstack_client)
