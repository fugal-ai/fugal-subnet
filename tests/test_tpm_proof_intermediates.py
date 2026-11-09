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


# The content hash of _proof() before intermediates were bound (release/v2
# 7aa1028). A proof without intermediates must keep it, or every proof from a
# validator or miner not yet upgraded would stop verifying.
HASH_WITHOUT_INTERMEDIATES = "742f6951bf7663561ea587dbad6c6df57fedc7e48b07be1ee397510d5e86a078"


def test_a_proof_without_intermediates_hashes_exactly_as_before():
    assert _proof().content_hash() == HASH_WITHOUT_INTERMEDIATES
    assert _proof([]).content_hash() == HASH_WITHOUT_INTERMEDIATES


def test_intermediates_are_bound_into_the_hash():
    """One proof_hash, one set of intermediates: a miner can no longer serve a
    valid chain to one validator and junk to another under the same hash (I1)."""
    bound = _proof([VENDORED]).content_hash()
    assert bound != HASH_WITHOUT_INTERMEDIATES
    assert _proof([b"\x30\x00"]).content_hash() != bound
    assert _proof([VENDORED, VENDORED]).content_hash() != bound


def test_swapping_the_intermediates_after_attestation_breaks_report_data():
    from fugal_subnet.tee.runtime import TEERuntime

    p = _proof([VENDORED])
    p.attestation_quote = TEERuntime(mock=True).generate_attestation(bytes.fromhex(p.content_hash()))
    ok = verify_mod.verify_proof(p, set(), p.questions_hash, p.nonce, {}, mock=True)
    assert "report_data" not in ok.reason
    p.tpm_intermediates = []
    r = verify_mod.verify_proof(p, set(), p.questions_hash, p.nonce, {}, mock=True)
    assert not r.valid and "report_data mismatch" in r.reason


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


# --- Same-named intermediates: the issuer is chosen by signature ------------
#
# Google names every EK/AK intermediate "CN=EK/AK CA Intermediate, OU=Google
# Cloud, ...", one per CA instance. A name lookup took the FIRST match, the
# vendored us-central1 one, so a leaf from any other instance failed even with
# its own intermediate in the proof (found 2026-11-08). The rotation tests above
# remove the vendored file, which hid it: the realistic case keeps it.

def _decoy_named_like(real_der):
    """A CA certificate with the real intermediate's exact subject and issuer
    names, under a key that signed nothing real."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    real = x509.load_der_x509_certificate(real_der)
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(real.subject).issuer_name(real.issuer)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    return cert.public_bytes(serialization.Encoding.DER)


def _vendored_is(monkeypatch, *ders):
    from cryptography import x509

    real = tpm_mod._load_pinned
    root = real()[0]
    monkeypatch.setattr(tpm_mod, "_load_pinned", lambda *a, **k: (
        root, [x509.load_der_x509_certificate(d) for d in ders]))


def test_a_same_named_vendored_intermediate_does_not_hide_the_right_one(monkeypatch):
    """A miner from another CA instance: validators have vendored a DIFFERENT
    intermediate with the same name, and the proof carries the right one."""
    blob = _blob()
    _vendored_is(monkeypatch, _decoy_named_like(VENDORED))
    with pytest.raises(ValueError, match="TPM quote"):
        verify_mod.unwrap_attestation(blob)                       # decoy alone: refused
    quote, _ = verify_mod.unwrap_attestation(blob, extra_certs=[VENDORED])
    assert quote


def test_the_issuer_choice_does_not_depend_on_order(monkeypatch):
    blob = _blob()
    decoy = _decoy_named_like(VENDORED)
    for vendored in ([decoy, VENDORED], [VENDORED, decoy]):
        _vendored_is(monkeypatch, *vendored)
        assert verify_mod.unwrap_attestation(blob)[0]


# --- The miner half: intermediates_for, at startup --------------------------

def _ak(name="A"):
    from fugal_subnet.scale import decode_dstack_attestation
    return decode_dstack_attestation(_blob(name))["tpm"]["ak_cert"]


def _aia_url(der):
    from cryptography import x509
    aia = x509.load_der_x509_certificate(der).extensions.get_extension_for_class(
        x509.AuthorityInformationAccess).value
    return aia[0].access_location.value


def _no_network(url):
    raise AssertionError(f"fetched {url} although a vendored intermediate signed the AK")


@pytest.mark.parametrize("name", ["A", "B"])
def test_a_vendored_issuer_is_carried_without_a_fetch(name):
    """The vendored intermediate is carried too, so the proof does not depend on
    which intermediates a validator happens to have vendored."""
    assert tpm_mod.intermediates_for(_ak(name), fetch=_no_network) == [VENDORED]


def test_after_a_rotation_the_issuer_is_fetched_from_the_aia_url(monkeypatch):
    _rotated(monkeypatch)
    asked = []

    def fetch(url):
        asked.append(url)
        return VENDORED

    assert tpm_mod.intermediates_for(_ak(), fetch=fetch) == [VENDORED]
    assert asked == [_aia_url(_ak())]
    # And the proof then verifies with what the miner built.
    quote, _ = verify_mod.unwrap_attestation(_blob(), extra_certs=[VENDORED])
    assert quote


def test_a_pem_body_is_accepted(monkeypatch):
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    _rotated(monkeypatch)
    pem = x509.load_der_x509_certificate(VENDORED).public_bytes(serialization.Encoding.PEM)
    assert tpm_mod.intermediates_for(_ak(), fetch=lambda url: pem) == [VENDORED]


@pytest.mark.parametrize("fetch", [
    lambda url: b"not a certificate",
    lambda url: (_ for _ in ()).throw(OSError("network unreachable")),
    lambda url: _decoy_named_like(VENDORED),          # parses, same name, wrong key
])
def test_nothing_that_does_not_chain_is_ever_carried(monkeypatch, fetch):
    _rotated(monkeypatch)
    assert tpm_mod.intermediates_for(_ak(), fetch=fetch) == []


def test_the_miner_builds_them_once_from_a_startup_attestation():
    from neurons.miner import _tpm_intermediates_at_startup

    class Runtime:
        def __init__(self, blob=None, error=None):
            self.blob, self.error, self.asked = blob, error, []

        def generate_attestation(self, report_data):
            self.asked.append(report_data)
            if self.error:
                raise self.error
            return self.blob

    rt = Runtime(_blob())
    assert _tpm_intermediates_at_startup(rt) == [VENDORED] and len(rt.asked) == 1
    from fugal_subnet.tee.runtime import TEERuntime
    bare = TEERuntime(mock=True).generate_attestation(b"\x00" * 32)
    assert _tpm_intermediates_at_startup(Runtime(bare)) == []
    assert _tpm_intermediates_at_startup(Runtime(error=RuntimeError("no dstack socket"))) == []


def test_the_miner_attaches_them_to_v2_proofs_before_hashing():
    """Wiring inside main() and _run_epoch, checked on the syntax tree: built
    once (live only), passed to every epoch, attached before content_hash, and
    only for v2 so a v1 proof stays byte-identical for an old validator."""
    import ast

    import neurons.miner as miner_mod

    tree = ast.parse(pathlib.Path(miner_mod.__file__).read_text(encoding="utf-8"))
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    main_src = ast.unparse(fns["main"])
    assert "tpm_intermediates = [] if mock else _tpm_intermediates_at_startup(tee_runtime)" in main_src
    assert "tpm_intermediates=tpm_intermediates" in main_src

    run = ast.unparse(fns["_run_epoch"])
    attach = "proof.tpm_intermediates = list(tpm_intermediates)"
    assert run.index(attach) < run.index("proof.content_hash()")
    (guard,) = [n for n in ast.walk(fns["_run_epoch"]) if isinstance(n, ast.If)
                and attach in ast.unparse(n.body)]
    assert ast.unparse(guard.test) == "protocol_version >= 2 and tpm_intermediates"


def test_the_fetch_is_bounded_in_size(monkeypatch):
    """The miner reads one certificate over plain HTTP: an over-long body is
    refused (and intermediates_for then carries nothing), never read whole."""
    import io
    import urllib.request

    from fugal_subnet.tee.proof import TPM_INTERMEDIATE_MAX_BYTES

    class Body(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout: Body(b"\x30" * (TPM_INTERMEDIATE_MAX_BYTES + 10)))
    with pytest.raises(tpm_mod.TpmError, match="over"):
        tpm_mod._fetch_ca_issuer("http://example.invalid/ca.crt")
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: Body(VENDORED))
    assert tpm_mod._fetch_ca_issuer("http://example.invalid/ca.crt") == VENDORED


def test_a_chain_that_would_not_verify_is_not_carried(monkeypatch):
    """Signatures alone are not the whole check: an issuer that signed the AK
    but has expired would make every proof fail, so the miner re-verifies the
    whole chain (validity included) before carrying it."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    now = datetime.datetime.now(datetime.timezone.utc)
    day = datetime.timedelta(days=1)

    def issue(name, key, issuer_name, issuer_key, ca, before, after):
        return (x509.CertificateBuilder()
                .subject_name(x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, name)]))
                .issuer_name(x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, issuer_name)]))
                .public_key(key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(before).not_valid_after(after)
                .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
                .sign(issuer_key, hashes.SHA256()))

    rk, ik, lk = (ec.generate_private_key(ec.SECP256R1()) for _ in range(3))
    root = issue("R", rk, "R", rk, True, now - 9 * day, now + 999 * day)
    expired = issue("I", ik, "R", rk, True, now - 9 * day, now - day)
    leaf = issue("AK", lk, "I", ik, False, now - 9 * day, now + 999 * day)
    der = lambda c: c.public_bytes(serialization.Encoding.DER)     # noqa: E731
    monkeypatch.setattr(tpm_mod, "_load_pinned", lambda *a, **k: (root, []))
    monkeypatch.setattr(tpm_mod, "_aia_certificate", lambda cert, fetch: expired)
    assert tpm_mod.intermediates_for(der(leaf)) == []


# --- Hostile candidates and slow fetches (independent review, 2026-11-08) ----

def _dup_extension_ca(subject):
    """A CA certificate named `subject` whose DER carries basicConstraints
    twice, so reading its extensions raises x509.DuplicateExtension (an
    Exception, not a ValueError). Builders refuse to make one, so an unknown
    extension's OID is renamed in the DER."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ObjectIdentifier

    now = datetime.datetime.now(datetime.timezone.utc)
    key = ec.generate_private_key(ec.SECP256R1())
    bc = x509.BasicConstraints(ca=True, path_length=None)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
            .public_key(key.public_key()).serial_number(1)
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=30))
            .add_extension(bc, critical=True)
            .add_extension(x509.UnrecognizedExtension(ObjectIdentifier("2.5.29.99"), bc.public_bytes()),
                           critical=False)
            .sign(key, hashes.SHA256()))
    der = cert.public_bytes(serialization.Encoding.DER)
    assert der.count(b"\x06\x03\x55\x1d\x63") == 1
    return der.replace(b"\x06\x03\x55\x1d\x63", b"\x06\x03\x55\x1d\x13")


def _issuer_name():
    from cryptography import x509
    return x509.load_der_x509_certificate(_ak()).issuer


def test_an_unreadable_candidate_is_passed_over_never_raised(monkeypatch):
    """Candidates come from miner bytes; a bad one is a non-issuer. It used to
    raise DuplicateExtension out of verify_proof."""
    bad = _dup_extension_ca(_issuer_name())
    _vendored_is(monkeypatch, bad, VENDORED)
    assert tpm_mod.verify_ak_chain(_ak())                      # skipped, the real one found
    _vendored_is(monkeypatch, bad)
    assert tpm_mod.verify_ak_chain(_ak()) is False             # alone: refused, not raised
    _rotated(monkeypatch)
    with pytest.raises(ValueError, match="TPM quote"):
        verify_mod.unwrap_attestation(_blob(), extra_certs=[bad])


def test_a_malformed_fetched_certificate_never_stops_the_miner(monkeypatch):
    from neurons.miner import _tpm_intermediates_at_startup

    bad = _dup_extension_ca(_issuer_name())
    _rotated(monkeypatch)
    assert tpm_mod.intermediates_for(_ak(), fetch=lambda url: bad) == []
    monkeypatch.setattr(tpm_mod, "_fetch_ca_issuer", lambda url: bad)

    class Runtime:
        def generate_attestation(self, report_data):
            return _blob()

    assert _tpm_intermediates_at_startup(Runtime()) == []


def test_an_expired_copy_of_a_ca_does_not_hide_its_valid_reissue(monkeypatch):
    """Same name, same key, different validity: the expired copy is passed
    over wherever it sits in the candidate order."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    now = datetime.datetime.now(datetime.timezone.utc)
    day = datetime.timedelta(days=1)
    name = lambda n: x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, n)])   # noqa: E731

    def issue(subject, key, issuer, issuer_key, ca, before, after, serial):
        return (x509.CertificateBuilder().subject_name(name(subject)).issuer_name(name(issuer))
                .public_key(key.public_key()).serial_number(serial)
                .not_valid_before(before).not_valid_after(after)
                .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
                .sign(issuer_key, hashes.SHA256()))

    der = lambda c: c.public_bytes(serialization.Encoding.DER)    # noqa: E731
    rk, ik, lk = (ec.generate_private_key(ec.SECP256R1()) for _ in range(3))
    root = issue("R", rk, "R", rk, True, now - 9 * day, now + 999 * day, 1)
    old = issue("I", ik, "R", rk, True, now - 9 * day, now - day, 2)
    new = issue("I", ik, "R", rk, True, now - day, now + 999 * day, 3)
    leaf = der(issue("AK", lk, "I", ik, False, now - day, now + 999 * day, 4))
    monkeypatch.setattr(tpm_mod, "_load_pinned", lambda *a, **k: (root, [old]))
    assert tpm_mod.verify_ak_chain(leaf, extra_certs=[der(new)])
    monkeypatch.setattr(tpm_mod, "_load_pinned", lambda *a, **k: (root, []))
    assert tpm_mod.verify_ak_chain(leaf, extra_certs=[der(old), der(new)])
    assert tpm_mod.verify_ak_chain(leaf, extra_certs=[der(new), der(old)])


def test_the_fetch_has_a_wall_clock_deadline(monkeypatch):
    """urlopen's timeout bounds each read, not the fetch: a responder that
    drips one byte at a time must still be cut off."""
    import socket
    import threading
    import time

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    stop = threading.Event()

    def drip():
        conn, _ = srv.accept()
        conn.recv(4096)
        conn.sendall(b"HTTP/1.0 200 OK\r\nContent-Length: 8000\r\n\r\n")
        try:
            while not stop.is_set():
                conn.sendall(b"x")
                time.sleep(0.2)
        except OSError:
            pass
        finally:
            conn.close()

    threading.Thread(target=drip, daemon=True).start()
    monkeypatch.setattr(tpm_mod, "_FETCH_DEADLINE_S", 1.0)
    t0 = time.monotonic()
    try:
        with pytest.raises(tpm_mod.TpmError, match="took over"):
            tpm_mod._fetch_ca_issuer(f"http://127.0.0.1:{srv.getsockname()[1]}/ca.crt")
    finally:
        stop.set()
        srv.close()
    assert time.monotonic() - t0 < 4.0
