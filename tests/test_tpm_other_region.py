"""A TD outside us-central1 (issues 129 and 179).

Google signs attestation keys with one "EK/AK CA Intermediate" per CA instance,
every one under the same name and under the one EK/AK root this repo pins. So
"is this a Google vTPM key?" has the same answer in every region: the
certificate chains, by signature, to the pinned root. Validators have applied
exactly that since proofs began to carry their intermediates.

The operator's own check did not. `provision.push` and `provision_td.py
--inspect` verified a TD with the vendored us-central1 intermediate only, so a
TD anywhere else was refused before it could be handed its key and head, and
never started. These tests pin the repair:

- the operator's check builds the chain a proof would carry (vendored, else the
  certificate's own issuer address) and holds it to the pinned root;
- that address comes from a certificate nobody has verified yet, so only
  Google's CA content hosts are ever asked, and never through a redirect;
- a miner whose one fetch at startup failed tries again at the next epoch;
- nothing a validator runs ever fetches.

The "other region" here is built from real certificates only: fixture A's
attestation key (issued in us-central1), Google's real us-east1 intermediate
standing in the vendored set, and the real root.
"""
import ast
import contextlib
import hashlib
import importlib.util
import json
import pathlib
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from fugal_subnet.tee import provision as provision_mod
from fugal_subnet.tee import tpm as tpm_mod
from fugal_subnet.tee import verify as verify_mod

REPO = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = pathlib.Path(__file__).parent / "fixtures"
ROOTS = pathlib.Path(tpm_mod.__file__).parent / "roots"
CENTRAL = (ROOTS / "google_ek_ak_intermediate_us-central1.der").read_bytes()
EAST = (FIXTURES / "google_ek_ak_intermediate_us-east1.der").read_bytes()
BLOB = (FIXTURES / "attestation_A.bin").read_bytes()
GOOGLE_URL = ("http://privateca-content-6a296e11-0000-2725-92a1-883d24f8f850"
              ".storage.googleapis.com/c5d5570c2c122e4f7db7/ca.crt")
# Where the intermediates themselves say the root is published.
ROOT_URL = ("http://privateca-content-62d71773-0000-21da-852e-f4f5e80d7778"
            ".storage.googleapis.com/032bf9d39db4fa06aade/ca.crt")


def _other_region(monkeypatch):
    """This checkout has vendored a different Google intermediate (us-east1)
    than the one that signed the TD's key (us-central1): same name, other key."""
    from cryptography import x509

    root = tpm_mod._load_pinned()[0]
    monkeypatch.setattr(tpm_mod, "_load_pinned", lambda *a, **k: (
        root, [x509.load_der_x509_certificate(EAST)]))


def _fetch_is(monkeypatch, fn):
    """What Google's CA host answers, in place of the network."""
    asked = []

    def fetch(url):
        asked.append(url)
        return fn(url)

    monkeypatch.setattr(tpm_mod, "fetch_google_ca_issuer", fetch)
    return asked


def _no_network(monkeypatch):
    """Refuse every connection and say which were tried. The list matters: the
    chain builder turns any failure into "no chain", so an exception alone
    would be swallowed."""
    opened = []

    def refuse(*a, **k):
        opened.append(a)
        raise AssertionError(f"opened a connection: {a!r}")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", refuse)
    return opened


@contextlib.contextmanager
def _td(answer):
    """A provisioning receiver that answers /provision/attest with `answer`."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            raw = json.dumps(answer).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    http = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", seen
    finally:
        http.shutdown()


ANSWER = {"attestation": BLOB.hex(), "pubkey": "11" * 32}
PAYLOAD = {"openrouter_api_key": "sk-or-v1-secret", "hotkey_ss58": "5Fk"}


# --- The facts the issues reported, kept true --------------------------------

def test_the_regional_intermediates_share_a_name_and_a_root_and_nothing_else():
    from cryptography import x509

    assert hashlib.sha256(EAST).hexdigest() == (
        "6f7825727896925c53ed5483487ba58b9977feffa3ee7f9c8a236093df18dae8")
    east, central = (x509.load_der_x509_certificate(d) for d in (EAST, CENTRAL))
    root, _ = tpm_mod._load_pinned()
    assert east.subject == central.subject and east.issuer == central.issuer == root.subject
    assert east.public_key().public_numbers() != central.public_key().public_numbers()
    for cert in (east, central):                      # each signed by the pinned root
        assert tpm_mod._issuer_of(cert, root, []) is root


def test_the_vendored_set_alone_refuses_a_td_from_another_region(monkeypatch):
    """What both issues hit: nothing wrong with the TD, only with what the
    checker had to hand."""
    _other_region(monkeypatch)
    with pytest.raises(ValueError, match="does not chain to Google's pinned EK/AK root"):
        verify_mod.unwrap_attestation(BLOB)
    assert verify_mod.unwrap_attestation(BLOB, extra_certs=[CENTRAL])[0]


# --- The operator's check ------------------------------------------------------

def test_the_operator_check_builds_the_chain_a_proof_would_carry(monkeypatch):
    _other_region(monkeypatch)
    asked = _fetch_is(monkeypatch, lambda url: CENTRAL)
    assert provision_mod.operator_tpm_chain(BLOB) == [CENTRAL]
    assert asked == [GOOGLE_URL]                       # the certificate's own issuer address
    quote, events = provision_mod.unwrap_td_attestation(BLOB)
    assert quote and events


def test_push_gets_past_the_tpm_half_for_a_td_in_another_region(monkeypatch):
    """The fixture's quote is not over this push's nonce, so the push still
    refuses, but at the nonce: the TD's hardware has been accepted."""
    _other_region(monkeypatch)
    _fetch_is(monkeypatch, lambda url: CENTRAL)
    with _td(ANSWER) as (address, seen):
        with pytest.raises(provision_mod.ProvisionError, match="not over our nonce"):
            provision_mod.push(address, PAYLOAD, approved_measurements={"a" * 64})
    assert seen == ["/provision/attest"]               # and nothing was pushed


def _provision_td():
    spec = importlib.util.spec_from_file_location(
        "provision_td_other_region", REPO / "scripts" / "provision_td.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_inspect_reports_what_a_td_in_another_region_attests(monkeypatch, capsys):
    """--inspect is how an operator learns the base measurement and instance id
    of a new TD. It used to stop at the TPM half and print neither."""
    _other_region(monkeypatch)
    _fetch_is(monkeypatch, lambda url: CENTRAL)
    with _td(ANSWER) as (address, _seen):
        assert _provision_td().inspect(address, timeout=5) == 0
    out = capsys.readouterr().out
    assert "base_measurement" in out and "instance_id" in out
    assert "us-central1-a" in out                      # the zone the key was issued in
    assert "1 intermediate" in out


def test_a_vendored_issuer_needs_no_fetch(monkeypatch):
    """us-central1 today: the operator's check is unchanged and offline."""
    opened = _no_network(monkeypatch)
    assert provision_mod.operator_tpm_chain(BLOB) == [CENTRAL]
    assert provision_mod.unwrap_td_attestation(BLOB)[0]
    assert opened == []


def test_an_operator_can_supply_the_intermediate_and_skip_the_fetch(monkeypatch):
    _other_region(monkeypatch)
    asked = _fetch_is(monkeypatch, lambda url: CENTRAL)
    assert provision_mod.unwrap_td_attestation(BLOB, [CENTRAL])[0]
    with _td(ANSWER) as (address, _seen):
        with pytest.raises(provision_mod.ProvisionError, match="not over our nonce"):
            provision_mod.push(address, PAYLOAD, approved_measurements={"a" * 64},
                               tpm_intermediates=[CENTRAL])
    assert asked == []


@pytest.mark.parametrize("answer", [
    lambda url: b"not a certificate",
    lambda url: EAST,                                  # genuine, but did not sign this key
    lambda url: (_ for _ in ()).throw(OSError("network unreachable")),
])
def test_a_fetch_that_does_not_produce_the_issuer_leaves_the_td_refused(monkeypatch, answer):
    _other_region(monkeypatch)
    _fetch_is(monkeypatch, answer)
    assert provision_mod.operator_tpm_chain(BLOB) == []
    with _td(ANSWER) as (address, seen):
        with pytest.raises(ValueError, match="TPM quote verification failed"):
            provision_mod.push(address, PAYLOAD, approved_measurements={"a" * 64})
    assert seen == ["/provision/attest"]


def test_a_ca_of_the_attackers_own_is_fetched_and_still_refused(monkeypatch):
    """Anyone can publish a CA on Google's CA service, at an address the host
    rule allows. The rule decides where a request may go; only the pinned root
    decides what is believed."""
    from fugal_subnet.attacks.run_tee_attacks import _rogue_chain

    _other_region(monkeypatch)
    _leaf, rogue = _rogue_chain(tpm_mod._load_pinned()[1][0].subject)
    _fetch_is(monkeypatch, lambda url: rogue)
    assert provision_mod.operator_tpm_chain(BLOB) == []
    with pytest.raises(ValueError, match="TPM quote"):
        provision_mod.unwrap_td_attestation(BLOB)
    with pytest.raises(ValueError, match="TPM quote"):
        provision_mod.unwrap_td_attestation(BLOB, [rogue])


def test_a_bare_quote_or_junk_reaches_the_ordinary_refusal(monkeypatch):
    opened = _no_network(monkeypatch)
    bare = (4).to_bytes(2, "little") + b"\x00" * 64
    assert provision_mod.operator_tpm_chain(bare) == []
    assert provision_mod.unwrap_td_attestation(bare) == (bare, None)
    assert provision_mod.operator_tpm_chain(b"\x00") == []
    with pytest.raises(ValueError, match="neither a TDX quote nor a dstack blob"):
        provision_mod.unwrap_td_attestation(b"\x00")
    assert opened == []


# --- The address is read from a certificate nobody has verified yet -----------

HOSTILE = [
    "file:///etc/passwd",
    "ftp://privateca-content-6a296e11.storage.googleapis.com/ca.crt",
    "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
    "http://localhost:8092/provision/logs",
    "http://10.0.0.5/ca.crt",
    "http://example.com/ca.crt",
    "http://evil.storage.googleapis.com/ca.crt",                 # any bucket is not the CA's
    "http://privateca-content-6a296e11.storage.googleapis.com.evil.example/ca.crt",
    "http://privateca-content-6a296e11.storage.googleapis.com@evil.example/ca.crt",
    "http://evil.example/privateca-content-6a296e11.storage.googleapis.com/ca.crt",
    "http://privateca-content-6a296e11.storage.googleapis.com:8080/ca.crt",
    "http://privateca-content-.storage.googleapis.com/ca.crt",
    "http://PRIVATECA-CONTENT-6a296e11.storage.googleapis.com\\@evil.example/ca.crt",
    "http://privateca-content-6a296e11.storage.googleapis.com\n.evil.example/ca.crt",
    "//privateca-content-6a296e11.storage.googleapis.com/ca.crt",
    "",
]


@pytest.mark.parametrize("url", HOSTILE)
def test_only_googles_ca_hosts_are_ever_asked(monkeypatch, url):
    opened = _no_network(monkeypatch)
    with pytest.raises(tpm_mod.TpmError, match="not fetched"):
        tpm_mod.fetch_google_ca_issuer(url)
    assert opened == []


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/ca.crt", "gopher://x/", ""])
def test_the_miners_fetch_is_http_only(monkeypatch, url):
    """Inside the TD the address comes from the platform's own vTPM, so the
    host is not restricted there (a change of Google's would stop miners); the
    scheme is, because urllib opens files and ftp as readily."""
    opened = _no_network(monkeypatch)
    with pytest.raises(tpm_mod.TpmError, match="not fetched"):
        tpm_mod._fetch_ca_issuer(url)
    assert opened == []


@pytest.mark.parametrize("url", [GOOGLE_URL, GOOGLE_URL.replace("http://", "https://"), ROOT_URL])
def test_googles_own_addresses_are_asked(monkeypatch, url):
    asked = []
    monkeypatch.setattr(tpm_mod, "_read_bounded", lambda u, open_: asked.append(u) or CENTRAL)
    assert tpm_mod.fetch_google_ca_issuer(url) == CENTRAL and asked == [url]


def _leaf_naming(url):
    """An attestation-key certificate, under a CA of the attacker's, whose
    issuer address is `url`."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import AuthorityInformationAccessOID

    ca, ak = (ec.generate_private_key(ec.SECP256R1()) for _ in range(2))
    name = x509.load_der_x509_certificate(CENTRAL).subject
    now = datetime.datetime.now(datetime.timezone.utc)
    return (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "AK")]))
            .issuer_name(name).public_key(ak.public_key()).serial_number(7)
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=30))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.AuthorityInformationAccess([x509.AccessDescription(
                AuthorityInformationAccessOID.CA_ISSUERS,
                x509.UniformResourceIdentifier(url))]), critical=False)
            .sign(ca, hashes.SHA256()).public_bytes(serialization.Encoding.DER))


@pytest.mark.parametrize("url", ["http://169.254.169.254/latest/meta-data/",
                                 "file:///etc/passwd", "http://127.0.0.1:8092/provision/logs"])
def test_a_td_cannot_point_the_operator_at_an_address_of_its_choosing(monkeypatch, url):
    """End to end through the chain builder the operator's check uses."""
    opened = _no_network(monkeypatch)
    _other_region(monkeypatch)
    assert tpm_mod.intermediates_for(_leaf_naming(url), fetch=tpm_mod.fetch_google_ca_issuer) == []
    assert opened == []


def test_a_redirect_is_not_followed():
    """A Google host that answered with a redirect would otherwise carry the
    request anywhere."""
    hits = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            if self.path == "/ca.crt":
                self.send_response(302)
                self.send_header("Location", "/elsewhere")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(CENTRAL)))
                self.end_headers()
                self.wfile.write(CENTRAL)

        def log_message(self, *a):
            pass

    http = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{http.server_port}"
        with pytest.raises(urllib.error.HTTPError):
            tpm_mod._read_bounded(f"{base}/ca.crt", tpm_mod._open_without_redirects)
        assert hits == ["/ca.crt"]
        assert tpm_mod._read_bounded(f"{base}/direct", tpm_mod._open_without_redirects) == CENTRAL
    finally:
        http.shutdown()


# --- The miner: one bad minute at boot is not the life of the process ----------

class _Runtime:
    def __init__(self, blob=BLOB):
        self.blob, self.asked = blob, 0

    def generate_attestation(self, report_data):
        self.asked += 1
        return self.blob


def test_a_failed_startup_fetch_is_tried_again_at_the_next_epoch(monkeypatch):
    from neurons.miner import _tpm_intermediates_at_startup, _tpm_intermediates_for_epoch

    _other_region(monkeypatch)
    answers = [OSError("network unreachable"), CENTRAL]

    def fetch(url):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(tpm_mod, "_fetch_ca_issuer", fetch)
    rt = _Runtime()
    have = _tpm_intermediates_at_startup(rt)
    assert have == []                                   # the fetch failed at boot
    have = _tpm_intermediates_for_epoch(have, rt, mock=False)
    assert have == [CENTRAL]                            # the next epoch has its chain
    assert _tpm_intermediates_for_epoch(have, rt, mock=False) is have and rt.asked == 2


def test_a_chain_already_built_or_a_mock_miner_asks_for_nothing():
    from neurons.miner import _tpm_intermediates_for_epoch

    rt = _Runtime()
    assert _tpm_intermediates_for_epoch([CENTRAL], rt, mock=False) == [CENTRAL]
    assert _tpm_intermediates_for_epoch([], rt, mock=True) == []
    assert rt.asked == 0


def test_the_epoch_loop_asks_before_each_epoch():
    import neurons.miner as miner_mod

    tree = ast.parse(pathlib.Path(miner_mod.__file__).read_text(encoding="utf-8"))
    main = ast.unparse(next(n for n in tree.body
                            if isinstance(n, ast.FunctionDef) and n.name == "main"))
    retry = "tpm_intermediates = _tpm_intermediates_for_epoch(tpm_intermediates, tee_runtime, mock)"
    assert retry in main
    assert main.index(retry) < main.index("_run_epoch(")


# --- Consensus: a validator's verdict never depends on the network -------------

@pytest.mark.parametrize("path", ["fugal_subnet/tee/verify.py", "neurons/validator.py"])
def test_nothing_a_validator_runs_can_fetch_a_certificate(path):
    """The proof carries its chain so that every validator reads the same
    bytes. A fetch on the validator's side would make the verdict depend on
    when it asked."""
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    used = ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            | {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
               for a in n.names})
    fetchers = {"intermediates_for", "fetch_google_ca_issuer", "_fetch_ca_issuer",
                "_aia_certificate", "_read_bounded", "operator_tpm_chain",
                "unwrap_td_attestation"}
    assert not used & fetchers, f"{path} uses {sorted(used & fetchers)}"


def test_the_validators_check_refuses_offline_and_accepts_what_the_proof_carries(monkeypatch):
    _other_region(monkeypatch)
    opened = _no_network(monkeypatch)
    with pytest.raises(ValueError, match="TPM quote"):
        verify_mod.unwrap_attestation(BLOB)
    assert verify_mod.unwrap_attestation(BLOB, extra_certs=[CENTRAL])[0]
    assert opened == []
