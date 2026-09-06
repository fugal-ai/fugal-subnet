"""TPM quote verification against two real dstack attestations.

On GCP, dstack's attestation carries a TPM quote alongside the TDX one, and a
validator that checked only the TDX half would disagree with one that checked
both — a consensus fork. So this is a consensus rule, tested against artifacts
rather than against a specification.
"""
import datetime
import pathlib

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

from fugal_subnet.scale import decode_dstack_attestation
from fugal_subnet.tee.tpm import (
    TpmError,
    certificate_subject,
    parse_attest,
    verify_ak_chain,
    verify_quote_signature,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _quote(name):
    p = FIXTURES / f"attestation_{name}.bin"
    if not p.exists():
        pytest.skip(f"fixture {p.name} not present")
    return decode_dstack_attestation(p.read_bytes())["tpm"]


@pytest.mark.parametrize("name", ["A", "B"])
def test_a_real_quote_signature_verifies(name):
    q = _quote(name)
    assert verify_quote_signature(q["message"], q["signature"], q["ak_cert"])


@pytest.mark.parametrize("name", ["A", "B"])
def test_a_single_flipped_bit_fails(name):
    """A signature check that cannot fail is not a check."""
    q = _quote(name)
    tampered = bytearray(q["message"])
    tampered[100] ^= 0x01
    assert not verify_quote_signature(bytes(tampered), q["signature"], q["ak_cert"])


def test_a_quote_cannot_be_verified_under_another_machines_key():
    """Two different GCP instances, two different attestation keys. Presenting
    one machine's quote with another's certificate must fail — otherwise a
    miner could replay somebody else's platform attestation."""
    a, b = _quote("A"), _quote("B")
    assert a["ak_cert"] != b["ak_cert"]
    assert not verify_quote_signature(a["message"], a["signature"], b["ak_cert"])


@pytest.mark.parametrize("name", ["A", "B"])
def test_the_attest_structure_describes_its_own_buffer(name):
    """Trailing bytes mean the structure does not describe the buffer, so
    something else is present that we are not accounting for."""
    q = _quote(name)
    a = parse_attest(q["message"])
    assert a["pcr_selections"][0]["pcrs"] == [0, 2, 14]
    assert len(a["pcr_digest"]) == 64


def test_a_non_quote_attestation_is_refused():
    """The same key signs other TPM structures. One of those proves nothing
    about PCR state and must not be accepted as if it did."""
    q = _quote("A")
    other = bytearray(q["message"])
    other[4:6] = (0x8017).to_bytes(2, "big")     # ATTEST_CERTIFY, not a quote
    with pytest.raises(TpmError, match="not a quote"):
        parse_attest(bytes(other))


def test_malformed_input_is_refused_not_guessed():
    for bad in (b"", b"\x00" * 8, b"\xffTCG" + b"\x00" * 4):
        with pytest.raises(TpmError):
            parse_attest(bad)
    with pytest.raises(TpmError, match="does not parse"):
        verify_quote_signature(b"m", b"\x00" * 8, b"not a certificate")


def test_the_certificate_names_the_project_and_instance():
    """Recorded because it is a privacy fact miners must be told, not because
    it is a vulnerability: a proof is not anonymous."""
    subject = certificate_subject(_quote("A")["ak_cert"])
    assert "Google Compute Engine" in subject


# --- Chain validation -------------------------------------------------------
#
# The signature check alone proves only that SOME key signed the quote. Without
# the chain a miner signs a fabricated quote with a key they generated and a
# certificate they wrote, and every check passes. These two halves are only a
# security property together.

NOW = datetime.datetime.now(datetime.timezone.utc)


def _selfsigned(subject, key=None, ca=False, issuer_key=None, issuer_name=None):
    key = key or ec.generate_private_key(ec.SECP256R1())
    b = (x509.CertificateBuilder()
         .subject_name(subject).issuer_name(issuer_name or subject)
         .public_key(key.public_key()).serial_number(1)
         .not_valid_before(NOW - datetime.timedelta(days=1))
         .not_valid_after(NOW + datetime.timedelta(days=365))
         .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True))
    return b.sign(issuer_key or key, hashes.SHA256()).public_bytes(serialization.Encoding.DER)


@pytest.mark.parametrize("name", ["A", "B"])
def test_a_real_ak_certificate_chains_to_the_pinned_google_root(name):
    assert verify_ak_chain(_quote(name)["ak_cert"])


def test_a_self_signed_certificate_does_not_chain():
    """The whole attack: fabricate a quote, sign it with your own key, present
    your own certificate. The signature verifies; the chain must not."""
    genuine = x509.load_der_x509_certificate(_quote("A")["ak_cert"])
    assert not verify_ak_chain(_selfsigned(genuine.subject))


def test_a_genuine_ak_cannot_mint_further_attestation_keys():
    """A leaf is not a CA. If it could issue, one honest GCP box would let its
    owner attest for machines they do not have."""
    q = _quote("A")
    leaf = x509.load_der_x509_certificate(q["ak_cert"])
    minted = _selfsigned(x509.Name([]), issuer_name=leaf.subject,
                         issuer_key=ec.generate_private_key(ec.SECP256R1()))
    assert not verify_ak_chain(minted, extra_certs=[q["ak_cert"]])


def test_a_supplied_intermediate_must_itself_chain_to_the_root():
    """extra_certs exists so an unvendored region still verifies. It must not
    become a way to supply your own trust anchor."""
    rogue_key = ec.generate_private_key(ec.SECP256R1())
    rogue_name = x509.Name([
        x509.NameAttribute(x509.NameOID.COMMON_NAME, "EK/AK CA Intermediate")])
    rogue_ca = _selfsigned(rogue_name, key=rogue_key, ca=True)
    leaf = _selfsigned(x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "x")]),
                       issuer_key=rogue_key, issuer_name=rogue_name)
    assert not verify_ak_chain(leaf, extra_certs=[rogue_ca])


def test_an_expired_certificate_is_refused():
    """`at` is the one time-dependent input, and it is a parameter so that a
    verdict is reproducible rather than dependent on when it was asked."""
    q = _quote("A")
    assert not verify_ak_chain(q["ak_cert"], at=NOW - datetime.timedelta(days=3650))


def test_the_pinned_root_is_the_one_we_reviewed():
    """A swapped trust anchor accepts anything, so the file is hash-pinned the
    way graders.py and data/models.json are."""
    import hashlib
    import pathlib

    from fugal_subnet.tee import tpm
    raw = (pathlib.Path(tpm.__file__).parent / "roots" / "google_ek_ak_root.der").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == tpm._ROOT_SHA256
    root = x509.load_der_x509_certificate(raw)
    assert "EK/AK CA Root" in root.subject.rfc4514_string()
    assert root.issuer == root.subject


@pytest.mark.parametrize("name", ["A", "B"])
def test_the_composed_check_passes_on_real_quotes(name):
    from fugal_subnet.tee.tpm import verify_tpm_quote
    assert verify_tpm_quote(_quote(name))


def test_the_composed_check_fails_if_either_half_fails():
    """Each half is defeated separately; neither defeat may survive composition."""
    from fugal_subnet.tee.tpm import verify_tpm_quote

    q = dict(_quote("A"))
    good_cert = q["ak_cert"]

    tampered = bytearray(q["message"]); tampered[100] ^= 0x01
    assert not verify_tpm_quote({**q, "message": bytes(tampered)})   # chain ok, sig bad

    genuine = x509.load_der_x509_certificate(good_cert)
    key = ec.generate_private_key(ec.SECP256R1())
    forged = _selfsigned(genuine.subject, key=key)
    r, s = utils.decode_dss_signature(key.sign(q["message"], ec.ECDSA(hashes.SHA256())))
    sig = (b"\x00\x18\x00\x0b" + len(r.to_bytes(32, "big")).to_bytes(2, "big")
           + r.to_bytes(32, "big") + (32).to_bytes(2, "big") + s.to_bytes(32, "big"))
    assert verify_quote_signature(q["message"], sig, forged)         # sig ok on its own
    assert not verify_tpm_quote({**q, "signature": sig, "ak_cert": forged})


# --- Unwrapping, where the two attestation shapes meet ----------------------

def test_a_bare_tdx_quote_passes_through_unchanged():
    """The existing shape must keep working; this is not a migration."""
    from fugal_subnet.tee.runtime import TEERuntime
    from fugal_subnet.tee.verify import unwrap_attestation

    bare = TEERuntime(mock=True).generate_attestation(b"\x01" * 32)
    quote, events = unwrap_attestation(bare)
    assert quote == bare
    assert events is None


@pytest.mark.parametrize("name", ["A", "B"])
def test_a_dstack_envelope_yields_its_inner_quote_and_log(name):
    from fugal_subnet.tee.verify import unwrap_attestation

    p = FIXTURES / f"attestation_{name}.bin"
    if not p.exists():
        pytest.skip(f"fixture {p.name} not present")
    quote, events = unwrap_attestation(p.read_bytes())
    assert int.from_bytes(quote[:2], "little") in (4, 5)
    assert events and all("digest" in e for e in events)


def test_an_envelope_whose_tpm_half_fails_is_rejected_whole():
    """A validator checking only the TDX half would accept what another rejects.
    The TDX quote here is genuine; the proof must still be refused."""
    from fugal_subnet.tee.verify import unwrap_attestation

    p = FIXTURES / "attestation_A.bin"
    if not p.exists():
        pytest.skip("fixture not present")
    blob = bytearray(p.read_bytes())

    # Flip a bit inside the TPMS_ATTEST message, leaving the TDX quote intact.
    d = decode_dstack_attestation(bytes(blob))
    at = blob.index(d["tpm"]["message"])
    blob[at + 40] ^= 0x01
    with pytest.raises(ValueError, match="TPM quote verification failed"):
        unwrap_attestation(bytes(blob))


def test_bytes_that_are_neither_shape_are_refused():
    from fugal_subnet.tee.verify import unwrap_attestation

    with pytest.raises(ValueError, match="neither a TDX quote nor a dstack blob"):
        unwrap_attestation(b"\x00\x01" + b"\xff" * 64)


@pytest.mark.parametrize("name", ["A", "B"])
def test_the_envelopes_event_log_reproduces_its_own_rtmr3(name):
    """The end-to-end property the approved list depends on: the log is untrusted
    until it replays to the register the CPU signed, and only then may a single
    field of it (the compose hash) be compared against anything."""
    from fugal_subnet.tee.attestation import parse_quote, replay_event_log
    from fugal_subnet.tee.verify import unwrap_attestation

    p = FIXTURES / f"attestation_{name}.bin"
    if not p.exists():
        pytest.skip(f"fixture {p.name} not present")
    quote, events = unwrap_attestation(p.read_bytes())
    replayed, seen = replay_event_log(events, imr=3)
    assert replayed == parse_quote(quote).rtmr3
    assert len(seen["compose-hash"]) == 32
    assert seen["instance-id"]


def test_the_two_boxes_differ_where_they_should_and_agree_where_they_must():
    """Recorded as a fact about the fixtures, because it is easy to misread.

    Different instance-id and different RTMR3 are EXPECTED — instance-id is
    extended on every deploy, which is why an approved entry can never hold an
    RTMR3 value. The compose hashes also differ, which means these are two
    different applications rather than one application deployed twice; anyone
    building an approved entry from these fixtures needs to know which.
    """
    from fugal_subnet.tee.attestation import replay_event_log
    from fugal_subnet.tee.verify import unwrap_attestation

    seen = {}
    for name in ("A", "B"):
        p = FIXTURES / f"attestation_{name}.bin"
        if not p.exists():
            pytest.skip(f"fixture {p.name} not present")
        _, events = unwrap_attestation(p.read_bytes())
        seen[name] = replay_event_log(events, imr=3)[1]

    assert seen["A"]["instance-id"] != seen["B"]["instance-id"]
    assert seen["A"]["compose-hash"] != seen["B"]["compose-hash"]


# --- The whole app-identity chain, on real data -----------------------------

def test_the_compose_file_bytes_reach_the_signed_register():
    """Every link, end to end, against hardware output rather than argument:

        sha256(RAW FILE BYTES)  ==  the compose-hash event in the log
        the log                 replays to RTMR3
        RTMR3                   is inside the Intel-signed quote

    This is what makes `compute_app_identity.py --compose` trustworthy, and it
    is the check that was missing when the compose hash was computed from
    normalised JSON — that version produced a hash dstack never extends, so it
    would have rejected every honest miner while looking correct.
    """
    import hashlib

    from fugal_subnet.tee.attestation import parse_quote, replay_event_log
    from fugal_subnet.tee.verify import unwrap_attestation

    compose = FIXTURES / "app-compose_A.json"
    blob = FIXTURES / "attestation_A.bin"
    if not (compose.exists() and blob.exists()):
        pytest.skip("fixtures not present")

    quote, events = unwrap_attestation(blob.read_bytes())
    replayed, seen = replay_event_log(events, imr=3)

    assert hashlib.sha256(compose.read_bytes()).hexdigest() == seen["compose-hash"].hex()
    assert replayed == parse_quote(quote).rtmr3


def test_the_compose_hash_is_raw_bytes_and_not_a_reserialisation():
    """dstack's fields are in INSERTION order, not sorted, and the compose YAML
    is one JSON string with escaped newlines whose trailing newline counts. Any
    re-serialisation changes the hash, so the only correct answer is the bytes."""
    import hashlib
    import json

    from scripts.compute_app_identity import compose_hash

    compose = FIXTURES / "app-compose_A.json"
    if not compose.exists():
        pytest.skip("fixture not present")

    raw = compose.read_bytes()
    digest, _ = compose_hash(str(compose))
    assert digest == hashlib.sha256(raw).hexdigest()

    parsed = json.loads(raw)
    assert list(parsed) != sorted(parsed), "fields are not sorted; do not sort them"

    # The normalisation that was wrong, and the compact form: both change the
    # hash, which is what would have rejected every honest miner.
    for variant in (json.dumps(parsed, sort_keys=True, separators=(",", ":")),
                    json.dumps(parsed, separators=(",", ":")),
                    json.dumps(parsed, sort_keys=True, indent=2)):
        assert hashlib.sha256(variant.encode()).hexdigest() != digest

    # dstack's own bytes happen to be exactly json.dumps(obj, indent=2) with
    # fields left in insertion order and no trailing newline. Asserted so that a
    # change is noticed, NOT so anything depends on it: the moment we hash a
    # re-serialisation instead of the bytes we are guessing at their writer
    # again, and we have already paid for that once.
    assert json.dumps(parsed, indent=2).encode() == raw


def test_report_data_is_right_zero_padded_not_hashed():
    """Measured on the guest: a 5-byte report_data came back as those 5 bytes
    followed by 59 zeros. The guest pads; it does not hash and does not reject a
    short value. `verify_proof` must expect exactly that padding for a 32-byte
    content hash, or every live proof fails a binding check that looks like
    tampering rather than a convention mismatch."""
    from fugal_subnet.tee.attestation import parse_quote
    from fugal_subnet.tee.verify import unwrap_attestation

    blob = FIXTURES / "attestation_A.bin"
    if not blob.exists():
        pytest.skip("fixture not present")
    quote, _ = unwrap_attestation(blob.read_bytes())
    observed = parse_quote(quote).report_data
    assert observed == b"fugal".hex() + "00" * 59

    # What the miner will actually send: a 32-byte content hash. This is the
    # expression verify_proof uses, checked against the guest's rule.
    content_hash = "ab" * 32
    expected = bytes.fromhex(content_hash).ljust(64, b"\x00")[:64]
    assert expected.hex() == content_hash + "00" * 32
