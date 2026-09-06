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

from fugal_subnet.scale import decode_dstack_attestation, decode_tpm_quote
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
    blob = p.read_bytes()
    d = decode_dstack_attestation(blob)
    return decode_tpm_quote(blob, blob.index(d["tpm_quote"]) - 2)


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
