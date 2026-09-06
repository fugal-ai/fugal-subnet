"""TPM quote verification, for dstack attestations on GCP.

On GCP, dstack's attestation is a distinct variant carrying a TPM quote
alongside the TDX one. Checking only the TDX half is not an option, and
validators split on whether to check the TPM half would disagree on validity —
a consensus fork with no bug behind it. So this is a consensus rule.

WHAT THE TPM HALF ADDS, AND WHAT IT COSTS. It attests that a genuine Google
Confidential VM produced the quote, signed by an attestation key whose
certificate chains to Google's EK/AK CA. That is a NEW TRUST ROOT: verifying it
means trusting Google's CA in addition to Intel's. Recorded in INVARIANTS.md
next to the Intel PCS dependency rather than buried here, because it is a change
to the threat model and not an implementation detail.

The certificate also names the miner's GCP project and instance. That is not a
vulnerability, but it does mean a proof is not anonymous, and miners are told so
in MINER_GUIDE.md.
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import pathlib
import struct

logger = logging.getLogger(__name__)

TPM_GENERATED = b"\xffTCG"
TPM_ST_ATTEST_QUOTE = 0x8018
TPM_ALG_ECDSA = 0x0018
TPM_ALG_SHA256 = 0x000B


class TpmError(ValueError):
    """A TPM quote that cannot be trusted. Always a rejection."""


def parse_attest(message: bytes) -> dict:
    """Parse TPMS_ATTEST. Strict on every length, because a miner supplies it."""
    try:
        pos = 0
        magic, attest_type = struct.unpack_from(">IH", message, pos); pos += 6
        if magic.to_bytes(4, "big") != TPM_GENERATED:
            raise TpmError(f"not a TPM-generated structure: magic {magic:#x}")
        if attest_type != TPM_ST_ATTEST_QUOTE:
            raise TpmError(
                f"attestation type {attest_type:#x} is not a quote "
                f"({TPM_ST_ATTEST_QUOTE:#x}) — a different structure signed by "
                "the same key proves nothing about PCR state"
            )

        def tpm2b():
            nonlocal pos
            (n,) = struct.unpack_from(">H", message, pos); pos += 2
            if pos + n > len(message):
                raise TpmError("TPM2B field runs past the end of the structure")
            out = message[pos:pos + n]; pos += n
            return out

        signer, extra = tpm2b(), tpm2b()
        pos += 17                                     # clockInfo
        (firmware,) = struct.unpack_from(">Q", message, pos); pos += 8
        (count,) = struct.unpack_from(">I", message, pos); pos += 4
        selections = []
        for _ in range(count):
            (alg,) = struct.unpack_from(">H", message, pos); pos += 2
            size = message[pos]; pos += 1
            mask = message[pos:pos + size]; pos += size
            pcrs = [i * 8 + b for i, byte in enumerate(mask)
                    for b in range(8) if byte & (1 << b)]
            selections.append({"algorithm": alg, "pcrs": pcrs})
        digest = tpm2b()
    except TpmError:
        raise
    except (struct.error, IndexError) as e:
        raise TpmError(f"malformed TPMS_ATTEST: {e}") from e

    if pos != len(message):
        raise TpmError(
            f"TPMS_ATTEST has {len(message) - pos} trailing bytes; the structure "
            "does not describe the buffer and should not be trusted"
        )
    return {
        "qualified_signer": signer.hex(),
        "extra_data": extra.hex(),
        "firmware_version": firmware,
        "pcr_selections": selections,
        "pcr_digest": digest.hex(),
    }


def _signature_rs(signature: bytes) -> tuple[int, int]:
    """Pull (r, s) out of a TPMT_SIGNATURE, refusing anything but ECDSA/SHA256."""
    if len(signature) < 8:
        raise TpmError("signature is too short to be a TPMT_SIGNATURE")
    sig_alg, hash_alg = struct.unpack_from(">HH", signature, 0)
    if sig_alg != TPM_ALG_ECDSA:
        raise TpmError(f"signature algorithm {sig_alg:#x} is not ECDSA")
    if hash_alg != TPM_ALG_SHA256:
        raise TpmError(f"hash algorithm {hash_alg:#x} is not SHA256")
    pos = 4
    (rn,) = struct.unpack_from(">H", signature, pos); pos += 2
    r = int.from_bytes(signature[pos:pos + rn], "big"); pos += rn
    (sn,) = struct.unpack_from(">H", signature, pos); pos += 2
    s = int.from_bytes(signature[pos:pos + sn], "big"); pos += sn
    if pos > len(signature):
        raise TpmError("signature is truncated inside r or s")
    # Real dstack blobs carry five trailing bytes after the TPMT_SIGNATURE
    # (observed: 0000010000). Their meaning is not identified here and they are
    # NOT guessed at. Ignoring them is safe: r and s are read from fixed offsets
    # at the front, so nothing appended can change what gets verified — but the
    # field is not fully understood and this comment exists so nobody later
    # mistakes silence for knowledge.
    if pos != len(signature):
        logger.debug("TPMT_SIGNATURE has %d unidentified trailing bytes",
                     len(signature) - pos)
    return r, s


def verify_quote_signature(message: bytes, signature: bytes, ak_cert_der: bytes) -> bool:
    """Verify the quote signature under the attestation key's own certificate.

    This proves the message was signed by the key in that certificate. It does
    NOT prove the certificate is Google's — that is `verify_ak_chain`, and both
    are required. A signature check alone would accept a quote signed by any key
    accompanied by any self-made certificate.
    """
    from cryptography import x509
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    try:
        cert = x509.load_der_x509_certificate(ak_cert_der)
    except Exception as e:  # noqa: BLE001 - miner-supplied bytes
        raise TpmError(f"attestation key certificate does not parse: {e}") from e

    key = cert.public_key()
    if not isinstance(key, ec.EllipticCurvePublicKey):
        raise TpmError(f"attestation key is {type(key).__name__}, expected EC")

    r, s = _signature_rs(signature)
    try:
        key.verify(
            utils.encode_dss_signature(r, s), message, ec.ECDSA(hashes.SHA256()),
        )
    except InvalidSignature:
        return False
    return True


def certificate_subject(ak_cert_der: bytes) -> str:
    """Human-readable subject, for logs and for telling a miner what they leak."""
    from cryptography import x509

    return x509.load_der_x509_certificate(ak_cert_der).subject.rfc4514_string()


# The Google EK/AK CA root, vendored rather than fetched. Its fingerprint is
# pinned so that swapping the file is a loud failure, the same discipline
# graders.py and data/models.json get: this is the trust anchor for every TPM
# quote the subnet accepts, and a silently substituted anchor accepts anything.
_ROOT_SHA256 = "594759594b9b61524f6c8ef668f7177f7b9066bc0674f2f49fe9e052be78beb2"
_ROOTS_DIR = pathlib.Path(__file__).parent / "roots"


def _load_pinned(name_hint: str = "") -> tuple[object, list]:
    """Load the pinned root and the vendored intermediates. No network, ever.

    Fetching the chain per proof is what the AIA extension invites and it is
    refused on three grounds: the URL is plain HTTP, it is a live availability
    dependency for every validator, and — decisively — two validators fetching
    at different moments can get different answers, which is a consensus fork
    with no bug behind it. Pinned bytes give every validator the same verdict.
    """
    from cryptography import x509

    root_path = _ROOTS_DIR / "google_ek_ak_root.der"
    raw = root_path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != _ROOT_SHA256:
        raise TpmError(
            f"pinned Google EK/AK root has hash {got}, expected {_ROOT_SHA256}. "
            "Refusing to verify against an unrecognised trust anchor."
        )
    root = x509.load_der_x509_certificate(raw)

    intermediates = []
    for p in sorted(_ROOTS_DIR.glob("*intermediate*.der")):
        intermediates.append(x509.load_der_x509_certificate(p.read_bytes()))
    return root, intermediates


def verify_ak_chain(ak_cert_der: bytes, extra_certs: "list[bytes] | None" = None,
                    at=None) -> bool:
    """Chain the attestation key certificate to the pinned Google root.

    `extra_certs` lets a proof carry its own intermediates. That is safe for the
    same reason TLS is: a supplied certificate is worthless unless it chains to
    the anchor we pinned. It exists because Google's intermediates rotate and
    live at per-CA-instance URLs that cannot be enumerated ahead of time, so a
    miner in a region whose intermediate we have not vendored would otherwise be
    rejected for a reason that has nothing to do with their honesty.

    `at` is the only time-dependent input. It defaults to now; pass an explicit
    time to make a verdict reproducible. Two validators evaluating a certificate
    across its expiry boundary would disagree — the leaf observed in practice is
    valid for 30 years, so the window is wide, but the hazard is real and is why
    the parameter exists rather than a hidden call to the clock.
    """
    from cryptography import x509
    from cryptography.exceptions import InvalidSignature

    if at is None:
        at = datetime.datetime.now(datetime.timezone.utc)

    try:
        leaf = x509.load_der_x509_certificate(ak_cert_der)
    except Exception as e:  # noqa: BLE001 - miner-supplied bytes
        raise TpmError(f"attestation key certificate does not parse: {e}") from e

    root, pool = _load_pinned()
    for raw in extra_certs or []:
        try:
            pool.append(x509.load_der_x509_certificate(raw))
        except Exception as e:  # noqa: BLE001 - miner-supplied bytes
            raise TpmError(f"supplied intermediate does not parse: {e}") from e

    # Walk leaf -> ... -> root. Depth is bounded so a cycle among supplied
    # certificates cannot spin here; a miner controls part of this pool.
    cert = leaf
    for depth in range(8):
        if not (cert.not_valid_before_utc <= at <= cert.not_valid_after_utc):
            logger.warning("certificate %s is outside its validity window at %s",
                           cert.subject.rfc4514_string(), at)
            return False

        issuer = root if cert.issuer == root.subject else next(
            (c for c in pool if c.subject == cert.issuer), None)
        if issuer is None:
            logger.warning("no issuer for %s (issuer %s) among pinned certificates",
                           cert.subject.rfc4514_string(), cert.issuer.rfc4514_string())
            return False

        # An issuer must actually be allowed to issue. Without this a leaf
        # certificate could be used to sign another certificate, and any miner
        # holding a genuine AK could mint attestation keys for machines they do
        # not own.
        try:
            bc = issuer.extensions.get_extension_for_class(x509.BasicConstraints).value
        except x509.ExtensionNotFound:
            logger.warning("issuer %s has no basicConstraints",
                           issuer.subject.rfc4514_string())
            return False
        if not bc.ca:
            logger.warning("issuer %s is not a CA", issuer.subject.rfc4514_string())
            return False

        try:
            issuer.public_key().verify(
                cert.signature, cert.tbs_certificate_bytes,
                *_verifier_args(cert, issuer),
            )
        except InvalidSignature:
            logger.warning("signature on %s does not verify under %s",
                           cert.subject.rfc4514_string(), issuer.subject.rfc4514_string())
            return False

        if issuer is root:
            return True
        cert = issuer

    logger.warning("certificate chain deeper than 8; refusing")
    return False


def _verifier_args(cert, issuer):
    """Padding/algorithm arguments for verifying `cert` under `issuer`'s key."""
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

    key = issuer.public_key()
    algo = cert.signature_hash_algorithm
    if isinstance(key, rsa.RSAPublicKey):
        return (padding.PKCS1v15(), algo)
    if isinstance(key, ec.EllipticCurvePublicKey):
        return (ec.ECDSA(algo),)
    raise TpmError(f"unsupported issuer key type {type(key).__name__}")


def verify_tpm_quote(quote: dict, extra_certs: "list[bytes] | None" = None,
                     at=None) -> bool:
    """Both halves, in one call, because either alone proves nothing.

    The signature check without the chain accepts a fabricated quote signed by
    the miner's own key. The chain check without the signature accepts a genuine
    certificate stapled to somebody else's quote. This function exists so that
    no caller can accidentally do one and believe it did both.

    `quote` is the dict from `scale.decode_tpm_quote`.
    """
    if not verify_ak_chain(quote["ak_cert"], extra_certs=extra_certs, at=at):
        return False
    return verify_quote_signature(quote["message"], quote["signature"], quote["ak_cert"])
