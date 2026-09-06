"""Per-miner data delivered INTO the TD after boot, over an attested channel.

Three facts decide this design.

**Per-miner values cannot be measured.** The compose hash IS the app identity in
the approved list, so it must be identical for every honest miner. A head that
changes every epoch, or a key that differs per operator, can never live in a
measured file.

**The unmeasured file that does exist is not confidential.** `.user-config`
rides the shared disk, which is a plain FAT32 image uploaded to cloud storage.
Anything in it is readable by the cloud provider. Fine for a head, fatal for a key.

**A TD can prove what it is without holding a secret first.** It can produce a
fresh Intel-signed quote over any nonce. So the miner does not have to hand the
TD a credential to bootstrap trust — the TD authenticates itself.

Those three together invert the obvious design. Rather than the TD fetching
secrets from somewhere (which needs an address and a credential, both per-miner,
both therefore in plaintext), THE MINER PUSHES. The miner's own process created
this TD and knows its address; it asks the TD to prove itself, checks the proof
against the approved measurement and app identity, and only then sends. Nothing
per-miner is written anywhere the cloud can read, and `.user-config` drops out of
the critical path entirely.

WHAT MAY CROSS, and this list is the security boundary:

    head            per-miner, changes every epoch, already bound by the
                    on-chain weights_hash committed before the nonce. Untrusted
                    bytes that are *bound* are safe; that is the existing
                    pattern for the head and this changes nothing about it.
    hotkey ss58     public — it is on chain. The TD needs it to bind the proof
                    to a miner, which is what kills proof relay
                    cryptographically rather than statistically. A miner who
                    pushes someone else's hotkey produces a proof bound to that
                    hotkey, which their own uid cannot use.
    api key         a secret, and only the miner's own money.
    hotkey keyfile  a secret, and only the miner's own identity. The TD has to
                    sign two extrinsics itself — serve_axon and the head
                    commitment — so the signing key must be inside it, and
                    this channel is the only path that never touches storage
                    the cloud provider can read (the compose is public, the
                    shared disk is plaintext FAT32 in GCS). Held in tmpfs,
                    never on the data volume. A miner who pushes it to a TD
                    they did not verify has handed over their own identity and
                    nobody else's — the same blast radius as the API key.
    coldkeypub      public. The SDK's serve_axon reads the coldkey ADDRESS
                    off the wallet, so the hotkey alone does not serve.

WHAT MUST NEVER CROSS: the model upstream, the pool, the grader, or anything
else `runtime_identity()` covers. Those are inside the measurement precisely so
a miner cannot choose them. A channel that can carry FUGAL_OPENROUTER_BASE hands
a miner the upstream-substitution exploit that `scripts/stub_upstream.py` exists
to demonstrate — perfect accuracy at near-zero attested cost, with every hash
binding and the approved measurement still passing.

ORDERING: THE AXON DOES NOT SERVE UNTIL PROVISIONING COMPLETES.

A miner today serves immediately and commits afterwards, deliberately — committing
first gets the serve extrinsic rate-limited and leaves the miner unreachable.
Provisioning inserts a step before readiness, and the failure mode needs choosing
rather than defaulting: an axon that serves before the TD has a head and a key
answers a validator with no proof, and reads as a dead miner rather than a
starting one.

The choice here is to HOLD THE AXON. An unprovisioned miner is not partially
ready, it is not ready, and pretending otherwise spends a validator's query
budget to learn nothing. The alternative — serving and answering with an explicit
not-ready — needs a protocol field and gives a validator something it must then
decide what to do with, for a state that lasts one round trip.

Holding fails loudly in the miner's own log, which is the property that matters:
"waiting to be provisioned" is a sentence an operator can act on. Silent
emptiness is the failure shape this project keeps finding, and it is the one
thing this must not be.

That is why `ALLOWED_FIELDS` is a closed allow-list and an unknown field is a
hard rejection rather than an ignored extra. Silently dropping what it does not
recognise is how a channel grows a second purpose that nobody reviewed.
"""
from __future__ import annotations

import logging
import secrets as _secrets

logger = logging.getLogger(__name__)

# The complete set of values that may be pushed into a TD. Adding to this is a
# consensus review, not a config change: tests/test_provision_schema.py reads
# this literal out of the AST and fails when it changes, so a new field cannot
# arrive in a diff nobody reads. Nothing here may be an input the grader, the
# pool, the slice or the cost model depends on.
ALLOWED_FIELDS = frozenset({
    "head_b64",
    "hotkey_ss58",
    "openrouter_api_key",
    "hotkey_keyfile_b64",
    "coldkeypub_b64",
})

# Nonce the miner chooses per provisioning attempt. The TD signs it into
# report_data, which is what makes a captured attestation useless later: a
# replayed quote carries the wrong nonce and fails the miner's own check.
NONCE_BYTES = 32


class ProvisionError(RuntimeError):
    """Provisioning failed. Never raised with a secret in the message."""


def new_nonce() -> str:
    """A fresh challenge for one provisioning attempt."""
    return _secrets.token_hex(NONCE_BYTES)


def validate_payload(payload: dict) -> dict[str, str]:
    """Accept exactly the allowed fields, reject anything else.

    Fails closed at the boundary rather than after a value has been used, and
    rejects unknown keys rather than ignoring them — an ignored extra is a
    channel quietly acquiring a second purpose.

    Values are returned as-is. This function decides WHAT may cross, never
    whether a particular value is sensible; that belongs to whoever consumes it.
    """
    if not isinstance(payload, dict):
        raise ProvisionError(f"payload must be an object, got {type(payload).__name__}")

    unknown = set(payload) - ALLOWED_FIELDS
    if unknown:
        # Names only. A rejected field's VALUE never reaches a log, because the
        # most likely unknown field is a secret someone added by mistake.
        raise ProvisionError(
            f"refusing unknown provisioning field(s): {sorted(unknown)}. "
            f"Allowed: {sorted(ALLOWED_FIELDS)}. Adding one is a consensus "
            f"review — see fugal_subnet/tee/provision.py."
        )
    for name, value in payload.items():
        if not isinstance(value, str):
            raise ProvisionError(
                f"field {name!r} must be a string, got {type(value).__name__}"
            )
    return dict(payload)


# --- transport -------------------------------------------------------------
#
# Small, and deliberately no cleverer than it must be. The security lives in
# what the miner CHECKS before it pushes: the TD proves what it is with an
# Intel-signed quote over a nonce the miner chose. But a quote AUTHENTICATES; it
# does not make the wire confidential, and an earlier version of this module
# POSTed the API key and the hotkey keyfile over plain HTTP once the quote
# checked out — verified who it was talking to, then shouted the secrets down
# the street. Found 2026-09-06 while designing the log path.
#
# So the quote now also binds a key to encrypt TO. The TD generates an
# ephemeral X25519 key when the receiver starts and puts sha256(pubkey) in the
# second half of report_data, next to the miner's nonce. The Intel signature
# therefore covers "this TD, running this measured image, holds this key", and
# the miner encrypts the payload to it: X25519 with a fresh operator key, HKDF
# salted with the nonce, ChaCha20-Poly1305 with the nonce as associated data.
# The TD refuses plaintext, refuses a nonce it never attested, and consumes
# each nonce once. Nothing about the checks before the push changed.
#
# The same shared secret authorises an operator-only, encrypted log pull, so a
# miner's own logs stop being a black box without setting public_logs.

PROVISION_PORT = 8092

_ATTEST_PATH = "/provision/attest"
_PUSH_PATH = "/provision"
_STATUS_PATH = "/provision/status"
_LOGS_PATH = "/provision/logs"

_HKDF_INFO_KEY = b"fugal-provision-v1/key"
_HKDF_INFO_LOG_TOKEN = b"fugal-provision-v1/log-token"
_AEAD_NONCE_BYTES = 12
# How many attested-but-unused nonces the TD remembers. Bounded so a flood of
# /attest requests cannot grow memory; the operator needs exactly one.
_MAX_PENDING_NONCES = 64
# The in-memory log ring the operator may pull. Lines, not bytes, so a burst of
# long tracebacks cannot silently evict the line that explains them.
LOG_RING_LINES = 5000


def _require_crypto():
    """The `cryptography` package ships in the `tee` extra (the miner image and
    the operator both install it). Fail with the install command, not a stack
    trace, when it is absent."""
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import x25519
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    except ImportError as e:  # pragma: no cover - environment, not logic
        raise ProvisionError(
            "the provisioning channel needs the `cryptography` package: "
            "install with `uv sync --extra tee`"
        ) from e
    return hashes, x25519, ChaCha20Poly1305, HKDF


def report_data_for(nonce_hex: str, td_pubkey: bytes) -> bytes:
    """The 64 bytes the TD attests over: the operator's nonce, then the hash of
    the TD's ephemeral public key. Both halves fixed width, so neither can be
    confused for the other."""
    import hashlib

    nonce = bytes.fromhex(nonce_hex)
    if len(nonce) != NONCE_BYTES:
        raise ProvisionError(f"nonce must be {NONCE_BYTES} bytes")
    if len(td_pubkey) != 32:
        raise ProvisionError("X25519 public key must be 32 bytes")
    return nonce + hashlib.sha256(td_pubkey).digest()


def _derive(shared: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    hashes, _x, _c, HKDF = _require_crypto()
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(shared)


def seal(payload: dict, td_pubkey: bytes, nonce_hex: str) -> tuple[dict, bytes]:
    """Encrypt `payload` to the TD's attested key. Returns (envelope, session_key).

    The envelope carries the operator's ephemeral public key, the AEAD nonce,
    the ciphertext and the provisioning nonce it belongs to — nothing secret.
    `session_key` is what the operator keeps to pull logs later; it never
    crosses the wire.
    """
    import json
    import os as _os

    _h, x25519, ChaCha20Poly1305, _k = _require_crypto()
    validate_payload(payload)
    nonce = bytes.fromhex(nonce_hex)
    eph = x25519.X25519PrivateKey.generate()
    shared = eph.exchange(x25519.X25519PublicKey.from_public_bytes(td_pubkey))
    key = _derive(shared, nonce, _HKDF_INFO_KEY)
    iv = _os.urandom(_AEAD_NONCE_BYTES)
    ct = ChaCha20Poly1305(key).encrypt(iv, json.dumps(payload).encode(), nonce)
    epk = eph.public_key().public_bytes_raw()
    return {"nonce": nonce_hex, "epk": epk.hex(), "iv": iv.hex(), "ct": ct.hex()}, key


def unseal(td_private, envelope: dict) -> tuple[dict, bytes]:
    """TD side of `seal`. Returns (payload, session_key). Raises ProvisionError
    on anything malformed or unauthentic; the message never carries a value."""
    import json

    _h, x25519, ChaCha20Poly1305, _k = _require_crypto()
    if not isinstance(envelope, dict) or set(envelope) != {"nonce", "epk", "iv", "ct"}:
        raise ProvisionError(
            "push must be a sealed envelope {nonce, epk, iv, ct}; plaintext is refused"
        )
    try:
        nonce = bytes.fromhex(envelope["nonce"])
        epk = bytes.fromhex(envelope["epk"])
        iv = bytes.fromhex(envelope["iv"])
        ct = bytes.fromhex(envelope["ct"])
    except (ValueError, TypeError) as e:
        raise ProvisionError(f"envelope field is not hex: {type(e).__name__}") from e
    if len(nonce) != NONCE_BYTES or len(epk) != 32 or len(iv) != _AEAD_NONCE_BYTES:
        raise ProvisionError("envelope field has the wrong length")
    shared = td_private.exchange(x25519.X25519PublicKey.from_public_bytes(epk))
    key = _derive(shared, nonce, _HKDF_INFO_KEY)
    try:
        plaintext = ChaCha20Poly1305(key).decrypt(iv, ct, nonce)
    except Exception as e:  # noqa: BLE001 - one error class, no detail leaks
        raise ProvisionError("envelope does not decrypt: wrong key, or tampered") from e
    try:
        payload = json.loads(plaintext)
    except ValueError as e:
        raise ProvisionError("decrypted payload is not JSON") from e
    return validate_payload(payload), key


def log_token(session_key: bytes) -> str:
    """Bearer token for the log pull, derived from the session key so the
    token itself never has to be exchanged."""
    return _derive(session_key, b"", _HKDF_INFO_LOG_TOKEN).hex()


def seal_bytes(session_key: bytes, data: bytes, aad: bytes) -> dict:
    import os as _os

    _h, _x, ChaCha20Poly1305, _k = _require_crypto()
    iv = _os.urandom(_AEAD_NONCE_BYTES)
    return {"iv": iv.hex(), "ct": ChaCha20Poly1305(session_key).encrypt(iv, data, aad).hex()}


def unseal_bytes(session_key: bytes, body: dict, aad: bytes) -> bytes:
    _h, _x, ChaCha20Poly1305, _k = _require_crypto()
    try:
        return ChaCha20Poly1305(session_key).decrypt(
            bytes.fromhex(body["iv"]), bytes.fromhex(body["ct"]), aad,
        )
    except Exception as e:  # noqa: BLE001
        raise ProvisionError("log response does not decrypt") from e


class RingLogHandler(logging.Handler):
    """Keeps the last LOG_RING_LINES formatted records in memory, numbered, so
    an operator can pull them incrementally. Never touches disk."""

    def __init__(self, capacity: int = LOG_RING_LINES) -> None:
        super().__init__()
        from collections import deque

        self._lines: deque = deque(maxlen=capacity)
        self._seq = 0
        self._lock = __import__("threading").Lock()
        self.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:  # noqa: BLE001 - a bad record must not kill the miner
            return
        with self._lock:
            self._seq += 1
            self._lines.append((self._seq, line))

    def since(self, seq: int) -> tuple[int, list[str]]:
        with self._lock:
            lines = [ln for s, ln in self._lines if s > seq]
            return self._seq, lines


class ProvisionStore:
    """What the miner pushed, held in memory for the life of the process.

    NEVER WRITTEN TO DISK. The dstack data volume is encrypted and would be
    safe, but persisting a key means it outlives the attestation that justified
    releasing it: a later boot with a different measurement would find it
    already there. Re-provisioning on every start costs one round trip and keeps
    "the miner saw a valid quote" and "the key is present" the same event.
    """

    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        # The session key the successful push was sealed with. Held so the
        # operator who provisioned this TD — and only them — can pull its logs.
        self.session_key: bytes = b""

    @property
    def ready(self) -> bool:
        return bool(self._values)

    def accept(self, payload: dict, session_key: bytes = b"") -> None:
        self._values = validate_payload(payload)
        self.session_key = session_key
        # Field NAMES only. The values are the secrets this whole design exists
        # to protect, and a log line is the easiest place to lose one.
        logger.info("provisioned with fields: %s", sorted(self._values))

    def get(self, field: str) -> str:
        if field not in ALLOWED_FIELDS:
            raise ProvisionError(f"{field!r} is not a provisioning field")
        return self._values.get(field, "")


def serve(store: ProvisionStore, port: int = PROVISION_PORT, host: str = "0.0.0.0"):
    """Run the TD-side receiver. Returns the HTTPServer; caller owns shutdown.

    Three endpoints and a status. `/provision/attest` takes the miner's nonce
    and returns a fresh attestation over `nonce || sha256(td_pubkey)` plus the
    public key — that is how the TD proves what it is, and what it holds,
    without ever holding a credential. `/provision` takes the SEALED payload once
    the miner is satisfied; plaintext is refused, and the envelope's nonce must
    be one this receiver attested and has not consumed. `/provision/logs` hands
    the operator who provisioned it — proven by a token derived from the
    session key — the miner's recent log lines, encrypted.

    Binds 0.0.0.0 because the miner reaches it from outside the guest. That is
    safe only because of what the endpoint does NOT do: it hands out an
    attestation and a public key, which are public information, and it accepts
    a payload only when it decrypts under a key bound into that attestation and
    validates against a closed allow-list. An attacker who reaches this port can
    learn what image is running, which is already discoverable, and can push a
    payload of their own, which gets them a miner running with THEIR API key and
    THEIR head under the miner's own hotkey — a denial-of-service on one miner,
    not a consensus break — so the operator firewalls the port to their own
    address regardless. What they can no longer do is read the operator's push.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse

    from fugal_subnet.tee import dstack_client

    _h, x25519, _c, _k = _require_crypto()
    td_private = x25519.X25519PrivateKey.generate()
    td_public = td_private.public_key().public_bytes_raw()
    pending: dict[str, None] = {}          # attested nonces awaiting a push
    pending_lock = threading.Lock()

    ring = RingLogHandler()
    logging.getLogger().addHandler(ring)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's name
            url = urlparse(self.path)
            if url.path == _STATUS_PATH:
                return self._send(200, {"provisioned": store.ready})
            if url.path == _LOGS_PATH:
                # Operator-only. The bearer token is derived from the session
                # key of the push that provisioned this TD, so it exists only
                # for the party that held the other half of the exchange, and
                # the lines go back encrypted under that same key.
                auth = self.headers.get("Authorization", "")
                if not store.session_key or auth != f"Bearer {log_token(store.session_key)}":
                    return self._send(403, {"error": "not the provisioning operator"})
                try:
                    since = int(parse_qs(url.query).get("since", ["0"])[0])
                except ValueError:
                    return self._send(400, {"error": "since must be an integer"})
                seq, lines = ring.since(since)
                body = json.dumps({"next": seq, "lines": lines}).encode()
                return self._send(200, seal_bytes(store.session_key, body, b"logs"))
            self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError as e:
                return self._send(400, {"error": f"bad json: {e}"})

            if self.path == _ATTEST_PATH:
                nonce = body.get("nonce", "")
                try:
                    report_data = report_data_for(nonce, td_public)
                except (ProvisionError, ValueError) as e:
                    return self._send(400, {"error": str(e)})
                try:
                    blob = dstack_client.attest(report_data)
                except Exception as e:  # noqa: BLE001 - reported, not swallowed
                    logger.exception("attestation failed during provisioning")
                    return self._send(503, {"error": f"attestation unavailable: {e}"})
                with pending_lock:
                    pending[nonce] = None
                    while len(pending) > _MAX_PENDING_NONCES:
                        pending.pop(next(iter(pending)))
                return self._send(200, {"attestation": blob.hex(), "pubkey": td_public.hex()})

            if self.path == _PUSH_PATH:
                nonce = body.get("nonce", "") if isinstance(body, dict) else ""
                with pending_lock:
                    attested = pending.pop(nonce, "absent") is None
                if not attested:
                    # Either never attested here, or already spent: a replayed
                    # envelope, or a push aimed at a TD that did not sign for it.
                    return self._send(400, {"error": "nonce was not attested by this TD, or was already used"})
                try:
                    payload, key = unseal(td_private, body)
                    store.accept(payload, key)
                except ProvisionError as e:
                    # The message names rejected FIELDS, never their values.
                    return self._send(400, {"error": str(e)})
                return self._send(200, {"provisioned": True})

            self._send(404, {"error": "not found"})

        def log_message(self, fmt, *args):
            # Default logs the request line to stderr, which would print a
            # rejected field name on every bad push and interleave with the
            # miner's own output. Route it through our logger instead.
            logger.debug("provision http: " + fmt, *args)

    server = HTTPServer((host, port), Handler)
    logger.info("provisioning receiver listening on %s:%d", host, port)
    return server


def push(
    address: str,
    payload: dict,
    *,
    approved_measurements,
    expected_app_identity: str = "",
    expected_instance_id: str = "",
    timeout: int = 30,
) -> bytes:
    """Verify a TD is what we expect, then send it per-miner data, sealed.

    Returns the session key. Keep it (0600) if you want to pull the TD's logs
    later with `pull_logs`; it never crosses the wire and cannot be recovered.

    THE MINER SIDE. Runs outside the enclave, on the machine that created the
    TD and therefore knows its address. Everything before the final POST exists
    so that a secret is never sent to something we have not identified.

    The order of checks is the security property, not a style choice:

      1. The quote is a genuine, Intel-signed quote over OUR nonce. The nonce is
         why a captured attestation cannot be replayed later — an old quote
         carries an old nonce.
      2. The base measurement is on the approved list. This is the image.
      3. The event log REPLAYS to the RTMR3 in that quote. Until this passes,
         nothing in the log means anything; it arrived from the thing we are
         trying to authenticate.
      4. Only now read the log's payloads: the compose hash is the approved app,
         and the instance-id is the instance WE created.

    Step 4 is what defeats a cloud provider booting the approved image to
    harvest a key. They would produce a valid quote with a correct measurement
    and a correct compose hash — and a different instance-id, because they did
    not create this instance. `expected_instance_id` is the check that turns
    that from an accepted risk into a refusal, so callers should supply it.
    """
    import json
    import urllib.request

    from fugal_subnet.tee.attestation import measurement_id, parse_quote, replay_event_log
    from fugal_subnet.tee.verify import unwrap_attestation

    validate_payload(payload)          # fail before contacting anything
    nonce = new_nonce()

    def _post(path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{address.rstrip('/')}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    answer = _post(_ATTEST_PATH, {"nonce": nonce})
    blob = bytes.fromhex(answer["attestation"])
    try:
        td_pubkey = bytes.fromhex(answer.get("pubkey", ""))
    except ValueError:
        td_pubkey = b""
    if len(td_pubkey) != 32:
        raise ProvisionError(
            "TD returned no X25519 public key with its attestation — an old "
            "receiver that would accept plaintext. Refusing to push to it."
        )
    quote_bytes, event_log = unwrap_attestation(blob)
    quote = parse_quote(quote_bytes)

    # 1. our nonce AND the hash of the key we are about to encrypt to, both
    #    under the Intel signature. Without the second half a genuine TD's
    #    quote would prove who we are talking to and nothing about who can read
    #    what we send.
    expected_rd = report_data_for(nonce, td_pubkey).hex()
    if quote.report_data != expected_rd:
        raise ProvisionError(
            "attestation is not over our nonce and the TD's key — this quote was "
            "produced for someone else, replayed from an earlier session, or the "
            "key offered is not the one the hardware signed for"
        )

    # 2. the image
    measured = measurement_id(quote)
    if measured not in set(approved_measurements):
        raise ProvisionError(
            f"TD reports measurement {measured[:16]}..., which is not approved. "
            f"Refusing to send anything to an image we do not recognise."
        )

    # 3. the log must reproduce the signed register before it is read
    if expected_app_identity or expected_instance_id:
        if not event_log:
            raise ProvisionError(
                "no event log in the attestation, so the app identity cannot be "
                "checked. A bare quote proves the image booted, not what ran."
            )
        replayed, events = replay_event_log(event_log)
        if replayed != quote.rtmr3:
            raise ProvisionError(
                "event log does not replay to the attested RTMR3 — the log is "
                "not the one this hardware signed, so none of it is believable"
            )
        # 4. and only now are its payloads worth reading
        if expected_app_identity:
            got = events.get("compose-hash", b"").hex()
            if got != expected_app_identity:
                raise ProvisionError(
                    f"TD is running app {got[:16] or '(none)'}..., expected "
                    f"{expected_app_identity[:16]}..."
                )
        if expected_instance_id:
            got = events.get("instance-id", b"").hex()
            if got != expected_instance_id:
                raise ProvisionError(
                    f"TD is instance {got[:16] or '(none)'}..., expected "
                    f"{expected_instance_id[:16]}.... This is the check that "
                    f"stops a correctly-imaged TD we did not create from being "
                    f"handed our key."
                )

    envelope, session_key = seal(payload, td_pubkey, nonce)
    _post(_PUSH_PATH, envelope)
    logger.info("provisioned TD at %s with fields: %s (sealed)", address, sorted(payload))
    return session_key


def pull_logs(address: str, session_key: bytes, since: int = 0,
              timeout: int = 30) -> tuple[int, list[str]]:
    """Fetch the TD's log lines after `since`, as the operator who provisioned it.

    Returns (next_seq, lines). Both the request (bearer token derived from the
    session key) and the response (sealed under it) stay between the two
    parties that did the key exchange; nobody who can merely reach the port
    learns anything.
    """
    import json
    import urllib.request

    req = urllib.request.Request(
        f"{address.rstrip('/')}{_LOGS_PATH}?since={int(since)}",
        headers={"Authorization": f"Bearer {log_token(session_key)}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    data = json.loads(unseal_bytes(session_key, body, b"logs"))
    return int(data["next"]), list(data["lines"])
