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
ALLOWED_FIELDS = frozenset({"head_b64", "hotkey_ss58", "openrouter_api_key"})

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
# Deliberately small and deliberately boring. The security lives in what the
# miner CHECKS before it pushes, not in the wire format: the TD proves what it
# is with an Intel-signed quote over a nonce the miner chose, and everything
# else is an ordinary HTTP round trip. A clever protocol here would add surface
# without adding a property.

PROVISION_PORT = 8092

_ATTEST_PATH = "/provision/attest"
_PUSH_PATH = "/provision"
_STATUS_PATH = "/provision/status"


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

    @property
    def ready(self) -> bool:
        return bool(self._values)

    def accept(self, payload: dict) -> None:
        self._values = validate_payload(payload)
        # Field NAMES only. The values are the secrets this whole design exists
        # to protect, and a log line is the easiest place to lose one.
        logger.info("provisioned with fields: %s", sorted(self._values))

    def get(self, field: str) -> str:
        if field not in ALLOWED_FIELDS:
            raise ProvisionError(f"{field!r} is not a provisioning field")
        return self._values.get(field, "")


def serve(store: ProvisionStore, port: int = PROVISION_PORT, host: str = "0.0.0.0"):
    """Run the TD-side receiver. Returns the HTTPServer; caller owns shutdown.

    Two endpoints and a status. `/provision/attest` takes the miner's nonce and
    returns a fresh attestation over it — that is how the TD proves what it is
    without ever holding a credential. `/provision` takes the payload once the
    miner is satisfied.

    Binds 0.0.0.0 because the miner reaches it from outside the guest. That is
    safe only because of what the endpoint does NOT do: it hands out an
    attestation, which is public information — a TDX quote is Intel-signed
    evidence, not a secret — and it accepts a payload it validates against a
    closed allow-list. An attacker who reaches this port can learn what image is
    running, which is already discoverable, and can push a payload, which gets
    them a miner running with their API key and their head under the miner's own
    hotkey. That is a denial-of-service on one miner, not a consensus break, and
    the operator should firewall the port to their own address regardless.
    """
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from fugal_subnet.tee import dstack_client

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's name
            if self.path != _STATUS_PATH:
                return self._send(404, {"error": "not found"})
            self._send(200, {"provisioned": store.ready})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError as e:
                return self._send(400, {"error": f"bad json: {e}"})

            if self.path == _ATTEST_PATH:
                nonce = body.get("nonce", "")
                try:
                    report_data = bytes.fromhex(nonce)
                except ValueError:
                    return self._send(400, {"error": "nonce must be hex"})
                if len(report_data) != NONCE_BYTES:
                    return self._send(
                        400,
                        {"error": f"nonce must be {NONCE_BYTES} bytes, got {len(report_data)}"},
                    )
                try:
                    blob = dstack_client.attest(report_data)
                except Exception as e:  # noqa: BLE001 - reported, not swallowed
                    logger.exception("attestation failed during provisioning")
                    return self._send(503, {"error": f"attestation unavailable: {e}"})
                return self._send(200, {"attestation": blob.hex()})

            if self.path == _PUSH_PATH:
                try:
                    store.accept(body)
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
) -> None:
    """Verify a TD is what we expect, then send it per-miner data.

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
    quote_bytes, event_log = unwrap_attestation(blob)
    quote = parse_quote(quote_bytes)

    # 1. our nonce, padded the way the agent pads it
    expected_rd = bytes.fromhex(nonce).ljust(64, b"\0").hex()
    if quote.report_data != expected_rd:
        raise ProvisionError(
            "attestation is not over our nonce — this quote was produced for "
            "someone else, or replayed from an earlier session"
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

    _post(_PUSH_PATH, payload)
    logger.info("provisioned TD at %s with fields: %s", address, sorted(payload))
