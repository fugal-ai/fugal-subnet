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
