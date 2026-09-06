"""TDX quote parsing and DCAP verification.

Forked from ThirtySpokes/Chutes (MIT licensed) attestation patterns.
Parses Intel TDX v4 quotes and verifies the DCAP certificate chain.

TDX quote v4 binary layout:
  Header (48 bytes):
    [0:2]   version
    [2:4]   att_key_type
    [4:8]   tee_type
    [8:28]  reserved / vendor_id
    [28:48] user_data

  Body — td_report_body_t (584 bytes, starts at offset 48):
    [184:232] mrtd             (48 bytes) — initial TD measurement
    [376:424] rtmr0            (48 bytes) — runtime measurement register 0
    [424:472] rtmr1            (48 bytes) — runtime measurement register 1
    [472:520] rtmr2            (48 bytes) — runtime measurement register 2
    [520:568] rtmr3            (48 bytes) — runtime measurement register 3
    [568:632] report_data      (64 bytes) — user-supplied nonce/data
"""
from __future__ import annotations

import hashlib
import logging
import struct
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_FIELDS = [
    ("tee_tcb_svn",     48,  16),
    ("mrseam",          64,  48),
    ("mrsignerseam",   112,  48),
    ("seam_attributes", 160,   8),
    ("td_attributes",  168,   8),
    ("xfam",           176,   8),
    ("mrtd",           184,  48),
    ("mrconfigid",     232,  48),
    ("mrowner",        280,  48),
    ("mrownerconfig",  328,  48),
    ("rtmr0",          376,  48),
    ("rtmr1",          424,  48),
    ("rtmr2",          472,  48),
    ("rtmr3",          520,  48),
    ("report_data",    568,  64),
]
_MIN_QUOTE_LEN = 632
_TDX_QUOTE_VERSION = 4
_TEE_TYPE_TDX = 0x00000081


@dataclass
class TDXQuote:
    version: int
    tee_type: int
    tee_tcb_svn: str
    mrseam: str
    mrsignerseam: str
    seam_attributes: str
    td_attributes: str
    xfam: str
    mrtd: str
    mrconfigid: str
    mrowner: str
    mrownerconfig: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    rtmr3: str
    report_data: str
    raw: bytes

    @property
    def report_data_bytes(self) -> bytes:
        return bytes.fromhex(self.report_data)


def parse_quote(data: bytes) -> TDXQuote:
    """Parse a raw TDX v4 quote binary into structured fields."""
    if len(data) < _MIN_QUOTE_LEN:
        raise ValueError(
            f"Quote is {len(data)} bytes — expected at least {_MIN_QUOTE_LEN}"
        )
    version = struct.unpack_from("<H", data, 0)[0]
    tee_type = struct.unpack_from("<I", data, 4)[0]
    if version != _TDX_QUOTE_VERSION:
        raise ValueError(
            f"Unsupported quote version {version} (expected {_TDX_QUOTE_VERSION}); "
            "the field offsets below are v4-specific"
        )
    if tee_type != _TEE_TYPE_TDX:
        raise ValueError(
            f"Quote tee_type is 0x{tee_type:x}, not TDX (0x{_TEE_TYPE_TDX:x})"
        )
    fields = {name: data[off: off + size].hex() for name, off, size in _FIELDS}
    return TDXQuote(
        version=version,
        tee_type=tee_type,
        raw=data,
        **fields,
    )


def measurement_id(quote: TDXQuote) -> str:
    """Identity of the runtime image the hardware actually measured.

    This is what an approved-image check must compare against. It is derived
    from the quote's own measurement registers, which the CPU fills in and the
    Intel-signed attestation covers — not from any value the workload chose.

    MRTD is the initial TD measurement (the VM image). RTMR1 covers the kernel
    and RTMR2 the kernel cmdline and initrd — the boot chain that determines
    which code the image starts.

    RTMR0 is deliberately EXCLUDED. It records the TDVF configuration the host
    builds: virtual hardware setup, CPU count, memory size, device config. That
    is chosen by the cloud provider, not by us, and it is not a per-image
    discriminator — measured directly, two identical images on different machine
    shapes produce different RTMR0 and therefore different identities. Including
    it forks the approved list by instance size while proving nothing about the
    code. See docs/INVARIANTS.md I8.

    RTMR3 is excluded HERE but is not ignorable: it is the application register,
    and `runtime_identity()` is what belongs in it. It is kept out of this
    function because a userspace extend is only load-bearing inside a locked
    image — an attacker on an unlocked image simply extends the expected value.
    Binding it is a verification step against a replayed event log, not a term
    in this hash. Until the image is locked, this function proves which OS
    booted and nothing about which Fugal code ran.

    A workload can put anything it likes in report_data, and it can claim any
    `source_hash` it likes inside its own proof. It cannot forge these.
    """
    payload = bytes.fromhex(quote.mrtd + quote.rtmr1 + quote.rtmr2)
    return hashlib.sha256(payload).hexdigest()


# The kernel's unified TSM measurement-register ABI. Writing a register-width
# digest to this file extends that RTMR. Verified on a live GCP c3 TD running
# 6.17.0-1022-gcp: the file exists, the write succeeds, and the new value
# appears in the next Intel-signed quote. Absent on older guest kernels, which
# is why extend_rtmr3 reports failure rather than assuming success.
#
# The register name carries its hash algorithm, and the directory is
# `measurements`, not `mr`. Both were wrong in the first draft of this file and
# the only symptom was extend_rtmr3 quietly returning False forever — which is
# why the path is asserted against hardware and not against documentation.
_TSM_MR_RTMR3 = "/sys/class/misc/tdx_guest/measurements/rtmr3:sha384"

# RTMRs are SHA384 registers. The value written must be register-width, so the
# runtime identity is SHA384 and not the SHA256 used everywhere else.
_RTMR_DIGEST_BYTES = 48


def runtime_identity(source_hash: str, pool_hash: str, grader_hash: str,
                     upstream: str = "") -> str:
    """Digest of what this runtime *is*, for extension into RTMR3.

    Four things decide what a proof means: the code that produced it, the pool
    the slice was drawn from, the grader that judged the answers, and WHERE THE
    ANSWERS CAME FROM. A change to any of them changes every grade, so they are
    the runtime's identity.

    The upstream is here because leaving it out was a complete break of the
    incentive mechanism. It is read from FUGAL_OPENROUTER_BASE inside the
    miner's own TD, and the pool in that TD carries every question's gold
    answer — so a miner pointing it at a server of their own returns gold with
    two tokens of usage and collects perfect accuracy at near-zero attested
    cost, with every hash binding, the DCAP signature and the approved
    measurement still passing. Nothing else in the proof records it.

    Including it does not forbid configuration, which is the right shape: an
    operator may legitimately front OpenRouter with a gateway or a regional
    endpoint, and a local testnet may point at a stub. What changes is that a
    different upstream produces a different identity, so the network decides
    which configurations it will accept rather than trusting that nobody
    changed one. Advisory until the image is locked, like everything else in
    this register — see docs/INVARIANTS.md I8.

    Deliberately excludes anything per-epoch — nonce, slice, results. A register
    that moves with runtime data can never be on an approved list, which is the
    reason RTMR3 was dropped from `measurement_id` in the first place. This
    value is fixed for a given deployment and computable off-hardware from the
    repo, so an approved list of it is reviewable in a pull request.

    SHA384 to match the RTMR register width.
    """
    # v2: the version tag is part of the payload, so adding the upstream
    # deliberately changes every identity rather than silently colliding with
    # a v1 value computed without it.
    payload = "|".join(
        ("fugal-runtime-v2", source_hash, pool_hash, grader_hash, upstream)
    )
    return hashlib.sha384(payload.encode()).hexdigest()


def replay_rtmr(extends, initial: bytes = b"\x00" * _RTMR_DIGEST_BYTES) -> str:
    """Recompute an RTMR from the sequence of values extended into it.

    THE REGISTER NEVER HOLDS THE VALUE WRITTEN TO IT. The hardware computes
    `RTMR = SHA384(RTMR || input)` starting from 48 zero bytes at boot, so a
    verifier can never compare an RTMR to the digest someone claims to have
    extended — it has to replay the chain and check the result matches what the
    CPU signed. Confirmed on live TDX rather than assumed: writing
    sha384(b"fugal-runtime-test") to a fresh TD produced exactly
    sha384(bytes(48) + input).

    That is also what makes an untrusted extend log safe to read. The log
    arrives from a miner and is believable only because replaying it must
    reproduce a value inside the Intel-signed quote; anything invented fails to
    reproduce it. Verify the replay before trusting a single field.
    """
    reg = initial
    for i, item in enumerate(extends):
        digest = bytes.fromhex(item) if isinstance(item, str) else bytes(item)
        if len(digest) != _RTMR_DIGEST_BYTES:
            raise ValueError(
                f"extend {i} is {len(digest)} bytes; RTMR extends are "
                f"{_RTMR_DIGEST_BYTES} (SHA384)"
            )
        reg = hashlib.sha384(reg + digest).digest()
    return reg.hex()


def expected_rtmr3(runtime_identity_hex: str) -> str:
    """RTMR3 as it should read after exactly one extend of this identity.

    The single-entry case of `replay_rtmr`, which is what a Fugal miner
    produces today: one extend at startup, from a register the hardware zeroed
    at boot. A dstack image extends RTMR3 several times from its initrd, so
    that case needs the full event log — this is the shape the verifier grows
    into, not a different mechanism.
    """
    return replay_rtmr([runtime_identity_hex])


# dstack tags every runtime event it extends with this type.
_DSTACK_EVENT_TYPE = 0x08000001


def dstack_event_digest(name: str, payload: bytes, version: int = 2) -> str:
    """The 48-byte value dstack extends for one runtime event.

    Two encodings, both SHA384:

      v1  sha384( u32_le(type) || b":" || name || b":" || payload )
      v2  sha384( RFC 8785 canonical JSON of
                  {"name": name, "type": type, "payload": hex(payload)} )

    NOTE THE TWO DIFFERENT HASHES. Event *payloads* are SHA-256 content hashes
    (the compose hash is sha256 of the docker-compose), while the *extend* over
    them is SHA-384 because that is the register width. Conflating them
    produces a value that never reproduces the quote.

    json.dumps with sorted keys and no whitespace equals JCS **for this object
    shape only** — flat, ASCII keys, string and small-integer values. JCS and
    json.dumps diverge on floats and non-ASCII escaping, so if this object ever
    grows a field that is not one of those, this shortcut stops being correct
    and a real JCS encoder is required.
    """
    import json
    import struct

    if version == 1:
        blob = (struct.pack("<I", _DSTACK_EVENT_TYPE) + b":"
                + name.encode("utf-8") + b":" + bytes(payload))
        return hashlib.sha384(blob).hexdigest()
    canonical = json.dumps(
        {"name": name, "payload": bytes(payload).hex(), "type": _DSTACK_EVENT_TYPE},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha384(canonical.encode("utf-8")).hexdigest()


def replay_event_log(events, imr: int = 3, upto: str | None = None) -> tuple[str, dict]:
    """Replay one measurement register from an event log; return (rtmr, events).

    THE LOG IS UNTRUSTED. It reaches a validator from the miner, and the only
    reason any field in it can be believed is that replaying it must reproduce a
    register value inside the Intel-signed quote. So the caller must compare the
    returned register to the quote BEFORE reading a single event payload — a log
    that does not reproduce it is discarded whole, not partially trusted.

    Order is load-bearing and preserved: extends are not commutative, so a
    verifier that sorted or de-duplicated the log would let a miner rearrange it
    to reach a chosen value.

    `upto` stops after the named event, which is how a verifier can bind only as
    far as `compose-hash` and ignore per-instance events that follow it.
    """
    reg = b"\x00" * _RTMR_DIGEST_BYTES
    seen: dict[str, bytes] = {}
    for i, ev in enumerate(events):
        if int(ev.get("imr", -1)) != imr:
            continue
        digest_hex = ev.get("digest", "")
        digest = bytes.fromhex(digest_hex) if isinstance(digest_hex, str) else bytes(digest_hex)
        if len(digest) != _RTMR_DIGEST_BYTES:
            raise ValueError(f"event {i} digest is {len(digest)} bytes, need 48")

        # v2 events may carry the preimage. When present it must hash to the
        # digest, or the log is lying about what it extended.
        preimage = ev.get("preimage")
        if preimage:
            raw = bytes.fromhex(preimage) if isinstance(preimage, str) else bytes(preimage)
            if hashlib.sha384(raw).hexdigest() != digest.hex():
                raise ValueError(
                    f"event {i} ({ev.get('event','?')}) preimage does not hash to "
                    "its digest"
                )
        reg = hashlib.sha384(reg + digest).digest()
        name = str(ev.get("event", ""))
        payload = ev.get("event_payload", b"")
        seen[name] = bytes.fromhex(payload) if isinstance(payload, str) else bytes(payload)
        if upto is not None and name == upto:
            break
    return reg.hex(), seen


def extend_rtmr3(identity_hex: str) -> bool:
    """Extend RTMR3 with the runtime identity. True if the hardware took it.

    ADVISORY UNTIL THE IMAGE IS LOCKED, and the docstring says so because the
    code cannot. On an unmeasured filesystem an attacker running modified code
    simply extends the value we expect, so a matching RTMR3 proves nothing on
    its own. It becomes load-bearing when the extend is performed from a
    measured initrd inside a dm-verity image, at which point the same value is
    evidence. Writing it now means that migration completes a design rather
    than introducing one.

    Never raises: a miner that cannot extend must still run, because the value
    is not yet enforced. It returns False instead, and callers are expected to
    say so loudly — a silent failure here would be indistinguishable from a
    binding that never happened.

    This is an EXTEND, not a set. The hardware computes
    `RTMR = SHA384(RTMR || input)`, starting from 48 zero bytes at boot, so the
    register never holds the value written to it. A verifier therefore cannot
    compare RTMR3 to `runtime_identity()` directly — it must replay the chain
    of extends from zero and check the result equals the quote's RTMR3. That is
    the same event-log replay dstack requires, and it is why binding RTMR3 is a
    verification procedure rather than a term in `measurement_id`.

    Confirmed on live hardware: writing sha384(b"fugal-runtime-test") to a
    freshly booted TD moved RTMR3 from zeros to
    f6fdca9f66372d80685dc6be023d9e6ae25a42334f3b59de24c78da941883a64..., which
    equals sha384(bytes(48) + input) and appeared unchanged in the next quote.
    """
    try:
        digest = bytes.fromhex(identity_hex)
    except ValueError:
        logger.error("runtime identity %r is not hex; RTMR3 not extended", identity_hex)
        return False
    if len(digest) != _RTMR_DIGEST_BYTES:
        logger.error(
            "runtime identity is %d bytes, RTMR3 needs %d (SHA384); not extended",
            len(digest), _RTMR_DIGEST_BYTES,
        )
        return False
    try:
        with open(_TSM_MR_RTMR3, "wb") as f:
            f.write(digest)
    except OSError as e:
        logger.warning(
            "RTMR3 not extended (%s): %s. The runtime identity is NOT bound to "
            "this TD's measurement registers. This kernel may predate the "
            "tsm-mr interface, or the process may lack write access.",
            _TSM_MR_RTMR3, e,
        )
        return False
    logger.info("RTMR3 extended with runtime identity %s", identity_hex[:16])
    return True


_pccs_logged: set[str] = set()


def _log_pccs_endpoint(url: str) -> None:
    """Say where collateral comes from, once per endpoint per process.

    A per-proof log line would be 256 lines an epoch; silence is how the
    dependency stayed invisible in the first place. Once is the compromise.
    """
    if url not in _pccs_logged:
        _pccs_logged.add(url)
        logger.info("DCAP collateral endpoint: %s", url)


def verification_order(nonce_hex: str, n: int) -> list[int]:
    """The order proofs are verified in, derived from the epoch nonce.

    Order is invisible until the collateral ceiling binds, and then it decides
    who goes unverified. In UID order that is the same high-UID miners every
    epoch -- a standing disadvantage no miner caused and none could escape,
    which is an I4 problem even though no miner produced it.

    The nonce is `derive_nonce(epoch_id, block_hash)`, so this permutation is:

      - **identical on every validator** (I1) -- it depends only on the nonce
        and the field size, both of which are chain-derived;
      - **not miner-influenceable** (I4) -- no part of it comes from anything a
        miner sends;
      - **different every epoch**, so the exposure rotates instead of settling.

    A miner can compute it once the nonce is public and know it is late in the
    queue. It still cannot know where the cut falls, and UID order leaked
    strictly more, so this is not a regression.

    Extracted from the validator loop so it can be tested at all. Inline, the
    property that two validators agree was asserted only by argument.
    """
    return sorted(
        range(n),
        key=lambda uid: hashlib.sha256(f"{nonce_hex}:{uid}".encode()).digest(),
    )


class CollateralBudget:
    """A per-epoch ceiling on time spent fetching DCAP collateral.

    WHY A SECOND BOUND EXISTS, when every fetch is already bounded.

    The per-proof timeout stops one call hanging forever. It does not stop the
    EPOCH hanging, because the verify loop is serial over every UID: 256 proofs
    against a slow-but-alive PCCS costs 256 x the per-proof budget, which is
    2560 s at the default against an 1800 s post-collection window. The epoch
    returns and still misses its weight-setting window, which is an I6 failure
    wearing the costume of a successful run. Bounding one call was never
    sufficient; bounding the sum is.

    This is the *aggregate* half of that. Once the budget is spent, remaining
    proofs are reported unverifiable with **no network call at all**, so the
    phase has a hard ceiling no upstream can raise.

    A DESIGN TENSION, recorded because the rejected option looks better than it
    is. A budget measured in TIME adapts -- a healthy epoch verifies every
    proof -- but where it runs out depends on each validator's network luck, so
    two validators hold different verified sets. Capping the COUNT of fetches
    instead would cut at the same place on every validator and be deterministic,
    but it would also cap a healthy epoch below the field size, which is the
    normal case. Time is chosen because it only binds when something is already
    wrong, and because the divergence it leaves is the one INVARIANTS.md already
    accepts as unavoidable locally: the input is nondeterministic, and no error
    handling repairs a nondeterministic input. Only collateral-in-proof does.

    So this does NOT make validators agree. It converts "misses the weight
    window entirely" into "sets weights with a bounded unverifiable set", which
    is strictly better and still not a fix.
    """

    __slots__ = (
        "_max_timeouts", "_proofs_left", "_remaining", "_timeouts", "_total",
    )

    def __init__(
        self,
        total_s: float,
        max_timeouts: int | None = None,
        expected_proofs: int | None = None,
    ):
        from fugal_subnet.config import TEE_COLLATERAL_MAX_TIMEOUTS

        self._total = float(total_s)
        self._remaining = float(total_s)
        self._timeouts = 0
        self._max_timeouts = (
            TEE_COLLATERAL_MAX_TIMEOUTS if max_timeouts is None else int(max_timeouts)
        )
        # None means "unknown", and then no per-proof fair share is enforced.
        self._proofs_left = (
            None if expected_proofs is None else max(1, int(expected_proofs))
        )

    @property
    def remaining(self) -> float:
        return max(0.0, self._remaining)

    @property
    def exhausted(self) -> bool:
        """Spent, or the endpoint has proven it is not going to answer.

        The second half is not a time bound at all -- it is what keeps the file
        descriptors leaked by abandoned fetches to a handful per epoch instead
        of one per miner. A fetch cannot be cancelled without crashing the
        interpreter, so it is abandoned and keeps its socket; the only way to
        leak fewer is to dial fewer times.
        """
        return self._remaining <= 0.0 or self._timeouts >= self._max_timeouts

    @property
    def circuit_open(self) -> bool:
        """True when it was the endpoint, not the clock, that stopped us."""
        return self._timeouts >= self._max_timeouts

    def record_timeout(self) -> None:
        """One fetch was ABANDONED, and the endpoint is why.

        Only abandonment counts, because only abandonment leaks a descriptor --
        that is the entire quantity this breaker bounds. A fetch that ended in
        any other way, however unhappily, closed its socket on the way out.
        """
        self._timeouts += 1

    def record_completion(self) -> None:
        """One fetch finished on its own: a result, or an error, either way.

        Named for COMPLETION rather than success, and that distinction was a
        bug before it was a name. The reset used to fire only on a successful
        fetch, so a 5xx, a DNS failure or a reset left the counter standing --
        and three timeouts separated by two instant HTTP 500s opened a breaker
        that three comments described as "consecutive". Measured: the breaker
        opened after 5 dials, having seen the endpoint answer twice in ~0 ms.

        An endpoint that returns an error ANSWERED. It leaked no descriptor and
        it proved it is reachable, which is the opposite of the evidence this
        counter accumulates.
        """
        self._timeouts = 0

    def allowance(self, per_call_max: float) -> float:
        """The bound for the next fetch: whichever limit runs out first.

        Three limits, and the third is what makes a shared ceiling safe to
        have at all.

        1. `per_call_max` -- the hang guard.
        2. What is left. Handing the last fetch a full per-call budget would let
           it overshoot the epoch ceiling by that much, which is the quantity
           this class exists to bound.
        3. **A fair share of what is left.** A ceiling is a shared resource, so
           I4 applies to it: without this, one miner whose fetch stalls can
           consume time that belonged to everyone behind it in the queue.
           Whether that is reachable in practice turns on how long a PCCS takes
           to refuse an FMSPC it does not know -- a quantity nobody has
           measured. Rather than make the safety of the design depend on an
           unmeasured fact, the share is enforced and the question stops
           mattering.

        Dynamic rather than a static `total/N`: unused time from fast successes
        flows to later proofs, so a healthy epoch is untouched while the bound
        on any single proof still holds. With 600 s over 256 proofs the floor is
        2.34 s against a measured 690 ms warm round trip.
        """
        limits = [float(per_call_max), self.remaining]
        if self._proofs_left is not None:
            limits.append(self.remaining / max(1, self._proofs_left))
        return min(limits)

    def spend(self, elapsed_s: float) -> None:
        self._remaining -= max(0.0, float(elapsed_s))
        # Every fetch that got as far as costing time has had its turn, so the
        # share the REMAINING proofs are entitled to grows accordingly.
        if self._proofs_left is not None:
            self._proofs_left = max(1, self._proofs_left - 1)


class CollateralUnavailable(Exception):
    """DCAP collateral could not be fetched, so the quote was never judged.

    This is NOT "the proof is bad". It is "this validator could not check it",
    and the two must not be spelled the same way. Collapsing them is what makes
    a transient network failure indistinguishable from a forged quote, and it
    is why a bare timeout on the fetch would convert a hang into a false
    accusation against every honest miner in the field at once.

    Same category as the ImportError raised when `dcap_qvl` is missing: the
    operator's infrastructure, not the miner's bytes. Handled the same way.
    """


def verify_dcap(
    quote_bytes: bytes,
    pccs_url: str | None = None,
    budget: "CollateralBudget | None" = None,
) -> bool:
    """Verify TDX quote via Intel DCAP collateral.

    Returns True if the quote is genuine and Intel-signed. Returns False when
    the MINER's bytes are at fault -- an unparseable quote, a signature that
    does not check out. Raises when THIS VALIDATOR is at fault, which is the
    distinction the return type alone cannot carry.

    The fetch and the verification are separate calls rather than
    `get_collateral_and_verify`, so that a fetch failure is attributable. The
    quote is parsed BEFORE anything is fetched, because a quote that will not
    parse is the miner's problem and must never be reported as a collateral
    outage.

    `pccs_url` defaults to `config.TEE_PCCS_URL` and is ALWAYS passed to the
    library explicitly. Omitting it makes `dcap-qvl` fall back to its own
    default, which is how every --live validator came to fetch collateral from
    Phala without anyone choosing it. The endpoint is this project's decision,
    so this project names it; see the config entry for why that is a choice
    about availability and privacy rather than about trust.

    KNOWN RESIDUAL (I4). `get_collateral` derives its request from the FMSPC
    inside the quote, so a well-formed quote naming an FMSPC the upstream does
    not know produces a fetch failure that is the MINER's doing and is reported
    here as unavailable. dcap-qvl surfaces every failure as ValueError, so the
    two are separable only by message text, and branching on prose is the thing
    this exception exists to avoid. Parsing first removes the malformed-quote
    cases but not this one. It closes only when the fetch goes away entirely --
    see the collateral-in-proof design in docs/INVARIANTS.md.

    The fetch is bounded by `config.TEE_COLLATERAL_TIMEOUT`. A stalled PCCS is
    reported as unavailable, never as invalid: a timeout says this validator
    ran out of patience, which is a statement about the validator and not about
    the miner.

    `budget`, when given, bounds the collateral time of the WHOLE EPOCH rather
    than of this one call, and it is the half that actually satisfies I6 -- the
    verify loop is serial, so 256 individually-bounded fetches still overrun the
    epoch. Once it is spent this returns unavailable without touching the
    network. Omitting it leaves only the per-call bound, which is correct for a
    one-off caller and NOT sufficient inside the epoch loop.

    Raises:
        ImportError: If dcap_qvl is not installed (configuration error).
        ValueError: If no PCCS endpoint is configured (configuration error).
        CollateralUnavailable: If collateral could not be fetched, INCLUDING
            when the fetch timed out.
    """
    try:
        from dcap_qvl import get_collateral
        from dcap_qvl import parse_quote as _parse
        from dcap_qvl import verify as _verify
    except ImportError:
        raise ImportError(
            "dcap_qvl not installed — DCAP verification requires it. "
            "Install with: pip install dcap-qvl"
        )

    from fugal_subnet.config import (
        TEE_COLLATERAL_MAX_TIMEOUTS,
        TEE_COLLATERAL_TIMEOUT,
        TEE_PCCS_URL,
    )

    url = (pccs_url or TEE_PCCS_URL or "").strip()
    if not url:
        # An empty url is not a harmless omission: the library would silently
        # substitute its own default, which is the exact invisibility this
        # exists to remove. Refuse rather than fetch from somewhere nobody named.
        raise ValueError(
            "No PCCS URL configured. Set FUGAL_PCCS_URL (or config.TEE_PCCS_URL) "
            "to the endpoint this validator should fetch DCAP collateral from. "
            "Leaving it empty would let dcap-qvl choose one silently."
        )
    _log_pccs_endpoint(url)

    # 1. Parse before fetching. Miner-controlled bytes, judged locally, with no
    #    network involved -- so a malformed quote can never be misreported as
    #    an infrastructure failure, and never costs a round trip either.
    try:
        _parse(quote_bytes)
    except Exception as e:  # noqa: BLE001 - miner-controlled bytes
        logger.warning("DCAP: quote does not parse (%s: %s)", type(e).__name__, e)
        return False

    # 2. Fetch collateral. Everything here is the operator's infrastructure,
    #    so a failure raises rather than condemning the miner.
    # 1b. Refuse before dialling if the epoch's collateral budget is gone.
    #     Checked BEFORE the fetch, not after: the point is to stop spending
    #     time, and a call that is going to be abandoned anyway should never
    #     leave the process. This is what gives the phase a hard ceiling.
    if budget is not None and budget.exhausted:
        why = (
            f"{url} failed to answer the last "
            f"{TEE_COLLATERAL_MAX_TIMEOUTS} attempts, so this validator stopped "
            "dialling it for the rest of this epoch"
            if budget.circuit_open
            else "this epoch's collateral budget is exhausted"
        )
        raise CollateralUnavailable(
            f"{why}, so the fetch was not attempted. The proof was never "
            "judged -- this is an outage, not a bad proof."
        )

    allowance = (
        TEE_COLLATERAL_TIMEOUT if budget is None
        else budget.allowance(TEE_COLLATERAL_TIMEOUT)
    )
    started = time.monotonic()
    try:
        collateral = _run_coro(
            lambda: get_collateral(url, quote_bytes), timeout=allowance,
        )
    except TimeoutError as e:
        # Blame the endpoint ONLY if it had the full budget and still did not
        # answer. When the epoch ceiling has nearly run out, `allowance` is
        # whatever remains -- so a healthy endpoint answering in a flat 100 ms
        # "times out" at 0.047 s and gets a strike it did not earn. That is the
        # outage-versus-accusation distinction the unverifiable path exists for,
        # one level down: the clock ran out, the endpoint did not fail.
        endpoint_had_a_fair_chance = allowance >= TEE_COLLATERAL_TIMEOUT
        if budget is not None and endpoint_had_a_fair_chance:
            budget.record_timeout()
        # Named separately because `str(TimeoutError())` is empty, and the
        # generic branch below would render this as "TimeoutError: " with no
        # indication of what was waited on or for how long. An operator reading
        # the epoch log needs to see the budget to know whether to raise it or
        # to go and look at their PCCS.
        raise CollateralUnavailable(
            f"DCAP collateral fetch from {url} exceeded {allowance:g}s and was "
            + ("abandoned" if endpoint_had_a_fair_chance else
               "abandoned -- but that was this epoch's REMAINING budget, not "
               f"the {TEE_COLLATERAL_TIMEOUT:g}s an endpoint is normally given, "
               "so it is not evidence against the endpoint")
            + ". The proof was never judged -- this is an outage, not a bad proof."
        ) from e
    except Exception as e:  # noqa: BLE001 - network, DNS, TLS, upstream 5xx
        # It answered, badly. That still resets the breaker: the socket closed,
        # nothing leaked, and the endpoint demonstrated it is reachable.
        if budget is not None:
            budget.record_completion()
        raise CollateralUnavailable(
            f"could not fetch DCAP collateral from {url}: {type(e).__name__}: {e}"
        ) from e
    else:
        if budget is not None:
            budget.record_completion()
    finally:
        # Charged on EVERY exit, not only success. A fetch that timed out or
        # errored still consumed wall-clock the epoch does not get back, and a
        # budget that only counts successes would be spent slowest exactly when
        # the upstream is at its worst -- which is the case it exists for.
        if budget is not None:
            budget.spend(time.monotonic() - started)

    # 3. Judge the quote against the collateral. Local and deterministic given
    #    both inputs, so a failure here is the miner's again.
    try:
        result = _verify(quote_bytes, collateral, int(time.time()))
    except Exception:
        logger.exception("DCAP verification failed")
        return False

    # Record the verdict rather than only the fact one was reached.
    #
    # `verify` returns a report, and `report.status` is the TCB state of the
    # platform that produced the quote: "OK", but also "OUT_OF_DATE",
    # "CONFIGURATION_NEEDED", "SW_HARDENING_NEEDED". This function accepts all
    # of them, so today "the quote is genuine" is all a passing DCAP check
    # means, NOT "the platform is patched". Measured on real hardware: a live
    # dstack CVM reports UpToDate with no advisories, and an UpToDate box and a
    # REVOKED box are currently indistinguishable to this validator.
    #
    # Whether to enforce a status is a policy decision with real cost: on GCP
    # and Azure the firmware is the cloud provider's to patch, so rejecting
    # OUT_OF_DATE removes honest miners for their host's maintenance schedule,
    # and does it to every miner on a platform generation at once when Intel
    # publishes. So: log it, change nothing. A status is a consensus input the
    # moment it is enforced -- two validators with different thresholds
    # disagree about identical bytes -- and it is not enforced here.
    status = getattr(result, "status", None)
    advisories = list(getattr(result, "advisory_ids", None) or [])
    if status is not None and str(status).upper() not in ("OK", "UPTODATE"):
        logger.warning(
            "DCAP verification passed but the platform TCB status is %s%s. "
            "The quote is genuine; the hardware is not necessarily patched. "
            "Not enforced -- see docs/INVARIANTS.md.",
            status,
            f" (advisories: {', '.join(advisories)})" if advisories else "",
        )
    else:
        logger.info("DCAP verification passed: status=%s", status)
    return True


def _run_coro(make_coro, timeout: float | None = None):
    """Run one coroutine from sync code, whatever loop state we are in.

    `make_coro` is a factory rather than a coroutine because dcap-qvl's bindings
    require the coroutine to be CREATED inside the running loop -- constructing
    it first raises "no running event loop".

    `timeout` ABANDONS the work; it does not cancel it. That distinction is the
    whole design, and it was arrived at by crashing the interpreter.

    THE OBVIOUS IMPLEMENTATION IS UNSAFE HERE. `asyncio.wait_for` bounds the
    call correctly and then cancels the coroutine, and cancelling *this*
    coroutine unwinds a Rust future inside pyo3. Its tokio worker then touches
    the interpreter during finalization:

        thread 'tokio-rt-worker' panicked at pyo3/src/interpreter_lifecycle.rs:
        assertion failed: The Python interpreter is not initialized

    Measured against a local socket that accepts and never answers: the bound
    fires correctly at 2.01 s, the process stays usable, and then exits 134 or
    139 -- 4 runs in 10. The validator has no `async def` anywhere, so it takes
    this path for every proof of every epoch.

    So the work is handed to a DAEMON THREAD and the JOIN is what gets bounded.
    On timeout the thread is left running: the fetch finishes on its own, or the
    process exits without waiting for it. Nothing is ever cancelled, so pyo3 is
    never unwound. Measured with the same stalled socket: 12 runs, 12 clean
    exits.

    A thread rather than the old `ThreadPoolExecutor` because the pool's context
    manager calls `shutdown(wait=True)` on exit, which blocks until the hung
    call finishes and is exactly why the previous `timeout=30` did nothing.

    THE COST, stated because it is real and is not fully closed: an abandoned
    fetch keeps its socket, so each one leaks a file descriptor until the
    connection dies on its own -- measured at 1.02 fds per abandoned fetch.
    `CollateralBudget` bounds how many can be abandoned per epoch, and its
    circuit breaker stops dialling a PCCS that has already proven unresponsive,
    which is what keeps that number small. See `record_timeout`.
    """
    import asyncio
    import threading

    box: dict = {}

    async def _outer():
        return await make_coro()

    def _worker():
        try:
            box["value"] = asyncio.run(_outer())
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller
            box["error"] = e

    # Daemon so an abandoned fetch can never hold the process open at exit.
    worker = threading.Thread(
        target=_worker, name="dcap-collateral", daemon=True,
    )
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        # Deliberately still running. Raising here abandons it; joining or
        # cancelling is what we are avoiding.
        raise TimeoutError(
            f"collateral fetch abandoned after {timeout}s (still running)"
        )
    if "error" in box:
        raise box["error"]
    return box["value"]


def extract_report_data(quote_bytes: bytes) -> bytes:
    """Extract the 64-byte report_data from a raw TDX quote."""
    if len(quote_bytes) < _MIN_QUOTE_LEN:
        raise ValueError(f"Quote too short: {len(quote_bytes)} bytes")
    return quote_bytes[568:632]
