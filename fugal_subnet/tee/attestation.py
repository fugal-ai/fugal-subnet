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


# The kernel's unified TSM measurement-register ABI (tsm-mr). Writing a
# register-width digest to this file extends that RTMR. Present on guest
# kernels carrying the tsm-mr series; absent on older ones, which is why
# extend_rtmr3 reports failure rather than assuming success.
_TSM_MR_RTMR3 = "/sys/class/misc/tdx_guest/mr/rtmr3"

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


def verify_dcap(quote_bytes: bytes) -> bool:
    """Verify TDX quote via Intel DCAP collateral.

    Requires the dcap_qvl package. Returns True if the quote signature
    and collateral chain are valid. In mock mode, this is skipped.

    Raises:
        ImportError: If dcap_qvl is not installed (configuration error).
    """
    try:
        from dcap_qvl import get_collateral_and_verify
    except ImportError:
        raise ImportError(
            "dcap_qvl not installed — DCAP verification requires it. "
            "Install with: pip install dcap-qvl"
        )

    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run, get_collateral_and_verify(quote_bytes)
                ).result(timeout=30)
        else:
            result = loop.run_until_complete(
                get_collateral_and_verify(quote_bytes)
            )
        logger.info("DCAP verification passed: %s", result)
        return True
    except Exception:
        logger.exception("DCAP verification failed")
        return False


def extract_report_data(quote_bytes: bytes) -> bytes:
    """Extract the 64-byte report_data from a raw TDX quote."""
    if len(quote_bytes) < _MIN_QUOTE_LEN:
        raise ValueError(f"Quote too short: {len(quote_bytes)} bytes")
    return quote_bytes[568:632]
