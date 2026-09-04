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
    fields = {name: data[off: off + size].hex() for name, off, size in _FIELDS}
    return TDXQuote(
        version=version,
        tee_type=tee_type,
        raw=data,
        **fields,
    )


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
