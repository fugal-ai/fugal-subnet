"""SCALE codec primitives, enough to read a TDX event log off the wire.

dstack ships its attestation as a SCALE-encoded blob — a 4-byte prefix, the TDX
quote, then the TPM quote and event log — not as JSON. SCALE (Simple
Concatenated Aggregate Little-Endian) is Substrate's codec, published and
stable, so this module implements the primitives and nothing dstack-specific.

That split is deliberate. The primitives can be verified against the codec's own
published test vectors with no hardware and no dstack, so they are trustworthy
today. The struct layout on top of them cannot be verified without a real blob,
and twice already a documented structure has turned out to describe a
neighbouring thing — the compose-hash normalisation rules governed runtime
events rather than the compose file, and the RTMR3 sysfs path in the docs was
not the path the kernel exposes. Both times the artifact settled it and the
specification did not. So the parts that can be proven live here; the parts that
need an artifact are marked as needing one.

Decoding is strict: every function reports how many bytes it consumed, an
over-long length is refused against the remaining buffer rather than trusted,
and nothing silently returns a partial result. The bytes arrive from a miner.
"""
from __future__ import annotations


class ScaleError(ValueError):
    """Malformed SCALE input. Always a rejection, never a partial decode."""


def decode_compact(buf: bytes, offset: int = 0) -> tuple[int, int]:
    """SCALE compact integer. Returns (value, bytes_consumed).

    The low two bits select the width, which is why a compact value cannot be
    read by looking at a fixed number of bytes:

        0b00  single byte,   value = b >> 2                    0 .. 63
        0b01  two bytes,     value = u16 >> 2                  0 .. 16383
        0b10  four bytes,    value = u32 >> 2                  0 .. 2**30-1
        0b11  big integer,   (b >> 2) + 4 bytes follow, LE
    """
    if offset >= len(buf):
        raise ScaleError(f"compact integer at {offset} is past the end of {len(buf)} bytes")
    first = buf[offset]
    mode = first & 0b11
    if mode == 0b00:
        return first >> 2, 1
    if mode == 0b01:
        if offset + 2 > len(buf):
            raise ScaleError("truncated two-byte compact integer")
        return int.from_bytes(buf[offset:offset + 2], "little") >> 2, 2
    if mode == 0b10:
        if offset + 4 > len(buf):
            raise ScaleError("truncated four-byte compact integer")
        return int.from_bytes(buf[offset:offset + 4], "little") >> 2, 4
    n = (first >> 2) + 4
    if offset + 1 + n > len(buf):
        raise ScaleError(f"truncated {n}-byte big compact integer")
    return int.from_bytes(buf[offset + 1:offset + 1 + n], "little"), 1 + n


def encode_compact(value: int) -> bytes:
    """Inverse of decode_compact. Present so the decoder can be round-trip
    tested against arbitrary values rather than only against a handful of
    published vectors."""
    if value < 0:
        raise ScaleError("compact integers are unsigned")
    if value < 0b1 << 6:
        return bytes([value << 2])
    if value < 0b1 << 14:
        return ((value << 2) | 0b01).to_bytes(2, "little")
    if value < 0b1 << 30:
        return ((value << 2) | 0b10).to_bytes(4, "little")
    raw = value.to_bytes((value.bit_length() + 7) // 8, "little")
    return bytes([((len(raw) - 4) << 2) | 0b11]) + raw


def decode_bytes(buf: bytes, offset: int = 0) -> tuple[bytes, int]:
    """A SCALE Vec<u8>: compact length, then that many bytes.

    The length is attacker-controlled, so it is checked against what is actually
    present instead of being used to slice — a Python slice past the end returns
    a short result silently, which would turn a malformed log into a plausible
    one.
    """
    length, used = decode_compact(buf, offset)
    start = offset + used
    if start + length > len(buf):
        raise ScaleError(
            f"Vec<u8> claims {length} bytes at offset {start} but only "
            f"{len(buf) - start} remain"
        )
    return buf[start:start + length], used + length


def decode_str(buf: bytes, offset: int = 0) -> tuple[str, int]:
    raw, used = decode_bytes(buf, offset)
    try:
        return raw.decode("utf-8"), used
    except UnicodeDecodeError as e:
        raise ScaleError(f"string is not valid UTF-8: {e}") from e


def decode_u32(buf: bytes, offset: int = 0) -> tuple[int, int]:
    if offset + 4 > len(buf):
        raise ScaleError("truncated u32")
    return int.from_bytes(buf[offset:offset + 4], "little"), 4


def decode_vec(buf: bytes, offset: int, item) -> tuple[list, int]:
    """A SCALE Vec<T>: compact count, then that many items.

    `item(buf, offset) -> (value, consumed)`. The count is checked against the
    remaining buffer before looping, so a log claiming four billion entries is
    refused immediately rather than after four billion iterations.
    """
    count, used = decode_compact(buf, offset)
    if count > len(buf) - offset - used:
        raise ScaleError(
            f"Vec claims {count} items but only {len(buf) - offset - used} "
            "bytes remain; no item can be shorter than one byte"
        )
    out, pos = [], offset + used
    for _ in range(count):
        value, consumed = item(buf, pos)
        out.append(value)
        pos += consumed
    return out, pos - offset
