"""SCALE primitives, and their behaviour on hostile input.

dstack ships attestation as a SCALE blob rather than JSON, so reading the event
log means decoding one — and the blob arrives from a miner. Every length in it
is attacker-controlled, which makes "refuses politely" the property under test
rather than "decodes correctly".
"""
import pytest

from fugal_subnet.scale import (
    ScaleError,
    decode_bytes,
    decode_compact,
    decode_str,
    decode_u32,
    decode_vec,
    encode_compact,
)


@pytest.mark.parametrize("value,encoded", [
    (0, "00"), (1, "04"), (42, "a8"), (69, "1501"), (65535, "feff0300"),
])
def test_compact_matches_the_published_vectors(value, encoded):
    assert encode_compact(value).hex() == encoded
    assert decode_compact(bytes.fromhex(encoded)) == (value, len(encoded) // 2)


@pytest.mark.parametrize("value", [
    0, 1, 63, 64, 16383, 16384, 2**30 - 1, 2**30, 2**32, 2**64 - 1,
])
def test_compact_round_trips_across_every_width(value):
    """The low two bits select the width, so the boundaries are where a decoder
    goes wrong. Round-tripping proves more than the five published vectors."""
    buf = encode_compact(value)
    assert decode_compact(buf) == (value, len(buf))


def test_truncated_input_is_refused_not_guessed():
    for buf in (b"", b"\x01", b"\x02\x00", b"\x03\x00\x00"):
        with pytest.raises(ScaleError):
            decode_compact(buf)
    with pytest.raises(ScaleError):
        decode_u32(b"\x00\x00")


def test_an_overlong_length_is_refused_against_the_buffer():
    """A Python slice past the end returns a SHORT result silently, which would
    turn a malformed log into a plausible one. The length must be checked."""
    buf = encode_compact(9999) + b"only a few bytes"
    with pytest.raises(ScaleError, match="claims 9999 bytes"):
        decode_bytes(buf)


def test_an_absurd_item_count_is_refused_before_looping():
    """A log claiming four billion entries must fail immediately, not after
    four billion iterations — that is the difference between a rejection and a
    denial of service on every validator at once."""
    buf = encode_compact(2**30 - 1) + b"\x00"
    with pytest.raises(ScaleError, match="claims"):
        decode_vec(buf, 0, decode_u32)


def test_non_utf8_strings_are_refused():
    buf = encode_compact(2) + b"\xff\xfe"
    with pytest.raises(ScaleError, match="UTF-8"):
        decode_str(buf)


def test_round_trip_through_a_vec_of_byte_strings():
    items = [b"", b"compose-hash", bytes(range(256)), b"\x00" * 48]
    blob = encode_compact(len(items)) + b"".join(
        encode_compact(len(i)) + i for i in items)
    decoded, used = decode_vec(blob, 0, decode_bytes)
    assert decoded == items
    assert used == len(blob)


def test_offsets_are_reported_so_a_caller_can_walk_a_struct():
    """Every decoder returns what it consumed. A struct is field after field,
    and a decoder that did not report its own length could not be composed."""
    blob = encode_compact(3) + b"abc" + (7).to_bytes(4, "little")
    raw, used = decode_bytes(blob, 0)
    assert raw == b"abc"
    assert decode_u32(blob, used) == (7, 4)
