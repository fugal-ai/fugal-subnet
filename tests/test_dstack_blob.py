"""Decode and replay two REAL dstack attestation blobs.

Captured from two dstack CVMs on GCP c3-standard-4 that differ by one line of
docker-compose. Committed as fixtures because they are the only thing in this
repository that can falsify the event-log code without a confidential VM, and
because every previous attempt to write this code from a specification produced
something confident and wrong.

They contain nothing secret: a TDX quote is a public, Intel-signed attestation.
"""
import pathlib

import pytest

from fugal_subnet.scale import ScaleError, decode_dstack_attestation
from fugal_subnet.tee.attestation import measurement_id, parse_quote, replay_event_log

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _blob(name):
    p = FIXTURES / f"attestation_{name}.bin"
    if not p.exists():
        pytest.skip(f"fixture {p.name} not present")
    return p.read_bytes()


@pytest.mark.parametrize("name", ["A", "B"])
def test_the_event_log_replays_to_the_signed_register(name):
    """The property everything else rests on. If the replay does not reproduce
    the value the CPU signed, no field of the log may be believed."""
    d = decode_dstack_attestation(_blob(name))
    quote = parse_quote(d["quote"])
    replayed, _ = replay_event_log(d["events"], imr=3)
    assert replayed == quote.rtmr3


def test_the_base_measurement_is_stable_across_an_app_change():
    """What makes the two halves independently rotatable: changing the app must
    not move the base, or every app change would force a base rotation too."""
    a = measurement_id(parse_quote(decode_dstack_attestation(_blob("A"))["quote"]))
    b = measurement_id(parse_quote(decode_dstack_attestation(_blob("B"))["quote"]))
    assert a == b


def test_the_app_change_is_visible_in_the_compose_hash():
    seen = {}
    for name in ("A", "B"):
        d = decode_dstack_attestation(_blob(name))
        _, events = replay_event_log(d["events"], imr=3)
        seen[name] = events["compose-hash"].hex()
    assert seen["A"] != seen["B"], "one line of compose changed and the hash did not"


def test_rtmr3_differs_between_deploys_of_the_same_shape():
    """THE reason an approved list can never hold an RTMR3 value.

    A dstack TD extends RTMR3 nine times during boot and one of those events is
    instance-id, which is new on every deploy. An approved list storing RTMR3
    would pass its first test and then reject every honest miner from their
    second deploy onward — and the symptom would look like an attack.
    """
    rtmr3 = {}
    instance = {}
    for name in ("A", "B"):
        d = decode_dstack_attestation(_blob(name))
        rtmr3[name] = parse_quote(d["quote"]).rtmr3
        _, events = replay_event_log(d["events"], imr=3)
        instance[name] = events["instance-id"].hex()
    assert instance["A"] != instance["B"], "instance-id was expected to be per-deploy"
    assert rtmr3["A"] != rtmr3["B"]


def test_the_key_provider_is_visible_to_a_validator():
    """key-provider is its own RTMR3 event, so a validator can check it directly
    from the replayed log rather than trusting it to be baked into the compose."""
    d = decode_dstack_attestation(_blob("A"))
    _, events = replay_event_log(d["events"], imr=3)
    assert b"name" in events["key-provider"]


def test_a_truncated_blob_is_refused():
    blob = _blob("A")
    for cut in (2, 100, 8000, 20_000):
        with pytest.raises(ScaleError):
            decode_dstack_attestation(blob[:cut])


def test_a_corrupted_layout_is_caught_by_the_tpm_magic():
    """The magic is a canary: if the layout shifts, everything decoded before it
    is suspect and should not be reported as a successful parse."""
    blob = bytearray(_blob("A"))
    blob[2] ^= 0xFF                      # damage the quote length
    with pytest.raises(ScaleError):
        decode_dstack_attestation(bytes(blob))
