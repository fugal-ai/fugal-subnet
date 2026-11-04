"""Protocol v2 model list: a second, separately pinned price table selected by protocol version.

A release/v2 validator scores v1 epochs until the activation block, so the v1 table must stay
exactly as it is and every caller must ask for the table of the epoch it is scoring.
"""
import hashlib
import json

import pytest

from fugal_subnet import fingerprint
from fugal_subnet import routing_protocol as rp


def test_both_tables_ship_and_the_version_selects_one():
    v1, v2 = rp.models_for_protocol(1), rp.models_for_protocol(2)
    assert len(v1) == 17 and len(v2) == 19
    assert v1 == sorted(v1) and v2 == sorted(v2)          # the global index space is the sorted ids
    assert not set(v1) & set(v2)                          # a one-time list replacement
    assert set(rp.price_table(2)) == set(v2)
    with pytest.raises(ValueError):
        rp.price_table(3)


def test_v1_identity_is_unchanged_by_shipping_the_v2_table():
    """identity() is hashed into every v1 reveal and the v1 frame; it must not move."""
    assert rp.identity().startswith("49390166e93fbf5c")


def test_every_v2_model_pins_exactly_one_reasoning_setting_and_v1_sends_none():
    """OpenRouter's unified `reasoning` takes effort OR max_tokens, never both."""
    for m in rp.models_for_protocol(2):
        r = rp.reasoning_for(2, m)
        assert isinstance(r, dict) and len(r) == 1
        assert set(r) <= {"effort", "max_tokens", "enabled"}
    assert rp.reasoning_for(1, rp.models_for_protocol(1)[0]) is None
    with pytest.raises(KeyError):
        rp.reasoning_for(2, "not/a-model")


def test_v2_prices_are_per_token_and_positive():
    for m, (pin, pout) in rp.price_table(2).items():
        assert 0 < pin < 1e-4 and 0 < pout < 1e-4, m


def test_the_v2_table_hash_is_in_the_consensus_fingerprint():
    fp = fingerprint.environment_fingerprint()
    expected = hashlib.sha256(rp.PRICES_PATH_V2.read_bytes()).hexdigest()
    assert fp["epoch"]["price_table_v2"] == expected


def test_the_two_shipped_copies_are_identical():
    root = rp.ROOT.parent
    assert (root / "data" / "models_v2.json").read_bytes() == rp.PRICES_PATH_V2.read_bytes()
    json.loads(rp.PRICES_PATH_V2.read_text())
