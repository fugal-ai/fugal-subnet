"""The v5 consensus section (fugal_subnet/consensus_v5.py, skeleton)."""
from __future__ import annotations

import json

from fugal_subnet import consensus_v5


def test_section_is_canonical_json_and_records_every_registered_name():
    s = consensus_v5.section()
    assert set(s) == {"activation", "constants", "files"}
    json.dumps(s, sort_keys=True)
    names = {f"{m}:{n}" for m, ns in consensus_v5.REGISTERED.items() for n in ns}
    assert set(s["constants"]) == names
    assert set(s["files"]) == set(consensus_v5.PINNED_FILES)


def test_seat_constants_are_readable():
    c = consensus_v5.constants()
    for name in consensus_v5.REGISTERED["fugal_subnet.seat_v5"]:
        assert not str(c[f"fugal_subnet.seat_v5:{name}"]).startswith("unavailable")


def test_problems_lists_what_is_not_built_yet():
    # Until the config block (lane V1) and the pinned files (lane P) land,
    # problems() names them; it never raises.
    assert isinstance(consensus_v5.problems(), list)
