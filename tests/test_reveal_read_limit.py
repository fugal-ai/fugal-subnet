"""One reveal size limit for every reader of a root's reveal
(reveal_sync.MAX_REVEAL_BYTES): a reveal just under it is fetched and read
by each of them (the menu step's summaries, the join step's served set, the
miner's holder read), and one byte over it by none, so a large field's
reveal never leaves one kind of node without what the others read."""
from __future__ import annotations

import hashlib
import io
import json

import pytest

from fugal_subnet import chain_proposals as cp
from fugal_subnet import menu_inputs as mi
from fugal_subnet import menu_publish as mp
from fugal_subnet import reveal_sync
from fugal_subnet.benchmarks.slicer import epoch_id_for_block
from fugal_subnet.tee import td_v4

LIMIT = reveal_sync.MAX_REVEAL_BYTES
EPOCH = 104


def test_every_reader_names_the_one_limit():
    assert mp.MAX_REVEAL_READ_BYTES == td_v4.MAX_HOLDER_REVEAL_BYTES == LIMIT == 96 * 2**20


def _reveal(size: int) -> bytes:
    """A root reveal of exactly `size` bytes: a menu summary, the served
    hotkeys and padding."""
    eid = epoch_id_for_block(EPOCH)
    doc = {"epoch_id": eid, "protocol_version": 4,
           mp.REVEAL_SUMMARY: mi.summary_to_dict(mi.EpochSummary(
               epoch=EPOCH, votes=(), calls=(), pairs=(), use=())),
           cp.SERVED_FIELD: ["5Fminer-a", "5Fminer-b"], "zz": ""}
    body = json.dumps(doc).encode()
    doc["zz"] = "x" * (size - len(body))
    raw = json.dumps(doc).encode()
    assert len(raw) == size
    return raw


@pytest.fixture(scope="module")
def under():
    return _reveal(LIMIT - 1)


def _read_all(raw: bytes, tmp_path):
    from neurons import validator

    eid = epoch_id_for_block(EPOCH)
    (tmp_path / eid).mkdir(exist_ok=True)
    path = tmp_path / eid / "reveal.json"
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    summary = mp.root_summaries(str(tmp_path), None, check=mi.summary_from_dict)(EPOCH)
    served = validator._v4_served(str(path), sha)

    def opener(req, timeout=None):
        return io.BytesIO(raw)
    fetched = reveal_sync.fetch_reveal_bytes(eid, sha, bucket="fugal-test-reveals", netuid=7,
                                             opener=opener)
    try:
        td_v4.seat_holder_from_reveal(raw, eid, sha)
        holder = True
    except ValueError:
        holder = False
    path.unlink()
    return summary, served, fetched, holder


def test_a_reveal_just_under_the_limit_is_read_by_every_reader(under, tmp_path):
    summary, served, fetched, holder = _read_all(under, tmp_path)
    assert summary is not None and summary["epoch"] == EPOCH
    assert served == ("5Fminer-a", "5Fminer-b")
    assert fetched == under
    assert holder


def test_a_reveal_over_the_limit_is_read_by_none(tmp_path):
    over = _reveal(LIMIT + 1)
    summary, served, fetched, holder = _read_all(over, tmp_path)
    assert (summary, served, fetched, holder) == (None, None, None, False)
