"""Protocol v4: the epoch summary is public, and a validator that lost its
window rebuilds it from the root's (docs/decisions/LIVE_MENU_V4.md, 4.3).

The root's reveal of epoch k carries epoch k's summary (`menu_summary`,
`menu_inputs.summary_to_dict`). A validator whose window is missing epochs
fills them from the root's reveals, each checked against the root's reveal
commitment for that epoch (the pointer ledger follower sync keeps, or the
commitment read at the chain block where it is the root's for that epoch),
and never in place of a summary of its own. Its `menu_audit` says which
epochs of the window it took from the root.
"""
from __future__ import annotations

import hashlib
import json

from fugal_subnet import menu_inputs, v2_sync
from fugal_subnet import menu_publish as mp
from fugal_subnet import menu_state as ms
from fugal_subnet import reveal_publisher as rp
from fugal_subnet.benchmarks.slicer import epoch_id_for_block


def _summary(epoch: int, pairs: int = 5) -> dict:
    cell = menu_inputs.PairCell("lab-a/model-one@high", 1, "math", False, pairs, 0, 10, 20)
    return menu_inputs.summary_to_dict(menu_inputs.EpochSummary(epoch, (), (), (cell,), ()))


def _reveal(directory, k: int, summary: dict | None) -> str:
    folder = directory / epoch_id_for_block(k)
    folder.mkdir(parents=True)
    doc = {"epoch_id": epoch_id_for_block(k), "protocol_version": 4}
    if summary is not None:
        doc[mp.REVEAL_SUMMARY] = summary
    raw = json.dumps(doc).encode()
    (folder / "reveal.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_the_summary_is_a_root_field_outside_the_compared_view():
    assert mp.REVEAL_SUMMARY == v2_sync.MENU_SUMMARY_FIELD == "menu_summary"
    assert mp.REVEAL_SUMMARY in v2_sync.PER_VALIDATOR_FIELDS
    assert mp.REVEAL_SUMMARY not in v2_sync.comparable_reveal({mp.REVEAL_SUMMARY: {}, "x": 1})


def test_a_root_summary_is_taken_only_from_a_reveal_the_root_committed(tmp_path):
    good = _reveal(tmp_path, 100, _summary(100))
    _reveal(tmp_path, 101, _summary(101))
    _reveal(tmp_path, 102, _summary(999))                     # another epoch's summary
    _reveal(tmp_path, 103, None)                               # no summary in it
    shas = {100: good, 101: "ab" * 32, 102: None, 103: None}
    get = mp.root_summaries(str(tmp_path), lambda k: shas.get(k), check=menu_inputs.summary_from_dict)
    assert get(100) == _summary(100)
    assert get(101) is None                                    # does not match the commitment
    assert get(102) is None and get(103) is None and get(104) is None
    # The root reads its own reveals, which it wrote: no commitment needed.
    own = mp.root_summaries(str(tmp_path), None, check=menu_inputs.summary_from_dict)
    assert own(101) == _summary(101) and own(102) is None


def test_the_reveal_commitment_names_the_sha_of_its_own_epoch_only():
    sha, h = "ab" * 32, "cd" * 32
    data = rp.commitment_for(epoch_id_for_block(100), sha, h)
    assert mp.reveal_sha_in_commitment(data, 100) == sha
    assert mp.reveal_sha_in_commitment(rp.commitment_for(epoch_id_for_block(100), sha), 100) == sha
    assert mp.reveal_sha_in_commitment(data, 101) is None
    assert mp.reveal_sha_in_commitment("garbage", 100) is None


def test_own_summaries_are_kept_and_only_the_missing_epochs_are_filled(tmp_path):
    book = mp.MenuBook(str(tmp_path / "node"))
    book.keep_summary(100, _summary(100, 1))
    filled = book.fill_from_root(range(98, 103), lambda k: _summary(k, 9))
    assert filled == [98, 99, 101, 102]
    assert [d["epoch"] for d in book.summaries(through=102, size=5)] == [98, 99, 100, 101, 102]
    assert book.summaries(through=100, size=1) == [_summary(100, 1)]     # its own, not the root's
    assert book.from_root(through=102, size=5) == [98, 99, 101, 102]
    # Its own summary of 101 arrives (a retried epoch): it replaces the root's.
    book.keep_summary(101, _summary(101, 1))
    assert book.from_root(through=102, size=5) == [98, 99, 102]
    # And all of it survives a restart.
    again = mp.MenuBook(str(tmp_path / "node"))
    assert again.from_root(through=102, size=5) == [98, 99, 102]
    assert again.summaries(through=101, size=1) == [_summary(101, 1)]


def test_the_audit_says_which_epochs_of_its_window_came_from_the_root(tmp_path):
    node = mp.MenuNode(first_epoch=90, book=mp.MenuBook(None), lists=mp.ListBook(None, shipped=False),
                       read_commitment=lambda e: None, fetch_menu=None,
                       step_params=ms.MenuParams(mix=(("math", 1),), min_category_accuracy_bp=0))
    from tests import menu_fixtures as fx
    node.book.hold(fx.state(epoch=101))
    node.keep_inputs(100, fx.inputs(epoch=100), rebuilt=(97, 99))
    read = node.at_boundary(102)
    audit = node.audit(read)
    assert audit["window_from_root"] == [epoch_id_for_block(97), epoch_id_for_block(99)]
    node.keep_inputs(101, fx.inputs(epoch=101))
    node.book.hold(read.state)
    assert "window_from_root" not in node.audit(node.at_boundary(103))
