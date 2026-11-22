"""Protocol v4: a successor's odds before its commit are its predecessors'
(labels_v4.predecessor_odds), through the shipped loops (tests/v4_world.py
SeatLoop: `validator.main` for a root and a follower given every input in
reverse order, and the miners' real `_run_epoch`).

The holder A's hotkey commits a successor of its head every epoch from
SWITCH (its rows byte for byte, new bytes), and so does one other hotkey,
with the late-odds bound at 2 head-epochs a reveal, so the two heads share
it as 32 such heads share the design's 32. B, sharper, is committed at
ARRIVES, after the successions have begun, and challenges from then. B
takes the seat at the epoch it takes it from a holder that keeps its head
in the same loops (both runs here; under the fast crown's floors that is a
few epochs after it arrives). Root and follower agree every epoch,
a rebuild from the root's reveals reaches its seat and window, and a replay
of its epochs across the successions from its state at SWITCH reaches the
state it holds.
"""
from __future__ import annotations

import hashlib

import pytest

from tests import v4_world as w
from tests.test_menu_publish import LIST_BYTES, signer  # noqa: F401 - the fixture
from tests.test_v4_replay_seat import _digest, _replayer, _root_at, _root_saved

SWITCH = w.SEAT_FIRST + 6
ARRIVES = SWITCH + 2
LAST = SWITCH + 9


@pytest.fixture
def _signature(signer):  # noqa: F811
    return signer.sign(LIST_BYTES)


def _loop(tmp_path, monkeypatch, signature, *, recommit: bool = True):
    from fugal_subnet import config

    monkeypatch.setattr(config, "LATE_ODDS_SLOTS_V4", 2)
    spec: list = []
    loop = w.SeatLoop(tmp_path, monkeypatch, spec, last=LAST, plan=lambda e, lp: None,
                      signature=signature)
    rows_a, rows_b = loop.all_rows(w.SHARP), loop.all_rows(w.SHARPER)
    other_row = {loop.base.ids()[1]: 0.0}
    spec += [
        (w.SEAT_A, "5Fcold-a", loop.head(rows_a, "a"), "on", 1),
        (w.SEAT_L, "5Fcold-l", loop.head({loop.base.ids()[0]: 0.0}, "l"), "off",
         w.SEAT_V4_EPOCH * w.BPE + 9),
        (w.SEAT_C, "5Fcold-c", loop.head(other_row, "c-0"), "off", w.SEAT_V4_EPOCH * w.BPE + 11),
        (w.SEAT_B, "5Fcold-b", loop.head(rows_b, "b-pre"), "on", w.SEAT_V4_EPOCH * w.BPE + 3),
    ]
    b_head = loop.head(rows_b, "b")

    def plan(e, lp):
        if e == ARRIVES:
            lp.set_head(w.SEAT_B, b_head)
        if e >= SWITCH and recommit:
            lp.set_head(w.SEAT_A, lp.head(rows_a, f"a-{e}"))
            lp.set_head(w.SEAT_C, lp.head(other_row, f"c-{e}"))
        lp.down = set() if e >= ARRIVES else {w.SEAT_B}

    loop.plan = plan
    return loop, hashlib.sha256(b_head).hexdigest()


def test_a_challenger_takes_from_a_holder_re_committed_each_epoch_and_every_node_agrees(
        tmp_path, monkeypatch, _signature):
    from fugal_subnet import seat_v4, v2_epoch

    real = v2_epoch.run_v2_epoch

    def took_at(loop, b):
        return next((k for k in loop.epochs if loop.holder(k) == b), None)

    # The take against a holder that keeps its head, in the same loops.
    (tmp_path / "static").mkdir()
    (tmp_path / "recommit").mkdir()
    static, b0 = _loop(tmp_path / "static", monkeypatch, _signature, recommit=False)
    static.run()
    monkeypatch.setattr(v2_epoch, "run_v2_epoch", real)
    takes = took_at(static, b0)
    assert takes is not None and ARRIVES < takes <= LAST, takes
    loop, b = _loop(tmp_path / "recommit", monkeypatch, _signature)
    loop.run()
    monkeypatch.setattr(v2_epoch, "run_v2_epoch", real)
    loop.check_agreement_and_rebuild()
    took = took_at(loop, b)
    labels = {k: loop.reveal(k)[seat_v4.REVEAL_KEY]["heads"][b]["vs_bar"]["n"]
              for k in loop.epochs
              if (loop.reveal(k)[seat_v4.REVEAL_KEY]["heads"].get(b) or {}).get("vs_bar")}
    assert took == takes, (took, takes, labels)
    # Every epoch from SWITCH the holder is a successor, read through its
    # predecessors.
    for k in range(SWITCH, takes):
        hs = seat_v4.from_dict(loop.reveal(k)["v2_state"]["seat"]).head(
            loop.reveal(k)["champion"]["champion"])
        assert len(hs.predecessors) == min(k - SWITCH + 1, seat_v4.MAX_PREDECESSORS), k
    # The root's epochs across the successions, replayed from its state at
    # SWITCH (rebuilt from its reveals), end in the state it holds.
    got = _replayer(loop, monkeypatch, w.held_memo_for(loop.dirs["root"], loop.pool))(
        _root_at(loop, SWITCH), SWITCH, LAST + 1)
    assert got.epochs == tuple(range(SWITCH + 1, LAST + 1))
    assert _digest(got.window, got.seat, got.copy_state) == _digest(*_root_saved(loop))
