"""v0.8.1: protocol v5 reads a head's RAW commitment block.

v0.8.0 recorded the v1 seniority key (registration x 10^10 + commit) as a
head's first commitment and turned blocks into epochs by dividing by
EPOCH_INTERVAL (seconds). The key made every head look committed in a far
future epoch, so every question published in an earlier reveal was dropped:
each epoch scored only its own exam and no comparison reached N_MIN (reported
on 2026-11-25). Dividing a raw block by seconds would instead have made every
published question eligible. From config.v5_commit_fix_block the head book
holds raw blocks and a block's epoch is slicer's epoch index."""
from __future__ import annotations

import pytest

from fugal_subnet import config, seat_v5, v5_epoch
from fugal_subnet.benchmarks.slicer import blocks_per_epoch, epoch_index_for_block
from tests.test_v5_epoch import COST, ENTRIES, Book, make_record, stratum

BPE = blocks_per_epoch(v5_epoch.EPOCH_INTERVAL)
START = 26629                       # the first v5 exam epoch on finney
COMMIT = (START - 8) * BPE + 9      # a head committed eight epochs before it
REGISTERED = 9576077                # its hotkey's registration block


def published_in_own_epoch(q: str) -> int | None:
    """Each exam question is first published in its own epoch's reveal (the
    make_record ids are q<epoch * 1000 + i>); coverage questions never."""
    return int(q[1:]) // 1000 if q.startswith("q") else None


def run(book, names_by_epoch, *, raw, heads=None, start=START):
    state, heads, records, res = seat_v5.SeatStateV5(), dict(heads or {}), [], None
    for i, names in enumerate(names_by_epoch):
        e = start + i
        rec, _spec = make_record(e, book, names)
        res = v5_epoch.score_epoch(epoch=e, state=state, heads=heads, record=rec,
                                   window=records, served=book.served(names),
                                   full_entries=ENTRIES, stratum_of=stratum, burn_uid=0,
                                   first_published=published_in_own_epoch,
                                   raw_commit_blocks=raw)
        records.append(rec)
        state, heads = res.decision.state, res.heads
    return res, heads


@pytest.fixture
def needs_three_epochs(monkeypatch):
    # Each synthetic epoch has 60 exam questions: a comparison must add epochs up.
    monkeypatch.setattr(seat_v5, "N_MIN", 150)
    monkeypatch.setattr(seat_v5, "N0", 150)


def test_a_blocks_epoch_is_slicers_epoch_index():
    for b in (0, BPE - 1, BPE, COMMIT, 9_585_720, 9_591_840):
        assert v5_epoch.commit_epoch(b) == epoch_index_for_block(b, BPE)
    assert v5_epoch.commit_epoch(9_585_720) == 9_585_720 // BPE
    # v0.8.0's arithmetic survives only to replay the epochs before the fix
    assert v5_epoch.commit_epoch_v080(COMMIT) == COMMIT // v5_epoch.EPOCH_INTERVAL


def test_the_seat_fills_across_epochs_on_raw_commit_blocks(needs_three_epochs):
    book = Book()
    book.add("a", "ck-a", 1, COMMIT, "m/good")
    res, heads = run(book, [["a"]] * 5, raw=True)
    assert res.decision.holder == "a"
    assert heads["a"].first_commit_block == COMMIT


def test_v080_seniority_keys_never_fill_the_seat(needs_three_epochs):
    # The v0.8.0 report, replayed: the key puts the commit in a far future
    # epoch, every earlier exam drops out, and n never reaches N_MIN.
    book = Book()
    book.add("a", "ck-a", 1, REGISTERED * 10**10 + COMMIT, "m/good")
    res, _heads = run(book, [["a"]] * 5, raw=False)
    assert res.decision.holder is None and res.decision.event == "empty"


def test_a_book_of_v080_keys_is_read_as_raw_blocks_from_the_fix(needs_three_epochs):
    book = Book()
    book.add("a", "ck-a", 1, REGISTERED * 10**10 + COMMIT, "m/good")
    _res, keyed = run(book, [["a"]] * 2, raw=False)
    assert keyed["a"].first_commit_block == REGISTERED * 10**10 + COMMIT
    book.h["a"] = (*book.h["a"][:3], COMMIT, book.h["a"][4])   # served with its raw block
    res, heads = run(book, [["a"]] * 5, raw=True, heads=keyed, start=START + 2)
    assert heads["a"].first_commit_block == COMMIT
    assert res.decision.holder == "a"


def test_questions_published_before_the_commit_still_do_not_count(needs_three_epochs):
    # The memorisation rule holds on raw blocks: a head committed after the
    # exams it would be judged on has no items from them.
    book = Book()
    book.add("late", "ck-l", 1, (START + 10) * BPE + 1, "m/good")
    res, _heads = run(book, [["late"]] * 5, raw=True)
    assert res.decision.holder is None


def test_the_book_owner_is_the_earliest_raw_commitment():
    served = [v5_epoch.Served("h", 2, "hk-x", "ck-x", COMMIT + 50),
              v5_epoch.Served("h", 1, "hk-o", "ck-o", COMMIT)]
    book = v5_epoch.observe({}, served, START)
    assert book["h"].coldkey == "ck-o" and book["h"].first_commit_block == COMMIT
    keyed = {"h": v5_epoch.HeadFacts("h", "ck-o", "hk-o", REGISTERED * 10**10 + COMMIT, START)}
    assert v5_epoch.raw_commit_book(keyed)["h"].first_commit_block == COMMIT
    assert v5_epoch.raw_commit_book(book) == book


def test_finney_fixes_at_a_boundary_after_the_v5_block():
    fix = config.FINNEY_V5_COMMIT_FIX_BLOCK
    assert fix is not None and fix % 360 == 0 and fix > config.FINNEY_V5_ACTIVATION_BLOCK
    assert config.v5_commit_fix_block(config.FINNEY_GENESIS_HASH) == fix
    assert COST and ENTRIES
