"""Protocol v5: the memorisation rule (v5_epoch.eligible_after, the fresh-only
exam draw), menu entries that leave between a spec and its exam, the kept head
bytes, and a stale node's rebuild from the root's reveals (v5_node.resync)."""
from __future__ import annotations

import hashlib

import pytest

from fugal_subnet import exam_v5, memo_v4, seat_v5, v5_epoch, v5_node
from fugal_subnet.exam_v5 import Vote
from tests.test_v5_epoch import COST, ENTRIES, Book, make_record, run, stratum


@pytest.fixture(autouse=True)
def small_bounds(monkeypatch):
    monkeypatch.setattr(seat_v5, "N_MIN", 60)
    monkeypatch.setattr(seat_v5, "N0", 60)
    monkeypatch.setattr(v5_epoch, "BASELINE_MIN_PAIRS", 5)


# --- the memorisation rule ---------------------------------------------------------------


def test_eligible_after_counts_only_what_no_compared_head_could_have_read():
    pub = {"seen": 9, "same": 10, "later": 11}
    ok = v5_epoch.eligible_after(pub.get, 10)
    # A reveal of epoch f is public from f + 1: a head committed in epoch 10
    # may have read epoch 9's, not epoch 10's.
    assert not ok("seen")
    assert ok("same") and ok("later") and ok("never")


def test_commit_epoch_is_the_blocks_epoch_index():
    n = v5_epoch.BLOCKS_PER_EPOCH          # blocks, not EPOCH_INTERVAL's seconds (v0.8.1)
    assert v5_epoch.commit_epoch(0) == 0
    assert v5_epoch.commit_epoch(5 * n - 1) == 4 and v5_epoch.commit_epoch(5 * n) == 5


def test_published_questions_give_no_items_and_the_seat_does_not_fill_on_them():
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    fresh = run(book, [["a"]] * 4)[0]
    assert fresh.decision.holder == "a"
    # Every question published before the head's commit epoch (0): nothing
    # counts, so the head cannot fill the seat on labels it could have read.
    seen = run(book, [["a"]] * 4, first_published=lambda _q: -1)[0]
    assert seen.decision.holder is None and seen.decision.event == "empty"
    # Published in the head's own commit epoch or later: it could not have.
    later = run(book, [["a"]] * 4, first_published=lambda _q: 0)[0]
    assert later.decision.holder == fresh.decision.holder


def test_a_comparison_reads_the_later_heads_commit():
    book = Book()
    n = v5_epoch.BLOCKS_PER_EPOCH
    book.add("old", "ck-o", 1, 10, "m/good")
    book.add("new", "ck-n", 2, 50 * n, "m/cheap")
    _res, _s, heads, records, _r = run(book, [["old", "new"]] * 3)
    specs = [r.spec for r in records]
    votes = [v for r in records for v in r.counted_votes()]
    first = {q: 30 for s in specs for q in s.questions}       # between the two commits
    items, base = v5_epoch.item_sources(records, heads, stratum, first.get)
    assert sum(len(v) for v in items("old", "new").values()) == 0
    assert sum(len(v) for v in base("old").values()) > 0         # old alone could not have
    assert sum(len(v) for v in base("new").values()) == 0
    everything = exam_v5.comparison_items(specs, votes, "old", "new",
                                          frozenset({"ck-o", "ck-n"}), {"old": 1, "new": 1},
                                          stratum)
    assert sum(len(v) for v in everything.values()) > 0


def test_the_exam_draws_only_the_unpublished_reserve_and_no_question_twice():
    by = {"d1": ["a", "b", "c"], "d2": ["x", "y"], "d3": ["p"]}
    first = {"a": 5, "x": 7}
    got = v5_epoch.reserve(by, first.get, used={"b"})
    assert got == {"d1": ["c"], "d2": ["y"], "d3": ["p"]}
    assert v5_epoch.reserve({"d": ["q"]}, {"q": 1}.get) == {}
    r = v5_epoch.FRESH_RESERVE_EPOCHS
    assert v5_epoch.exam_size(0) == 0 and v5_epoch.exam_size(1) == 1
    assert v5_epoch.exam_size(r) == 1 and v5_epoch.exam_size(r + 1) == 2
    assert v5_epoch.exam_size(10**9) == exam_v5.EXAM_QUESTIONS
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    _r1, s1 = make_record(100, book, ["a"])
    _r2, s2 = make_record(101, book, ["a"])
    assert v5_epoch.spec_questions([s1, None, s2]) == set(s1.questions) | set(s2.questions)


def test_v5_votes_are_published_outcomes():
    rec, _spec = make_record(100, Book(), [])
    doc = {"votes_v5": rec.to_reveal()}
    ids = memo_v4.published_ids(doc)
    assert ids == {v.qid for v in rec.votes} | {v.qid for v in rec.coverage}
    with pytest.raises(ValueError):
        memo_v4.published_ids({"votes_v5": {"votes": [[None]]}})
    assert memo_v4.published_ids({}) == set()


# --- an entry that leaves the menu between the spec and its exam --------------------------


def test_a_cell_on_an_entry_off_the_menu_gives_no_item_and_counts_as_no_failure():
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    book.add("b", "ck-b", 2, 20, "m/cheap")
    rec, spec = make_record(100, book, ["a", "b"])
    # The TDs do not call a cell whose entry left the epoch's menu and the
    # validator expects no such call (tee/verify.expected_v5): no vote at all.
    left = [v for v in rec.votes if v.entry != "m/cheap"]
    ex = frozenset({"ck-a", "ck-b"})
    full = exam_v5.comparison_items([spec], rec.votes, "a", "b", ex, {"a": 1, "b": 1}, stratum)
    gone = exam_v5.comparison_items([spec], left, "a", "b", ex, {"a": 1, "b": 1}, stratum)
    assert sum(len(v) for v in full.values()) == len(spec.questions)
    assert sum(len(v) for v in gone.values()) == 0
    # Failed votes from two coldkeys would have been a wrong answer: an absent
    # entry is not that.
    failed = left + [Vote(100, q, e, hks[0], dict(book.ring(["a", "b"]))[hks[0]], 0, 0, True)
                     for (q, e), hks in spec.assignment.items() if e == "m/cheap"]
    one = exam_v5.comparison_items([spec], failed, "a", "b", ex, {"a": 1, "b": 1}, stratum)
    assert sum(len(v) for v in one.values()) == 0


# --- kept head bytes ---------------------------------------------------------------------


def test_head_bytes_survive_a_restart_and_leave_with_the_head(tmp_path):
    path = str(tmp_path / "state.v5")
    data = b"head-bytes"
    h = hashlib.sha256(data).hexdigest()
    other = hashlib.sha256(b"other").hexdigest()
    node = v5_node.V5Node(path, root=True)
    node.keep_head(h, data)
    node.keep_head(other, b"other")
    with pytest.raises(ValueError):
        node.keep_head("0" * 64, data)
    node.heads = {h: v5_epoch.HeadFacts(h, "ck", "hk", 10, 100)}
    node.save()
    again = v5_node.V5Node(path, root=True)
    assert again.head_bytes == {h: data}
    # A tampered file is ignored.
    with open(tmp_path / "state.v5.heads" / f"{h}.npz", "wb") as f:
        f.write(b"tampered")
    assert v5_node.V5Node(path, root=True).head_bytes == {}
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    res, _st, _hd, records, _rv = run(book, [["a"]])
    again.commit(100, res, records[-1], None, {})
    assert not (tmp_path / "state.v5.heads" / f"{other}.npz").exists()


# --- resync ------------------------------------------------------------------------------


def test_a_fresh_follower_converges_to_the_roots_state(tmp_path):
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    book.add("b", "ck-b", 2, 20, "m/cheap")
    res, state, heads, records, reveals = run(book, [["a", "b"]] * 5)
    last = 104
    # The specs ahead (epochs 105 and 106) as the root's reveals of 103 and 104 publish them.
    _r, s105 = make_record(105, book, ["a", "b"])
    _r, s106 = make_record(106, book, ["a", "b"])
    reveals[103] = {**reveals[103], **v5_epoch.spec_reveal(s105, COST)}
    reveals[104] = {**reveals[104], **v5_epoch.spec_reveal(s106, COST)}
    node = v5_node.V5Node(str(tmp_path / "f.v5"), root=False)
    assert node.stale(last + 1, first_epoch=100)
    assert node.resync(reveals, last + 1)
    assert node.seat == state and node.heads == heads
    assert ([r.to_reveal() for r in node.records]
            == [r.to_reveal() for r in v5_epoch.window_of(records, last)])
    assert node.specs[105][0] == s105 and node.specs[106][0] == s106
    assert not node.stale(last + 1, first_epoch=100)
    # It persisted: a restart holds the same state, and scores the next epoch
    # exactly as the root does.
    again = v5_node.V5Node(str(tmp_path / "f.v5"), root=False)
    assert again.seat == state and again.through == last
    rec, _spec = make_record(105, book, ["a", "b"])
    mine = v5_epoch.score_epoch(epoch=105, state=again.seat, heads=again.heads, record=rec,
                                window=again.records, served=book.served(["a", "b"]),
                                full_entries=ENTRIES, stratum_of=stratum, burn_uid=0,
                                first_published=lambda _q: None)
    root = v5_epoch.score_epoch(epoch=105, state=state, heads=heads, record=rec,
                                window=records, served=book.served(["a", "b"]),
                                full_entries=ENTRIES, stratum_of=stratum, burn_uid=0,
                                first_published=lambda _q: None)
    assert mine.reveal == root.reveal and mine.weights == root.weights


def test_resync_needs_the_reveal_before_the_epoch(tmp_path):
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    *_x, reveals = run(book, [["a"]] * 3)
    node = v5_node.V5Node(None, root=False)
    assert not node.resync({k: v for k, v in reveals.items() if k != 102}, 103)
    assert node.through is None and node.stale(103, first_epoch=100)
    assert not node.stale(100, first_epoch=100)


def test_reference_reveals_reads_the_window_and_skips_what_is_not_held():
    import json

    held = {e: json.dumps({"epoch": e}).encode() for e in range(40, 60)}
    held[55] = b"not json"
    got = v5_node.reference_reveals(held.get, 60)
    lo = 60 - v5_epoch.WINDOW_EPOCHS - v5_epoch.SPEC_LAG
    assert set(got) == {e for e in range(max(lo, 40), 60) if e != 55}
