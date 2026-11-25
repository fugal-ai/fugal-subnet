"""Protocol v5's epoch (fugal_subnet/v5_epoch.py), end to end on synthetic
epochs: specs built by exam_v5, votes cast by the assigned hotkeys, the seat
decided by seat_v5, the weights, the reveal, and the rebuild from reveals."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from fugal_subnet import exam_v5, seat_v5, v5_epoch
from fugal_subnet.exam_v5 import Vote

ENTRIES = ("m/base", "m/cheap", "m/good", "m/best")
ACC = {"m/base": 50, "m/cheap": 60, "m/good": 90, "m/best": 99}       # percent
COST = {"m/base": 1500, "m/cheap": 500, "m/good": 2000, "m/best": 2000}  # nanodollars
VOTERS = [("hk-v1", "ck-v1"), ("hk-v2", "ck-v2"), ("hk-v3", "ck-v3"), ("hk-v4", "ck-v4")]
BASE = "m/base"


def NOTHING(_q):
    """No question published before the exam (every question eligible)."""
    return None


@pytest.fixture(autouse=True)
def small_bounds(monkeypatch):
    monkeypatch.setattr(seat_v5, "N_MIN", 60)
    monkeypatch.setattr(seat_v5, "N0", 60)
    monkeypatch.setattr(v5_epoch, "BASELINE_MIN_PAIRS", 5)


def right(q: str, entry: str) -> bool:
    h = int.from_bytes(hashlib.sha256(f"{q}|{entry}".encode()).digest()[:2], "big") % 100
    return h < ACC[entry]


def stratum(q: str) -> str:
    return "s" + str(int(q[1:]) % 2)


class Book:
    """Heads (name -> (coldkey, hotkey, uid, commit block, entry it always picks))."""

    def __init__(self):
        self.h = {}

    def add(self, name, ck, uid, block, entry):
        self.h[name] = (ck, f"hk-{name}", uid, block, entry)
        return name

    def served(self, names):
        return [v5_epoch.Served(n, self.h[n][2], self.h[n][1], self.h[n][0], self.h[n][3])
                for n in names]

    def ring(self, names):
        return VOTERS + [(self.h[n][1], self.h[n][0]) for n in names]


def make_record(epoch, book, names, *, n_q=60, void=None, liar=None):
    pool = [f"q{epoch * 1000 + i}" for i in range(n_q)]
    picks = {n: [book.h[n][4]] * n_q for n in names}
    spec = exam_v5.build_spec(epoch, b"nonce" + bytes([epoch % 256]), picks, ENTRIES, pool,
                              book.ring(names), BASE)
    hk_ck = dict(book.ring(names))
    votes = []
    for (q, e), hks in spec.assignment.items():
        for hk in hks:
            y = 10000 if right(q, e) else 0
            if liar is not None and hk_ck[hk] == liar:
                y = 10000 - y
            votes.append(Vote(epoch, q, e, hk, hk_ck[hk], y, COST[e], False))
    cover = []
    for i, e in enumerate(ENTRIES):
        for j in range(8):
            q = f"c{epoch}-{i}-{j}"
            for hk, ck in VOTERS[:2]:
                cover.append(Vote(epoch, q, e, hk, ck, 10000 if right(q, e) else 0, COST[e],
                                  False))
    rec = v5_epoch.EpochRecord(epoch=epoch, spec=None if void else spec, costs_nd=dict(COST),
                               votes=tuple(votes), coverage=tuple(cover),
                               flagged=frozenset(v5_epoch.implausible(votes)), void=void)
    return rec, spec


def run(book, names_by_epoch, start=100, state=None, heads=None, records=(),
        first_published=NOTHING, **kw):
    state = state or seat_v5.SeatStateV5()
    heads = dict(heads or {})
    records = list(records)
    reveals = {}
    res = None
    for i, names in enumerate(names_by_epoch):
        e = start + i
        rec, spec = make_record(e, book, names, **kw)
        res = v5_epoch.score_epoch(epoch=e, state=state, heads=heads, record=rec,
                                   window=records, served=book.served(names),
                                   full_entries=ENTRIES, stratum_of=stratum, burn_uid=0,
                                   first_published=first_published)
        records.append(rec)
        state, heads = res.decision.state, res.heads
        reveals[e] = dict(res.reveal)
        reveals[e - v5_epoch.SPEC_LAG] = {**reveals.get(e - v5_epoch.SPEC_LAG, {}),
                                          **v5_epoch.spec_reveal(spec, COST)}
    return res, state, heads, records, reveals


def test_day_one_no_evidence_everything_burns():
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    rec = v5_epoch.EpochRecord(100, None, {}, (), ())
    res = v5_epoch.score_epoch(epoch=100, state=seat_v5.SeatStateV5(), heads={}, record=rec,
                               window=[], served=book.served(["a"]), full_entries=ENTRIES,
                               stratum_of=stratum, burn_uid=0, first_published=NOTHING)
    assert res.decision.event == "empty" and res.decision.holder is None
    assert (res.uids, res.weights) == ([0], [1.0])
    assert res.baseline == ""


def test_the_seat_fills_once_a_head_beats_the_best_single_model():
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    book.add("b", "ck-b", 2, 20, "m/cheap")
    res, state, *_ = run(book, [["a", "b"]] * 3)
    assert res.baseline in ENTRIES
    assert res.decision.event in ("fill", "hold") and res.decision.holder == "a"
    assert res.decision.payable == ("a",)
    # empty history seats go to the holder; half the miner emission burns (uid 0)
    assert dict(zip(res.uids, res.weights)) == {0: pytest.approx(0.5), 1: pytest.approx(0.5)}


def test_a_head_worse_than_the_single_model_never_fills():
    book = Book()
    book.add("b", "ck-b", 2, 20, "m/base")
    res, *_ = run(book, [["b"]] * 3)
    assert res.decision.holder is None and res.weights == [1.0] and res.uids == [0]


def test_a_better_challenger_takes_the_seat():
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    _, state, heads, records, _ = run(book, [["a"]] * 2)
    assert state.holder == "a"
    book.add("c", "ck-c", 3, 30, "m/best")
    res, state, *_ = run(book, [["a", "c"]] * 3, start=102, state=state, heads=heads,
                         records=records)
    assert state.holder == "c" and "a" in state.history


def test_the_holders_own_successor_upgrades_when_not_worse():
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    _, state, heads, records, _ = run(book, [["a"]] * 2)
    book.add("a2", "ck-a", 4, 40, "m/best")
    res, state, *_ = run(book, [["a", "a2"]] * 3, start=102, state=state, heads=heads,
                         records=records)
    assert state.holder == "a2" and "a" not in state.history


def test_a_void_exam_gives_no_items():
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    res, *_ = run(book, [["a"]] * 3, void="spec mismatch")
    assert res.decision.holder is None


def test_spec_agreement():
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    _, s1 = make_record(100, book, ["a"])
    _, s2 = make_record(101, book, ["a"])
    assert v5_epoch.spec_agrees(s1, s1) is None
    assert "not the published" in v5_epoch.spec_agrees(s1, s2)
    assert v5_epoch.spec_agrees(s1, None) and v5_epoch.spec_agrees(None, s1)


def test_a_lying_operator_is_flagged_and_its_votes_dropped():
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    rec, _ = make_record(100, book, ["a"], n_q=80, liar="ck-v1")
    assert "ck-v1" in rec.flagged and all(v.coldkey != "ck-v1" for v in rec.counted_votes())


def test_rebuild_from_reveals_gives_the_same_state():
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    book.add("b", "ck-b", 2, 20, "m/cheap")
    res, state, heads, records, reveals = run(book, [["a", "b"]] * 4)
    seat, book2, recs = v5_epoch.rebuild(reveals, 103)
    assert seat == state and book2 == heads
    assert [r.to_reveal() for r in recs] == [r.to_reveal() for r in v5_epoch.window_of(records, 103)]
    again = v5_epoch.score_epoch(epoch=103, state=seat_v5.SeatStateV5(), heads={},
                                 record=recs[-1], window=recs[:-1], served=book.served(["a", "b"]),
                                 full_entries=ENTRIES, stratum_of=stratum, burn_uid=0,
                                   first_published=NOTHING)
    assert again.decision.holder == res.decision.holder


def test_the_record_round_trips_through_its_reveal_and_checks_the_spec():
    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    rec, spec = make_record(100, book, ["a"])
    back = v5_epoch.EpochRecord.from_reveal(rec.to_reveal(), spec, COST)
    assert back.to_reveal() == rec.to_reveal()
    _, other = make_record(101, book, ["a"])
    with pytest.raises(ValueError):
        v5_epoch.EpochRecord.from_reveal(rec.to_reveal(), other, COST)
    s, c = v5_epoch.spec_from_reveal(v5_epoch.spec_reveal(spec, COST))
    assert s == spec and c == COST


def test_a_copy_served_by_another_coldkey_is_not_live_for_the_owner():
    heads = v5_epoch.observe({}, [v5_epoch.Served("h", 1, "hk-o", "ck-o", 10)], 100)
    served = [v5_epoch.Served("h", 2, "hk-x", "ck-x", 50)]
    book = v5_epoch.observe(heads, served, 101)
    assert book["h"].coldkey == "ck-o" and book["h"].last_seen == 100
    (c,) = v5_epoch.contenders(book, served, 101)
    assert not c.live and c.standing
    assert v5_epoch.serving_uids(book, served) == {}


def test_votes_from_proof_splits_exam_and_coverage_and_reads_billed_cost():
    r = lambda **k: SimpleNamespace(**{"question_id": "q1", "routed_model": "m/good", "y": 10000,
                                       "failed": False, "provider_cost": 1234, "cost_usd": 9.0,
                                       **k})
    proof = SimpleNamespace(hotkey="hk", results=[
        r(exam=True), r(exam=False, provider_cost=None, cost_usd=0.5),
        r(exam=True, failed=True, y=0), r(exam=None)])
    exam, cover = v5_epoch.votes_from_proof(7, proof, "ck")
    assert [v.cost_nd for v in exam] == [1234, 1234] and exam[1].failed
    assert cover[0].cost_nd == 500_000_000 and len(cover) == 1


def test_the_node_keeps_its_state_and_judges_the_published_spec(tmp_path):
    from fugal_subnet import v5_node

    book = Book()
    book.add("a", "ck-a", 1, 10, "m/good")
    res, state, heads, records, _ = run(book, [["a"]] * 2)
    _, spec = make_record(102, book, ["a"])
    path = str(tmp_path / "state.v5")
    node = v5_node.V5Node(path, root=True)
    node.commit(101, res, records[-1], spec, COST)
    again = v5_node.V5Node(path, root=True)
    assert again.seat == state and again.heads == heads
    assert [r.to_reveal() for r in again.records] == [records[-1].to_reveal()]
    assert again.exam_for(102) == (spec, COST, None)
    assert again.exam_for(103)[2] == "no exam spec for this epoch"
    _, other = make_record(103, book, ["a"])
    follower = v5_node.V5Node(path, root=False, read_published=lambda e: (other, COST))
    assert "not the published" in follower.exam_for(102)[2]
    same = v5_node.V5Node(path, root=False, read_published=lambda e: (spec, COST))
    assert same.exam_for(102) == (spec, COST, None)
    broken = v5_node.V5Node(path, root=False, read_published=lambda e: 1 / 0)
    assert broken.exam_for(102)[0] is None and broken.exam_for(102)[2]
