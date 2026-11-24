#!/usr/bin/env python3
"""Protocol v5 flip rehearsal, in process: the shipped v5 epoch
(fugal_subnet/v5_epoch.py, seat_v5, exam_v5) from the v5 block onward, with a
synthetic world for the models and the operators. No Docker, no chain, no
network, no paid call:

    python scripts/rehearsal_v5.py          # prints a table, exit 1 on a FAIL

Timeline (A the first v5 epoch; the shipped constants unless named below):

    A-1       protocol v4 (slicer.protocol_version_for_boundary says 4).
    A, A+1    protocol v5, coverage only: no spec has been published for
              them (SPEC_LAG_V5 = 2), so the seat is empty and every share
              burns ("day one").
    A+2..     the first exams: each epoch the root builds the spec of
              E + SPEC_LAG from the heads served by now, the assigned
              operators label its cells (EXAM_VOTES_V5 distinct coldkeys),
              and the seat is decided on the window.
    fill      the seat fills only once a head's bound over the best single
              entry (B*) passes. B* is labelled on EXAM_BASELINE_QUESTIONS_V5
              (100) questions an epoch, and the bound needs N_MIN_V5 (2,000)
              of them: no seat fills before about A+21 (~25 h after the
              block). A head that is no better than B* never fills it.
    take      a later head that beats the holder by TAKE_MARGIN_V5 takes it.
    follower  a second validator that rebuilds its state from the root's
              reveals alone (v5_epoch.rebuild) decides the same seat and
              recomputes the same spec every epoch.

The world: four menu entries with fixed accuracies and costs; heads that
route per question with a fixed rule. SHORTENED FOR THE REHEARSAL, and only here:
EXAM_QUESTIONS (the candidates per epoch) is 300, so the run stays under a
few minutes; the seat's N_MIN / N0 / margins are the shipped ones.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fugal_subnet import exam_v5, seat_v5, v5_epoch  # noqa: E402
from fugal_subnet.benchmarks import slicer  # noqa: E402
from fugal_subnet.exam_v5 import Vote  # noqa: E402

ENTRIES = ("m/base", "m/cheap", "m/good", "m/best")
ACC = {"m/base": 70, "m/cheap": 62, "m/good": 88, "m/best": 96}         # percent
COST = {"m/base": 600_000, "m/cheap": 100_000, "m/good": 1_500_000,   # nanodollars a call
        "m/best": 15_000_000}
VOTERS = [(f"hk-v{i}", f"ck-v{i}") for i in range(1, 7)]
QUESTIONS = 300
A = 30_000                  # first v5 epoch (synthetic)
TAKE_AT = 26                # head c is first served at A + TAKE_AT
EPOCHS = 56


def right(q: str, entry: str) -> bool:
    h = int.from_bytes(hashlib.sha256(f"{q}|{entry}".encode()).digest()[:2], "big") % 100
    return h < ACC[entry]


def stratum(q: str) -> str:
    return "s" + str(int(hashlib.sha256(q.encode()).digest()[0]) % 3)


class World:
    def __init__(self):
        self.heads = {}          # name -> (coldkey, hotkey, uid, commit block, pick(q) -> entry)

    def add(self, name, ck, uid, block, entry):
        self.heads[name] = (ck, f"hk-{name}", uid, block, entry)

    def served(self, names):
        return [v5_epoch.Served(n, self.heads[n][2], self.heads[n][1], self.heads[n][0],
                                self.heads[n][3]) for n in names]

    def ring(self, names):
        return VOTERS + [(self.heads[n][1], self.heads[n][0]) for n in names]


def coverage(epoch: int) -> tuple[Vote, ...]:
    out = []
    for i, e in enumerate(ENTRIES):
        for j in range(60):
            q = f"c{epoch}-{i}-{j}"
            for hk, ck in VOTERS[:2]:
                out.append(Vote(epoch, q, e, hk, ck, 10000 if right(q, e) else 0, COST[e], False))
    return tuple(out)


def spec_for(epoch: int, world: World, names, baseline: str):
    pool = [f"q{epoch}-{i}" for i in range(QUESTIONS)]
    picks = {n: [world.heads[n][4](q) for q in pool] for n in names}
    return exam_v5.build_spec(epoch, hashlib.sha256(f"nonce{epoch}".encode()).digest(), picks,
                              ENTRIES, pool, world.ring(names), baseline)


def record(epoch: int, spec, world: World, names) -> v5_epoch.EpochRecord:
    if spec is None:
        return v5_epoch.EpochRecord(epoch, None, {}, (), coverage(epoch))
    hk_ck = dict(world.ring(names))
    votes = [Vote(epoch, q, e, hk, hk_ck[hk], 10000 if right(q, e) else 0, COST[e], False)
             for (q, e), hks in spec.assignment.items() for hk in hks]
    return v5_epoch.EpochRecord(epoch, spec, dict(COST), tuple(votes), coverage(epoch),
                                flagged=frozenset(v5_epoch.implausible(votes)))


def main() -> int:
    t0 = time.time()
    rows, fails = [], []

    def check(name, ok, detail=""):
        rows.append(("PASS" if ok else "FAIL", name, detail))
        if not ok:
            fails.append(name)

    bpe = 360
    v2_block, v3_block, v4_block, v5_block = 7_000 * bpe, 8_000 * bpe, 9_000 * bpe, A * bpe
    pv = [slicer.protocol_version_for_boundary(b, v2_block, v3_block,
                                               v4_activation_block=v4_block,
                                               v5_activation_block=v5_block)
          for b in (v5_block - bpe, v5_block)]
    check("flip: v4 before the block, v5 from it", pv == [4, 5], f"{pv}")

    world = World()
    # B* is the best single entry on S (m/good here). 'a' routes: the cheap
    # entry where it is right, else m/good -- a real router that beats B*.
    # 'z' always picks m/good: it IS B*, so it can never fill. 'c', later,
    # routes better still (cheapest right entry): it takes the seat.
    world.add("a", "ck-a", 1, 100, lambda q: "m/cheap" if right(q, "m/cheap") else "m/good")
    world.add("z", "ck-z", 2, 110, lambda q: "m/good")
    specs, reveals, records = {}, {}, []
    state, heads = seat_v5.SeatStateV5(), {}
    events, holder_at = {}, {}
    names_at = lambda e: ["a", "z"] if e < A + TAKE_AT else ["a", "z", "c"]  # noqa: E731
    for e in range(A, A + EPOCHS):
        if e == A + TAKE_AT:
            world.add("c", "ck-c", 3, 120, lambda q: next(
                (x for x in ("m/cheap", "m/good", "m/base") if right(q, x)), "m/best"))
        names = names_at(e)
        rec = record(e, specs.get(e), world, names)
        res = v5_epoch.score_epoch(epoch=e, state=state, heads=heads, record=rec, window=records,
                                   served=world.served(names), full_entries=ENTRIES,
                                   stratum_of=stratum, burn_uid=0,
                                   # Every exam question here is drawn fresh, unpublished.
                                   first_published=lambda _q: None)
        records.append(rec)
        state, heads = res.decision.state, res.heads
        events[e], holder_at[e] = res.decision.event, res.decision.holder
        reveals[e] = dict(res.reveal)
        nxt = spec_for(e + v5_epoch.SPEC_LAG, world, names, res.baseline or "m/good")
        specs[e + v5_epoch.SPEC_LAG] = nxt
        reveals[e] = {**reveals[e], **v5_epoch.spec_reveal(nxt, COST)}
        # the follower: everything from the root's reveals alone
        f_state, f_heads, f_recs = v5_epoch.rebuild(reveals, e)
        if (f_state, f_heads) != (state, heads):
            check(f"follower agrees at A+{e - A}", False, "rebuilt state differs")
        pub, _ = v5_epoch.spec_from_reveal(reveals[e])
        if v5_epoch.spec_agrees(nxt, pub) is not None:
            check(f"spec of A+{e - A + 2} agrees", False, "published spec differs")
        if res.decision.holder is None and res.uids != [0]:
            check(f"burn at A+{e - A}", False, f"uids {res.uids}")

    check("day one: A and A+1 burn, no spec", all(events[A + i] in ("empty", "burn")
                                                  and holder_at[A + i] is None for i in (0, 1)),
          f"{events[A]}, {events[A + 1]}")
    filled = [e for e in sorted(events) if holder_at[e] == "a"]
    first = filled[0] if filled else None
    check("fill on merit: head 'a' (beats B*) fills", first is not None,
          f"first at A+{first - A}" if first else "never")
    if first is not None:
        n_base = (first - (A + 2) + 1) * exam_v5.EXAM_BASELINE_QUESTIONS
        check("no fill before N_MIN questions paired with B*", n_base >= seat_v5.N_MIN,
              f"{n_base} B* questions by then, N_MIN {seat_v5.N_MIN}")
    check("head 'z' (= B*) never holds", all(h != "z" for h in holder_at.values()))
    took = [e for e in sorted(events) if holder_at[e] == "c"]
    check("take: head 'c' takes the seat by the margin", bool(took),
          f"at A+{took[0] - A}" if took else "never")
    check("follower rebuilds the same seat every epoch", not any("follower" in f for f in fails))

    width = max(len(r[1]) for r in rows)
    for status, name, detail in rows:
        print(f"{status}  {name:<{width}}  {detail}")
    print("events: " + " ".join(f"A+{e - A}:{events[e]}" for e in sorted(events)))
    print(f"{len(rows) - len(fails)}/{len(rows)} passed in {time.time() - t0:.0f} s")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
