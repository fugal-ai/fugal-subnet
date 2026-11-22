"""A synthetic protocol v4 world for the seat's tests and attack cases
(fugal_subnet/seat_v4.py; tests/test_seat_v4.py, run_miner_attacks.SEAT_ATTACKS).

A `World` holds what the seat reads, built directly: each epoch's labels
(labels_v4.LabelPairs, voted by coldkeys that own no head unless a test says
so), every head's odds on them, the real-picks records (picks_v4.PicksEpoch),
the copy rule's ownership state and the seat state. A head is described by
how much it knows of each label's outcome (`skill`, 0 knows nothing and
predicts the entry's rate, 1 knows the outcome to within a tenth) and by its
real picks (`value`, its net value on each challenge question, in points: a
challenger's item is its value less the value of the holder its epoch's
challenges pair with, seat_v4.challenge_holder), so each rule's verdict is
set by construction rather than by luck.

Odds are given only for the epochs a head is verified in; the epochs it
missed get theirs when it is next verified (late odds), as the live epoch
computes them: all at once, or, with `late_slots` set, on the live epoch's
schedule (v2_epoch.late_epochs: at most that many head-epochs an epoch,
the seat's heads first, none for an epoch a successor's predecessors' odds
stand for).
"""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np

from fugal_subnet import copy_rule, labels_v4, seat_v4
from fugal_subnet.champion import EpochPairs, Window
from fugal_subnet.picks_v4 import PicksEpoch

ENTRIES = ("lab-a/model@high", "lab-b/model@low")
SCALE = 10_000
BPE = 10
UNIT = seat_v4.UNITS_PER_POINT
VOTERS = ("ck-voter-1", "ck-voter-2", "ck-voter-3")


def hash_of(name: str) -> str:
    return hashlib.sha256(f"seat-v4-world:{name}".encode()).hexdigest()


@dataclasses.dataclass
class Head:
    name: str
    owner: str
    hotkey: str
    block: int
    skill: float = 0.0
    noise: int = 0              # a seed: odds perturbed by +-1 unit on some labels
    rows: tuple[str, ...] = ENTRIES
    digest_salt: str = ""       # rows' bytes: a different salt is a different row
    row_salt: dict = dataclasses.field(default_factory=dict)
    row_skill: dict = dataclasses.field(default_factory=dict)   # entry -> its row's skill

    @property
    def hash(self) -> str:
        return hash_of(self.name)

    def digests(self) -> dict[str, str]:
        return {e: hashlib.sha256(f"{e}|{self.row_salt.get(e, self.digest_salt)}".encode())
                .hexdigest() for e in self.rows}


class World:
    def __init__(self, *, labels: int = 500, seed: int = 0, floor_block: int | None = None,
                 salt: str = "salt", questions: int = 100, late_slots: int | None = None):
        self.labels, self.questions = labels, questions
        self.late_slots = late_slots
        self.rng = np.random.default_rng(seed)
        self.window = Window.empty(48)
        self.heads: dict[str, Head] = {}
        self.state = seat_v4.SeatStateV4()
        self.floor_block, self.salt = floor_block, salt
        self.commits: dict[str, tuple[str, str, int]] = {}   # hotkey -> (head name, coldkey, block)
        self.copy = copy_rule.CopyState()
        self.decisions: dict[int, seat_v4.SeatDecisionV4] = {}
        # entry -> the epoch its allowance ends (a missing row is certainly
        # wrong from it; at the entry's rate, 5000, before). An entry not
        # named is a seed entry: no allowance.
        self.missing_after: dict[str, int] = {}
        # entry -> the epoch it becomes full: its labels drawn before are
        # provisional (labels_v4.PROVISIONAL_LABEL, read by no comparison).
        self.provisional_until: dict[str, int] = {}
        self.voters: dict[int, tuple[str, ...]] = {}

    # --- heads ---------------------------------------------------------------------
    def add(self, name: str, *, owner: str | None = None, hotkey: str | None = None,
            block: int = 5, **kw) -> Head:
        h = Head(name, owner or f"ck-{name}", hotkey or f"hk-{name}", block, **kw)
        self.heads[name] = h
        self.commits[h.hotkey] = (name, h.owner, block)
        return h

    def odds(self, h: Head, pairs: labels_v4.LabelPairs, epoch: int) -> np.ndarray:
        y = pairs.y.astype(np.int64)
        base = np.full(len(y), SCALE // 2, dtype=np.int64)
        target = np.where(y >= SCALE // 2, 9 * SCALE // 10, SCALE // 10)
        skill = np.array([h.row_skill.get(m, h.skill) for m in pairs.models])
        o = np.rint(base + skill * (target - base)).astype(np.int64)
        if h.noise:
            r = np.random.default_rng(h.noise * 1_000_003 + epoch).integers(-1, 2, len(y))
            o = np.clip(o + r, 0, SCALE)
        has = np.array([m in h.rows for m in pairs.models])
        return np.where(has, o, -1).astype(np.int16)

    # --- an epoch ------------------------------------------------------------------
    def epoch(self, e: int, *, served=(), answered=(), value=None, memo=(), allow_take=True,
              voters=None, outage=False) -> seat_v4.SeatDecisionV4:
        """Run epoch `e`: the heads named in `served` are verified (served by
        their owners), those in `answered` also have a counted challenge of
        `questions` questions, each at `value[name]` points a question
        (default 0) less the value of the holder it pairs with."""
        value = value or {}
        n = self.labels
        y = (self.rng.random(n) < 0.5).astype(np.int32) * SCALE
        models = tuple(ENTRIES[i % 2] for i in range(n))
        qids = tuple(f"q{e}-{i:05d}" for i in range(n))
        missing = np.array([labels_v4.PROVISIONAL_LABEL if self.provisional_until.get(m, 0) > e
                            else -1 if self.missing_after.get(m, 0) <= e else 5000
                            for m in models], dtype=np.int16)
        vote = voters or VOTERS
        pairs = labels_v4.label_pairs(
            EpochPairs(question_ids=qids, models=models, benchmarks=("b1",) * n,
                       categories=("c1",) * n, y=y, prompt_tokens=np.zeros(n, np.int32),
                       completion_tokens=np.zeros(n, np.int32)),
            [(vote[i % len(vote)],) for i in range(n)], missing)
        commits = [copy_rule.Commitment(hk, ck, hash_of(name), block)
                   for hk, (name, ck, block) in sorted(self.commits.items())]
        self.copy = copy_rule.observe(self.copy, e, e * BPE, commits, ghost_retire=24)
        served = [self.heads[s] for s in served]
        serving = {h.hash: 100 + i for i, h in enumerate(sorted(served, key=lambda h: h.name))}
        pairs_w = dict(self.window.pairs)
        pairs_w[e] = pairs
        odds = {hh: dict(per) for hh, per in self.window.odds.items()}
        rows = {h.hash: h.digests() for h in served}
        state = seat_v4.observe(self.state, epoch=e, serving=serving, rows=rows,
                                copy_state=self.copy, menu_ids=ENTRIES, blocks_per_epoch=BPE)
        cats = ("c1",) * self.questions
        picks = dict(self.window.picks)
        holder = seat_v4.challenge_holder(self.state, e)
        base = value.get(self.name(holder), 0) if holder is not None else 0
        picks[e] = PicksEpoch(e, cats, {self.heads[a].hash: (int((value.get(a, 0) - base) * UNIT),)
                                        * self.questions for a in answered}, outage, holder)
        if self.late_slots is not None:
            self._live_late_odds(state, served, pairs_w, odds, e, picks)
        else:
            for h in served:
                since = seat_v4.odds_since(state, h.hash, e)
                for k, p in pairs_w.items():
                    if k >= since and k not in odds.get(h.hash, {}):
                        odds.setdefault(h.hash, {})[k] = labels_v4.effective(self.odds(h, p, k), p)
        self.window = dataclasses.replace(self.window, pairs=pairs_w, odds=odds, picks=picks)
        d = seat_v4.seat_step(self.state, epoch=e, window=self.window, serving=serving, rows=rows,
                              copy_state=self.copy, menu_ids=ENTRIES, blocks_per_epoch=BPE,
                              floor_block=self.floor_block, salt=self.salt,
                              memo_refused={self.heads[m].hash for m in memo},
                              allow_take=allow_take)
        self.state = d.state
        self.decisions[e] = d
        self.serving = serving
        return d

    def _live_late_odds(self, state, served, pairs_w, odds, e, picks) -> None:
        """Odds now for every served head, then the late odds the live epoch
        computes (v2_epoch._add_v4_epoch): `late_slots` head-epochs, each
        successor read through its seat view (seat_v4.side_of), the seat's
        heads first (seat_v4.late_odds_first, on the window holding the
        epoch's picks)."""
        from fugal_subnet.v2_epoch import late_epochs

        by = {h.hash: h for h in served}
        for h in served:
            odds.setdefault(h.hash, {})[e] = labels_v4.effective(self.odds(h, pairs_w[e], e),
                                                                  pairs_w[e])
        win = dataclasses.replace(self.window, pairs=pairs_w, odds=odds, picks=picks)
        since = {hh: seat_v4.odds_since(state, hh, e) for hh in by}
        sides = {hs.head: seat_v4.side_of(hs) for hs in state.heads if hs.predecessors}
        first = seat_v4.late_odds_first(state, win, epoch=e)
        for hh, eps in late_epochs(win, by, since, self.late_slots, sides, first).items():
            for k in eps:
                odds[hh][k] = labels_v4.effective(self.odds(by[hh], pairs_w[k], k), pairs_w[k])

    def name(self, h: str | None) -> str | None:
        if h is None:
            return None
        return next((n for n, x in self.heads.items() if x.hash == h), None)

    def holder(self) -> str | None:
        return self.name(self.state.champion)
