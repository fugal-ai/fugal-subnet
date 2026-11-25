"""A synthetic protocol v5 world for the seat's tests and attack cases
(fugal_subnet/seat_v5.py; tests/test_seat_v5.py).

Each exam question has one latent difficulty u in [0, 1). A head with
accuracy a answers it correctly when u < a, except on a share `noise` of the
questions, where it draws its own outcome from its own seed: so two heads of
equal accuracy disagree on some questions and agree on the rest, and a
head's edge over another is set by construction rather than by luck. B*,
the best single model, is a head like any other. Costs are left out (items
are accuracy differences only): the item of x over y on a question is
100 x (correct_x - correct_y) S-points, in seat_bound units.
"""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np

from fugal_subnet import seat_v5

UNIT = seat_v5.UNITS_PER_POINT
STRATA = ("knowledge", "trivia", "medical")


def hash_of(name: str) -> str:
    return hashlib.sha256(f"seat-v5-world:{name}".encode()).hexdigest()


@dataclasses.dataclass
class Head:
    name: str
    coldkey: str
    block: int
    accuracy: float
    noise: float = 0.2
    seed: int = 0
    standing: bool = True
    memo_refused: bool = False
    live: bool = True
    copy: bool = False
    predecessor: str | None = None

    @property
    def hash(self) -> str:
        return hash_of(self.name)

    def contender(self) -> seat_v5.Contender:
        return seat_v5.Contender(
            head=self.hash, coldkey=self.coldkey, hotkey=f"hk-{self.name}",
            first_commit_block=self.block, standing=self.standing,
            memo_refused=self.memo_refused, predecessor=self.predecessor,
            live=self.live, copy=self.copy)


class World:
    def __init__(self, *, questions: int = 6000, baseline_accuracy: float = 0.70, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n = questions
        self.u = rng.random(questions)
        self.stratum = [STRATA[i % len(STRATA)] for i in range(questions)]
        self.heads: dict[str, Head] = {}
        self.baseline = Head("B*", "ck-baseline", 0, baseline_accuracy, seed=10_000 + seed)
        self._correct: dict[str, np.ndarray] = {}

    def add(self, head: Head) -> Head:
        self.heads[head.hash] = head
        return head

    def correct(self, head: Head) -> np.ndarray:
        if head.hash not in self._correct:
            rng = np.random.default_rng(int(hashlib.sha256(
                f"{head.name}:{head.seed}".encode()).hexdigest()[:8], 16))
            own = rng.random(self.n)
            redraw = rng.random(self.n) < head.noise
            c = np.where(redraw, own < head.accuracy, self.u < head.accuracy)
            self._correct[head.hash] = c.astype(np.int64)
        return self._correct[head.hash]

    def _head(self, h: str) -> Head:
        return self.baseline if h == self.baseline.hash else self.heads[h]

    def items(self, x: str, y: str) -> dict[str, list[int]]:
        d = (self.correct(self._head(x)) - self.correct(self._head(y))) * 100 * UNIT
        out: dict[str, list[int]] = {s: [] for s in STRATA}
        for s, v in zip(self.stratum, d.tolist()):
            out[s].append(int(v))
        return out

    def baseline_items(self, x: str) -> dict[str, list[int]]:
        return self.items(x, self.baseline.hash)

    def decide(self, state: seat_v5.SeatStateV5, epoch: int = 100,
               order: list[str] | None = None) -> seat_v5.SeatDecisionV5:
        heads = [self.heads[h] for h in (order or list(self.heads))]
        return seat_v5.decide(state, epoch, [h.contender() for h in heads],
                              self.items, self.baseline_items)
