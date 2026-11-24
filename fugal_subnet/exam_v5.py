"""Protocol v5: one epoch's exam and the paired items read from it (I1, I3, I4, I8).

A v5 head is judged on the value of its real picks on questions drawn after it
was committed, against the holder and the best single entry (B*) on the same
questions (arena_score). To judge a router only the outcomes of the entries it
picks are needed, so the subnet labels exactly those cells, and only where a
comparison needs them:

1. **Questions.** `draw_exam` draws candidate questions by the epoch nonce
   on the exam mix (each dataset's quota by largest remainder of its weight,
   each dataset's questions ranked by HMAC(nonce, question id), the draw put
   in the same keyed order). Which questions are eligible (fresh or already
   published, the memorisation check) is the caller's (memo_v4).
2. **Picks.** `exam_picks`: each contender's pick per question, by the v4
   pick function with the head's OWN lam (in (0, HEAD_LAM_MAX_V5]).
3. **Cells.** `build_spec` walks the candidates in order. A question needs a
   cell for each distinct pick when the contenders' picks differ, and on the
   first EXAM_BASELINE_QUESTIONS_V5 questions also B*'s cell (plus the
   contenders' cells) when any pick differs from B*'s. A question where
   every pick is equal needs no call and still counts (its items are 0).
   Questions are added while cells x votes fits the epoch's budget
   (EXAM_CALLS_PER_HOTKEY_V5 x the ring's hotkeys), at most EXAM_QUESTIONS_V5.
4. **Votes.** Each cell is assigned EXAM_VOTES_V5 hotkeys of distinct
   coldkeys, walking the sorted ring from an HMAC(nonce, cell) offset and
   skipping a hotkey already at its budget. The TD labels only the cells
   assigned to it (I8: `assigned_cells`).
5. **Items.** `comparison_items` / `baseline_items`: per question, the
   paired item of arena_score from the cells' pooled votes. Votes of BOTH
   compared operators never count (I4: no operator judges a comparison it is
   in, and a third party alone cannot move a cell it holds one vote of).
   A vote counts only from a hotkey assigned to that cell. Failed calls are
   dropped; a cell whose usable votes are all failures, from at least
   FAILED_CELL_MIN_VOTERS coldkeys, is a wrong answer at no cost (as
   RouterArena scores a row that never returns); a cell with no usable vote
   loses the question for that comparison only.

DETERMINISM. Keyed hashes, sorted orders, integers and Fractions only; the
spec has one canonical JSON form and `spec_hash` is its sha256.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from fugal_subnet import arena_score, config

EXAM_QUESTIONS = int(config.EXAM_QUESTIONS_V5)
EXAM_BASELINE_QUESTIONS = int(config.EXAM_BASELINE_QUESTIONS_V5)
EXAM_VOTES = int(config.EXAM_VOTES_V5)
EXAM_CALLS_PER_HOTKEY = int(config.EXAM_CALLS_PER_HOTKEY_V5)
HEAD_LAM_MAX = int(config.HEAD_LAM_MAX_V5)
Y_SCALE = int(config.Y_SCALE_V5)
# A cell whose usable votes all failed counts as a wrong answer only when at
# least this many coldkeys failed it (one failure alone could be the voter's).
FAILED_CELL_MIN_VOTERS = 2

_NO_PICK = ""


# --- questions -----------------------------------------------------------------------------


def exam_nonce(nonce: bytes) -> bytes:
    """The exam's domain-separated nonce."""
    return hashlib.sha256(bytes(nonce) + b":exam-v5").digest()


def _rank(key: bytes, label: str) -> bytes:
    return hmac.new(key, label.encode("utf-8"), hashlib.sha256).digest()


def draw_exam(nonce: bytes, by_dataset: Mapping[str, Sequence[str]],
              weights_ppm: Mapping[str, int], n: int) -> list[str]:
    """`n` candidate question ids on the exam mix, in keyed order.

    Each dataset with a positive weight and questions gets floor(n w / W) by
    largest remainder (ties by name), capped at what it has; its questions
    are the lowest HMAC(exam nonce, "q:" + id). The result is ordered by the
    same keyed digest, so datasets interleave. A dataset short of its quota
    leaves the draw short (no refill)."""
    if type(n) is not int or n < 0:
        raise ValueError(f"n must be a non-negative int, got {n!r}")
    key = exam_nonce(nonce)
    live = sorted(d for d, w in weights_ppm.items()
                  if int(w) > 0 and len(by_dataset.get(d, ())) > 0)
    total = sum(int(weights_ppm[d]) for d in live)
    if not live or n == 0:
        return []
    quota = {d: n * int(weights_ppm[d]) // total for d in live}
    rest = n - sum(quota.values())
    order = sorted(live, key=lambda d: (-(n * int(weights_ppm[d]) % total), d))
    for d in order[:rest]:
        quota[d] += 1
    chosen: list[tuple[bytes, str]] = []
    for d in live:
        ids = sorted(set(str(q) for q in by_dataset[d]))
        ranked = sorted(((_rank(key, "q:" + q), q) for q in ids))
        chosen.extend(ranked[:quota[d]])
    return [q for _, q in sorted(chosen)]


# --- picks ---------------------------------------------------------------------------------


def head_lam(head) -> Fraction:
    """A v5 head's own lam (the price its picks weigh cost at), exact: the
    head file's `lam`, finite, in (0, HEAD_LAM_MAX_V5]."""
    z = getattr(head, "success", None) or {}
    if "lam" not in z:
        raise ValueError("a v5 head carries its lam")
    raw = z["lam"]
    try:
        value = float(raw)
        lam = Fraction(value)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"the head's lam is not a finite number: {raw!r}") from e
    if not 0 < lam <= HEAD_LAM_MAX:
        raise ValueError(f"the head's lam must be in (0, {HEAD_LAM_MAX}], got {value!r}")
    return lam


def exam_picks(head, state, hidden, *, exclude: frozenset = frozenset()) -> tuple[str, ...]:
    """A head's pick per question (row of `hidden`) on the epoch's menu, by
    the v4 pick function at the head's own lam; "" where it has no pick."""
    from fugal_subnet.head_odds import picks_and_odds_v4

    out = picks_and_odds_v4(head, state, hidden, lam=head_lam(head), exclude=exclude)
    return tuple(_NO_PICK if p is None else str(p) for p in out.picks)


# --- the spec ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExamSpec:
    """One epoch's exam: what every validator recomputes and the root publishes."""

    epoch: int
    questions: tuple[str, ...]
    picks: Mapping[str, tuple[str, ...]]
    baseline: str
    baseline_questions: tuple[str, ...]
    cells: tuple[tuple[str, str], ...]
    ring: tuple[tuple[str, str], ...]
    assignment: Mapping[tuple[str, str], tuple[str, ...]] = field(default_factory=dict)

    def to_json(self) -> str:
        doc = {
            "epoch": int(self.epoch),
            "questions": list(self.questions),
            "picks": {h: list(p) for h, p in sorted(self.picks.items())},
            "baseline": self.baseline,
            "baseline_questions": list(self.baseline_questions),
            "cells": [list(c) for c in self.cells],
            "ring": [list(r) for r in self.ring],
            "assignment": [[q, e, list(self.assignment[(q, e)])] for q, e in self.cells],
        }
        return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def spec_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()

    @classmethod
    def from_json(cls, s: str) -> ExamSpec:
        d = json.loads(s)
        if not isinstance(d, dict):
            raise ValueError("an exam spec is a JSON object")
        questions = tuple(str(q) for q in d["questions"])
        picks = {str(h): tuple(str(x) for x in p) for h, p in d["picks"].items()}
        for h, p in picks.items():
            if len(p) != len(questions):
                raise ValueError(f"head {h[:16]} has {len(p)} picks for {len(questions)} questions")
        cells = tuple((str(q), str(e)) for q, e in d["cells"])
        assignment = {(str(q), str(e)): tuple(str(h) for h in hks)
                      for q, e, hks in d["assignment"]}
        if set(assignment) != set(cells):
            raise ValueError("the spec's assignment does not cover exactly its cells")
        spec = cls(epoch=int(d["epoch"]), questions=questions, picks=picks,
                   baseline=str(d["baseline"]),
                   baseline_questions=tuple(str(q) for q in d["baseline_questions"]),
                   cells=cells, ring=tuple((str(h), str(c)) for h, c in d["ring"]),
                   assignment=assignment)
        if spec.to_json() != s:
            raise ValueError("the exam spec is not in its canonical form")
        return spec


def _cells_for(values: Sequence[str], baseline: str, on_baseline: bool) -> set[str]:
    distinct = set(values)
    out: set[str] = set()
    if len(distinct) >= 2:
        out |= distinct
    if on_baseline and baseline and any(v != baseline for v in values):
        out |= distinct | {baseline}
    out.discard(_NO_PICK)
    return out


def _ring(ring: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    out = tuple(sorted((str(h), str(c)) for h, c in ring))
    hks = [h for h, _ in out]
    if len(set(hks)) != len(hks):
        raise ValueError("a hotkey appears twice in the ring")
    return out


def _assign(nonce: bytes, cells: Sequence[tuple[str, str]],
            ring: Sequence[tuple[str, str]], votes: int) -> dict[tuple[str, str], tuple[str, ...]]:
    key = exam_nonce(nonce)
    load = dict.fromkeys((h for h, _ in ring), 0)
    out: dict[tuple[str, str], tuple[str, ...]] = {}
    n = len(ring)
    for cell in sorted(cells):
        picked: list[str] = []
        seen_ck: set[str] = set()
        if n:
            start = int.from_bytes(_rank(key, f"cell:{cell[0]}\x00{cell[1]}")[:8], "big") % n
            for i in range(n):
                hk, ck = ring[(start + i) % n]
                if ck in seen_ck or load[hk] >= EXAM_CALLS_PER_HOTKEY:
                    continue
                picked.append(hk)
                seen_ck.add(ck)
                load[hk] += 1
                if len(picked) == votes:
                    break
        out[cell] = tuple(sorted(picked))
    return out


def build_spec(epoch: int, nonce: bytes, contenders: Mapping[str, Sequence[str]],
               menu: Iterable[str], pool: Sequence[str], ring: Iterable[tuple[str, str]],
               baseline: str) -> ExamSpec:
    """One epoch's exam spec.

    `pool` is the candidate question ids in exam order (`draw_exam`, with
    the memorisation check's eligibility applied by the caller), and
    `contenders` each contender head's picks aligned with `pool`
    (`exam_picks`). A pick that is not a `menu` entry counts as no pick.
    `ring` is the epoch's verified (hotkey, coldkey) pairs; `baseline` is
    B* (a `menu` entry, or "" for none)."""
    entries = set(str(e) for e in menu)
    base = str(baseline) if str(baseline) in entries else _NO_PICK
    pool = [str(q) for q in pool]
    if len(set(pool)) != len(pool):
        raise ValueError("the exam's candidate questions repeat")
    heads = sorted(str(h) for h in contenders)
    picks_in: dict[str, list[str]] = {}
    for h in heads:
        p = [str(x) if str(x) in entries else _NO_PICK for x in contenders[h]]
        if len(p) != len(pool):
            raise ValueError(f"head {h[:16]} has {len(p)} picks for {len(pool)} candidates")
        picks_in[h] = p
    ring_t = _ring(ring)
    votes = min(EXAM_VOTES, len({c for _, c in ring_t}))
    budget = EXAM_CALLS_PER_HOTKEY * len(ring_t)
    used, taken = 0, 0
    cells: set[tuple[str, str]] = set()
    for i, q in enumerate(pool):
        if taken >= EXAM_QUESTIONS:
            break
        need = _cells_for([picks_in[h][i] for h in heads], base,
                          on_baseline=taken < EXAM_BASELINE_QUESTIONS)
        cost = len(need) * votes
        if need and (votes == 0 or used + cost > budget):
            break
        used += cost
        cells |= {(q, e) for e in need}
        taken += 1
    questions = tuple(pool[:taken])
    cells_t = tuple(sorted(cells))
    return ExamSpec(
        epoch=int(epoch), questions=questions,
        picks={h: tuple(picks_in[h][:taken]) for h in heads},
        baseline=base,
        baseline_questions=questions[:EXAM_BASELINE_QUESTIONS] if base else (),
        cells=cells_t, ring=ring_t,
        assignment=_assign(nonce, cells_t, ring_t, votes))


def assigned_cells(spec: ExamSpec, hotkey: str) -> list[tuple[str, str]]:
    """The cells this hotkey's TD labels this epoch (I8), sorted."""
    hk = str(hotkey)
    return [c for c in spec.cells if hk in spec.assignment.get(c, ())]


# --- votes and items -----------------------------------------------------------------------


@dataclass(frozen=True)
class Vote:
    """One assigned exam call from one verified proof."""

    epoch: int
    qid: str
    entry: str
    hotkey: str
    coldkey: str
    y: int              # 0..Y_SCALE_V5, the pinned grader's label
    cost_nd: int        # what the call cost (provider_cost, else price x tokens), nanodollars
    failed: bool


def _median(values: list[Fraction]) -> Fraction:
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _index_votes(specs: Sequence[ExamSpec], votes: Iterable[Vote]):
    """{(epoch, qid, entry): [vote, ...]} keeping only votes from a hotkey
    assigned to that cell; a hotkey with two votes for one cell has neither."""
    assigned = {(s.epoch, q, e): set(hks) for s in specs for (q, e), hks in s.assignment.items()}
    seen: dict[tuple, list[Vote]] = {}
    for v in votes:
        k = (int(v.epoch), str(v.qid), str(v.entry))
        if v.hotkey not in assigned.get(k, ()):
            continue
        seen.setdefault(k + (v.hotkey,), []).append(v)
    out: dict[tuple, list[Vote]] = {}
    for k, vs in seen.items():
        if len(vs) == 1:
            out.setdefault(k[:3], []).append(vs[0])
    return out


def _cell_label(votes: list[Vote], exclude: frozenset[str]) -> tuple[Fraction, Fraction] | None:
    usable = [v for v in votes if v.coldkey not in exclude]
    ok = [v for v in usable if not v.failed]
    if ok:
        for v in ok:
            if type(v.y) is not int or not 0 <= v.y <= Y_SCALE:
                raise ValueError(f"a vote's label must be an int 0..{Y_SCALE}, got {v.y!r}")
            if type(v.cost_nd) is not int or v.cost_nd < 0:
                raise ValueError(f"a vote's cost must be a non-negative int, got {v.cost_nd!r}")
        return (_median([Fraction(v.y) for v in ok]),
                _median([Fraction(v.cost_nd) for v in ok]))
    if len({v.coldkey for v in usable}) >= FAILED_CELL_MIN_VOTERS:
        return Fraction(0), Fraction(0)
    return None


def mean_pick_cost_nd(specs: Sequence[ExamSpec], head: str, cost_nd: Mapping[str, int]) -> int:
    """A head's mean pick-table cost per question over the specs it is in, in
    nanodollars (floor): label-free, so it may set V. A question it has no pick
    on costs 0."""
    total, n = 0, 0
    for s in specs:
        for e in s.picks.get(head, ()):
            total += int(cost_nd[e]) if e else 0
            n += 1
    return total // n if n else 0


def _items(index, pairs, exclude, v_nd, stratum_of) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    labels: dict[tuple, tuple | None] = {}

    def label(epoch, q, e):
        if not e:
            return Fraction(0), Fraction(0)
        k = (epoch, q, e)
        if k not in labels:
            labels[k] = _cell_label(index.get(k, []), exclude)
        return labels[k]

    for spec, q, px, py in pairs:
        stratum = str(stratum_of(q))
        if px == py:
            out.setdefault(stratum, []).append(0)
            continue
        lx, ly = label(spec.epoch, q, px), label(spec.epoch, q, py)
        if lx is None or ly is None:
            continue
        out.setdefault(stratum, []).append(
            arena_score.item_units(lx[0], ly[0], lx[1], ly[1], v_nd))
    return out


def comparison_items(specs: Sequence[ExamSpec], votes: Iterable[Vote], x: str, y: str,
                     exclude_coldkeys: frozenset[str], mean_cost_nd: Mapping[str, int],
                     stratum_of: Callable[[str], str]) -> dict[str, list[int]]:
    """X's items against Y by stratum, over every spec both are contenders of
    (in seat_bound units: arena_score.item_units). `exclude_coldkeys` are the
    two compared operators; `mean_cost_nd[h]` each head's mean pick-table cost
    (`mean_pick_cost_nd`), which sets V_XY."""
    specs = [s for s in specs if x in s.picks and y in s.picks]
    index = _index_votes(specs, votes)
    v_nd = arena_score.value_scale_nd(int(mean_cost_nd[x]), int(mean_cost_nd[y]))
    pairs = [(s, q, s.picks[x][i], s.picks[y][i])
             for s in specs for i, q in enumerate(s.questions)]
    return _items(index, pairs, frozenset(exclude_coldkeys), v_nd, stratum_of)


def baseline_items(specs: Sequence[ExamSpec], votes: Iterable[Vote], x: str,
                   exclude_coldkeys: frozenset[str], mean_cost_nd: Mapping[str, int],
                   stratum_of: Callable[[str], str]) -> dict[str, list[int]]:
    """X's items against B* on each spec's baseline questions (B* is the
    spec's `baseline` entry; `mean_cost_nd` holds the entry's pick-table cost
    per question under its id). Specs without X or without a baseline are
    skipped; `exclude_coldkeys` is X's operator."""
    specs = [s for s in specs if x in s.picks and s.baseline]
    index = _index_votes(specs, votes)
    out: dict[str, list[int]] = {}
    for s in specs:
        v_nd = arena_score.value_scale_nd(int(mean_cost_nd[x]), int(mean_cost_nd[s.baseline]))
        pos = {q: i for i, q in enumerate(s.questions)}
        pairs = [(s, q, s.picks[x][pos[q]], s.baseline) for q in s.baseline_questions]
        for k, v in _items(index, pairs, frozenset(exclude_coldkeys), v_nd, stratum_of).items():
            out.setdefault(k, []).extend(v)
    return out
