"""Protocol v5: the seat, decided on real picks at RouterArena's trade-off.

One rule, re-run every epoch on the window's paired exam items
(fugal_subnet/exam_v5.py; docs/INVARIANTS.md, "I3 — v5: the seat in one
rule"). An item is one exam question's difference, in S-points (accuracy
after RouterArena's log-cost adjustment, arena_score.py), between two heads'
picks, judged only on votes from operators other than the two compared
(I4). Every comparison is `seat_bound.lower_bound`, valid at every look.

    B*         the best single menu entry on the window (the baseline).
    incumbent  the holder, if it stands, is not memo-refused, is not an
               exact copy, and is not shown worse than B* (the bound of
               B* over the holder does not pass). Otherwise none.
    upgrade    a newer head of the holder's own coldkey replaces it when it
               is NOT SHOWN WORSE than the holder (the bound of the holder
               over it does not pass -UPGRADE_MARGIN = +1/4; with fewer than
               N_MIN paired questions nothing is shown, so a fresh retrain is
               not shown worse) and its own bound over B* passes 0 (it is
               payable). It keeps the holder's place and since.
    walk       the other contenders, earliest first commit first: C replaces
               the incumbent when its bound over the incumbent passes
               TAKE_MARGIN (+1/4), or there is no incumbent, and its bound
               over B* passes 0. A later head must beat whoever holds the
               seat at its turn, so a near-copy of an earlier head needs the
               margin over it.
    pay        a seat (holder, then the history, 85/10/5 through
               payout.seat_weights) is paid while its head is live this
               epoch, is not memo-refused, and its bound over B* passes 0;
               an unpaid share burns.

Standing: a holder that stops standing (not verified in the last STANDING
epochs) goes to the history and the seat refills on merit in the same epoch.
A returning former holder has no resume right: it walks as an ordinary
contender in first-commit order and must beat whoever then holds the seat.

Contenders: per coldkey the newest standing head, plus, for the holder's
coldkey, the holder itself; at most CONTENDERS_MAX, earliest first commit
kept. Exact copies (copy_rule ownership: served by a coldkey that is not the
head's owner) and memo-refused heads never contend.

DETERMINISM. Integers and Fractions only; the order of the contenders given
does not matter (they are sorted by (first commit block, head hash)); the
item callables are called once per ordered pair and must be pure.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from fugal_subnet import config, seat_bound

UNITS_PER_POINT = seat_bound.UNITS_PER_POINT

# v5 constants (fugal_subnet/config.py, the `# --- v5 ---` block). The
# fallbacks are the design's starting values (v5/DESIGN.md 2) and are used
# only until the config block exists.
TAKE_MARGIN = Fraction(getattr(config, "TAKE_MARGIN_V5", "1/4"))
UPGRADE_MARGIN = Fraction(getattr(config, "UPGRADE_MARGIN_V5", "-1/4"))
N_MIN = int(getattr(config, "N_MIN_V5", 2000))
N0 = int(getattr(config, "N0_V5", 2000))
CONTENDERS_MAX = int(getattr(config, "CONTENDERS_MAX_V5", 24))
SEAT_SHARES = tuple(config.PAYOUT_SEAT_SHARES)
HISTORY_MAX = len(SEAT_SHARES) - 1

Items = Mapping[str, Sequence[int]]


def _units(points: Fraction) -> int:
    """A margin in S-points as bound units, rounded toward -inf (exact)."""
    q = points * UNITS_PER_POINT
    return q.numerator // q.denominator


TAKE_UNITS = _units(TAKE_MARGIN)
UPGRADE_UNITS = _units(UPGRADE_MARGIN)


@dataclass(frozen=True)
class SeatStateV5:
    holder: str | None = None
    holder_since: int | None = None
    history: tuple[str, ...] = ()      # former holders, most recent first

    def to_reveal(self) -> dict:
        return {"holder": self.holder, "holder_since": self.holder_since,
                "history": list(self.history)}

    @classmethod
    def from_reveal(cls, d: Mapping | None) -> SeatStateV5:
        if not d:
            return cls()
        holder = d.get("holder")
        since = d.get("holder_since")
        history = d.get("history") or ()
        if holder is not None and type(holder) is not str:
            raise ValueError("holder must be a head hash or null")
        if since is not None and type(since) is not int:
            raise ValueError("holder_since must be an int or null")
        if not all(type(h) is str for h in history):
            raise ValueError("history must be head hashes")
        return cls(holder, since, tuple(history)[:HISTORY_MAX])


@dataclass(frozen=True)
class Contender:
    head: str
    coldkey: str
    hotkey: str
    first_commit_block: int
    standing: bool = True          # verified in the last STANDING epochs
    # Refused by a memorisation check. v5's caller never sets it: v5 keeps
    # what a head could have read out of its comparisons instead
    # (v5_epoch.eligible_after, INVARIANTS I5 — v5).
    memo_refused: bool = False
    predecessor: str | None = None
    live: bool = True              # verified this epoch (a paid seat must be)
    copy: bool = False             # an exact copy (copy_rule.is_exact_copy)


@dataclass(frozen=True)
class SeatDecisionV5:
    epoch: int
    event: str                     # take | upgrade | fill | hold | burn | empty
    holder: str | None
    holder_since: int | None
    history: tuple[str, ...]
    payable: tuple[str, ...]
    evidence: dict = field(default_factory=dict)

    @property
    def state(self) -> SeatStateV5:
        return SeatStateV5(self.holder, self.holder_since, self.history)


def _bound_doc(b: seat_bound.Bound) -> dict:
    return {"n": b.n, "lower": b.lower, "passes": b.passes}


def _order(c: Contender) -> tuple[int, str]:
    return (c.first_commit_block, c.head)


def select(contenders: Sequence[Contender], holder: str | None) -> list[Contender]:
    """The heads the walk may consider, earliest first commit first: exact
    copies and memo-refused or absent heads never; per coldkey the newest
    standing head; the holder kept as its coldkey's extra head; at most
    CONTENDERS_MAX, the holder always among them."""
    seen: dict[str, Contender] = {}
    for c in sorted(contenders, key=_order):
        if c.head in seen and seen[c.head] != c:
            raise ValueError(f"head {c.head[:16]} given twice with different facts")
        seen[c.head] = c
    ok = [c for c in seen.values() if c.standing and not c.memo_refused and not c.copy]
    newest: dict[str, Contender] = {}
    for c in sorted(ok, key=_order):
        if c.head != holder:
            newest[c.coldkey] = c          # sorted ascending: the last is the newest
    chosen = list(newest.values())
    held = next((c for c in ok if c.head == holder), None)
    chosen.sort(key=_order)
    if held is not None:
        chosen = [held] + chosen[: CONTENDERS_MAX - 1]
    else:
        chosen = chosen[:CONTENDERS_MAX]
    return sorted(chosen, key=_order)


def decide(state: SeatStateV5, epoch: int, contenders: Sequence[Contender],
           items: Callable[[str, str], Items],
           baseline_items: Callable[[str], Items]) -> SeatDecisionV5:
    """The epoch's seat. `items(x, y)` gives the paired items of x over y
    (S-point units, by stratum); `baseline_items(x)` those of x over B*."""
    if type(epoch) is not int:
        raise TypeError("epoch must be an int")
    pool = select(contenders, state.holder)
    by_head = {c.head: c for c in contenders}
    cache: dict[tuple, seat_bound.Bound] = {}

    def bound(key: tuple, fn: Callable[[], Items], margin: int) -> seat_bound.Bound:
        k = key + (margin,)
        if k not in cache:
            cache[k] = seat_bound.lower_bound(fn(), margin_units=margin, n_min=N_MIN, n0=N0)
        return cache[k]

    def over_b(h: str) -> seat_bound.Bound:
        return bound(("B", h), lambda: baseline_items(h), 0)

    def b_over(h: str) -> seat_bound.Bound:
        # B* over h: the negated items (shown worse than B* when it passes)
        def neg() -> Items:
            return {s: [-x for x in xs] for s, xs in baseline_items(h).items()}
        return bound(("-B", h), neg, 0)

    def beats(x: str, y: str, margin: int) -> seat_bound.Bound:
        return bound(("X", x, y), lambda: items(x, y), margin)

    heads_doc: dict[str, dict] = {}

    def doc(h: str) -> dict:
        return heads_doc.setdefault(h, {"reasons": []})

    for c in contenders:
        d = doc(c.head)
        d.update({"coldkey": c.coldkey, "hotkey": c.hotkey,
                  "first_commit_block": c.first_commit_block})
        if c.copy:
            d["reasons"].append("an exact copy of a head another coldkey owns")
        elif c.memo_refused:
            d["reasons"].append("refused by the memorisation check")
        elif not c.standing:
            d["reasons"].append("not standing")
        elif c not in pool:
            d["reasons"].append("not a contender (an older head of its coldkey, or over the cap)")

    holder = state.holder
    history = list(state.history)
    since = state.holder_since
    in_pool = {c.head for c in pool}

    # 1. the incumbent
    incumbent = holder if holder in in_pool else None
    if holder is not None and incumbent is None:
        doc(holder)["reasons"].append("holder does not stand: no incumbent")
    if incumbent is not None:
        worse = b_over(incumbent)
        doc(incumbent)["b_over_holder"] = _bound_doc(worse)
        if worse.passes:
            doc(incumbent)["reasons"].append("shown worse than the best single model: no incumbent")
            incumbent = None
    start = incumbent
    event = None

    # 2. upgrade: the holder's own coldkey, newer head, not shown worse
    if incumbent is not None:
        own = by_head[incumbent].coldkey
        succ = [c for c in pool if c.coldkey == own and c.head != incumbent
                and _order(c) > _order(by_head[incumbent])
                and (c.predecessor is None or c.predecessor == incumbent)]
        for c in sorted(succ, key=_order, reverse=True)[:1]:
            # shown worse: the holder's bound over the successor passes the
            # upgrade margin's magnitude (+1/4); before N_MIN it never passes
            worse = beats(incumbent, c.head, -UPGRADE_UNITS)
            vb = over_b(c.head)
            doc(c.head).update({"holder_over_upgrade": _bound_doc(worse),
                                "vs_baseline": _bound_doc(vb)})
            if worse.passes:
                doc(c.head)["reasons"].append("upgrade refused: shown worse than the holder")
            elif not vb.passes:
                doc(c.head)["reasons"].append("upgrade refused: does not beat the best single model")
            else:
                doc(c.head)["reasons"].append("upgraded the holder (same coldkey, not shown worse)")
                incumbent, event = c.head, "upgrade"

    # 3. the walk, earliest first commit first
    before_walk = incumbent
    took = False
    for c in pool:
        if c.head == incumbent or (event == "upgrade" and c.head == start):
            continue
        d = doc(c.head)
        vb = over_b(c.head)
        d["vs_baseline"] = _bound_doc(vb)
        if incumbent is not None:
            vs = beats(c.head, incumbent, TAKE_UNITS)
            d["vs_holder"] = _bound_doc(vs)
            d["vs_holder_head"] = incumbent
            if not vs.passes:
                d["reasons"].append(f"does not beat {incumbent[:16]} by the margin")
                continue
        if not vb.passes:
            d["reasons"].append("does not beat the best single model")
            continue
        d["reasons"].append("took the seat" if incumbent is not None else "filled the empty seat")
        incumbent, took = c.head, True
    if took:
        event = "fill" if before_walk is None else "take"

    # 4. the new state: a replaced or demoted holder goes to the history; an
    # upgraded one does not (its successor keeps the place and the since)
    if took and before_walk is not None:
        history = [before_walk] + history
    elif holder is not None and before_walk is None:
        history = [holder] + history
    seen_h: set[str] = set()
    history = [h for h in history if h != incumbent and not (h in seen_h or seen_h.add(h))]
    if incumbent is None:
        since = None
    elif took:
        since = epoch
    history = tuple(history[:HISTORY_MAX])

    # 5. pay
    def paid(h: str | None) -> bool:
        c = by_head.get(h) if h is not None else None
        if c is None or not c.live or c.memo_refused or c.copy:
            return False
        ok = over_b(h)
        doc(h)["paid_vs_baseline"] = _bound_doc(ok)
        return ok.passes

    payable = tuple(h for h in (incumbent, *history) if h is not None and paid(h))
    if event is None:
        event = "empty" if incumbent is None else "hold"
    if incumbent is not None and incumbent not in payable and event == "hold":
        event = "burn"
    evidence = {
        "epoch": epoch, "event": event, "holder_before": holder, "holder": incumbent,
        "holder_since": since, "history": list(history), "payable": list(payable),
        "order": [c.head for c in pool],
        "burned": incumbent not in payable,
        "constants": {"take_margin": str(TAKE_MARGIN), "upgrade_margin": str(UPGRADE_MARGIN),
                      "n_min": N_MIN, "n0": N0, "contenders_max": CONTENDERS_MAX},
        "heads": {h: heads_doc[h] for h in sorted(heads_doc)},
    }
    return SeatDecisionV5(epoch, event, incumbent, since, history, payable, evidence)


def weights(decision: SeatDecisionV5, serving_uid: Mapping[str, int], *,
            burn_uid: int) -> tuple[list[int], list[float]]:
    """The epoch's weights: 85/10/5 to the holder and history while payable
    (payout.seat_weights), every unpaid share to `burn_uid`."""
    from fugal_subnet import payout

    return payout.seat_weights(decision.holder, decision.history, serving_uid,
                               frozenset(decision.payable), shares=SEAT_SHARES,
                               paid_fraction=1.0, burn_uid=burn_uid)
