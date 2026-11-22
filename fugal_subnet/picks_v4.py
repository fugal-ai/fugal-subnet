"""Protocol v4: the real-picks check (rule B) as a one-epoch challenge, and
the absolute failure bound.

docs/decisions/SEAT_MINIMAL_V4.md, sections 3.4, 3.7, 5.4 and 8 (package S3),
as docs/decisions/FAST_CROWN_V4.md changes them (packages F1, F5, F7).
Nothing on the v1-v3 path reads this module; the seat (seat_v4) calls
`picks_check` and `answered_in`.

THE CHALLENGE. Each epoch has a challenge list of CHALLENGE_LIST_SIZE_V4
questions (epoch_plan.challenge_list_v4: the scored slice first, then
published questions on the scored mix, drawn by the epoch nonce). A miner
chooses each epoch whether its TD answers a challenge, and its size: the
first 500 or 1,000 questions of the list (CHALLENGE_SIZES_V4). For each
question the TD computes its own head's pick and the pick of the seat holder
the root's reveal of E - CHALLENGE_HOLDER_LAG_V4 names
(seat_v4.challenge_holder), from the holder's published bytes on the epoch's
menu and pick-cost table over the menu's full entries only
(`challenge_picks`: a provisional entry is never a challenge pick), calls
its own pick and, where the two differ, the holder's. Every validator recomputes both
picks (v2_epoch._challenge_epoch); a challenge naming another holder or
stating another pick is refused (not counted: the run restarts). The holder
answers nothing: it stands and is paid on its exploration (seat_v4).

THE ITEMS (FAST_CROWN_V4.md 2, `challenge_items`), one per question, in
units of 1 / BOUND_UNITS_PER_POINT_V4 of a point:

    same entry          0                      (one answer serves both)
    different entries   v_challenger - v_holder,   v = 100 x (grade - t / V)
    no holder pick      v_challenger           (the holder has odds for no
                                                entry on the menu: its side is 0)

t the epoch's pick cost of the entry in the question's category
(menu_state.pick_cost on THAT epoch's MenuState), V = 1 /
routing_protocol.BENCHMARK_LAMBDA dollars, so a cent is one point; the grade
the call's own (the TD's pinned grader on its reply). Failed calls fail
closed: the challenger's own failed call is graded 0 and the holder's failed
call 1, each at its table cost, so a failed call never raises a challenger's
items. A holder-side call the challenger's TD did not start is no item
(NO_ITEM). When more than HOLDER_FAIL_CAP_BP (10%) of the holder-side calls
failed or were not started the challenge is void (`challenge_verdict`):
every item NO_ITEM, counted for and against nobody, published with its
reason. Each call's value is rounded half to even once (`call_value`), and
an item is held within INT32_MIN + 1 .. INT32_MAX (as published).

RULE B. For a head H and a bar B, over H's ANSWERED RUN (its unbroken
sequence of epochs with a counted challenge, ending at the newest recorded
epoch), on the items of the run's epochs whose challenges paired with B (or
with a predecessor B succeeded: seat_v4.lineage_records), the one bound
(seat_bound.lower_bound) stratified by category passes with margin
-PICKS_MARGIN_POINTS_V4 (1.5 points) on at least PICKS_MIN_QUESTIONS_V4 (500)
items, n0 = BOUND_N0_V4 (100). A missed epoch restarts the run, so an
attempt is judged only with the attempts before it that were not withheld.
A void challenge keeps the run, but only VOID_MAX_IN_RUN (1) of them: a
second one back ends it, so voids cannot select a run's good epochs.
An epoch with no verified proof at all is not recorded and breaks nobody's
run.

THE FAILURE BOUND (3.7, 5.4). A failed call says nothing about a model, and
a miner controls its own network: a proof with more than FAIL_MAX_BP_V4 (34%)
of its exploration calls failed is not counted (`failure_refusals`), except
in an outage epoch: the median coldkey failed more than RETIRE_OUTAGE_RATE_BP
(30%) of all its exploration calls (`menu_inputs.outage`, the menu step's
health rule on the same verified proofs).

DETERMINISM. Integers and Fractions only, every collection read sorted.
"""
from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from fugal_subnet import config
from fugal_subnet.seat_bound import Bound, lower_bound

UNITS_PER_POINT = config.BOUND_UNITS_PER_POINT_V4
BP = 10_000
FAIL_MAX_BP = config.FAIL_MAX_BP_V4
PICKS_MARGIN_POINTS = Fraction(config.PICKS_MARGIN_POINTS_V4)
# The bound's margin in units: rule B asks that H is not more than the margin
# WORSE than the bar, so the margin is negative.
PICKS_MARGIN_UNITS = -int(PICKS_MARGIN_POINTS * UNITS_PER_POINT)
PICKS_MIN_QUESTIONS = config.PICKS_MIN_QUESTIONS_V4
# The evidence the bound's mixture is tuned to (FAST_CROWN_V4.md 0): 100 items
# for rules A and B, not the rule's minimum.
BOUND_N0 = config.BOUND_N0_V4
# The reveal field and the digest key this package publishes.
REVEAL_KEY = "picks_v4"
DIGEST_KEY = "picks_digest"
# Items are published as int32 (FAST_CROWN_V4.md 3); a value outside it is
# held at the nearest end, alike on every node, live or rebuilt.
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
# A question that holds no item: its holder-side call was not started (cut by
# the challenger's time or dollar budget), or the whole challenge is void. It
# is published as the int32 minimum, which no real item takes (real items are
# held within INT32_MIN + 1 .. INT32_MAX).
NO_ITEM = INT32_MIN
# The holder-side failure cap and the void epochs a run passes through
# (config.HOLDER_FAIL_CAP_BP_V4, VOID_MAX_IN_RUN_V4).
HOLDER_FAIL_CAP_BP = config.HOLDER_FAIL_CAP_BP_V4
VOID_MAX_IN_RUN = config.VOID_MAX_IN_RUN_V4


def value_per_correct() -> Fraction:
    """V, dollars a correct answer is worth: 1 / BENCHMARK_LAMBDA, read from
    routing_protocol and nowhere else (SEAT_MINIMAL_V4.md 3.4)."""
    from fugal_subnet.routing_protocol import BENCHMARK_LAMBDA

    lam = Fraction(BENCHMARK_LAMBDA)
    if lam <= 0:
        raise ValueError("BENCHMARK_LAMBDA must be positive")
    return 1 / lam


# --- the failure bound ------------------------------------------------------------------


def over_bound(failed: int, calls: int) -> bool:
    """More than FAIL_MAX_BP of `calls` failed (exactly: failed x BP > bound x calls)."""
    return calls > 0 and failed * BP > FAIL_MAX_BP * calls


@dataclass(frozen=True)
class FailureVerdict:
    """The epoch's failure bound: whether it is an outage epoch, and the
    hotkeys whose proofs are not counted, each with (failed, calls) of its
    exploration. Nothing is refused in an outage epoch."""

    outage: bool
    refused: Mapping[str, tuple[int, int]] = field(default_factory=dict)


def failure_refusals(proofs: Iterable, state) -> FailureVerdict:
    """The absolute exploration failure bound over the epoch's verified v4
    proofs (`menu_inputs.VerifiedProof`, each with the chain's coldkey) on
    the epoch's MenuState `state`. The outage is `menu_inputs.outage` of
    these proofs' exploration calls (per coldkey, summed over its hotkeys):
    the definition the menu step's health rule reads."""
    from fugal_subnet import menu_inputs as mi

    proofs = mi._sorted_proofs(proofs)
    provisional = mi.provisional_ids(state)
    summary = mi.EpochSummary(epoch=int(state.epoch), votes=(),
                              calls=mi.exploration_calls(proofs, state), pairs=(), use=(),
                              provisional=provisional)
    if mi.outage(summary):
        return FailureVerdict(True, {})
    refused = {}
    for vp in proofs:
        # A provisional entry's calls count against its own answered share only.
        ex = [r for r in vp.proof.exploration_results if r.routed_model not in provisional]
        failed = sum(1 for r in ex if r.failed)
        if over_bound(failed, len(ex)):
            refused[vp.hotkey] = (failed, len(ex))
    return FailureVerdict(False, refused)


# --- the challenge's items --------------------------------------------------------------


def call_value(grade: Fraction, cost: Fraction) -> int:
    """One call's value, 100 x (grade - cost / V) points, in units of
    1 / UNITS_PER_POINT of a point, rounded half to even (so the value of -x
    is minus the value of x)."""
    return round(Fraction(grade) * (100 * UNITS_PER_POINT)
                 - Fraction(cost) / value_per_correct() * (100 * UNITS_PER_POINT))


def _clamp32(x: int) -> int:
    return INT32_MIN + 1 if x <= INT32_MIN else INT32_MAX if x > INT32_MAX else x


@dataclass(frozen=True)
class Answer:
    """One challenge call as the proof reports it: the entry called, its
    grade, whether it failed, and whether it was started at all (a
    holder-side call the challenger's TD did not start is reported failed
    and not started)."""

    entry: str
    correct: bool
    failed: bool
    started: bool = True


def challenge_items(state, categories: Sequence[str], own: Sequence[Answer],
                    holder_picks: Sequence[str | None],
                    holder: Mapping[int, Answer]) -> tuple[int, ...]:
    """A challenge's items in question order (the module docstring): `own`
    the challenger's call on each question, `holder_picks` the holder's
    computed pick on each (None: no entry it has odds for), `holder` the
    holder-side call by question position wherever the two picks differ.
    `state` is the epoch's MenuState (its pick-cost table)."""
    from fugal_subnet import menu_state as ms

    if not (len(categories) == len(own) == len(holder_picks)):
        raise ValueError("a challenge's answers do not align with its questions")
    memo: dict[tuple, int] = {}

    def value(grade: int, entry: str, cat: str) -> int:
        # a call's value depends only on its grade, entry and category
        key = (grade, entry, cat)
        if key not in memo:
            memo[key] = call_value(Fraction(grade), ms.pick_cost(state, entry, cat))
        return memo[key]

    out = []
    for i, (cat, mine, hp) in enumerate(zip(categories, own, holder_picks)):
        if hp == mine.entry:
            out.append(0)
            continue
        v_c = value(0 if mine.failed else int(bool(mine.correct)), mine.entry, cat)
        if hp is None:
            out.append(_clamp32(v_c))
            continue
        h = holder.get(i)
        if h is None or h.entry != hp:
            raise ValueError(f"question {i}: no holder-side call to its pick {hp!r}")
        if not h.started:
            out.append(NO_ITEM)          # not made: no item at all
            continue
        v_h = value(1 if h.failed else int(bool(h.correct)), hp, cat)
        out.append(_clamp32(v_c - v_h))
    return tuple(out)


def challenge_verdict(holder: Mapping[int, Answer]) -> str | None:
    """None when a challenge counts, else why it is void: more than
    HOLDER_FAIL_CAP_BP of its holder-side calls (one per question where the
    two picks differ) failed or were not started. A void challenge is
    counted for and against nobody (`void_items`)."""
    planned = len(holder)
    bad = sum(1 for h in holder.values() if h.failed or not h.started)
    if planned and bad * BP > HOLDER_FAIL_CAP_BP * planned:
        not_started = sum(1 for h in holder.values() if not h.started)
        return (f"{bad} of its {planned} holder-side calls failed or were not made "
                f"({not_started} not made), more than {HOLDER_FAIL_CAP_BP // 100}%")
    return None


def void_items(n: int) -> tuple[int, ...]:
    """The items of a void challenge of `n` questions: none counts."""
    return (NO_ITEM,) * int(n)


def is_void(items: Sequence[int]) -> bool:
    """Whether a challenge's items are a void challenge's (no item counts)."""
    return len(items) > 0 and all(x == NO_ITEM for x in items)


def counted(items: Sequence[int]) -> int:
    """How many of a challenge's items count (every one but NO_ITEM)."""
    return sum(1 for x in items if x != NO_ITEM)


def challenge_picks(head, state, hidden) -> tuple[str | None, ...]:
    """A head's picks in a challenge: head_odds.picks_and_odds_v4 on the
    epoch's menu over its FULL entries only. A provisional entry is never a
    challenge pick, on either side: it has no measured rate, and its calls
    run under the provisional deadline and budget, which the holder could
    otherwise choose for its own side. The challenger's TD, every validator
    and the published `holder_picks` compute them alike; None where the head
    has odds for no full entry."""
    from fugal_subnet import menu_state as ms
    from fugal_subnet.head_odds import picks_and_odds_v4

    not_full = frozenset(e.id for e in state.entries if not ms.is_full(e))
    return picks_and_odds_v4(head, state, hidden, exclude=not_full).picks


def call_order_key(nonce: bytes, question_id: str) -> bytes:
    """The order a challenger's TD starts a challenge's holder-side calls in
    (and its exploration's): HMAC-SHA256 of the epoch nonce over
    "order:<question id>". The holder-side calls it did not start are always
    the last in this order (verify._v4_scored_check holds a proof to it)."""
    import hmac

    return hmac.new(bytes(nonce), f"order:{question_id}".encode("utf-8"),
                    hashlib.sha256).digest()


def item_means(items: Sequence[int], own_entries: Sequence[str],
               holder_picks: Sequence[str | None]) -> dict[str, tuple[int, int]]:
    """{entry: (n, sum)} over the items whose two picks differ, by the
    challenger's entry. Summed over an epoch's counted challenges it is the
    mean challenge item per entry the reveal publishes (FAST_CROWN_V4.md 0:
    an epoch-wide shift can be seen, and is not guarded)."""
    out: dict[str, list[int]] = {}
    for x, e, hp in zip(items, own_entries, holder_picks):
        if hp == e or x == NO_ITEM:
            continue
        acc = out.setdefault(e, [0, 0])
        acc[0] += 1
        acc[1] += int(x)
    return {e: (n, t) for e, (n, t) in sorted(out.items())}


# --- the holder's routing against R's: published, deciding nothing ------------------------


def routing_vs_r(state, pairs, window, holder, hidden_of, *, now: int,
                 window_epochs: int = config.SEAT_WINDOW_EPOCHS_V4) -> dict:
    """The holder's routing value against the reference R's routing on the
    epoch's pooled labels, off-policy (FAST_CROWN_V4.md 19: published every
    epoch with its bound, never a rule). A label is the holder's when its
    entry is the holder's computed pick on its question, and R's when its
    entry is R's pick for its benchmark and category: the entry maximising
    R's rate (each entry's rate on the benchmark over the window's pooled
    labels before this epoch, shrunk to its overall rate with prior
    ODDS_REF_PRIOR, as odds_score reads R) minus its pick cost, the first in
    menu order on a tie. Each label's value is 100 x (y - t / V) points in
    units. Published: [n, sum, sum of squares] per side, and `lower`, the
    difference of the two means less sqrt(A_ALPHA x (s_h^2 / n_h +
    s_r^2 / n_r)), rounded down (None with fewer than two labels a side).
    The two sides are different labels, so the items are unpaired."""
    from fugal_subnet import menu_state as ms
    from fugal_subnet.head_odds import ODDS_SCALE
    from fugal_subnet.labels_v4 import PROVISIONAL_LABEL
    from fugal_subnet.seat_bound import A_ALPHA, _ceil_sqrt

    ids = list(state.ids())
    on_menu = set(ids)
    # A provisional entry has no rate the field measured yet and its labels
    # score nobody (labels_v4): R never picks one, and no provisional label is
    # read on either side.
    full = [m for m in ids if not state.entry(m).provisional]

    def counted(ep) -> list[bool]:
        missing = getattr(ep, "missing", None)
        return ([True] * len(ep.question_ids) if missing is None
                else (np.asarray(missing) != PROVISIONAL_LABEL).tolist())

    k_mb: dict[tuple[str, str], int] = {}
    n_mb: dict[tuple[str, str], int] = {}
    k_m: dict[str, int] = {}
    n_m: dict[str, int] = {}
    for e, ep in sorted(window.pairs.items()):
        if not (int(now) - int(window_epochs) < e < int(now)):
            continue
        for m, b, y, ok in zip(ep.models, ep.benchmarks, ep.y.tolist(), counted(ep)):
            if not ok:
                continue
            k_mb[(m, b)] = k_mb.get((m, b), 0) + int(y)
            n_mb[(m, b)] = n_mb.get((m, b), 0) + 1
            k_m[m] = k_m.get(m, 0) + int(y)
            n_m[m] = n_m.get(m, 0) + 1
    prior = int(config.ODDS_REF_PRIOR)

    def rate(m: str, b: str) -> Fraction:
        r_m = (Fraction(k_m.get(m, 0), ODDS_SCALE) + 1) / (n_m.get(m, 0) + 2)
        nc = n_mb.get((m, b), 0)
        return (Fraction(k_mb.get((m, b), 0), ODDS_SCALE) + prior * r_m) / (nc + prior)

    r_pick: dict[tuple[str, str], str] = {}
    lam = 1 / value_per_correct()

    def r_pick_of(b: str, c: str) -> str:
        if (b, c) not in r_pick:
            best, best_u = None, None
            for m in full:
                u = rate(m, b) - lam * ms.pick_cost(state, m, c)
                if best_u is None or u > best_u:
                    best, best_u = m, u
            r_pick[(b, c)] = best
        return r_pick[(b, c)]

    sides = {"holder": [0, 0, 0], "r": [0, 0, 0]}
    qs = list(pairs.question_ids)
    if qs:
        hp = challenge_picks(holder, state, np.stack([np.asarray(hidden_of(q)) for q in qs]))
        for i, (m, b, c, y, ok) in enumerate(zip(pairs.models, pairs.benchmarks, pairs.categories,
                                                 pairs.y.tolist(), counted(pairs))):
            if m not in on_menu or not ok:
                continue
            v = call_value(Fraction(int(y), ODDS_SCALE), ms.pick_cost(state, m, c))
            for side, hit in (("holder", m == hp[i]), ("r", m == r_pick_of(b, c))):
                if hit:
                    acc = sides[side]
                    acc[0] += 1
                    acc[1] += v
                    acc[2] += v * v
    (nh, sh, qh), (nr, sr, qr) = sides["holder"], sides["r"]
    lower = None
    if nh >= 2 and nr >= 2:
        var = (Fraction(qh * nh - sh * sh, nh * nh * (nh - 1))
               + Fraction(qr * nr - sr * sr, nr * nr * (nr - 1)))
        diff = Fraction(sh, nh) - Fraction(sr, nr)
        lower = (diff - _ceil_sqrt(var * A_ALPHA)).__floor__()
    return {"holder": sides["holder"], "r": sides["r"], "lower": lower}


# --- the record of an epoch -------------------------------------------------------------


@dataclass(frozen=True)
class PicksEpoch:
    """What one v4 epoch contributes to rule B: the categories of the
    epoch's challenge list (list order), the holder its challenges paired
    with (None: there was none), and each head with a counted challenge with
    its items on a prefix of the list (`challenge_items`). A head not in
    `values` has no counted challenge this epoch. `outage` is the epoch's
    outage flag."""

    epoch: int
    categories: tuple[str, ...]
    values: Mapping[str, tuple[int, ...]]
    outage: bool = False
    holder: str | None = None

    def __post_init__(self):
        q = len(self.categories)
        for h, v in self.values.items():
            if not 0 < len(v) <= q or not all(type(x) is int for x in v):
                raise ValueError(f"head {h[:12]}: items must be one int per question of a "
                                 "prefix of the challenge list")


# --- the answered run and rule B ----------------------------------------------------------


def answered_in(records: Mapping[int, PicksEpoch], head: str, epoch: int) -> bool:
    """Whether `head` has a counted challenge in `epoch` (False when the
    epoch is not recorded, and for a void challenge): what the seat reads
    for qualifiers and for the contenders that bar later heads."""
    rec = records.get(int(epoch))
    return rec is not None and head in rec.values and not is_void(rec.values[head])


def answered_run(records: Mapping[int, PicksEpoch], head: str, through: int) -> tuple[int, ...]:
    """`head`'s answered run: the recorded epochs up to `through` in which it
    has a counted or a void challenge, back from the newest recorded one to
    the first it missed, oldest first. Empty when it missed the newest. A
    void challenge neither counts nor breaks the run, but a run passes
    through at most VOID_MAX_IN_RUN of them: the next one back ends it, as
    a gap does."""
    run = []
    voids = 0
    for e in sorted((e for e in records if e <= int(through)), reverse=True):
        if head not in records[e].values:
            break
        if is_void(records[e].values[head]):
            voids += 1
            if voids > VOID_MAX_IN_RUN:
                break
        run.append(e)
    return tuple(sorted(run))


@dataclass(frozen=True)
class PicksEvidence:
    """Rule B for one head against a bar (None: the open-seat count only).
    `run` the head's answered run, `questions` the challenge items in it,
    `paired_epochs` and `paired` the run's epochs and items paired with the
    bar, `bound` the bound on the paired items (None with no bar)."""

    head: str
    bar: str | None
    run: tuple[int, ...]
    questions: int
    paired_epochs: int
    paired: int
    bound: Bound | None
    passes: bool

    @property
    def lower(self) -> int | None:
        return None if self.bound is None else self.bound.lower


def picks_items(records: Mapping[int, PicksEpoch], head: str, bar: str,
                run: Sequence[int]) -> dict[str, list[int]]:
    """The paired items of rule B, by category: `head`'s challenge items in
    the run's epochs whose challenges paired with `bar`."""
    items: dict[str, list[int]] = {}
    for e in sorted(run):
        rec = records[e]
        if head not in rec.values or rec.holder != bar:
            continue
        for cat, x in zip(rec.categories, rec.values[head]):
            if x != NO_ITEM:
                items.setdefault(cat, []).append(x)
    return items


def picks_check(records: Mapping[int, PicksEpoch], head: str, bar: str | None, *, through: int,
                margin_units: int = PICKS_MARGIN_UNITS, min_questions: int = PICKS_MIN_QUESTIONS
                ) -> PicksEvidence:
    """Rule B (FAST_CROWN_V4.md 2): does `head` pass the real-picks check
    against `bar` at epoch `through`? With a bar: the bound of its challenge
    items paired with the bar passes with `margin_units` (-1.5 points) on at
    least `min_questions` (500) items, n0 = BOUND_N0 (100). With no bar (an
    open seat): its answered run holds at least `min_questions` items."""
    run = answered_run(records, head, through)
    questions = sum(counted(records[e].values[head]) for e in run)
    if bar is None or bar == head:
        return PicksEvidence(head, None, run, questions, 0, 0, None, questions >= min_questions)
    items = picks_items(records, head, bar, run)
    paired_epochs = sum(1 for e in run if records[e].holder == bar)
    b = lower_bound(items, margin_units=margin_units, n_min=min_questions, n0=BOUND_N0)
    return PicksEvidence(head, bar, run, questions, paired_epochs, b.n, b, b.passes)


def open_seat_check(records: Mapping[int, PicksEpoch], head: str, last_holder: str | None, *,
                    through: int) -> PicksEvidence:
    """Rule B when the seat is open (SEAT_MINIMAL_V4.md 3.4): against the
    last holder when the head has at least PICKS_MIN_QUESTIONS items paired
    with it, else only that its answered run holds that many."""
    if last_holder is not None and last_holder != head:
        ev = picks_check(records, head, last_holder, through=through)
        if ev.paired >= PICKS_MIN_QUESTIONS:
            return ev
    return picks_check(records, head, None, through=through)


# --- published form -----------------------------------------------------------------------


def values_b64(values: Sequence[int]) -> str:
    """A challenger's items as published: little-endian int32, base64."""
    return base64.b64encode(np.asarray(list(values), dtype="<i4").tobytes()).decode("ascii")


def values_from_b64(text: str, n: int | None = None) -> tuple[int, ...]:
    """Inverse of `values_b64`, validated: a challenge size's worth of items
    (or exactly `n`)."""
    raw = base64.b64decode(str(text), validate=True)
    k, rest = divmod(len(raw), 4)
    if rest or (n is not None and k != n) or (n is None and k not in config.CHALLENGE_SIZES_V4):
        raise ValueError(f"challenge items hold {len(raw)} bytes, not a challenge size's worth")
    return tuple(int(x) for x in np.frombuffer(raw, dtype="<i4"))


def evidence_to_reveal(ev: PicksEvidence, *, answered_now: bool) -> dict:
    """One head's rule-B evidence as published: its run (first epoch and
    length), the items in it, whether it has a counted challenge this epoch,
    the bar and the paired count, and the bound's verdict and `lower`."""
    return {"answered": bool(answered_now), "run_from": ev.run[0] if ev.run else None,
            "run_epochs": len(ev.run), "questions": ev.questions, "bar": ev.bar,
            "paired_epochs": ev.paired_epochs, "paired": ev.paired,
            "passes": ev.passes, "lower": ev.lower}


def reveal_block(rec: PicksEpoch, evidence: Mapping[str, dict],
                 refused: Mapping[str, tuple[int, int]] | None = None, *,
                 questions: Sequence[str] = (), means: Mapping[str, tuple[int, int]] | None = None,
                 cells: Mapping[str, Sequence[int]] | None = None,
                 holder_picks: Mapping | None = None,
                 challenges_refused: Mapping[str, str] | None = None,
                 routing_vs_r: Mapping | None = None,
                 void: Mapping[str, str] | None = None) -> dict:
    """The epoch's `picks_v4` reveal field (FAST_CROWN_V4.md 3, 6, 19): the
    outage flag; the hotkeys the failure bound did not count; the holder the
    challenges paired with; the challenge list as [id, category] (the
    prompts past the slice are not published); every counted challenger's
    items as int32; the mean item per entry as [n, sum]; the challenge calls
    the slice's shared cells do not hold, summed per "entry category" as
    [calls, correct, failed]; `holder_picks`, the seat holder's computed
    pick on every question of the list (`holder_picks_doc`: the product's
    parity check; empty with no holder); the challenges refused and why;
    the void challenges and why (`void`: their items are all NO_ITEM); the
    holder's
    routing value against R's routing (published only: it decides nothing);
    and every serving head's evidence. Sorted keys."""
    if questions and len(questions) != len(rec.categories):
        raise ValueError("the challenge list's ids do not align with its categories")
    return {
        "outage": rec.outage,
        "failure_refused": {hk: list(refused[hk]) for hk in sorted(refused or {})},
        "holder": rec.holder,
        "list": [[q, c] for q, c in zip(questions, rec.categories)],
        "items": {h: values_b64(rec.values[h]) for h in sorted(rec.values)},
        "item_means": {e: list(v) for e, v in sorted((means or {}).items())},
        "cells": {k: list(v) for k, v in sorted((cells or {}).items())},
        "holder_picks": dict(holder_picks or {}),
        "refused": {h: why for h, why in sorted((challenges_refused or {}).items())},
        "void": {h: why for h, why in sorted((void or {}).items())},
        "routing_vs_r": dict(routing_vs_r or {}),
        "evidence": {h: evidence[h] for h in sorted(evidence)},
    }


def holder_picks_doc(head: str, picks: Sequence[str | None], models: Sequence[str]) -> dict:
    """The seat holder's picks as published (`picks_v4.holder_picks`):
    {"head": its hash, "picks": base64 of one little-endian int8 per question
    of the epoch's challenge list (`picks_v4.list`, in list order), each the
    index of the picked entry in the reveal's `models`, -1 where the head
    picks nothing (it has odds for no entry on the menu)}. The picks are
    head_odds.picks_and_odds_v4 on the epoch's menu and pick-cost table, as
    rule B computes the holder's side."""
    index = {m: i for i, m in enumerate(models)}
    if len(models) > 127:
        raise ValueError("more than 127 models: a pick does not fit an int8")
    out = []
    for p in picks:
        if p is not None and p not in index:
            raise ValueError(f"the holder picks {p!r}, which the reveal's models do not list")
        out.append(-1 if p is None else index[p])
    raw = np.asarray(out, dtype="<i1").tobytes()
    return {"head": head, "picks": base64.b64encode(raw).decode("ascii")}


def picks_from_doc(doc: Mapping, models: Sequence[str]) -> tuple[str, tuple[str | None, ...]]:
    """Inverse of `holder_picks_doc`: (head, entry or None per question)."""
    raw = base64.b64decode(str(doc["picks"]), validate=True)
    picks = []
    for i in np.frombuffer(raw, dtype="<i1").tolist():
        if i < -1 or i >= len(models):
            raise ValueError("a published pick names no model of the reveal")
        picks.append(None if i == -1 else models[i])
    return str(doc["head"]), tuple(picks)


def digest(block: Mapping) -> str:
    """One hash over the epoch's published `picks_v4` field, for `v2_state`
    (the consensus view two validators are compared on)."""
    canonical = json.dumps(block, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_HEX = frozenset("0123456789abcdef")
_MAX_ID = 256


def epoch_from_reveal(block, epoch: int, categories: Sequence[str] | None = None) -> PicksEpoch:
    """The epoch's record from its published `picks_v4` field, strictly: the
    categories are the published challenge list's, each challenger's items a
    challenge size's worth of int32 within the list, the holder a head hash
    or null. `categories` is not read (callers pass the slice's)."""
    if not isinstance(block, Mapping):
        raise ValueError("`picks_v4` must be an object")
    outage, values, lst, holder = (block.get("outage"), block.get("items"), block.get("list"),
                                   block.get("holder"))
    if not isinstance(outage, bool) or not isinstance(values, Mapping):
        raise ValueError("`picks_v4` needs a boolean `outage` and an `items` map")
    if (not isinstance(lst, list) or len(lst) > config.CHALLENGE_LIST_SIZE_V4
            or not all(isinstance(x, list) and len(x) == 2
                       and all(isinstance(y, str) and 0 < len(y) <= _MAX_ID for y in x)
                       for x in lst)
            or len({x[0] for x in lst}) != len(lst)):
        raise ValueError("`picks_v4.list` must be distinct [question id, category] pairs")
    if holder is not None and not (isinstance(holder, str) and len(holder) == 64
                                   and set(holder) <= _HEX):
        raise ValueError("`picks_v4.holder` must be a head hash or null")
    if list(values) != sorted(values):
        raise ValueError("`picks_v4.items` must be sorted by head")
    void = block.get("void", {})
    if not isinstance(void, Mapping) or not all(isinstance(v, str) for v in void.values()):
        raise ValueError("`picks_v4.void` must map heads to reasons")
    if set(void) - set(values):
        raise ValueError("`picks_v4.void` names a head with no items")
    cats = tuple(x[1] for x in lst)
    items = {}
    for h, v in values.items():
        got = values_from_b64(v)
        if len(got) > len(cats):
            raise ValueError(f"head {str(h)[:12]}: more items than the challenge list holds")
        if is_void(got) != (h in void):
            raise ValueError(f"head {str(h)[:12]}: its items and `picks_v4.void` disagree")
        items[str(h)] = got
    return PicksEpoch(int(epoch), cats, items, outage, holder)


__all__ = [
    "Answer", "FailureVerdict", "PicksEpoch", "PicksEvidence", "answered_in", "answered_run",
    "NO_ITEM", "call_order_key", "call_value", "challenge_items", "challenge_picks",
    "challenge_verdict", "counted", "digest", "is_void", "void_items", "epoch_from_reveal", "evidence_to_reveal",
    "holder_picks_doc", "picks_from_doc",
    "failure_refusals", "item_means", "open_seat_check", "over_bound", "picks_check",
    "picks_items", "reveal_block", "routing_vs_r", "value_per_correct", "values_b64",
    "values_from_b64",
]
