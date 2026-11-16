"""Protocol v2 champion scoring: the paired net-value test (I3, I1).

From the activation block the miner share goes to a CHAMPION, and a head takes
the seat only by beating the champion on the same questions:

    NV_i = u_i * (g_challenger,i - g_champion,i) - (s_challenger,i - s_champion,i) / V
    V    = CHAMPION_ACCURACY_VALUE * the pinned anchor's expected spend per
           question (config.CHAMPION_ANCHOR_MODEL; the champion's own spend
           only when no anchor is given, which protocol v2 never does)
    u_i  = CHAMPION_HARD_WEIGHT on a hard question (the hard-reasoning
           category), else 1
    NV   = 100 * mean_i NV_i                       (percentage points)

g is the shared cell grade (fugal_subnet/shared_grades.py) and s the policy's
spend for the question: the pool's prompt tokens plus the cell's coldkey-median
attested completion tokens at the pinned prices (never the frame's verbosity),
so neither side of NV is a miner's self-report. V comes from pinned files
(the v2 token manifest and price table), so every validator prices the same.

The challenger takes the seat when BOTH its point estimate and the one-sided
paired-bootstrap lower bound of NV exceed CHAMPION_MARGIN points; the
champion keeps it with no margin. The bootstrap resamples questions with
replacement within category, keeping both heads' answers to a question
together, which is what makes the comparison paired.

FRESH-DATA CONFIRMATION (from config.CONFIRM_ACTIVATION_BLOCK). Passing that
test once does not take the seat from a settled champion. The epoch a
challenger first passes is recorded (a pending record in the seat state), and
it takes the seat only when it passes the same test again on questions drawn
AFTER that epoch: at least CHAMPION_CONFIRM_QUESTIONS of them, while the
first pass is still inside the window. The champion holds and is paid meanwhile. The
first pass is a selection: the test is re-run every epoch for every
challenger, so with k challengers that have no real edge somebody passes by
luck about k times as often as one would. The second pass is on data the
selection never saw, so luck has to strike twice.

DETERMINISM. Two validators must reach the same verdict on every CPU. Each
NV_i is computed elementwise (exact IEEE operations, no reductions) and then
quantised to an integer number of NV_UNIT points, so every sum below is an
exact integer sum, independent of summation order or SIMD width. V is a
math.fsum mean, which is correctly rounded and order-independent. The
resampling stream is numpy's PCG64 seeded from the epoch and both head
hashes, so it is identical everywhere and different for every pair and
epoch. The lower bound is an order statistic of integer sums, so it is exact
too, and the margin comparison is done in the same integer units.
"""
from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

# Integer units per NV point. 1e-6 of a point is far below any decision the
# margin (0.2 points) makes, and keeps a 2,000 x 15,000-question resample sum
# well inside int64.
NV_UNITS_PER_POINT = 1_000_000
# Resamples are drawn in blocks of this many to bound memory.
_BLOCK = 100
# Per-question NV is clipped to +/- this many units (a million points) before
# it becomes an integer. Real values are within a few hundred points; the clip
# only stops a pathological spend ratio from overflowing int64 in a sum.
UNIT_CLIP = 10**12
# Former champions kept in the seat state, most recent first. The payout reads
# two; a few more keep the record readable without letting it grow.
HISTORY_KEEP = 4


def question_nv_units(
    g_challenger: float,
    g_champion: float,
    s_challenger: float,
    s_champion: float,
    *,
    hard: bool,
    value_per_correct: float,
    hard_weight: float,
) -> int:
    """NV of one question, in integer NV_UNITS_PER_POINT units of a point."""
    if value_per_correct <= 0:
        raise ValueError(f"value_per_correct must be positive, got {value_per_correct!r}")
    u = hard_weight if hard else 1.0
    nv = u * (float(g_challenger) - float(g_champion)) - (
        float(s_challenger) - float(s_champion)
    ) / float(value_per_correct)
    x = 100.0 * nv * NV_UNITS_PER_POINT
    if x != x:
        raise ValueError("NV is not a number")
    return int(round(max(-UNIT_CLIP, min(UNIT_CLIP, x))))


def pair_seed(epoch_id: str, champion_hash: str, challenger_hash: str) -> int:
    """The resampling seed for one (champion, challenger) pair in one epoch."""
    material = f"fugal-v2-champion:{epoch_id}:{champion_hash}:{challenger_hash}".encode()
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


@dataclass(frozen=True)
class PairedResult:
    n_questions: int
    mean_units: int          # floor of the exact mean, in NV units
    lcb_units: int           # floor of the lower-bound resample mean, in NV units
    takes_seat: bool

    @property
    def mean_points(self) -> float:
        return self.mean_units / NV_UNITS_PER_POINT

    @property
    def lcb_points(self) -> float:
        return self.lcb_units / NV_UNITS_PER_POINT


def paired_test(
    units_by_category: Mapping[str, np.ndarray | list[int]],
    *,
    margin_points: float,
    confidence: float,
    resamples: int,
    seed: int,
) -> PairedResult:
    """Does the challenger beat the champion by the margin? Exact and deterministic.

    `units_by_category` holds each category's per-question NV in integer units
    (question_nv_units). Categories are visited in sorted order, so the result
    does not depend on mapping order.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if resamples < 1:
        raise ValueError(f"resamples must be positive, got {resamples!r}")
    cats = [
        (name, np.asarray(units_by_category[name], dtype=np.int64))
        for name in sorted(units_by_category)
    ]
    cats = [(name, x) for name, x in cats if x.size]
    n = int(sum(x.size for _, x in cats))
    if n == 0:
        return PairedResult(0, 0, 0, False)

    total = int(sum(int(x.sum()) for _, x in cats))
    margin_sum = round(float(margin_points) * NV_UNITS_PER_POINT) * n
    mean_units = total // n
    # The point estimate is a necessary condition, decided before any
    # resampling: a challenger whose mean does not clear the margin does not
    # take the seat, and pays no bootstrap.
    if total <= margin_sum:
        return PairedResult(n, mean_units, mean_units, False)

    rng = np.random.Generator(np.random.PCG64(seed))
    sums = np.zeros(int(resamples), dtype=np.int64)
    for start in range(0, int(resamples), _BLOCK):
        stop = min(start + _BLOCK, int(resamples))
        for _name, x in cats:
            idx = rng.integers(0, x.size, size=(stop - start, x.size))
            sums[start:stop] += x[idx].sum(axis=1)
    sums.sort(kind="stable")
    # One-sided lower bound at `confidence`: the k-th smallest resample sum,
    # k = round((1 - confidence) * resamples), 0-based (200 of 2,000 at 90%).
    # round, not int: (1 - 0.9) * 2000 is 199.99999999999997 in floating point.
    k = min(round((1.0 - float(confidence)) * int(resamples)), int(resamples) - 1)
    lcb_sum = int(sums[k])
    return PairedResult(n, mean_units, lcb_sum // n, lcb_sum > margin_sum)


# ---------------------------------------------------------------------------
# The comparison window: what every head answered, epoch by epoch.
#
# Every miner answers the SAME scored slice in an epoch, so two heads are
# paired question by question within each epoch they were both scored. The
# window keeps the last CHAMPION_WINDOW_EPOCHS epochs by calendar. Per epoch it
# holds what is common to every head (the slice, each question's category and
# hard flag, the shared-cell grade and the policy spend of every (question,
# model)) and per head only which model it routed each question to and which of
# its calls failed. Everything in it is in the published reveal, so the window
# is rebuilt from reveals exactly.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochSlice:
    """What one epoch's scored slice is worth, for every model in the pool.

    Arrays are (questions, models) in the global model index space. A grade is
    NaN where no operator routed that question to that model; a head's own
    answer always has a cell, so a NaN is never read for a head.
    """

    epoch: int
    question_ids: tuple[str, ...]
    categories: tuple[str, ...]
    hard: tuple[bool, ...]
    grades: np.ndarray
    spends: np.ndarray

    def __post_init__(self):
        q = len(self.question_ids)
        if len(self.categories) != q or len(self.hard) != q:
            raise ValueError("categories and hard must align with question_ids")
        for name in ("grades", "spends"):
            arr = getattr(self, name)
            if arr.ndim != 2 or arr.shape[0] != q:
                raise ValueError(f"{name} must be (questions, models), got {arr.shape}")
        if self.spends.shape != self.grades.shape:
            raise ValueError("grades and spends must share a shape")


@dataclass(frozen=True)
class HeadEpoch:
    """One head's answers in one epoch, aligned to that epoch's slice."""

    routed: np.ndarray   # (questions,) int model index
    failed: np.ndarray   # (questions,) bool


@dataclass(frozen=True)
class EpochPairs:
    """One epoch's pooled exploration pairs (protocol v3; explore_pool.py).

    Aligned tuples and arrays, in (question, model) order. `y` is each pair's
    outcome in ODDS_SCALE units; the token arrays are the attested medians.
    """

    question_ids: tuple[str, ...]
    models: tuple[str, ...]
    benchmarks: tuple[str, ...]
    categories: tuple[str, ...]
    y: np.ndarray                 # (pairs,) int32
    prompt_tokens: np.ndarray     # (pairs,) int32
    completion_tokens: np.ndarray  # (pairs,) int32

    def __post_init__(self):
        n = len(self.question_ids)
        if not (len(self.models) == len(self.benchmarks) == len(self.categories) == n):
            raise ValueError("pair fields must align")
        for name in ("y", "prompt_tokens", "completion_tokens"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must hold one value per pair")


@dataclass
class Window:
    """The last `size` epochs of slices, and every head's answers in them.

    From protocol v3 it also holds each epoch's pooled exploration pairs and
    every scored head's odds on them (`pairs`, `odds`: int16 per pair, -1
    where the head has no row for the pair's model). Both are empty before v3.
    """

    size: int
    slices: dict[int, EpochSlice]
    heads: dict[str, dict[int, HeadEpoch]]
    pairs: dict[int, EpochPairs] = dataclasses.field(default_factory=dict)
    odds: dict[str, dict[int, np.ndarray]] = dataclasses.field(default_factory=dict)

    @classmethod
    def empty(cls, size: int) -> Window:
        return cls(size=int(size), slices={}, heads={})

    def add_epoch(self, sl: EpochSlice, answers: Mapping[str, HeadEpoch],
                  pairs: EpochPairs | None = None,
                  odds: Mapping[str, np.ndarray] | None = None) -> Window:
        """A new window with this epoch added and anything older than `size` dropped.

        Copy-on-write, like the rest of the epoch state (I1/I6: an epoch is
        applied once or not at all). Idempotent: adding an epoch that is
        already present REPLACES it (its slice and every head's answers in it),
        so a retry of the same epoch after a crash between writing the window
        and the JSON state, or a `--once` re-run, reaches the same window
        rather than failing the epoch.
        """
        q = len(sl.question_ids)
        m = sl.grades.shape[1]
        for h, he in answers.items():
            if he.routed.shape != (q,) or he.failed.shape != (q,):
                raise ValueError(f"head {h[:12]}: answers do not align with the slice")
            if q and (int(he.routed.min()) < 0 or int(he.routed.max()) >= m):
                raise ValueError(f"head {h[:12]}: a routed model index is outside 0..{m - 1}")
        oldest = sl.epoch - self.size + 1
        slices = {e: s for e, s in self.slices.items() if e >= oldest and e != sl.epoch}
        slices[sl.epoch] = sl
        heads: dict[str, dict[int, HeadEpoch]] = {}
        for h, per in self.heads.items():
            kept = {e: he for e, he in per.items() if e >= oldest and e != sl.epoch}
            if kept:
                heads[h] = kept
        for h, he in answers.items():
            heads.setdefault(h, {})[sl.epoch] = he
        kept_pairs = {e: p for e, p in self.pairs.items() if e >= oldest and e != sl.epoch}
        kept_odds: dict[str, dict[int, np.ndarray]] = {}
        for h, per in self.odds.items():
            kept = {e: o for e, o in per.items() if e >= oldest and e != sl.epoch}
            if kept:
                kept_odds[h] = kept
        if pairs is not None:
            kept_pairs[sl.epoch] = pairs
            for h, o in (odds or {}).items():
                if h not in answers:
                    raise ValueError(f"head {h[:12]}: odds for a head with no answers this epoch")
                if o.shape != (len(pairs.question_ids),):
                    raise ValueError(f"head {h[:12]}: odds do not align with the epoch's pairs")
                kept_odds.setdefault(h, {})[sl.epoch] = o
        elif odds:
            raise ValueError("odds were given without the epoch's pairs")
        return Window(size=self.size, slices=slices, heads=heads, pairs=kept_pairs,
                      odds=kept_odds)

    def head_vectors(self, head: str, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        """(grades, spends) of `head` on `epoch`'s slice. A failed call is 0 and 0."""
        sl = self.slices[epoch]
        he = self.heads[head][epoch]
        rows = np.arange(len(sl.question_ids))
        grades = sl.grades[rows, he.routed].astype(np.float64)
        grades = np.where(he.failed, 0.0, grades)
        # A failed call answered nothing and cost nothing: grade 0, spend 0.
        spends = np.where(he.failed, 0.0, sl.spends[rows, he.routed].astype(np.float64))
        if np.isnan(grades).any() or not np.isfinite(spends).all():
            raise ValueError(f"head {head[:12]} epoch {epoch}: an answered question has no cell")
        return grades, spends

    def epochs_of(self, head: str, since_epoch: int | None = None) -> list[int]:
        per = self.heads.get(head, {})
        return sorted(e for e in per if since_epoch is None or e >= since_epoch)

    def current_run(self, head: str, since_epoch: int | None = None) -> list[int]:
        """The head's current unbroken run: the window epochs it was scored in,
        back from the newest epoch to the first one it missed.

        A challenger is measured only on this run. Otherwise it could sit out
        epochs it did badly on (unpaid, so free) and then serve a burn-in's
        worth in a row, and be compared on epochs it chose. An epoch with no
        proofs at all is not in the window and does not break a run.
        """
        per = self.heads.get(head, {})
        run: list[int] = []
        for e in sorted(self.slices, reverse=True):
            if e not in per or (since_epoch is not None and e < since_epoch):
                break
            run.append(e)
        return sorted(run)


def _units(
    g_a: np.ndarray, g_b: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
    hard: np.ndarray, value_per_correct: float | None, hard_weight: float,
) -> np.ndarray:
    """Vectorised question_nv_units: elementwise only, so exact on any CPU.

    `value_per_correct` None drops the cost term (a bar that spent nothing has
    no price to discount by).
    """
    u = np.where(hard, float(hard_weight), 1.0)
    nv = u * (g_a - g_b)
    if value_per_correct is not None:
        nv = nv - (s_a - s_b) / float(value_per_correct)
    x = 100.0 * nv * NV_UNITS_PER_POINT
    if np.isnan(x).any():
        raise ValueError("NV is not a number")
    return np.rint(np.clip(x, -UNIT_CLIP, UNIT_CLIP)).astype(np.int64)


@dataclass(frozen=True)
class PairUnits:
    """A challenger's per-category NV over a bar, and the V it was priced at."""

    by_category: dict[str, np.ndarray]
    value_per_correct: float | None
    n_questions: int


def pair_units(
    window: Window, challenger: str, champion: str, *,
    accuracy_value: float, hard_weight: float, since_epoch: int | None = None,
    anchor_spend: float | None = None,
) -> PairUnits:
    """Per-category NV units of `challenger` over `champion`, paired by question.

    Only epochs in the challenger's current unbroken run (Window.current_run,
    from `since_epoch`, its first commit) in which the champion was also
    scored count. The champion's own missed epochs are simply absent (see the
    withholding residual in INVARIANTS). V is `accuracy_value` times
    `anchor_spend`, a pinned anchor's expected spend per question, when given
    (protocol v2 passes it); without one it falls back to the champion's own
    mean spend on exactly these questions.
    """
    import math

    both = sorted(
        set(window.current_run(challenger, since_epoch)) & set(window.epochs_of(champion))
    )
    parts = []
    for e in both:
        sl = window.slices[e]
        g_a, s_a = window.head_vectors(challenger, e)
        g_b, s_b = window.head_vectors(champion, e)
        parts.append((g_a, g_b, s_a, s_b, np.asarray(sl.hard, dtype=bool),
                      np.asarray(sl.categories)))
    if not parts:
        return PairUnits({}, None, 0)
    g_a, g_b, s_a, s_b, hard, cats = (np.concatenate(x) for x in zip(*parts))
    if anchor_spend is not None:
        if not anchor_spend > 0:
            raise ValueError(f"anchor_spend must be positive, got {anchor_spend!r}")
        value = float(accuracy_value) * float(anchor_spend)
    else:
        mean_spend = math.fsum(float(x) for x in s_b) / len(s_b)
        value = float(accuracy_value) * mean_spend if mean_spend > 0 else None
    units = _units(g_a, g_b, s_a, s_b, hard, value, hard_weight)
    return PairUnits(
        {c: units[cats == c] for c in sorted(set(cats.tolist()))}, value, int(units.size),
    )


def solo_value(window: Window, head: str, params: "SeatParams",
               since_epoch: int | None = None):
    """A head's measured net value over its current run, as an exact Fraction.

    The seat test's per-question NV units against an empty baseline (no
    answer, no spend): u * grade - spend / V, in integer units, so the sum is
    exact on any CPU and the Fraction compares exactly (I1). Used to choose
    who takes an EMPTY seat; between heads it orders the same way as the
    paired comparison does, because every head answers the same slice.
    """

    value = (float(params.accuracy_value) * float(params.anchor_spend)
             if params.anchor_spend else None)
    total, n = 0, 0
    for e in window.current_run(head, since_epoch):
        sl = window.slices[e]
        g, s = window.head_vectors(head, e)
        zero = np.zeros_like(g)
        units = _units(g, zero, s, zero, np.asarray(sl.hard, dtype=bool), value,
                       params.hard_weight)
        total += int(units.sum())
        n += int(units.size)
    return Fraction(total, n) if n else Fraction(0)


def window_accuracy(
    window: Window, head: str, since_epoch: int | None = None, hard_only: bool = False,
    run_only: bool = False,
) -> tuple[float, int]:
    """(mean shared grade, questions) of `head` over the window, exactly summed."""
    import math

    total: list[float] = []
    epochs = window.current_run(head, since_epoch) if run_only else window.epochs_of(head, since_epoch)
    for e in epochs:
        g, _ = window.head_vectors(head, e)
        if hard_only:
            g = g[np.asarray(window.slices[e].hard, dtype=bool)]
        total.extend(float(x) for x in g)
    if not total:
        return 0.0, 0
    return math.fsum(total) / len(total), len(total)


# ---------------------------------------------------------------------------
# The seat: who is champion, and how the seat changes hands.
#
#   FIRST  with no champion and no ghost, the most senior eligible head that
#          passes the high-water gates takes the empty seat; there is nothing
#          to compare it with yet. Until any head is eligible the champion
#          share burns, as in v1;
#   HOLD   a champion that is live and eligible keeps the seat with no margin;
#   TAKE   an eligible challenger takes it only by beating the bar by the
#          margin (paired_test), and passing the accuracy gates;
#   CONFIRM  (from the confirmation activation block) against a settled
#          champion, or its ghost while it has only missed, the first pass
#          only records the challenger as pending; it takes the seat when it
#          also passes on the questions drawn since. A seat filled from empty
#          is open to the plain test for a few epochs first, and a retired
#          champion's ghost never needs confirming against: it answers no
#          new questions;
#   GHOST  a champion that stops being live or eligible leaves a ghost bar for
#          CHAMPION_GHOST_OFF epochs (it missed) or CHAMPION_GHOST_RETIRE (its
#          owner serves a different head); while it stands, a challenger must
#          beat the ghost, the seat is otherwise empty and its share burns, and
#          a champion that only missed resumes its seat if it returns.
#
# The bar is never "the best of the field": what a challenger has to beat is
# the champion or its ghost and, under the copy rule (copy_rule.py), every
# seat holder SENIOR to it in the comparison set. No junior head's presence moves it
# (I4).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ghost:
    head_hash: str
    last_epoch: int          # the last epoch this ghost still bars
    reason: str              # "missed" or "retired"


@dataclass(frozen=True)
class Pending:
    """A challenger that has passed the take test and awaits confirmation."""

    head_hash: str
    since_epoch: int         # the epoch it first passed; later epochs are fresh
    bar: str                 # the champion (or its missed ghost) it passed against


@dataclass(frozen=True)
class SeatState:
    champion: str | None = None
    champion_since: int | None = None
    # Former champions, most recent first, for the payout seats.
    history: tuple[str, ...] = ()
    ghosts: tuple[Ghost, ...] = ()
    # Accuracy gates. Zero until the first champion, then its lower bound,
    # then each new champion's. They never fall while a champion or its ghost
    # stands, and reset to zero when the last ghost expires with the seat
    # empty.
    high_water: float = 0.0
    high_water_hard: float = 0.0
    # Fresh-data confirmation (empty and None before its activation block, so
    # the published seat is byte-identical to the one before the rule).
    # Challengers that passed the take test and await confirmation, by hash.
    pending: tuple[Pending, ...] = ()
    # The first epoch at which taking the seat from this champion needs
    # confirmation. A champion seated from an empty seat won it against
    # nobody, so for CHAMPION_CONFIRM_OPEN_EPOCHS it can be taken by the plain
    # test; a champion that was itself confirmed is settled at once. None on a
    # champion seated before the rule: settled.
    confirm_from: int | None = None

    def __post_init__(self):
        # Bounded: the payout reads two former champions; HISTORY_KEEP keep
        # the record readable without letting it grow for the life of the
        # subnet.
        object.__setattr__(self, "history", tuple(self.history)[:HISTORY_KEEP])
        object.__setattr__(
            self, "pending", tuple(sorted(self.pending, key=lambda p: p.head_hash)))


@dataclass(frozen=True)
class Candidate:
    """A verified head this epoch, as the seat logic needs to see it."""

    head_hash: str
    # Senior first; a tuple so the caller decides what seniority is.
    seniority: tuple
    # Live this epoch, passes the quality floor, not an exact copy: enough to
    # HOLD a seat or RESUME one inside its ghost window. Decided by the caller.
    eligible: bool
    # The first epoch the challenger's comparison may use (its first commit).
    since_epoch: int
    # Full burn-in since the last reset or miss (E''): additionally required
    # to TAKE a seat, first or by challenge. A seated champion that misses one
    # epoch restarts its burn-in but keeps (or resumes) its seat; a head that
    # is not seated must re-earn burn-in before it can take one.
    burned_in: bool = True
    # The head's real first-commit block on chain, unfloored. Breaks an EXACT
    # tie for an empty seat before seniority does: every pre-flip commit ties
    # at the seniority floor, so byte-different copies of a head (which measure
    # identically) otherwise split the first seat by the salted draw.
    first_block: int | None = None


@dataclass(frozen=True)
class OddsParams:
    """The odds objective (protocol v3, from the odds seat block). With these
    set on SeatParams the seat is decided on how well heads predict the pooled
    exploration pairs (fugal_subnet/odds_score.py) instead of on net value."""

    margin_points: float       # Brier points a challenger's lower bound must clear
    min_pairs: int             # shared fresh pairs a take needs
    nv_noninferiority: float   # net-value points a challenger may not be worse by
    ref_prior: int             # strength of the reference's shrinkage (A)
    confirm_pairs: int         # fresh pairs the confirmation needs


@dataclass(frozen=True)
class SeatParams:
    margin_points: float
    confidence: float
    resamples: int
    accuracy_value: float
    hard_weight: float
    hard_gate_min_questions: int
    ghost_off: int
    ghost_retire: int
    wilson_confidence: float
    # Absolute floor on window accuracy, for taking and holding the seat.
    min_accuracy: float = 0.0
    # A take needs at least this many paired questions against the bar.
    min_paired_questions: int = 0
    # Pinned anchor spend per question for V (v2); None = the champion's own.
    anchor_spend: float | None = None
    # Fresh-data confirmation. 0 questions = the rule is not in force (every
    # epoch before config.CONFIRM_ACTIVATION_BLOCK): a take is immediate.
    confirm_questions: int = 0
    confirm_open_epochs: int = 0
    # None: the seat is decided on net value (v2, and v3 before the odds seat
    # block). Set: on odds, with net value kept as a guard.
    odds: OddsParams | None = None


@dataclass(frozen=True)
class SeatDecision:
    state: SeatState
    champion: str | None
    event: str               # "first", "hold", "take", "ghost", "empty"
    bar: str | None          # what a challenger had to beat this epoch
    detail: tuple[tuple[str, str], ...] = ()


def _wilson_lower(p: float, n: int, confidence: float) -> float:
    from fugal_subnet.evidence import _wilson_lower_bound

    return _wilson_lower_bound(p, float(n), confidence)


def _passes_gates(
    state: SeatState, window: Window, cand: Candidate, params: SeatParams,
    detail: list[tuple[str, str]],
) -> bool:
    h = cand.head_hash
    acc, n = window_accuracy(window, h, cand.since_epoch, run_only=True)
    if n == 0:
        detail.append((h, "no scored questions in the window"))
        return False
    if acc < params.min_accuracy:
        detail.append((h, f"accuracy {acc:.4f} below the floor {params.min_accuracy:.4f}"))
        return False
    if acc < state.high_water:
        detail.append((h, f"accuracy {acc:.4f} below the high-water mark {state.high_water:.4f}"))
        return False
    acc_hard, n_hard = window_accuracy(window, h, cand.since_epoch, hard_only=True,
                                       run_only=True)
    if n_hard >= params.hard_gate_min_questions and acc_hard < state.high_water_hard:
        detail.append((h, f"hard accuracy {acc_hard:.4f} below {state.high_water_hard:.4f}"))
        return False
    return True


def _seated(state: SeatState, window: Window, winner: Candidate, epoch: int,
            history: tuple[str, ...], params: SeatParams,
            confirm_from: int | None = None) -> SeatState:
    """The state with `winner` newly seated and the high-water marks raised.

    Every pending record is dropped: each was a pass against the previous
    holder, and says nothing about this one."""
    acc, n = window_accuracy(window, winner.head_hash, winner.since_epoch, run_only=True)
    acc_h, n_h = window_accuracy(window, winner.head_hash, winner.since_epoch, hard_only=True,
                                 run_only=True)
    hw = max(state.high_water, _wilson_lower(acc, n, params.wilson_confidence))
    hw_h = state.high_water_hard
    if n_h >= params.hard_gate_min_questions:
        hw_h = max(hw_h, _wilson_lower(acc_h, n_h, params.wilson_confidence))
    return SeatState(
        champion=winner.head_hash, champion_since=epoch,
        history=tuple(x for x in history if x != winner.head_hash),
        ghosts=(),  # the new champion beat every standing bar, or none stood
        high_water=hw, high_water_hard=hw_h,
        pending=(), confirm_from=confirm_from,
    )


def _challenge(
    window: Window, c: Candidate, bar: str, epoch_id: str, params: SeatParams,
    detail: list[tuple[str, str]], label: str = "", *, fresh_after: int | None = None,
    ref=None,
) -> int | None:
    """The paired test of `c` over `bar`: its mean NV units if it beats the bar
    by the margin on at least the minimum paired questions, else None.

    `fresh_after` runs the confirmation instead: the same test, on the
    questions of epochs after `fresh_after` only (the epoch `c` first passed
    in), needing `confirm_questions` of them. Its resampling stream is its
    own, so it shares no draw with the test that selected `c`.
    """
    fresh = fresh_after is not None
    since = max(int(c.since_epoch), int(fresh_after) + 1) if fresh else c.since_epoch
    if params.odds is not None:
        return _challenge_odds(window, c, bar, epoch_id, params, detail, label, since, fresh, ref)
    need = params.confirm_questions if fresh else params.min_paired_questions
    pu = pair_units(window, c.head_hash, bar, accuracy_value=params.accuracy_value,
                    hard_weight=params.hard_weight, since_epoch=since,
                    anchor_spend=params.anchor_spend)
    res = paired_test(
        pu.by_category, margin_points=params.margin_points,
        confidence=params.confidence, resamples=params.resamples,
        seed=pair_seed(f"{epoch_id}:confirm" if fresh else epoch_id, bar, c.head_hash),
    )
    v = "none" if pu.value_per_correct is None else f"{pu.value_per_correct:.6f}"
    if fresh:
        label = "fresh questions,"
    prefix = f"vs {label} " if label else "vs "
    detail.append((c.head_hash, f"{prefix}{bar[:16]}: n={res.n_questions} V={v} "
                                f"NV={res.mean_points:+.3f} LCB={res.lcb_points:+.3f}"))
    if res.n_questions < need:
        detail.append((c.head_hash, f"only {res.n_questions} "
                                    f"{'fresh' if fresh else 'paired'} questions; {need} needed"))
        return None
    if not res.takes_seat:
        return None
    return res.mean_units


def _challenge_odds(window: Window, c: Candidate, bar: str, epoch_id: str, params: SeatParams,
                    detail: list[tuple[str, str]], label: str, since: int, fresh: bool,
                    ref) -> int | None:
    """The odds test of `c` over `bar`: its mean Brier gain in units if it
    passes, else None. Three conditions, all on the same paired test:

    1. its odds beat the bar's by the Brier margin on the pairs both were
       scored on (at least `min_pairs`; `confirm_pairs` for a confirmation,
       which reads only pairs after the first pass);
    2. its odds beat the reference R (a rate per model and benchmark) at all:
       the same test with no margin, on its own pairs;
    3. it does not route worse: its net value over the bar on the scored
       questions has a lower bound above -`nv_noninferiority` points.

    A confirmation repeats only the first on the fresh pairs; the other two
    were just checked on its whole run by the plain test of the same epoch.
    """
    from fugal_subnet import odds_score

    o = params.odds
    tag = f"{epoch_id}:odds:confirm" if fresh else f"{epoch_id}:odds"
    prefix = "vs fresh pairs, " if fresh else (f"vs {label} " if label else "vs ")
    bu = odds_score.brier_units(window, ref, c.head_hash, bar, since)
    res = paired_test(bu.by_category, margin_points=o.margin_points,
                      confidence=params.confidence, resamples=params.resamples,
                      seed=pair_seed(tag, bar, c.head_hash))
    detail.append((c.head_hash, f"{prefix}{bar[:16]}: pairs={res.n_questions} "
                                f"Brier gain={res.mean_points:+.3f} LCB={res.lcb_points:+.3f}"))
    need = o.confirm_pairs if fresh else o.min_pairs
    if res.n_questions < need:
        detail.append((c.head_hash, f"only {res.n_questions} {'fresh' if fresh else 'shared'} "
                                    f"pairs; {need} needed"))
        return None
    if not res.takes_seat:
        return None
    if fresh:
        return res.mean_units

    su = odds_score.skill_units(window, ref, c.head_hash, since)
    skill = paired_test(su.by_category, margin_points=0.0, confidence=params.confidence,
                        resamples=params.resamples,
                        seed=pair_seed(f"{epoch_id}:skill", "reference", c.head_hash))
    detail.append((c.head_hash, f"vs the per-model, per-benchmark reference: pairs="
                                f"{skill.n_questions} skill={skill.mean_points:+.3f} "
                                f"LCB={skill.lcb_points:+.3f}"))
    if not skill.takes_seat:
        detail.append((c.head_hash, "its odds do not beat the reference"))
        return None

    pu = pair_units(window, c.head_hash, bar, accuracy_value=params.accuracy_value,
                    hard_weight=params.hard_weight, since_epoch=since,
                    anchor_spend=params.anchor_spend)
    nv = paired_test(pu.by_category, margin_points=-float(o.nv_noninferiority),
                     confidence=params.confidence, resamples=params.resamples,
                     seed=pair_seed(f"{epoch_id}:nv", bar, c.head_hash))
    detail.append((c.head_hash, f"net value vs {bar[:16]}: n={nv.n_questions} "
                                f"NV={nv.mean_points:+.3f} LCB={nv.lcb_points:+.3f}"))
    if not nv.takes_seat:
        detail.append((c.head_hash, f"routes worse than {bar[:16]} by more than "
                                    f"{o.nv_noninferiority:g} net-value points"))
        return None
    return res.mean_units


def seat_step(
    state: SeatState,
    epoch: int,
    epoch_id: str,
    window: Window,
    candidates: Mapping[str, Candidate],
    *,
    retired: frozenset[str] | set[str] = frozenset(),
    params: SeatParams,
    allow_take: bool = True,
    senior_bars: Mapping[str, Sequence[str]] | None = None,
) -> SeatDecision:
    """Advance the seat by one epoch. Pure and deterministic.

    `candidates` are this epoch's verified heads by hash; `retired` the hashes
    whose owner now serves a different head. The window must already include
    this epoch. `allow_take=False` (a validator whose window was lost and not
    yet rebuilt) keeps hold, resume and ghost logic but makes no first or take
    decision: it fails closed rather than judging on a partial window.

    `senior_bars` (protocol v2 copy rule, `copy_rule.senior_bars`) lists, per
    head, the senior seat holders it must ALSO beat by the margin before it can take
    a seat, first or by challenge. None (or a head with no entry) adds no bar,
    which is this function's behaviour before the copy rule.

    With `params.confirm_questions` set (the confirmation rule in force), a
    challenger that beats a settled champion is recorded as pending, and takes
    the seat only by passing again on the questions drawn since (module
    docstring). With it 0 nothing below differs from the rule before it.
    """
    detail: list[tuple[str, str]] = []
    confirm_on = params.confirm_questions > 0
    ref = None
    if params.odds is not None:
        from fugal_subnet import odds_score

        ref = odds_score.reference(window.pairs, params.odds.ref_prior)
    ghosts = tuple(g for g in state.ghosts if g.last_epoch >= epoch)
    if state.champion is None and state.ghosts and not ghosts:
        # The last ghost expired with the seat empty: the champion that set
        # the high-water marks is gone for good, so the marks reset and the
        # next eligible head can take the seat. Without this a departed
        # champion's bar could leave the share burning indefinitely.
        state = dataclasses.replace(state, high_water=0.0, high_water_hard=0.0,
                                    pending=(), confirm_from=None)
        detail.append(("", "last ghost expired with the seat empty; high-water marks reset"))
    champion = state.champion
    since = state.champion_since
    history = state.history
    confirm_from = state.confirm_from if confirm_on else None

    # A champion that went offline (a "missed" ghost) and whose owner then
    # commits a different head is retired: its ghost bar is extended to the
    # retire window, counted from when it first stood.
    upgraded = []
    for g in ghosts:
        if g.reason == "missed" and g.head_hash in retired:
            first = g.last_epoch - params.ghost_off + 1
            g = Ghost(g.head_hash, first + params.ghost_retire - 1, "retired")
            detail.append((g.head_hash, f"ghost upgraded to retired; bars until epoch {g.last_epoch}"))
        upgraded.append(g)
    ghosts = tuple(upgraded)

    def can_hold(h: str) -> bool:
        c = candidates.get(h)
        if c is None or not c.eligible:
            return False
        acc, n = window_accuracy(window, h)
        if n == 0:
            detail.append((h, "no scored questions in the window"))
            return False
        if acc < params.min_accuracy:
            detail.append((h, f"champion accuracy {acc:.4f} below the floor "
                              f"{params.min_accuracy:.4f}"))
            return False
        if ref is not None:
            # A seated head loses the seat when the reference's odds beat its
            # own on the window's pairs, by the test a challenger would have
            # to pass (no margin). Not a point estimate: a head whose skill is
            # near zero would otherwise lose and regain the seat on noise, and
            # every epoch without a champion burns the miner share.
            su = odds_score.skill_units(window, ref, h, run_only=False)
            worse = paired_test({c: -u for c, u in su.by_category.items()}, margin_points=0.0,
                                confidence=params.confidence, resamples=params.resamples,
                                seed=pair_seed(f"{epoch_id}:hold", "reference", h))
            if worse.takes_seat:
                detail.append((h, f"champion odds are worse than the reference "
                                  f"({-worse.mean_points:+.3f} Brier points over "
                                  f"{su.n_pairs} pairs, upper bound {-worse.lcb_points:+.3f})"))
                return False
        return True

    if champion is not None:
        if not can_hold(champion):
            reason = "retired" if champion in retired else "missed"
            span = params.ghost_retire if reason == "retired" else params.ghost_off
            ghosts = ghosts + (Ghost(champion, epoch + span - 1, reason),)
            detail.append((champion, f"champion lost the seat ({reason}); "
                                     f"ghost bar until epoch {epoch + span - 1}"))
            history = (champion,) + tuple(h for h in history if h != champion)
            champion = None
    else:
        # RESUME: a champion that only missed takes its own seat back while
        # its ghost bar still stands. That is what the ghost window is for: a
        # short outage must not hand the seat to whoever is next.
        for g in sorted(ghosts, key=lambda g: g.head_hash):
            if g.reason == "missed" and can_hold(g.head_hash):
                champion = g.head_hash
                since = epoch
                ghosts = tuple(x for x in ghosts if x.head_hash != champion)
                history = tuple(h for h in history if h != champion)
                detail.append((champion, "champion resumed its seat inside its ghost window"))
                break

    ordered = sorted(candidates.values(), key=lambda c: (c.seniority, c.head_hash))
    ghost_hashes = {g.head_hash for g in ghosts}
    challengers = [
        c for c in ordered
        if c.eligible and c.burned_in
        and c.head_hash != champion and c.head_hash not in ghost_hashes
    ] if allow_take else []
    if not allow_take:
        detail.append(("", "window not trusted: no first or take decision this epoch"))
    # A head seated by the plain test (an empty seat, a retired ghost, or a
    # champion still in its own open period) is open to the plain test for
    # this many epochs before a challenger needs confirming against it.
    open_until = epoch + params.confirm_open_epochs if confirm_on else None

    def beats_seniors(c: Candidate, already: Sequence[str] = ()) -> bool:
        # The copy rule: a head must beat every senior seat holder in the
        # comparison set by the same margin, on the same paired test, before
        # it may take a seat. A copy of a seat holder never does.
        for k in (senior_bars or {}).get(c.head_hash, ()):
            if k == c.head_hash or k in already:
                continue
            if _challenge(window, c, k, epoch_id, params, detail, "senior", ref=ref) is None:
                detail.append((c.head_hash, f"does not beat senior {k[:16]} by the margin"))
                return False
        return True

    if champion is None and not ghosts:
        # FIRST: an empty seat with nothing barring it goes to the eligible
        # head with the highest measured net value (solo_value) among those
        # that pass the gates and, under the copy rule, beat every senior seat
        # holder in the comparison set; seniority, then the hash, break exact
        # ties (the real first-commit block first, so an original beats its
        # pre-flip copies). It went to the most SENIOR such head, and at the flip every
        # pre-flip commit ties at the seniority floor, so the first seat was a
        # salted draw: N coldkeys bought N tickets, a junk head above the
        # accuracy floor could win it (v2 attack review, 2026-11-09).
        eligible = [c for c in challengers
                    if _passes_gates(state, window, c, params, detail) and beats_seniors(c)]
        value = {}
        if ref is not None:
            # On odds an empty seat goes to the best measured skill over the
            # reference, among heads with enough pairs and skill not below
            # zero (the rule a seated head is held to).
            kept = []
            for c in eligible:
                skill, n = odds_score.solo_skill(window, ref, c.head_hash, c.since_epoch)
                if n < params.odds.min_pairs:
                    detail.append((c.head_hash, f"only {n} pairs; {params.odds.min_pairs} needed"))
                elif skill < 0:
                    detail.append((c.head_hash, "its odds do not beat the reference "
                                                f"({odds_score.points(skill):+.3f} Brier points)"))
                else:
                    value[c.head_hash] = skill
                    kept.append(c)
            eligible = kept
        else:
            value = {c.head_hash: solo_value(window, c.head_hash, params, c.since_epoch)
                     for c in eligible}
        if eligible:
            ranked = sorted(eligible, key=lambda c: (
                -value[c.head_hash],
                c.first_block if c.first_block is not None else float("inf"),
                c.seniority, c.head_hash))
            c = ranked[0]
            new = _seated(state, window, c, epoch, history, params, open_until)
            detail.append((c.head_hash, f"took the empty seat (best of {len(eligible)} "
                                        + ("eligible by measured value)" if ref is None else
                                           "eligible by the skill of its odds)")))
            return SeatDecision(new, c.head_hash, "first", None, tuple(detail))
        new = SeatState(None, None, history, (), state.high_water, state.high_water_hard)
        return SeatDecision(new, None, "empty", None, tuple(detail))

    bars = [champion] if champion is not None else sorted(ghost_hashes)

    # CONFIRM. A bar needs confirming against when it is the champion, or its
    # ghost while it has only missed (a short outage must not reopen the seat
    # to a single lucky pass), and the champion is settled. A retired ghost
    # answers no new questions, so there is nothing fresh to pair with it.
    ghost_reason = {g.head_hash: g.reason for g in ghosts}
    settled = confirm_on and (confirm_from is None or epoch >= confirm_from)
    guarded = [b for b in bars
               if settled and (b == champion or ghost_reason.get(b) == "missed")]
    # A window that is not trusted decides nothing this epoch, so it does not
    # judge the pending records either: they pass through as published. Judged
    # against a lost or refilling window, every record reads as a broken run,
    # and a restarted root would publish a seat that dropped them all.
    if not allow_take:
        pending = {p.head_hash: p for p in (state.pending if confirm_on else ())}
    else:
        pending = _live_pending(state.pending if confirm_on else (), epoch, window,
                                candidates, guarded, params, detail)

    qualifying: list[tuple[int, tuple, str]] = []
    for c in challengers:
        if not _passes_gates(state, window, c, params, detail):
            continue
        worst_mean: int | None = None
        for bar in bars:
            mean = _challenge(window, c, bar, epoch_id, params, detail, ref=ref)
            if mean is None:
                worst_mean = None
                break
            worst_mean = mean if worst_mean is None else min(worst_mean, mean)
        # The copy rule is pass/fail; the ranking among qualifiers stays the
        # margin over the champion or its ghost.
        if worst_mean is None or not beats_seniors(c, bars):
            continue
        if guarded:
            p = pending.get(c.head_hash)
            if p is None:
                pending[c.head_hash] = Pending(c.head_hash, epoch, guarded[0])
                fresh_need = (f"{params.odds.confirm_pairs} pairs" if params.odds is not None
                              else f"{params.confirm_questions} questions")
                detail.append((c.head_hash, "passed the take test; takes the seat when it "
                                            f"passes again on {fresh_need} "
                                            "drawn from the next epoch on"))
                continue
            if any(_challenge(window, c, b, epoch_id, params, detail,
                              fresh_after=p.since_epoch, ref=ref) is None for b in guarded):
                continue
            detail.append((c.head_hash, "confirmed on questions drawn since epoch "
                                        f"{p.since_epoch}"))
        qualifying.append((worst_mean, c.seniority, c.head_hash))

    if qualifying:
        # Highest margin over the bar wins; ties go to the most senior.
        qualifying.sort(key=lambda t: (-t[0], t[1], t[2]))
        winner = candidates[qualifying[0][2]]
        if champion is not None:
            history = (champion,) + tuple(x for x in history if x != champion)
        # A confirmed winner has replicated, so it is settled at once; one that
        # took an open or retired seat by the plain test gets the open period.
        new = _seated(state, window, winner, epoch, history, params,
                      (epoch if guarded else open_until))
        return SeatDecision(new, winner.head_hash, "take", bars[0], tuple(detail))

    kept = tuple(pending[h] for h in sorted(pending))
    if champion is not None:
        new = SeatState(champion, since, history, ghosts,
                        state.high_water, state.high_water_hard, kept, confirm_from)
        return SeatDecision(new, champion, "hold", champion, tuple(detail))

    new = SeatState(None, None, history, ghosts, state.high_water, state.high_water_hard,
                    kept, confirm_from)
    return SeatDecision(new, None, "ghost", bars[0], tuple(detail))


def _live_pending(
    pending: Sequence[Pending], epoch: int, window: Window,
    candidates: Mapping[str, Candidate], guarded: Sequence[str], params: SeatParams,
    detail: list[tuple[str, str]],
) -> dict[str, Pending]:
    """The pending records that still stand this epoch, by head.

    A record stands while the bar it passed against is still guarded, its
    head is verified and eligible this epoch with an unbroken run since the
    pass (the comparison reads the current run only, so a miss leaves no fresh
    questions to confirm on), and the epoch of the pass is still in the
    window. That limit is what keeps the second test a second test: once the
    pass has left the window, every question in the window is "fresh", and
    testing on them is the first test over again. The head is then recorded
    anew the next time it passes, and its fresh questions start from there.
    """
    out: dict[str, Pending] = {}
    for p in pending:
        c = candidates.get(p.head_hash)
        if p.bar not in guarded:
            continue
        if c is None or not c.eligible:
            detail.append((p.head_hash, "pending confirmation dropped: not verified this epoch"))
            continue
        if epoch - p.since_epoch >= window.size:
            detail.append((p.head_hash, "pending confirmation lapsed: not confirmed within "
                                        f"the {window.size}-epoch window of epoch "
                                        f"{p.since_epoch}"))
            continue
        run = window.current_run(p.head_hash, c.since_epoch)
        if not run or run[0] > p.since_epoch:
            detail.append((p.head_hash, "pending confirmation dropped: its run was broken "
                                        f"after epoch {p.since_epoch}"))
            continue
        out[p.head_hash] = p
    return out
