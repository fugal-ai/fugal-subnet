"""Protocol v2 champion scoring: the paired net-value test (I3, I1).

From the activation block the miner share goes to a CHAMPION, and a head takes
the seat only by beating the champion on the same questions:

    NV_i = u_i * (g_challenger,i - g_champion,i) - (s_challenger,i - s_champion,i) / V
    V    = CHAMPION_ACCURACY_VALUE * the champion's own mean spend per question
           over the same paired questions
    u_i  = CHAMPION_HARD_WEIGHT on a hard question (the hard-reasoning
           category), else 1
    NV   = 100 * mean_i NV_i                       (percentage points)

g is the shared cell grade (fugal_subnet/shared_grades.py) and s the policy's
cost for the question from pinned prices and the frame's verbosity, so
neither side of NV is a miner's self-report. V is measured from the window,
so nothing about it is pinned in advance and it is rebuilt from reveals.

The challenger takes the seat when BOTH its point estimate and the one-sided
paired-bootstrap lower bound of NV exceed CHAMPION_MARGIN points; the
champion keeps it with no margin. The bootstrap resamples questions with
replacement within category, keeping both heads' answers to a question
together, which is what makes the comparison paired.

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


@dataclass
class Window:
    """The last `size` epochs of slices, and every head's answers in them."""

    size: int
    slices: dict[int, EpochSlice]
    heads: dict[str, dict[int, HeadEpoch]]

    @classmethod
    def empty(cls, size: int) -> Window:
        return cls(size=int(size), slices={}, heads={})

    def add_epoch(self, sl: EpochSlice, answers: Mapping[str, HeadEpoch]) -> Window:
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
        return Window(size=self.size, slices=slices, heads=heads)

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

    def __post_init__(self):
        # Bounded: the payout reads two former champions; HISTORY_KEEP keep
        # the record readable without letting it grow for the life of the
        # subnet.
        object.__setattr__(self, "history", tuple(self.history)[:HISTORY_KEEP])


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
            history: tuple[str, ...], params: SeatParams) -> SeatState:
    """The state with `winner` newly seated and the high-water marks raised."""
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
    )


def _challenge(
    window: Window, c: Candidate, bar: str, epoch_id: str, params: SeatParams,
    detail: list[tuple[str, str]], label: str = "",
) -> int | None:
    """The paired test of `c` over `bar`: its mean NV units if it beats the bar
    by the margin on at least the minimum paired questions, else None."""
    pu = pair_units(window, c.head_hash, bar, accuracy_value=params.accuracy_value,
                    hard_weight=params.hard_weight, since_epoch=c.since_epoch,
                    anchor_spend=params.anchor_spend)
    res = paired_test(
        pu.by_category, margin_points=params.margin_points,
        confidence=params.confidence, resamples=params.resamples,
        seed=pair_seed(epoch_id, bar, c.head_hash),
    )
    v = "none" if pu.value_per_correct is None else f"{pu.value_per_correct:.6f}"
    prefix = f"vs {label} " if label else "vs "
    detail.append((c.head_hash, f"{prefix}{bar[:16]}: n={res.n_questions} V={v} "
                                f"NV={res.mean_points:+.3f} LCB={res.lcb_points:+.3f}"))
    if res.n_questions < params.min_paired_questions:
        detail.append((c.head_hash, f"only {res.n_questions} paired questions; "
                                    f"{params.min_paired_questions} needed"))
        return None
    if not res.takes_seat:
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
    """
    detail: list[tuple[str, str]] = []
    ghosts = tuple(g for g in state.ghosts if g.last_epoch >= epoch)
    if state.champion is None and state.ghosts and not ghosts:
        # The last ghost expired with the seat empty: the champion that set
        # the high-water marks is gone for good, so the marks reset and the
        # next eligible head can take the seat. Without this a departed
        # champion's bar could leave the share burning indefinitely.
        state = dataclasses.replace(state, high_water=0.0, high_water_hard=0.0)
        detail.append(("", "last ghost expired with the seat empty; high-water marks reset"))
    champion = state.champion
    since = state.champion_since
    history = state.history

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

    def beats_seniors(c: Candidate, already: Sequence[str] = ()) -> bool:
        # The copy rule: a head must beat every senior seat holder in the
        # comparison set by the same margin, on the same paired test, before
        # it may take a seat. A copy of a seat holder never does.
        for k in (senior_bars or {}).get(c.head_hash, ()):
            if k == c.head_hash or k in already:
                continue
            if _challenge(window, c, k, epoch_id, params, detail, "senior") is None:
                detail.append((c.head_hash, f"does not beat senior {k[:16]} by the margin"))
                return False
        return True

    if champion is None and not ghosts:
        # FIRST: an empty seat with nothing barring it goes to the most senior
        # eligible head that passes the gates (and, under the copy rule, beats
        # every senior seat holder in the comparison set).
        for c in challengers:
            if _passes_gates(state, window, c, params, detail) and beats_seniors(c):
                new = _seated(state, window, c, epoch, history, params)
                detail.append((c.head_hash, "took the empty seat"))
                return SeatDecision(new, c.head_hash, "first", None, tuple(detail))
        new = SeatState(None, None, history, (), state.high_water, state.high_water_hard)
        return SeatDecision(new, None, "empty", None, tuple(detail))

    bars = [champion] if champion is not None else sorted(ghost_hashes)
    qualifying: list[tuple[int, tuple, str]] = []
    for c in challengers:
        if not _passes_gates(state, window, c, params, detail):
            continue
        worst_mean: int | None = None
        for bar in bars:
            mean = _challenge(window, c, bar, epoch_id, params, detail)
            if mean is None:
                worst_mean = None
                break
            worst_mean = mean if worst_mean is None else min(worst_mean, mean)
        # The copy rule is pass/fail; the ranking among qualifiers stays the
        # margin over the champion or its ghost.
        if worst_mean is not None and beats_seniors(c, bars):
            qualifying.append((worst_mean, c.seniority, c.head_hash))

    if qualifying:
        # Highest margin over the bar wins; ties go to the most senior.
        qualifying.sort(key=lambda t: (-t[0], t[1], t[2]))
        winner = candidates[qualifying[0][2]]
        if champion is not None:
            history = (champion,) + tuple(x for x in history if x != champion)
        new = _seated(state, window, winner, epoch, history, params)
        return SeatDecision(new, winner.head_hash, "take", bars[0], tuple(detail))

    if champion is not None:
        new = SeatState(champion, since, history, ghosts,
                        state.high_water, state.high_water_hard)
        return SeatDecision(new, champion, "hold", champion, tuple(detail))

    new = SeatState(None, None, history, ghosts, state.high_water, state.high_water_hard)
    return SeatDecision(new, None, "ghost", bars[0], tuple(detail))
