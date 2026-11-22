"""Protocol v4: the menu step's inputs, from verified proofs (docs/decisions/LIVE_MENU_V4.md,
section 4.4; work package W4).

    MenuState(k + 2) = menu_state.step(MenuState(k + 1), inputs(k), params)

This module is `inputs(k)`: pure functions from epoch k's verified v4 proofs,
and from a window of earlier epochs' summaries where the design reads a
window, to each member of `menu_state.StepInputs`. Nothing here reads a
clock, the network, the chain or a file. The validator hands in, for each
verified proof, the coldkey that owns its hotkey **as read from the chain**
(`VerifiedProof.coldkey`): no proof field names a coldkey, and none is read.

Two stages, because some inputs are votes of one epoch and some are sums
over a window:

1. `summarize_epoch(epoch, MenuState(epoch), proofs, ...)` -> `EpochSummary`:
   the epoch's endpoint vote, each coldkey's exploration calls and failures
   per entry revision, its pooled pairs by entry revision and category, the scored picks of the heads that count for use, and its
   graded scored-question cells by entry revision and category (the
   pick-cost table's cells, W13). A
   summary is integers and strings only, with one JSON form
   (`summary_to_dict` / `summary_from_dict`), so a validator keeps it in its
   window and a restart reads back exactly what was written.
2. `step_inputs(MenuState(k + 1), MenuState(k), window, proofs_k, ...)` ->
   `StepInputs`: the endpoint vote of epoch k (section 7.1), token sums and
   exploration-pair counts per entry for its current revision (7.2, 8), the
   graded cells the pick-cost table is computed from (W13),
   health (9.2, the replay's rule), accuracy for the "plausibly useful" gate,
   entry use (9.2, the cap), and what the gates read of each provisional
   entry the step decides (`gate_stats`: the field's labels and each
   coldkey's answered share; FAST_CROWN_V4.md 18).

ONE VOTE PER COLDKEY. Every vote and every count that one operator could
otherwise inflate is taken per coldkey (`_voter`): an operator with fifty
hotkeys is one voter on a price and one coldkey in the health rule.

EXACT AND ORDER-FREE. Medians are of Fractions and rounded half up once;
bounds use an integer square root at a fixed scale. Every collection is
sorted by its key before it is read, and every output is sorted, so the
order proofs, reads or summaries arrive in never changes a byte (I1).

Nothing on the v1-v3 path imports this module.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from fugal_subnet import menu_rules
from fugal_subnet import menu_state as ms
from fugal_subnet.head_odds import ODDS_SCALE
from fugal_subnet.tee.endpoint_reads import (
    MAX_ANSWER_LIMIT,
    MAX_PRICE_PICO,
    EndpointRead,
)
from fugal_subnet.tee.proof import BenchmarkProof

# The one-sided 95% normal quantile, exactly 1.645, for the accuracy bounds
# of the "plausibly useful" gate (section 5).
Z_95 = Fraction(329, 200)
# Fixed-point scale of the integer square root in the bounds.
_SQRT_SCALE = 10**12


# --- what the validator hands in -------------------------------------------------


@dataclass(frozen=True)
class VerifiedProof:
    """A proof that passed `verify_proof` under protocol v4 for the epoch,
    and the coldkey that owns its hotkey, read from the chain at the epoch's
    boundary. The proof names no coldkey; this is the only one used."""

    proof: BenchmarkProof
    coldkey: str

    @property
    def hotkey(self) -> str:
        return self.proof.hotkey


def _voter(vp: VerifiedProof) -> str:
    """Whose vote a proof is: its operator's, the coldkey. Every vote and
    count below is taken per voter, so hotkeys of one coldkey are one vote."""
    return vp.coldkey


def _sorted_proofs(proofs: Iterable[VerifiedProof]) -> list[VerifiedProof]:
    out = []
    for vp in proofs:
        if not isinstance(vp, VerifiedProof) or not isinstance(vp.proof, BenchmarkProof):
            raise ValueError("menu inputs are read from VerifiedProof values")
        if not isinstance(vp.coldkey, str) or not vp.coldkey:
            raise ValueError(f"proof of {vp.proof.hotkey!r}: its coldkey is read from the chain")
        out.append(vp)
    hotkeys = [vp.hotkey for vp in out]
    if len(hotkeys) != len(set(hotkeys)):
        raise ValueError("two verified proofs of one hotkey in one epoch")
    return sorted(out, key=lambda vp: vp.hotkey)


def _round_half_up(x: Fraction) -> int:
    return int((2 * x + 1) // 2)


def _median(values: Sequence[Fraction]) -> Fraction:
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# --- 1. the endpoint vote (section 7.1) -------------------------------------------


def _read_ok(read: EndpointRead) -> bool:
    """A read in the proof field's one shape (endpoint_reads.reads_from_json
    already holds a parsed proof to it): each host once, every number an
    integer in range, a host that is not listed carrying no value. Checked
    again here so the vote never counts a host twice or a value that is not
    whole picodollars, whatever built the read."""
    if not isinstance(read, EndpointRead) or not isinstance(read.hosts, tuple):
        return False
    names = [getattr(h, "host", None) for h in read.hosts]
    if not names or names != sorted(set(names)) or not all(isinstance(n, str) for n in names):
        return False
    for h in read.hosts:
        if not isinstance(h.listed, bool):
            return False
        for v, hi in ((h.price_in, MAX_PRICE_PICO), (h.price_out, MAX_PRICE_PICO),
                      (h.limit, MAX_ANSWER_LIMIT)):
            if not _is_int(v) or not 0 <= v <= hi:
                return False
        if not h.listed and (h.price_in, h.price_out, h.limit) != (0, 0, 0):
            return False
    return True


def _proof_reads(proof: BenchmarkProof) -> tuple[EndpointRead, ...]:
    """The proof's reads, or none when any of them is malformed or a model is
    read twice: such a proof has no vote (a failed read is no vote)."""
    reads = tuple(proof.endpoint_reads or ())
    models = [getattr(r, "model", None) for r in reads]
    if len(models) != len(set(models)) or not all(_read_ok(r) for r in reads):
        return ()
    return reads


def endpoint_vote(proofs: Iterable[VerifiedProof]) -> tuple[ms.HostRead, ...]:
    """The epoch's voted read of every (model, host) any verified proof read,
    sorted by (model, host): section 7.1's median across coldkeys.

    Per coldkey, its hotkeys' reads of the host are one vote: listed when at
    least half of them read it listed, and then the median of its listed
    hotkeys' prices and limit. Across coldkeys, the host is listed when at
    least half of the coldkeys that read it vote it listed; then `voters` is
    the number of coldkeys voting it listed and each value is their median,
    rounded half up to a whole picodollar (or token). A host the vote shows
    unlisted has `voters` the coldkeys that read it and no value. The step
    moves nothing on fewer than ENDPOINT_MIN_VOTERS voters (menu_state), so
    one operator's reads, however many hotkeys carry them, move nothing."""
    # (model, host) -> voter -> hotkey -> HostListing
    by_host: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for vp in _sorted_proofs(proofs):
        for r in _proof_reads(vp.proof):
            for h in r.hosts:
                by_host.setdefault((r.model, h.host), {}).setdefault(_voter(vp), {})[vp.hotkey] = h
    out = []
    for (model, host), voters in sorted(by_host.items()):
        listed_votes: list[tuple[Fraction, Fraction, Fraction]] = []
        for v in sorted(voters):
            hs = [voters[v][hk] for hk in sorted(voters[v])]
            listed = [h for h in hs if h.listed]
            if 2 * len(listed) < len(hs):
                continue
            listed_votes.append((_median([Fraction(h.price_in) for h in listed]),
                                 _median([Fraction(h.price_out) for h in listed]),
                                 _median([Fraction(h.limit) for h in listed])))
        if 2 * len(listed_votes) >= len(voters):
            out.append(ms.HostRead(
                model=model, host=host, voters=len(listed_votes), listed=True,
                price_in=_round_half_up(_median([x[0] for x in listed_votes])),
                price_out=_round_half_up(_median([x[1] for x in listed_votes])),
                limit=_round_half_up(_median([x[2] for x in listed_votes]))))
        else:
            out.append(ms.HostRead(model=model, host=host, voters=len(voters), listed=False))
    return tuple(out)


# --- 2. one epoch's summary ---------------------------------------------------------


@dataclass(frozen=True)
class CallCount:
    """One coldkey's exploration calls to one entry revision in one epoch,
    and how many of them failed."""

    entry_id: str
    rev: int
    coldkey: str
    calls: int
    failed: int


@dataclass(frozen=True)
class PairCell:
    """One epoch's pooled exploration pairs of one entry revision in one
    category. `measured` is always false (no proposer measures any more; the
    field is kept so a summary has one shape). `y` is the sum of the pairs'
    outcomes in ODDS_SCALE units; tokens are attested sums."""

    entry_id: str
    rev: int
    category: str
    measured: bool
    pairs: int
    y: int
    tokens_in: int
    tokens_out: int


@dataclass(frozen=True)
class GradedCell:
    """One epoch's graded scored-question cells of one entry revision in one
    category (`graded_cells`): how many, and the sums of their consensus
    tokens (each cell's coldkey median, whole tokens, bounded by the answer
    cap). What the pick-cost table is computed from (W13)."""

    entry_id: str
    rev: int
    category: str
    cells: int
    tokens_in: int
    tokens_out: int


@dataclass(frozen=True)
class EpochSummary:
    """What one epoch contributes to the window the step's inputs read."""

    epoch: int
    votes: tuple[ms.HostRead, ...]
    calls: tuple[CallCount, ...]
    pairs: tuple[PairCell, ...]
    use: tuple[ms.EntryUse, ...]
    graded: tuple[GradedCell, ...] = ()
    # The entries provisional on the epoch's menu (FAST_CROWN_V4.md 18): their
    # calls are kept for their gates and left out of the outage rule and the
    # health counts.
    provisional: tuple[str, ...] = ()


def exploration_calls(proofs: Iterable[VerifiedProof], state: ms.MenuState
                      ) -> tuple[CallCount, ...]:
    """Each coldkey's exploration calls (the shared set and both private
    parts) per entry revision of the epoch's menu
    `state`, summed over its hotkeys, sorted by (entry, rev, coldkey)."""
    revs = {e.id: e.rev for e in state.entries}
    acc: dict[tuple[str, int, str], list[int]] = {}
    for vp in _sorted_proofs(proofs):
        for r in vp.proof.exploration_results:
            if r.routed_model not in revs:
                raise ValueError(f"exploration routed to {r.routed_model!r}, which is not on the "
                                 f"menu of epoch {state.epoch}: the proof was not verified on it")
            c = acc.setdefault((r.routed_model, revs[r.routed_model], _voter(vp)), [0, 0])
            c[0] += 1
            c[1] += 1 if r.failed else 0
    return tuple(CallCount(e, rev, ck, n, f) for (e, rev, ck), (n, f) in sorted(acc.items()))


def pair_cells(pooled: Iterable, proofs: Iterable[VerifiedProof], state: ms.MenuState,
               category_of: Mapping[str, str]) -> tuple[PairCell, ...]:
    """The epoch's pairs by entry revision and category, sorted.

    `pooled` is the epoch's pooled exploration pairs (explore_pool.Pair, as
    pooled under v4's weights and cap); each entry's revision is the one on
    the epoch's menu `state`. `proofs` is accepted for the summary's one
    signature and read for nothing."""
    revs = {e.id: e.rev for e in state.entries}
    acc: dict[tuple[str, int, str, bool], list[int]] = {}

    def add(entry_id, rev, qid, measured, y, t_in, t_out):
        if qid not in category_of:
            raise ValueError(f"question {qid!r} has no category")
        c = acc.setdefault((entry_id, rev, str(category_of[qid]), measured), [0, 0, 0, 0])
        c[0] += 1
        c[1] += int(y)
        c[2] += int(t_in)
        c[3] += int(t_out)

    for p in sorted(pooled, key=lambda p: (p.question_id, p.model)):
        if p.model not in revs:
            raise ValueError(f"pooled pair on {p.model!r}, which is not on the menu of epoch "
                             f"{state.epoch}")
        add(p.model, revs[p.model], p.question_id, False, p.y, p.prompt_tokens,
            p.completion_tokens)
    return tuple(PairCell(e, rev, cat, m, *v) for (e, rev, cat, m), v in sorted(acc.items()))


def epoch_use(proofs: Iterable[VerifiedProof], counted_heads: Iterable[str]
              ) -> tuple[ms.EntryUse, ...]:
    """The epoch's scored picks per entry by the heads that count for use
    (section 9.2: those that held the seat or were senior bars), given by
    weights hash. A head is counted once whichever hotkeys ran it: on the
    proof of its lowest hotkey."""
    counted = set(counted_heads)
    seen: set[str] = set()
    acc: dict[str, int] = {}
    for vp in _sorted_proofs(proofs):
        h = vp.proof.weights_hash
        if h not in counted or h in seen:
            continue
        seen.add(h)
        for r in vp.proof.scored_results:
            acc[r.routed_model] = acc.get(r.routed_model, 0) + 1
    return tuple(ms.EntryUse(e, n) for e, n in sorted(acc.items()))


def cell_token_bounds(entry: ms.Entry) -> tuple[int, int]:
    """(prompt, completion) tokens one graded cell of `entry` counts at
    most: the answer cap (menu_rules.MAX_ANSWER_TOKENS), and for the answer
    also the entry's own answer limit on the epoch's menu."""
    cap = menu_rules.MAX_ANSWER_TOKENS
    return cap, min(cap, entry.max_tokens)


def graded_cells(proofs: Iterable[VerifiedProof], state: ms.MenuState,
                 category_of: Mapping[str, str]) -> tuple[GradedCell, ...]:
    """The epoch's graded scored-question cells, by entry revision of the
    epoch's menu `state` and category, sorted: the pick-cost table's cells
    (W13; SEAT_MINIMAL_V4.md 3.4).

    A cell is a (question, entry) of the scored questions that at least one
    verified proof answered without failing (a failed call is not a cell,
    as it has no grade, I4). Its tokens are a vote of operators, as a shared
    cell's grade is (shared_grades) and a pooled pair's tokens are
    (explore_pool.pool_pairs): each coldkey's answers are one vote, the mean
    over its hotkeys; the cell's tokens are the median across coldkeys,
    rounded half up to whole tokens. One operator however many hotkeys it
    runs is one vote in a cell, and alone in a cell it is bounded: no cell
    counts more completion tokens than the entry's answer limit on the
    epoch's menu, nor more prompt tokens than the answer cap
    (menu_rules.MAX_ANSWER_TOKENS)."""
    entries = {e.id: e for e in state.entries}
    by_cell: dict[tuple[str, str], dict[str, list[tuple[int, int]]]] = {}
    for vp in _sorted_proofs(proofs):
        for r in vp.proof.scored_results:
            if r.routed_model not in entries:
                raise ValueError(f"a scored call routed to {r.routed_model!r}, which is not on the "
                                 f"menu of epoch {state.epoch}: the proof was not verified on it")
            if r.failed:
                continue
            by_cell.setdefault((r.question_id, r.routed_model), {}).setdefault(
                _voter(vp), []).append((int(r.prompt_tokens), int(r.completion_tokens)))
    acc: dict[tuple[str, int, str], list[int]] = {}
    for (qid, eid), votes in sorted(by_cell.items()):
        if qid not in category_of:
            raise ValueError(f"question {qid!r} has no category")
        e = entries[eid]
        p_tok = [Fraction(sum(v[0] for v in votes[ck]), len(votes[ck])) for ck in sorted(votes)]
        c_tok = [Fraction(sum(v[1] for v in votes[ck]), len(votes[ck])) for ck in sorted(votes)]
        a = acc.setdefault((eid, e.rev, str(category_of[qid])), [0, 0, 0])
        a[0] += 1
        most_in, most_out = cell_token_bounds(e)
        a[1] += min(most_in, _round_half_up(_median(p_tok)))
        a[2] += min(most_out, _round_half_up(_median(c_tok)))
    return tuple(GradedCell(eid, rev, cat, *v) for (eid, rev, cat), v in sorted(acc.items()))


def summarize_epoch(epoch: int, state: ms.MenuState, proofs: Iterable[VerifiedProof], *,
                    pooled: Iterable, category_of: Mapping[str, str],
                    counted_heads: Iterable[str]) -> EpochSummary:
    """The window record of v4 epoch `epoch`, from its menu `state`, its
    verified proofs, its pooled exploration pairs and the heads that count
    for use."""
    ms.validate(state)
    if state.epoch != int(epoch):
        raise ValueError(f"the menu given is the menu of epoch {state.epoch}, not of {epoch}")
    proofs = _sorted_proofs(proofs)
    return EpochSummary(epoch=int(epoch), votes=endpoint_vote(proofs),
                        calls=exploration_calls(proofs, state),
                        pairs=pair_cells(pooled, proofs, state, category_of),
                        use=epoch_use(proofs, counted_heads),
                        graded=graded_cells(proofs, state, category_of),
                        provisional=provisional_ids(state))


def provisional_ids(state: ms.MenuState) -> tuple[str, ...]:
    """The entries provisional on `state`, sorted: their exploration calls
    count against nothing but their own answered share."""
    return tuple(sorted(e.id for e in state.entries if e.provisional))


# --- 3. the window: token sums, exploration counts, accuracy (sections 7.2, 8, 5) ---


def check_summary_window(window_epochs: int, params: ms.MenuParams | None = None) -> None:
    """Refuse a window of summaries longer than `menu_state.rejoin_gap_epochs`:
    with it, the statistics of an id that came back could read pairs of its
    previous run, which carry the same (id, rev). ValueError, naming both."""
    gap = ms.rejoin_gap_epochs(params)
    if not isinstance(window_epochs, int) or isinstance(window_epochs, bool) or window_epochs < 1:
        raise ValueError(f"the summary window must be a positive number of epochs, not "
                         f"{window_epochs!r}")
    if window_epochs > gap:
        raise ValueError(
            f"the summary window is {window_epochs} epochs, longer than the {gap} epochs between "
            "two runs of one id (cooldown + 1): an id that came back would be "
            "measured with pairs of its previous run")


def _window(window: Iterable[EpochSummary], through: int) -> list[EpochSummary]:
    """The summaries handed in up to epoch `through`, sorted by epoch, each
    epoch once."""
    out = sorted((s for s in window if s.epoch <= through), key=lambda s: s.epoch)
    epochs = [s.epoch for s in out]
    if len(epochs) != len(set(epochs)):
        raise ValueError("the window holds two summaries of one epoch")
    return out


def _cells_of(entry: ms.Entry, window: Sequence[EpochSummary], *, measured: bool | None):
    """{category: [pairs, y, tokens_in, tokens_out]} of the entry's current
    revision. Exploration pairs count from the epoch it joined on (an entry
    is explored only while on the menu, so earlier ones are another run of
    the id); measurement pairs of its revision count wherever they are."""
    acc: dict[str, list[int]] = {}
    for s in window:
        for c in s.pairs:
            if (c.entry_id, c.rev) != (entry.id, entry.rev):
                continue
            if measured is not None and c.measured is not measured:
                continue
            if not c.measured and s.epoch < entry.joined_epoch:
                continue
            a = acc.setdefault(c.category, [0, 0, 0, 0])
            for i, v in enumerate((c.pairs, c.y, c.tokens_in, c.tokens_out)):
                a[i] += v
    return acc


def token_stats(state: ms.MenuState, window: Iterable[EpochSummary], *, through: int
                ) -> tuple[ms.TokenStat, ...]:
    """Each entry's attested tokens over its window pairs of its current
    revision (exploration and measurement pairs), by category, sorted by
    entry: what the step weights to the scored mix (section 7.2)."""
    w = _window(window, through)
    out = []
    for e in state.entries:
        cells = _cells_of(e, w, measured=None)
        if cells:
            out.append(ms.TokenStat(e.id, e.rev, tuple(
                ms.TokenCell(cat, a[0], a[2], a[3]) for cat, a in sorted(cells.items()))))
    return tuple(out)


def explore_counts(state: ms.MenuState, window: Iterable[EpochSummary], *, through: int
                   ) -> tuple[ms.ExploreCount, ...]:
    """Every entry's pooled exploration pairs in the window for its current
    revision (section 8's n_e), sorted by entry. Measurement pairs never
    count; an entry with none in the window counts 0."""
    w = _window(window, through)
    return tuple(ms.ExploreCount(e.id, e.rev, sum(a[0] for a in _cells_of(e, w, measured=False)
                                                   .values()))
                 for e in state.entries)


def pick_stats(state: ms.MenuState, window: Iterable[EpochSummary], *, through: int
               ) -> tuple[ms.PickStat, ...]:
    """Each entry's graded scored-question cells over the window, for its
    current revision, by category, sorted by entry: what the step's
    pick-cost table is computed from (W13). Cells count from the epoch the
    entry joined on (an entry is routed to only while on the menu, so an
    earlier cell under its id is another run's). Entries with no cell are
    left out: the step prices them at their call cost."""
    w = _window(window, through)
    out = []
    for e in state.entries:
        acc: dict[str, list[int]] = {}
        for s in w:
            if s.epoch < e.joined_epoch:
                continue
            for c in s.graded:
                if (c.entry_id, c.rev) != (e.id, e.rev):
                    continue
                a = acc.setdefault(c.category, [0, 0, 0])
                for i, v in enumerate((c.cells, c.tokens_in, c.tokens_out)):
                    a[i] += v
        if acc:
            out.append(ms.PickStat(e.id, e.rev, tuple(
                ms.PickCell(cat, *a) for cat, a in sorted(acc.items()))))
    return tuple(out)


def _sqrt(x: Fraction) -> Fraction:
    """sqrt(x) to 1 / _SQRT_SCALE, rounded down: an integer square root, the
    same on every host."""
    if x <= 0:
        return Fraction(0)
    return Fraction(math.isqrt(x.numerator * _SQRT_SCALE**2 // x.denominator), _SQRT_SCALE)


def weighted_bounds(cells: Mapping[str, tuple[int, Fraction]],
                    mix: Iterable[tuple[str, int]]) -> tuple[Fraction, Fraction, Fraction]:
    """(rate, lower, upper): the accuracy weighted to the scored mix and its
    one-sided 95% bounds (section 5's "plausibly useful" gate).

    `cells` is {category: (graded, correct)}, `correct` possibly fractional
    (a pooled pair's outcome is a median across coldkeys). The rate weights
    each measured category's rate by its share of the mix, renormalised over
    the categories measured, as the token means are
    (menu_state.weighted_token_means); with no measured category in the mix,
    the plain rate. The bounds are Wilson's at z = 1.645 on that rate, with
    the stratified sample's effective size W^2 / sum(w_c^2 / n_c). Exact up
    to an integer square root; (0, 0, 1) with nothing graded."""
    weight = dict(mix)
    live = {c: (int(n), Fraction(k)) for c, (n, k) in cells.items() if int(n) > 0}
    if not live:
        return Fraction(0), Fraction(0), Fraction(1)
    total_w = sum(weight.get(c, 0) for c in live)
    if total_w:
        rate = sum(weight.get(c, 0) * k / n for c, (n, k) in live.items()) / total_w
        n_eff = Fraction(total_w * total_w) / sum(
            Fraction(weight.get(c, 0) ** 2, n) for c, (n, _k) in live.items() if weight.get(c, 0))
    else:
        n = sum(n for n, _k in live.values())
        rate = sum(k for _n, k in live.values()) / n
        n_eff = Fraction(n)
    z2 = Z_95 * Z_95
    denom = 1 + z2 / n_eff
    center = (rate + z2 / (2 * n_eff)) / denom
    half = Z_95 / denom * _sqrt(rate * (1 - rate) / n_eff + z2 / (4 * n_eff * n_eff))
    return rate, max(Fraction(0), center - half), min(Fraction(1), center + half)


def _ppm_floor(x: Fraction) -> int:
    return int(x * ms.PPM // 1)


def _ppm_ceil(x: Fraction) -> int:
    return -int(-x * ms.PPM // 1)


def accuracy_stats(state: ms.MenuState, window: Iterable[EpochSummary], *, through: int,
                   mix: Iterable[tuple[str, int]]) -> tuple[ms.AccuracyStat, ...]:
    """Each entry's measured accuracy over its window pairs of its current
    revision, weighted to the scored mix, with its one-sided 95% lower bound
    (`weighted_bounds`), in parts per million: the rate rounded half up, the
    bound rounded down. Sorted by entry; entries with no pair are left out."""
    w = _window(window, through)
    mix = tuple(mix)
    out = []
    for e in state.entries:
        cells = _cells_of(e, w, measured=None)
        if not cells:
            continue
        rate, lower, _upper = weighted_bounds(
            {c: (a[0], Fraction(a[1], ODDS_SCALE)) for c, a in cells.items()}, mix)
        out.append(ms.AccuracyStat(e.id, e.rev, sum(a[0] for a in cells.values()),
                                   _round_half_up(rate * ms.PPM), _ppm_floor(lower)))
    return tuple(out)


def entry_use(window: Iterable[EpochSummary], *, through: int
              ) -> tuple[tuple[ms.EntryUse, ...], int]:
    """(each entry's counted picks over the window, all counted picks): the
    step's `use` and `use_total` (section 9.2, the cap)."""
    acc: dict[str, int] = {}
    for s in _window(window, through):
        for u in s.use:
            acc[u.entry_id] = acc.get(u.entry_id, 0) + int(u.picks)
    return tuple(ms.EntryUse(e, n) for e, n in sorted(acc.items())), sum(acc.values())


# --- 4. health (section 9.2, the rule W0's replay fixed) -------------------------------


def outage(summary: EpochSummary) -> bool:
    """Whether the epoch is an outage: the median coldkey failed more than
    RETIRE_OUTAGE_RATE_BP of all its exploration calls, to every entry. Such
    an epoch is not counted by the health rule. An epoch with no calls is
    not an outage. Calls to the epoch's provisional entries are left out."""
    per: dict[str, list[int]] = {}
    skip = set(summary.provisional)
    for c in summary.calls:
        if c.entry_id in skip:
            continue
        a = per.setdefault(c.coldkey, [0, 0])
        a[0] += c.calls
        a[1] += c.failed
    rates = [Fraction(f, n) for n, f in (per[ck] for ck in sorted(per)) if n > 0]
    if not rates:
        return False
    return _median(rates) * ms.BP > ms.RETIRE_OUTAGE_RATE_BP


def counted_epochs(window: Iterable[EpochSummary], *, through: int) -> list[EpochSummary]:
    """The last RETIRE_WINDOW_EPOCHS counted epochs (outages left out) among
    the summaries up to `through`, oldest first."""
    return [s for s in _window(window, through) if not outage(s)][-ms.RETIRE_WINDOW_EPOCHS:]


def field_size(summary: EpochSummary) -> int:
    """The coldkeys serving in the epoch: those with exploration calls in a
    verified proof."""
    return len({c.coldkey for c in summary.calls})


def listing_outage(summary: EpochSummary, *, min_voters: int = ms.ENDPOINT_MIN_VOTERS) -> bool:
    """Whether the epoch's listing reads are an outage for the "gone" rule
    (the guard "failing" has in `outage`): of the hosts the vote decided (at
    least `min_voters` coldkeys), more than RETIRE_OUTAGE_RATE_BP read as
    not listed. A listing that stops showing most hosts at once (a format
    change, an outage that still answers) is not hosts leaving, and such an
    epoch is not counted. An epoch with no decided vote is not an outage."""
    decided = [r for r in summary.votes if r.voters >= min_voters]
    if not decided:
        return False
    unlisted = sum(1 for r in decided if not r.listed)
    return unlisted * ms.BP > ms.RETIRE_OUTAGE_RATE_BP * len(decided)


def _unlisted_epochs(entry: ms.Entry, window: Sequence[EpochSummary], *, through: int,
                     min_voters: int) -> int:
    """Consecutive epochs, ending with `through`, in which the vote showed
    every pinned host of the entry unlisted, each by at least `min_voters`
    coldkeys (in a smaller field, `ms.retire_minimum` of the epoch's
    `field_size`). An epoch without a summary, or with a host listed or voted
    on by fewer coldkeys, ends the run. A listing outage (`listing_outage`)
    is not counted: it neither adds to the run nor ends it."""
    by_epoch = {s.epoch: s for s in window}
    n, k = 0, through
    while k in by_epoch:
        need = ms.retire_minimum(field_size(by_epoch[k]), min_voters)
        if listing_outage(by_epoch[k], min_voters=need):
            k -= 1
            continue
        votes = {(r.model, r.host): r for r in by_epoch[k].votes}
        rs = [votes.get((entry.model, h)) for h in entry.provider_order]
        if not all(r is not None and not r.listed and r.voters >= need for r in rs):
            break
        n, k = n + 1, k - 1
    return n


def health_stats(state: ms.MenuState, window: Iterable[EpochSummary], *, through: int,
                 min_voters: int = ms.ENDPOINT_MIN_VOTERS) -> tuple[ms.HealthStat, ...]:
    """Every entry's health for its current revision, sorted by entry
    (section 9.2):

    - over the last RETIRE_WINDOW_EPOCHS counted epochs (`counted_epochs`),
      a coldkey with at least RETIRE_MIN_CALLS exploration calls to the entry
      is eligible; an eligible coldkey is failing on it when more than
      RETIRE_FAIL_RATE_BP of those calls failed and its failed calls fall in
      at least RETIRE_FAIL_MIN_EPOCHS distinct epochs;
    - `unlisted_epochs` from the votes (`_unlisted_epochs`).

    The step retires an entry when at least RETIRE_FAIL_MIN_COLDKEYS are
    eligible (in a smaller field, `ms.retire_minimum` of the coldkeys serving
    in the counted epochs, `field`) and more than half of them are failing
    (menu_state.health_retires). A coldkey is one operator however many
    hotkeys it runs, so one operator dropping its own calls is one failing
    coldkey."""
    w = _window(window, through)
    counted = counted_epochs(w, through=through)
    field = len({c.coldkey for s in counted for c in s.calls})
    out = []
    for e in state.entries:
        per: dict[str, list] = {}
        # Calls while it was provisional count against its gates, not its health.
        start = e.joined_epoch if e.provisional else max(e.joined_epoch, e.scored_from)
        for s in counted:
            if s.epoch < start:
                continue
            for c in s.calls:
                if (c.entry_id, c.rev) != (e.id, e.rev):
                    continue
                a = per.setdefault(c.coldkey, [0, 0, set()])
                a[0] += c.calls
                a[1] += c.failed
                if c.failed:
                    a[2].add(s.epoch)
        eligible = [per[ck] for ck in sorted(per) if per[ck][0] >= ms.RETIRE_MIN_CALLS]
        failing = [a for a in eligible
                   if a[1] * ms.BP > ms.RETIRE_FAIL_RATE_BP * a[0]
                   and len(a[2]) >= ms.RETIRE_FAIL_MIN_EPOCHS]
        out.append(ms.HealthStat(e.id, e.rev, eligible=len(eligible), failing=len(failing),
                                 unlisted_epochs=_unlisted_epochs(e, w, through=through,
                                                                  min_voters=min_voters),
                                 field=field))
    return tuple(out)


# --- 5. the gates of provisional entries (FAST_CROWN_V4.md 18) ------------------------


def gate_stats(start: ms.MenuState, window: Iterable[EpochSummary], *, through: int,
               mix: Iterable[tuple[str, int]]) -> tuple[ms.GateStat, ...]:
    """What the gates read of every provisional entry of `start` that joined
    by epoch `through`, over the window's epochs from its join through
    `through`, for its current revision, sorted by entry:

    - each coldkey's exploration calls to it and how many failed (summed over
      those epochs), for the median coldkey's answered share;
    - its labels (pooled pairs) by category: how many, how many correct
      (the outcome sum in whole labels, rounded down) and their tokens;
    - the one-sided 95% bounds on its accuracy weighted to the scored
      mix (`weighted_bounds`), in parts per million rounded up.

    An entry with no call and no label in the window has none: the step
    then refuses it at the answered gate."""
    w = _window(window, through)
    mix = tuple(mix)
    out = []
    for e in start.entries:
        if not e.provisional or e.joined_epoch > through:
            continue
        per: dict[str, list[int]] = {}
        cells: dict[str, list[int]] = {}
        for s in w:
            if s.epoch < e.joined_epoch:
                continue
            for c in s.calls:
                if (c.entry_id, c.rev) == (e.id, e.rev):
                    a = per.setdefault(c.coldkey, [0, 0])
                    a[0] += c.calls
                    a[1] += c.failed
            for c in s.pairs:
                if (c.entry_id, c.rev) == (e.id, e.rev) and not c.measured:
                    a = cells.setdefault(c.category, [0, 0, 0, 0])
                    for i, v in enumerate((c.pairs, c.y, c.tokens_in, c.tokens_out)):
                        a[i] += v
        if not per and not cells:
            continue
        _rate, lower, upper = weighted_bounds(
            {c: (a[0], Fraction(a[1], ODDS_SCALE)) for c, a in cells.items()}, mix)
        out.append(ms.GateStat(
            entry_id=e.id, rev=e.rev,
            coldkeys=tuple(ms.CallTally(ck, *per[ck]) for ck in sorted(per)),
            cells=tuple(ms.MeasureCell(cat, a[0], a[0], a[1] // ODDS_SCALE, a[2], a[3])
                        for cat, a in sorted(cells.items())),
            upper_ppm=min(ms.PPM, _ppm_ceil(upper)), lower_ppm=min(ms.PPM, _ppm_floor(lower))))
    return tuple(out)


# --- 6. all of them --------------------------------------------------------------------


def step_inputs(start: ms.MenuState, epoch_state: ms.MenuState,
                window: Iterable[EpochSummary], proofs: Iterable[VerifiedProof], *,
                epoch: int, nonce: bytes, lab_list, params: ms.MenuParams,
                category_of: Mapping[str, str], device_list=None,
                lab_list_in_force=None) -> ms.StepInputs:
    """`StepInputs` of scored epoch k = `epoch` for the step from `start`,
    MenuState(k + MENU_LAG - 1). `epoch_state` is MenuState(k), the menu the
    epoch's proofs were verified on; `window` the summaries the validator
    holds, the one of epoch k included (`summarize_epoch`); `proofs` epoch
    k's verified proofs (checked, read for nothing else); `lab_list` /
    `device_list` the signed lists the node holds (menu_rules.
    verify_signed_list's results), and `lab_list_in_force` the one `start`
    names. Pure. `category_of` is kept for the one signature."""
    k = int(epoch)
    if start.epoch != k + params.menu_lag - 1 or epoch_state.epoch != k:
        raise ValueError(f"the step for scored epoch {k} starts from the menu of epoch "
                         f"{k + params.menu_lag - 1}, with the proofs of the menu of epoch {k}")
    w = _window(window, k)
    this = [s for s in w if s.epoch == k]
    if not this:
        raise ValueError(f"the window holds no summary of the scored epoch {k}")
    _sorted_proofs(proofs)
    use, use_total = entry_use(w, through=k)
    return ms.StepInputs(
        epoch=k, nonce=bytes(nonce), lab_list=lab_list, lab_list_in_force=lab_list_in_force,
        device_list=device_list,
        host_reads=this[0].votes,
        token_stats=token_stats(start, w, through=k),
        explore_counts=explore_counts(start, w, through=k),
        pick_stats=pick_stats(start, w, through=k),
        health=health_stats(start, w, through=k, min_voters=params.endpoint_min_voters),
        accuracy=accuracy_stats(start, w, through=k, mix=params.mix),
        use=use, use_total=use_total,
        gates=gate_stats(start, w, through=k, mix=params.mix))


# --- the summary's one JSON form ----------------------------------------------------------

_VOTE_KEYS = ("model", "host", "voters", "listed", "price_in", "price_out", "limit")
_CALL_KEYS = ("entry_id", "rev", "coldkey", "calls", "failed")
_PAIR_KEYS = ("entry_id", "rev", "category", "measured", "pairs", "y", "tokens_in", "tokens_out")
_USE_KEYS = ("entry_id", "picks")
_GRADED_KEYS = ("entry_id", "rev", "category", "cells", "tokens_in", "tokens_out")
# The fields of a summary row that are strings; every other non-boolean field
# is an integer.
_SUMMARY_STRINGS = frozenset({"model", "host", "entry_id", "coldkey", "category"})


def summary_to_dict(s: EpochSummary) -> dict:
    """The summary as JSON-ready integers, strings and booleans, every list
    in its sorted order."""
    return {"epoch": s.epoch,
            "votes": [{k: getattr(r, k) for k in _VOTE_KEYS} for r in s.votes],
            "calls": [{k: getattr(c, k) for k in _CALL_KEYS} for c in s.calls],
            "pairs": [{k: getattr(c, k) for k in _PAIR_KEYS} for c in s.pairs],
            "use": [{k: getattr(u, k) for k in _USE_KEYS} for u in s.use],
            "graded": [{k: getattr(c, k) for k in _GRADED_KEYS} for c in s.graded],
            **({"provisional": list(s.provisional)} if s.provisional else {})}


def summary_from_dict(d) -> EpochSummary:
    """`summary_to_dict`'s inverse, strict: exactly those keys, each field of
    exactly its type (`_SUMMARY_STRINGS` are 1..MAX_STRING printable ASCII
    characters, `listed` and `measured` booleans, every other field an
    integer in 0..2^53-1, a `rev` at least 1, a graded row with at least one
    cell and no more tokens than its cells hold at the answer cap), and each
    list in its sorted
    order with no key twice. ValueError otherwise: a summary that is not
    well-typed is refused whole, and its epoch is absent from a window built
    from it."""
    def rows(name, keys, bools=()):
        items = d.get(name) if isinstance(d, Mapping) else None
        if not isinstance(items, list):
            raise ValueError(f"summary: {name} must be a list")
        out = []
        for x in items:
            if not isinstance(x, Mapping) or set(x) != set(keys):
                raise ValueError(f"summary: a {name} row must have exactly {list(keys)}")
            for k in keys:
                v = x[k]
                if k in bools:
                    if not isinstance(v, bool):
                        raise ValueError(f"summary: {name}.{k} must be a boolean")
                elif k in _SUMMARY_STRINGS:
                    ms._str(v, f"summary: {name}.{k}")
                elif not _is_int(v) or not (1 if k == "rev" else 0) <= v <= ms.MAX_INT:
                    raise ValueError(f"summary: {name}.{k} must be an integer in range")
            out.append(tuple(x[k] for k in keys))
        return out

    if not isinstance(d, Mapping) or set(d) - {"provisional"} != {"epoch", "votes", "calls",
                                                                  "pairs", "use", "graded"}:
        raise ValueError("summary: exactly epoch, votes, calls, pairs, use, graded (and "
                         "provisional when any entry was)")
    provisional = d.get("provisional", [])
    if (not isinstance(provisional, list) or "provisional" in d and not provisional
            or provisional != sorted(set(provisional))):
        raise ValueError("summary: provisional is a sorted, non-empty list of entry ids")
    for eid in provisional:
        ms._str(eid, "summary: provisional")
    if not _is_int(d["epoch"]) or d["epoch"] < 0:
        raise ValueError("summary: epoch must be a non-negative integer")
    votes = [ms.HostRead(*v) for v in rows("votes", _VOTE_KEYS, ("listed",))]
    calls = [CallCount(*v) for v in rows("calls", _CALL_KEYS)]
    pairs = [PairCell(*v) for v in rows("pairs", _PAIR_KEYS, ("measured",))]
    use = [ms.EntryUse(*v) for v in rows("use", _USE_KEYS)]
    graded = [GradedCell(*v) for v in rows("graded", _GRADED_KEYS)]
    for c in graded:
        ms._pick_category(c.category, "summary: graded.category")
    for name, items, key in (("votes", votes, lambda r: (r.model, r.host)),
                             ("calls", calls, lambda c: (c.entry_id, c.rev, c.coldkey)),
                             ("pairs", pairs, lambda c: (c.entry_id, c.rev, c.category,
                                                         c.measured)),
                             ("use", use, lambda u: u.entry_id),
                             ("graded", graded, lambda c: (c.entry_id, c.rev, c.category))):
        keys = [key(x) for x in items]
        if keys != sorted(set(keys)):
            raise ValueError(f"summary: {name} must be sorted with no key twice")
    for c in calls:
        if c.failed > c.calls:
            raise ValueError("summary: more failed calls than calls")
    for c in pairs:
        if c.y > ODDS_SCALE * c.pairs:
            raise ValueError("summary: an outcome sum above its pairs")
    for c in graded:
        if c.cells < 1 or max(c.tokens_in, c.tokens_out) > c.cells * menu_rules.MAX_ANSWER_TOKENS:
            raise ValueError("summary: a graded row needs a cell and no more tokens than its "
                             "cells hold at the answer cap")
    return EpochSummary(d["epoch"], tuple(votes), tuple(calls), tuple(pairs), tuple(use),
                        tuple(graded), tuple(provisional))


def _check_compiled_window() -> None:
    """At import: the window the validator loop keeps (CHAMPION_WINDOW_EPOCHS,
    which `_v4_menu_epoch` is handed) against the compiled constants."""
    from fugal_subnet.config import CHAMPION_WINDOW_EPOCHS
    check_summary_window(CHAMPION_WINDOW_EPOCHS)


_check_compiled_window()
