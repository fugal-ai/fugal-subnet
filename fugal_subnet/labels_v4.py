"""Protocol v4: the labels a head is scored on, and its odds on every epoch
(docs/decisions/SEAT_MINIMAL_V4.md 3.2, 4.1 to 4.3 and 4.5).

A LABEL is one pooled exploration pair (explore_pool.pool_pairs): a question,
an entry, and whether the entry answered it correctly, as the coldkey median.
Every v4 comparison of two heads X and Y (Y may be the reference R) reads
only the labels that

  1. were drawn in the last SEAT_WINDOW_EPOCHS (48) epochs;
  2. were drawn after both rows they read were committed: a head's row for an
     entry counts from the first epoch whose boundary is at or after the
     head's commit (`Side.since`; the seat package passes an earlier epoch
     for a row a successor inherited, `Side.rows_since`), and R has no
     commit;
  3. were supplied by neither head's operator: no coldkey that voted on the
     label is X's or Y's owner (R has none);
  4. are not proposal measurement pairs. As built, measurement results never
     enter the pool, so no window holds one.

`eligible` is that rule; `gain_items` and `gain_strata` are the per-category
integer sums `seat_bound` consumes, in units of 1 / BOUND_UNITS_PER_POINT_V4 of
a Brier point (one unit of odds_score's loss), each item a label's weighted
loss of Y minus that of X.

ODDS FOR EVERY EPOCH. A head's odds are a function of its committed bytes
and the pinned embeddings, not of whether it served. Validators compute them
on an epoch's pooled pairs in the epoch they verify the head; for each window
epoch from the head's commit on that they hold no odds of it for (it was not
verified then, so they did not hold its bytes), they compute them when the
head is next verified (`late_odds`), from the pairs the window stores, and
the reveal of that epoch publishes them (`late_for_reveal`). What is stored
per epoch is the `LabelPairs` (the pairs, their voters and the odds a missing
row gives) and, per head, one int16 per pair: the window file keeps both, the
reveal publishes both, and a rebuild reads them back, so the odds a node holds
for an epoch never depend on when it computed them.

NEW ENTRIES (FAST_CROWN_V4.md 18). An entry joins provisional: its labels
drawn while it is provisional (`Entry.in_allowance` of the label's own epoch)
are scored for no head, which is its one epoch of allowance, and stored with
`missing` = PROVISIONAL_LABEL so no comparison reads them. From the epoch the
gates make it full, a head with a row is scored on its own odds and a head
without one as odds 0, "never answers correctly". The pair's own epoch
decides, through the epoch's menu, and the value is stored with the pair
(`LabelPairs.missing`), so a pair keeps it after the entry leaves the menu.
The stored odds are the odds a head is scored on: its row's, or that value.
A row for an entry not on the epoch's menu is never read, and a request
update changes neither the row nor the allowance.

Nothing on the v1-v3 path imports this module.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from fugal_subnet import config
from fugal_subnet.champion import EpochPairs, Window
from fugal_subnet.head_odds import ODDS_SCALE
from fugal_subnet.seat_bound import Stratum

SEAT_WINDOW_EPOCHS = config.SEAT_WINDOW_EPOCHS_V4
# `LabelPairs.missing` of a label drawn while its entry was provisional: no
# comparison reads it (`eligible`).
PROVISIONAL_LABEL = -2
# A loss unit (odds_score: (p - y)^2 in ODDS_SCALE units) is exactly the
# bound's unit, a millionth of a Brier point.
assert ODDS_SCALE * ODDS_SCALE // 100 == config.BOUND_UNITS_PER_POINT_V4


@dataclass(frozen=True)
class LabelPairs(EpochPairs):
    """One v4 epoch's pooled pairs: EpochPairs, and per pair the coldkeys
    that voted on it (sorted) and what a head with no row for its entry is
    scored at, `missing` (int16: -1, odds 0, for an entry that was full when
    the pair was drawn; PROVISIONAL_LABEL for one that was provisional, which
    no comparison reads). `scored` is None: the provisional mark is the mask."""

    voters: tuple[tuple[str, ...], ...] = ()
    missing: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        n = len(self.question_ids)
        if self.scored is not None:
            raise ValueError("a v4 epoch masks no pair: its labels are scored from the join")
        if len(self.voters) != n or any(
                not isinstance(v, tuple) or not v or list(v) != sorted(set(v))
                or not all(isinstance(c, str) and c for c in v) for v in self.voters):
            raise ValueError("voters must hold one sorted, non-empty tuple of coldkeys per pair")
        m = self.missing
        if m is None or m.shape != (n,) or (n and (int(m.min()) < PROVISIONAL_LABEL
                                                   or int(m.max()) > ODDS_SCALE)):
            raise ValueError("missing must hold one value in -2..ODDS_SCALE per pair")


def missing_odds(menu_state, entries: Iterable[str]) -> np.ndarray:
    """Per pair entry, what a head with no row for it is scored at on a pair
    drawn in `menu_state`'s epoch: PROVISIONAL_LABEL when the entry was
    provisional then (scored for no head), else -1 (odds 0). Every entry
    must be on that menu."""
    out = []
    for eid in entries:
        e = menu_state.entry(str(eid))
        rate = e.allowance_rate(menu_state.epoch)
        out.append(PROVISIONAL_LABEL if e.in_allowance(menu_state.epoch)
                   else -1 if rate is None else int(rate))
    return np.array(out, dtype=np.int16)


def label_pairs(pairs: EpochPairs, voters: Iterable[Iterable[str]], missing) -> LabelPairs:
    """`pairs` as a v4 epoch's labels."""
    fields = {f.name: getattr(pairs, f.name) for f in dataclasses.fields(EpochPairs)}
    return LabelPairs(**fields, voters=tuple(tuple(v) for v in voters),
                      missing=np.asarray(missing, dtype=np.int16))


def effective(raw: np.ndarray, pairs: LabelPairs) -> np.ndarray:
    """The odds a head is scored on: its row's (`raw`, -1 where it has no
    row), with each missing row replaced by the pair's `missing`."""
    raw = np.asarray(raw, dtype=np.int16)
    if raw.shape != pairs.missing.shape:
        raise ValueError("odds do not align with the epoch's pairs")
    # A label of a provisional entry is read by no comparison: a missing row
    # there is stored as -1 like any other.
    return np.where(raw < 0, np.maximum(pairs.missing, -1), raw).astype(np.int16)


def odds_on(head: tuple, pairs: LabelPairs, hidden_of: Callable) -> np.ndarray:
    """A head's scored odds on one epoch's labels. `head` is (W, b, entry ids)
    as the validator loads it; rows are matched by entry id, exactly as
    head_odds.pair_odds_v4 matches them on the epoch's menu (the pairs hold
    only entries of that menu), so these are the same integers whether they
    are computed in the epoch or later."""
    return odds_on_many([head], pairs, hidden_of)[0]


def odds_on_many(heads: Iterable[tuple], pairs: LabelPairs, hidden_of: Callable
                 ) -> list[np.ndarray]:
    """[odds_on(head, pairs, hidden_of) for head in heads]: each head's rows
    matched by entry id as head_odds.head_odds matches them (a repeated id:
    its last row), the odds those of head_odds.odds (head_odds.odds_grid,
    identical by construction), the epoch's embeddings read once."""
    from fugal_subnet import head_odds

    heads = list(heads)
    if not heads:
        return []
    hidden = [hidden_of(q) for q in pairs.question_ids]
    code_of: dict[str, int] = {}
    codes = [code_of.setdefault(m, len(code_of)) for m in pairs.models]
    grid = []
    for W, b, models in heads:
        row_of = {m: i for i, m in enumerate(models)}
        grid.append((W, b, np.array([row_of.get(m, -1) for m in code_of], dtype=np.int64)))
    out = head_odds.odds_grid(grid, codes, hidden)
    return [effective(out[k], pairs) for k in range(len(heads))]


def late_odds(window: Window, heads: Mapping[str, tuple], since: Mapping[str, int],
              hidden_of: Callable) -> dict[str, dict[int, np.ndarray]]:
    """{head: {epoch: odds}} for every v4 epoch of the window, from the
    head's commit epoch (`since`) on, on which the window holds no odds of
    the head: the epochs it was not verified in. `heads` are the heads
    verified now (their bytes are what the odds are computed from)."""
    wanted: dict[str, list[int]] = {}
    for h in sorted(heads):
        have = window.odds.get(h, {})
        for e in sorted(window.pairs):
            pairs = window.pairs[e]
            if (isinstance(pairs, LabelPairs) and e >= int(since[h]) and e not in have):
                wanted.setdefault(h, []).append(e)
    by_epoch: dict[int, list[str]] = {}
    for h, epochs in wanted.items():
        for e in epochs:
            by_epoch.setdefault(e, []).append(h)
    got = {}
    for e, hs in sorted(by_epoch.items()):          # each epoch's embeddings read once
        for h, o in zip(hs, odds_on_many([heads[h] for h in hs], window.pairs[e], hidden_of)):
            got[(h, e)] = o
    return {h: {e: got[(h, e)] for e in epochs} for h, epochs in wanted.items()}


def with_odds(window: Window, odds: Mapping[str, Mapping[int, np.ndarray]]) -> Window:
    """`window` with these odds added (copy-on-write). Each must be for an
    epoch whose pairs the window holds, one value per pair."""
    merged = {h: dict(per) for h, per in window.odds.items()}
    for h in sorted(odds):
        for e, o in sorted(odds[h].items()):
            pairs = window.pairs.get(e)
            if pairs is None:
                raise ValueError(f"head {h[:12]}: odds for epoch {e}, whose pairs the window "
                                 "does not hold")
            o = np.asarray(o, dtype=np.int16)
            if o.shape != (len(pairs.question_ids),):
                raise ValueError(f"head {h[:12]}: odds do not align with epoch {e}'s pairs")
            merged.setdefault(h, {})[int(e)] = o
    return dataclasses.replace(window, odds=merged)


# --- which labels a comparison reads -----------------------------------------------------


@dataclass(frozen=True)
class Side:
    """One side of a comparison: a head (its hash, its owner's coldkey and
    the epoch its rows count from), or the reference R (`REFERENCE`).

    `since` is the first epoch whose boundary is at or after the head's
    commit (copy_rule.first_epoch). `rows_since` overrides it per entry: the
    seat package passes, for a successor, its predecessor's epoch for the
    rows it inherited byte for byte (SEAT_MINIMAL_V4.md 4.4).

    `chain` is, for a successor, its predecessors most recent first, each
    with the epoch it was its line's head until (the `since` of the head
    after it): `predecessor_odds` reads their stored odds for the epochs
    before the successor's own commit.

    A missing row counts from the head's LINE (`line_since`): the line
    origin's own commit (`line`, the oldest predecessor's `since`, published
    in the seat state as "hash:since") for a successor, its own commit for a
    head that succeeded nobody. Re-committing a successor does not restart a
    missing row's count. `added` names the entries whose row no head of the
    line had when it was first carried: a row this head added, or one an
    earlier head of its line added and it inherited. Such a row counts from
    its own start (`row_since`), and the entry's labels from `line_since` up
    to that start are scored as missing, odds 0 (the line had no row then).
    An entry the head has a row for that is neither inherited nor added (a
    changed row) counts from `since`, as a new row does."""

    head: str | None
    owner: str | None = None
    since: int | None = None
    rows_since: Mapping[str, int] = field(default_factory=dict)
    chain: tuple[tuple[str, int], ...] = ()
    added: frozenset = frozenset()
    line: int | None = None

    def __post_init__(self):
        if self.head is None:
            if (self.owner is not None or self.since is not None or self.rows_since
                    or self.chain or self.added or self.line is not None):
                raise ValueError("the reference has no owner and no commit")
        elif not isinstance(self.owner, str) or not self.owner or not isinstance(self.since, int):
            raise ValueError("a head's side needs its owner's coldkey and its commit epoch")
        elif not all(isinstance(p, str) and type(u) is int for p, u in self.chain):
            raise ValueError("a side's chain holds (predecessor, epoch) pairs")
        elif self.line is not None and type(self.line) is not int:
            raise ValueError("a side's line origin is an epoch")

    def row_since(self, entry: str) -> int | None:
        if self.head is None:
            return None
        return int(self.rows_since.get(entry, self.since))

    def line_since(self) -> int | None:
        """The first epoch a missing row of this head counts from: the
        earliest of its own commit, its inherited rows' epochs and its line
        origin's commit (`line`)."""
        if self.head is None:
            return None
        return min([int(self.since), *(int(v) for v in self.rows_since.values()),
                    *(() if self.line is None else (int(self.line),))])

    def _starts(self, index, raw) -> list:
        """Per entry of `index`, the first epoch its labels are read for this
        side: the line's for an added row and for a missing one (`raw`, the
        odds the side is scored from that epoch, `_stored`, below zero on its
        labels); an inherited row's epoch; `since` otherwise."""
        if self.head is None:
            return [None] * len(index.entries)
        line = self.line_since()
        missing = None
        if raw is not None and len(index.entries):
            _u, first = np.unique(index.entry, return_index=True)
            missing = np.asarray(raw)[first] < 0
        out = []
        for j, entry in enumerate(index.entries):
            if entry in self.added:
                out.append(line)
            elif entry in self.rows_since:
                out.append(int(self.rows_since[entry]))
            elif missing is not None and bool(missing[j]):
                out.append(line)
            else:
                out.append(int(self.since))
        return out

    def late_added(self, epoch: int) -> frozenset:
        """The added entries whose row starts after `epoch`: scored as missing
        there (the line had no row for them yet)."""
        if not self.added:
            return frozenset()
        return frozenset(m for m in self.added if int(epoch) < self.row_since(m))


REFERENCE = Side(None)


def in_window(epoch: int, now: int, window_epochs: int = SEAT_WINDOW_EPOCHS) -> bool:
    """Condition 1: `epoch` is one of the last `window_epochs` through `now`."""
    return now - window_epochs < int(epoch) <= now


@dataclass(frozen=True)
class _LabelIndex:
    """What every comparison reads of one epoch's labels and no head
    changes: the distinct entries (in order of first appearance) and each
    label's, the labels each coldkey voted on, and each label's category
    (as gain_items names it) by code. Built from the LabelPairs' tuples,
    which a frozen LabelPairs never changes, once per LabelPairs object.

    `order` lists the labels category by category (a label's POSITION is
    its place in it); `entry_pos` and `voted_pos` are what `entry` and
    `voted` select, as sorted positions, and `segment` each position's
    category code (`gain_strata_many` works on positions)."""

    entries: tuple[str, ...]
    entry: np.ndarray                    # (labels,) index into `entries`
    voted: Mapping[str, np.ndarray]      # coldkey -> the labels it voted on
    categories: tuple[str, ...]
    by_category: tuple[np.ndarray, ...]  # per category, its labels in order
    order: np.ndarray                    # the labels by category, each in order
    starts: np.ndarray                   # where each category starts in `order`
    entry_pos: tuple[np.ndarray, ...] = ()            # per entry, its labels' positions
    voted_pos: Mapping[str, np.ndarray] = field(default_factory=dict)
    segment: np.ndarray | None = None                 # (labels,) category code per position


_INDEXES: dict[int, tuple] = {}


def _label_index(pairs: LabelPairs) -> _LabelIndex:
    """The LabelPairs' _LabelIndex, kept while the LabelPairs lives (keyed by
    the object, checked by identity, dropped when it is collected)."""
    import weakref

    key = id(pairs)
    hit = _INDEXES.get(key)
    if hit is not None and hit[0]() is pairs:
        return hit[1]
    n = len(pairs.question_ids)
    entries: dict = {}
    entry = np.fromiter((entries.setdefault(m, len(entries)) for m in pairs.models),
                        dtype=np.int64, count=n)
    voted: dict = {}
    for i, v in enumerate(pairs.voters):
        for c in v:
            voted.setdefault(c, []).append(i)
    cats: dict = {}
    category = np.fromiter((cats.setdefault(str(c), len(cats))
                            for c in np.asarray(pairs.categories).tolist()),
                           dtype=np.int64, count=n)
    by_category = tuple(np.flatnonzero(category == c) for c in range(len(cats)))
    sizes = [len(ix) for ix in by_category]
    order = np.concatenate(by_category) if n else np.zeros(0, np.int64)
    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n, dtype=np.int64)
    by_entry = np.argsort(entry[order], kind="stable") if n else np.zeros(0, np.int64)
    entry_pos = tuple(np.split(by_entry, np.cumsum(np.bincount(entry[order],
                                                               minlength=len(entries)))[:-1]))
    voted_ix = {c: np.array(ix, dtype=np.int64) for c, ix in voted.items()}
    index = _LabelIndex(tuple(entries), entry, voted_ix,
                        tuple(cats), by_category, order,
                        np.array([0] + sizes[:-1], dtype=np.int64).cumsum(),
                        entry_pos, {c: np.sort(position[ix]) for c, ix in voted_ix.items()},
                        category[order] if n else np.zeros(0, np.int64))
    _INDEXES[key] = (weakref.ref(pairs, lambda _r, k=key: _INDEXES.pop(k, None)), index)
    return index


def eligible(pairs: LabelPairs, epoch: int, x: Side, y: Side, *, now: int,
             window_epochs: int = SEAT_WINDOW_EPOCHS, raw_x=None, raw_y=None) -> np.ndarray:
    """Per label of `epoch`, whether a comparison of `x` and `y` reads it
    (conditions 1-3 of the module docstring): each side's row for the label's
    entry counts from an epoch at or before it (`Side._starts`: a missing
    row from the side's line when its stored odds `raw_x` / `raw_y` show it
    missing), the label's entry was not provisional, and none of its voters
    owns either head. Decided per entry and per owner, then spread to the
    labels (`_label_index`)."""
    if not isinstance(pairs, LabelPairs):
        raise ValueError(f"epoch {epoch} is not a v4 epoch: its labels carry no voters")
    n = len(pairs.question_ids)
    if not in_window(epoch, now, window_epochs):
        return np.zeros(n, dtype=bool)
    index = _label_index(pairs)
    on = np.ones(len(index.entries), dtype=bool)
    for side, raw in ((x, raw_x), (y, raw_y)):
        for j, start in enumerate(side._starts(index, raw)):
            if start is not None and epoch < start:
                on[j] = False
    keep = on[index.entry] & (pairs.missing != PROVISIONAL_LABEL)
    for owner in sorted({s.owner for s in (x, y) if s.owner is not None}):
        ix = index.voted.get(owner)
        if ix is not None:
            keep[ix] = False
    return keep


def reference(window: Window, *, now: int, window_epochs: int = SEAT_WINDOW_EPOCHS):
    """R, the reference, as a head: odds_score.reference (each entry's rate on
    each benchmark, shrunk to its overall rate, leave one epoch out) over the
    pooled labels of the window's last `window_epochs` epochs."""
    import weakref

    from fugal_subnet import odds_score

    kept = {e: p for e, p in window.pairs.items() if in_window(e, now, window_epochs)}
    # The memorisation check, the seat and the published standing each read
    # R of the same epochs' labels: computed once for the same LabelPairs
    # objects (frozen; checked by identity, held weakly).
    key = (config.ODDS_REF_PRIOR, tuple(sorted((e, id(p)) for e, p in kept.items())))
    hit = _REFERENCES.get(key)
    if hit is not None and all(r() is kept[e] for e, r in hit[0]):
        return hit[1]
    ref = odds_score.reference(kept, config.ODDS_REF_PRIOR)
    try:
        refs = [(e, weakref.ref(p)) for e, p in sorted(kept.items())]
    except TypeError:
        return ref
    _REFERENCES.clear()
    _REFERENCES[key] = (refs, ref)
    return ref


_REFERENCES: dict = {}


def predecessor_odds(window: Window, side: Side, epoch: int) -> np.ndarray | None:
    """A successor's odds for an epoch before its own commit, read from its
    predecessors (SEAT_MINIMAL_V4.md 4.4: its odds on their rows are
    theirs): the stored odds of the most recent predecessor in `side.chain`
    that has odds for `epoch` and was its line's head then (`epoch` before
    the epoch it was head until). None for an epoch at or after the
    successor's commit, for a head with no chain, and when no such
    predecessor has odds for it (its odds are then computed from its own
    bytes, as late odds).

    Exact on every label a comparison reads: before its commit a
    successor's labels are those of its inherited rows (`rows_since`; its
    added rows and missing rows count from its commit), each inherited byte
    for byte from every head of the line since the one that added it. A
    predecessor read at an epoch before it was succeeded carries every row
    counted by then, and a head's odds on a pair depend only on the row for
    the pair's entry (head_odds.head_odds). Its odds for an epoch after it
    was succeeded (the bytes served again) are not read: rows added after
    it may count by then."""
    if side.head is None or not side.chain or epoch >= side.since:
        return None
    for p, until in side.chain:
        if epoch >= until:
            return None
        o = window.odds.get(p, {}).get(epoch)
        if o is not None:
            return o
    return None


def _stored(window: Window, side: Side, epoch: int) -> np.ndarray | None:
    """The stored odds a head's side is scored from in `epoch`: its own, else
    its predecessors' (`predecessor_odds`); None when the window holds
    neither. The one test of whether a comparison reads a head's epoch,
    shared by the per-pair and the batched sums."""
    o = window.odds.get(side.head, {}).get(epoch)
    if o is None:
        o = predecessor_odds(window, side, epoch)
    return o


def _scored(window: Window, ref, side: Side, epoch: int) -> np.ndarray | None:
    """The odds `side` is scored on in `epoch` (int64), with a missing row
    (negative) as odds 0, and an added row before its start as missing
    (`Side.late_added`); None when the window holds none yet. A successor's
    are its predecessors' where it has none of its own (`predecessor_odds`)."""
    if side.head is None:
        return ref.of(epoch).astype(np.int64)
    o = _stored(window, side, epoch)
    if o is None:
        return None
    late = side.late_added(epoch)
    if late:
        # An added row counts from its own start: before it the line had none,
        # so its labels are scored as a missing row is on each pair (the
        # pair's stored `missing`, as `effective` stores it: odds 0 for -1).
        pairs = window.pairs[epoch]
        mask = np.array([m in late for m in pairs.models])
        o = np.where(mask, np.maximum(pairs.missing, -1), o)
    missing = o < 0
    o = o.astype(np.int64)
    if not missing.any():
        return o                              # every row held: nothing to replace
    return np.where(missing, 0, o)


def _raw(window: Window, side: Side, epoch: int):
    """The odds `side` is scored from in `epoch` (`_stored`; None for R or
    none yet): what `eligible` reads a missing row from."""
    return None if side.head is None else _stored(window, side, epoch)


def gain_items(window: Window, ref, x: Side, y: Side, *, now: int,
               window_epochs: int = SEAT_WINDOW_EPOCHS) -> dict[str, list[int]]:
    """{category: items}: per label the comparison reads, y's weighted Brier
    loss minus x's (positive where x was closer to what happened), exact
    integers in loss units. An epoch for which the window holds no odds of a
    head yet is not read (it is read once the head is next verified)."""
    from fugal_subnet.explore_weights import apply

    out: dict[str, list[int]] = {}
    for e in sorted(window.pairs):
        if not in_window(e, now, window_epochs):
            continue
        pairs = window.pairs[e]
        ox, oy = _scored(window, ref, x, e), _scored(window, ref, y, e)
        if ox is None or oy is None:
            continue
        keep = eligible(pairs, e, x, y, now=now, window_epochs=window_epochs,
                        raw_x=_raw(window, x, e), raw_y=_raw(window, y, e))
        if not keep.any():
            continue
        t = pairs.y.astype(np.int64)
        units = (oy - t) ** 2 - (ox - t) ** 2
        if pairs.weight is not None:
            units = apply(units, pairs.weight)
        for c, u in zip(np.asarray(pairs.categories)[keep].tolist(), units[keep].tolist()):
            out.setdefault(str(c), []).append(int(u))
    return out


def gain_strata(window: Window, ref, x: Side, y: Side, *, now: int,
                window_epochs: int = SEAT_WINDOW_EPOCHS) -> dict[str, Stratum]:
    """{category: Stratum(n, total, squares)} of `gain_items`: what
    seat_bound.bound_from_strata reads for "x beats y" (rule A: margin
    ODDS_MARGIN, n0 = ODDS_MIN_PAIRS).

    The same items as `gain_items` (the same int64 expressions on the same
    labels), summed per category with integer array sums that are exact in
    any order (seat_bound.int_sum, int_sum_squares), categories in the order
    `gain_items` first meets them."""
    from fugal_subnet.explore_weights import apply

    out: dict[str, list[int]] = {}
    for e in sorted(window.pairs):
        if not in_window(e, now, window_epochs):
            continue
        for c, n, total, squares in _epoch_sums(window, ref, x, y, e, now, window_epochs, apply):
            acc = out.setdefault(c, [0, 0, 0])
            acc[0] += n
            acc[1] += total
            acc[2] += squares
    return {c: Stratum(n, total, squares) for c, (n, total, squares) in out.items()}


def _epoch_sums(window: Window, ref, x: Side, y: Side, e: int, now: int, window_epochs: int,
                apply) -> list[tuple[str, int, int, int]]:
    """One window epoch's part of `gain_strata` (empty when either side has
    no odds there or no label is read)."""
    pairs = window.pairs[e]
    ox, oy = _scored(window, ref, x, e), _scored(window, ref, y, e)
    if ox is None or oy is None:
        return []
    keep = eligible(pairs, e, x, y, now=now, window_epochs=window_epochs,
                    raw_x=_raw(window, x, e), raw_y=_raw(window, y, e))
    if not keep.any():
        return []
    t = pairs.y.astype(np.int64)
    units = (oy - t) ** 2 - (ox - t) ** 2
    if pairs.weight is not None:
        units = apply(units, pairs.weight)
    return category_sums(_label_index(pairs), units, keep)


# --- many comparisons at once -------------------------------------------------------------
#
# Rule C compares a qualifier candidate with every contender earlier than it,
# so under a field ordered as a ladder the seat makes a number of comparisons
# quadratic in the heads, each over every label of the window. What one
# comparison reads of a head on an epoch (its scored odds, its loss on each
# label, the labels its owner voted on, the entries its rows do not count
# for yet) does not depend on the other head: `gain_strata_many` computes it
# once per head and epoch, and each pair's sums with a few integer array
# passes over the epoch's labels.
#
# THE ITEM. A label's item is apply(Ly - Lx, w) = floor((2 (Ly - Lx) w + S)
# / 2S) (explore_weights.apply, S = WEIGHT_SCALE, even), L the side's loss
# (o - t)^2. With v = L w that is floor((v_y + S/2 - v_x) / S). Write
# v_x = q_x S + r_x and v_y + S/2 = q'_y S + r'_y (0 <= r < S < 2^32) and
# pack K_x = q_x 2^32 + r_x, K'_y = q'_y 2^32 + r'_y: then
#     floor((K'_y - K_x) / 2^32) = q'_y - q_x + floor((r'_y - r_x) / 2^32)
#                                = q'_y - q_x - [r'_y < r_x]
#                                = floor((v_y + S/2 - v_x) / S)      (the item)
# because -S < r'_y - r_x < S. So a pair's item on every label is one int64
# subtraction and the high 32 bits of the result. (The rounding of each
# label's weighted difference is why the pair's sums are not a bilinear form
# of per-head sums: the carry [r'_y < r_x] is a comparison between the two
# heads, so the pair-by-label pass cannot be replaced by a matrix product of
# per-head quantities.) Unweighted labels pack K = L 2^32 on both sides.
#
# NO OVERFLOW, asserted where it is computed: 0 <= q, q' < 2^31 (checked:
# L_max w_max + S/2 < 2^31 S, else the epoch is summed by `_epoch_sums`), so
# 0 <= K < 2^63, |K'_y - K_x| < 2^63 and the item lies in [-2^31, 2^31) (its
# high word, an int32). A pass reads at most _CHUNK = 1,024 labels, so an
# item sum lies within 2^10 2^31 = 2^41. With every |item| at most `bound`
# (max q'_y, max q_x + 1 over the pass), squares are summed m = floor((2^63 -
# 1) / bound^2) >= 1 at a time in int64 (each such sum below 2^63), and the
# partial sums are carried in two non-negative int64 limbs (s >> 31 < 2^32
# and s & (2^31 - 1)), added at most 2^10 per pass and 2^20 passes per pair:
# below 2^52 each. A label taken out (`_corrections`) has its square split
# into limbs below 2^31 before it is summed. The old expressions are exact in
# int64 under the same condition (2 L_max w_max + S < 2^63), so the items are
# those `gain_strata` reads, and every sum is an exact integer whatever the
# order, the chunking or the thread that adds it.
#
# WHICH LABELS. A pair reads the labels neither side excludes (`eligible`:
# the side's row for the entry counts from a later epoch, or the side's owner
# voted on it). Each side's excluded positions are a short sorted list; a
# pair's sums are the sums over every label of the epoch minus those over
# the union of its two lists (the fixed side's, then the other side's not in
# it), so no pair-by-label mask is built.

_SHIFT = 32
_Q_LIMIT = 1 << 31
_INT64_MAX = (1 << 63) - 1
_LIMB = 31
_LIMB_MASK = (1 << _LIMB) - 1
_ROW_BLOCK = 64          # pairs of one side computed together
_CHUNK = 1024            # labels per array pass (cache-sized)
_TILE = 32               # y rows read together by every x of a run (`_tile_sums`)
_X_GROUP = 4             # x's that read a whole tile, computed together
_TILE_CHUNK = 1024       # labels per tile pass (_X_GROUP x _TILE x it: 1 MiB)
# Window epochs summed at once (threads; the sums do not depend on it). One:
# measured at 128 and 256 UIDs, two or four threads were slower (the passes
# are short enough that the interpreter lock serialises them).
_THREADS = 1


def _side_key(s: Side):
    """Every field of a Side that what it is scored on reads: two sides with
    the same key are scored identically on every label."""
    if s.head is None:
        return None
    return (s.head, s.owner, s.since,
            tuple(sorted((str(k), int(v)) for k, v in s.rows_since.items())),
            tuple((str(p), int(u)) for p, u in s.chain),
            tuple(sorted(str(m) for m in s.added)), s.line)


class _EpochSides:
    """One epoch's per-side quantities for `gain_strata_many`: each side's
    packed losses in the roles it is compared in (rows of `mat[role]`), the
    largest q per category (`qmax`), and its excluded positions (CSR in
    `zcat`, `zptr`)."""

    def __init__(self, n_seg: int):
        self.row = {"x": {}, "y": {}}
        self.mat = {}
        self.qmax = {}
        self.zcat = {}
        self.zptr = {}
        self.row_of = {}                 # role -> (sides,) row, -1 for none
        self.n_seg = n_seg


def gain_strata_many(window: Window, ref, requests, *, now: int,
                     window_epochs: int = SEAT_WINDOW_EPOCHS,
                     threads: int | None = None) -> list[dict[str, Stratum]]:
    """`gain_strata` for many (x, y) pairs on one window, R and epoch: one
    {category: Stratum} per request, in request order, with the same
    n, total and squares as `gain_strata(window, ref, x, y, ...)` (the
    categories in sorted order; nothing reads their order). What does not
    depend on the pair is computed once per side and epoch, the rest with
    integer array passes (the comment above). Requests whose y side repeats
    are best given with the y sides in a consistent order (rule C asks each
    candidate about a prefix of one precedence order): those pairs then read
    contiguous rows. `threads` (default _THREADS) only sets how many epochs
    are summed at once."""
    from fugal_subnet.explore_weights import WEIGHT_SCALE, apply

    requests = [(x, y) for x, y in requests]
    if not requests:
        return []
    if WEIGHT_SCALE % 2:
        raise AssertionError("the item's rounding needs an even weight scale")
    key_of: dict = {}
    sides: list[Side] = []

    def side_id(s):
        k = _side_key(s)
        if k not in key_of:
            key_of[k] = len(sides)
            sides.append(s)
        return key_of[k]

    rx = [side_id(x) for x, _ in requests]
    ry = [side_id(y) for _, y in requests]
    epochs = [e for e in sorted(window.pairs) if in_window(e, now, window_epochs)]
    cats = sorted({str(c) for e in epochs for c in (
        _label_index(window.pairs[e]).categories if isinstance(window.pairs[e], LabelPairs)
        else np.asarray(window.pairs[e].categories).tolist())})
    gcat = {c: i for i, c in enumerate(cats)}
    n_req = len(requests)

    # Blocks: a fixed side and the requests it shares; x first, then the
    # requests whose x is alone, by their y.
    by_x: dict[int, list[int]] = {}
    for i, s in enumerate(rx):
        by_x.setdefault(s, []).append(i)
    blocks: list[tuple[int, bool, list[int]]] = []
    alone: dict[int, list[int]] = {}
    for s, ids in by_x.items():
        if len(ids) >= 2:
            blocks.append((s, True, ids))
        else:
            alone.setdefault(ry[ids[0]], []).append(ids[0])
    blocks += [(s, False, ids) for s, ids in alone.items()]
    roles = {"x": list(dict.fromkeys(rx)), "y": list(dict.fromkeys(ry))}
    rx_a, ry_a = np.array(rx, dtype=np.int64), np.array(ry, dtype=np.int64)
    plan = [(f, f_is_x, np.array(ids, dtype=np.int64), (ry_a if f_is_x else rx_a)[ids])
            for f, f_is_x, ids in blocks]

    work = []
    for e in epochs:
        pairs = window.pairs[e]
        have = [s.head is None or _stored(window, s, e) is not None for s in sides]
        have_a = np.array(have, dtype=bool)
        live = np.flatnonzero(have_a[rx_a] & have_a[ry_a]).tolist()
        if not live:
            continue
        if not isinstance(pairs, LabelPairs):
            raise ValueError(f"epoch {e} is not a v4 epoch: its labels carry no voters")
        if len(pairs.question_ids) == 0:
            continue
        work.append((e, have, live))

    def run(todo, acc, extra) -> None:
        for e, have, live in todo:
            _epoch_requests(window, ref, requests, e, now, window_epochs, sides, roles, have,
                            live, plan, gcat, acc, extra, WEIGHT_SCALE, apply)

    def blank():
        return ({k: np.zeros((n_req, max(1, len(cats))), dtype=np.int64)
                 for k in ("n", "t", "hi", "lo")}, {})

    acc, extra = blank()
    threads = _THREADS if threads is None else int(threads)
    if threads <= 1 or len(work) < 2:
        run(work, acc, extra)
    else:
        # Epochs are independent: each thread sums its own (every sum is an
        # exact integer, so the split and the order of the additions change
        # nothing), and numpy's array passes run outside the interpreter lock.
        from concurrent.futures import ThreadPoolExecutor

        parts = [blank() for _ in range(min(threads, len(work)))]
        with ThreadPoolExecutor(len(parts)) as pool:
            for f in [pool.submit(run, work[k::len(parts)], *parts[k]) for k in range(len(parts))]:
                f.result()
        for part_acc, part_extra in parts:
            for k in acc:
                acc[k] += part_acc[k]
            for key, v in part_extra.items():
                a = extra.setdefault(key, [0, 0, 0])
                for j in range(3):
                    a[j] += v[j]

    out: list[dict[str, Stratum]] = []
    ns, ts, his, los = (acc[k].tolist() for k in ("n", "t", "hi", "lo"))
    for i in range(n_req):
        row = {}
        for g, c in enumerate(cats):
            n, total = ns[i][g], ts[i][g]
            squares = (his[i][g] << _LIMB) + los[i][g]
            more = extra.get((i, c))
            if more is not None:
                n, total, squares = n + more[0], total + more[1], squares + more[2]
            if n:
                row[c] = Stratum(n, total, squares)
        out.append(row)
    return out


def _epoch_requests(window, ref, requests, e, now, window_epochs, sides, roles, have, live, plan,
                    gcat, acc, extra, scale, apply) -> None:
    """One window epoch's part of `gain_strata_many`, added to `acc` (and to
    `extra` when the epoch is summed the old way)."""
    pairs = window.pairs[e]
    ep = _epoch_sides(window, ref, pairs, e, sides, roles, have, scale)
    if ep is None:                         # values past the packed range: the old sums
        for i in live:
            for c, n, total, squares in _epoch_sums(window, ref, requests[i][0], requests[i][1],
                                                    e, now, window_epochs, apply):
                a = extra.setdefault((i, c), [0, 0, 0])
                a[0] += n
                a[1] += total
                a[2] += squares
        return
    index = _label_index(pairs)
    seg_g = np.array([gcat[str(c)] for c in index.categories], dtype=np.int64)

    def add(ids, n, t, hi, lo):
        r = ids[:, None]
        if n is not None:
            acc["n"][r, seg_g] += n
        acc["t"][r, seg_g] += t
        acc["hi"][r, seg_g] += hi
        acc["lo"][r, seg_g] += lo

    tiles: dict[int, list] = {}
    for f, f_is_x, ids, other in plan:
        f_role, o_role = ("x", "y") if f_is_x else ("y", "x")
        frow = int(ep.row_of[f_role][f])
        if frow < 0:
            continue
        rows = ep.row_of[o_role][other]
        on = rows >= 0
        ids, rows = ids[on], rows[on]
        if not len(ids):
            continue
        r0 = int(rows[0])
        contiguous = bool(np.array_equal(rows, np.arange(r0, r0 + len(rows))))
        add(ids, *_corrections(ep, index, frow, f_is_x, rows, contiguous))
        if f_is_x and contiguous:
            # rule C's shape: x against a run of y rows, shared with other
            # x's (`_tile_sums`)
            for t0 in range(r0 - r0 % _TILE, r0 + len(rows), _TILE):
                lo_, hi_ = max(r0, t0), min(r0 + len(rows), t0 + _TILE)
                tiles.setdefault(t0, []).append((frow, lo_, hi_, ids[lo_ - r0:hi_ - r0]))
            continue
        for k in range(0, len(ids), _ROW_BLOCK):
            add(ids[k:k + _ROW_BLOCK], None,
                *_block_sums(ep, index, frow, f_is_x, rows[k:k + _ROW_BLOCK], contiguous))
    for t0, items in tiles.items():
        for ids, t, hi, lo in _tile_sums(ep, index, t0, items):
            add(ids, None, t, hi, lo)


def _epoch_sides(window: Window, ref, pairs: LabelPairs, e: int, sides: list[Side],
                 roles: Mapping[str, list[int]], have: list[bool], scale: int):
    """The `_EpochSides` of epoch `e` for the sides that have odds there, or
    None when a loss times a weight leaves the packed range (the module
    comment: then the epoch is summed per pair the old way)."""
    index = _label_index(pairs)
    order = index.order
    t = pairs.y.astype(np.int64)[order]
    w = None if pairs.weight is None else np.asarray(pairs.weight, dtype=np.int64)[order]
    used = sorted({s for r in roles.values() for s in r if have[s]})
    loss = {}
    worst = 0
    for s in used:
        o = _scored(window, ref, sides[s], e)[order]
        d = o - t
        worst = max(worst, int(np.abs(d).max()))
        loss[s] = d
    l_max = worst * worst
    half = scale // 2
    if w is None:
        if l_max >= _Q_LIMIT:
            return None
    else:
        w_max = int(w.max())
        if l_max * w_max + half >= _Q_LIMIT * scale or 2 * l_max * w_max + scale > _INT64_MAX:
            return None
    starts = index.starts
    n_lab = len(order)
    provisional = np.flatnonzero(np.asarray(pairs.missing)[order] == PROVISIONAL_LABEL)
    ep = _EpochSides(len(starts))
    excluded, value = {}, {}
    for s in used:
        side = sides[s]
        z = []
        if len(provisional):
            z.append(provisional)              # read by no comparison (`eligible`)
        if side.head is not None:
            if side.owner in index.voted_pos:
                z.append(index.voted_pos[side.owner])
            # the entries whose labels this side does not read yet: exactly
            # `eligible`'s per-entry starts (Side._starts on the odds it is
            # scored from)
            row_starts = side._starts(index, _stored(window, side, e))
            z += [index.entry_pos[j] for j, st in enumerate(row_starts) if e < st]
        zs = np.unique(np.concatenate(z)) if z else np.zeros(0, np.int64)
        d = loss.pop(s)
        if len(zs) == n_lab:
            continue                    # it reads no label of this epoch with anyone
        v = d * d                                   # <= l_max < 2^63
        if w is not None:
            v *= w                                  # <= l_max w_max < 2^31 S
        excluded[s], value[s] = zs, v
    divided: dict = {}
    # Each role's rows in the order its sides first appear in the requests,
    # so a run of requests reads a run of rows.
    for role in ("x", "y"):
        mine = [s for s in roles[role] if s in value]
        ep.row_of[role] = np.full(len(sides), -1, dtype=np.int64)
        if not mine:
            continue
        mat = np.empty((len(mine), n_lab), dtype=np.int64)
        qmax = np.empty((len(mine), len(starts)), dtype=np.int64)
        for r, s in enumerate(mine):
            v = value[s]
            if w is None:
                q = v
                np.left_shift(v, _SHIFT, out=mat[r])
            else:
                if s not in divided:
                    divided[s] = np.divmod(v, scale)            # v = q S + rem
                q, rem = divided[s]
                if role == "y":
                    # v + S/2 = q' S + rem': q' = q + [rem >= S/2], rem' =
                    # rem + S/2 - S [rem >= S/2]
                    carry = rem >= half
                    q = q + carry
                    rem = rem + half
                    rem -= scale * carry
                np.left_shift(q, _SHIFT, out=mat[r])
                mat[r] |= rem
            qmax[r] = np.maximum.reduceat(q, starts)
            ep.row[role][s] = r
            ep.row_of[role][s] = r
        zs = [excluded[s] for s in mine]
        ep.mat[role], ep.qmax[role] = mat, qmax
        ep.zptr[role] = np.concatenate([[0], np.cumsum([len(z) for z in zs])]).astype(np.int64)
        ep.zcat[role] = np.concatenate(zs) if int(ep.zptr[role][-1]) else np.zeros(0, np.int64)
    return ep


def _high(d: np.ndarray, out: np.ndarray) -> np.ndarray:
    """floor(d / 2^32) of an int64 array into `out` (int64)."""
    if np.little_endian and d.strides[-1] == 8:
        np.copyto(out, d.view(np.int32)[..., 1::2])   # the high word, as int32
    else:
        np.right_shift(d, _SHIFT, out=out)
    return out


def _limbs(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return s >> _LIMB, s & _LIMB_MASK


def _segments(index: _LabelIndex, n_lab: int) -> tuple[list[int], list[int]]:
    starts = index.starts.tolist()
    return starts, starts[1:] + [n_lab]


def _chunk(bound: int) -> int:
    """How many items of magnitude at most `bound` may be squared and summed
    in int64: their sum of squares stays below 2^63 (the comment above)."""
    assert 0 < bound <= _Q_LIMIT
    m = max(1, _INT64_MAX // (bound * bound))
    assert m * bound * bound <= _INT64_MAX
    return m


def _squares(it: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """The sum of squares of `it` along its last axis, as (high limb, low
    limb): summed m items at a time in int64 (each such sum is below 2^63,
    `_chunk`), the partial sums split into limbs (each below 2^32) and
    those added: one einsum over the pass whatever m is."""
    n = it.shape[-1]
    k = n // m
    full = k * m
    hi = np.zeros(it.shape[:-1], dtype=np.int64)
    lo = np.zeros(it.shape[:-1], dtype=np.int64)
    if k:
        a = it[..., :full].reshape(it.shape[:-1] + (k, m))
        part = np.einsum("...cl,...cl->...c", a, a)
        hi += (part >> _LIMB).sum(axis=-1)
        lo += (part & _LIMB_MASK).sum(axis=-1)
    if full < n:
        rest = np.einsum("...l,...l->...", it[..., full:], it[..., full:])
        hi += rest >> _LIMB
        lo += rest & _LIMB_MASK
    return hi, lo


def _block_sums(ep: _EpochSides, index: _LabelIndex, frow: int, f_is_x: bool, rows: np.ndarray,
                contiguous: bool):
    """(total, squares high limb, squares low limb), each (pairs,
    categories) int64: the sums over EVERY label of the epoch of the pairs
    of the fixed side's row `frow` (x when `f_is_x`, else y) with each of
    the other role's `rows` (`_corrections` takes the excluded labels out)."""
    f_role, o_role = ("x", "y") if f_is_x else ("y", "x")
    fvec = ep.mat[f_role][frow]
    mat = ep.mat[o_role]
    b = len(rows)
    block = mat[int(rows[0]):int(rows[0]) + b] if contiguous else mat[rows]
    starts, ends = _segments(index, len(fvec))
    n_seg = len(starts)
    tot = np.zeros((b, n_seg), dtype=np.int64)
    hi = np.zeros((b, n_seg), dtype=np.int64)
    lo = np.zeros((b, n_seg), dtype=np.int64)
    q_other = ep.qmax[o_role][rows].max(axis=0).tolist()
    q_f = ep.qmax[f_role][frow].tolist()
    d_buf = np.empty((b, _CHUNK), dtype=np.int64)
    it_buf = np.empty((b, _CHUNK), dtype=np.int64)
    for s in range(n_seg):
        # every item lies in [-(q_x max) - 1, q'_y max]
        m = _chunk(max(q_other[s], q_f[s] + 1) if f_is_x else max(q_f[s], q_other[s] + 1))
        for c0 in range(starts[s], ends[s], _CHUNK):
            c1 = min(ends[s], c0 + _CHUNK)
            d = d_buf[:, :c1 - c0]
            if f_is_x:
                np.subtract(block[:, c0:c1], fvec[c0:c1], out=d)
            else:
                np.subtract(fvec[c0:c1], block[:, c0:c1], out=d)
            it = _high(d, it_buf[:, :c1 - c0])
            tot[:, s] += it.sum(axis=1)
            h, low = _squares(it, m)
            hi[:, s] += h
            lo[:, s] += low
    return tot, hi, lo


def _tile_sums(ep: _EpochSides, index: _LabelIndex, t0: int, items: list):
    """`_block_sums` for rule C's shape: x rows each against a run of the y
    rows of the tile [t0, t0 + _TILE), the tile's labels read once per chunk
    for every x (cache-sized passes, _X_GROUP x's at a time when they read
    the whole tile). Yields (request ids, total, squares high and low limbs)
    per item."""
    ky, kx = ep.mat["y"], ep.mat["x"]
    t1 = min(t0 + _TILE, len(ky))
    starts, ends = _segments(index, ky.shape[1])
    n_seg = len(starts)
    xrows = np.array([frow for frow, _a, _b, _ids in items], dtype=np.int64)
    q_y = ep.qmax["y"][t0:t1].max(axis=0).tolist()
    q_x = ep.qmax["x"][xrows].max(axis=0).tolist()
    full = [k for k, (_f, a, b, _ids) in enumerate(items) if a == t0 and b == t1]
    part = [k for k, (_f, a, b, _ids) in enumerate(items) if not (a == t0 and b == t1)]
    width = t1 - t0
    sums = {k: [np.zeros((items[k][2] - items[k][1], n_seg), dtype=np.int64) for _ in range(3)]
            for k in part}
    full_sums = [np.zeros((len(full), width, n_seg), dtype=np.int64) for _ in range(3)]
    y_buf = np.empty((width, _TILE_CHUNK), dtype=np.int64)
    d_buf = np.empty((_X_GROUP, width, _TILE_CHUNK), dtype=np.int64)
    it_buf = np.empty((_X_GROUP, width, _TILE_CHUNK), dtype=np.int64)
    full_x = xrows[full]
    for s in range(n_seg):
        m = _chunk(max(q_y[s], q_x[s] + 1))
        for c0 in range(starts[s], ends[s], _TILE_CHUNK):
            c1 = min(ends[s], c0 + _TILE_CHUNK)
            L = c1 - c0
            yb = y_buf[:, :L]
            np.copyto(yb, ky[t0:t1, c0:c1])
            for g in range(0, len(full), _X_GROUP):
                xg = full_x[g:g + _X_GROUP]
                d = d_buf[:len(xg), :, :L]
                np.subtract(yb[None, :, :], kx[xg, c0:c1][:, None, :], out=d)
                it = _high(d, it_buf[:len(xg), :, :L])
                full_sums[0][g:g + len(xg), :, s] += it.sum(axis=2)
                h, low = _squares(it, m)
                full_sums[1][g:g + len(xg), :, s] += h
                full_sums[2][g:g + len(xg), :, s] += low
            for k in part:
                frow, a, b, _ids = items[k]
                d = d_buf[0, :b - a, :L]
                np.subtract(yb[a - t0:b - t0], kx[frow, c0:c1], out=d)
                it = _high(d, it_buf[0, :b - a, :L])
                sums[k][0][:, s] += it.sum(axis=1)
                h, low = _squares(it, m)
                sums[k][1][:, s] += h
                sums[k][2][:, s] += low
    for j, k in enumerate(full):
        yield items[k][3], full_sums[0][j], full_sums[1][j], full_sums[2][j]
    for k in part:
        yield items[k][3], sums[k][0], sums[k][1], sums[k][2]


def _corrections(ep: _EpochSides, index: _LabelIndex, frow: int, f_is_x: bool, rows: np.ndarray,
                 contiguous: bool):
    """(n, total, squares high limb, squares low limb), each (pairs,
    categories) int64, of the pairs of row `frow` with `rows` (as
    `_block_sums`): n the labels each pair reads, and minus the sums over
    the labels it does not read (the union of the two sides' excluded
    positions), which `_block_sums` or `_tile_sums` counted."""
    f_role, o_role = ("x", "y") if f_is_x else ("y", "x")
    fvec = ep.mat[f_role][frow]
    mat = ep.mat[o_role]
    b = len(rows)
    starts, ends = _segments(index, len(fvec))
    n_seg = len(starts)
    seg_len = np.array([e - a for a, e in zip(starts, ends)], dtype=np.int64)
    n = np.broadcast_to(seg_len, (b, n_seg)).copy()
    tot = np.zeros((b, n_seg), dtype=np.int64)
    hi = np.zeros((b, n_seg), dtype=np.int64)
    lo = np.zeros((b, n_seg), dtype=np.int64)

    def remove(r_ids, pos):
        """Take the labels at (rows[r_ids[k]], pos[k]) out; the pairs run
        row by row, positions increasing in each."""
        if not len(pos):
            return
        if f_is_x:
            d = mat[rows[r_ids], pos] - fvec[pos]
        else:
            d = fvec[pos] - mat[rows[r_ids], pos]
        it = d >> _SHIFT
        sq_hi, sq_lo = _limbs(it * it)              # each square < 2^62
        seg = index.segment[pos]
        key = r_ids * n_seg + seg
        first = np.concatenate([[0], np.flatnonzero(np.diff(key)) + 1])
        count = np.diff(np.concatenate([first, [len(key)]]))
        where_r, where_s = r_ids[first], seg[first]
        n[where_r, where_s] -= count
        tot[where_r, where_s] -= np.add.reduceat(it, first)
        hi[where_r, where_s] -= np.add.reduceat(sq_hi, first)
        lo[where_r, where_s] -= np.add.reduceat(sq_lo, first)

    zf = ep.zcat[f_role][ep.zptr[f_role][frow]:ep.zptr[f_role][frow + 1]]
    if len(zf):
        # the fixed side's positions, the same for every pair: one 2-D pass
        r0 = int(rows[0])
        other = mat[r0:r0 + b][:, zf] if contiguous else mat[rows][:, zf]
        d = other - fvec[zf] if f_is_x else fvec[zf] - other
        it = d >> _SHIFT
        sq_hi, sq_lo = _limbs(it * it)              # each square < 2^62
        seg = index.segment[zf]
        first = np.concatenate([[0], np.flatnonzero(np.diff(seg)) + 1])
        where_s = seg[first]
        n[:, where_s] -= np.diff(np.concatenate([first, [len(zf)]]))[None, :]
        tot[:, where_s] -= np.add.reduceat(it, first, axis=1)
        hi[:, where_s] -= np.add.reduceat(sq_hi, first, axis=1)
        lo[:, where_s] -= np.add.reduceat(sq_lo, first, axis=1)
    zptr, zcat = ep.zptr[o_role], ep.zcat[o_role]
    if contiguous:
        r0 = int(rows[0])
        lens = np.diff(zptr[r0:r0 + b + 1])
        pos = zcat[zptr[r0]:zptr[r0 + b]]
    else:
        lens = zptr[rows + 1] - zptr[rows]
        pos = (np.concatenate([zcat[zptr[r]:zptr[r + 1]] for r in rows.tolist()]) if lens.sum()
               else np.zeros(0, np.int64))
    if len(pos):
        r_ids = np.repeat(np.arange(b, dtype=np.int64), lens)
        if len(zf):
            outside = np.ones(len(fvec), dtype=bool)
            outside[zf] = False
            keep = outside[pos]
            r_ids, pos = r_ids[keep], pos[keep]
        remove(r_ids, pos)
    return n, tot, hi, lo


def category_sums(index: _LabelIndex, units: np.ndarray, keep: np.ndarray
                  ) -> list[tuple[str, int, int, int]]:
    """[(category, n, sum, sum of squares)] of the kept labels' int64
    `units`, exact Python ints, the categories in the order of their first
    kept label (as gain_items meets them).

    One pass for every category: the labels in category order
    (`_LabelIndex.order`), the dropped ones zeroed (they add nothing), each
    value split as v = h 2^s + l (seat_bound.int_sum_squares) and the parts
    summed per category with np.add.reduceat. No int64 sum can overflow:
    with n kept labels of at most b bits, every partial sum of v, h^2, h l
    and l^2 lies within n x 2^(b + 1), checked below 2^63; otherwise each
    category is summed by seat_bound.int_sum / int_sum_squares."""
    from fugal_subnet.seat_bound import INT64_LIMIT, _magnitude, int_sum, int_sum_squares

    order, starts = index.order, index.starts
    kept = keep[order]
    first = np.minimum.reduceat(np.where(kept, order, len(keep)), starts).tolist()
    met = sorted((f, c) for c, f in enumerate(first) if f < len(keep))
    v = np.where(kept, units[order], 0)
    counts = np.add.reduceat(kept.astype(np.int64), starts).tolist()
    n = sum(counts)
    b = _magnitude(v).bit_length()
    if n * (1 << (b + 1)) >= INT64_LIMIT:
        out = []
        for _f, c in met:
            sel = units[index.by_category[c][keep[index.by_category[c]]]]
            out.append((index.categories[c], len(sel), int_sum(sel), int_sum_squares(sel)))
        return out
    s = (b + 1) // 2
    h = v >> s
    lo = v & ((1 << s) - 1)
    totals = np.add.reduceat(v, starts).tolist()
    hh = np.add.reduceat(h * h, starts).tolist()
    hl = np.add.reduceat(h * lo, starts).tolist()
    ll = np.add.reduceat(lo * lo, starts).tolist()
    return [(index.categories[c], counts[c], totals[c],
             (hh[c] << (2 * s)) + (hl[c] << (s + 1)) + ll[c]) for _f, c in met]


def skill_strata(window: Window, ref, x: Side, *, now: int,
                 window_epochs: int = SEAT_WINDOW_EPOCHS) -> dict[str, Stratum]:
    """x's gain over R, by category."""
    return gain_strata(window, ref, x, REFERENCE, now=now, window_epochs=window_epochs)


# --- the reveal ---------------------------------------------------------------------------


def labels_for_reveal(pairs: LabelPairs) -> dict:
    """The v4 facts of an epoch's labels, aligned with `exploration_pool`:
    the voting coldkeys (as indexes into one sorted list) and the odds a
    missing row is scored at."""
    coldkeys = sorted({c for v in pairs.voters for c in v})
    index = {c: i for i, c in enumerate(coldkeys)}
    return {"coldkeys": coldkeys,
            "voters": [[index[c] for c in v] for v in pairs.voters],
            "missing_odds": [int(m) for m in pairs.missing.tolist()]}


def labels_from_reveal(doc, pairs: EpochPairs, what: str = "labels_v4") -> LabelPairs:
    """The inverse of `labels_for_reveal` on the epoch's pool, validated: a
    published reveal is someone else's file (I2)."""
    if not isinstance(doc, Mapping) or set(doc) != {"coldkeys", "voters", "missing_odds"}:
        raise ValueError(f"`{what}` must hold coldkeys, voters and missing_odds")
    coldkeys, voters, missing = doc["coldkeys"], doc["voters"], doc["missing_odds"]
    n = len(pairs.question_ids)
    if (not isinstance(coldkeys, list) or coldkeys != sorted(set(coldkeys))
            or not all(isinstance(c, str) and c for c in coldkeys)):
        raise ValueError(f"`{what}.coldkeys` must be sorted distinct coldkeys")
    if not isinstance(voters, list) or len(voters) != n or not all(
            isinstance(v, list) and v and v == sorted(set(v))
            and all(type(i) is int and 0 <= i < len(coldkeys) for i in v) for v in voters):
        raise ValueError(f"`{what}.voters` must hold one sorted list of coldkey indexes per pair")
    if not isinstance(missing, list) or len(missing) != n or not all(
            type(m) is int and PROVISIONAL_LABEL <= m <= ODDS_SCALE for m in missing):
        raise ValueError(f"`{what}.missing_odds` must hold one value in -2..{ODDS_SCALE} per pair")
    return label_pairs(pairs, [tuple(coldkeys[i] for i in v) for v in voters], missing)


def late_for_reveal(late: Mapping[str, Mapping[int, np.ndarray]]) -> dict:
    """{head: [[epoch index, odds as published (v2_epoch.odds_b64)], ...]}."""
    from fugal_subnet.v2_epoch import odds_b64

    return {h: [[int(e), odds_b64(o)] for e, o in sorted(late[h].items())]
            for h in sorted(late) if late[h]}


def late_from_reveal(doc, window: Window, what: str = "late_odds") -> dict[str, dict[int, np.ndarray]]:
    """The inverse of `late_for_reveal`, validated against the window's pairs.
    An epoch the window no longer holds (it left the window) is skipped."""
    from fugal_subnet.v2_epoch import odds_from_b64

    if not isinstance(doc, Mapping):
        raise ValueError(f"`{what}` must be a map")
    out: dict[str, dict[int, np.ndarray]] = {}
    for h in sorted(doc):
        rows = doc[h]
        if not isinstance(rows, list) or not all(
                isinstance(r, list) and len(r) == 2 and type(r[0]) is int and isinstance(r[1], str)
                for r in rows):
            raise ValueError(f"`{what}[{str(h)[:12]}]` must be [epoch, odds] rows")
        for e, text in rows:
            pairs = window.pairs.get(e)
            if pairs is None:
                continue
            if not isinstance(pairs, LabelPairs):
                raise ValueError(f"`{what}[{h[:12]}]`: epoch {e} is not a v4 epoch")
            out.setdefault(str(h), {})[e] = odds_from_b64(text, len(pairs.question_ids))
    return out


def standing_for_reveal(window: Window, serving: Mapping[str, int], owners: Mapping[str, str],
                        since: Mapping[str, int], *, now: int) -> dict:
    """Per serving head, its labels and mean gain over R on them (Brier
    points), and the window epochs it has odds for: what this package
    computes, published every v4 epoch. The seat package publishes the
    bound."""
    from fractions import Fraction

    ref = reference(window, now=now)
    out = {}
    heads = [h for h in sorted(serving) if h in owners and h in since]
    # skill_strata of every head, computed together (gain_strata_many)
    strata = gain_strata_many(window, ref, [(Side(h, owners[h], int(since[h])), REFERENCE)
                                            for h in heads], now=now)
    for h, s in zip(heads, strata):
        n = sum(x.n for x in s.values())
        total = sum(x.total for x in s.values())
        out[h] = {"uid": int(serving[h]), "labels": n,
                  "gain_points": float(Fraction(total, n) / config.BOUND_UNITS_PER_POINT_V4) if n else 0.0,
                  "odds_epochs": sorted(window.odds.get(h, {}))}
    return out


def window_to_arrays(window: Window) -> tuple[dict, dict]:
    """(meta, arrays) of the window's v4 facts, for v2_epoch.save_window:
    each LabelPairs epoch's voters and missing odds, and the odds of heads
    the window holds no answers of (heads verified without the scored
    questions, and late odds of heads that never answered)."""
    meta: dict = {}
    arrays: dict = {}
    labels = {e: p for e, p in window.pairs.items() if isinstance(p, LabelPairs)}
    if labels:
        meta["labels_v4"] = {str(e): {"voters": [list(v) for v in p.voters]}
                             for e, p in sorted(labels.items())}
        for e, p in labels.items():
            arrays[f"x_m_{e}"] = p.missing
    only = sorted(h for h in window.odds if h not in window.heads and window.odds[h])
    if only:
        meta["odds_only"] = {h: {"key": i, "epochs": sorted(window.odds[h])}
                             for i, h in enumerate(only)}
        for i, h in enumerate(only):
            for e, o in window.odds[h].items():
                arrays[f"q_{i}_{e}"] = o
    return meta, arrays


def window_from_arrays(window: Window, meta: Mapping, z) -> Window:
    """The inverse of `window_to_arrays` on a loaded window."""
    pairs = dict(window.pairs)
    for e, v in (meta.get("labels_v4") or {}).items():
        e = int(e)
        pairs[e] = label_pairs(pairs[e], [tuple(x) for x in v["voters"]],
                               np.array(z[f"x_m_{e}"], dtype=np.int16))
    odds = {h: dict(per) for h, per in window.odds.items()}
    for h, v in (meta.get("odds_only") or {}).items():
        odds[h] = {int(e): np.array(z[f"q_{v['key']}_{e}"], dtype=np.int16) for e in v["epochs"]}
    return dataclasses.replace(window, pairs=pairs, odds=odds)


__all__ = [
    "REFERENCE", "SEAT_WINDOW_EPOCHS", "LabelPairs", "Side", "effective", "eligible",
    "gain_items", "gain_strata", "in_window", "label_pairs", "labels_for_reveal",
    "labels_from_reveal", "late_for_reveal", "late_from_reveal", "late_odds", "missing_odds",
    "odds_on", "reference", "skill_strata", "standing_for_reveal", "window_from_arrays",
    "window_to_arrays", "with_odds", "predecessor_odds",
]
