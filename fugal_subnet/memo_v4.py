"""Protocol v4: the memorisation check and the unpublished reserve
(docs/decisions/SEAT_MINIMAL_V4.md, section 6; package S5).

A head is small on purpose (a linear layer on a locked public encoder), so it
has little room to memorise the public question pool. This module is the
second line: it refuses a head whose skill comes from questions whose
outcomes were already published, and it keeps a reserve of questions whose
outcomes nobody has published, so the comparison has something to compare.

THE PUBLISHED SET (`Published`). Per pool question, the first epoch whose
reveal published an outcome on it: a scored question, an exploration answer
or a pooled label (`published_ids`). It is consensus state: seeded at the v4
activation from a pinned file built from named reveals
(`routing_data/published_v4.json`, scripts/build_published_v4.py), then
folded forward (`fold`) by one rule every node follows, the root, every
other validator and every miner's TD alike: the root's reveals, each at the
epoch the root's commitment read pinned at a boundary names it
(fugal_subnet/memo_sync.py). `fold_reveal` and `rebuild` fold a sequence of
reveals as given (the seed's build, the tests). The view an epoch reads has
its digest in each v4 reveal (`memo_v4`), so a node whose state differs is
visible in the epoch it diverges. Questions published
before the seed's last reveal are recorded at that epoch: every v4 head is
committed after it, so no v4 rule reads an earlier first epoch.

THE RESERVE AND THE FRESH DRAW. An epoch E reads the set as published
through E - 2 (every node holds that reveal at E's boundary, as it holds the
menu committed in it). The RESERVE is the pool's questions not published
through E - 2. The epoch spends
    F = ceil(R / FRESH_RESERVE_EPOCHS_V4)        (R the reserve's size)
of them, drawn once for the whole field by the epoch nonce on the pinned v4
mix as far as the reserve allows (`fresh_questions`), and EVERY hotkey's
private exploration set contains all F: what a hotkey count changes is how
many labels the fresh questions give, never how many questions are spent. So
the reserve shrinks by a 1/1200 share an epoch (rounded up), whatever the
number of hotkeys: a reserve of 1,200 or more lasts 60 days (1,200 epochs),
one of 2,400 or more keeps a quarter of itself to then, and questions a
release adds enter it unpublished and raise F at once. Every
other draw of the epoch (the scored slice, the rest of each private set,
proposal measurement) reads only questions published through E - 2
(`ReserveIndex`), so the reserve is spent by the fresh draw alone. A
category with fewer than `PUBLISHED_MIN_PER_CATEGORY` published questions is
left out of those draws (its share goes to the other categories, as the
slicer does for an absent category), so no draw can exhaust it; it is
reached by the fresh draw until enough of it is published. A category whose
reserve is empty gives its fresh slots to the categories that still have
one.

The private set keeps its size and its split: the fresh questions are
members of it like any other, ranked into the cost-aware and coverage parts
by the miner's own keyed rank, and each sent to an entry drawn by the part's
shares. A question's entry is therefore drawn with exactly the probability
the pair's weight undoes (explore_weights.SharesV4), fresh or not.

THE CHECK (`memo_evidence`). For a head H, the labels of its comparison with
R (labels_v4.eligible: the window, after the row's commit, not its own
operator's) split into SEEN (the question was published in an epoch before
the row counts from, `Side.row_since`) and UNSEEN (the rest). Within each
benchmark with at least MEMO_MIN_QUESTIONS_V4 questions in both parts, the
gap is H's mean skill over R on seen labels minus that on unseen ones
(skill: R's weighted Brier loss minus H's, labels_v4's units); benchmarks
are averaged with weights 1 / (1/n_seen + 1/n_unseen), n in labels. H is
REFUSED when, with MEMO_MIN_LABELS_V4 labels in each part, the one bound
(seat_bound.bound_from_strata, n0 = MEMO_MIN_LABELS_V4) passes on the gap
with margin MEMO_MARGIN_POINTS_V4. The bound's items are the questions, not
the labels: every hotkey answers every fresh question, so the labels of one
question are correlated, and a question's labels enter as one item (their
weighted sum, scaled by the label's share of the gap), each item in the
stratum of its benchmark and part. The scaling is exact (a common integer
multiple of every coefficient), so the verdict is the bound's inequality on
the gap itself, in integers and Fractions.

Nothing on the v1-v3 path imports this module.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType

import numpy as np

from fugal_subnet import config
from fugal_subnet.benchmarks.slicer import PoolIndex, select_private_explore
from fugal_subnet.seat_bound import Bound, Stratum, bound_from_strata

SEED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "routing_data",
                         "published_v4.json")
SEED_FORMAT = "published-v4-v1"
STATE_FORMAT = "published-v4-state-v1"

FRESH_RESERVE_EPOCHS = config.FRESH_RESERVE_EPOCHS_V4
MEMO_MARGIN_UNITS = config.MEMO_MARGIN_POINTS_V4 * config.BOUND_UNITS_PER_POINT_V4
MEMO_MIN_LABELS = config.MEMO_MIN_LABELS_V4
MEMO_MIN_QUESTIONS = config.MEMO_MIN_QUESTIONS_V4
# Epochs between the last reveal an epoch's draws read and the epoch: the
# reveal of E - 2 is out at E's boundary on every node.
READ_LAG = 2

SEEN = "seen"
UNSEEN = "unseen"


# Questions a draw once also excluded for a proposer's measurement, 150 an
# epoch. No proposer measures any more (FAST_CROWN_V4.md 18: the field
# measures a new entry through its exploration); the room is kept so that the
# rule for which categories enter the published draws, and so every draw of
# the published set, is exactly what it was.
_FORMER_MEASURE_ROOM = 150


# A category enters the published draws only with more published questions
# than one draw could ever take or exclude in it: the scored slice, a private
# set and the former measurement room. Below that a private draw could
# run out of questions and come back short, and a short private set would
# change its parts' sizes and so every private pair's weight.
PUBLISHED_MIN_PER_CATEGORY = (config.SLICE_SIZE_V4 + config.EXPLORE_PRIVATE_SIZE_V4
                              + _FORMER_MEASURE_ROOM + 1)


def epoch_index(epoch_id: str) -> int:
    """The number in a published epoch id (ids are read here, never
    formatted: slicer.epoch_id_for_block is the only formatter)."""
    text = str(epoch_id)
    if not (text.startswith("e") and text[1:].isdigit()):
        raise ValueError(f"epoch_id {text!r} is not of the form e<digits>")
    return int(text[1:])


# --- the published set --------------------------------------------------------------------


@dataclass(frozen=True)
class Published:
    """Per question, the first epoch a reveal published an outcome on it
    (`first`; a question absent from it is unpublished), folded through epoch
    `through`. Immutable: `fold` returns a new state."""

    first: Mapping[str, int]
    through: int

    def __post_init__(self):
        if type(self.through) is not int:
            raise ValueError("through must be an epoch index")
        if not isinstance(self.first, MappingProxyType):
            object.__setattr__(self, "first", MappingProxyType(dict(self.first)))
        for q, e in self.first.items():
            if not isinstance(q, str) or not q or type(e) is not int or e > self.through:
                raise ValueError("the published set maps question ids to epochs at or before "
                                 "its last folded epoch")

    def first_epoch(self, question_id: str) -> int | None:
        return self.first.get(question_id)

    def published_by(self, question_id: str, epoch: int) -> bool:
        """Whether the question was published in a reveal of `epoch` or before.
        The state must be folded through `epoch`."""
        self.require(epoch)
        f = self.first.get(question_id)
        return f is not None and f <= epoch

    def require(self, epoch: int) -> None:
        if int(epoch) > self.through:
            raise ValueError(f"the published set is folded through epoch {self.through}, not "
                             f"{int(epoch)}: fold the reveals in between first")

    def view(self, epoch: int) -> Published:
        """The state as it was after folding epoch `epoch`: what an epoch that
        reads through `epoch` reads, whatever was folded since."""
        self.require(epoch)
        e = int(epoch)
        return Published({q: f for q, f in self.first.items() if f <= e}, e)

    def to_doc(self) -> dict:
        by: dict[int, list[str]] = {}
        for q, e in self.first.items():
            by.setdefault(e, []).append(q)
        return {"format": STATE_FORMAT, "through": self.through,
                "first": {str(e): sorted(by[e]) for e in sorted(by)}}

    @functools.cached_property
    def digest(self) -> str:
        """sha256 of the canonical document: the published set two nodes are
        compared on."""
        text = json.dumps(self.to_doc(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def published_from_doc(doc) -> Published:
    """The inverse of `Published.to_doc`, strictly read (a state file or a
    peer's is someone else's bytes): any value of another type than the one
    expected is a ValueError, never another exception."""
    if (not isinstance(doc, Mapping) or set(doc) != {"format", "through", "first"}
            or doc["format"] != STATE_FORMAT or type(doc["through"]) is not int
            or not isinstance(doc["first"], Mapping)):
        raise ValueError(f"not a {STATE_FORMAT} document")
    first: dict[str, int] = {}
    for e, ids in doc["first"].items():
        if (not (isinstance(e, str) and e.lstrip("-").isascii() and e.lstrip("-").isdigit())
                or type(ids) is not list or any(type(q) is not str for q in ids)):
            raise ValueError("the published set's epochs must map to lists of question ids")
        if ids != sorted(set(ids)) or any(q in first for q in ids):
            raise ValueError("each question appears once, under its first epoch, in id order")
        first.update(dict.fromkeys(ids, int(e)))
    return Published(first, doc["through"])


# The most bytes a snapshot of the published set may be (`snapshot_bytes`):
# the canonical document lists each published question once, so it is the
# pool's ids and a few bytes each. The pinned pool's 48,320 ids average 28
# characters and are at most 85, so the whole pool published is about 1.5 MB
# and at most 4.3 MB; this is twice that. Bytes over it are refused unread.
MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024


def snapshot_bytes(published: Published) -> bytes:
    """The published set as a node publishes it and another reads it: the
    canonical document whose sha256 is `Published.digest` (sorted keys, no
    spaces, UTF-8). The root uploads the bytes of the view a reveal names
    (memo_v4.published_digest) under that digest."""
    return json.dumps(published.to_doc(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def published_from_snapshot(raw: bytes, digest: str, *, pool_ids=None,
                            through: int | None = None) -> Published:
    """The published set in snapshot bytes someone else served, strictly read:
    at most MAX_SNAPSHOT_BYTES, hashing to `digest` (the reveal's
    `published_digest`), the canonical bytes of a `published_from_doc`
    document (so a question appears once, under one epoch, in id order),
    every question in `pool_ids` when given (a set folded from this pool's
    reveals holds no other), and folded through `through` when given (the
    view a reveal names: its `memo_v4.through`). ValueError otherwise."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > MAX_SNAPSHOT_BYTES:
        raise ValueError(f"a snapshot is bytes, at most {MAX_SNAPSHOT_BYTES}")
    raw = bytes(raw)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("the snapshot's bytes do not hash to the digest the reveal names")
    try:
        doc = json.loads(raw)
    except (ValueError, RecursionError) as e:
        raise ValueError(f"the snapshot is not JSON ({type(e).__name__})") from None
    published = published_from_doc(doc)
    if snapshot_bytes(published) != raw:
        raise ValueError("the snapshot is not in canonical form")
    if through is not None and published.through != int(through):
        raise ValueError(f"the snapshot is folded through {published.through}, not {int(through)}")
    if pool_ids is not None:
        outside = [q for q in published.first if q not in pool_ids]
        if outside:
            raise ValueError(f"the snapshot names {len(outside)} question(s) not in the pool, "
                             f"e.g. {sorted(outside)[0][:64]!r}")
    return published


def seed_through(doc: Mapping | None = None) -> int:
    """The epoch the pinned seed is observed through (its `through`)."""
    doc = seed_doc() if doc is None else doc
    return epoch_index(doc["through"])


def published_ids(reveal: Mapping) -> set[str]:
    """The questions a reveal publishes an outcome on: its scored questions,
    every exploration answer it lists (failed or not, excluded from the pool
    or not), its pooled labels and, from protocol v5, its exam and coverage
    votes. Over-inclusion only shrinks the reserve; a
    question left out would be counted unseen when it is not."""
    out: set[str] = set()
    for key in ("questions", "exploration", "exploration_pool"):
        rows = reveal.get(key) or []
        if not isinstance(rows, list):
            raise ValueError(f"reveal `{key}` must be a list")
        for row in rows:
            q = row.get("question_id") if isinstance(row, Mapping) else None
            if not isinstance(q, str) or not q:
                raise ValueError(f"reveal `{key}` holds a row with no question id")
            out.add(q)
    # Protocol v5 (v5_epoch.RECORD_KEY): the record's exam and coverage votes
    # are outcomes too, rows [question id, entry, hotkey, coldkey, y, cost,
    # failed]. A v4 reveal has no such field.
    v5 = reveal.get("votes_v5")
    if v5 is not None:
        if not isinstance(v5, Mapping):
            raise ValueError("reveal `votes_v5` must be an object")
        for key in ("votes", "coverage"):
            rows = v5.get(key) or []
            if not isinstance(rows, list):
                raise ValueError(f"reveal `votes_v5.{key}` must be a list")
            for row in rows:
                q = row[0] if isinstance(row, list) and row else None
                if not isinstance(q, str) or not q:
                    raise ValueError(f"reveal `votes_v5.{key}` holds a row with no question id")
                out.add(q)
    return out


def fold(published: Published, epoch: int, ids: Iterable[str]) -> Published:
    """`published` with epoch `epoch`'s questions folded in: a question not
    yet published gets `epoch` as its first epoch. Epochs are folded in
    order; an epoch no reveal exists for is skipped by folding the next.
    Folding an epoch already folded is a retry and changes nothing, provided
    it publishes nothing new."""
    e = int(epoch)
    ids = set(ids)
    if e <= published.through:
        if any(published.first.get(q) is None or published.first[q] > e for q in ids):
            raise ValueError(f"epoch {e} is already folded and this reveal publishes questions "
                             "the state does not hold as published by it")
        return published
    first = dict(published.first)
    for q in sorted(ids):
        first.setdefault(q, e)
    return Published(first, e)


def fold_reveal(published: Published, reveal: Mapping) -> Published:
    """Fold one published reveal (any protocol): `fold` of its epoch and
    `published_ids`. The live validator folds the reveal it publishes, a
    rebuild the reveals it reads; both call this."""
    return fold(published, epoch_index(reveal["epoch_id"]), published_ids(reveal))


def rebuild(start: Published, reveals: Iterable[Mapping]) -> Published:
    """`start` with every reveal after its last epoch folded, oldest first.
    Two reveals for one epoch are refused."""
    rows = sorted(reveals, key=lambda r: epoch_index(r["epoch_id"]))
    seen = [epoch_index(r["epoch_id"]) for r in rows]
    if len(set(seen)) != len(seen):
        raise ValueError("two reveals for one epoch")
    out = start
    for r in rows:
        if epoch_index(r["epoch_id"]) > out.through:
            out = fold_reveal(out, r)
    return out


@functools.lru_cache(maxsize=1)
def seed_doc() -> dict:
    """The pinned seed (routing_data/published_v4.json), read once. Its bytes
    are pinned by scripts/check_safety_invariants.py."""
    with open(SEED_PATH, "rb") as f:
        doc = json.loads(f.read())
    check_seed(doc)
    return doc


def seed_label() -> str:
    """The pinned seed as the READY lines name it: the first eight hex of its
    sha256 and the reveal it is observed through. Raises OSError or
    ValueError when it cannot be read or does not read as a seed."""
    with open(SEED_PATH, "rb") as f:
        raw = f.read()
    doc = json.loads(raw)
    check_seed(doc)
    return f"{hashlib.sha256(raw).hexdigest()[:8]} through {doc['through']}"


def check_seed(doc) -> None:
    """The seed's shape: its last named reveal, the pool it was observed on,
    and per question set the sorted ids of the set's questions no named
    reveal published."""
    if (not isinstance(doc, Mapping) or doc.get("format") != SEED_FORMAT
            or set(doc) != {"format", "through", "pool", "observations_sha256", "unpublished"}):
        raise ValueError(f"not a {SEED_FORMAT} file")
    epoch_index(doc["through"])
    pool = doc["pool"]
    if (not isinstance(pool, Mapping) or set(pool) != {"pool_hash", "n_questions", "per_set"}
            or not isinstance(pool["per_set"], Mapping)):
        raise ValueError("the seed names the pool it was observed on: its hash, size and sets")
    unpublished = doc["unpublished"]
    if not isinstance(unpublished, Mapping) or set(unpublished) - set(pool["per_set"]):
        raise ValueError("the seed's unpublished questions are listed under the pool's sets")
    for s, ids in unpublished.items():
        if (not isinstance(ids, list) or ids != sorted(set(ids))
                or not all(isinstance(q, str) and q for q in ids)
                or len(ids) > pool["per_set"][s]):
            raise ValueError(f"set {s}: the unpublished ids must be sorted, distinct and "
                             "no more than the set holds")


def seed_published(pool: Iterable[Mapping], doc: Mapping | None = None) -> Published:
    """The published set at the seed's last named reveal, for `pool`: every
    question of a set the seed names, except those it lists unpublished, is
    published (recorded at the seed's epoch); a question of a set the seed
    does not name entered the pool after it and is unpublished. A listed id
    missing from its set in `pool` means another pool: refused."""
    doc = seed_doc() if doc is None else doc
    check_seed(doc)
    through = epoch_index(doc["through"])
    named = doc["pool"]["per_set"]
    unpublished = {s: set(ids) for s, ids in doc["unpublished"].items()}
    first: dict[str, int] = {}
    held: dict[str, set[str]] = {}
    for q in pool:
        qid, s = q["question_id"], q.get("benchmark", "")
        held.setdefault(s, set()).add(qid)
        if s in named and qid not in unpublished.get(s, ()):
            first[qid] = through
    for s, ids in unpublished.items():
        missing = ids - held.get(s, set())
        if missing and s in held:
            raise ValueError(f"set {s}: {len(missing)} of the seed's unpublished questions are "
                             "not in this pool; the seed was observed on another pool")
    return Published(first, through)


# --- the reserve and the fresh draw ---------------------------------------------------------


def fresh_count(reserve_size: int) -> int:
    """F = ceil(R / FRESH_RESERVE_EPOCHS): the epoch's fresh questions for the
    whole field."""
    r = int(reserve_size)
    if r < 0:
        raise ValueError("a reserve size is not negative")
    return -(-r // FRESH_RESERVE_EPOCHS)


def fresh_nonce(nonce: bytes) -> bytes:
    """The key of an epoch's fresh draw: the epoch nonce, domain-separated, the
    same for every hotkey."""
    return hashlib.sha256(bytes(nonce) + b":fresh-v4").digest()


class ReserveIndex(PoolIndex):
    """An epoch's pool index for protocol v4 with the published set: as a
    PoolIndex it holds only the questions published through `through` (in
    categories with at least PUBLISHED_MIN_PER_CATEGORY of them), which every
    draw but the fresh one reads (the private draw, proposal measurement);
    `published_pool` is the same questions for the scored slice; `reserve`
    indexes the questions not published through `through`, which only the
    fresh draw reads; `fresh_count` is F; `digest` is the published set's
    digest as of `through`."""

    def __init__(self, published_pool: list[dict], reserve_pool: list[dict],
                 bench_category: Mapping[str, str], bench_weight: Mapping[str, int] | None, *,
                 epoch: int, through: int, digest: str):
        super().__init__(published_pool, dict(bench_category),
                         dict(bench_weight) if bench_weight is not None else None)
        self.published_pool = published_pool
        self.reserve = PoolIndex(reserve_pool, dict(bench_category),
                                 dict(bench_weight) if bench_weight is not None else None)
        self.reserve_size = len(reserve_pool)
        self.fresh_count = fresh_count(len(reserve_pool))
        self.epoch = int(epoch)
        self.through = int(through)
        self.digest = digest


def reserve_index(published: Published, pool: list[dict], epoch: int,
                  bench_category: Mapping[str, str] | None = None,
                  bench_weight: Mapping[str, int] | None = None) -> ReserveIndex:
    """Epoch `epoch`'s ReserveIndex of `pool`, reading the published set as
    of `epoch` - READ_LAG. The pinned v4 mix's sets and weights unless given.
    A question of a set with no category is refused, as the slicer refuses
    it."""
    if bench_category is None:
        from fugal_subnet import epoch_plan

        mix = epoch_plan.mix_v4()
        bench_category, bench_weight = mix.bench_category, mix.bench_weight
    through = int(epoch) - READ_LAG
    view = published.view(through)
    done: list[dict] = []
    left: list[dict] = []
    per_cat: dict[str, int] = {}
    for q in pool:
        b = q.get("benchmark", "")
        if b not in bench_category:
            raise ValueError(f"benchmark {b!r} has no category")
        if view.first.get(q["question_id"]) is not None:
            done.append(q)
            per_cat[bench_category[b]] = per_cat.get(bench_category[b], 0) + 1
        else:
            left.append(q)
    done = [q for q in done if per_cat[bench_category[q.get("benchmark", "")]]
            >= PUBLISHED_MIN_PER_CATEGORY]
    return ReserveIndex(done, left, bench_category, bench_weight, epoch=int(epoch),
                        through=through, digest=view.digest)


def fresh_questions(nonce: bytes, index: ReserveIndex, targets: Mapping[str, float]) -> list[dict]:
    """The epoch's fresh questions: `index.fresh_count` questions of the
    reserve, drawn by the epoch nonce (`fresh_nonce`) on the category mix
    `targets` over the categories that still have a reserve, each slot's set
    by the mix's weights inside its category (slicer.select_private_explore,
    keyed by the epoch alone). The same list for every hotkey and every node:
    no miner chooses which of its questions are fresh."""
    if not isinstance(index, ReserveIndex):
        raise ValueError("the fresh draw reads an epoch's ReserveIndex")
    if index.fresh_count <= 0 or not index.reserve.by_bench:
        return []
    return select_private_explore(fresh_nonce(nonce), index.reserve, set(), index.fresh_count,
                                  dict(targets))


# --- the check ------------------------------------------------------------------------------


class _Questions(Mapping):
    """A Part's `by_question` as `split_labels` builds it: {question: (labels,
    summed skill)} in question order, held as arrays (an index into the
    split's sorted question ids, the counts, the sums) and read-only. It is
    equal to the mapping of the same items, and reads as one; the dict is
    built only when it is read as a mapping. `sums` (count, total, sum of
    squares of the per-question sums, exact Python ints) is what the bound
    reads, without building it."""

    __slots__ = ("_ids", "_index", "_counts", "_sums", "_map", "_cache")

    def __init__(self, ids, index, counts, sums):
        self._ids, self._index, self._counts, self._sums = ids, index, counts, sums
        self._map = None
        self._cache = None

    def _items(self) -> dict:
        if self._map is None:
            ids = self._ids
            self._map = {ids[i]: (c, s) for i, c, s in zip(self._index.tolist(),
                                                           self._counts.tolist(),
                                                           self._sums.tolist())}
        return self._map

    def __getitem__(self, q):
        return self._items()[q]

    def __iter__(self):
        return iter(self._items())

    def __len__(self) -> int:
        return len(self._counts)

    def __eq__(self, other):
        if isinstance(other, _Questions):
            other = other._items()
        if not isinstance(other, Mapping):
            return NotImplemented
        return self._items() == dict(other.items())

    __hash__ = None

    def __repr__(self) -> str:
        return f"mappingproxy({self._items()!r})"

    def sums(self) -> tuple[int, int, int]:
        if self._cache is None:
            from fugal_subnet.seat_bound import int_sum, int_sum_squares

            self._cache = (len(self._counts), int_sum(self._sums), int_sum_squares(self._sums))
        return self._cache


def _question_sums(part: Part) -> tuple[int, int, int]:
    """(questions, sum of the per-question summed skill, sum of its squares)."""
    if isinstance(part.by_question, _Questions):
        return part.by_question.sums()
    xs = [x for _, x in part.by_question.values()]
    return len(xs), sum(xs), sum(x * x for x in xs)


@dataclass(frozen=True)
class Part:
    """One benchmark's labels on one side of the split: their count, the
    questions they are on, and per question its labels and summed weighted
    skill (loss units)."""

    labels: int
    by_question: Mapping[str, tuple[int, int]]

    @property
    def questions(self) -> int:
        return len(self.by_question)

    @property
    def total(self) -> int:
        if isinstance(self.by_question, _Questions):
            return self.by_question.sums()[1]
        return sum(u for _, u in self.by_question.values())


@dataclass(frozen=True)
class MemoEvidence:
    """A head's memorisation evidence. `gap` is the weighted within-benchmark
    gap (seen minus unseen, mean skill per label in loss units, a Fraction;
    0 with no benchmark compared); `lower` the bound's lower end of it in
    units, rounded down (None without items); `refused` the verdict: enough
    labels in both parts and the bound passing at MEMO_MARGIN."""

    benchmarks: Mapping[str, tuple[Part, Part]]
    seen_labels: int
    unseen_labels: int
    gap: Fraction
    lower: int | None
    bound: Bound | None
    refused: bool
    left_out: tuple[str, ...] = field(default=())

    @property
    def enough(self) -> bool:
        return self.seen_labels >= MEMO_MIN_LABELS and self.unseen_labels >= MEMO_MIN_LABELS


def split_labels(window, ref, side, published: Published, *, now: int,
                 window_epochs: int | None = None) -> dict[str, dict[str, Part]]:
    """{benchmark: {SEEN: Part, UNSEEN: Part}}: `side`'s labels against R
    (labels_v4.eligible, R as the other side), each label's skill (R's
    weighted Brier loss minus the head's) under its question, in the part
    its question's first published epoch puts it in: SEEN when that epoch is
    before the epoch the label's row counts from (`Side.row_since`), UNSEEN
    otherwise (a question not yet in the state is published by the label's
    own epoch, which is never before the row). `published` must be folded
    through `now` - 1, so the first epoch of every label's question but the
    current epoch's is known."""
    if side.head is None:
        raise ValueError("the memorisation check is of a head, not of R")
    return _Split(window, ref, published, now=now, window_epochs=window_epochs).split(side)


def _narrow(a: np.ndarray) -> np.ndarray:
    """A non-negative int64 array as int32 when every value fits (they are
    read back as the same Python ints): half the memory a kept head's
    questions hold."""
    return a.astype(np.int32) if len(a) == 0 or int(a.max()) < 2**31 else a


class _Split:
    """`split_labels` for many heads on one window, R, published set and
    epoch: what does not depend on the head is computed once (per epoch:
    each label's (benchmark, question) slot, its question's first published
    epoch as a rank, its entry, the coldkeys that voted on it, R's loss), and
    each head's split is a few integer array operations per epoch.

    EXACT. A label's skill is computed by the same int64 expressions as
    before; the per-question counts are integer bincounts and the sums
    int64 `np.add.at`, whose result cannot depend on the order of the adds
    because no partial sum can overflow: every one lies within n x m, n the
    head's labels and m their largest |skill|, and that product is checked
    below 2^63 (`seat_bound.INT64_LIMIT`) before int64 is used; otherwise the
    sums are of Python ints. The first-epoch comparison (published before
    the row counts from) is made on ranks in the sorted list of the
    published set's epochs, so no epoch is narrowed to int64."""

    def __init__(self, window, ref, published: Published, *, now: int,
                 window_epochs: int | None = None):
        from fugal_subnet import labels_v4

        self.window, self.ref, self.published, self.now = window, ref, published, int(now)
        self.w = labels_v4.SEAT_WINDOW_EPOCHS if window_epochs is None else int(window_epochs)
        self.epochs = [e for e in sorted(window.pairs) if labels_v4.in_window(e, now, self.w)]
        self._built = False
        self._per: dict[int, dict] = {}

    def _build(self) -> None:
        """The slots: every (benchmark, question) of the window's labels,
        ranked by benchmark then question, as the split's output is sorted."""
        if self._built:
            return
        benches, questions = set(), set()
        for e in self.epochs:
            p = self.window.pairs[e]
            benches.update(str(b) for b in p.benchmarks)
            questions.update(p.question_ids)
        self.benches = sorted(benches)
        self.questions = sorted(questions)
        b_rank = {b: i for i, b in enumerate(self.benches)}
        q_rank = {q: i for i, q in enumerate(self.questions)}
        nq = max(1, len(self.questions))
        keys, self._q = {}, {}
        for e in self.epochs:
            p = self.window.pairs[e]
            n = len(p.question_ids)
            q = np.fromiter((q_rank[x] for x in p.question_ids), dtype=np.int64, count=n)
            b = np.fromiter((b_rank[str(x)] for x in p.benchmarks), dtype=np.int64, count=n)
            keys[e] = b * nq + q
            self._q[e] = q
        # the slots: each (benchmark, question) the window holds, in that order
        self.keys = (np.unique(np.concatenate(list(keys.values()))) if keys
                     else np.zeros(0, np.int64))
        self._slot = {e: np.searchsorted(self.keys, k) for e, k in keys.items()}
        self.slot_bench = self.keys // nq
        self.slot_question = self.keys % nq
        first = self.published.first
        firsts = [first.get(q) for q in self.questions]
        self.first_values = sorted({f for f in firsts if f is not None})
        rank = {f: i for i, f in enumerate(self.first_values)}
        # an unpublished question ranks after every epoch: never seen
        self.first_rank = np.array([len(self.first_values) if f is None else rank[f]
                                    for f in firsts], dtype=np.int64)
        self._built = True

    def _epoch(self, e: int) -> dict:
        d = self._per.get(e)
        if d is not None:
            return d
        self._build()
        pairs = self.window.pairs[e]
        n = len(pairs.question_ids)
        entries: dict[str, int] = {}
        entry = np.fromiter((entries.setdefault(m, len(entries)) for m in pairs.models),
                            dtype=np.int64, count=n)
        voted: dict[str, list[int]] = {}
        for i, v in enumerate(pairs.voters):
            for c in v:
                voted.setdefault(c, []).append(i)
        d = {"pairs": pairs, "entries": list(entries), "entry": entry,
             "slot": self._slot[e] * 2, "first": self.first_rank[self._q[e]],
             "voted": {c: np.array(ix, dtype=np.int64) for c, ix in voted.items()},
             "t": pairs.y.astype(np.int64), "ref_loss": None, "w2": None}
        self._per[e] = d
        return d

    def split(self, side) -> dict[str, dict[str, Part]]:
        from bisect import bisect_left

        from fugal_subnet import labels_v4
        from fugal_subnet.explore_weights import WEIGHT_SCALE
        from fugal_subnet.seat_bound import INT64_LIMIT, _magnitude

        if side.head is None:
            raise ValueError("the memorisation check is of a head, not of R")
        self.published.require(self.now - 1)
        slots, units_kept = [], []
        for e in self.epochs:
            pairs = self.window.pairs[e]
            ox = labels_v4._scored(self.window, self.ref, side, e)
            if ox is None:
                continue
            if not isinstance(pairs, labels_v4.LabelPairs):
                labels_v4.eligible(pairs, e, side, labels_v4.REFERENCE, now=self.now,
                                   window_epochs=self.w)        # raises: not a v4 epoch
            d = self._epoch(e)
            if not len(d["t"]):
                continue
            mine = d["voted"].get(side.owner)
            provisional = np.asarray(pairs.missing) == labels_v4.PROVISIONAL_LABEL
            if (side.rows_since or side.added or side.line_since() != int(side.since)
                    or provisional.any()):
                # labels_v4.eligible's per-entry starts (Side._starts on the
                # odds the side is scored from: a missing or added row from
                # the head's line, an inherited row from its own epoch)
                starts = side._starts(labels_v4._label_index(pairs),
                                      labels_v4._stored(self.window, side, e))
                # labels_v4.eligible: the row counts from its start (R has no row)
                keep = np.array([not (e < s) for s in starts], dtype=bool)[d["entry"]]
                # labels_v4.eligible: a label drawn while its entry was
                # provisional is read by none
                keep &= ~provisional
                # SEEN: first published before the row counts from (an
                # unpublished question's rank is past every epoch)
                before = np.array([bisect_left(self.first_values, s) for s in starts],
                                  dtype=np.int64)[d["entry"]]
            else:
                # every row counts from the head's own commit: one start
                if e < int(side.since):
                    continue
                keep = None
                before = bisect_left(self.first_values, int(side.since))
            if mine is not None:
                if keep is None:
                    keep = np.ones(len(d["t"]), dtype=bool)
                keep[mine] = False
            if keep is not None and not keep.any():
                continue
            if d["ref_loss"] is None:
                oy = labels_v4._scored(self.window, self.ref, labels_v4.REFERENCE, e)
                d["ref_loss"] = (oy - d["t"]) ** 2
            units = ox - d["t"]
            units *= units
            np.subtract(d["ref_loss"], units, out=units)
            if pairs.weight is not None:
                # explore_weights.apply, (2 u w + S) // 2S, in place (the
                # same int64 operations: 2 u w is u (2 w) to the bit)
                if d["w2"] is None:
                    d["w2"] = 2 * np.asarray(pairs.weight, dtype=np.int64)
                units *= d["w2"]
                units += WEIGHT_SCALE
                np.floor_divide(units, 2 * WEIGHT_SCALE, out=units)
            slot = d["first"] >= before
            slot = slot + d["slot"]                            # 2 x slot, + 1 when unseen
            slots.append(slot if keep is None else slot[keep])
            units_kept.append(units if keep is None else units[keep])
        if not slots:
            return {}
        slot = np.concatenate(slots)
        units = np.concatenate(units_kept)
        size = 2 * len(self.keys)
        counts = np.bincount(slot, minlength=size)
        if len(units) * _magnitude(units) < INT64_LIMIT:
            sums = np.zeros(size, dtype=np.int64)
        else:
            sums = np.zeros(size, dtype=object)
            sums[:] = 0
            units = units.astype(object)
        np.add.at(sums, slot, units)
        used = np.flatnonzero(counts)
        part_of = used % 2
        bench_of = self.slot_bench[used // 2]
        q_of = self.slot_question[used // 2]
        out: dict[str, dict[str, Part]] = {}
        cuts = np.flatnonzero(np.diff(bench_of)) + 1
        for block in np.split(np.arange(len(used)), cuts):
            b = self.benches[int(bench_of[block[0]])]
            parts = {}
            for bit, name in ((0, SEEN), (1, UNSEEN)):
                sel = block[part_of[block] == bit]
                c = counts[used[sel]]
                labels = int(c.sum())
                parts[name] = Part(labels, _Questions(self.questions, _narrow(q_of[sel]),
                                                      _narrow(c), sums[used[sel]]))
            out[b] = parts
        return out


def evidence_from_parts(parts: Mapping[str, Mapping[str, Part]]) -> MemoEvidence:
    """The gap, its bound and the verdict from a head's split labels.

    Benchmarks with fewer than MEMO_MIN_QUESTIONS questions in either part
    are left out. With w_b = n_s n_u / (n_s + n_u) (labels) and W their sum,
    the gap is sum_b (w_b / W)(S_s,b / n_s,b - S_u,b / n_u,b): each question's
    summed skill x_k enters with coefficient c = w_b / (W n) (negative on the
    unseen side), so the gap is sum c x_k over the questions. The bound reads
    items y_k = N K c x_k, N the number of questions and K the least common
    denominator of the coefficients, in the stratum of their benchmark and
    part, with margin MEMO_MARGIN x K: its D = N K (gap - margin) and its V is
    (N K)^2 times the within-stratum spread of c x, so the inequality it
    evaluates is the one on the gap, exactly."""
    kept = {b: (p[SEEN], p[UNSEEN]) for b, p in sorted(parts.items())
            if p[SEEN].questions >= MEMO_MIN_QUESTIONS and p[UNSEEN].questions >= MEMO_MIN_QUESTIONS}
    left_out = tuple(sorted(set(parts) - set(kept)))
    n_s = sum(s.labels for s, _ in kept.values())
    n_u = sum(u.labels for _, u in kept.values())
    if not kept:
        return MemoEvidence(MappingProxyType({}), n_s, n_u, Fraction(0), None, None, False, left_out)
    w = {b: Fraction(s.labels * u.labels, s.labels + u.labels) for b, (s, u) in kept.items()}
    big_w = sum(w.values())
    coef: dict[tuple[str, str], Fraction] = {}
    for b, (s, u) in kept.items():
        coef[(b, SEEN)] = w[b] / (big_w * s.labels)
        coef[(b, UNSEEN)] = -w[b] / (big_w * u.labels)
    gap = sum((coef[(b, SEEN)] * s.total + coef[(b, UNSEEN)] * u.total
               for b, (s, u) in kept.items()), Fraction(0))
    k = 1
    for c in coef.values():
        k = math.lcm(k, c.denominator)
    n_items = sum(s.questions + u.questions for s, u in kept.values())
    strata: dict[str, Stratum] = {}
    for b, (s, u) in kept.items():
        for name, part in ((SEEN, s), (UNSEEN, u)):
            a = n_items * int(coef[(b, name)] * k)
            count, total, squares = _question_sums(part)
            strata[f"{b}:{name}"] = Stratum(count, a * total, a * a * squares)
    bound = bound_from_strata(strata, margin_units=MEMO_MARGIN_UNITS * k, n_min=0,
                              n0=MEMO_MIN_LABELS)
    lower = None if bound.lower is None else bound.lower // k
    refused = n_s >= MEMO_MIN_LABELS and n_u >= MEMO_MIN_LABELS and bound.passes
    return MemoEvidence(MappingProxyType(kept), n_s, n_u, gap, lower, bound, refused, left_out)


def memo_evidence(window, ref, side, published: Published, *, now: int,
                  window_epochs: int | None = None) -> MemoEvidence:
    """A head's memorisation evidence on the window: `split_labels` then
    `evidence_from_parts`. What the seat package reads: `.refused` (the head
    cannot take, hold, bar or be paid while it is True) and the counts."""
    return evidence_from_parts(split_labels(window, ref, side, published, now=now,
                                            window_epochs=window_epochs))


def refused(evidence: MemoEvidence) -> bool:
    """The verdict: the bound passes on the gap at MEMO_MARGIN with
    MEMO_MIN_LABELS labels in each part."""
    return bool(evidence.refused)


# --- the reveal ---------------------------------------------------------------------------


def evidence_for_reveal(ev: MemoEvidence) -> dict:
    """A head's evidence as published: integers and a bool only."""
    gap = ev.gap
    return {"seen": ev.seen_labels, "unseen": ev.unseen_labels,
            "benchmarks": len(ev.benchmarks),
            "gap_units": gap.numerator // gap.denominator,
            "lower_units": ev.lower, "refused": bool(ev.refused)}


@dataclass(frozen=True)
class EpochMemo:
    """What a v4 epoch's memorisation fields are computed from: the epoch's
    ReserveIndex (the published set as its draws read it, and F), its fresh
    questions, and the published set folded through the epoch before
    (for the split of every label up to this epoch)."""

    index: ReserveIndex
    fresh: tuple[str, ...]
    published: Published


def epoch_memo(published: Published, index: ReserveIndex, nonce: bytes,
               targets: Mapping[str, float]) -> EpochMemo:
    """The EpochMemo of the epoch `index` was built for."""
    published.require(index.epoch - 1)
    fresh = tuple(sorted(q["question_id"] for q in fresh_questions(nonce, index, targets)))
    return EpochMemo(index, fresh, published)


def _require_epoch(memo: EpochMemo, now: int) -> None:
    if memo.index.epoch != int(now):
        raise ValueError(f"the memorisation input is for epoch {memo.index.epoch}, not {now}")


def verdicts(memo: EpochMemo, window, sides: Mapping[str, object], *, now: int
             ) -> dict[str, MemoEvidence]:
    """{head: its MemoEvidence} on `window` (holding epoch `now`), each head
    against R from its `labels_v4.Side`: the seat's view of it (seat_v4.side_of),
    so a successor's inherited rows count from its predecessor's commit.
    What the seat reads (`refused`) and the reveal publishes
    (`memo_for_reveal`): computed once, so the two are the same."""
    from fugal_subnet import labels_v4

    _require_epoch(memo, now)
    ref = labels_v4.reference(window, now=now)
    split = _Split(window, ref, memo.published, now=now)    # what the heads share, once
    return {h: evidence_from_parts(split.split(sides[h])) for h in sorted(sides)}


def memo_for_reveal(memo: EpochMemo, evidence: Mapping[str, MemoEvidence], *, now: int) -> dict:
    """The reveal's `memo_v4`: the published set's digest and epoch as the
    epoch's draws read it, the reserve's size, F, the fresh questions, and
    the evidence of each head the seat judged (`verdicts`)."""
    _require_epoch(memo, now)
    return {"through": memo.index.through, "published_digest": memo.index.digest,
            "reserve": memo.index.reserve_size, "fresh_count": memo.index.fresh_count,
            "fresh": list(memo.fresh),
            "evidence": {h: evidence_for_reveal(ev) for h, ev in sorted(evidence.items())}}


__all__ = [
    "EpochMemo", "MemoEvidence", "Part", "Published", "ReserveIndex", "SEEN", "UNSEEN",
    "MAX_SNAPSHOT_BYTES", "check_seed", "epoch_memo", "evidence_for_reveal",
    "evidence_from_parts", "fold", "fold_reveal", "fresh_count", "fresh_nonce",
    "fresh_questions", "memo_evidence", "memo_for_reveal", "published_from_doc",
    "published_from_snapshot", "published_ids", "rebuild", "refused", "reserve_index",
    "seed_doc", "seed_published", "seed_through", "snapshot_bytes", "split_labels", "verdicts",
]
