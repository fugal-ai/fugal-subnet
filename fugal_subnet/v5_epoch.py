"""Protocol v5: one validator epoch, as a pure function of published inputs.

What a v5 epoch decides (~/projects/fugal-v2-handoff/v5/DESIGN.md;
docs/INVARIANTS.md, the v5 entries):

1. **Votes.** Every verified proof's exam calls become `exam_v5.Vote`s
   (`votes_from_proof`), and its coverage calls become coverage votes. A
   coldkey whose exam votes stray too far from the other coldkeys' on the
   same cells is flagged (`implausible`): its votes count nowhere this epoch.
2. **The window.** The last WINDOW_EPOCHS_V5 epochs' records (`EpochRecord`:
   the epoch's spec when its exam counts, its votes and coverage, the
   flagged coldkeys, the spec's pick costs). A record is exactly what the
   epoch's reveal publishes (`EpochRecord.to_reveal` / `from_reveal`), so the
   window rebuilds from reveals alone.
3. **B*.** The best FULL menu entry by RouterArena's own score on the
   window's coverage votes (`baseline_entry`).
4. **The seat.** `seat_v5.decide` on the window's paired items
   (`exam_v5.comparison_items` / `baseline_items`, both compared operators
   excluded) and the weights (`seat_v5.weights`).
5. **The next exam.** The spec of epoch E + SPEC_LAG_V5, built from the
   heads verified by now (`next_spec`); the root publishes it in this
   epoch's reveal and every validator recomputes it (`spec_agrees`).

MEMORISATION. A head is fixed by its commitment, so it can only have learned
the labels of questions published before it was committed. Two rules, both on
the published set (memo_v4.Published: per question, the first epoch whose
reveal published an outcome on it, the chain-driven view every node holds):

- The exam is drawn from the pool's UNPUBLISHED reserve only (`next_spec`),
  FRESH_RESERVE_EPOCHS_V5-paced like v4's fresh draw (F = ceil(R / that)),
  leaving out every question of a spec already built: when it is drawn no
  label of it is public.
- A question counts in a comparison only if its labels were first published
  in an epoch at or after the later of the compared heads' commit epochs
  (`eligible_after`): a reveal of epoch f is published during epoch f + 1,
  so a head committed in epoch c can have read it only if f <= c - 1. This
  catches what the draw cannot see (a question published by another path
  between the draw and the exam) and makes the rule exact rather than a
  statistical refusal: v4's per-head memorisation verdict is not needed.

DETERMINISM. Integers and Fractions only; sorted orders; the reveal fields
are canonical JSON-able values.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from fugal_subnet import arena_score, config, exam_v5, seat_v5
from fugal_subnet.exam_v5 import ExamSpec, Vote

WINDOW_EPOCHS = int(config.WINDOW_EPOCHS_V5)
EPOCH_INTERVAL = int(config.EPOCH_INTERVAL)
FRESH_RESERVE_EPOCHS = int(config.FRESH_RESERVE_EPOCHS_V5)
SPEC_LAG = int(config.SPEC_LAG_V5)
STANDING_EPOCHS = int(config.STANDING_EPOCHS_V4)
Y_SCALE = int(config.Y_SCALE_V5)
BASELINE_MIN_PAIRS = int(getattr(config, "BASELINE_MIN_PAIRS_V5", 100))
IMPLAUSIBLE_Z = int(config.EXPLORE_PRIVATE_Z)
IMPLAUSIBLE_MIN_SAMPLES = int(config.EXPLORE_PRIVATE_MIN_SAMPLES_V4)
NANO = 1_000_000_000

# Reveal keys.
SPEC_KEY = "exam_spec_v5"          # {"epoch", "spec", "spec_hash"}: the spec of E + SPEC_LAG
COSTS_KEY = "exam_costs_v5"        # {"epoch", "costs"}: that spec's pick costs, nanodollars
RECORD_KEY = "votes_v5"            # this epoch's EpochRecord
SEAT_KEY = "seat_v5"               # the decision's evidence and the new seat state
BASELINE_KEY = "baseline_v5"       # B* and the single-entry table
HEADS_KEY = "heads_v5"             # what the seat knows of every head


# --- votes ----------------------------------------------------------------------------------


def _cost_nd(result) -> int:
    """A call's cost in nanodollars: what the provider billed when the proof
    carries it, else the proxy-metered dollars (exact from the attested
    float), floored."""
    pc = getattr(result, "provider_cost", None)
    if type(pc) is int and pc >= 0:
        return pc
    usd = Fraction(float(getattr(result, "cost_usd", 0.0) or 0.0))
    return max(0, int(usd * NANO))


def votes_from_proof(epoch: int, proof, coldkey: str) -> tuple[list[Vote], list[Vote]]:
    """(exam votes, coverage votes) of one verified v5 proof."""
    exam: list[Vote] = []
    cover: list[Vote] = []
    hk = str(getattr(proof, "hotkey", ""))
    for r in proof.results:
        if r.exam is None:
            continue
        y = int(r.y) if r.y is not None else 0
        v = Vote(epoch=int(epoch), qid=str(r.question_id), entry=str(r.routed_model),
                 hotkey=hk, coldkey=str(coldkey), y=0 if r.failed else y,
                 cost_nd=_cost_nd(r), failed=bool(r.failed))
        (exam if r.exam else cover).append(v)
    return exam, cover


def implausible(votes: Iterable[Vote], *, z: int = IMPLAUSIBLE_Z,
                min_samples: int = IMPLAUSIBLE_MIN_SAMPLES) -> dict[str, str]:
    """{coldkey: why} for operators whose exam votes disagree with the other
    coldkeys' on the same cells by more than z standard deviations.

    For each of a coldkey's non-failed votes on a cell other coldkeys also
    answered, p is the others' mean label (in [0, 1]) and y its own. It is
    flagged when (sum y - sum p)^2 > z^2 sum p(1 - p) over at least
    `min_samples` such votes (explore_pool.private_implausible's test, the
    reference being the cell's other votes). Exact Fractions."""
    by_cell: dict[tuple, list[Vote]] = {}
    for v in votes:
        if not v.failed:
            by_cell.setdefault((v.epoch, v.qid, v.entry), []).append(v)
    stats: dict[str, list[Fraction]] = {}
    for vs in by_cell.values():
        if len({v.coldkey for v in vs}) < 2:
            continue
        for v in vs:
            others = [Fraction(o.y, Y_SCALE) for o in vs if o.coldkey != v.coldkey]
            if not others:
                continue
            p = sum(others, Fraction(0)) / len(others)
            s = stats.setdefault(v.coldkey, [Fraction(0), Fraction(0), Fraction(0), Fraction(0)])
            s[0] += Fraction(v.y, Y_SCALE)
            s[1] += p
            s[2] += p * (1 - p)
            s[3] += 1
    out: dict[str, str] = {}
    for ck in sorted(stats):
        sy, sp, var, n = stats[ck]
        if n < min_samples:
            continue
        if (sy - sp) ** 2 > z * z * max(var, Fraction(1, 4 * int(n))):
            out[ck] = (f"exam votes disagree with the other coldkeys' on {int(n)} cells "
                       f"by more than {z} standard deviations")
    return out


# --- the epoch record ------------------------------------------------------------------------


def _vote_row(v: Vote) -> list:
    return [v.qid, v.entry, v.hotkey, v.coldkey, int(v.y), int(v.cost_nd), 1 if v.failed else 0]


def _vote_of(epoch: int, row) -> Vote:
    q, e, hk, ck, y, c, f = row
    if type(y) is not int or type(c) is not int or f not in (0, 1):
        raise ValueError("a published vote is [qid, entry, hotkey, coldkey, y, cost_nd, failed]")
    return Vote(epoch=int(epoch), qid=str(q), entry=str(e), hotkey=str(hk), coldkey=str(ck),
                y=y, cost_nd=c, failed=bool(f))


@dataclass(frozen=True)
class EpochRecord:
    """One epoch's evidence: its exam spec (None when the epoch had no exam
    or its exam is void here), the pick costs that spec was priced at, its
    exam and coverage votes, and the coldkeys flagged this epoch."""

    epoch: int
    spec: ExamSpec | None
    costs_nd: Mapping[str, int]
    votes: tuple[Vote, ...]
    coverage: tuple[Vote, ...]
    flagged: frozenset[str] = frozenset()
    void: str | None = None

    def counted_votes(self) -> list[Vote]:
        return [v for v in self.votes if v.coldkey not in self.flagged]

    def counted_coverage(self) -> list[Vote]:
        return [v for v in self.coverage if v.coldkey not in self.flagged]

    def to_reveal(self) -> dict:
        return {
            "epoch": int(self.epoch),
            "spec_hash": self.spec.spec_hash() if self.spec is not None else None,
            "void": self.void,
            "flagged": sorted(self.flagged),
            "votes": sorted(_vote_row(v) for v in self.votes),
            "coverage": sorted(_vote_row(v) for v in self.coverage),
        }

    @classmethod
    def from_reveal(cls, doc: Mapping, spec: ExamSpec | None,
                    costs_nd: Mapping[str, int]) -> EpochRecord:
        """The record a reveal published, with the spec (and its costs) the
        reveal of epoch - SPEC_LAG published; the spec must hash to the one
        the record names."""
        epoch = int(doc["epoch"])
        want = doc.get("spec_hash")
        if doc.get("void") is not None or want is None:
            spec = None
        elif spec is None or spec.spec_hash() != want:
            raise ValueError(f"epoch {epoch}'s record names spec {str(want)[:16]}, "
                             "which its spec reveal does not carry")
        return cls(epoch=epoch, spec=spec,
                   costs_nd={str(k): int(v) for k, v in (costs_nd or {}).items()},
                   votes=tuple(_vote_of(epoch, r) for r in doc.get("votes", ())),
                   coverage=tuple(_vote_of(epoch, r) for r in doc.get("coverage", ())),
                   flagged=frozenset(str(c) for c in doc.get("flagged", ())),
                   void=doc.get("void"))


def window_of(records: Iterable[EpochRecord], epoch: int) -> list[EpochRecord]:
    """The records of the WINDOW_EPOCHS epochs up to and including `epoch`,
    oldest first (one per epoch: a repeated epoch is refused)."""
    lo = int(epoch) - WINDOW_EPOCHS + 1
    out = sorted((r for r in records if lo <= r.epoch <= int(epoch)), key=lambda r: r.epoch)
    if len({r.epoch for r in out}) != len(out):
        raise ValueError("two records for one epoch")
    return out


# --- B* --------------------------------------------------------------------------------------


def _arena_exact(acc: Fraction, cost_nd: int) -> Fraction:
    """RouterArena's score (without the 100) of accuracy `acc` at a mean cost
    per query of `cost_nd` nanodollars, exactly (arena_score.cost_score)."""
    c = arena_score.cost_score(int(cost_nd))
    b = arena_score.ARENA_BETA
    den = b * acc + c
    return (1 + b) * acc * c / den if den > 0 else Fraction(0)


def baseline_table(records: Sequence[EpochRecord], full: Iterable[str]) -> dict[str, dict]:
    """{entry: {"n", "accuracy", "cost_nd", "arena"}} over the window's
    coverage votes of FULL entries: one vote per (epoch, question, entry,
    coldkey), failed calls left out."""
    full = set(full)
    seen: set[tuple] = set()
    acc: dict[str, list[int]] = {}
    for r in records:
        for v in r.counted_coverage():
            if v.failed or v.entry not in full:
                continue
            k = (v.epoch, v.qid, v.entry, v.coldkey)
            if k in seen:
                continue
            seen.add(k)
            s = acc.setdefault(v.entry, [0, 0, 0])
            s[0] += 1
            s[1] += int(v.y)
            s[2] += int(v.cost_nd)
    out = {}
    for e in sorted(acc):
        n, ys, cs = acc[e]
        a = Fraction(ys, Y_SCALE * n)
        c = cs // n
        out[e] = {"n": n, "accuracy": a, "cost_nd": c, "arena": _arena_exact(a, c)}
    return out


def baseline_entry(records: Sequence[EpochRecord], full: Iterable[str]) -> tuple[str, dict]:
    """(B*, the table): the full entry with the highest RouterArena score on
    at least BASELINE_MIN_PAIRS coverage votes (ties: the lower id), or ""."""
    table = baseline_table(records, full)
    ok = [(-t["arena"], e) for e, t in table.items() if t["n"] >= BASELINE_MIN_PAIRS]
    return (min(ok)[1] if ok else ""), table


# --- heads -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadFacts:
    """What the seat knows of a head: its owner (the coldkey of its first
    commitment), the hotkey that committed it then, that block, and the last
    epoch it was verified in."""

    head: str
    coldkey: str
    hotkey: str
    first_commit_block: int
    last_seen: int

    def to_reveal(self) -> list:
        return [self.coldkey, self.hotkey, int(self.first_commit_block), int(self.last_seen)]


def heads_from_reveal(doc: Mapping | None) -> dict[str, HeadFacts]:
    out = {}
    for h, (ck, hk, b, seen) in sorted((doc or {}).items()):
        if type(b) is not int or type(seen) is not int:
            raise ValueError("a published head's blocks and epochs are ints")
        out[str(h)] = HeadFacts(str(h), str(ck), str(hk), b, seen)
    return out


@dataclass(frozen=True)
class Served:
    """A head verified this epoch: the uid, hotkey and coldkey that served it
    and the block of that hotkey's commitment."""

    head: str
    uid: int
    hotkey: str
    coldkey: str
    commit_block: int


def observe(heads: Mapping[str, HeadFacts], served: Sequence[Served],
            epoch: int) -> dict[str, HeadFacts]:
    """The head book after epoch `epoch`: a head first committed by the
    earliest commitment seen of it (its owner); `last_seen` moves on when the
    owner's coldkey served it this epoch (a copy served by another coldkey
    does not keep it standing). Heads unseen for the whole window are dropped."""
    out = dict(heads)
    for s in sorted(served, key=lambda s: (s.commit_block, s.head, s.hotkey)):
        h = out.get(s.head)
        if h is None or s.commit_block < h.first_commit_block:
            out[s.head] = HeadFacts(s.head, s.coldkey, s.hotkey, s.commit_block, epoch)
            continue
        if s.coldkey == h.coldkey:
            out[s.head] = HeadFacts(h.head, h.coldkey, h.hotkey, h.first_commit_block, epoch)
    return {h: f for h, f in sorted(out.items()) if f.last_seen > epoch - WINDOW_EPOCHS}


def contenders(heads: Mapping[str, HeadFacts], served: Sequence[Served],
               epoch: int) -> list[seat_v5.Contender]:
    """seat_v5's contenders: every head in the book, standing when its owner
    served it in the last STANDING_EPOCHS epochs, live when its owner served
    it this epoch. A head served this epoch only by another coldkey is an
    exact copy where it is served (copy_rule ownership): it gets no seat of
    its own, and the owner's is unaffected."""
    live = {s.head for s in served if s.head in heads and s.coldkey == heads[s.head].coldkey}
    out = []
    for h, f in sorted(heads.items()):
        out.append(seat_v5.Contender(
            head=h, coldkey=f.coldkey, hotkey=f.hotkey, first_commit_block=f.first_commit_block,
            standing=f.last_seen > epoch - STANDING_EPOCHS, live=h in live))
    return out


def serving_uids(heads: Mapping[str, HeadFacts], served: Sequence[Served]) -> dict[str, int]:
    """{head: the uid of its owner's lowest hotkey serving it this epoch}:
    where a seat's weight goes. A copy's uid is never a head's."""
    out: dict[str, int] = {}
    for s in sorted(served, key=lambda s: s.uid):
        f = heads.get(s.head)
        if f is not None and s.coldkey == f.coldkey and s.head not in out:
            out[s.head] = s.uid
    return out


# --- memorisation ----------------------------------------------------------------------------


def commit_epoch(block: int) -> int:
    """The epoch a commitment at `block` was made in."""
    return int(block) // EPOCH_INTERVAL


def eligible_after(first_published: Callable[[str], int | None],
                   epoch: int) -> Callable[[str], bool]:
    """The question filter of a comparison whose later head was committed in
    epoch `epoch`: a question counts when its labels were never published
    (`first_published` None) or first published in an epoch at or after
    `epoch` (a reveal of epoch f is public from epoch f + 1 on)."""
    e = int(epoch)

    def ok(question_id: str) -> bool:
        f = first_published(question_id)
        return f is None or int(f) >= e
    return ok


def published_reader(published) -> Callable[[str], int | None]:
    """`first_published` over a memo_v4.Published (or None: nothing published)."""
    if published is None:
        return lambda _q: None
    return published.first_epoch


# --- items -----------------------------------------------------------------------------------


def _mean_cost(records: Sequence[EpochRecord], head: str) -> int:
    """A head's mean pick cost per question over the window's specs it is in,
    each spec priced at its own costs (label-free). 0 with no spec."""
    total, n = 0, 0
    for r in records:
        if r.spec is None:
            continue
        for e in r.spec.picks.get(head, ()):
            total += int(r.costs_nd.get(e, 0)) if e else 0
            n += 1
    return total // n if n else 0


def _baseline_cost(records: Sequence[EpochRecord]) -> dict[str, int]:
    """{B* entry: its pick cost} over the window's specs (the newest spec's
    price of each baseline entry)."""
    out: dict[str, int] = {}
    for r in records:
        if r.spec is not None and r.spec.baseline:
            out[r.spec.baseline] = int(r.costs_nd.get(r.spec.baseline, 0))
    return out


def item_sources(records: Sequence[EpochRecord], heads: Mapping[str, HeadFacts],
                 stratum_of: Callable[[str], str],
                 first_published: Callable[[str], int | None]):
    """(items(x, y), baseline_items(x)) for seat_v5.decide over the window,
    each on the questions eligible for that comparison (`eligible_after`)."""
    specs = [r.spec for r in records if r.spec is not None]
    votes = [v for r in records for v in r.counted_votes()]
    costs: dict[str, int] = {}

    def mean(h: str) -> int:
        if h not in costs:
            costs[h] = _mean_cost(records, h)
        return costs[h]

    base_cost = _baseline_cost(records)

    def committed(h: str) -> int:
        return commit_epoch(heads[h].first_commit_block)

    def items(x: str, y: str):
        ex = frozenset({heads[x].coldkey, heads[y].coldkey})
        ok = eligible_after(first_published, max(committed(x), committed(y)))
        return exam_v5.comparison_items(specs, votes, x, y, ex, {x: mean(x), y: mean(y)},
                                        stratum_of, ok)

    def base(x: str):
        ex = frozenset({heads[x].coldkey})
        ok = eligible_after(first_published, committed(x))
        return exam_v5.baseline_items(specs, votes, x, ex, {x: mean(x), **base_cost},
                                      stratum_of, ok)

    return items, base


# --- the epoch -------------------------------------------------------------------------------


@dataclass(frozen=True)
class V5Result:
    decision: seat_v5.SeatDecisionV5
    uids: list[int]
    weights: list[float]
    heads: dict[str, HeadFacts]
    baseline: str
    reveal: dict = field(default_factory=dict)


def score_epoch(*, epoch: int, state: seat_v5.SeatStateV5, heads: Mapping[str, HeadFacts],
                record: EpochRecord, window: Sequence[EpochRecord], served: Sequence[Served],
                full_entries: Iterable[str], stratum_of: Callable[[str], str], burn_uid: int,
                first_published: Callable[[str], int | None]) -> V5Result:
    """Epoch `epoch`'s seat and weights. `window` is the earlier epochs'
    records (this epoch's `record` is added); `heads` the book before this
    epoch; `first_published` the published set as of this epoch
    (`published_reader`). Epochs A and A + 1 have no spec: the seat is
    decided on the window as it is (empty at activation: every share burns)."""
    book = observe(heads, served, epoch)
    recs = window_of([*[r for r in window if r.epoch != record.epoch], record], epoch)
    base, table = baseline_entry(recs, full_entries)
    items, base_items = item_sources(recs, book, stratum_of, first_published)
    decision = seat_v5.decide(state, epoch, contenders(book, served, epoch), items, base_items)
    uids, weights = seat_v5.weights(decision, serving_uids(book, served), burn_uid=burn_uid)
    reveal = {
        SEAT_KEY: {"state": decision.state.to_reveal(), "evidence": _jsonable(decision.evidence)},
        BASELINE_KEY: {"entry": base, "table": {
            e: {"n": t["n"], "accuracy": str(t["accuracy"]), "cost_nd": t["cost_nd"],
                "arena": str(t["arena"])} for e, t in table.items()}},
        HEADS_KEY: {h: f.to_reveal() for h, f in book.items()},
        RECORD_KEY: record.to_reveal(),
    }
    return V5Result(decision, uids, weights, book, base, reveal)


def _jsonable(x):
    if isinstance(x, Fraction):
        return str(x)
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# --- the next exam ---------------------------------------------------------------------------


def spec_nonce(nonce: bytes, spec_epoch: int) -> bytes:
    """The nonce the spec of `spec_epoch` is drawn with: this epoch's nonce,
    domain separated (known only once this epoch's boundary exists, after
    every head it judges was committed)."""
    return hashlib.sha256(bytes(nonce) + f":spec-v5:{int(spec_epoch)}".encode()).digest()


def next_spec(*, epoch: int, nonce: bytes, decision: seat_v5.SeatDecisionV5,
              heads: Mapping[str, HeadFacts], served: Sequence[Served],
              loaded: Mapping[str, object], state_menu, hidden_of: Callable[[Sequence[str]], object],
              by_dataset: Mapping[str, Sequence[str]], weights_ppm: Mapping[str, int],
              baseline: str, first_published: Callable[[str], int | None],
              used: Iterable[str] = ()) -> tuple[ExamSpec, dict]:
    """(the spec of epoch + SPEC_LAG, its pick costs in nanodollars).

    The candidate questions are the reserve's (`reserve`: unpublished by
    `first_published`, not in `used`, the questions of every spec already
    built), FRESH_RESERVE_EPOCHS-paced (`exam_size`). The contenders are
    seat_v5.select's for the seat as just decided; each picks on the
    candidate questions with its own lam on this epoch's menu
    (`exam_v5.exam_picks`); a contender whose bytes are not at hand
    (`loaded`) is left out. The ring is this epoch's verified (hotkey,
    coldkey) pairs."""
    from fugal_subnet import menu_state as _ms

    spec_epoch = int(epoch) + SPEC_LAG
    key = spec_nonce(nonce, spec_epoch)
    fresh = reserve(by_dataset, first_published, used)
    pool = exam_v5.draw_exam(key, fresh, weights_ppm,
                             exam_size(sum(len(v) for v in fresh.values())))
    chosen = seat_v5.select(contenders(heads, served, epoch), decision.holder)
    picks: dict[str, tuple[str, ...]] = {}
    if pool:
        hidden = hidden_of(pool)
        for c in chosen:
            head = loaded.get(c.head)
            if head is None:
                continue
            try:
                picks[c.head] = exam_v5.exam_picks(head, state_menu, hidden)
            except ValueError:
                continue
    entries = [e.id for e in state_menu.entries if _ms.is_full(e)]
    ring = sorted({(s.hotkey, s.coldkey) for s in served})
    spec = exam_v5.build_spec(spec_epoch, key, picks, entries, pool, ring, baseline)
    costs = {e: int(c * NANO) for e, c in sorted(_ms.pick_costs(state_menu).items())}
    return spec, costs


def reserve(by_dataset: Mapping[str, Sequence[str]], first_published: Callable[[str], int | None],
            used: Iterable[str] = ()) -> dict[str, list[str]]:
    """{dataset: its questions no reveal has published an outcome on and no
    built spec holds}, ids sorted: what an exam may draw."""
    used = set(used)
    out = {}
    for d in sorted(by_dataset):
        qs = sorted({str(q) for q in by_dataset[d]
                     if str(q) not in used and first_published(str(q)) is None})
        if qs:
            out[d] = qs
    return out


def exam_size(reserve_size: int) -> int:
    """An exam's candidate questions from a reserve of R: ceil(R /
    FRESH_RESERVE_EPOCHS), at most EXAM_QUESTIONS_V5. The reserve then lasts:
    it shrinks by about 1/FRESH_RESERVE_EPOCHS an epoch (a release that adds
    questions raises it at once)."""
    r = int(reserve_size)
    if r < 0:
        raise ValueError("a reserve size is not negative")
    return min(exam_v5.EXAM_QUESTIONS, -(-r // FRESH_RESERVE_EPOCHS))


def spec_questions(specs: Iterable[ExamSpec | None]) -> set[str]:
    """Every question of the given specs: the `used` an exam draw leaves out."""
    return {q for s in specs if s is not None for q in s.questions}


def spec_reveal(spec: ExamSpec, costs_nd: Mapping[str, int]) -> dict:
    """The reveal fields that publish the next exam (td_v5 reads SPEC_KEY)."""
    return {SPEC_KEY: {"epoch": int(spec.epoch), "spec": spec.to_json(),
                       "spec_hash": spec.spec_hash()},
            COSTS_KEY: {"epoch": int(spec.epoch),
                        "costs": {k: int(v) for k, v in sorted(costs_nd.items())}}}


def spec_from_reveal(doc: Mapping) -> tuple[ExamSpec | None, dict[str, int]]:
    """(the spec a reveal publishes, its costs), or (None, {})."""
    f = doc.get(SPEC_KEY)
    if not f:
        return None, {}
    spec = ExamSpec.from_json(f["spec"])
    if spec.spec_hash() != f["spec_hash"] or int(spec.epoch) != int(f["epoch"]):
        raise ValueError("the reveal's exam spec does not match the hash and epoch it names")
    c = doc.get(COSTS_KEY) or {}
    if c and int(c.get("epoch", -1)) != int(spec.epoch):
        raise ValueError("the reveal's exam costs are for another epoch")
    return spec, {str(k): int(v) for k, v in (c.get("costs") or {}).items()}


def spec_agrees(own: ExamSpec | None, published: ExamSpec | None) -> str | None:
    """None when this validator's spec for an epoch is the root's published
    one; otherwise why the epoch's exam is void here."""
    if published is None:
        return "no published exam spec for this epoch"
    if own is None:
        return "this validator built no exam spec for this epoch"
    if own.spec_hash() != published.spec_hash():
        return (f"this validator's exam spec {own.spec_hash()[:16]} is not the published "
                f"{published.spec_hash()[:16]}")
    return None


# --- rebuild ---------------------------------------------------------------------------------


def rebuild(reveals: Mapping[int, Mapping], epoch: int):
    """(seat state, head book, window records) after epoch `epoch`, from the
    reveals of the window and the SPEC_LAG epochs before it alone: a pure
    function of what was published. A missing reveal leaves its epoch out."""
    records = []
    for e in range(int(epoch) - WINDOW_EPOCHS + 1, int(epoch) + 1):
        doc = reveals.get(e)
        if not doc or RECORD_KEY not in doc:
            continue
        spec, costs = spec_from_reveal(reveals.get(e - SPEC_LAG) or {})
        records.append(EpochRecord.from_reveal(doc[RECORD_KEY], spec, costs))
    last = reveals.get(int(epoch)) or {}
    seat = last.get(SEAT_KEY) or {}
    return (seat_v5.SeatStateV5.from_reveal(seat.get("state")),
            heads_from_reveal(last.get(HEADS_KEY)), records)
