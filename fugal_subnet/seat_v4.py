"""Protocol v4: the seat (docs/decisions/SEAT_MINIMAL_V4.md 3.3 to 3.8, 4.3,
4.4; package S4). Nothing on the v1-v3 path imports this module.

Each epoch decides which head holds the seat and who is paid, from the
window (labels, every head's odds, the real-picks records), the epoch's menu,
the copy rule's ownership state and the previous seat state. Three rules on
one bound (seat_bound), and two additions:

  A  the odds test: X beats Y when the bound passes on X's Brier gain over Y
     with margin ODDS_MARGIN (0.1 Brier points) on at least ODDS_MIN_PAIRS
     (1,500) labels (labels_v4.gain_strata);
  B  the real-picks check: a head about to take passes when the bound of its
     challenge items over the holder passes with margin -1.5 points on 500
     items of its answered run of challenge epochs, n0 100
     (picks_v4.picks_check; FAST_CROWN_V4.md 2); when the seat is open, only
     that its run holds 500 items, unless 500 are paired with the last
     holder;
  C  precedence: a head may take the seat only if it beats (A) every
     contender earlier than it. Heads are ordered by the first commit of
     themselves or of their earliest predecessor (succession), commits at or
     before the v4 activation block tying at it, broken by the salted digest
     of copy_rule.seniority. R, the reference, is the earliest head of all.
  standing  a head, the holder included, stands while it (or a predecessor
     it succeeded) was verified in this epoch or one of the STANDING_EPOCHS - 1
     (2) before it: the holder stands on its exploration proof alone and
     answers no scored question (FAST_CROWN_V4.md 6);
  ownership  copy_rule.classify, unchanged: an exact copy is not a head.

THE SEAT, EACH EPOCH (`decide`), in this order:

  1. The bar is the holder if it stands, R does not beat it (A) and the
     memorisation verdict does not refuse it; otherwise the bar is R and the
     seat is open.
  2. A contender is a head other than the holder that stands, has a counted
     challenge in one of the last STANDING_EPOCHS epochs, is not refused,
     beats the bar (A), and is not shown to route worse than the holder (the
     bound of the holder's net value over it, margin +1.5 points, on its
     answered run, does not pass). R is always a contender.
  3. A qualifier is a contender with a counted challenge this epoch that
     passes B and beats (A) every contender earlier than it, R first
     (when the bar is the holder; when it is R, beating the bar is beating
     R). Rule C is tested earliest first and stops at the first failure.
  4. If there are qualifiers, the one committed last takes the seat; the
     holder it replaces goes to the history. Every other qualifier is a
     contender earlier than it, so it beat each of them. Otherwise the holder
     keeps the seat, open or not: the seat is never emptied without a
     replacement.

PAY (3.8): 85 / 10 / 5 over the holder and the two holders before it
(payout.seat_weights). A seat pays when its head is live (verified this
epoch: the holder on its exploration proof alone), R does not beat it and
the memorisation verdict does not refuse it. An unpaid seat's share goes to the
holder; an unpaid holder's share burns.

SUCCESSION (4.4, `observe`). A head H' first seen this epoch is the
successor of H when the same hotkey first committed H before it, both belong
to the same coldkey, H is not served this epoch, and every row H has for an
entry on this epoch's menu is in H' byte for byte (per-row digests,
`row_digests`, published in the seat state for every head it keeps). It
inherits H's seat or history seat, its precedence, its standing and its
evidence: each inherited row counts from H's row (`rows_since`), so a
validator computes H''s odds from H''s first row (`odds_since`), H's
challenges count as H''s, and a challenge paired with H counts as paired
with H' (`lineage_records`).

THE HOLDER A CHALLENGE PAIRS WITH (FAST_CROWN_V4.md 6, `challenge_holder`)
is the holder after epoch E - CHALLENGE_HOLDER_LAG_V4, which the root's
newest reveal at E's boundary names (it is held one epoch, as the menu is).
The state keeps the holder after each of its last CHALLENGE_HOLDER_LAG_V4
epochs (`holders`, published), so a rebuild and a re-anchor know it too. Its new rows count
from its own commit. A head that changes an existing row is a new head.

THE MEMORISATION VERDICT is an input (`memo_refused`, a set of head hashes).
The live epoch computes it when handed the epoch's memo (v2_epoch.run_v2_epoch:
memo_v4.verdicts for every head `observe` keeps, each through `side_of`, on
the window holding the epoch) and publishes the same evidence; without a
memo it is empty. Where the design applies it: the bar (a refused holder
opens the seat), contenders and qualifiers (a refused head is neither), and
pay (a refused head is not paid).

DETERMINISM. Every comparison is the bound on exact integer sums; heads are
visited in sorted order and ranked by precedence; no arrival order, no
random stream and no ranking of margins is read.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from fugal_subnet import config
from fugal_subnet.seat_bound import Bound, bound_from_strata, lower_bound

UNITS_PER_POINT = config.BOUND_UNITS_PER_POINT_V4
STANDING_EPOCHS = config.STANDING_EPOCHS_V4
ODDS_MARGIN_POINTS = Fraction(config.ODDS_MARGIN_POINTS_V4)
ODDS_MARGIN_UNITS = int(ODDS_MARGIN_POINTS * UNITS_PER_POINT)
ODDS_MIN_PAIRS = config.ODDS_MIN_PAIRS_V4
# Rule B and its reverse ("shown to route worse"): margin 1.5 points, 1,000
# questions, n0 the same (picks_v4).
PICKS_MARGIN_UNITS = int(Fraction(config.PICKS_MARGIN_POINTS_V4) * UNITS_PER_POINT)
PICKS_MIN_QUESTIONS = config.PICKS_MIN_QUESTIONS_V4
# Rules A and B tune the bound to BOUND_N0_V4 items (FAST_CROWN_V4.md 0).
BOUND_N0 = config.BOUND_N0_V4
# Former holders kept, most recent first (the payout reads two).
HISTORY_KEEP = 4
# A successor keeps at most this many predecessors, most recent first: their
# stored odds stand for its own before its commit (labels_v4.predecessor_odds)
# and their real-picks answers are read as its own (`lineage_records`). The
# evidence window: each predecessor was its line's head for at least one
# epoch, so the predecessors it no longer keeps were head only in epochs that
# have left the window, however often the line is re-committed.
MAX_PREDECESSORS = config.SEAT_WINDOW_EPOCHS_V4
# The holders after the last epochs the state keeps: a challenge in E pairs
# with the holder after E - CHALLENGE_HOLDER_LAG_V4.
HOLDER_LAG = config.CHALLENGE_HOLDER_LAG_V4
# The published state's bounds (I2: a reveal is someone else's file). The
# state keeps the heads that stand (at most one new head per hotkey and
# epoch: 256 x STANDING_EPOCHS) and the seat's.
MAX_STATE_HEADS = 4096
MAX_ROWS = 72                    # head_eval.HEAD_MAX_ROWS_V4
MAX_ENTRY_ID = 256
REVEAL_KEY = "seat_v4"
DIGEST_KEY = "seat_digest"
# What the published evidence names the reference.
R = "R"
_HASH = re.compile(r"[0-9a-f]{64}")


# --- state ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadState:
    """A head the seat keeps: one that stands, or holds the seat or a history
    seat.

    `block` is its own first-commit block and `since` the first epoch whose
    boundary is at or after it (copy_rule.first_epoch); `seen` the last epoch
    it, or a predecessor it succeeded, was verified. `origin`, `origin_block`
    and `origin_owner` name the earliest predecessor, whose first commit is
    the head's precedence (itself when it succeeded nobody). `rows_since`
    maps each row inherited by succession to the epoch it counts from;
    `predecessors` are the heads it succeeded, most recent first, and
    `predecessor_since` each one's own `since`, aligned with them; `rows` its
    per-row digests (`row_digests`). `added` names the entries whose row no
    head of its line had when it was first carried (a row this head added at
    its succession, or one an earlier head of its line added and it
    inherited): labels_v4.Side scores such an entry's labels from the line's
    start up to the row's own start as missing (FAST_CROWN_V4.md 8, 18)."""

    head: str
    owner: str
    hotkey: str
    block: int
    since: int
    seen: int
    origin: str
    origin_block: int
    origin_owner: str
    rows_since: Mapping[str, int] = field(default_factory=dict)
    predecessors: tuple[str, ...] = ()
    rows: Mapping[str, str] = field(default_factory=dict)
    predecessor_since: tuple[int, ...] = ()
    added: frozenset = frozenset()

    def __post_init__(self):
        if len(self.predecessor_since) != len(self.predecessors):
            raise ValueError(f"head {self.head[:12]}: one `since` per predecessor")

    def row_since(self, entry: str) -> int:
        return int(self.rows_since.get(entry, self.since))

    @property
    def line_since(self) -> int | None:
        """Its line origin's own commit epoch (the oldest predecessor it
        keeps), None for a head that succeeded nobody."""
        return self.predecessor_since[-1] if self.predecessor_since else None

    @property
    def odds_since(self) -> int:
        """The first epoch a validator computes this head's odds for: its
        own commit, its oldest inherited row's, or its line origin's commit
        (from which a missing row counts)."""
        line = self.line_since
        return min([self.since, *self.rows_since.values(), *(() if line is None else (line,))])


@dataclass(frozen=True)
class SeatStateV4:
    """The v4 seat: the holder (`champion`, the head hash a miner reads), the
    epoch it took the seat, former holders most recent first, and the heads
    the seat keeps (sorted by hash). `epoch` is the last epoch observed.
    `holders` is the holder after each of the last HOLDER_LAG epochs decided,
    newest first, as (epoch, head hash or None): what `challenge_holder`
    reads."""

    champion: str | None = None
    champion_since: int | None = None
    history: tuple[str, ...] = ()
    heads: tuple[HeadState, ...] = ()
    epoch: int | None = None
    holders: tuple[tuple[int, str | None], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "history", tuple(self.history)[:HISTORY_KEEP])
        object.__setattr__(self, "heads", tuple(sorted(self.heads, key=lambda h: h.head)))
        object.__setattr__(self, "holders", tuple(sorted(
            ((int(e), h) for e, h in self.holders), key=lambda t: -t[0]))[:HOLDER_LAG])

    # v4 has no ghosts; the v1-v3 wiring that reads `ghosts` sees none.
    @property
    def ghosts(self) -> tuple:
        return ()

    def head(self, h: str) -> HeadState | None:
        for hs in self.heads:
            if hs.head == h:
                return hs
        return None

    def by_head(self) -> dict[str, HeadState]:
        return {hs.head: hs for hs in self.heads}


def to_dict(state: SeatStateV4) -> dict:
    """The seat state as published (`v2_state.seat`) and saved. Entry ids
    are written once, in `entries`; a head's rows are one string of
    "index:digest" and its inherited rows one of "index:epoch", each
    space-separated in entry order (one line each in the indented reveal)."""
    entries = sorted({e for hs in state.heads for e in (*hs.rows, *hs.rows_since, *hs.added)})
    index = {e: i for i, e in enumerate(entries)}

    def line(m):
        return " ".join(f"{index[e]}:{v}" for e, v in sorted(m.items(), key=lambda kv: index[kv[0]]))

    return {
        "protocol": 4,
        "champion": state.champion,
        "champion_since": state.champion_since,
        "history": list(state.history),
        "epoch": state.epoch,
        "holders": {str(e): h for e, h in state.holders},
        "entries": entries,
        "heads": {
            hs.head: {
                "owner": hs.owner, "hotkey": hs.hotkey, "block": hs.block, "since": hs.since,
                "seen": hs.seen, "origin": hs.origin, "origin_block": hs.origin_block,
                "origin_owner": hs.origin_owner, "rows_since": line(hs.rows_since),
                "predecessors": [f"{p}:{s}" for p, s in _chain_items(hs)], "rows": line(hs.rows),
                "added": " ".join(str(i) for i in sorted(index[e] for e in hs.added)),
            }
            for hs in state.heads
        },
    }


def is_v4_dict(raw) -> bool:
    return isinstance(raw, Mapping) and raw.get("protocol") == 4


def _int(x, what: str, lo: int = 0) -> int:
    if type(x) is not int or x < lo:
        raise ValueError(f"v4 seat state: {what} must be an integer of at least {lo}")
    return x


def _hash(x, what: str) -> str:
    if not (isinstance(x, str) and _HASH.fullmatch(x)):
        raise ValueError(f"v4 seat state: {what} must be a head hash")
    return x


def _address(x, what: str) -> str:
    if not (isinstance(x, str) and 0 < len(x) <= 64):
        raise ValueError(f"v4 seat state: {what} must be an address")
    return x


_EPOCH_KEY = re.compile(r"0|[1-9][0-9]{0,11}")
_ROW_ITEM = re.compile(r"(0|[1-9][0-9]{0,3}):([0-9a-f]{64})")
_EPOCH_ITEM = re.compile(r"(0|[1-9][0-9]{0,3}):(0|[1-9][0-9]{0,11})")


def _chain_items(hs: HeadState) -> list[tuple[str, int]]:
    return list(zip(hs.predecessors, hs.predecessor_since, strict=True))


def _predecessor(item, h: str) -> tuple[str, int]:
    """A published predecessor, "hash:since" (`to_dict`)."""
    head, _, since = item.partition(":") if isinstance(item, str) else ("", "", "")
    if (not _HASH.fullmatch(head) or not (since.isascii() and since.isdigit())
            or len(since) > 12 or (since[0] == "0" and since != "0")):
        raise ValueError(f"v4 seat state: head {h[:12]} `predecessors` is malformed")
    return head, int(since)


def _line(text, item: re.Pattern, h: str) -> list[tuple[int, str]]:
    """The "index:value" items of one published line, in increasing index
    order, at most MAX_ROWS."""
    if not isinstance(text, str):
        raise ValueError(f"v4 seat state: head {h[:12]} has a row line that is not a string")
    if text == "":
        return []
    parts = text.split(" ")
    out = []
    for p in parts:
        m = item.fullmatch(p)
        if m is None:
            raise ValueError(f"v4 seat state: head {h[:12]} has a malformed row item")
        out.append((int(m.group(1)), m.group(2)))
    idx = [i for i, _v in out]
    if len(out) > MAX_ROWS or idx != sorted(set(idx)):
        raise ValueError(f"v4 seat state: head {h[:12]} names a row twice, out of order or "
                         f"more than {MAX_ROWS}")
    return out


_INDEX_ITEM = re.compile(r"0|[1-9][0-9]{0,3}")


def _index_line(text, h: str) -> list[int]:
    """The entry indexes of a published `added` line: space-separated, in
    increasing order, each once, at most MAX_ROWS."""
    if not isinstance(text, str):
        raise ValueError(f"v4 seat state: head {h[:12]} has an `added` line that is not a string")
    if text == "":
        return []
    parts = text.split(" ")
    if not all(_INDEX_ITEM.fullmatch(p) for p in parts):
        raise ValueError(f"v4 seat state: head {h[:12]} has a malformed `added` item")
    out = [int(p) for p in parts]
    if len(out) > MAX_ROWS or out != sorted(set(out)):
        raise ValueError(f"v4 seat state: head {h[:12]} names an added row twice, out of order "
                         f"or more than {MAX_ROWS}")
    return out


def from_dict(raw: Mapping) -> SeatStateV4:
    """The inverse of `to_dict`, validated and bounded (I2)."""
    if not is_v4_dict(raw):
        raise ValueError("not a v4 seat state")
    keys = {"protocol", "champion", "champion_since", "history", "epoch", "holders", "entries",
            "heads"}
    if set(raw) != keys:
        raise ValueError(f"v4 seat state: fields must be {sorted(keys)}")
    entries, heads_raw = raw["entries"], raw["heads"]
    if (not isinstance(entries, list) or len(entries) > MAX_STATE_HEADS
            or entries != sorted(set(entries))
            or not all(isinstance(e, str) and 0 < len(e) <= MAX_ENTRY_ID for e in entries)):
        raise ValueError("v4 seat state: `entries` must be sorted distinct entry ids")
    if not isinstance(heads_raw, Mapping) or len(heads_raw) > MAX_STATE_HEADS:
        raise ValueError(f"v4 seat state: `heads` must be a map of at most {MAX_STATE_HEADS}")

    def entry(i):
        if type(i) is not int or not 0 <= i < len(entries):
            raise ValueError("v4 seat state: an entry index is out of range")
        return entries[i]

    heads = []
    for h, v in heads_raw.items():
        _hash(h, "a head")
        want = {"owner", "hotkey", "block", "since", "seen", "origin", "origin_block",
                "origin_owner", "rows_since", "predecessors", "rows", "added"}
        if not isinstance(v, Mapping) or set(v) != want:
            raise ValueError(f"v4 seat state: head {h[:12]} must hold {sorted(want)}")
        preds = v["predecessors"]
        if not isinstance(preds, list) or len(preds) > MAX_PREDECESSORS:
            raise ValueError(f"v4 seat state: head {h[:12]} `predecessors` is malformed")
        chain = [_predecessor(p, h) for p in preds]
        rows_since = {entry(i): int(s) for i, s in _line(v["rows_since"], _EPOCH_ITEM, h)}
        digests = {entry(i): d for i, d in _line(v["rows"], _ROW_ITEM, h)}
        added = frozenset(entry(i) for i in _index_line(v["added"], h))
        heads.append(HeadState(
            head=h, owner=_address(v["owner"], "an owner"), hotkey=_address(v["hotkey"], "a hotkey"),
            block=_int(v["block"], "a block", 1), since=_int(v["since"], "an epoch"),
            seen=_int(v["seen"], "an epoch"), origin=_hash(v["origin"], "an origin"),
            origin_block=_int(v["origin_block"], "a block", 1),
            origin_owner=_address(v["origin_owner"], "an owner"), rows_since=rows_since,
            predecessors=tuple(p for p, _s in chain), rows=digests,
            predecessor_since=tuple(s for _p, s in chain), added=added))
    history = raw["history"]
    if not isinstance(history, list) or len(history) > HISTORY_KEEP:
        raise ValueError(f"v4 seat state: `history` must be at most {HISTORY_KEEP} heads")
    champion = raw["champion"]
    kept = {hs.head for hs in heads}
    for h in ([champion] if champion is not None else []) + list(history):
        if _hash(h, "a seat") not in kept:
            raise ValueError("v4 seat state: a seat names a head the state does not keep")
    since = raw["champion_since"]
    if (champion is None) != (since is None):
        raise ValueError("v4 seat state: a holder and its epoch go together")
    epoch = raw["epoch"]
    holders_raw = raw["holders"]
    if not isinstance(holders_raw, Mapping) or len(holders_raw) > HOLDER_LAG:
        raise ValueError(f"v4 seat state: `holders` must be a map of at most {HOLDER_LAG}")
    holders = []
    for e, h in holders_raw.items():
        if not (isinstance(e, str) and _EPOCH_KEY.fullmatch(e)):
            raise ValueError("v4 seat state: a `holders` key must be an epoch")
        holders.append((int(e), None if h is None else _hash(h, "a holder")))
    return SeatStateV4(
        champion=champion, champion_since=None if since is None else _int(since, "an epoch"),
        history=tuple(history), heads=tuple(heads),
        epoch=None if epoch is None else _int(epoch, "an epoch"), holders=tuple(holders))


# --- succession and standing ---------------------------------------------------------------


def row_digests(W, b, models: Iterable[str]) -> dict[str, str]:
    """{entry: sha256(entry id || 0x00 || the row's weights || its bias)},
    the weights and bias as little-endian float64 (the values the odds are
    computed from: head_odds.logit_exact)."""
    W = np.asarray(W)
    b = np.asarray(b)
    out = {}
    for i, m in enumerate(models):
        payload = (str(m).encode("utf-8") + b"\x00"
                   + np.asarray(W[i], dtype="<f8").tobytes()
                   + np.asarray(b[i], dtype="<f8").reshape(1).tobytes())
        out[str(m)] = hashlib.sha256(payload).hexdigest()
    return out


def is_successor(new_rows: Mapping[str, str], pred: HeadState, menu_ids: Iterable[str]) -> bool:
    """Every row `pred` has for an entry on the menu is in `new_rows` with
    the same digest (rows may be added, and rows of entries off the menu
    dropped or changed)."""
    if not new_rows or not pred.rows:
        return False
    on_menu = set(menu_ids)
    return all(new_rows.get(e) == d for e, d in pred.rows.items() if e in on_menu)


def observe(prev: SeatStateV4, *, epoch: int, serving: Mapping[str, int],
            rows: Mapping[str, Mapping[str, str]], copy_state, menu_ids: Iterable[str],
            blocks_per_epoch: int) -> SeatStateV4:
    """The seat state with this epoch's verified heads recorded: each head
    served by its owner (`serving`, copy_rule.classify's owners) is seen now;
    a head first seen now is a successor (module docstring) or a new head;
    a head that no longer stands and holds no seat is dropped. Pure."""
    from fugal_subnet import copy_rule

    epoch = int(epoch)
    menu_ids = tuple(menu_ids)
    on_menu = set(menu_ids)
    heads = prev.by_head()
    champion, history = prev.champion, tuple(prev.history)
    rename: dict[str, str] = {}
    new = []

    def menu_rows(h):
        # The rows a successor must keep are those on the menu; others are
        # not recorded (a row for an entry that joins later counts from the
        # commit of the head that carries it, as any new row does).
        return {e: d for e, d in (rows.get(h) or {}).items() if e in on_menu}

    for h in sorted(serving):
        rec = copy_state.get(h)
        if rec is None:
            continue
        if h in heads:
            hs = heads[h]
            heads[h] = dataclasses.replace(hs, seen=epoch, owner=rec.owner,
                                           rows=menu_rows(h) if h in rows else dict(hs.rows))
        else:
            new.append(rec)
    for rec in sorted(new, key=lambda r: (r.first_block, r.head_hash)):
        h = rec.head_hash
        since = int(copy_rule.first_epoch(copy_state, h, blocks_per_epoch))
        mine = menu_rows(h)
        preds = [p for p in heads.values()
                 if p.hotkey == rec.first_hotkey and p.owner == rec.owner and p.head != h
                 and p.head not in serving and p.block < rec.first_block]
        pred = max(preds, key=lambda p: (p.block, p.head)) if preds else None
        if pred is not None and is_successor(mine, pred, menu_ids):
            heads.pop(pred.head)
            rename[pred.head] = h
            inherited = {e: pred.row_since(e) for e, d in mine.items() if pred.rows.get(e) == d}
            heads[h] = HeadState(
                head=h, owner=rec.owner, hotkey=rec.first_hotkey, block=rec.first_block,
                since=since, seen=epoch, origin=pred.origin, origin_block=pred.origin_block,
                origin_owner=pred.origin_owner,
                # the rows it adds, and those an earlier head of its line added
                added=frozenset(e for e in mine if e not in inherited)
                | (pred.added & frozenset(inherited)),
                rows_since=inherited,
                predecessors=((pred.head,) + pred.predecessors)[:MAX_PREDECESSORS], rows=mine,
                predecessor_since=((pred.since,) + pred.predecessor_since)[:MAX_PREDECESSORS])
        else:
            heads[h] = HeadState(head=h, owner=rec.owner, hotkey=rec.first_hotkey,
                                 block=rec.first_block, since=since, seen=epoch, origin=h,
                                 origin_block=rec.first_block, origin_owner=rec.owner, rows=mine)
    if champion is not None:
        champion = rename.get(champion, champion)
    history = tuple(rename.get(h, h) for h in history)
    seats = ({champion} if champion is not None else set()) | set(history)
    kept = [hs for h, hs in sorted(heads.items())
            if h in seats or hs.seen >= epoch - STANDING_EPOCHS + 1]
    # Row digests are kept only where a successor can need them: each
    # hotkey's newest head (a successor's predecessor is its hotkey's head
    # before it) and the seat's heads. A head served again gets its own back.
    newest = {}
    for hs in kept:
        if hs.hotkey not in newest or (hs.block, hs.head) > (newest[hs.hotkey].block,
                                                           newest[hs.hotkey].head):
            newest[hs.hotkey] = hs
    kept = [hs if hs.head in seats or newest[hs.hotkey] is hs or not hs.rows
            else dataclasses.replace(hs, rows={}) for hs in kept]
    return SeatStateV4(champion=champion, champion_since=prev.champion_since, history=history,
                       heads=tuple(kept), epoch=epoch, holders=prev.holders)


def seat_serving(state: SeatStateV4, serving: Mapping[str, int], copy_state) -> dict[str, int]:
    """`serving` (copy_rule.classify's owners) as the seat reads it: a head
    the seat keeps whose copy record now names a different first commit is
    not that head. The copy rule drops a retired head's record after a few
    epochs, and a hash committed afresh after that is owned by whoever
    committed it; the seat keeps its own heads (a holder keeps the seat until
    replaced), so it serves, stands and pays only the head it recorded: the
    same commit, or the same bytes committed again by the coldkey that owns
    the head (its operator back after a lapse), never another coldkey's."""
    out = {}
    for h, uid in sorted(serving.items()):
        hs, rec = state.head(h), copy_state.get(h)
        if (hs is not None and rec is not None and rec.first_block != hs.block
                and rec.owner != hs.owner):
            continue
        out[h] = int(uid)
    return out


def odds_since(state: SeatStateV4, head: str, default: int) -> int:
    """The first epoch `head`'s odds are computed for (`HeadState.odds_since`),
    `default` for a head the state does not keep."""
    hs = state.head(head)
    return int(default) if hs is None else hs.odds_since


def lineage_records(records: Mapping[int, object], state: SeatStateV4) -> dict:
    """The challenge records with each successor's predecessors' challenges
    read as its own in the epochs it has none, and a challenge paired with a
    predecessor read as paired with its successor: its evidence is theirs.
    A predecessor's challenges are read only from epochs before the
    successor's commit (`since`): after it the owner's hotkey commits the
    successor, so the predecessor's hash served later is another coldkey's
    commit of the same bytes, whose answers are never the successor's."""
    preds = {hs.head: hs.predecessors for hs in state.heads if hs.predecessors}
    if not preds:
        return dict(records)
    since = {hs.head: hs.since for hs in state.heads}
    successor = {p: h for h, ps in sorted(preds.items()) for p in ps}
    out = {}
    for e, rec in records.items():
        extra = {}
        for h, ps in sorted(preds.items()):
            if h in rec.values or e >= since[h]:
                continue
            for p in ps:
                if p in rec.values:
                    extra[h] = rec.values[p]
                    break
        holder = successor.get(rec.holder, rec.holder)
        if extra or holder != rec.holder:
            rec = dataclasses.replace(rec, values={**rec.values, **extra}, holder=holder)
        out[e] = rec
    return out


def predecessor_chain(hs: HeadState) -> tuple[tuple[str, int], ...]:
    """The successor's predecessors as labels_v4.Side.chain reads them: most
    recent first, each with the epoch it was the line's head until (the
    `since` of the head after it: the successor's own for the first)."""
    until = ((hs.since,) + tuple(hs.predecessor_since))[:len(hs.predecessors)]
    return tuple(zip(hs.predecessors, until, strict=True))
def challenge_holder(state: SeatStateV4, epoch: int) -> str | None:
    """The holder epoch `epoch`'s challenges pair with: the holder after
    epoch - HOLDER_LAG, as the state records it after the newest epoch it
    decided at or before that one (a node that did not run an epoch has the
    seat as it last decided it, as every node does at a gap), or None when
    it records none (as at the v4 activation)."""
    want = int(epoch) - HOLDER_LAG
    for e, h in state.holders:              # newest first
        if e <= want:
            return h
    return None


def side_of(hs: HeadState):
    """The head as a comparison reads it (labels_v4.Side): its owner, its
    commit epoch and the epochs its inherited rows count from. Rules A and C
    and the memorisation verdict read a head through this one view; a
    successor's carries its predecessors (`predecessor_chain`), whose stored
    odds stand for its own before its commit (labels_v4.predecessor_odds),
    the rows its line added (`added`) and its line origin's commit (`line`),
    from which a missing row counts. All of it is in the published state,
    so a rebuild, the v4 replay and a follower read the same sides."""
    from fugal_subnet import labels_v4

    return labels_v4.Side(hs.head, hs.owner, hs.since, dict(hs.rows_since),
                          predecessor_chain(hs), added=frozenset(hs.added),
                          line=hs.line_since)


def challenged_recently(records: Mapping[int, object], head: str, *, epoch: int) -> bool:
    """Whether `head` has a counted challenge in `epoch` or one of the
    STANDING_EPOCHS - 1 before it, in `records` (the challenge records read
    through `lineage_records`, so a predecessor's challenges count): a
    contender's condition (`decide`), and with the holder the heads whose
    late odds come first (`late_odds_first`). One predicate for both."""
    from fugal_subnet.picks_v4 import answered_in

    return any(answered_in(records, head, e)
               for e in range(int(epoch) - STANDING_EPOCHS + 1, int(epoch) + 1))


def late_odds_first(state: SeatStateV4, window, *, epoch: int) -> frozenset[str]:
    """The heads whose late odds are computed before any other's
    (v2_epoch.late_epochs `first`): the seat holder and every head the seat
    keeps with a counted challenge in the last STANDING_EPOCHS epochs
    (`challenged_recently`, the contenders' condition) on the window holding
    `epoch`. A function of the window and the seat state alone."""
    records = lineage_records(window.picks, state)
    out = {hs.head for hs in state.heads if challenged_recently(records, hs.head, epoch=epoch)}
    if state.champion is not None:
        out.add(state.champion)
    return frozenset(out)


def precedence(hs: HeadState, *, floor_block: int | None, salt: str) -> tuple:
    """Rule C's order, lower is earlier: copy_rule.seniority of the head's
    origin (commits at or before `floor_block`, the v4 activation block, tie
    there and are ordered by a digest salted with its block hash), then the
    head's own hash."""
    from fugal_subnet import copy_rule

    rec = copy_rule.HeadRecord(hs.origin, hs.origin_block, hs.origin_owner, True)
    key = copy_rule.seniority(copy_rule.CopyState(heads=(rec,)), hs.origin,
                              floor_block=floor_block, salt=salt)
    return (int(key[0]), str(key[1]), hs.head)


# --- the decision --------------------------------------------------------------------------


@dataclass(frozen=True)
class SeatDecisionV4:
    """One epoch's seat. `event`: "fill" (an empty seat taken), "take",
    "hold" (the holder is the bar and keeps the seat), "open" (the bar is R
    and no qualifier: the holder keeps the seat), "empty" (no holder and no
    qualifier). `bar` is the holder when it is the bar, None when it is R.
    `evidence` is the published `seat_v4` block; `payable` the seat heads
    that are paid; `tested` each qualifier candidate's rule C comparisons
    (the earlier contenders it was tested against)."""

    state: SeatStateV4
    champion: str | None
    event: str
    bar: str | None
    detail: tuple[tuple[str, str], ...]
    evidence: dict
    payable: frozenset[str]
    tested: dict[str, tuple[str, ...]]


def _bound_doc(b: Bound | None) -> dict | None:
    return None if b is None else {"n": b.n, "lower": b.lower, "passes": b.passes}


def _earlier_item(who: str, b: Bound) -> str:
    """One rule C comparison as published: "who:n:lower:passes", `who` the
    contender's index in the evidence's `order` (or R), `lower` the bound in
    units ("-" with no labels), `passes` 1 or 0. One short line each: rule C
    can test every qualifier against every earlier contender."""
    return f"{who}:{b.n}:{'-' if b.lower is None else b.lower}:{int(b.passes)}"


def earlier_rows(evidence: Mapping, head: str) -> list[tuple[str, int, int | None, bool]]:
    """A head's published rule C comparisons as (the contender's hash, or R;
    labels; the bound in units; whether it beat it)."""
    out = []
    for item in evidence["heads"][head]["earlier"]:
        who, n, lower, passes = item.split(":")
        out.append((R if who == R else evidence["order"][int(who)], int(n),
                    None if lower == "-" else int(lower), passes == "1"))
    return out


def _picks(records, head: str, holder: str | None, *, open_seat: bool, through: int):
    """Rule B: against the holder; when the seat is open, against the last
    holder if 1,000 questions are paired with it, else the count alone."""
    from fugal_subnet.picks_v4 import picks_check

    kw = dict(through=through, margin_units=-PICKS_MARGIN_UNITS, min_questions=PICKS_MIN_QUESTIONS)
    if not open_seat:
        return picks_check(records, head, holder, **kw)
    if holder is not None and holder != head:
        ev = picks_check(records, head, holder, **kw)
        if ev.paired >= PICKS_MIN_QUESTIONS:
            return ev
    return picks_check(records, head, None, **kw)


def routes_worse(records, head: str, holder: str, *, through: int) -> Bound:
    """The bound of the holder's net value over `head` on `head`'s answered
    run, margin +1.5 points: passing it shows `head` routes worse."""
    from fugal_subnet.picks_v4 import answered_run, picks_items

    run = answered_run(records, head, through)
    items = picks_items(records, head, holder, run)
    return lower_bound({c: [-x for x in v] for c, v in items.items()},
                       margin_units=PICKS_MARGIN_UNITS, n_min=PICKS_MIN_QUESTIONS,
                       n0=BOUND_N0)


def must_beat(order: list[str], i: int, *, open_seat: bool) -> list[str | None]:
    """Rule C: the contenders the i-th contender in precedence order must
    beat on odds to qualify, earliest first: R (None), which is the earliest
    head of all, then every contender earlier than it. When the seat is open
    the bar is R and beating the bar is beating R, so R is not tested twice."""
    return ([None] if not open_seat else []) + list(order[:i])


# Rule C's comparisons are computed for every candidate at once: first each
# candidate's earliest RULE_C_FIRST_ROUND, then the rest for those that passed
# all of them. Only what the rule then reads is published; a comparison past a
# candidate's first failure is computed at most within its round.
RULE_C_FIRST_ROUND = 16


def _prepare_rule_c(prepare, order: list[str], picks_of: Mapping, *, open_seat: bool,
                    passes) -> None:
    """Fill rule C's comparisons for every contender that may qualify (it
    answered this epoch and passed rule B), earliest first, in two rounds
    (RULE_C_FIRST_ROUND). Decides nothing: `decide` reads them in its order."""
    alive = [(i, h) for i, h in enumerate(order) if h in picks_of and picks_of[h].passes]
    done = dict.fromkeys((h for _i, h in alive), 0)
    step = RULE_C_FIRST_ROUND
    while alive:
        wanted, parts = [], []
        for i, h in alive:
            part = must_beat(order, i, open_seat=open_seat)[done[h]:done[h] + step]
            parts.append(part)
            wanted += [(h, k) for k in part]
        prepare(wanted)
        nxt = []
        for (i, h), part in zip(alive, parts):
            done[h] += len(part)
            if (all(passes(h, k) for k in part)
                    and done[h] < len(must_beat(order, i, open_seat=open_seat))):
                nxt.append((i, h))
        alive, step = nxt, len(order) + 1


def decide(state: SeatStateV4, *, epoch: int, window, serving: Mapping[str, int],
           floor_block: int | None, salt: str, memo_refused: Iterable[str] = frozenset(),
           allow_take: bool = True, per_pair: bool = False) -> SeatDecisionV4:
    """Rules A, B and C on a state `observe` has brought to `epoch`, and the
    window holding `epoch` (module docstring, in that order). Pure.

    `per_pair` computes every rule A comparison with its own
    labels_v4.gain_strata call instead of the batches of gain_strata_many:
    the same decision by construction (tests/test_seat_v4_batched.py holds
    the batched path to it)."""
    from fugal_subnet import labels_v4
    from fugal_subnet.picks_v4 import answered_in, evidence_to_reveal

    epoch = int(epoch)
    refused = frozenset(memo_refused)
    heads = state.by_head()
    records = lineage_records(window.picks, state)
    ref = labels_v4.reference(window, now=epoch)
    detail: list[tuple[str, str]] = []
    cache: dict[tuple, Bound] = {}

    def side(h):
        return labels_v4.REFERENCE if h is None else side_of(heads[h])

    def prepare(wanted) -> None:
        """Rule A's bounds for these (x, y) pairs, computed together
        (labels_v4.gain_strata_many: what a head contributes is computed once
        per epoch, not once per pair). Each is the bound `beats` gives."""
        todo = list(dict.fromkeys(p for p in wanted if p not in cache))
        if not todo:
            return
        if per_pair:
            sums = [labels_v4.gain_strata(window, ref, side(x), side(y), now=epoch)
                    for x, y in todo]
        else:
            sums = labels_v4.gain_strata_many(window, ref,
                                              [(side(x), side(y)) for x, y in todo], now=epoch)
        for p, s in zip(todo, sums):
            cache[p] = bound_from_strata(s, margin_units=ODDS_MARGIN_UNITS,
                                         n_min=ODDS_MIN_PAIRS, n0=BOUND_N0)

    def beats(x, y) -> Bound:
        """Rule A: x beats y (None is R)."""
        if (x, y) not in cache:
            prepare([(x, y)])
        return cache[(x, y)]

    def answered_recently(h):
        return challenged_recently(records, h, epoch=epoch)

    def stands(h):
        return heads[h].seen >= epoch - STANDING_EPOCHS + 1

    def prec(h):
        return precedence(heads[h], floor_block=floor_block, salt=salt)

    # 1. The bar.
    holder = state.champion
    bar = None
    holder_doc = None
    if holder is not None:
        h_stands, r_beats, h_refused = stands(holder), beats(None, holder), holder in refused
        holder_doc = {"head": holder, "stands": h_stands, "r_beats": _bound_doc(r_beats),
                      "refused": h_refused}
        if h_stands and not r_beats.passes and not h_refused:
            bar = holder
        else:
            why = (f"has not been verified in the last {STANDING_EPOCHS} epochs" if not h_stands
                   else "the reference beats its odds" if r_beats.passes
                   else "refused by the memorisation check")
            detail.append((holder, f"the seat is open: the holder {why}; the bar is R"))
    open_seat = bar is None

    # 2. Contenders. Every head that reaches rule A against the bar is
    # compared with it in one batch first.
    prepare([(h, bar) for h in sorted(heads) if h != holder and stands(h) and h not in refused
             and answered_recently(h)])
    docs: dict[str, dict] = {}
    contenders: list[str] = []
    for h in sorted(heads):
        if h == holder:
            continue
        hs = heads[h]
        doc = {"uid": serving.get(h), "precedence": list(prec(h)[:2]), "origin": hs.origin,
               "stands": stands(h), "seen": hs.seen, "answered_recent": answered_recently(h),
               "answered": answered_in(records, h, epoch), "refused": h in refused,
               "vs_bar": None, "routes_worse": None, "contender": False, "picks": None,
               "earlier": [], "qualifies": False, "reasons": []}
        docs[h] = doc
        why = doc["reasons"]
        if not doc["stands"]:
            why.append(f"not verified in the last {STANDING_EPOCHS} epochs")
            continue
        if doc["refused"]:
            why.append("refused by the memorisation check")
            continue
        if not doc["answered_recent"]:
            why.append(f"has no counted challenge in the last {STANDING_EPOCHS} epochs, so it "
                       "bars no one")
            continue
        vb = beats(h, bar)
        doc["vs_bar"] = _bound_doc(vb)
        if not vb.passes:
            why.append("its odds do not beat the " + ("reference" if open_seat else "holder's")
                       + f" by {float(ODDS_MARGIN_POINTS):g} Brier points (rule A)")
            continue
        if holder is not None:
            rw = routes_worse(records, h, holder, through=epoch)
            doc["routes_worse"] = _bound_doc(rw)
            if rw.passes:
                why.append("its real picks are shown to be worse than the holder's by more than "
                           f"{PICKS_MARGIN_UNITS / UNITS_PER_POINT:g} points")
                continue
        doc["contender"] = True
        contenders.append(h)
    order = sorted(contenders, key=prec)
    position = {h: i for i, h in enumerate(order)}

    # 3. Qualifiers.
    qualifiers: list[str] = []
    tested: dict[str, tuple[str, ...]] = {}
    last_holder = holder if holder is not None else (state.history[0] if state.history else None)
    picks_of = {h: _picks(records, h, last_holder if open_seat else holder, open_seat=open_seat,
                          through=epoch)
                for h in order if allow_take and docs[h]["answered"]}
    _prepare_rule_c(prepare, order, picks_of, open_seat=open_seat,
                    passes=lambda x, y: cache[(x, y)].passes)
    for i, h in enumerate(order):
        doc, why = docs[h], docs[h]["reasons"]
        if not allow_take:
            why.append("this validator's window is not trusted yet: no take this epoch")
            continue
        if not doc["answered"]:
            why.append("has no counted challenge this epoch")
            continue
        pe = picks_of[h]
        doc["picks"] = evidence_to_reveal(pe, answered_now=True)
        if not pe.passes:
            why.append(("its answered run holds fewer than "
                        f"{PICKS_MIN_QUESTIONS} challenge items" if pe.bar is None
                        else "its real picks do not pass the check against "
                        f"{pe.bar[:16]} (rule B)"))
            continue
        earlier = must_beat(order, i, open_seat=open_seat)
        ok = True
        for k in earlier:
            b = beats(h, k)
            doc["earlier"].append(_earlier_item(R if k is None else str(position[k]), b))
            if not b.passes:
                why.append(f"does not beat the earlier contender {R if k is None else k[:16]} "
                           "on odds (rule C)")
                ok = False
                break
        tested[h] = tuple(k for k in earlier[:len(doc["earlier"])] if k is not None)
        if ok:
            doc["qualifies"] = True
            qualifiers.append(h)

    # 4. The seat.
    history = tuple(state.history)
    if qualifiers:
        winner = qualifiers[-1]
        event = "take" if holder is not None else "fill"
        if holder is not None:
            history = (holder,) + tuple(x for x in history if x not in (holder, winner))
        else:
            history = tuple(x for x in history if x != winner)
        champion, since = winner, epoch
        detail.append((winner, ("took the empty seat" if holder is None else
                                f"took the seat from {holder[:16]}")
                       + f", the latest of {len(qualifiers)} qualifier(s)"))
    else:
        winner = None
        champion, since = holder, state.champion_since
        event = "empty" if holder is None else ("open" if open_seat else "hold")
    new_state = dataclasses.replace(
        state, champion=champion, champion_since=since, history=history,
        # The holders after this epoch and the one before, keyed by epoch
        # whether or not this node ran that one (the newest it recorded at
        # or before it), so the published state does not depend on which
        # epochs a node ran.
        holders=((epoch, champion), (epoch - 1, challenge_holder(state, epoch + HOLDER_LAG - 1))))

    # Pay.
    payable = set()
    pay_doc = {}
    for i, h in enumerate(([champion] if champion is not None else []) + list(history[:2])):
        live = h in serving
        ok = live and h not in refused and not beats(None, h).passes
        pay_doc[h] = ok
        if ok:
            payable.add(h)
    evidence = {
        "epoch": epoch, "event": event, "bar": bar, "open": open_seat, "holder": holder_doc,
        "order": order, "qualifiers": qualifiers, "winner": winner, "paid": pay_doc,
        "heads": {h: docs[h] for h in sorted(docs)},
    }
    return SeatDecisionV4(new_state, champion, event, bar, tuple(detail), evidence,
                          frozenset(payable), tested)


def seat_step(prev: SeatStateV4, *, epoch: int, window, serving: Mapping[str, int],
              rows: Mapping[str, Mapping[str, str]], copy_state, menu_ids: Iterable[str],
              blocks_per_epoch: int, floor_block: int | None, salt: str,
              memo_refused: Iterable[str] = frozenset(), allow_take: bool = True
              ) -> SeatDecisionV4:
    """The v4 seat for one epoch, a pure function of the previous seat state,
    the window (labels, odds and real-picks records through `epoch`), the
    verified heads served by their owners (`serving`) and their row digests
    (`rows`), the copy rule's state (`copy_state`), the epoch's menu
    (`menu_ids`), the precedence floor and salt, and the memorisation
    verdict. `observe`, then `decide`."""
    state = observe(prev, epoch=epoch, serving=serving, rows=rows, copy_state=copy_state,
                    menu_ids=menu_ids, blocks_per_epoch=blocks_per_epoch)
    return decide(state, epoch=epoch, window=window, serving=serving, floor_block=floor_block,
                  salt=salt, memo_refused=memo_refused, allow_take=allow_take)


def decision_for_reveal(decision: SeatDecisionV4) -> dict:
    """The reveal's `champion` block under v4: the holder a miner reads
    (`champion`), the event, the bar (None: R) and the reasons in words. The
    seat state is in `v2_state.seat`, the evidence in `seat_v4`."""
    return {"event": decision.event, "champion": decision.champion, "bar": decision.bar,
            "detail": [[h, why] for h, why in decision.detail]}


def digest(block: Mapping) -> str:
    """One hash over the epoch's published `seat_v4` block, for `v2_state`."""
    import json

    canonical = json.dumps(block, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def carry_seen(state: SeatStateV4, own: Iterable[tuple[int, Mapping]]) -> SeatStateV4:
    """A re-anchored seat state (v2_sync.resync) brought forward over this
    validator's own reveals after the snapshot, as `observe` recorded those
    epochs: each head that a reveal of epoch j lists as served by its owner
    is seen at j; a head the snapshot does not keep is taken as that reveal's
    own seat state records it (a new head, or a successor, which replaces its
    predecessors and takes over their seats). Seat decisions after the
    snapshot are not replayed: the snapshot's seat is the reference's."""
    heads = state.by_head()
    champion, history = state.champion, list(state.history)
    for j, reveal in sorted(own, key=lambda t: t[0]):
        j = int(j)
        block = reveal.get("v2_state") or {}
        try:
            recorded = from_dict(block["seat"]) if is_v4_dict(block.get("seat")) else None
        except ValueError:
            recorded = None
        for h in sorted(block.get("serving") or {}):
            if h in heads:
                if heads[h].seen < j:
                    heads[h] = dataclasses.replace(heads[h], seen=j)
                continue
            hs = recorded.head(h) if recorded is not None else None
            if hs is None:
                continue
            heads[h] = dataclasses.replace(hs, seen=j)
            for p in hs.predecessors:
                if heads.pop(p, None) is not None:
                    champion = h if champion == p else champion
                    history = [h if x == p else x for x in history]
    return dataclasses.replace(state, champion=champion, history=tuple(history),
                               heads=tuple(heads.values()))


__all__ = [
    "HeadState", "MAX_STATE_HEADS", "ODDS_MARGIN_UNITS", "ODDS_MIN_PAIRS", "PICKS_MIN_QUESTIONS",
    "R", "REVEAL_KEY", "STANDING_EPOCHS", "SeatDecisionV4", "SeatStateV4", "carry_seen",
    "decide", "decision_for_reveal", "digest", "earlier_rows", "from_dict", "is_successor",
    "is_v4_dict", "seat_serving", "challenge_holder",
    "lineage_records", "must_beat", "observe", "odds_since", "precedence", "routes_worse", "row_digests",
    "seat_step", "side_of", "to_dict",
]
