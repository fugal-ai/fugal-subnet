"""Protocol v2 copy rule: who is senior, who owns a head, and ghost incumbents.

From the activation block, a head that is a copy of an older head must not
earn. v1 tried to RECOGNISE copies (routing agreement, weight distance, the
epsilon clause in `dedup.py`), and measured offline that cannot work: a lightly
perturbed copy sits further from its source than two honest refits on the
same data sit from each other, so no threshold separates
them. v2 stops recognising copies and asks one question instead: is this head
measurably BETTER than every head older than it? A copy is not, so it never
takes the seat, whatever it looks like.

Three rules, all v2-only (v1 keeps `commitments.seniority_key` and `dedup.py`
unchanged):

1. **Seniority is the head's first commit.** A head's age is the block at
   which its exact weights hash was first committed by any hotkey, not the
   registration block of whoever serves it now. An older hotkey that copies a
   newer head is junior to it. Ties inside one block are broken by
   sha256("fugal-v2-owner:" || head hash || ":" || coldkey), lowest first:
   deterministic and not chosen by the committer's hotkey address.
2. **A head belongs to the coldkey that first committed it.** The same hash
   served under another coldkey is an exact copy and is ineligible. A chain
   coldkey swap moves the first committer's hotkey to a new coldkey, and
   ownership follows that hotkey. Keyed by
   coldkey, not hotkey, so an operator whose hotkey is pruned and who
   re-registers keeps the head's age. Two hotkeys of the OWNER serving one
   hash are one head: the canonical one is decided by `classify`, the other
   is a duplicate and earns nothing.
3. **Ghost incumbents.** A head is retired the first time its owner no longer
   commits it at a boundary (it committed a different head, or its hotkey
   left the subnet). A retired head that was scored stays in the comparison
   set, unpaid, for `CHAMPION_GHOST_RETIRE` epochs from that first
   retirement, so perturbed copies of it cannot become incumbents the moment
   its author moves on. A head its owner still commits but that missed stays
   in the set for `CHAMPION_GHOST_OFF` epochs after the last epoch it was
   scored, the same span as a champion that missed.

The bar a junior must clear is the champion test itself: `senior_bars` lists,
for each head, the senior SEAT HOLDERS in the comparison set (the paid seats
and the champion's ghosts, `seat_holders`; a retired seat holder keeps
barring through its retire window), and
`champion.seat_step` requires a head to beat each of them by
`CHAMPION_MARGIN` (paired LCB, the same `paired_test`) before it can take a
seat, first or by challenge. There is no second margin.

STATE. First-commit blocks, owners and retirements are history, so every
validator must hold the same record. It is a fold over boundary snapshots of
the chain's commitments (`observe`), read pinned at each epoch's boundary
block exactly as the validator already reads them. The post-epoch state is
published in the v2 reveal's single `v2_state` block
(`CopyState.to_reveal`, under `REVEAL_KEY`), so it is rebuilt from the
newest published reveal (`rebuild_from_reveals`) or re-derived from an
archive node (`replay`), never from one validator's memory.

BOUNDS (I2). A commitment is miner-chosen text. Only a 64-character
lowercase hex hash, an ss58-length hotkey and coldkey and a block in
(0, boundary] are accepted; anything else is ignored. A snapshot is capped at
`MAX_COMMITMENTS` entries, kept in (block, hotkey) order. The state holds a
record only while some hotkey commits the hash or while it is a ghost, so it
is bounded by the number of hotkeys times (CHAMPION_GHOST_RETIRE + 1).

Not covered here, stated in docs/INVARIANTS.md: commitments are the plain
public hash, so whoever gets a seen hash on chain first is its first
committer (binding the commitment to the coldkey is deferred to a later
release).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# A subnet has at most 256 UIDs; this is generous headroom and only bounds work.
MAX_COMMITMENTS = 4096
# A loaded state larger than this is refused. A live state holds at most one
# head per hotkey per epoch of the retire window plus the live ones, far below.
MAX_STATE_HEADS = 65536
# This module's key inside the v2 reveal's single `v2_state` block.
REVEAL_KEY = "copy_rule"
# ss58 addresses are 47-48 characters; anything much longer is not one.
MAX_ADDRESS_LEN = 64
_HASH = re.compile(r"[0-9a-f]{64}")
_EPOCH_ID = re.compile(r"e(\d{1,12})")


@dataclass(frozen=True)
class Commitment:
    """One hotkey's commitment as read at an epoch boundary."""

    hotkey: str
    coldkey: str
    head_hash: str
    block: int


@dataclass(frozen=True)
class HeadRecord:
    head_hash: str
    first_block: int
    owner: str                    # the coldkey that first committed it
    active: bool                  # its owner committed it at the last boundary
    retired_at: int | None = None  # the epoch it FIRST stopped being active
    # The hotkey of that first commit. Informational (ownership is the
    # coldkey); among one coldkey's hotkeys in one block, the lowest address.
    first_hotkey: str = ""


@dataclass(frozen=True)
class CopyState:
    """The copy rule's history. Immutable; `observe` returns a new one."""

    epoch: int | None = None
    heads: tuple[HeadRecord, ...] = ()   # sorted by head_hash

    def __post_init__(self):
        object.__setattr__(
            self, "heads", tuple(sorted(self.heads, key=lambda r: r.head_hash)))
        object.__setattr__(self, "_by_hash", {r.head_hash: r for r in self.heads})

    def get(self, head_hash: str) -> HeadRecord | None:
        return self._by_hash.get(head_hash)

    def __contains__(self, head_hash: str) -> bool:
        return head_hash in self._by_hash

    def to_reveal(self) -> dict:
        """The post-epoch state as a plain JSON-serialisable dict.

        This is the copy rule's part of the v2 reveal's single `v2_state`
        block (under `REVEAL_KEY`). Every map is keyed by weights hash and
        emitted in sorted order, so two validators with the same state
        publish the same bytes. A ghost bars until
        `retired_at + CHAMPION_GHOST_RETIRE - 1`; the expiry is derived, not
        stored, so it cannot disagree with `retired_at`.
        """
        return {
            "epoch": self.epoch,
            "first_commit": {r.head_hash: r.first_block for r in self.heads},
            "ownership": {r.head_hash: r.owner for r in self.heads},
            "first_hotkey": {r.head_hash: r.first_hotkey for r in self.heads},
            "active": [r.head_hash for r in self.heads if r.active],
            "retired_at": {r.head_hash: r.retired_at for r in self.heads
                           if r.retired_at is not None},
        }

    @classmethod
    def from_reveal(cls, d: Mapping) -> CopyState:
        """Inverse of `to_reveal`. Validates and bounds what it reads (I2).

        A published reveal is a file someone else wrote: a hash that is not
        64 lowercase hex, an owner that is not an address, a block or epoch
        that is not a positive integer, a map larger than `MAX_STATE_HEADS`,
        or an entry in one map with no first commit is refused with
        ValueError rather than half-loaded.
        """
        if not isinstance(d, Mapping):
            raise ValueError("copy-rule state must be a mapping")
        first = d.get("first_commit", {})
        owner = d.get("ownership", {})
        active = d.get("active", [])
        retired = d.get("retired_at", {})
        hotkeys = d.get("first_hotkey", {})
        if not (isinstance(first, Mapping) and isinstance(owner, Mapping)
                and isinstance(active, list) and isinstance(retired, Mapping)
                and isinstance(hotkeys, Mapping)):
            raise ValueError("copy-rule state has a field of the wrong type")
        if len(first) > MAX_STATE_HEADS:
            raise ValueError(f"copy-rule state holds more than {MAX_STATE_HEADS} heads")
        if set(owner) != set(first) or set(hotkeys) != set(first) \
                or not set(active) <= set(first) or not set(retired) <= set(first):
            raise ValueError("copy-rule state maps do not share one set of heads")
        active_set = set(active)
        heads = []
        for h, block in first.items():
            ck = owner[h]
            hk = hotkeys[h]
            ra = retired.get(h)
            if not (isinstance(h, str) and _HASH.fullmatch(h)):
                raise ValueError("copy-rule state holds an invalid head hash")
            if not (isinstance(ck, str) and 0 < len(ck) <= MAX_ADDRESS_LEN):
                raise ValueError("copy-rule state holds an invalid owner")
            if not (isinstance(hk, str) and 0 < len(hk) <= MAX_ADDRESS_LEN):
                raise ValueError("copy-rule state holds an invalid first hotkey")
            if not _is_int(block) or block <= 0:
                raise ValueError("copy-rule state holds an invalid first-commit block")
            if ra is not None and (not _is_int(ra) or ra < 0):
                raise ValueError("copy-rule state holds an invalid retirement epoch")
            is_active = h in active_set
            if not is_active and ra is None:
                raise ValueError("an inactive head must carry its retirement epoch")
            heads.append(HeadRecord(h, block, ck, is_active, ra, hk))
        e = d.get("epoch")
        if e is not None and (not _is_int(e) or e < 0):
            raise ValueError("copy-rule state holds an invalid epoch")
        return cls(epoch=e, heads=tuple(heads))


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _claim(block: int, head_hash: str, coldkey: str) -> tuple[int, str]:
    """Who committed first: by block, then by a digest no hotkey address can steer."""
    digest = hashlib.sha256(f"fugal-v2-owner:{head_hash}:{coldkey}".encode()).hexdigest()
    return int(block), digest


def seniority(state: CopyState, head_hash: str) -> tuple:
    """The head's seniority key, lower is senior: (first-commit block, owner digest).

    A head with no record (never seen committed) is junior to every head with one.
    """
    r = state.get(head_hash)
    if r is None:
        return (float("inf"), head_hash)
    return _claim(r.first_block, r.head_hash, r.owner)


# --- The names the v2 validator wiring calls. -------------------------------


def owner(state: CopyState, head_hash: str) -> tuple[str, str] | None:
    """(coldkey, hotkey) that FIRST committed `head_hash`, or None if never seen.

    The coldkey is the owner (rule 1b); the hotkey is the one it committed
    from, for logs and for the wiring's choice of which UID serves the head.
    """
    r = state.get(head_hash)
    return None if r is None else (r.owner, r.first_hotkey)


def first_commit_block(state: CopyState, head_hash: str) -> int | None:
    """The block at which `head_hash` was first committed by any hotkey."""
    r = state.get(head_hash)
    return None if r is None else r.first_block


def is_exact_copy(
    state: CopyState, coldkey: str, head_hash: str, hotkey: str | None = None,
) -> bool:
    """True iff `head_hash` is owned, and the server is not its owner.

    The serving UID's coldkey (read pinned at the boundary) goes in
    `coldkey`, its hotkey in `hotkey`. The hotkey that first committed the
    head is never a copy of it: a chain coldkey swap moves that hotkey to a
    new coldkey, and ownership follows it. Otherwise the coldkeys are
    compared. An unknown hash is not a copy but is not eligible either; see
    `classify`, which is this applied to every serving UID at once.
    """
    r = state.get(head_hash)
    if r is None:
        return False
    if hotkey is not None and hotkey == r.first_hotkey:
        return False
    return r.owner != coldkey


def v2_seniority(state: CopyState, head_hash: str) -> tuple:
    """Seniority for protocol v2: the head's first-commit block, never the
    registration block. An alias of `seniority`, named for the wiring."""
    return seniority(state, head_hash)


def valid_commitments(
    commitments: Iterable[Commitment], boundary_block: int,
) -> list[Commitment]:
    """The bounded, well-formed subset of a boundary snapshot, one per hotkey.

    Ordered by (block, hotkey). A hotkey listed twice keeps its first entry in
    that order (a chain read cannot list a hotkey twice; a hand-edited reveal
    can).
    """
    out: list[Commitment] = []
    for c in commitments:
        if not (isinstance(c.hotkey, str) and isinstance(c.coldkey, str)
                and isinstance(c.head_hash, str)):
            continue
        if not (0 < len(c.hotkey) <= MAX_ADDRESS_LEN and 0 < len(c.coldkey) <= MAX_ADDRESS_LEN):
            continue
        if not _HASH.fullmatch(c.head_hash):
            continue
        if isinstance(c.block, bool) or not isinstance(c.block, int):
            continue
        if not 0 < c.block <= int(boundary_block):
            continue
        out.append(c)
    out.sort(key=lambda c: (c.block, c.hotkey, c.coldkey, c.head_hash))
    seen: set[str] = set()
    unique = []
    for c in out:
        if c.hotkey in seen:
            continue
        seen.add(c.hotkey)
        unique.append(c)
    return unique[:MAX_COMMITMENTS]


def observe(
    state: CopyState,
    epoch: int,
    boundary_block: int,
    commitments: Iterable[Commitment],
    *,
    ghost_retire: int,
) -> CopyState:
    """Fold one epoch's boundary snapshot into the state. Pure; order-free.

    `commitments` are the hotkeys registered on the subnet at the boundary,
    each with its owning coldkey and its commitment, read pinned at
    `boundary_block`. Epochs must be applied in increasing order and once.
    """
    if state.epoch is not None and int(epoch) <= state.epoch:
        raise ValueError(f"epoch {epoch} is not after the last observed epoch {state.epoch}")
    valid = valid_commitments(commitments, boundary_block)

    recs: dict[str, HeadRecord] = {r.head_hash: r for r in state.heads}
    for c in valid:
        r = recs.get(c.head_hash)
        if r is not None and c.hotkey == r.first_hotkey and c.coldkey != r.owner:
            # The first committer's hotkey now sits under another coldkey: a
            # chain coldkey swap. Ownership follows the hotkey, age is kept.
            r = HeadRecord(r.head_hash, r.first_block, c.coldkey, r.active,
                           r.retired_at, r.first_hotkey)
            recs[c.head_hash] = r
        claim = (_claim(c.block, c.head_hash, c.coldkey), c.hotkey)
        if r is None or claim < (_claim(r.first_block, r.head_hash, r.owner), r.first_hotkey):
            # An earlier claim than the one on record can only come from chain
            # state the record never saw; the chain is authoritative.
            keep_retired = r is not None and r.owner == c.coldkey
            recs[c.head_hash] = HeadRecord(
                c.head_hash, c.block, c.coldkey, False,
                r.retired_at if keep_retired else None, c.hotkey,
            )

    active = {c.head_hash for c in valid if recs[c.head_hash].owner == c.coldkey}
    committed = {c.head_hash for c in valid}
    out = []
    for h in sorted(recs):
        r = recs[h]
        is_active = h in active
        retired_at = r.retired_at
        if not is_active and retired_at is None:
            retired_at = int(epoch)
        # Kept while anyone still commits the hash (so a copier holding it
        # never inherits its age: once the record is gone, a fresh commit is
        # a new head at a new block) or while it is a ghost.
        if (not is_active and h not in committed
                and int(epoch) > retired_at + int(ghost_retire) - 1):
            continue
        out.append(HeadRecord(h, r.first_block, r.owner, is_active, retired_at, r.first_hotkey))
    return CopyState(epoch=int(epoch), heads=tuple(out))


def retired_hashes(state: CopyState) -> frozenset[str]:
    """Heads whose owner no longer commits them (`champion.seat_step`'s `retired`)."""
    return frozenset(r.head_hash for r in state.heads if not r.active)


def first_epoch(state: CopyState, head_hash: str, blocks_per_epoch: int) -> int | None:
    """The first epoch whose boundary is at or after the head's first commit.

    That is the first epoch the head can have been scored in, so it is the
    `since_epoch` of the head's `champion.Candidate`.
    """
    r = state.get(head_hash)
    if r is None:
        return None
    bpe = max(1, int(blocks_per_epoch))
    return -(-r.first_block // bpe)


def classify(state: CopyState, served: Mapping[int, Commitment]) -> dict[int, str]:
    """Per UID that served a verified proof this epoch: whose head is it?

    `served[uid]` is that UID's commitment at the boundary (the hash it served
    must equal it; the validator already enforces that). Verdicts:

    - "owner":     the canonical serving of a head by its owner. Its answers
                   are the head's answers in the comparison window.
    - "copy":      the hash is owned by another coldkey (rule 1b). Ineligible.
    - "duplicate": the owner serves the same hash from another hotkey too; the
                   one with the earliest commitment block (then hotkey, then
                   UID) is canonical and this one earns nothing.
    - "unknown":   no record of the hash (the snapshot it was in was not
                   observed). Ineligible.
    """
    verdict: dict[int, str] = {}
    owners: dict[str, list[tuple[int, str, int]]] = {}
    for uid in sorted(served):
        c = served[uid]
        r = state.get(c.head_hash)
        if r is None:
            verdict[uid] = "unknown"
        elif is_exact_copy(state, c.coldkey, c.head_hash, c.hotkey):
            verdict[uid] = "copy"
        else:
            owners.setdefault(c.head_hash, []).append((int(c.block), c.hotkey, uid))
    for _h, uids in owners.items():
        uids.sort()
        verdict[uids[0][2]] = "owner"
        for _, _, uid in uids[1:]:
            verdict[uid] = "duplicate"
    return verdict


def comparison_set(
    state: CopyState, epoch: int, window, *, ghost_off: int, ghost_retire: int,
) -> dict[str, str]:
    """The heads that bar their juniors at `epoch`: {head_hash: status}.

    `window` is the champion comparison window (`champion.Window`), already
    holding this epoch; a head is live when it has answers for `epoch`. Only
    heads with answers somewhere in the window can bar: a head never scored
    has no evidence for a junior to beat. Status:

    - "live":    scored this epoch;
    - "missed":  its owner still commits it, last scored within `ghost_off`
                 epochs;
    - "retired": its owner no longer commits it, within `ghost_retire` epochs
                 of its first retirement.
    """
    out: dict[str, str] = {}
    for r in state.heads:
        epochs = window.epochs_of(r.head_hash)
        if not epochs:
            continue
        if epochs[-1] == epoch:
            out[r.head_hash] = "live"
        elif r.active:
            if epoch <= epochs[-1] + int(ghost_off):
                out[r.head_hash] = "missed"
        elif r.retired_at is not None and epoch <= r.retired_at + int(ghost_retire) - 1:
            out[r.head_hash] = "retired"
    return out


def seat_holders(seat, n_seats: int) -> frozenset[str]:
    """The heads the copy rule protects: the paid seats and the champion's ghosts.

    `seat` is the `champion.SeatState` the epoch starts from. The seats are
    the ones `payout.seat_weights` pays: the champion, then former champions
    most recent first, `n_seats` in all (`len(PAYOUT_SEAT_SHARES)`). Only
    seats pay, so a copy of a head that holds no seat earns nothing anyway,
    and barring it would only add tests a better head can fail on noise.
    """
    holders: list[str] = [seat.champion] if seat.champion is not None else []
    for h in seat.history:
        if len(holders) >= int(n_seats) - (0 if seat.champion is not None else 1):
            break
        if h not in holders:
            holders.append(h)
    return frozenset(holders) | frozenset(g.head_hash for g in seat.ghosts)


def senior_bars(
    state: CopyState, epoch: int, window, heads: Iterable[str], *,
    holders: Iterable[str], ghost_off: int, ghost_retire: int,
) -> dict[str, tuple[str, ...]]:
    """For each head, the senior seat holders in the comparison set, most senior first.

    `holders` is `seat_holders(seat, n_seats)`. Pass the result to
    `champion.seat_step(senior_bars=...)`: a head takes a seat only by
    beating each of these by the champion margin. A seat holder that has
    retired keeps barring through its retire window (rule 1c), so copies of a
    retired seat holder cannot become incumbents.
    """
    cs = comparison_set(state, epoch, window, ghost_off=ghost_off, ghost_retire=ghost_retire)
    protected = set(holders)
    ranked = sorted((h for h in cs if h in protected), key=lambda h: (seniority(state, h), h))
    out: dict[str, tuple[str, ...]] = {}
    for h in sorted(set(heads)):
        mine = seniority(state, h)
        out[h] = tuple(k for k in ranked if k != h and seniority(state, k) < mine)
    return out


# ---------------------------------------------------------------------------
# Rebuild: from the chain (fold the boundary snapshots) or from a reveal (load
# the published post-epoch state).
# ---------------------------------------------------------------------------


def replay(
    snapshots: Iterable[tuple[int, int, Iterable[Commitment]]], *, ghost_retire: int,
    state: CopyState | None = None,
) -> CopyState:
    """Fold (epoch, boundary_block, commitments) snapshots, in epoch order.

    The chain-derived rebuild: with an archive node, read `CommitmentOf` (and
    each hotkey's owner) pinned at every v2 boundary since the activation
    block and fold them. It reaches the same state as the live fold because
    the live fold reads exactly those snapshots.
    """
    s = state or CopyState()
    for epoch, boundary, commitments in sorted(snapshots, key=lambda t: t[0]):
        s = observe(s, epoch, boundary, commitments, ghost_retire=ghost_retire)
    return s


def _epoch_index(epoch_id: str) -> int | None:
    m = _EPOCH_ID.fullmatch(str(epoch_id))
    return int(m.group(1)) if m else None


def from_rebuild(rebuilt) -> CopyState | None:
    """The copy-rule state from `v2_rebuild.rebuild_v2_from_reveals`'s result.

    That rebuild carries every later v2 state block verbatim in
    `V2Rebuild.extra`; this module's is under `REVEAL_KEY`. It is validated
    and bounded like any reveal input (`CopyState.from_reveal`), and refused
    if it is for a different epoch than the newest reveal it came from. None
    if the newest reveal carries no copy-rule state.
    """
    block = (getattr(rebuilt, "extra", None) or {}).get(REVEAL_KEY)
    if block is None:
        return None
    state = CopyState.from_reveal(block)
    newest = _epoch_index(getattr(rebuilt, "newest_epoch_id", None) or "")
    if newest is not None and state.epoch is not None and state.epoch != newest:
        raise ValueError(
            f"newest reveal is epoch {newest} but its copy-rule state is for epoch {state.epoch}")
    return state


def rebuild_from_reveals(source) -> CopyState | None:
    """The copy-rule state from the newest v2 reveal that carries it.

    Each v2 reveal carries the post-epoch state in its single `v2_state`
    block, this module's part under `REVEAL_KEY`, so a rebuild is a load of
    the newest one (by epoch id, never by file order), not a replay. None if
    no reveal carries it. The trust assumption is the reference frame's
    (`reference_frame.rebuild_from_reveals`): loading your own reveal is
    self-trusting; loading someone else's is a snapshot you choose to trust.
    Every field is chain state, so `replay` against an archive node checks it.
    """
    import glob
    import os

    if isinstance(source, (str, os.PathLike)):
        paths = sorted(glob.glob(os.path.join(str(source), "*", "reveal.json")))
    else:
        paths = sorted(source)

    best: tuple[int, Mapping] | None = None
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                reveal = json.load(f)
        except Exception as e:  # noqa: BLE001 - one bad file must not lose the rest
            logger.warning("Skipping unreadable reveal %s: %s", path, e)
            continue
        if not isinstance(reveal, Mapping):
            continue
        block = reveal.get("v2_state")
        epoch = _epoch_index(reveal.get("epoch_id", ""))
        if not isinstance(block, Mapping) or REVEAL_KEY not in block or epoch is None:
            continue
        if best is not None and epoch == best[0]:
            raise ValueError(f"two reveals for epoch {epoch}")
        if best is None or epoch > best[0]:
            best = (epoch, block[REVEAL_KEY])
    if best is None:
        return None
    state = CopyState.from_reveal(best[1])
    if state.epoch is not None and state.epoch != best[0]:
        raise ValueError(f"reveal for epoch {best[0]} carries copy-rule state of epoch {state.epoch}")
    return state
