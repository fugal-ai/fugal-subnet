"""Protocol v4: how the memorisation check's published set moves between nodes
(docs/decisions/SEAT_MINIMAL_V4.md 6; `memo_v4` decides what the set is).

THE RULE. The published set advances only by root reveals the chain shows
committed, and every node (the root, any other validator, a miner's TD)
advances it the same way, from the same pinned reads (`ChainSet`):

  at each epoch boundary b, read the root's reveal commitment pinned at b's
  block (the read the menu already makes). If it names a reveal c newer than
  every reveal folded so far, fold that reveal's questions (its bytes must
  hash to the committed sha256) as first published at b - READ_LAG;
  otherwise fold nothing at b - READ_LAG. The set as of epoch E is the state
  after boundary E: what epoch E draws on (`epoch_plan.plan_v4`) and what its
  memorisation verdicts read (`memo_v4.epoch_memo`), one view (`view`).

In steady state the root commits reveal E - 2 during E - 1, boundary E names
it, and the set is the root's reveals through E - 2, each at its own epoch.
When the root is down or late, nothing is folded and nothing is assumed: the
set stays as it was, and every node still builds and checks the same plan.
A reveal committed late is folded at the first boundary whose pinned read
names it, on every node alike; a reveal the chain never names at a boundary
(one published in a batch whose newest alone was committed, or committed
older than one already folded) is never folded. Nothing ever waits on the
root.

THE SNAPSHOT. Each v4 reveal names the view its epoch used (`memo_v4`:
`through`, `published_digest`). Before committing it, the root puts that
view's canonical bytes (`memo_v4.snapshot_bytes`) in its store under
`<netuid>/published/<digest>.json`, create-only, and reads them back
(`SnapshotPublisher`). A node without state anchors on the reveal the
current pinned read names: that reveal's named view by snapshot (checked
against the digest), then the boundaries after it. Before the first v4
reveal it starts from the pinned seed and replays the boundaries after it.
Every node checks each reveal it folds against its own state: a reveal
whose named view differs is an ALARM, and a TD makes no proof on it.

Nothing on the v1-v3 path imports this module.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping

from fugal_subnet import memo_v4
from fugal_subnet.benchmarks.slicer import epoch_id_for_block

logger = logging.getLogger(__name__)

# Where a snapshot is published, beside the menus: <netuid>/published/<digest>.json.
SNAPSHOT_PREFIX = "published"
# A validator's published set, beside its state file (written before it, as
# the window is).
STATE_SUFFIX = ".published_v4.json"
# A miner's TD keeps its set in its v4 state directory under this name.
TD_FILE = "published_v4.json"
# What either file holds: the set, the newest reveal folded, the boundary.
HELD_FORMAT = "published-v4-held-v1"
# One snapshot's upload and read-back, in seconds; past it the reveal is
# committed without the snapshot (a node starting then replays boundaries).
PUBLISH_BUDGET_S = 120.0
# The most boundaries one node replays in one go (one pinned read each): the
# seed is at most 150 epochs before the activation (the release gate).
MAX_FOLD_EPOCHS = 512
# How far back a node's chain endpoint must serve pinned reads, in epochs,
# for a node that missed epochs to re-anchor on a snapshot (`advance_to`):
# the boundary of the reveal the current commitment names (two epochs back
# in steady state) and every boundary after it, with the current epoch's
# own offset. Older reads need an archive endpoint.
ANCHOR_HISTORY_EPOCHS = 3
# What a pinned read of a block whose state the endpoint no longer keeps
# raises (async_substrate_interface.errors.StateDiscardedError, and the
# messages pruning nodes send), as opposed to a transient failure.
_PRUNED_MARKERS = ("state discarded", "state already discarded", "pruned", "unknownblock",
                   "unknown block")

_HEX64 = re.compile(r"[0-9a-f]{64}")


class PublishedUnavailable(ValueError):
    """This node does not hold the published set an epoch reads, and cannot
    obtain it: a TD produces no proof for the epoch, and a validator scores
    none of it (it treats it as an epoch with no proof it could check)."""


def is_pruned(e: BaseException) -> bool:
    """Whether a failed pinned chain read says the endpoint no longer keeps
    that block's state (it will never serve it), rather than a transient
    failure worth retrying."""
    if any(t.__name__ == "StateDiscardedError" for t in type(e).__mro__):
        return True
    text = str(e).lower()
    return any(m in text for m in _PRUNED_MARKERS)


def snapshot_object(netuid: int, digest: str) -> str:
    """The store key of the snapshot with this digest."""
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise ValueError("a published-set digest is 64 lowercase hex characters")
    return f"{int(netuid)}/{SNAPSHOT_PREFIX}/{digest}.json"


def named_view(reveal) -> tuple[int, str] | None:
    """(through, digest) of the published set a v4 reveal names
    (`memo_v4.through`, `memo_v4.published_digest`), or None when it names
    none (a v3 reveal) or names it in another shape."""
    block = reveal.get("memo_v4") if isinstance(reveal, Mapping) else None
    if not isinstance(block, Mapping):
        return None
    through, digest = block.get("through"), block.get("published_digest")
    if type(through) is not int or not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        return None
    return through, digest


def reveal_index(reveal) -> int | None:
    try:
        return memo_v4.epoch_index(reveal["epoch_id"])
    except (KeyError, TypeError, ValueError):
        return None


def fetch_snapshot(digest: str, *, through: int | None, bucket: str | None, netuid: int,
                   mirrors=None, pool_ids=None, opener=None,
                   timeout: float | None = None) -> memo_v4.Published | None:
    """The published set whose snapshot hashes to `digest`, from the root's
    bucket or a mirror (reveal_sync.fetch_object: the first source whose bytes
    hash to it), read strictly (`memo_v4.published_from_snapshot`); None when
    no source serves such bytes or they do not read as the view named."""
    from fugal_subnet import reveal_sync

    kw = {} if timeout is None else {"timeout": timeout}
    raw = reveal_sync.fetch_object(
        snapshot_object(netuid, digest), bucket=bucket, mirrors=mirrors,
        limit=memo_v4.MAX_SNAPSHOT_BYTES, opener=opener,
        check=lambda b: hashlib.sha256(b).hexdigest() == digest, **kw)
    if raw is None:
        return None
    return read_snapshot(raw, digest, through=through, pool_ids=pool_ids)


def read_snapshot(raw, digest: str, *, through: int | None, pool_ids=None):
    """`memo_v4.published_from_snapshot`, logged and None on refusal."""
    try:
        return memo_v4.published_from_snapshot(raw, digest, pool_ids=pool_ids, through=through)
    except ValueError as e:
        logger.error("ALARM: the published-set snapshot %s is refused: %s", str(digest)[:16], e)
        return None



def check_named(published: memo_v4.Published, reveal) -> str | None:
    """A finding when `published` disagrees with the view `reveal` names;
    None when they agree, or when either cannot say (no named view, or the
    set is not folded that far)."""
    named = named_view(reveal)
    if named is None or published.through < named[0]:
        return None
    mine = published.view(named[0]).digest
    if mine == named[1]:
        return None
    return (f"the root's reveal {reveal.get('epoch_id')} names the published set through "
            f"{epoch_id_for_block(named[0])} as {named[1][:16]}, this node's fold gives "
            f"{mine[:16]}")



# --- the root's publication ---------------------------------------------------------


class SnapshotPublisher:
    """The `snapshot` hook of reveal_publisher.publish_due_reveals, for the
    root: for a reveal that names a view of the published set (its
    `memo_v4`), the canonical bytes of that view of the set this node holds
    (`held()`), checked to hash to the digest the reveal names, put in the
    store under `snapshot_object`, create-only, and read back. Returns
    whether the store serves them; never raises (readers fold from an older
    snapshot when one is missing)."""

    def __init__(self, *, netuid: int, store, held: Callable[[], memo_v4.Published | None],
                 budget_s: float = PUBLISH_BUDGET_S, clock: Callable[[], float] = time.monotonic):
        self.netuid, self.store, self.held = int(netuid), store, held
        self.budget_s, self.clock = float(budget_s), clock

    def __call__(self, epoch_id: str, reveal: Mapping) -> bool:
        try:
            return self._publish(str(epoch_id), reveal)
        except Exception as e:  # noqa: BLE001 - a missing snapshot only lengthens a reader's fold
            logger.warning("The published-set snapshot of reveal %s was not published (%s: %s)",
                           epoch_id, type(e).__name__, e)
            return False

    def _publish(self, epoch_id: str, reveal: Mapping) -> bool:
        named = named_view(reveal)
        if named is None:
            return False
        through, digest = named
        held = self.held()
        if held is None or held.through < through:
            logger.warning("The published-set snapshot of reveal %s was not published: this node "
                           "does not hold the set through %s", epoch_id, epoch_id_for_block(through))
            return False
        raw = memo_v4.snapshot_bytes(held.view(through))
        if hashlib.sha256(raw).hexdigest() != digest:
            logger.error("ALARM: reveal %s names the published set through %s as %s, which is not "
                         "the set this node holds; its snapshot is not published", epoch_id,
                         epoch_id_for_block(through), digest[:16])
            return False
        key = snapshot_object(self.netuid, digest)
        deadline = self.clock() + self.budget_s
        self.store.put(key, raw)
        if self.clock() > deadline:
            raise TimeoutError("the publication budget ran out while uploading")
        if self.store.get(key, len(raw) + 1) != raw:
            raise ValueError(f"the store did not serve back what was put under {key}")
        logger.info("Published-set snapshot through %s published (%s, %d bytes)",
                    epoch_id_for_block(through), digest[:16], len(raw))
        return True


# --- one node's set ----------------------------------------------------------------------


def _read_json(path: str, limit: int):
    with open(path, "rb") as f:
        raw = f.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"larger than {limit} bytes")
    return raw


class ChainSet:
    """One node's published set across epochs, advanced by the root's
    commitments read at pinned boundary blocks (the module's rule).

    - `read_at(b)`: the root's reveal commitment read pinned at boundary b's
      block, or None when there is none; raises when the chain cannot be
      read (the caller's epoch fails and is retried, as for the menu);
    - `reveal_bytes(epoch_id, sha256)`: bytes offered for that root reveal
      (this node's own epoch directory, a follower's reference reveals, the
      root's store), or None; checked here against the sha256;
    - `snapshot(digest)`: bytes offered for that snapshot, or None; read
      strictly here (`memo_v4.published_from_snapshot`);
    - `path`: where the state is kept (HELD_FORMAT), or None;
    - `refuse`: a TD refuses to go on (PublishedUnavailable) past a reveal
      whose named view contradicts its own state; a validator logs an ALARM
      and goes on (the root's set is the reference);
    - `other_pool_ids`: question ids of another pool the root's reveals
      publish outcomes on. From protocol v5 a reveal's `votes_v5` names v5
      pool questions (memo_v4.published_ids), which the set folds like any
      other, so a kept state or a snapshot names them too; a node with v5
      scheduled passes the v5 pool's ids. Only `pool` seeds the set, and an
      id in neither is still refused.

    State: `held` (memo_v4.Published, folded through `at` - READ_LAG),
    `named` (the newest root reveal folded), `at` (the last boundary read)."""

    def __init__(self, *, pool, read_at: Callable[[int], str | None],
                 reveal_bytes: Callable[[str, str], bytes | None],
                 snapshot: Callable[[str], bytes | None] | None = None, path=None,
                 refuse: bool = False, seed_doc: Mapping | None = None,
                 other_pool_ids=()):
        self.pool = pool
        # What a kept state or a snapshot may name (`_load`, `_anchor`).
        self.pool_ids = (frozenset(q["question_id"] for q in pool)
                         | frozenset(str(q) for q in other_pool_ids))
        self.read_at, self.reveal_bytes, self.snapshot = read_at, reveal_bytes, snapshot
        self.path = None if path is None else os.fspath(path)
        self.refuse, self.seed_doc = bool(refuse), seed_doc
        self.findings: list[str] = []
        self.held: memo_v4.Published | None = None
        self.named = self.at = None
        self._load()

    # The file ---------------------------------------------------------------------

    def _load(self) -> None:
        if self.path is None or not os.path.exists(self.path):
            return
        try:
            try:
                doc = json.loads(_read_json(self.path, memo_v4.MAX_SNAPSHOT_BYTES + 4096))
            except RecursionError:
                raise ValueError("nested too deeply to be the state") from None
            if (not isinstance(doc, dict) or set(doc) != {"format", "set", "named", "at"}
                    or doc["format"] != HELD_FORMAT or type(doc["named"]) is not int
                    or type(doc["at"]) is not int):
                raise ValueError(f"not a {HELD_FORMAT} document")
            held = memo_v4.published_from_doc(doc["set"])
            if held.through != doc["at"] - memo_v4.READ_LAG:
                raise ValueError("the set is not folded through its boundary")
            if any(q not in self.pool_ids for q in held.first):
                raise ValueError("it names questions outside this pool")
        except (OSError, ValueError) as e:
            logger.error("The kept published set %s is set aside (%s); it is brought forward "
                         "again from the chain", self.path, e)
            return
        self.held, self.named, self.at = held, doc["named"], doc["at"]

    def save(self) -> None:
        """Write the state atomically (nothing when none is held)."""
        if self.held is None or self.path is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        doc = {"format": HELD_FORMAT, "set": self.held.to_doc(), "named": self.named,
               "at": self.at}
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    # Reading the chain ------------------------------------------------------------------

    def _commitment(self, b: int) -> tuple[int, str] | None:
        from fugal_subnet import reveal_sync

        parsed = reveal_sync.parse_commitment(self.read_at(int(b)))
        if parsed is None:
            return None
        return memo_v4.epoch_index(parsed[0]), parsed[1]

    def _reveal(self, c: int, sha: str) -> dict:
        eid = epoch_id_for_block(c)
        raw = self.reveal_bytes(eid, sha)
        if not isinstance(raw, (bytes, bytearray)) or hashlib.sha256(bytes(raw)).hexdigest() != sha:
            raise PublishedUnavailable(f"the root's reveal {eid}, which the chain names, is not "
                                       "to be had")
        try:
            doc = json.loads(bytes(raw))
        except (ValueError, RecursionError):
            doc = None
        if not isinstance(doc, dict) or doc.get("epoch_id") != eid:
            raise PublishedUnavailable(f"the bytes committed for {eid} are not its reveal")
        return doc

    def _check(self, reveal: dict) -> None:
        finding = check_named(self.held, reveal)
        if finding is None:
            return
        self.findings.append(finding)
        logger.error("ALARM: %s", finding)
        if self.refuse:
            raise PublishedUnavailable(finding)

    def _step(self, b: int) -> None:
        """Boundary b: fold the reveal its pinned read newly names, or nothing."""
        got = self._commitment(b)
        ids: set[str] = set()
        named = self.named
        if got is not None and got[0] > self.named:
            reveal = self._reveal(*got)
            self._check(reveal)
            ids, named = memo_v4.published_ids(reveal), got[0]
        self.held = memo_v4.fold(self.held, b - memo_v4.READ_LAG, ids)
        self.named, self.at = named, b

    def _anchor(self, epoch: int, after: int | None = None) -> bool:
        """Anchor on the snapshot the reveal the current pinned read names
        names (then its boundary's named reveal is read), when that reveal
        is after boundary `after`: True when it did. A failed chain read
        raises as it is."""
        got = self._commitment(epoch) if self.snapshot is not None else None
        if got is None or (after is not None and got[0] <= after):
            return False
        try:
            reveal = self._reveal(*got)
            named = named_view(reveal)
            raw = self.snapshot(named[1]) if named is not None else None
            if raw is None:
                return False
            c = got[0]
            held = memo_v4.published_from_snapshot(raw, named[1], pool_ids=self.pool_ids,
                                                   through=c - memo_v4.READ_LAG)
            before = self._commitment(c)
        except (PublishedUnavailable, ValueError) as e:
            logger.warning("Published set: no snapshot to anchor on (%s)", e)
            return False
        self.held, self.at = held, c
        self.named = before[0] if before is not None else memo_v4.seed_through(self.seed_doc)
        logger.info("Published set: anchored on the snapshot the root's reveal %s names",
                    epoch_id_for_block(c))
        return True

    def _start(self, epoch: int) -> None:
        """No state, or too far behind to replay: the snapshot the reveal the
        current pinned read names names (`_anchor`), else the pinned seed
        (then every boundary after it)."""
        if self._anchor(epoch):
            return
        if self.snapshot is not None:
            logger.warning("Published set: starting from the pinned seed")
        seed = memo_v4.seed_published(self.pool, self.seed_doc)
        self.held, self.named, self.at = seed, seed.through, seed.through + memo_v4.READ_LAG

    def advance_to(self, epoch: int) -> None:
        """Read every boundary from the last one read through `epoch`'s.
        Raises PublishedUnavailable when a reveal the chain names cannot be
        had (or, on a TD, contradicts its state); a failed chain read raises
        as it is, except one the endpoint says it can never serve
        (`is_pruned`, a block whose state it no longer keeps): then the node
        re-anchors once on the snapshot the current commitment names
        (`_anchor`, which reads ANCHOR_HISTORY_EPOCHS back) and goes on from
        there, or, when it cannot, raises PublishedUnavailable saying the
        endpoint does not serve the history this needs (an archive endpoint
        does). The next call starts from whatever was reached."""
        epoch = int(epoch)
        if self.at is not None and self.at >= epoch:
            return
        if self.held is None or epoch - self.at > MAX_FOLD_EPOCHS:
            self._start(epoch)
        if epoch - self.at > MAX_FOLD_EPOCHS:
            raise PublishedUnavailable(f"{epoch - self.at} boundaries to replay since "
                                       f"{epoch_id_for_block(self.at)}, more than {MAX_FOLD_EPOCHS}")
        reanchored = False
        while self.at < epoch:
            b = self.at + 1
            try:
                self._step(b)
            except PublishedUnavailable:
                raise
            except Exception as e:
                if not is_pruned(e):
                    raise
                if reanchored:
                    raise self._history_missing(b, e) from e
                reanchored = True
                logger.warning("Published set: the chain endpoint no longer serves boundary %s "
                               "(%s); re-anchoring on the root's newest snapshot",
                               epoch_id_for_block(b), type(e).__name__)
                try:
                    anchored = self._anchor(epoch, after=self.at)
                except Exception as e2:  # noqa: BLE001 - classified below
                    if not is_pruned(e2):
                        raise
                    anchored = False
                if not anchored:
                    raise self._history_missing(b, e) from e

    @staticmethod
    def _history_missing(b: int, e: BaseException) -> PublishedUnavailable:
        return PublishedUnavailable(
            f"the chain endpoint does not serve the pinned read of boundary "
            f"{epoch_id_for_block(b)} ({type(e).__name__}) and no snapshot within "
            f"{ANCHOR_HISTORY_EPOCHS} epochs of history could be anchored on: a v4 node needs an "
            "endpoint that keeps that much state, or an archive endpoint "
            "(FUGAL_ARCHIVE_ENDPOINTS)")

    def view(self, epoch: int) -> memo_v4.Published:
        """The set as of epoch E (its boundary read): what its plan draws on
        and its verdicts read, folded through E - 1 with nothing at E - 1 (so
        `plan_v4` and `memo_v4.epoch_memo` take it as it is). The same
        whatever the state has read since."""
        epoch = int(epoch)
        if self.at is None or self.at < epoch:
            raise PublishedUnavailable(f"the published set is read only through boundary "
                                       f"{self.at}, not {epoch}")
        return memo_v4.fold(self.held.view(epoch - memo_v4.READ_LAG), epoch - 1, ())

    def view_for(self, epoch: int) -> memo_v4.Published:
        """`advance_to` then `view`: what a TD's session reads each epoch."""
        self.advance_to(epoch)
        return self.view(epoch)


# How long after the seed's `through` the startup probe still reads back to it
# (the release gate's PUBLISHED_V4_SEED_MAX_AGE_EPOCHS plus a margin): after
# that a node holds the set in its state and re-anchors ANCHOR_HISTORY_EPOCHS back.
_seed_history_epochs = 200


def history_line(subtensor, netuid: int, blocks_per_epoch: int) -> str:
    """Whether this node's chain reads reach the history a v4 node needs, for
    the startup and `--check` output: one pinned read of the commitments at
    the older of the pinned seed's boundary (what the published set is read
    from across the activation) and ANCHOR_HISTORY_EPOCHS back (a re-anchor
    after missed epochs), through the same reader the node uses, so through
    its archive fallback (chain_reads). Never raises."""
    from fugal_subnet.commitments import get_commitments_with_blocks

    try:
        current = int(subtensor.get_current_block())
        block = current - ANCHOR_HISTORY_EPOCHS * int(blocks_per_epoch)
        try:
            seed_block = memo_v4.seed_through() * int(blocks_per_epoch)
            if current - seed_block <= _seed_history_epochs * int(blocks_per_epoch):
                block = min(block, seed_block)          # the activation is near: the seed's depth
        except (OSError, ValueError, KeyError, TypeError):
            pass
        block = max(0, block)
        get_commitments_with_blocks(subtensor, netuid, block=block)
        return f"chain history: pinned reads {current - block} blocks back are served"
    except Exception as e:  # noqa: BLE001 - reported, never fatal
        why = "the endpoint no longer keeps that state" if is_pruned(e) else type(e).__name__
        return (f"chain history: a pinned read back to the published-set seed FAILED ({why}); "
                "at the v4 activation this node cannot read the published set: keep an archive "
                "endpoint in FUGAL_ARCHIVE_ENDPOINTS (finney's public archive is the default)")
