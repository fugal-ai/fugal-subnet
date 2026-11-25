"""Protocol v5 on one validator: the state v5_epoch's pure functions read,
kept on disk between epochs, and the root's published exam spec.

`V5Node` keeps the seat state, the head book, the window's records, the exam
specs this validator built for the epochs ahead, and the bytes of the heads
in its book (a head's picks are recomputed from its bytes, so they are kept
on disk beside the state, `<state>.heads/<sha256>.npz`, each checked against
its name when read back: a restart still gives the holder and the contenders
their picks). The root's spec for an epoch is the one this validator compares
its own with (`exam_for`): the root reads its own; a follower reads the
root's reveal of E - SPEC_LAG as the TD does (`published_spec_reader`).

RESYNC. A node whose state is missing or behind (`stale`) rebuilds it from
the root's reveals alone (`resync`: v5_epoch.rebuild over the window, and the
specs ahead from the reveals that publish them), so a fresh or long-stopped
follower holds the root's seat, book, window and specs before it scores.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Mapping

from fugal_subnet import seat_v5, v5_epoch
from fugal_subnet.exam_v5 import ExamSpec

logger = logging.getLogger(__name__)

SpecAndCosts = tuple[ExamSpec | None, dict]
_SHA = re.compile(r"^[0-9a-f]{64}$")
# The most bytes a kept head file may be (a v4/v5 head is well under 1 MB).
MAX_HEAD_BYTES = 8 * 1024 * 1024


class V5Node:
    def __init__(self, path: str | None, *, root: bool,
                 read_published: Callable[[int], SpecAndCosts] | None = None):
        self.path = path
        self.root = bool(root)
        self.read_published = read_published
        self.seat = seat_v5.SeatStateV5()
        self.heads: dict[str, v5_epoch.HeadFacts] = {}
        self.records: list[v5_epoch.EpochRecord] = []
        self.specs: dict[int, SpecAndCosts] = {}
        self.head_bytes: dict[str, bytes] = {}
        self.through: int | None = None     # the last epoch committed
        if path and os.path.exists(path):
            self._load()
        self._load_heads()

    # --- persistence ----------------------------------------------------------------------

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            d = json.load(f)
        self.seat = seat_v5.SeatStateV5.from_reveal(d.get("seat"))
        self.heads = v5_epoch.heads_from_reveal(d.get("heads"))
        self.specs = {int(e): (ExamSpec.from_json(s["spec"]), {k: int(v) for k, v in
                                                                s["costs"].items()})
                      for e, s in (d.get("specs") or {}).items()}
        recs = []
        for r in d.get("records") or ():
            spec = ExamSpec.from_json(r["spec"]) if r.get("spec") else None
            recs.append(v5_epoch.EpochRecord.from_reveal(r["record"], spec, r.get("costs") or {}))
        self.records = recs
        t = d.get("through")
        self.through = (int(t) if type(t) is int else
                        max((r.epoch for r in recs), default=None))

    # --- head bytes -----------------------------------------------------------------------

    @property
    def heads_dir(self) -> str | None:
        return self.path + ".heads" if self.path else None

    def keep_head(self, head: str, data: bytes) -> None:
        """Keep a verified head's bytes (memory, and on disk when the node has
        a path); `head` must be their sha256."""
        if not _SHA.match(str(head)) or hashlib.sha256(bytes(data)).hexdigest() != head:
            raise ValueError("a kept head's name is the sha256 of its bytes")
        self.head_bytes[head] = bytes(data)
        folder = self.heads_dir
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{head}.npz")
        if os.path.exists(path):
            return
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".head.")
        with os.fdopen(fd, "wb") as f:
            f.write(bytes(data))
        os.replace(tmp, path)

    def _load_heads(self) -> None:
        """Read back the kept bytes of the heads in the book: a file whose
        bytes do not hash to its name, or too big, is ignored."""
        folder = self.heads_dir
        if not folder or not os.path.isdir(folder):
            return
        for h in sorted(self.heads):
            if h in self.head_bytes or not _SHA.match(h):
                continue
            path = os.path.join(folder, f"{h}.npz")
            try:
                with open(path, "rb") as f:
                    data = f.read(MAX_HEAD_BYTES + 1)
            except OSError:
                continue
            if len(data) <= MAX_HEAD_BYTES and hashlib.sha256(data).hexdigest() == h:
                self.head_bytes[h] = data

    def _prune_heads(self) -> None:
        folder = self.heads_dir
        self.head_bytes = {h: b for h, b in self.head_bytes.items() if h in self.heads}
        if not folder or not os.path.isdir(folder):
            return
        for name in os.listdir(folder):
            h = name[:-4] if name.endswith(".npz") else None
            if h and _SHA.match(h) and h not in self.heads:
                try:
                    os.remove(os.path.join(folder, name))
                except OSError:
                    pass

    def save(self) -> None:
        if not self.path:
            return
        d = {
            "through": self.through,
            "seat": self.seat.to_reveal(),
            "heads": {h: f.to_reveal() for h, f in sorted(self.heads.items())},
            "specs": {str(e): {"spec": s.to_json(), "costs": dict(sorted(c.items()))}
                      for e, (s, c) in sorted(self.specs.items()) if s is not None},
            "records": [{"record": r.to_reveal(),
                         "spec": r.spec.to_json() if r.spec is not None else None,
                         "costs": dict(sorted(r.costs_nd.items()))} for r in self.records],
        }
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".v5_state.")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(d, f, sort_keys=True, separators=(",", ":"))
        os.replace(tmp, self.path)

    # --- the epoch's exam -----------------------------------------------------------------

    def exam_for(self, epoch: int) -> tuple[ExamSpec | None, dict, str | None]:
        """(the spec the TDs answer in `epoch`, its costs, why the exam does
        not count here or None). The TDs answer the root's published spec;
        its items count only when this validator built the same spec."""
        own, own_costs = self.specs.get(int(epoch), (None, {}))
        if self.root:
            return own, own_costs, None if own is not None else "no exam spec for this epoch"
        published, costs = None, {}
        if self.read_published is not None:
            try:
                published, costs = self.read_published(int(epoch))
            except Exception as e:  # noqa: BLE001 - the exam is void here; logged
                logger.warning("Epoch %d: the root's exam spec could not be read (%s: %s)",
                               epoch, type(e).__name__, e)
        return published, costs, v5_epoch.spec_agrees(own, published)

    def commit(self, epoch: int, result: v5_epoch.V5Result, record: v5_epoch.EpochRecord,
               next_spec: ExamSpec | None, next_costs: dict) -> None:
        """Advance to after `epoch` and save."""
        self.seat = result.decision.state
        self.heads = dict(result.heads)
        self.records = v5_epoch.window_of(
            [*[r for r in self.records if r.epoch != record.epoch], record], epoch)
        if next_spec is not None:
            self.specs[int(next_spec.epoch)] = (next_spec, dict(next_costs))
        self.specs = {e: s for e, s in self.specs.items() if e > int(epoch)}
        self.through = int(epoch)
        self._prune_heads()
        self.save()

    # --- resync ---------------------------------------------------------------------------

    def stale(self, epoch: int, first_epoch: int) -> bool:
        """Whether this node's state is not the state after epoch - 1: it
        holds none, or an older one, while epoch - 1 is a v5 epoch."""
        prev = int(epoch) - 1
        if prev < int(first_epoch):
            return False
        return self.through is None or self.through < prev

    def resync(self, reveals: Mapping[int, Mapping], epoch: int) -> bool:
        """Rebuild the state after epoch - 1 from the root's reveals
        ({epoch index: reveal}, covering the window, the SPEC_LAG epochs
        before it and the specs ahead): v5_epoch.rebuild, and the specs of
        epochs epoch .. epoch - 1 + SPEC_LAG from the reveals that publish
        them. False (state unchanged) when the reveal of epoch - 1 is not
        among them. Kept head bytes stay; heads not in the rebuilt book go."""
        prev = int(epoch) - 1
        last = reveals.get(prev)
        if not last or v5_epoch.SEAT_KEY not in last:
            return False
        seat, heads, records = v5_epoch.rebuild(reveals, prev)
        specs = {}
        for e in range(int(epoch), prev + v5_epoch.SPEC_LAG + 1):
            spec, costs = v5_epoch.spec_from_reveal(reveals.get(e - v5_epoch.SPEC_LAG) or {})
            if spec is not None and int(spec.epoch) == e:
                specs[e] = (spec, costs)
        self.seat, self.heads, self.records, self.specs = seat, heads, records, specs
        self.through = prev
        self._load_heads()
        self._prune_heads()
        self.save()
        logger.warning("v5 state rebuilt from the root's reveals through epoch %d: holder %s, "
                       "%d heads, %d window records, specs for %s", prev,
                       (seat.holder or "none")[:16], len(heads), len(records), sorted(specs))
        return True


def reference_reveals(read: Callable[[int], bytes | None], epoch: int) -> dict[int, dict]:
    """{epoch index: reveal} of the epochs a resync before `epoch` reads (the
    window, the SPEC_LAG epochs before it and the specs ahead), from
    `read(epoch index)` (the reveal's bytes, or None when not held). A
    reveal that is not JSON is left out."""
    out = {}
    lo = int(epoch) - 1 - v5_epoch.WINDOW_EPOCHS + 1 - v5_epoch.SPEC_LAG
    for e in range(lo, int(epoch)):
        raw = read(e)
        if raw is None:
            continue
        try:
            doc = json.loads(raw)
        except (ValueError, RecursionError):
            continue
        if isinstance(doc, dict):
            out[e] = doc
    return out


def published_spec_reader(subtensor, *, netuid: int, root_hotkey, blocks_per_epoch: int,
                          fetch_reveal: Callable[[str, str], bytes | None]
                          ) -> Callable[[int], SpecAndCosts]:
    """A follower's reader of the root's spec for epoch E: the reveal the
    root's commitment (read pinned at E's boundary) names must be the reveal
    of E - SPEC_LAG, its bytes the committed sha256 (td_v5.exam_spec_from_
    reveal); its spec and that spec's costs."""
    from fugal_subnet import menu_publish as mp
    from fugal_subnet import reveal_sync
    from fugal_subnet.commitments import get_commitments_with_blocks
    from fugal_subnet.memo_v4 import epoch_index
    from fugal_subnet.tee import td_v5

    def read(epoch: int) -> SpecAndCosts:
        hotkey = mp.hotkey_for_epoch(root_hotkey, epoch)
        data, _b = get_commitments_with_blocks(
            subtensor, netuid, block=int(epoch) * blocks_per_epoch).get(hotkey, ("", 0))
        got = reveal_sync.parse_commitment(data)
        if got is None:
            return None, {}
        epoch_id, sha, _menu = got
        if epoch_index(epoch_id) != int(epoch) - v5_epoch.SPEC_LAG:
            return None, {}
        raw = fetch_reveal(epoch_id, sha)
        if raw is None:
            raise LookupError(f"the root's reveal {epoch_id} is not to be had")
        spec = td_v5.exam_spec_from_reveal(raw, epoch_id, sha)
        if spec is None:
            return None, {}
        _s, costs = v5_epoch.spec_from_reveal(json.loads(raw))
        return spec, costs
    return read
