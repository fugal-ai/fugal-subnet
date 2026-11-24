"""Protocol v5 on one validator: the state v5_epoch's pure functions read,
kept on disk between epochs, and the root's published exam spec.

`V5Node` keeps the seat state, the head book, the window's records, the exam
specs this validator built for the epochs ahead, and the bytes of the heads
it verified (memory only: a head's picks are recomputed from its bytes). The
root's spec for an epoch is the one this validator compares its own with
(`exam_for`): the root reads its own; a follower reads the root's reveal of
E - SPEC_LAG as the TD does (`published_spec_reader`).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable

from fugal_subnet import seat_v5, v5_epoch
from fugal_subnet.exam_v5 import ExamSpec

logger = logging.getLogger(__name__)

SpecAndCosts = tuple[ExamSpec | None, dict]


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
        if path and os.path.exists(path):
            self._load()

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

    def save(self) -> None:
        if not self.path:
            return
        d = {
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
        self.head_bytes = {h: b for h, b in self.head_bytes.items() if h in self.heads}
        self.save()


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
