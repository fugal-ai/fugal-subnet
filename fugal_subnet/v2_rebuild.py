"""Rebuild protocol v2 state from published reveals: snapshot + replay.

A validator that loses its state, or has just joined, must reach the state
its peers hold or it scores the field differently (I1). The v2 decision
state cannot be replayed from reveals alone, because some of what decided it
(dedup verdicts, per-head evidence, chain commitments, seniority) is not in
them. So it is rebuilt in two parts:

  SNAPSHOT  the seat, head owners, the UID chosen for each head, each head's
            burn-in progress, and any later v2 state, read from the `v2_state`
            block of the NEWEST v2 reveal (v2_epoch.v2_state_for_reveal);
  REPLAY    the comparison window, re-derived from the last
            CHAMPION_WINDOW_EPOCHS v2 reveals: the slice, the shared cells with
            their tokens, and each head's routing and failed calls, at the
            pinned prices.

THE TRUST MODEL, stated plainly. The snapshot is the publishing validator's
word: a validator that rebuilds trusts it, as it trusts a frame rebuilt from
someone else's reveals. That trust is bounded. Every reveal's sha256 is
committed on chain by the validator that published it, so a rebuilt snapshot
is pinned to a specific, attributable, unalterable artifact; and the window
part is re-derived, not trusted, from the cells and routing every reveal
carries. Rebuilding from your own reveals after a disk loss is self-trust.

Deterministic: reveals are ordered by epoch id, never by filesystem order.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from fugal_subnet.champion import HeadEpoch, SeatState, Window
from fugal_subnet.v2_epoch import build_epoch_slice, seat_from_dict

logger = logging.getLogger(__name__)

_SNAPSHOT_KEYS = ("seat", "owners", "serving", "burn_in")


@dataclass
class V2Rebuild:
    window: Window
    seat: SeatState
    owners: dict[str, str]
    burn_in: dict[str, float]
    serving: dict[str, int]
    # Later v2 state published in the same block (the copy rule's), verbatim.
    extra: dict[str, object] = field(default_factory=dict)
    epochs_replayed: int = 0
    newest_epoch_id: str | None = None
    # The epoch whose v2_state block is the snapshot (the newest unless asked).
    snapshot_epoch_id: str | None = None


def _epoch_index(epoch_id: str) -> int:
    # slicer.epoch_id_for_block is the only formatter: "e%08d".
    text = str(epoch_id)
    if not (text.startswith("e") and text[1:].isdigit()):
        raise ValueError(f"epoch_id {text!r} is not of the form e<digits>")
    return int(text[1:])


def _require(cond: bool, epoch_id: str, what: str) -> None:
    """A published reveal is someone else's file (I2): a field that is not the
    shape the live epoch wrote is refused by name, never guessed at."""
    if not cond:
        raise ValueError(f"reveal {epoch_id}: {what}")


def _load(source: str | os.PathLike | Iterable[str]) -> list[dict]:
    if isinstance(source, (str, os.PathLike)):
        paths = sorted(glob.glob(os.path.join(str(source), "*", "reveal.json")))
    else:
        paths = sorted(source)
    out = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                reveal = json.load(f)
        except Exception as e:  # noqa: BLE001 - one bad file must not lose the rest
            logger.warning("Skipping unreadable reveal %s: %s", path, e)
            continue
        if not isinstance(reveal, Mapping):
            logger.warning("Skipping reveal %s: not a JSON object", path)
            continue
        if int(reveal.get("protocol_version", 1)) >= 2 and reveal.get("epoch_id"):
            _epoch_index(reveal["epoch_id"])        # malformed id: refuse by name, below
            out.append(dict(reveal, _path=str(path)))
    return out


def epoch_from_reveal(
    reveal: Mapping,
    prices: Mapping[str, tuple[float, float]],
    bench_category: Mapping[str, str] | None = None,
):
    """(EpochSlice, {head hash: HeadEpoch}) for one v2 reveal, exactly as the
    live epoch built them."""
    eid = str(reveal.get("epoch_id", "?"))
    epoch = _epoch_index(eid)
    models, questions, grades = reveal.get("models"), reveal.get("questions"), reveal.get("shared_grades")
    _require(isinstance(models, list) and all(isinstance(m, str) for m in models),
             eid, "`models` must be a list of model ids")
    _require(isinstance(questions, list) and all(isinstance(q, Mapping) and "question_id" in q
                                                 for q in questions),
             eid, "`questions` must be a list of question objects")
    _require(isinstance(grades, list) and all(isinstance(c, Mapping) for c in grades),
             eid, "`shared_grades` must be a list of cells")
    try:
        cells = {(str(c["question_id"]), str(c["model"])): float(c["grade"]) for c in grades}
        tokens = {(str(c["question_id"]), str(c["model"])): float(c.get("completion_tokens", 0.0))
                  for c in grades}
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"reveal {eid}: a shared_grades cell is malformed ({e})") from None
    sl = build_epoch_slice(epoch, list(questions), list(models), cells, tokens, prices,
                           bench_category)

    # routing_decisions are per uid, in the global model index space, ORDERED
    # BY QUESTION ID (validator._extract_routing_decisions); realign them to
    # the slice's order.
    by_id = sorted(sl.question_ids)
    v2_state = reveal.get("v2_state") or {}
    serving = v2_state.get("serving") or {}
    failed = reveal.get("failed") or {}
    decisions = reveal.get("routing_decisions") or {}
    _require(isinstance(serving, Mapping) and isinstance(failed, Mapping)
             and isinstance(decisions, Mapping), eid, "serving, failed and routing_decisions must be maps")
    answers: dict[str, HeadEpoch] = {}
    for h in sorted(serving):
        uid = str(serving[h])
        if uid not in decisions:
            continue
        row = decisions[uid]
        _require(isinstance(row, list) and len(row) == len(by_id)
                 and all(isinstance(x, int) and not isinstance(x, bool) for x in row),
                 eid, f"routing_decisions[{uid}] must hold one model index per question")
        _require(all(0 <= x < len(models) for x in row),
                 eid, f"routing_decisions[{uid}] names a model index outside 0..{len(models) - 1}")
        routed_by_id = dict(zip(by_id, row))
        bad = set(failed.get(uid) or [])
        routed = np.array([routed_by_id[q] for q in sl.question_ids], dtype=np.int64)
        fail = np.array([q in bad for q in sl.question_ids], dtype=bool)
        answers[h] = HeadEpoch(routed=routed, failed=fail)
    return sl, answers


def rebuild_v2_from_reveals(
    source: str | os.PathLike | Iterable[str],
    *,
    window_size: int,
    prices: Mapping[str, tuple[float, float]],
    bench_category: Mapping[str, str] | None = None,
    snapshot_epoch_id: str | None = None,
) -> V2Rebuild:
    """Rebuild the v2 window and decision state from published v2 reveals.

    `snapshot_epoch_id` takes the snapshot (seat, owners, serving, burn-in,
    copy rule) from that epoch's reveal instead of the newest one, while the
    window still replays every reveal given (fugal_subnet/v2_sync.py re-anchors
    on a reference validator's state at an epoch, then keeps its own
    observations of the epochs since). Two reveals for one epoch are refused.
    """
    reveals = sorted(_load(source), key=lambda r: _epoch_index(r["epoch_id"]))
    ids = [r["epoch_id"] for r in reveals]
    if len(set(ids)) != len(ids):
        raise ValueError("two reveals for one epoch: pass one source per epoch")
    if not reveals:
        return V2Rebuild(Window.empty(window_size), SeatState(), {}, {}, {})
    newest = reveals[-1]
    if snapshot_epoch_id is not None:
        found = [r for r in reveals if r["epoch_id"] == snapshot_epoch_id]
        if not found:
            raise ValueError(f"no reveal for the snapshot epoch {snapshot_epoch_id}")
        snapshot = found[0]
    else:
        snapshot = newest
    oldest_kept = _epoch_index(newest["epoch_id"]) - int(window_size) + 1
    window = Window.empty(window_size)
    replayed = 0
    for r in reveals:
        if _epoch_index(r["epoch_id"]) < oldest_kept:
            continue
        try:
            sl, answers = epoch_from_reveal(r, prices, bench_category)
            window = window.add_epoch(sl, answers)
        except ValueError as e:
            # Fail closed, by name: a window missing one epoch is a different
            # window from every peer's (I1), so the operator fixes the file.
            raise ValueError(f"{r.get('_path', r.get('epoch_id'))}: {e}") from None
        replayed += 1
    state = dict(snapshot.get("v2_state") or {})
    rebuilt = V2Rebuild(
        window=window,
        seat=seat_from_dict(state.get("seat")),
        owners={str(k): str(v) for k, v in (state.get("owners") or {}).items()},
        burn_in={str(k): float(v) for k, v in (state.get("burn_in") or {}).items()},
        serving={str(k): int(v) for k, v in (state.get("serving") or {}).items()},
        extra={k: v for k, v in state.items() if k not in _SNAPSHOT_KEYS},
        epochs_replayed=replayed,
        newest_epoch_id=str(newest["epoch_id"]),
        snapshot_epoch_id=str(snapshot["epoch_id"]),
    )
    logger.info(
        "Rebuilt v2 state from %d reveals (window %d epochs, newest %s, champion %s)",
        len(reveals), replayed, rebuilt.newest_epoch_id,
        (rebuilt.seat.champion or "none")[:16],
    )
    return rebuilt
