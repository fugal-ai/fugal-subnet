"""V2 scoring state whose records are bound to both UID and hotkey."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

COMPOSITE_ACCURACY = 0.55
COMPOSITE_COST = 0.35
COMPOSITE_KL = 0.10
SCORE_ROUNDING_DECIMALS = 12


@dataclass(frozen=True)
class MinerRecord:
    uid: int
    hotkey: str
    epochs_seen: int = 0
    epochs_missed: int = 0
    current_head_hash: str = ""
    accuracy: float = 0.0
    cost_efficiency: float = 0.0
    kl_score: float = 0.0
    composite_score: float = 0.0


@dataclass(frozen=True)
class OwnershipReconciliation:
    records: dict[int, MinerRecord]
    new_uids: frozenset[int]
    reset_uids: frozenset[int]
    removed_uids: frozenset[int]


def composite_score(
    accuracy: float,
    cost_efficiency: float,
    kl_score: float,
    *,
    rounding_decimals: int = SCORE_ROUNDING_DECIMALS,
) -> float:
    """Compute the manifest-defined v2 score with bounded finite inputs."""
    values = (accuracy, cost_efficiency, kl_score)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
        raise ValueError("score components must be numeric")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("score components must be finite")
    if not 0 <= accuracy <= 1 or not 0 <= cost_efficiency <= 1:
        raise ValueError("accuracy and cost efficiency must be between zero and one")
    if not 0 <= rounding_decimals <= 15:
        raise ValueError("rounding_decimals must be between zero and fifteen")
    normalized_kl = 1.0 / (1.0 + math.exp(min(700.0, -float(kl_score) - 1.0)))
    result = (
        COMPOSITE_ACCURACY * float(accuracy)
        + COMPOSITE_COST * float(cost_efficiency)
        + COMPOSITE_KL * normalized_kl
    )
    return round(result, rounding_decimals)


def reconcile_uid_ownership(
    records: Mapping[int, MinerRecord],
    current_hotkeys: Mapping[int, str],
) -> OwnershipReconciliation:
    """Reset state when a metagraph UID is reassigned to another hotkey."""
    if any(not isinstance(uid, int) or isinstance(uid, bool) or uid < 0 for uid in current_hotkeys):
        raise ValueError("metagraph UIDs must be non-negative integers")
    hotkeys = list(current_hotkeys.values())
    if any(not isinstance(hotkey, str) or not hotkey for hotkey in hotkeys):
        raise ValueError("metagraph hotkeys must be non-empty strings")
    if len(hotkeys) != len(set(hotkeys)):
        raise ValueError("metagraph hotkeys must be unique")

    reconciled: dict[int, MinerRecord] = {}
    new_uids: set[int] = set()
    reset_uids: set[int] = set()
    for uid in sorted(current_hotkeys):
        hotkey = current_hotkeys[uid]
        previous = records.get(uid)
        if previous is None:
            new_uids.add(uid)
            reconciled[uid] = MinerRecord(uid=uid, hotkey=hotkey)
        elif previous.hotkey != hotkey:
            reset_uids.add(uid)
            reconciled[uid] = MinerRecord(uid=uid, hotkey=hotkey)
        else:
            # Normalize the uid field too; malformed persisted state must not
            # let one UID carry another UID's history.
            reconciled[uid] = replace(previous, uid=uid)

    removed = frozenset(set(records) - set(current_hotkeys))
    return OwnershipReconciliation(
        records=reconciled,
        new_uids=frozenset(new_uids),
        reset_uids=frozenset(reset_uids),
        removed_uids=removed,
    )
