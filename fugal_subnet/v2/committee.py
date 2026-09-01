"""Deterministic v2 matrix-builder committee selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

MAX_BUILDERS = 5
MIN_REPORTS = 3


class QuorumUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Builder:
    uid: int
    hotkey: str


def _boundary_bytes(boundary_hash: str) -> bytes:
    value = boundary_hash.removeprefix("0x")
    if len(value) != 64:
        raise ValueError("boundary_hash must be 32-byte hex")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("boundary_hash must be 32-byte hex") from exc


def select_builders(
    boundary_hash: str,
    hotkeys: Sequence[str],
    validator_permits: Sequence[bool],
    *,
    maximum_builders: int = MAX_BUILDERS,
    minimum_reports: int = MIN_REPORTS,
) -> tuple[Builder, ...]:
    if len(hotkeys) != len(validator_permits):
        raise ValueError("metagraph hotkeys and permits do not align")
    if not 1 <= maximum_builders <= MAX_BUILDERS:
        raise ValueError("maximum_builders is outside the v2 bound")
    if not 1 <= minimum_reports <= maximum_builders:
        raise ValueError("minimum_reports is invalid")
    boundary = _boundary_bytes(boundary_hash)
    eligible = []
    seen = set()
    for uid, (hotkey, permit) in enumerate(zip(hotkeys, validator_permits)):
        if type(permit) is not bool:
            raise ValueError("validator permits must be booleans")
        if not permit:
            continue
        if not isinstance(hotkey, str) or not hotkey or hotkey in seen:
            raise ValueError("eligible validator hotkeys are invalid or duplicated")
        seen.add(hotkey)
        rank = hashlib.sha256(
            b"fugal-builder-v2\x00" + boundary + b"\x00" + hotkey.encode("utf-8")
        ).digest()
        eligible.append((rank, hotkey, uid))
    selected = tuple(
        Builder(uid=uid, hotkey=hotkey)
        for _, hotkey, uid in sorted(eligible)[:maximum_builders]
    )
    if len(selected) < minimum_reports:
        raise QuorumUnavailable(
            f"only {len(selected)} permitted validators; v2 requires {minimum_reports}"
        )
    return selected
