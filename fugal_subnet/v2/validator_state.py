"""Atomic UID/hotkey-bound validator liveness state for v2."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

from fugal_subnet.consensus_manifest import canonical_json
from fugal_subnet.v2.scoring import MinerRecord, reconcile_uid_ownership

SCHEMA_VERSION = 1
LIVENESS_MISS_LIMIT = 3


class ValidatorStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class LivenessSnapshot:
    records: dict[int, MinerRecord]
    eligible_uids: frozenset[int]
    forced_zero_uids: frozenset[int]
    reset_uids: frozenset[int]


class ValidatorStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _decode(self) -> tuple[str | None, dict[int, MinerRecord]]:
        if not self.path.exists():
            return None, {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidatorStateError("v2 validator state is unreadable") from exc
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version", "last_epoch_id", "records"
        }:
            raise ValidatorStateError("v2 validator state schema differs")
        if raw["schema_version"] != SCHEMA_VERSION or not isinstance(raw["records"], list):
            raise ValidatorStateError("v2 validator state version differs")
        last_epoch_id = raw["last_epoch_id"]
        if last_epoch_id is not None and (
            not isinstance(last_epoch_id, str) or not last_epoch_id
        ):
            raise ValidatorStateError("v2 validator last epoch is invalid")
        records = {}
        expected = set(MinerRecord.__dataclass_fields__)
        for item in raw["records"]:
            if not isinstance(item, dict) or set(item) != expected:
                raise ValidatorStateError("v2 miner state record schema differs")
            try:
                record = MinerRecord(**item)
            except (TypeError, ValueError) as exc:
                raise ValidatorStateError("v2 miner state record is invalid") from exc
            if record.uid in records:
                raise ValidatorStateError("v2 miner state UIDs are duplicated")
            records[record.uid] = record
        return last_epoch_id, records

    def _write(self, epoch_id: str, records: Mapping[int, MinerRecord]) -> None:
        payload = canonical_json({
            "schema_version": SCHEMA_VERSION,
            "last_epoch_id": epoch_id,
            "records": [asdict(records[uid]) for uid in sorted(records)],
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        temporary.replace(self.path)
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _validate_update(
        epoch_id: str,
        current_hotkeys: Mapping[int, str],
        responding_uids: set[int],
        head_hashes: Mapping[int, str],
    ) -> None:
        if not isinstance(epoch_id, str) or not epoch_id:
            raise ValidatorStateError("epoch_id is invalid")
        if not set(responding_uids) <= set(current_hotkeys):
            raise ValidatorStateError("responding UIDs are outside the metagraph")
        if not set(head_hashes) <= set(responding_uids):
            raise ValidatorStateError("head hashes are outside responding UIDs")

    @staticmethod
    def _derive(
        epoch_id: str,
        last_epoch_id: str | None,
        persisted: Mapping[int, MinerRecord],
        current_hotkeys: Mapping[int, str],
        responding_uids: set[int],
        head_hashes: Mapping[int, str],
    ) -> LivenessSnapshot:
        ownership = reconcile_uid_ownership(persisted, current_hotkeys)
        records = {}
        for uid, record in ownership.records.items():
            if last_epoch_id == epoch_id:
                records[uid] = record
            elif uid in responding_uids:
                records[uid] = replace(
                    record,
                    epochs_seen=record.epochs_seen + 1,
                    epochs_missed=0,
                    current_head_hash=head_hashes.get(uid, record.current_head_hash),
                )
            else:
                records[uid] = replace(
                    record,
                    epochs_missed=record.epochs_missed + 1,
                )
        forced = frozenset(
            uid for uid, record in records.items()
            if record.epochs_missed >= LIVENESS_MISS_LIMIT
        )
        return LivenessSnapshot(
            records=records,
            eligible_uids=frozenset(set(records) - set(forced)),
            forced_zero_uids=forced,
            reset_uids=ownership.reset_uids,
        )

    def preview_epoch(
        self,
        epoch_id: str,
        current_hotkeys: Mapping[int, str],
        responding_uids: set[int],
        head_hashes: Mapping[int, str] | None = None,
    ) -> LivenessSnapshot:
        """Derive liveness without persisting an epoch that may still abort."""
        normalized_hashes = head_hashes or {}
        self._validate_update(
            epoch_id, current_hotkeys, responding_uids, normalized_hashes
        )
        with self._lock:
            last_epoch_id, persisted = self._decode()
            return self._derive(
                epoch_id,
                last_epoch_id,
                persisted,
                current_hotkeys,
                responding_uids,
                normalized_hashes,
            )

    def update_epoch(
        self,
        epoch_id: str,
        current_hotkeys: Mapping[int, str],
        responding_uids: set[int],
        head_hashes: Mapping[int, str] | None = None,
    ) -> LivenessSnapshot:
        """Derive and atomically persist liveness after an epoch succeeds."""
        normalized_hashes = head_hashes or {}
        self._validate_update(
            epoch_id, current_hotkeys, responding_uids, normalized_hashes
        )
        with self._lock:
            last_epoch_id, persisted = self._decode()
            snapshot = self._derive(
                epoch_id,
                last_epoch_id,
                persisted,
                current_hotkeys,
                responding_uids,
                normalized_hashes,
            )
            self._write(epoch_id, snapshot.records)
            return snapshot
