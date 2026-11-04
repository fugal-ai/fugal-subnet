"""Private on-disk archive of verified miner heads (operational; feeds no score).

Every axon response already carries the miner's head (`head_npz_b64`). The
validator checks those bytes against the attested `weights_hash` and the
on-chain commitment, then used to throw them away. With
`FUGAL_HEAD_ARCHIVE_DIR` set, the heads that VERIFIED are kept:

    <dir>/<weights_hash>.npz     raw bytes, written once per hash, mode 0600
    <dir>/index.jsonl            one row per (epoch, uid): who served which hash

Rules this module keeps:

- **Bytes only.** A head is never parsed here: no `np.load`, no unzip. The
  archive holds exactly what the miner sent, so it can be re-verified later
  against `weights_hash`.
- **Only verified heads.** The caller passes heads that already passed
  `verify_proof`. The sha256 is recomputed anyway, and the result is the file
  name. A mismatch is skipped, so a miner-supplied string never becomes a path.
- **Private.** The directory is created 0700 and files 0600. Publishing is a
  separate, announced step; nothing here uploads.
- **Never fatal.** Archiving is diagnostics. A disk error costs the archive
  row, not the epoch.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time

logger = logging.getLogger("fugal.head_archive")

_HEX64 = re.compile(r"[0-9a-f]{64}")


def archive_verified_heads(
    archive_dir: str,
    epoch_id: str,
    heads: dict[int, bytes],
    weights_hashes: dict[int, str],
    hotkeys: dict[int, str],
    coldkeys: dict[int, str],
) -> dict[str, int]:
    """Write each verified head once per hash and append one index row per UID.

    Returns counts: `first_seen`, `repeat`, `skipped`. With an empty
    `archive_dir` this does nothing and returns zeros.
    """
    counts = {"first_seen": 0, "repeat": 0, "skipped": 0}
    if not archive_dir or not heads:
        return counts

    os.makedirs(archive_dir, mode=0o700, exist_ok=True)
    rows = []
    for uid in sorted(heads):
        head_bytes = heads[uid]
        digest = hashlib.sha256(head_bytes).hexdigest()
        claimed = weights_hashes.get(uid, "")
        if not _HEX64.fullmatch(digest) or digest != claimed:
            # Cannot happen for a head verify_proof accepted; refuse rather
            # than archive something the index would misdescribe.
            logger.warning("UID %d: head sha256 %s... != weights_hash %s...; not archived",
                           uid, digest[:12], str(claimed)[:12])
            counts["skipped"] += 1
            continue
        path = os.path.join(archive_dir, f"{digest}.npz")
        if os.path.exists(path):
            status = "repeat"
        else:
            tmp = f"{path}.tmp.{os.getpid()}"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(head_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            status = "first_seen"
        counts[status] += 1
        rows.append({
            "epoch_id": epoch_id,
            "uid": int(uid),
            "hotkey": hotkeys.get(uid, ""),
            "coldkey": coldkeys.get(uid, ""),
            "weights_hash": digest,
            "status": status,
            "bytes": len(head_bytes),
            "archived_at": time.time(),
        })

    if rows:
        index = os.path.join(archive_dir, "index.jsonl")
        fd = os.open(index, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    return counts
