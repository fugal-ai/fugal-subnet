"""Publish verified heads, by hash, with the reveal that scores them.

A reveal names every verified head by its sha256 (`head_hashes`) and says how
each routed, but the head's bytes stayed with the miner and the validators
that queried it. With this module the validator that publishes reveals also
publishes those bytes, so anyone can fetch the champion's head from a public
place and check it against the reveal and the miner's on-chain commitment.

Operational, not consensus. Nothing here is read by a score, a weight, the
reveal, the commit hash or the consensus digest, and a validator with the
feature off publishes byte-identical reveals (tests/test_head_publisher.py).

LAYOUT, in the reveal bucket:

    heads/<netuid>/<sha256>.npz      the head, exactly the bytes the miner served
    <netuid>/<epoch_id>/heads.json   beside the reveal: where each of its heads is

The head objects are outside `<netuid>/` on purpose. Followers list that
prefix every epoch to find reveals (reveal_sync.list_reveals), within a page
limit, and the number of distinct heads is decided by miners; the sidecar adds
one small object per epoch there and no more. The locations go in a sidecar,
not in the reveal: a field in the reveal would make the reveals of a
validator that publishes heads differ from those of one that does not, and
reveals are what validators are compared on (I1).

WHEN. A head becomes public only once a published reveal lists it, never
earlier: an epoch's heads are uploaded after that epoch's reveal carries its
`.published` marker, and only when the epoch is past the reveal's own hold
(`reveal_publisher.delay_epochs_for`); both are checked here. The hashes are
read from the reveal file whose sha256 the marker recorded, so what is
uploaded is exactly what the public reveal names. A head verified in a later
epoch waits for that epoch's reveal.

WHAT. Only heads the validator verified (the caller passes the same
`verified_heads` the reveal's `head_hashes` come from). They wait in a private
outbox, `<sha256>.npz`, mode 0600 in a 0700 directory. The sha256 is
recomputed when a head enters the outbox and again before it is uploaded, the
size cap (`HEAD_MAX_BYTES`) is applied both times, and a file that fails
either is removed, never sent. A head is never parsed here: no `np.load`, no
unzip. The private archive (head_archive.py) is a separate directory and is
never uploaded: heads verified before this feature was switched on are not
published unless a miner serves them again.

FAILURE. Uploading is the last thing an epoch does, after the weights are set,
the reveal is published and the state is saved. It runs in a worker thread
with a time budget; a store that fails, or does not answer, costs a warning
and nothing else, and what was not sent is tried again on later epochs: the
bytes stay in the outbox, and an epoch whose heads are not all settled stays
open for `RETRY_EPOCHS` epochs. A head still being served is kept again each
epoch, so it is published with a later reveal even after that. Existing
objects are never overwritten (`upload_if_absent`), and a `<sha256>.sent`
record stops a head being sent twice.

READERS. The store is not trusted and does not need to be: check that the
sha256 of the fetched bytes equals the head hash (the object's name, the
reveal's `head_hashes` value, the miner's commitment), and load them with
pickle disabled, `np.load(..., allow_pickle=False)`, within the size limits
(`head_eval.load_head_from_npz` does both). `reveal_sync.fetch_head` fetches
and checks the hash.

Log lines and the sidecar carry hashes, epoch ids, counts and object names:
no local path, no credential, and no text from an upload error beyond the
exception's type (a failed command line would name local files).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time

logger = logging.getLogger("fugal.head_publisher")

_HEX64 = re.compile(r"[0-9a-f]{64}")
_EPOCH_ID = re.compile(r"e\d{1,12}")

HEADS_PREFIX = "heads"
SIDECAR_NAME = "heads.json"
# The reveal publisher's marker (reveal_publisher.MARKER): the reveal is public.
REVEAL_MARKER = ".published"
# This module's marker: every head of the epoch is settled and the sidecar is up.
EPOCH_MARKER = ".heads_published"
SENT_SUFFIX = ".sent"

# The outbox holds at most this many unsent heads (at most HEAD_MAX_BYTES
# each). A subnet has 256 UIDs, so this is several epochs of a field that
# changes every head every epoch; past it the heads seen longest ago go.
SPOOL_MAX_HEADS = 1024
# How long an epoch whose heads are not all settled is retried.
RETRY_EPOCHS = 20
# A store that failed this many times in one run is left until next epoch.
MAX_FAILURES_PER_RUN = 3
# Seconds one epoch may spend uploading heads.
PUBLISH_BUDGET_S = 120.0

VERIFY_NOTE = ("the sha256 of the fetched bytes must equal the head hash (the object's name "
               "and the reveal's head_hashes value); load with "
               "numpy.load(..., allow_pickle=False)")


def head_key(netuid: int, sha256: str) -> str:
    """The object name of a head. Refuses anything that is not a sha256."""
    if not isinstance(sha256, str) or not _HEX64.fullmatch(sha256):
        raise ValueError("a head is named by its 64-character lowercase hex sha256")
    return f"{HEADS_PREFIX}/{int(netuid)}/{sha256}.npz"


def sidecar_key(netuid: int, epoch_id: str) -> str:
    return f"{int(netuid)}/{epoch_id}/{SIDECAR_NAME}"


def spool_dir_for(epoch_dir: str) -> str:
    """The default outbox: beside the epoch directory, never inside it."""
    parent = os.path.dirname(os.path.abspath(epoch_dir.rstrip("/") or "/"))
    return os.path.join(parent, "head-outbox")


def _write_private(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        _remove(tmp)
        raise


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _evict(spool_dir: str) -> int:
    """Keep the outbox within SPOOL_MAX_HEADS; the heads seen longest ago go."""
    aged = []
    for name in os.listdir(spool_dir):
        if name.endswith(".npz") and _HEX64.fullmatch(name[:-4]):
            try:
                aged.append((os.path.getmtime(os.path.join(spool_dir, name)), name))
            except OSError:
                continue
    aged.sort()
    dropped = 0
    for _mtime, name in aged[:max(0, len(aged) - SPOOL_MAX_HEADS)]:
        _remove(os.path.join(spool_dir, name))
        dropped += 1
    return dropped


def spool_verified_heads(spool_dir: str, heads: dict[int, bytes],
                         weights_hashes: dict[int, str], max_bytes: int) -> dict[str, int]:
    """Keep each verified head in the private outbox until its reveal is public.

    Returns counts: `kept` (new), `present` (already waiting), `sent` (already
    published, not kept again), `skipped` (bytes that do not match their hash,
    are empty or exceed `max_bytes`: cannot happen for a head the validator
    accepted, refused anyway so a miner-supplied string never becomes a path).
    """
    counts = {"kept": 0, "present": 0, "sent": 0, "skipped": 0}
    if not spool_dir or not heads:
        return counts
    made = False
    for uid in sorted(heads):
        data = heads[uid]
        if not isinstance(data, (bytes, bytearray)) or not 0 < len(data) <= max_bytes:
            counts["skipped"] += 1
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest != weights_hashes.get(uid, ""):
            logger.warning("UID %d: head sha256 %s... is not its weights_hash; not kept for "
                           "publication", uid, digest[:12])
            counts["skipped"] += 1
            continue
        if not made:
            os.makedirs(spool_dir, mode=0o700, exist_ok=True)
            made = True
        path = os.path.join(spool_dir, f"{digest}.npz")
        if os.path.exists(os.path.join(spool_dir, digest + SENT_SUFFIX)):
            counts["sent"] += 1
        elif os.path.exists(path):
            # Seen again: the newest in the outbox, so the last to be dropped.
            os.utime(path, None)
            counts["present"] += 1
        else:
            _write_private(path, bytes(data))
            counts["kept"] += 1
    if made:
        dropped = _evict(spool_dir)
        if dropped:
            logger.warning("Head outbox is full: dropped %d unsent head(s) seen longest ago "
                           "(they are kept again if a miner still serves them)", dropped)
    return counts


def _sent_record(spool_dir: str, sha256: str) -> dict | None:
    try:
        with open(os.path.join(spool_dir, sha256 + SENT_SUFFIX), encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, ValueError):
        return None
    if (isinstance(record, dict) and record.get("key") and isinstance(record.get("url"), str)
            and isinstance(record.get("bytes"), int)):
        return {"key": record["key"], "url": record["url"], "bytes": record["bytes"]}
    return None


def _send(spool_dir: str, sha256: str, netuid: int, uploader, max_bytes: int) -> dict | None:
    """Upload one waiting head. Returns its record, or None when this
    validator holds no publishable bytes for the hash. Raises what the
    uploader raises."""
    path = os.path.join(spool_dir, f"{sha256}.npz")
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes + 1)
    except OSError:
        return None
    if not 0 < len(data) <= max_bytes or hashlib.sha256(data).hexdigest() != sha256:
        logger.warning("Head %s...: the waiting file is not that head any more; removed, "
                       "not uploaded", sha256[:12])
        _remove(path)
        return None
    key = head_key(netuid, sha256)
    url = uploader.upload_if_absent(path, key)
    record = {"key": key, "url": str(url), "bytes": len(data)}
    _write_private(os.path.join(spool_dir, sha256 + SENT_SUFFIX),
                   json.dumps(record, sort_keys=True).encode("utf-8"))
    _remove(path)
    return record


def _published_reveal(folder: str, epoch_id: str) -> tuple[dict, str] | None:
    """The reveal in `folder` and its sha256, if it is the one that was published."""
    try:
        with open(os.path.join(folder, REVEAL_MARKER), encoding="utf-8") as f:
            marker = json.load(f)
        with open(os.path.join(folder, "reveal.json"), "rb") as f:
            data = f.read()
        reveal = json.loads(data)
    except (OSError, ValueError):
        return None
    digest = hashlib.sha256(data).hexdigest()
    if (not isinstance(marker, dict) or marker.get("sha256") != digest
            or marker.get("epoch_id") != epoch_id
            or not isinstance(reveal, dict) or reveal.get("epoch_id") != epoch_id):
        return None
    return reveal, digest


def _reveal_hashes(reveal: dict) -> list[str]:
    """The reveal's head hashes: the champion's first, then in sorted order."""
    listed = reveal.get("head_hashes")
    hashes = sorted({h for h in (listed.values() if isinstance(listed, dict) else [])
                     if isinstance(h, str) and _HEX64.fullmatch(h)})
    decision = reveal.get("champion")
    champion = decision.get("champion") if isinstance(decision, dict) else None
    if champion in hashes:
        hashes.remove(champion)
        hashes.insert(0, champion)
    return hashes


def _due_epochs(epoch_dir: str, current_epoch_index: int, delay_epochs: int,
                retry_epochs: int) -> list[str]:
    """Epoch ids past the hold, inside the retry window, published and not yet
    settled here; oldest first."""
    newest = current_epoch_index - max(1, int(delay_epochs))
    try:
        names = os.listdir(epoch_dir)
    except OSError:
        return []
    due = []
    for name in names:
        if not _EPOCH_ID.fullmatch(name):
            continue
        index = int(name[1:])
        if not newest - retry_epochs < index <= newest:
            continue
        folder = os.path.join(epoch_dir, name)
        if (os.path.exists(os.path.join(folder, REVEAL_MARKER))
                and not os.path.exists(os.path.join(folder, EPOCH_MARKER))):
            due.append((index, name))
    return [name for _index, name in sorted(due)]


def publish_due_heads(epoch_dir: str, spool_dir: str, current_epoch_index: int,
                      delay_epochs: int, netuid: int, uploader, *, max_bytes: int,
                      deadline: float | None = None,
                      retry_epochs: int = RETRY_EPOCHS) -> dict:
    """Upload the heads of every published reveal that is past its hold.

    `uploader.upload_if_absent(path, key) -> url` must not overwrite an
    existing object; `uploader.upload(path, key) -> url` sends the sidecar.
    `deadline` is a `time.monotonic()` instant after which nothing more is
    started. Never raises for a store failure. Returns what happened:
    `epochs` settled (sidecar published), heads `uploaded`, hashes
    `unavailable` (listed by a reveal, no bytes kept here), `failed` uploads,
    and `out_of_time`.
    """
    out = {"epochs": [], "uploaded": 0, "unavailable": 0, "failed": 0, "out_of_time": False}

    def stop() -> bool:
        if out["failed"] >= MAX_FAILURES_PER_RUN:
            return True
        if deadline is not None and time.monotonic() >= deadline:
            out["out_of_time"] = True
            return True
        return False

    for epoch_id in _due_epochs(epoch_dir, current_epoch_index, delay_epochs, retry_epochs):
        if stop():
            break
        folder = os.path.join(epoch_dir, epoch_id)
        found = _published_reveal(folder, epoch_id)
        if found is None:
            logger.warning("Epoch %s: the local reveal is not the one that was published; its "
                           "heads are not uploaded", epoch_id)
            continue
        reveal, reveal_sha256 = found
        records: dict[str, dict] = {}
        unavailable: list[str] = []
        settled = True
        for sha256 in _reveal_hashes(reveal):
            record = _sent_record(spool_dir, sha256)
            if record is None:
                if stop():
                    settled = False
                    break
                try:
                    record = _send(spool_dir, sha256, netuid, uploader, max_bytes)
                except Exception as e:  # noqa: BLE001 - reported, retried next epoch
                    out["failed"] += 1
                    settled = False
                    logger.warning("Head %s... not published (%s); will retry",
                                   sha256[:12], type(e).__name__)
                    continue
                if record is not None:
                    out["uploaded"] += 1
            if record is None:
                unavailable.append(sha256)
            else:
                records[sha256] = record
        if not settled or stop():
            continue
        sidecar = {
            "epoch_id": epoch_id,
            "netuid": int(netuid),
            "reveal_sha256": reveal_sha256,
            "layout": f"{HEADS_PREFIX}/<netuid>/<sha256>.npz",
            "heads": records,
            "unavailable": sorted(unavailable),
            "verify": VERIFY_NOTE,
        }
        try:
            local = os.path.join(folder, SIDECAR_NAME)
            _write_private(local, json.dumps(sidecar, sort_keys=True, indent=2).encode("utf-8"))
            uploader.upload(local, sidecar_key(netuid, epoch_id))
            _write_private(os.path.join(folder, EPOCH_MARKER),
                           json.dumps({"epoch_id": epoch_id, "heads": len(records),
                                       "unavailable": len(unavailable)}).encode("utf-8"))
        except Exception as e:  # noqa: BLE001 - reported, retried next epoch
            out["failed"] += 1
            logger.warning("Epoch %s: head list not published (%s); will retry",
                           epoch_id, type(e).__name__)
            continue
        out["epochs"].append(epoch_id)
        out["unavailable"] += len(unavailable)
        logger.info("Epoch %s: %d head(s) public by hash, %d not held here",
                    epoch_id, len(records), len(unavailable))
    return out


class HeadPublisher:
    """What the validator holds: the outbox, the store, and a time budget.

    Neither method raises, and `publish` returns within `budget_s` whatever
    the store does: the upload runs in a daemon thread, and one that is still
    running when the budget is spent is left to finish (its results are
    recorded as usual) while the epoch loop moves on. A new upload is not
    started while an old one is still running.
    """

    def __init__(self, epoch_dir: str, netuid: int, uploader, delay_epochs: int, *,
                 max_bytes: int, spool_dir: str | None = None,
                 budget_s: float = PUBLISH_BUDGET_S):
        self.epoch_dir = epoch_dir
        self.netuid = int(netuid)
        self.uploader = uploader
        self.delay_epochs = int(delay_epochs)
        self.max_bytes = int(max_bytes)
        self.spool_dir = spool_dir or spool_dir_for(epoch_dir)
        self.budget_s = float(budget_s)
        self._worker: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def spool(self, heads: dict[int, bytes], weights_hashes: dict[int, str]) -> None:
        """Keep this epoch's verified heads for publication with its reveal."""
        try:
            counts = spool_verified_heads(self.spool_dir, heads, weights_hashes, self.max_bytes)
            logger.info("Heads for publication: %d new, %d waiting, %d already public, "
                        "%d skipped", counts["kept"], counts["present"], counts["sent"],
                        counts["skipped"])
        except Exception as e:  # noqa: BLE001 - publication must not cost the epoch
            logger.warning("Heads not kept for publication this epoch (%s)", type(e).__name__)

    def publish(self, current_epoch_index: int) -> dict | None:
        """Upload what is due. None when nothing could be reported (the budget
        was spent, an earlier upload is still running, or an error)."""
        try:
            if self.busy:
                logger.warning("Head publication: the previous upload is still running; "
                               "nothing started this epoch")
                return None
            box: dict = {}
            deadline = time.monotonic() + self.budget_s

            def work() -> None:
                try:
                    box["out"] = publish_due_heads(
                        self.epoch_dir, self.spool_dir, current_epoch_index,
                        self.delay_epochs, self.netuid, self.uploader,
                        max_bytes=self.max_bytes, deadline=deadline)
                except Exception as e:  # noqa: BLE001
                    box["error"] = type(e).__name__

            worker = threading.Thread(target=work, name="fugal-head-publisher", daemon=True)
            self._worker = worker
            worker.start()
            worker.join(self.budget_s)
            if worker.is_alive():
                logger.warning("Head publication used its whole time budget (%.0fs) and is "
                               "still running; the epoch goes on, the rest is retried",
                               self.budget_s)
                return None
            if "out" not in box:
                logger.warning("Head publication failed this epoch (%s); will retry",
                               box.get("error", "unknown"))
                return None
            out = box["out"]
            if out["uploaded"] or out["failed"] or out["out_of_time"]:
                logger.info("Head publication: %d uploaded, %d failed, %d epoch(s) settled%s",
                            out["uploaded"], out["failed"], len(out["epochs"]),
                            ", out of time" if out["out_of_time"] else "")
            return out
        except Exception as e:  # noqa: BLE001 - publication must not cost the epoch
            logger.warning("Head publication failed this epoch (%s); will retry",
                           type(e).__name__)
            return None
