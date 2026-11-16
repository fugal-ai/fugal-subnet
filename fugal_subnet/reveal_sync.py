"""Follow the root validator's published reveals, with nothing to set up.

WHY. A protocol v2 validator that is not the root checks its seat state
against the root's published reveals and re-anchors on them after a missed
epoch or a lost proof (fugal_subnet/v2_sync.py). Until this module, getting
those reveals onto the validator's disk was the operator's job: a `gsutil
rsync` of the root's bucket on a timer, and FUGAL_FRAME_BOOTSTRAP pointed at
the directory. Each step failed silently when it was missing. A mainnet
validator that skipped them started from the bootstrap frame with an empty
seat, set all of its weight on the burn UID, and logged nothing an operator
would read as an error.

So the validator fetches them itself. The root's bucket is public-read, and a
public GCS object needs no credentials and no client library: this is two
plain HTTPS GETs (the JSON listing, then each object) from the standard
library. A follower on finney needs no variable at all; the root says it is
the root (FUGAL_ROOT=1) or is recognised by publishing to the reference
bucket itself.

TRUST. Unchanged from the directory an operator synced by hand: a follower
that re-anchors trusts the root's published state (v2_rebuild's docstring
states the model). The fetch adds nothing to it and removes nothing from it.
Transport integrity is HTTPS plus the object's MD5 from the listing; a file
that fails either, does not parse, or names another epoch is discarded, never
left on disk for the rebuild to read.

BOUNDS (I2, I6). Every request has a timeout and a size cap, a sync has a
total time budget, object names must be exactly `<netuid>/e<digits>/reveal.json`,
and nothing here raises into the epoch loop: `sync` returns what happened and
the validator carries on with what it already has.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

GCS = "https://storage.googleapis.com"
# The root validator's public reveal bucket on finney (netuid 70). Objects are
# `<netuid>/<epoch_id>/reveal.json` (reveal_publisher.publish_due_reveals).
FINNEY_REFERENCE_BUCKET = "fugal-reveals"
# A v2 reveal is about 0.6 MB; this only stops a hostile or broken object from
# filling the disk or the process.
MAX_REVEAL_BYTES = 64 * 1024 * 1024
MAX_LISTING_BYTES = 8 * 1024 * 1024
MAX_LISTING_PAGES = 64
REQUEST_TIMEOUT_S = 30.0
# GCS bucket names: 3-63 of lowercase letters, digits, dash, underscore, dot.
_BUCKET = re.compile(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]")
_EPOCH_ID = re.compile(r"e\d{8,12}")


def valid_bucket(name: str) -> bool:
    return bool(_BUCKET.fullmatch(str(name or "")))


def _epoch_index(epoch_id: str) -> int:
    return int(epoch_id[1:])


@dataclass(frozen=True)
class Listed:
    epoch_id: str
    size: int
    md5: str            # hex; "" when the listing carries none


@dataclass
class SyncResult:
    listed: int = 0                 # reveals the bucket holds in range
    downloaded: int = 0
    present: int = 0                # already on disk and intact
    failed: list[str] = field(default_factory=list)   # "<epoch id>: <why>"
    newest: str | None = None       # newest epoch id on disk after the sync
    error: str | None = None        # the listing itself failed
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and not self.failed

    def summary(self) -> str:
        if self.error:
            return f"listing failed ({self.error})"
        text = (f"{self.downloaded} fetched, {self.present} already here, newest "
                f"{self.newest or 'none'}, {self.seconds:.1f}s")
        if self.failed:
            text += f"; {len(self.failed)} FAILED ({'; '.join(self.failed[:3])})"
        return text


def _get(url: str, limit: int, timeout: float, opener: Callable | None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fugal-validator"})
    # Resolved here, not as a default argument, so the one place that opens a
    # socket can be replaced in a test.
    with (opener or urllib.request.urlopen)(req, timeout=timeout) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"response larger than {limit} bytes")
    return data


def list_reveals(bucket: str, netuid: int, *, timeout: float = REQUEST_TIMEOUT_S,
                 opener: Callable | None = None) -> dict[int, Listed]:
    """{epoch index: Listed} of every reveal object under `<netuid>/`.

    Raises on a failed or malformed listing; the caller (`sync`) turns that
    into a result. Names that are not exactly `<netuid>/e<digits>/reveal.json`
    are ignored: an object name is text someone else chose.
    """
    if not valid_bucket(bucket):
        raise ValueError(f"{bucket!r} is not a bucket name")
    prefix = f"{int(netuid)}/"
    out: dict[int, Listed] = {}
    token = ""
    for _page in range(MAX_LISTING_PAGES):
        query = {"prefix": prefix, "maxResults": "1000",
                 "fields": "items(name,size,md5Hash),nextPageToken"}
        if token:
            query["pageToken"] = token
        url = f"{GCS}/storage/v1/b/{bucket}/o?{urllib.parse.urlencode(query)}"
        page = json.loads(_get(url, MAX_LISTING_BYTES, timeout, opener).decode("utf-8"))
        if not isinstance(page, dict):
            raise ValueError("listing is not a JSON object")
        for item in page.get("items") or []:
            if not isinstance(item, dict):
                continue
            parts = str(item.get("name", "")).split("/")
            if len(parts) != 3 or parts[0] != str(int(netuid)) or parts[2] != "reveal.json" \
                    or not _EPOCH_ID.fullmatch(parts[1]):
                continue
            try:
                size = int(item.get("size", -1))
                md5 = base64.b64decode(str(item.get("md5Hash", "")), validate=True).hex()
            except (TypeError, ValueError):
                continue
            out[_epoch_index(parts[1])] = Listed(parts[1], size, md5)
        token = str(page.get("nextPageToken") or "")
        if not token:
            return out
    raise ValueError(f"listing did not end within {MAX_LISTING_PAGES} pages")


def _intact(path: str, item: Listed) -> bool:
    try:
        if os.path.getsize(path) != item.size:
            return False
        if not item.md5:
            return True
        h = hashlib.md5(usedforsecurity=False)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest() == item.md5
    except OSError:
        return False


def _fetch(bucket: str, netuid: int, item: Listed, dest: str, timeout: float,
           opener: Callable | None) -> None:
    """Download one reveal into `dest/<epoch id>/reveal.json`, atomically."""
    if not 0 < item.size <= MAX_REVEAL_BYTES:
        raise ValueError(f"size {item.size} outside 1..{MAX_REVEAL_BYTES}")
    url = f"{GCS}/{bucket}/{int(netuid)}/{item.epoch_id}/reveal.json"
    data = _get(url, MAX_REVEAL_BYTES, timeout, opener)
    if len(data) != item.size:
        raise ValueError(f"got {len(data)} bytes, the listing says {item.size}")
    if item.md5 and hashlib.md5(data, usedforsecurity=False).hexdigest() != item.md5:
        raise ValueError("MD5 differs from the listing")
    reveal = json.loads(data)
    if not isinstance(reveal, dict) or reveal.get("epoch_id") != item.epoch_id:
        raise ValueError("not a reveal for this epoch")
    folder = os.path.join(dest, item.epoch_id)
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, "reveal.json.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(folder, "reveal.json"))


def newest_local(dest: str) -> str | None:
    try:
        ids = [d for d in os.listdir(dest)
               if _EPOCH_ID.fullmatch(d) and os.path.exists(os.path.join(dest, d, "reveal.json"))]
    except OSError:
        return None
    return max(ids, key=_epoch_index) if ids else None


def sync(bucket: str, netuid: int, dest: str, *, since: int = 0, budget_s: float = 120.0,
         timeout: float = REQUEST_TIMEOUT_S, opener: Callable | None = None,
         clock: Callable[[], float] = time.monotonic) -> SyncResult:
    """Bring `dest` up to date with the bucket's reveals of epochs >= `since`.

    Oldest first, so an interrupted sync leaves a contiguous prefix. A reveal
    already on disk with the listed size and MD5 is not fetched again. Stops
    when `budget_s` is spent (the rest is fetched by the next call). Never
    raises.
    """
    start = clock()
    result = SyncResult()
    try:
        listed = list_reveals(bucket, netuid, timeout=timeout, opener=opener)
    except Exception as e:  # noqa: BLE001 - reported; the validator keeps what it has
        result.error = f"{type(e).__name__}: {e}"
        result.newest = newest_local(dest)
        result.seconds = clock() - start
        return result
    wanted = [listed[k] for k in sorted(listed) if k >= int(since)]
    result.listed = len(wanted)
    for item in wanted:
        path = os.path.join(dest, item.epoch_id, "reveal.json")
        if os.path.exists(path) and _intact(path, item):
            result.present += 1
            continue
        if clock() - start > budget_s:
            result.failed.append(f"{item.epoch_id}: time budget of {budget_s:.0f}s spent")
            break
        try:
            _fetch(bucket, netuid, item, dest, timeout, opener)
            result.downloaded += 1
        except Exception as e:  # noqa: BLE001 - one object, not the sync
            result.failed.append(f"{item.epoch_id}: {type(e).__name__}: {e}")
    result.newest = newest_local(dest)
    result.seconds = clock() - start
    return result


@dataclass(frozen=True)
class Role:
    """What this validator is in the v2 star: the root, or a follower of one."""

    root: bool
    # Where the reference reveals are read from (None for a root).
    reference_dir: str | None
    # The bucket this validator syncs `reference_dir` from itself (None when
    # the operator manages the directory, or for a root).
    bucket: str | None
    why: str

    def describe(self) -> str:
        if self.root:
            return f"ROOT ({self.why}): the other validators verify against this one's reveals"
        source = (f"gs://{self.bucket} (fetched by this validator)" if self.bucket
                  else "kept in sync by the operator")
        return f"follower of the reveals in {self.reference_dir}, {source} ({self.why})"


def resolve_role(*, finney: bool, root_flag: bool | None, own_bucket: str, reference_bucket: str,
                 explicit_dir: str | None, state_path: str, netuid: int) -> Role | None:
    """Root or follower, from what is configured. None: neither (a chain with
    no reference configured, where a validator runs alone as before).

    - FUGAL_ROOT=1, or publishing to the reference bucket itself, is the root.
      FUGAL_ROOT=0 is never the root, whatever it publishes to (a root that
      lost its state follows another validator's reveals until it catches up).
    - FUGAL_FRAME_BOOTSTRAP names a directory the operator keeps in sync (the
      arrangement before this module, still honoured as it is).
    - Otherwise a reference bucket (finney's is built in; elsewhere
      FUGAL_REFERENCE_BUCKET) is fetched into a directory beside the state.
    """
    bucket = reference_bucket or (FINNEY_REFERENCE_BUCKET if finney else "")
    if root_flag:
        return Role(True, None, None, "FUGAL_ROOT=1")
    if root_flag is None and own_bucket and bucket and own_bucket == bucket:
        return Role(True, None, None, f"FUGAL_REVEAL_BUCKET is the reference bucket {bucket}")
    if explicit_dir:
        return Role(False, explicit_dir, None, "FUGAL_FRAME_BOOTSTRAP")
    if bucket:
        dest = os.path.join(os.path.dirname(os.path.abspath(state_path)),
                            "reference-reveals", str(int(netuid)))
        return Role(False, dest, bucket, "the default" if not reference_bucket
                    else "FUGAL_REFERENCE_BUCKET")
    return None


def reference_status(reference_dir: str, own_dir: str, before: int, *,
                     differ_limit: int) -> dict:
    """What the newest reference reveal says, beside this validator's own.

    For the per-epoch `Following root` line: the root's seat (champion hash
    and the UID serving it), its fugal version, how many epochs the newest
    reference reveal is behind `before`, and for how many of the newest
    consecutive reference epochs this validator's own seat differed from the
    root's (`seat_differs`, counted up to `differ_limit`). Never raises: a
    missing or unreadable reveal reads as "nothing to report".
    """
    from fugal_subnet.v2_sync import load_reveals

    out = {"k": None, "lag": None, "champion": None, "uid": None, "version": None,
           "seat_differs": 0, "own_champion": None}
    try:
        since = before - 64
        ref, own = load_reveals(reference_dir, since), load_reveals(own_dir, since)
        ks = sorted((k for k in ref if k < before), reverse=True)
        if not ks:
            return out
        k = ks[0]
        rv = ref[k][1]
        state = rv.get("v2_state") or {}
        champion = (state.get("seat") or {}).get("champion")
        out.update(k=k, lag=before - 1 - k, champion=champion,
                   uid=(state.get("serving") or {}).get(champion) if champion else None,
                   version=(rv.get("environment") or {}).get("fugal_version"))
        for j in ks[:max(1, int(differ_limit))]:
            if j not in own:
                break
            mine = ((own[j][1].get("v2_state") or {}).get("seat") or {}).get("champion")
            theirs = ((ref[j][1].get("v2_state") or {}).get("seat") or {}).get("champion")
            if j == k:
                out["own_champion"] = mine
            if mine == theirs:
                break
            out["seat_differs"] += 1
    except Exception as e:  # noqa: BLE001 - a log line never costs an epoch
        logger.debug("reference status unavailable: %s", e)
    return out
