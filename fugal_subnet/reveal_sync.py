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

TRUST. A follower that re-anchors trusts the root's published state
(v2_rebuild's docstring states the model). What it must not trust is the
bucket on its own: followers verify downloaded reveals against the root's
on-chain commitment. The root commits `reveal:<epoch_id>:<sha256>` for the
newest reveal it publishes (reveal_publisher), and the pallet keeps one
commitment per hotkey, so the chain vouches for one reveal at a time. So:
- the reveal at the root's committed epoch must have the committed sha256;
- a reveal newer than the commitment is not vouched for yet: it is not
  fetched, and a copy already on disk is removed, so the re-anchor never
  reads it;
- every commitment the follower reads is recorded in a small file beside the
  reference directory (`<reference_dir>.pointers.json`), and an older reveal
  with a recorded commitment must match it;
- an older reveal with no recorded commitment (one published before this
  follower first ran) is accepted as before. That is the known limit: the
  chain no longer holds those pointers. Reveals that each carry the hash of
  the one before would close it (planned with the protocol v3 reset).
Without the commitment (it could not be read) nothing is fetched and what is
on disk stays. The root's hotkey is built in on finney
(config.FINNEY_ROOT_HOTKEY); elsewhere FUGAL_ROOT_HOTKEY names it, and without
it a follower fetches unverified, as before. An operator-managed directory
(FUGAL_FRAME_BOOTSTRAP) is the operator's to keep. Transport integrity is
HTTPS plus the object's MD5 from the listing; a file that fails either, does
not parse, or names another epoch is discarded, never left on disk for the
rebuild to read.

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
# The most bytes of one reveal any node reads: the follower sync's fetch, the
# validator's read of its own reveal, the miner's holder read
# (td_v4.MAX_HOLDER_REVEAL_BYTES) and the root summaries' read
# (menu_publish.MAX_REVEAL_READ_BYTES) all name this one limit. A v2 reveal is
# about 0.6 MB; a protocol v4 reveal at 256 UIDs is at most three quarters of
# it (tests/test_v4_reveal_budget.py). It only stops a hostile or broken
# object from filling the disk or the process; a reveal's content never
# depends on it.
MAX_REVEAL_BYTES = 96 * 1024 * 1024
MAX_LISTING_BYTES = 8 * 1024 * 1024
MAX_LISTING_PAGES = 64
REQUEST_TIMEOUT_S = 30.0
# GCS bucket names: 3-63 of lowercase letters, digits, dash, underscore, dot.
_BUCKET = re.compile(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]")
_EPOCH_ID = re.compile(r"e\d{8,12}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# `reveal:<epoch_id>:<sha256>`, and from protocol v4 optionally `:<menu hash
# token>` (reveal_publisher.commitment_for). The first three fields are the
# same in both, so a follower reads the reveal pointer of either.
_POINTER = re.compile(r"reveal:(e\d{8,12}):([0-9a-f]{64})(?::([A-Za-z0-9_-]{43}))?")
# The recorded commitments: one per epoch the follower saw, so this bounds
# the file at a few hundred kilobytes (about 200 days of epochs).
MAX_POINTERS = 4096


def valid_bucket(name: str) -> bool:
    return bool(_BUCKET.fullmatch(str(name or "")))


def _epoch_index(epoch_id: str) -> int:
    return int(epoch_id[1:])


# --- the root's on-chain commitment -------------------------------------------


def parse_pointer(data) -> tuple[str, str] | None:
    """`reveal:<epoch_id>:<sha256>` (with or without a v4 menu field) ->
    (epoch_id, sha256); anything else None."""
    got = parse_commitment(data)
    return (got[0], got[1]) if got is not None else None


def parse_commitment(data) -> tuple[str, str, str | None] | None:
    """A reveal commitment -> (epoch_id, reveal sha256, menu hash or None).
    A fourth field that is not the canonical token of a menu hash makes the
    whole commitment unreadable (None), never a commitment without a menu."""
    from fugal_subnet.reveal_publisher import decode_menu_hash

    if not isinstance(data, str):
        return None
    m = _POINTER.fullmatch(data)
    if not m:
        return None
    if m.group(3) is None:
        return m.group(1), m.group(2), None
    menu_hash = decode_menu_hash(m.group(3))
    return (m.group(1), m.group(2), menu_hash) if menu_hash is not None else None


def read_root_pointer(subtensor, netuid: int, hotkey: str) -> tuple[str, str] | None:
    """The root hotkey's current reveal commitment, read at the chain head.

    Operational, not consensus (nothing scored reads it), so the head is the
    right block: the newest pointer. Never raises; None when it cannot be read
    or is not a reveal pointer."""
    from fugal_subnet.commitments import get_commitments_with_blocks

    try:
        data, _block = get_commitments_with_blocks(subtensor, netuid).get(hotkey, ("", 0))
    except Exception as e:  # noqa: BLE001 - reported by the caller as "no commitment"
        logger.warning("Could not read the root's reveal commitment: %s", e)
        return None
    return parse_pointer(data)


def pointer_ledger_path(reference_dir: str) -> str:
    return os.path.normpath(str(reference_dir)) + ".pointers.json"


def load_pointers(path: str) -> dict[str, str]:
    """{epoch_id: sha256} recorded so far; {} for a missing or damaged file
    (entries that are not an epoch id and a sha256 are dropped)."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items()
            if isinstance(k, str) and _EPOCH_ID.fullmatch(k)
            and isinstance(v, str) and _SHA256.fullmatch(v)}


def record_pointer(path: str, epoch_id: str, sha256: str) -> dict[str, str]:
    """Add one commitment to the record, atomically, and return the record.

    The chain is the authority: a commitment for an epoch already recorded
    with another hash replaces it (the root re-published), with a warning."""
    pointers = load_pointers(path)
    old = pointers.get(epoch_id)
    if old == sha256:
        return pointers
    if old is not None:
        logger.warning("The root's commitment for %s changed (%s -> %s); following the chain",
                       epoch_id, old[:16], sha256[:16])
    pointers[epoch_id] = sha256
    if len(pointers) > MAX_POINTERS:
        for k in sorted(pointers, key=_epoch_index)[:len(pointers) - MAX_POINTERS]:
            del pointers[k]
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(pointers.items())), f, indent=0)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return pointers


def _file_sha256(path: str) -> str | None:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


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
    # Reveals that differ from the root's on-chain commitment (verified sync).
    rejected: list[str] = field(default_factory=list)  # "<epoch id>: <why>"
    unvouched: int = 0              # listed past the root's commitment; not taken yet
    verified: str | None = None     # the committed epoch checked, when verifying
    newest: str | None = None       # newest epoch id on disk after the sync
    error: str | None = None        # the listing (or the commitment) failed
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and not self.failed and not self.rejected

    def summary(self) -> str:
        if self.error:
            return f"sync failed ({self.error})"
        text = (f"{self.downloaded} fetched, {self.present} already here, newest "
                f"{self.newest or 'none'}, {self.seconds:.1f}s")
        if self.verified:
            text += f", verified against the root's commitment at {self.verified}"
        if self.unvouched:
            text += f", {self.unvouched} newer awaiting the root's commitment"
        if self.rejected:
            text += (f"; {len(self.rejected)} REJECTED, differ from the root's on-chain "
                     f"commitment ({'; '.join(self.rejected[:3])})")
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
                 opener: Callable | None = None,
                 deadline: Callable[[], bool] | None = None) -> dict[int, Listed]:
    """{epoch index: Listed} of every reveal object under `<netuid>/`.

    Raises on a failed or malformed listing, or when `deadline()` says the
    time budget is spent before the last page; the caller (`sync`) turns that
    into a result. Names that are not exactly `<netuid>/e<digits>/reveal.json`
    are ignored: an object name is text someone else chose.
    """
    if not valid_bucket(bucket):
        raise ValueError(f"{bucket!r} is not a bucket name")
    prefix = f"{int(netuid)}/"
    out: dict[int, Listed] = {}
    token = ""
    for _page in range(MAX_LISTING_PAGES):
        if _page and deadline is not None and deadline():
            raise TimeoutError("time budget spent while listing")
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


class CommitmentMismatch(ValueError):
    """A downloaded reveal's sha256 differs from the root's on-chain commitment."""


def _fetch(bucket: str, netuid: int, item: Listed, dest: str, timeout: float,
           opener: Callable | None, expect_sha256: str | None = None) -> None:
    """Download one reveal into `dest/<epoch id>/reveal.json`, atomically.

    With `expect_sha256` (the root's commitment for this epoch) bytes with
    another hash are refused before anything is written."""
    if not 0 < item.size <= MAX_REVEAL_BYTES:
        raise ValueError(f"size {item.size} outside 1..{MAX_REVEAL_BYTES}")
    url = f"{GCS}/{bucket}/{int(netuid)}/{item.epoch_id}/reveal.json"
    data = _get(url, MAX_REVEAL_BYTES, timeout, opener)
    if len(data) != item.size:
        raise ValueError(f"got {len(data)} bytes, the listing says {item.size}")
    if item.md5 and hashlib.md5(data, usedforsecurity=False).hexdigest() != item.md5:
        raise ValueError("MD5 differs from the listing")
    if expect_sha256 is not None and hashlib.sha256(data).hexdigest() != expect_sha256:
        raise CommitmentMismatch(
            f"sha256 {hashlib.sha256(data).hexdigest()[:16]} is not the committed "
            f"{expect_sha256[:16]}")
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


def fetch_head(bucket: str, netuid: int, sha256: str, *, max_bytes: int | None = None,
               timeout: float = REQUEST_TIMEOUT_S, opener: Callable | None = None) -> bytes:
    """The bytes of a published head, checked against its hash.

    Verified heads are published by sha256, with or after the reveal that
    lists them (fugal_subnet/head_publisher.py says when). `sha256` is the head hash from a reveal's
    `head_hashes` (or its `champion`), which is also the miner's on-chain
    commitment. The store is not trusted: bytes with another sha256, or more
    than `max_bytes` (default `HEAD_MAX_BYTES`), are refused. The bytes are
    returned unparsed. Load them with pickle disabled,
    `np.load(io.BytesIO(data), allow_pickle=False)`, or with
    `head_eval.load_head_from_npz(data)`, which also bounds what they unpack to.
    Raises on any failure; nothing in the epoch loop calls this.
    """
    from fugal_subnet.head_publisher import head_key

    if not valid_bucket(bucket):
        raise ValueError(f"{bucket!r} is not a bucket name")
    if max_bytes is None:
        from fugal_subnet.config import HEAD_MAX_BYTES
        max_bytes = HEAD_MAX_BYTES
    data = _get(f"{GCS}/{bucket}/{head_key(netuid, sha256)}", int(max_bytes), timeout, opener)
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"sha256 of the fetched bytes is not {sha256[:16]}...")
    return data


def newest_local(dest: str) -> str | None:
    try:
        ids = [d for d in os.listdir(dest)
               if _EPOCH_ID.fullmatch(d) and os.path.exists(os.path.join(dest, d, "reveal.json"))]
    except OSError:
        return None
    return max(ids, key=_epoch_index) if ids else None


def _remove_unvouched(dest: str, pointer_index: int) -> None:
    """Remove local reveals newer than the root's commitment: nothing on chain
    vouches for them yet, so the re-anchor must not read them."""
    try:
        names = os.listdir(dest)
    except OSError:
        return
    for d in names:
        if _EPOCH_ID.fullmatch(d) and _epoch_index(d) > pointer_index:
            path = os.path.join(dest, d, "reveal.json")
            if os.path.exists(path):
                logger.warning("Removing reference reveal %s: newer than the root's on-chain "
                               "commitment", d)
                _remove(path)


def sync(bucket: str, netuid: int, dest: str, *, since: int = 0, budget_s: float = 120.0,
         timeout: float = REQUEST_TIMEOUT_S, opener: Callable | None = None,
         clock: Callable[[], float] = time.monotonic, verify: bool = False,
         pointer: tuple[str, str] | None = None,
         ledger_path: str | None = None) -> SyncResult:
    """Bring `dest` up to date with the bucket's reveals of epochs >= `since`.

    Oldest first, so an interrupted sync leaves a contiguous prefix. A reveal
    already on disk with the listed size and MD5 is not fetched again. Stops
    when `budget_s` is spent, in the listing or between downloads (the rest
    is fetched by the next call). Never raises.

    With `verify`, `pointer` is the root's on-chain commitment (epoch_id,
    sha256) and `ledger_path` the record of the commitments seen so far (the
    rules are in the module docstring, TRUST). A missing pointer fetches
    nothing.
    """
    start = clock()
    result = SyncResult()

    def done() -> SyncResult:
        result.newest = newest_local(dest)
        result.seconds = clock() - start
        return result

    pointers: dict[str, str] = {}
    pointer_index: int | None = None
    if verify:
        if pointer is None or ledger_path is None:
            result.error = ("the root's on-chain reveal commitment could not be read; "
                            "nothing fetched, the reveals on disk are kept")
            return done()
        try:
            pointers = record_pointer(ledger_path, pointer[0], pointer[1])
        except OSError as e:
            result.error = f"could not record the root's commitment ({e}); nothing fetched"
            return done()
        pointer_index = _epoch_index(pointer[0])
        result.verified = pointer[0]
        _remove_unvouched(dest, pointer_index)
    try:
        listed = list_reveals(bucket, netuid, timeout=timeout, opener=opener,
                              deadline=lambda: clock() - start > budget_s)
    except Exception as e:  # noqa: BLE001 - reported; the validator keeps what it has
        result.error = f"listing failed: {type(e).__name__}: {e}"
        return done()
    wanted = [listed[k] for k in sorted(listed) if k >= int(since)]
    if pointer_index is not None:
        result.unvouched = sum(1 for item in wanted if _epoch_index(item.epoch_id) > pointer_index)
        wanted = [item for item in wanted if _epoch_index(item.epoch_id) <= pointer_index]
        if pointer_index not in listed:
            result.failed.append(f"{pointer[0]}: committed on chain but not in the bucket")
    result.listed = len(wanted)
    for item in wanted:
        path = os.path.join(dest, item.epoch_id, "reveal.json")
        expect = pointers.get(item.epoch_id)
        if os.path.exists(path):
            if expect is not None and _file_sha256(path) != expect:
                logger.warning("Reference reveal %s on disk differs from the root's on-chain "
                               "commitment; fetching it again", item.epoch_id)
                _remove(path)
            elif _intact(path, item):
                result.present += 1
                continue
        if clock() - start > budget_s:
            result.failed.append(f"{item.epoch_id}: time budget of {budget_s:.0f}s spent")
            break
        try:
            _fetch(bucket, netuid, item, dest, timeout, opener, expect_sha256=expect)
            result.downloaded += 1
        except CommitmentMismatch as e:
            result.rejected.append(f"{item.epoch_id}: {e}")
            if os.path.exists(path) and _file_sha256(path) != expect:
                _remove(path)
        except Exception as e:  # noqa: BLE001 - one object, not the sync
            result.failed.append(f"{item.epoch_id}: {type(e).__name__}: {e}")
    return done()


# --- protocol v4: menus and signed lists, by hash ------------------------------
#
# The root publishes each menu it commits, and each signed list a menu newly
# names, beside its reveals (menu_publish.MenuPublisher), named by content:
#
#   <netuid>/menus/<menu hash>.json          the menu's canonical bytes
#   <netuid>/lists/<kind>/<sha256>.json      a signed list's bytes
#   <netuid>/lists/<kind>/<sha256>.json.sig  its signature
#
# A reader needs no listing and trusts no name: it asks for the hash the chain
# names and checks the bytes against it. These helpers are what a miner's TD
# (td_v4.V4Session's `menu_bytes` and `list_bytes`) and the product call.

# The most bytes read for one menu. Bytes with the committed hash are the
# committed menu whatever their size: the reader (menu_state.menu_for_epoch)
# refuses one over its MENU_MAX_BYTES (1 MiB) unparsed and every node carries
# the previous menu. So the fetch must read past that limit to see the hash,
# up to a bound of its own; past this bound there are no bytes (MenuUnavailable).
MAX_MENU_FETCH_BYTES = 4 * 1024 * 1024
# A signed list and its signature (menu_rules.MAX_LIST_BYTES and
# MAX_SIGNATURE_BYTES; anything larger is refused there unread).
MAX_LIST_FETCH_BYTES = 262144
MAX_SIGNATURE_FETCH_BYTES = 2048
_LIST_KIND = re.compile(r"fugal-[a-z-]{1,40}")


def menu_object(netuid: int, menu_hash: str) -> str:
    if not _SHA256.fullmatch(str(menu_hash)):
        raise ValueError("a menu hash is 64 lowercase hex characters")
    return f"{int(netuid)}/menus/{menu_hash}.json"


def list_object(netuid: int, kind: str, sha256: str) -> str:
    if not _LIST_KIND.fullmatch(str(kind)) or not _SHA256.fullmatch(str(sha256)):
        raise ValueError("a list is named by its kind and its sha256")
    return f"{int(netuid)}/lists/{kind}/{sha256}.json"


def signature_object(netuid: int, kind: str, sha256: str) -> str:
    return list_object(netuid, kind, sha256) + ".sig"


# A join delta (chain_proposals.DELTA_MAX_BYTES; one over it is refused there
# unparsed, so the fetch reads to the same bound and no further).
MAX_JOIN_FETCH_BYTES = 1_048_576


def join_object(netuid: int, delta_hash: str) -> str:
    """Where the root's join delta (protocol v4) with this sha256 lives."""
    if not _SHA256.fullmatch(str(delta_hash)):
        raise ValueError("a join delta is named by its sha256")
    return f"{int(netuid)}/joins/{delta_hash}.json"


def _sources(bucket: str | None, mirrors) -> list[str]:
    """Base URLs to ask, in order: the bucket, then each https mirror.
    `mirrors` None is config.MENU_MIRRORS (FUGAL_MENU_MIRRORS)."""
    if mirrors is None:
        from fugal_subnet import config
        mirrors = config.MENU_MIRRORS
    out = []
    if bucket:
        if not valid_bucket(bucket):
            raise ValueError(f"{bucket!r} is not a bucket name")
        out.append(f"{GCS}/{bucket}")
    for m in mirrors or ():
        text = str(m).rstrip("/")
        if text.startswith("https://"):
            out.append(text)
        else:
            logger.warning("Ignoring a menu mirror that is not an https URL")
    return out


def fetch_object(key: str, *, bucket: str | None, mirrors=None, limit: int,
                 check: Callable[[bytes], bool], timeout: float = REQUEST_TIMEOUT_S,
                 opener: Callable | None = None) -> bytes | None:
    """The first bytes for `key` that pass `check`, from the bucket or a
    mirror, in that order; None when no source serves such bytes. Never
    raises: a source that fails, is too large or serves other bytes is
    skipped, so one bad mirror never hides a good one."""
    for base in _sources(bucket, mirrors):
        url = f"{base}/{key}"
        try:
            data = _get(url, limit, timeout, opener)
        except Exception as e:  # noqa: BLE001 - one source, not the read
            logger.debug("No %s from %s: %s", key, base, type(e).__name__)
            continue
        if check(data):
            return data
        logger.warning("A source served bytes for %s that are not the ones named", key)
    return None


def fetch_menu_bytes(menu_hash: str, *, bucket: str | None, netuid: int, mirrors=None,
                     timeout: float = REQUEST_TIMEOUT_S,
                     opener: Callable | None = None) -> bytes | None:
    """The bytes of the menu with `menu_hash`, checked against it, from the
    root's bucket or a mirror; None when none serves them. The bytes are not
    parsed or judged here: that is menu_state.menu_for_epoch's."""
    return fetch_object(menu_object(netuid, menu_hash), bucket=bucket, mirrors=mirrors,
                        limit=MAX_MENU_FETCH_BYTES, timeout=timeout, opener=opener,
                        check=lambda b: hashlib.sha256(b).hexdigest() == menu_hash)


def fetch_join_bytes(delta_hash: str, *, bucket: str | None, netuid: int, mirrors=None,
                     timeout: float = REQUEST_TIMEOUT_S,
                     opener: Callable | None = None) -> bytes | None:
    """The bytes of the root's join delta with `delta_hash`, checked against
    it, from the root's bucket or a mirror; None when none serves them. Not
    parsed here: chain_proposals.check_delta judges them."""
    return fetch_object(join_object(netuid, delta_hash), bucket=bucket, mirrors=mirrors,
                        limit=MAX_JOIN_FETCH_BYTES, timeout=timeout, opener=opener,
                        check=lambda b: hashlib.sha256(b).hexdigest() == delta_hash)


def fetch_reveal_bytes(epoch_id: str, sha256: str, *, bucket: str | None, netuid: int,
                       mirrors=None, timeout: float = REQUEST_TIMEOUT_S,
                       opener: Callable | None = None) -> bytes | None:
    """The bytes of the root's reveal for `epoch_id` that hash to `sha256`
    (the root's commitment for it), from the root's bucket or a mirror;
    None when none serves them. Not parsed here."""
    if not _EPOCH_ID.fullmatch(str(epoch_id)) or not _SHA256.fullmatch(str(sha256)):
        raise ValueError("a reveal is named by its epoch id and its sha256")
    return fetch_object(f"{int(netuid)}/{epoch_id}/reveal.json", bucket=bucket, mirrors=mirrors,
                        limit=MAX_REVEAL_BYTES, timeout=timeout, opener=opener,
                        check=lambda b: hashlib.sha256(b).hexdigest() == sha256)


def fetch_signed_list(kind: str, sha256: str, *, bucket: str | None, netuid: int, mirrors=None,
                      timeout: float = REQUEST_TIMEOUT_S,
                      opener: Callable | None = None) -> tuple[bytes, bytes] | None:
    """(list bytes, signature) of the signed list of `kind` whose bytes hash
    to `sha256`; None when no source serves them. The signature is not
    checked here: the caller verifies both with menu_rules.verify_signed_list."""
    payload = fetch_object(list_object(netuid, kind, sha256), bucket=bucket, mirrors=mirrors,
                           limit=MAX_LIST_FETCH_BYTES, timeout=timeout, opener=opener,
                           check=lambda b: hashlib.sha256(b).hexdigest() == sha256)
    if payload is None:
        return None
    signature = fetch_object(signature_object(netuid, kind, sha256), bucket=bucket,
                             mirrors=mirrors, limit=MAX_SIGNATURE_FETCH_BYTES, timeout=timeout,
                             opener=opener, check=lambda b: len(b) > 0)
    return None if signature is None else (payload, signature)


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
    # The root's hotkey, whose on-chain reveal commitment every fetched reveal
    # is verified against (None: fetched unverified, or nothing fetched).
    root_hotkey: str | None = None

    def describe(self) -> str:
        if self.root:
            return f"ROOT ({self.why}): the other validators verify against this one's reveals"
        source = (f"gs://{self.bucket} (fetched by this validator)" if self.bucket
                  else "kept in sync by the operator")
        check = ""
        if self.bucket:
            check = (f", verified against the on-chain commitment of {self.root_hotkey}"
                     if self.root_hotkey else ", NOT verified against the chain (no root hotkey)")
        return f"follower of the reveals in {self.reference_dir}, {source}{check} ({self.why})"


def resolve_role(*, finney: bool, root_flag: bool | None, own_bucket: str, reference_bucket: str,
                 explicit_dir: str | None, state_path: str, netuid: int,
                 root_hotkey: str | None = None) -> Role | None:
    """Root or follower, from what is configured. None: neither (a chain with
    no reference configured, where a validator runs alone as before).

    - FUGAL_ROOT=1, or publishing to the reference bucket itself, is the root.
      FUGAL_ROOT=0 is never the root, whatever it publishes to (a root that
      lost its state follows another validator's reveals until it catches up).
    - FUGAL_FRAME_BOOTSTRAP names a directory the operator keeps in sync (the
      arrangement before this module, still honoured as it is).
    - Otherwise a reference bucket (finney's is built in; elsewhere
      FUGAL_REFERENCE_BUCKET) is fetched into a directory beside the state,
      verified against `root_hotkey`'s on-chain commitment when one is given
      (config.root_hotkey).
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
                    else "FUGAL_REFERENCE_BUCKET", root_hotkey=root_hotkey or None)
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
