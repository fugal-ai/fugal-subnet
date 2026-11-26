"""The signed protocol schedule (docs/decisions/SIGNED_SCHEDULE.md).

A switch whose code is in the image takes its block from a document the
owner signs, so a date can be set or moved with no new image or compose.

THE DOCUMENT. Canonical JSON (`canonical_bytes`): `activations` {name:
block}, `chain` (the genesis hash it is for), `format`, `kind`
("fugal-protocol-schedule") and `serial`. Signed with a release key under
the signed lists' namespace (menu_rules.LIST_NAMESPACE) and verified in
process against the pinned signers inside the image (`verify`); the `kind`
inside the signed bytes keeps a schedule and a list apart.

THE POINTER. The root's commitment carries `sched:<serial>:<digest>` in a
field after its reveal pointer (`Pointer.field`, `pointer_in_fields`). The
digest is sha256 of the document's bytes followed by its signature
(`bundle_digest`), so the pointer fixes both: whatever source serves them,
every node checks the same bytes and reaches the same verdict.
`P(e)` is that pointer read pinned at epoch e's boundary under the root's
hotkey for e (`read_pointer`).

THE RULE (`fold`). From the first epoch at or after SCHEDULE_FROM_BLOCK, for
each epoch e in order: an absent pointer, or the one in force, carries; a
pointer to bytes no source has stops the fold (UNRESOLVED); a document that
fails its signature, its format, the chain, its serial (it must exceed the
one in force: no replay) or the lead (every block it sets, moves or removes,
on both sides, must be at or after the boundary of e + LEAD) is refused and
the schedule in force carries; otherwise it is adopted at e. Every input is a
chain read at a named block, bytes checked against a hash and a signature,
or a compiled constant, so every node, validator or TD, folds the same
sequence or stops.

THE CONSUMER (`ScheduleTracker`, `ScheduleView`). A node keeps its fold on
disk and advances it at every boundary. `view.certain_through` is the last
epoch whose activations the fold decides (its last epoch + LEAD): a switch
consumer refuses an epoch beyond it. config.switch_block gives a switch's
block, a compiled one taking precedence.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

KIND = "fugal-protocol-schedule"
FORMAT = 1
# Serial 0 marks a draft: refused whatever signs it. A schedule starts at 1.
DRAFT_SERIAL = 0
MAX_SERIAL = 2**53 - 1
MAX_BLOCK = 2**53 - 1
MAX_DOCUMENT_BYTES = 16384
MAX_SIGNATURE_BYTES = 2048          # menu_rules.MAX_SIGNATURE_BYTES
MAX_ACTIVATIONS = 64
NAME_PATTERN = r"[a-z][a-z0-9_]{0,39}"
_NAME = re.compile(NAME_PATTERN)
_GENESIS = re.compile(r"0x[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# The commitment field: `sched:<serial>:<bundle digest>`, at most
# 6 + 16 + 1 + 64 = 87 bytes, inside the pallet's 128 per field.
POINTER_PREFIX = "sched:"
_POINTER = re.compile(r"sched:([1-9][0-9]{0,15}):([0-9a-f]{64})")
# Where a release ships signed schedules: <digest>.json and <digest>.json.sig.
SHIPPED_DIR = Path(__file__).resolve().parent / "signed_schedule"
STATE_FORMAT = 1


class ScheduleRefused(Exception):
    """A schedule document refused. `code`: oversized, malformed, duplicate,
    not_canonical, wrong_kind, draft, wrong_chain, serial_mismatch, rollback,
    not_boundary, too_soon (a block it sets or moves to is inside the lead),
    frozen (a block it moves or removes is inside the lead, or has passed),
    or a signature code of menu_rules.ListRefused."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


class ScheduleUnresolved(Exception):
    """The fold cannot go on: the root named a document no source serves
    (`epoch`, `pointer`), or the chain could not be read. Nothing after it
    is decided; `certain_through` epochs before it still are."""

    def __init__(self, message: str, *, epoch: int | None = None,
                 pointer: Pointer | None = None):
        super().__init__(message)
        self.epoch, self.pointer = epoch, pointer


class ScheduleConflict(Exception):
    """A release's compiled block and the schedule in force disagree."""


# ── the pointer ──────────────────────────────────────────────────────────────


def bundle_digest(payload: bytes, signature: bytes) -> str:
    """What a pointer names: sha256 of the document's bytes followed by its
    armored signature. A payload-only hash would let two sources serve the
    same document with different signatures, one valid and one not, and two
    nodes reach different verdicts on one pointer."""
    return hashlib.sha256(bytes(payload) + bytes(signature)).hexdigest()


@dataclass(frozen=True)
class Pointer:
    serial: int
    digest: str

    def field(self) -> str:
        return f"{POINTER_PREFIX}{self.serial}:{self.digest}"


def parse_pointer(value) -> Pointer | None:
    """A `sched:<serial>:<digest>` field, or None for anything else."""
    if not isinstance(value, str):
        return None
    m = _POINTER.fullmatch(value)
    if m is None or int(m.group(1)) > MAX_SERIAL:
        return None
    return Pointer(int(m.group(1)), m.group(2))


def pointer_in_fields(fields: Sequence[str]) -> Pointer | None:
    """The schedule pointer of a root commitment's fields: the first field
    after the reveal pointer (field 0) that starts with `sched:`, parsed;
    None when there is none or it is malformed."""
    for value in list(fields)[1:]:
        if isinstance(value, str) and value.startswith(POINTER_PREFIX):
            return parse_pointer(value)
    return None


def with_pointer(fields: Sequence[str], pointer: Pointer | None) -> list[str]:
    """`fields` (the root's reveal pointer, then its join pointer if any) with
    the schedule pointer as the last field, replacing any held one; at most
    three fields (the pallet's MaxFields)."""
    out = [f for f in fields if f and not str(f).startswith(POINTER_PREFIX)]
    if pointer is not None:
        out.append(pointer.field())
    if len(out) > 3:
        raise ValueError("a commitment holds at most three fields")
    return out


# ── the document ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Schedule:
    """A verified schedule (or EMPTY, serial 0: nothing scheduled)."""

    serial: int
    chain: str
    activations: tuple[tuple[str, int], ...]
    sha256: str = ""                 # of the document's bytes
    digest: str = ""                 # of the bytes and the signature (bundle_digest)
    signer: str | None = None

    def get(self, name: str) -> int | None:
        return dict(self.activations).get(name)

    def as_dict(self) -> dict[str, int]:
        return dict(self.activations)

    @property
    def pointer(self) -> Pointer | None:
        return Pointer(self.serial, self.digest) if self.serial > DRAFT_SERIAL else None


EMPTY = Schedule(serial=DRAFT_SERIAL, chain="", activations=())


def _is_int(value) -> bool:
    return type(value) is int


def canonical_bytes(serial: int, chain: str, activations: Mapping[str, int]) -> bytes:
    """The one byte string of a schedule: JSON with sorted keys, one space of
    indent, ASCII, a final newline (the signed lists' layout). Encoding only:
    `parse_unsigned` is what validates."""
    obj = {"activations": {name: int(activations[name]) for name in sorted(activations)},
           "chain": str(chain).lower(), "format": FORMAT, "kind": KIND, "serial": int(serial)}
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


class _Repeated(ValueError):
    pass


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise _Repeated(key)
        out[key] = value
    return out


def _no_number(text):
    raise ValueError(f"{text} is not an integer")


def parse_unsigned(payload: bytes) -> Schedule:
    """The format rules of a schedule, **with no signature check** (drafts,
    tools, tests). The neurons use `verify`."""
    if type(payload) is not bytes:
        raise ScheduleRefused("malformed", "a schedule is bytes")
    if len(payload) > MAX_DOCUMENT_BYTES:
        raise ScheduleRefused("oversized", f"{len(payload)} bytes, at most {MAX_DOCUMENT_BYTES}")
    try:
        obj = json.loads(payload.decode("ascii"), object_pairs_hook=_object,
                         parse_float=_no_number, parse_constant=_no_number)
    except _Repeated as e:
        raise ScheduleRefused("duplicate", f"key {str(e)[:40]!r} appears twice") from None
    except (ValueError, RecursionError):
        raise ScheduleRefused("malformed", "not ASCII JSON with integer numbers") from None
    if type(obj) is not dict:
        raise ScheduleRefused("malformed", "not a JSON object")
    if obj.get("kind") != KIND:
        raise ScheduleRefused("wrong_kind", f"kind {str(obj.get('kind'))[:60]!r}, expected {KIND!r}")
    expected = {"activations", "chain", "format", "kind", "serial"}
    if set(obj) != expected:
        raise ScheduleRefused("malformed", f"fields {sorted(obj)[:8]}, expected {sorted(expected)}")
    if not _is_int(obj["format"]) or obj["format"] != FORMAT:
        raise ScheduleRefused("malformed", f"format {obj['format']!r}, expected {FORMAT}")
    serial = obj["serial"]
    if not _is_int(serial) or not 0 <= serial <= MAX_SERIAL:
        raise ScheduleRefused("malformed", "serial is not an integer in 0..2^53-1")
    chain = obj["chain"]
    if not isinstance(chain, str) or not _GENESIS.fullmatch(chain):
        raise ScheduleRefused("malformed", "chain is not a genesis hash (0x and 64 lowercase hex)")
    acts = obj["activations"]
    if type(acts) is not dict:
        raise ScheduleRefused("malformed", "activations is not an object")
    if len(acts) > MAX_ACTIVATIONS:
        raise ScheduleRefused("oversized", f"{len(acts)} activations, at most {MAX_ACTIVATIONS}")
    for name, block in acts.items():
        if not _NAME.fullmatch(name):
            raise ScheduleRefused("malformed", f"switch name {name[:48]!r} is not {NAME_PATTERN}")
        if not _is_int(block) or not 0 < block <= MAX_BLOCK:
            raise ScheduleRefused("malformed", f"block of {name!r} is not an integer in 1..2^53-1")
    if payload != canonical_bytes(serial, chain, acts):
        raise ScheduleRefused("not_canonical", "the bytes are not the canonical encoding")
    return Schedule(serial=serial, chain=chain, activations=tuple(sorted(acts.items())),
                    sha256=hashlib.sha256(payload).hexdigest())


def verify(payload: bytes, signature: bytes, *, signers: bytes | None = None) -> Schedule:
    """The one way a schedule becomes trusted: the signature first (the
    signed lists' verifier, namespace and pinned signers: `signers` None is
    the shipped, pinned file inside the image), then the format, then serial
    1 or more. Returns the schedule with `signer` set; raises ScheduleRefused."""
    from fugal_subnet import menu_rules

    if type(payload) is not bytes or len(payload) > MAX_DOCUMENT_BYTES:
        raise ScheduleRefused("oversized" if type(payload) is bytes else "malformed",
                              f"a schedule is bytes, at most {MAX_DOCUMENT_BYTES}")
    try:
        signer = menu_rules.verify_list_signature(payload, signature, signers=signers)
    except menu_rules.ListRefused as e:
        raise ScheduleRefused(e.code, e.detail) from None
    parsed = parse_unsigned(payload)
    if parsed.serial <= DRAFT_SERIAL:
        raise ScheduleRefused("draft", "serial 0 marks a draft, not a schedule")
    return Schedule(serial=parsed.serial, chain=parsed.chain, activations=parsed.activations,
                    sha256=parsed.sha256, digest=bundle_digest(payload, signature),
                    signer=signer)


def check_transition(held: Schedule, new: Schedule, *, epoch: int, chain: str,
                     blocks_per_epoch: int, lead_epochs: int) -> None:
    """Whether `new`, first named at `epoch`'s boundary, may replace `held`.
    Raises ScheduleRefused (wrong_chain, rollback, not_boundary, too_soon,
    frozen). Pure: the verdict depends on the arguments alone."""
    bpe = int(blocks_per_epoch)
    if bpe < 1 or int(lead_epochs) < 1:
        raise ValueError("blocks_per_epoch and lead_epochs are positive")
    if new.chain != str(chain).lower():
        raise ScheduleRefused("wrong_chain", f"for chain {new.chain[:18]}..., not this one")
    if new.serial <= held.serial:
        raise ScheduleRefused("rollback", f"serial {new.serial} is not above the serial in "
                                          f"force, {held.serial}")
    for name, block in new.activations:
        if block % bpe:
            raise ScheduleRefused("not_boundary", f"{name} at block {block} is not an epoch "
                                                  f"boundary ({bpe} blocks an epoch)")
    floor = (int(epoch) + int(lead_epochs)) * bpe
    old, nxt = held.as_dict(), new.as_dict()
    for name in sorted(set(old) | set(nxt)):
        a, b = old.get(name), nxt.get(name)
        if a == b:
            continue
        if a is not None and a < floor:
            raise ScheduleRefused("frozen", f"{name} is at block {a}, before block {floor} "
                                            f"(epoch {epoch} + {lead_epochs}): it can no longer "
                                            "move or be removed")
        if b is not None and b < floor:
            raise ScheduleRefused("too_soon", f"{name} at block {b} is before block {floor} "
                                              f"(epoch {epoch} + {lead_epochs})")


# ── the fold ─────────────────────────────────────────────────────────────────


@dataclass
class FoldState:
    """Where a fold is: the schedule in force after epoch `next_epoch - 1`,
    its bytes and signature (kept so a restart re-verifies them), the epoch
    it was adopted, and what happened at each epoch that named something new."""

    start_epoch: int
    next_epoch: int
    in_force: Schedule = EMPTY
    payload: bytes = b""
    signature: bytes = b""
    adopted_epoch: int | None = None
    events: list = field(default_factory=list)       # [(epoch, field, outcome)]
    unresolved: tuple[int, str] | None = None        # (epoch, pointer field)

    @property
    def through(self) -> int:
        """The last epoch folded (start_epoch - 1 before any)."""
        return self.next_epoch - 1


Document = Callable[[Pointer], "tuple[bytes, bytes] | None"]
PointerAt = Callable[[int], "Pointer | None"]
_MAX_EVENTS = 256


def fold(state: FoldState, through_epoch: int, *, pointer_at: PointerAt, document: Document,
         chain: str, blocks_per_epoch: int, lead_epochs: int,
         signers: bytes | None = None) -> FoldState:
    """Advance `state` through `through_epoch` (inclusive), in place, and
    return it. Raises ScheduleUnresolved, with the state left at the epoch
    it could not decide, when the pointer cannot be read or names bytes no
    source serves; nothing past that epoch is folded."""
    state.unresolved = None
    for e in range(state.next_epoch, int(through_epoch) + 1):
        try:
            p = pointer_at(e)
        except Exception as ex:  # noqa: BLE001 - re-raised as "cannot decide e"
            raise ScheduleUnresolved(f"epoch {e}: the root's schedule pointer could not be "
                                     f"read ({type(ex).__name__})", epoch=e) from ex
        if p is None or p == state.in_force.pointer:
            state.next_epoch = e + 1
            continue
        got = document(p)
        if got is not None and bundle_digest(*got) != p.digest:
            got = None              # a source's error, never a verdict on the pointer
        if got is None:
            state.unresolved = (e, p.field())
            raise ScheduleUnresolved(f"epoch {e}: the root names schedule serial {p.serial} "
                                     f"({p.digest[:16]}...), which no source serves",
                                     epoch=e, pointer=p)
        payload, signature = got
        try:
            doc = verify(payload, signature, signers=signers)
            if doc.serial != p.serial:
                raise ScheduleRefused("serial_mismatch", f"the pointer says serial {p.serial}, "
                                                         f"the document {doc.serial}")
            check_transition(state.in_force, doc, epoch=e, chain=chain,
                             blocks_per_epoch=blocks_per_epoch, lead_epochs=lead_epochs)
        except ScheduleRefused as r:
            _event(state, e, p, f"refused:{r.code}")
            state.next_epoch = e + 1
            continue
        state.in_force, state.payload, state.signature = doc, bytes(payload), bytes(signature)
        state.adopted_epoch = e
        _event(state, e, p, "adopted")
        state.next_epoch = e + 1
    return state


def _event(state: FoldState, epoch: int, p: Pointer, outcome: str) -> None:
    last = state.events[-1] if state.events else None
    if last is not None and last[1] == p.field() and last[2] == outcome:
        return          # a refused pointer still named: logged once
    state.events.append((int(epoch), p.field(), outcome))
    del state.events[:-_MAX_EVENTS]


# ── persistence ──────────────────────────────────────────────────────────────


def _b64(data: bytes) -> str:
    return base64.b64encode(bytes(data)).decode("ascii")


def _unb64(text) -> bytes:
    return base64.b64decode(str(text), validate=True)


def state_dict(state: FoldState, params: Mapping) -> dict:
    return {"format": STATE_FORMAT, "params": dict(params), "start_epoch": state.start_epoch,
            "next_epoch": state.next_epoch, "adopted_epoch": state.adopted_epoch,
            "payload": _b64(state.payload), "signature": _b64(state.signature),
            "events": [list(ev) for ev in state.events]}


def state_from_dict(d: Mapping, params: Mapping, *, signers: bytes | None = None) -> FoldState | None:
    """The fold kept on disk, re-verified; None when it is for other
    parameters, damaged, or its schedule no longer verifies (the fold then
    starts again from the beginning, which gives the same answer)."""
    try:
        if d.get("format") != STATE_FORMAT or dict(d.get("params") or {}) != dict(params):
            return None
        state = FoldState(start_epoch=int(d["start_epoch"]), next_epoch=int(d["next_epoch"]),
                          adopted_epoch=d.get("adopted_epoch"),
                          events=[tuple(ev) for ev in d.get("events") or ()])
        payload, signature = _unb64(d.get("payload", "")), _unb64(d.get("signature", ""))
        if state.next_epoch < state.start_epoch:
            return None
        if payload:
            state.in_force = verify(payload, signature, signers=signers)
            state.payload, state.signature = payload, signature
            if state.adopted_epoch is None:
                return None
            state.adopted_epoch = int(state.adopted_epoch)
        elif state.adopted_epoch is not None:
            return None
        return state
    except (KeyError, TypeError, ValueError, binascii.Error, ScheduleRefused):
        return None


def _write_atomic(path: str, data: bytes) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".schedule.")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


# ── the node's view ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScheduleView:
    """The schedule a node holds after folding through `through`."""

    in_force: Schedule
    through: int
    lead_epochs: int
    blocks_per_epoch: int
    adopted_epoch: int | None = None
    unresolved: tuple[int, str] | None = None

    @property
    def certain_through(self) -> int:
        """The last epoch whose activations this view decides: a document
        named at through + 1 or later moves nothing before through + 1 + LEAD."""
        return self.through + self.lead_epochs

    def block(self, name: str) -> int | None:
        return self.in_force.get(name)

    def active(self, name: str, epoch: int) -> bool:
        """Whether switch `name` applies to `epoch` (its boundary is at or
        after the block). ScheduleUnresolved past `certain_through`."""
        if int(epoch) > self.certain_through:
            raise ScheduleUnresolved(f"epoch {epoch} is past the last epoch this schedule "
                                     f"decides ({self.certain_through})", epoch=int(epoch))
        b = self.block(name)
        return b is not None and int(epoch) * self.blocks_per_epoch >= b


class ScheduleTracker:
    """A node's fold, kept at `state_path` (None: memory only) and advanced at
    each boundary by `advance(epoch)`. `pointer_at(e)` reads P(e);
    `document(pointer)` returns (bytes, signature) or None. The same class
    runs on a validator and in a TD."""

    def __init__(self, *, chain: str, start_epoch: int, blocks_per_epoch: int,
                 lead_epochs: int, pointer_at: PointerAt, document: Document,
                 state_path: str | None = None, signers: bytes | None = None):
        self.chain = str(chain).lower()
        self.blocks_per_epoch, self.lead_epochs = int(blocks_per_epoch), int(lead_epochs)
        self.pointer_at, self.document = pointer_at, document
        self.state_path, self.signers = state_path, signers
        self.params = {"chain": self.chain, "start_epoch": int(start_epoch),
                       "blocks_per_epoch": self.blocks_per_epoch, "lead_epochs": self.lead_epochs,
                       "kind": KIND, "format": FORMAT}
        self.state = self._load() or FoldState(start_epoch=int(start_epoch),
                                               next_epoch=int(start_epoch))

    def _load(self) -> FoldState | None:
        if not self.state_path or not os.path.exists(self.state_path):
            return None
        try:
            with open(self.state_path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            return None
        got = state_from_dict(d, self.params, signers=self.signers)
        if got is None:
            logger.warning("Schedule state at %s is for other parameters or does not verify; "
                           "folding again from epoch %d", self.state_path,
                           self.params["start_epoch"])
        return got

    def _save(self) -> None:
        if self.state_path:
            _write_atomic(self.state_path, json.dumps(
                state_dict(self.state, self.params), sort_keys=True).encode("utf-8"))

    @property
    def view(self) -> ScheduleView:
        s = self.state
        return ScheduleView(in_force=s.in_force, through=s.through,
                            lead_epochs=self.lead_epochs, blocks_per_epoch=self.blocks_per_epoch,
                            adopted_epoch=s.adopted_epoch, unresolved=s.unresolved)

    def advance(self, epoch: int) -> ScheduleView:
        """Fold through `epoch` and keep the result. Raises ScheduleUnresolved
        (the state kept where it stopped); `view` then still decides every
        epoch through its certain_through."""
        before = (self.state.in_force.digest, len(self.state.events))
        try:
            fold(self.state, int(epoch), pointer_at=self.pointer_at, document=self.document,
                 chain=self.chain, blocks_per_epoch=self.blocks_per_epoch,
                 lead_epochs=self.lead_epochs, signers=self.signers)
        finally:
            self._save()
            for ev in self.state.events[before[1]:]:
                (logger.info if ev[2] == "adopted" else logger.warning)(
                    "Protocol schedule: epoch %d, %s: %s", ev[0], ev[1][:30], ev[2])
            if self.state.in_force.digest != before[0]:
                logger.info("Protocol schedule in force: serial %d, %s",
                            self.state.in_force.serial, describe(self.state.in_force))
        return self.view


def describe(schedule: Schedule) -> str:
    acts = ", ".join(f"{n} at {b:,}" for n, b in schedule.activations)
    return acts or "nothing scheduled"


# ── sources of bytes, and the chain read ─────────────────────────────────────


def shipped_documents(directory: Path | None = None) -> dict[str, tuple[bytes, bytes]]:
    """{digest: (bytes, signature)} of the schedules a release ships:
    `<digest>.json` with `<digest>.json.sig` (bundle_digest). A pair that
    does not hash to its name, or a file without its signature, is left out
    (the safety check refuses a release that has one)."""
    folder = SHIPPED_DIR if directory is None else Path(directory)
    out: dict[str, tuple[bytes, bytes]] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.json")):
        digest = path.stem
        sig = path.with_name(path.name + ".sig")
        if not _SHA256.fullmatch(digest) or not sig.is_file():
            continue
        payload, signature = path.read_bytes(), sig.read_bytes()
        if (len(payload) <= MAX_DOCUMENT_BYTES and len(signature) <= MAX_SIGNATURE_BYTES
                and bundle_digest(payload, signature) == digest):
            out[digest] = (payload, signature)
    return out


class DocumentSource:
    """`document(pointer)` for a node: the shipped schedules, then the node's
    cache directory, then `fetch(digest)` (the public store), whose answer is
    kept in the cache. Every answer is checked against the digest, which
    covers the signature; the signature's validity is the fold's verdict."""

    def __init__(self, *, directories: Iterable[Path] = (), cache_dir: str | None = None,
                 fetch: Callable[[str], "tuple[bytes, bytes] | None"] | None = None):
        self.directories = [Path(d) for d in directories]
        self.cache_dir, self.fetch = cache_dir, fetch
        self._held: dict[str, tuple[bytes, bytes]] = {}

    def _local(self, digest: str) -> tuple[bytes, bytes] | None:
        for folder in self.directories + ([Path(self.cache_dir)] if self.cache_dir else []):
            got = shipped_documents(folder).get(digest)
            if got is not None:
                return got
        return None

    def __call__(self, pointer: Pointer) -> tuple[bytes, bytes] | None:
        digest = pointer.digest
        if digest in self._held:
            return self._held[digest]
        got = self._local(digest)
        if got is None and self.fetch is not None:
            try:
                fetched = self.fetch(digest)
            except Exception as e:  # noqa: BLE001 - one source; the fold reports UNRESOLVED
                logger.warning("Schedule %s...: fetch failed (%s)", digest[:16],
                               type(e).__name__)
                fetched = None
            if fetched is not None and bundle_digest(*fetched) == digest:
                got = (bytes(fetched[0]), bytes(fetched[1]))
                if self.cache_dir:
                    _write_atomic(os.path.join(self.cache_dir, f"{digest}.json"), got[0])
                    _write_atomic(os.path.join(self.cache_dir, f"{digest}.json.sig"), got[1])
        if got is not None:
            self._held[digest] = got
        return got


def store_fetch(*, bucket: str | None, netuid: int, mirrors=None):
    """fetch(digest) from the public store, beside the signed lists
    (`store_object`: `<netuid>/lists/fugal-protocol-schedule/<digest>.json`
    and `.sig`), each part read to its bound; the caller checks the digest."""
    from fugal_subnet import reveal_sync

    def fetch(digest: str):
        payload = reveal_sync.fetch_object(store_object(netuid, digest), bucket=bucket,
                                           mirrors=mirrors, limit=MAX_DOCUMENT_BYTES,
                                           check=lambda b: len(b) > 0)
        if payload is None:
            return None
        signature = reveal_sync.fetch_object(
            store_object(netuid, digest) + ".sig", bucket=bucket, mirrors=mirrors,
            limit=MAX_SIGNATURE_BYTES, check=lambda s: bundle_digest(payload, s) == digest)
        return None if signature is None else (payload, signature)
    return fetch


def store_object(netuid: int, digest: str) -> str:
    """The store key of a schedule's bytes, by its bundle digest (its
    signature: the same key + '.sig')."""
    from fugal_subnet import reveal_sync
    return reveal_sync.list_object(netuid, KIND, digest)


def read_pointer(subtensor, netuid: int, root_hotkeys, epoch: int,
                 blocks_per_epoch: int) -> Pointer | None:
    """P(epoch): the schedule pointer in the root's commitment read pinned at
    the epoch's boundary block, under the root's hotkey for that epoch
    (menu_publish.hotkey_for_epoch). Raises when the chain cannot be read."""
    from fugal_subnet import chain_proposals, chain_reads
    from fugal_subnet.menu_publish import hotkey_for_epoch

    hotkey = hotkey_for_epoch(root_hotkeys, int(epoch))
    block = int(epoch) * int(blocks_per_epoch)
    raw = chain_reads.pinned(
        subtensor, block, lambda s: s.query_module("Commitments", "CommitmentOf",
                                                   params=[int(netuid), hotkey], block=block),
        what="Schedule pointer")
    return pointer_in_fields(chain_proposals.commitment_fields(raw))


def first_epoch(from_block: int, blocks_per_epoch: int) -> int:
    """The first epoch whose boundary is at or after `from_block`."""
    bpe = int(blocks_per_epoch)
    return -(-int(from_block) // bpe)


def root_candidate(in_force: Schedule, documents: Mapping[str, tuple[bytes, bytes]], *,
                   epoch: int, chain: str, blocks_per_epoch: int, lead_epochs: int,
                   signers: bytes | None = None) -> Schedule | None:
    """The document the root should name so that it is first seen at
    `epoch`'s boundary: of those it holds, the highest serial that verifies
    and may replace `in_force` there (check_transition). None when none may."""
    best = None
    for payload, signature in documents.values():
        try:
            doc = verify(payload, signature, signers=signers)
            check_transition(in_force, doc, epoch=epoch, chain=chain,
                             blocks_per_epoch=blocks_per_epoch, lead_epochs=lead_epochs)
        except ScheduleRefused:
            continue
        if best is None or doc.serial > best.serial:
            best = doc
    return best


# ── a node's wiring ──────────────────────────────────────────────────────────


def node_tracker(*, subtensor, netuid: int, genesis: str, root_hotkeys, blocks_per_epoch: int,
                 from_block: int, state_path: str | None, cache_dir: str | None,
                 bucket: str | None, lead_epochs: int | None = None,
                 signers: bytes | None = None) -> ScheduleTracker:
    """The tracker a validator or a TD runs: P(e) read pinned at each
    boundary (`read_pointer`), the bytes from the release, the node's cache
    or the public store (`DocumentSource`), the fold kept at `state_path`."""
    from fugal_subnet import config

    bpe = int(blocks_per_epoch)
    if int(from_block) % bpe:
        raise ValueError(f"the schedule's start block {from_block} is not an epoch boundary "
                         f"({bpe} blocks an epoch)")
    if not genesis:
        raise ValueError("the protocol schedule is for one chain: its genesis hash is unknown")
    source = DocumentSource(directories=[SHIPPED_DIR], cache_dir=cache_dir,
                            fetch=store_fetch(bucket=bucket, netuid=netuid))
    return ScheduleTracker(
        chain=genesis, start_epoch=first_epoch(from_block, bpe), blocks_per_epoch=bpe,
        lead_epochs=config.SCHEDULE_LEAD_EPOCHS if lead_epochs is None else int(lead_epochs),
        pointer_at=lambda e: read_pointer(subtensor, netuid, root_hotkeys, e, bpe),
        document=source, state_path=state_path, signers=signers)


def advance_quietly(tracker: ScheduleTracker | None, epoch: int) -> ScheduleView | None:
    """`tracker.advance(epoch)`, logging instead of raising: for a node that
    has no schedulable switch to decide yet. A switch's consumer calls
    `advance` and refuses an epoch past the view's `certain_through`."""
    if tracker is None:
        return None
    try:
        return tracker.advance(epoch)
    except ScheduleUnresolved as e:
        view = tracker.view
        logger.error("Protocol schedule: %s; the schedule held decides epochs through %d",
                     e, view.certain_through)
        return view
    except Exception as e:  # noqa: BLE001 - nothing reads the schedule yet: logged
        logger.error("Protocol schedule: the fold failed (%s: %s)", type(e).__name__, e)
        return tracker.view


def root_pointer(tracker: ScheduleTracker, *, epoch: int,
                 documents: Mapping[str, tuple[bytes, bytes]],
                 confirm: Callable[[str], bool]) -> Pointer | None:
    """What the root writes in its commitment during `epoch` (after folding
    through it): the newest document it holds that may replace the one in
    force even if first seen one epoch late (at epoch + 2's boundary) and
    that the public store serves back (`confirm(digest)`), else the pointer
    in force (naming it again changes nothing), else None."""
    in_force = tracker.state.in_force
    if int(epoch) + 1 < tracker.params["start_epoch"]:
        return None
    cand = root_candidate(in_force, documents, epoch=int(epoch) + 2, chain=tracker.chain,
                          blocks_per_epoch=tracker.blocks_per_epoch,
                          lead_epochs=tracker.lead_epochs, signers=tracker.signers)
    if cand is not None and cand.digest != in_force.digest:
        try:
            served = bool(confirm(cand.digest))
        except Exception:  # noqa: BLE001 - not confirmed: not named
            served = False
        if served:
            return cand.pointer
        logger.warning("Protocol schedule serial %d is not served by the public store "
                       "(<netuid>/lists/%s/%s.json and .sig): not naming it", cand.serial, KIND,
                       cand.digest)
    return in_force.pointer
