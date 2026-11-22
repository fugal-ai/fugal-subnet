"""Protocol v4: how the menu moves between nodes (docs/decisions/LIVE_MENU_V4.md, 4.3, 12, 15).

`menu_state` decides what a menu is, how one follows from the last
(`step_detail`) and which one an epoch runs on (`menu_for_epoch`). This
module carries menus from the node that derives them to every node that runs
on them:

  THE ROOT, scoring epoch k, steps from MenuState(k + MENU_LAG - 1) to
    MenuState(k + MENU_LAG) and writes the result into reveal k as
    `menu_next`, with the step's events (`root_reveal_fields`). When reveal k
    is published, `MenuPublisher` uploads the menu bytes, and any signed list
    the menu newly names, to the public store, reads each back, and only then
    hands the menu hash to the reveal commitment
    (reveal_publisher.publish_due_reveals): publish before commit. A failure
    anywhere leaves the hash out of that commitment, so the previous menu
    carries; it never costs the reveal, its commitment, the epoch or the
    weights.
  EVERY NODE, at the boundary of epoch E, reads the root's commitment for
    epoch E - MENU_LAG pinned at E's boundary block (`read_menu_commitment`),
    finds bytes for a changed hash (its own copy, the store, a mirror) and
    the signed list they name, and calls `menu_for_epoch`
    (`MenuNode.at_boundary`, `read_epoch_menu`). No commitment: carried. A
    refused menu: carried, with an alarm. No bytes or no list: no menu, and
    the node says it cannot verify the epoch's proofs.
  THE OTHER VALIDATORS run the same step on their own verified proofs and
    publish how the result differs from the root's (`menu_audit`), then run
    on the root's like everyone.
  A NODE THAT MISSED EPOCHS walks the root's commitments forward from the
    last menu it holds, or from the pinned seed (`rebuild`), each link
    through `menu_for_epoch`. A link it cannot read stops the walk there.
  THE JOINS (FAST_CROWN_V4.md 18): at every boundary, after the base menu,
    the root's join delta named by its second commitment field read at the
    same block (`read_join_pointer`) is checked and applied
    (`read_epoch_menu`, `chain_proposals.check_delta`); the root writes both
    fields together (`RootFields`).

The commitment is `reveal:<epoch_id>:<reveal sha256>:<menu hash token>`
(reveal_publisher.commitment_for); the store layout is in reveal_sync. Nothing
here is reachable before a v4 activation block, and nothing on the v1-v3 path
imports this module.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from fugal_subnet import reveal_sync
from fugal_subnet.benchmarks.slicer import epoch_id_for_block
from fugal_subnet.reveal_publisher import (
    COMMITMENT_MAX_BYTES,
    commitment_for,
    epoch_index_of,
    read_marker,
)
from fugal_subnet.v2_sync import MENU_FIELD, MENU_ROLE_FIELDS, MENU_SUMMARY_FIELD

logger = logging.getLogger(__name__)

# The reveal fields this module writes (v2_sync compares on MENU_FIELD and
# leaves the role fields out of the comparison).
REVEAL_NEXT, REVEAL_NEXT_HASH, REVEAL_EVENTS, REVEAL_LISTS, REVEAL_AUDIT = MENU_ROLE_FIELDS
REVEAL_SUMMARY = MENU_SUMMARY_FIELD

LIST_KINDS = (mr.KIND_MENU_ALLOWLIST, mr.KIND_DEVICE_LIST)
# One publication's uploads and read-backs together, in seconds: past it the
# hash stays out of the commitment and the menu carries. Every single call is
# bounded by the store's own timeout as well.
PUBLISH_BUDGET_S = 120.0
# A rebuild reads at most this many links (one pinned chain read each) per
# call; the next boundary goes on from where it stopped.
REBUILD_MAX_LINKS = 64
# How many differences an audit lists before it says how many more there are.
MAX_AUDIT_DIFFERENCES = 64
# Menus a node remembers by epoch.
MAX_HELD_EPOCHS = 1024


# The step inputs of a scored epoch k, or how to build them from the menu the
# step starts from: menu_inputs.step_inputs(start, MenuState(k), window,
# proofs_k, ...) needs MenuState(k + 1), which the root knows only once it has
# published reveal k - 1 and another validator only at the boundary of k + 1,
# so a validator hands in a function of it. Tests hand in the value itself.
Inputs = ms.StepInputs | Callable[[ms.MenuState], ms.StepInputs]


def _inputs_for(inputs: Inputs, start: ms.MenuState) -> ms.StepInputs:
    return inputs if isinstance(inputs, ms.StepInputs) else inputs(start)


# --- startup -------------------------------------------------------------------------


def startup_refusals(*, chain_hold: int, reveal_delay: int, root: bool, root_hotkey: str | None,
                     epoch_id: str, lag: int = ms.MENU_LAG) -> list[str]:
    """Why a validator must not run protocol v4 epochs, or [] (section 4.3).

    - The root's reveal k must be public, and its commitment on chain, before
      the boundary of k + lag: a hold longer than lag - 1 epochs (the chain's
      commit-reveal period, `reveal_publisher.delay_epochs_for`) means the
      commitment for k can never be read in time, so no menu would ever
      change. The root's own publishing delay is held to the same bound.
    - Without the root's hotkey no node can read the commitment, and would
      carry every menu while its peers moved on.
    - The commitment with its menu hash must fit one Commitments field
      (`COMMITMENT_MAX_BYTES`), for this epoch's id and for any epoch id the
      next 10^8 epochs bring.
    """
    out = []
    if chain_hold > lag - 1:
        out.append(f"the chain holds reveals for {chain_hold} epochs (its commit-reveal "
                   f"period); with a menu lag of {lag} the root's commitment for epoch k must be "
                   f"on chain before the boundary of k + {lag}, which needs a hold of at most "
                   f"{lag - 1}")
    if root and reveal_delay > lag - 1:
        out.append(f"this root publishes reveals {reveal_delay} epochs late "
                   f"(FUGAL_REVEAL_DELAY_EPOCHS); protocol v4 needs at most {lag - 1}")
    if not root_hotkey:
        out.append("no root hotkey is known, so the root's menu commitment cannot be read")
    for eid in (epoch_id, epoch_id_for_block(10**8 - 1)):
        size = len(commitment_for(eid, "f" * 64, "f" * 64).encode())
        if size > COMMITMENT_MAX_BYTES:
            out.append(f"a menu commitment for {eid} is {size} bytes, more than the "
                       f"{COMMITMENT_MAX_BYTES} one Commitments field holds")
    return out


# --- the chain read -------------------------------------------------------------------


def menu_hash_in_commitment(data, epoch: int, lag: int = ms.MENU_LAG) -> str | None:
    """The menu hash a root commitment names for epoch `epoch`: present only
    when the commitment is the root's reveal commitment for epoch
    `epoch - lag` and carries a menu field. Anything else (no commitment, an
    older or newer epoch's, one without a menu hash, one that does not parse)
    is no commitment for this epoch, and the previous menu carries."""
    if int(epoch) - lag < 0:
        return None
    got = reveal_sync.parse_commitment(data)
    if got is None:
        return None
    committed_epoch, _sha, menu_hash = got
    return menu_hash if committed_epoch == epoch_id_for_block(int(epoch) - lag) else None


def hotkey_for_epoch(hotkeys, epoch: int) -> str:
    """The root hotkey whose commitment names the menu of `epoch`: `hotkeys`
    is one hotkey for every epoch, or a table ((first epoch, hotkey), ...)
    sorted by first epoch (config.root_hotkeys_v4), of which the last row
    at or before `epoch` holds. ValueError for an epoch no row covers: the
    read cannot be made, and a walk stops there."""
    if isinstance(hotkeys, str):
        return hotkeys
    rows = [hk for first, hk in hotkeys if int(first) <= int(epoch)]
    if not rows:
        raise ValueError(f"no root hotkey covers epoch {int(epoch)}")
    return rows[-1]


def read_menu_commitment(subtensor, netuid: int, root_hotkey, epoch: int,
                         boundary_block: int) -> str | None:
    """The menu hash the root committed for `epoch`, read pinned at the
    epoch's boundary block, as head commitments are read: chain state at a
    named block is the same for every node. `root_hotkey` is the root's
    hotkey, or the table of its hotkeys by epoch (`hotkey_for_epoch`), so
    each epoch is read under the hotkey that was the root's then. Raises
    when the read fails, which is not the same as "no commitment" (that
    would carry on this node while its peers changed menu)."""
    from fugal_subnet.commitments import get_commitments_with_blocks

    hotkey = hotkey_for_epoch(root_hotkey, epoch)
    data, _block = get_commitments_with_blocks(subtensor, netuid, block=int(boundary_block)).get(
        hotkey, ("", 0))
    return menu_hash_in_commitment(data, epoch)


def read_join_pointer(subtensor, netuid: int, root_hotkey, epoch: int, boundary_block: int):
    """The root's join pointer for `epoch` (chain_proposals.JoinPointer), its
    commitment's second field read pinned at the epoch's boundary block under
    the root hotkey in force for the epoch; None when that field names
    nothing for the epoch (a late or absent root: no joins). Raises when the
    chain cannot be read."""
    from fugal_subnet import chain_proposals as cp

    hotkey = hotkey_for_epoch(root_hotkey, epoch)
    fields = cp.read_fields_at(subtensor, netuid, int(boundary_block)).get(hotkey, ())
    return cp.parse_join_field(fields[1] if len(fields) >= 2 else None, epoch)


def reveal_sha_in_commitment(data, k: int) -> str | None:
    """The sha256 of reveal k that a root commitment names, when it is the
    root's reveal commitment for epoch k (with or without a menu hash)."""
    got = reveal_sync.parse_commitment(data)
    if got is None or got[0] != epoch_id_for_block(int(k)):
        return None
    return got[1]


def read_reveal_commitment(subtensor, netuid: int, root_hotkey, k: int,
                           block: int) -> str | None:
    """The sha256 of reveal k the root committed, read pinned at `block`
    (the boundary of k + MENU_LAG, where the commitment for k is the one the
    menu of k + MENU_LAG is read from), under the root hotkey in force for
    the menu of k + MENU_LAG (`hotkey_for_epoch`). Raises when the read fails."""
    from fugal_subnet.commitments import get_commitments_with_blocks

    hotkey = hotkey_for_epoch(root_hotkey, int(k) + ms.MENU_LAG)
    data, _block = get_commitments_with_blocks(subtensor, netuid, block=int(block)).get(
        hotkey, ("", 0))
    return reveal_sha_in_commitment(data, k)


def anchor_from_reveal(raw: bytes, epoch_id: str, sha256: str
                       ) -> tuple[int, str, int, str] | None:
    """(k, the hash of the menu epoch k ran on, its source's epoch, its
    source's hash) from the bytes of the root's reveal k: only when they
    hash to `sha256` (the root's commitment for it), are reveal
    `epoch_id`'s, and carry a `menu` field for that epoch that
    `check_menu_field` accepts ({epoch, hash, source: {epoch, hash}}). None
    otherwise: a reveal whose menu field is malformed is not anchored on. A
    TD that starts or has fallen behind anchors on it (td_v4.V4Session)
    rather than replay every menu."""
    try:
        if hashlib.sha256(bytes(raw)).hexdigest() != sha256:
            return None
        reveal = json.loads(bytes(raw))
        if not isinstance(reveal, dict) or reveal.get("epoch_id") != epoch_id:
            return None
        return check_menu_field(reveal.get(MENU_FIELD), epoch_id)
    except (ValueError, TypeError, AttributeError):
        return None


def read_root_anchor(subtensor, netuid: int, root_hotkey, epoch: int, boundary_block: int, *,
                     fetch_reveal: Callable[[str, str], bytes | None]) -> tuple[int, str] | None:
    """The newest root reveal that is published and verified against the
    root's on-chain commitment read pinned at `boundary_block` (the boundary
    of `epoch`, a recent block), as (k, the hash of the menu epoch k ran
    on): `anchor_from_reveal` of the bytes `fetch_reveal(epoch_id, sha256)`
    serves. None when there is no commitment or no such reveal is to be
    had. Raises when the chain cannot be read."""
    from fugal_subnet.commitments import get_commitments_with_blocks

    hotkey = hotkey_for_epoch(root_hotkey, epoch)
    data, _block = get_commitments_with_blocks(subtensor, netuid, block=int(boundary_block)).get(
        hotkey, ("", 0))
    got = reveal_sync.parse_commitment(data)
    if got is None:
        return None
    epoch_id, sha, _menu = got
    raw = fetch_reveal(epoch_id, sha)
    return None if raw is None else anchor_from_reveal(raw, epoch_id, sha)


# The most bytes read of one reveal when taking its summary: the bound every
# reader of a reveal uses (reveal_sync.MAX_REVEAL_BYTES), so a reveal the
# follower sync fetches is one whose summary can be read.
MAX_REVEAL_READ_BYTES = reveal_sync.MAX_REVEAL_BYTES


def root_summaries(directory: str, sha_of: Callable[[int], str | None] | None, *,
                   check: Callable[[dict], object]) -> Callable[[int], dict | None]:
    """A source of the root's published epoch summaries, for
    `MenuBook.fill_from_root`: epoch k -> the `menu_summary` of the root's
    reveal k in `directory/<epoch id>/reveal.json`, or None.

    `sha_of(k)` is the sha256 the root committed for reveal k (follower sync's
    pointer ledger, else the chain); a reveal is read only when its bytes
    hash to it, as follower sync verifies reveals. `sha_of` None is the root
    reading its own reveals. `check(doc)` parses the summary
    (menu_inputs.summary_from_dict, which this module does not import) and
    the summary must be of epoch k. Never raises."""
    def get(k: int) -> dict | None:
        eid = epoch_id_for_block(int(k))
        try:
            expect = None if sha_of is None else sha_of(int(k))
            if sha_of is not None and expect is None:
                return None
            with open(os.path.join(directory, eid, "reveal.json"), "rb") as f:
                raw = f.read(MAX_REVEAL_READ_BYTES + 1)
            if len(raw) > MAX_REVEAL_READ_BYTES:
                return None
            if expect is not None and hashlib.sha256(raw).hexdigest() != expect:
                logger.warning("The root's reveal %s on disk is not the one it committed; its "
                               "summary is not used", eid)
                return None
            reveal = json.loads(raw)
            doc = reveal.get(REVEAL_SUMMARY) if reveal.get("epoch_id") == eid else None
            if not isinstance(doc, dict) or getattr(check(doc), "epoch", None) != int(k):
                return None
            return doc
        except Exception:  # noqa: BLE001 - no summary for this epoch; the window lacks it
            return None
    return get


# --- signed lists a node holds ------------------------------------------------------


class ListBook:
    """The signed lists this node holds, each verified with
    `menu_rules.verify_signed_list`, with the bytes and signature it was
    verified from (so a root can publish them): the lists this release ships,
    any the operator places in `directory/<kind>/` as `<name>.json` with
    `<name>.json.sig`, and any fetched (`fetch(kind, sha256) -> (bytes,
    signature) | None`) because a menu names them."""

    def __init__(self, directory: str | None = None, *,
                 fetch: Callable[[str, str], tuple[bytes, bytes] | None] | None = None,
                 signers: bytes | None = None, shipped: bool = True):
        self.directory = directory
        self.fetch = fetch
        self.signers = signers
        self._held: dict[tuple[str, str], tuple[object, bytes, bytes]] = {}
        if shipped:
            for kind in LIST_KINDS:
                name = mr.SIGNED_LISTS_DIR / mr.SHIPPED_LISTS[kind]
                try:
                    payload, signature = name.read_bytes(), (name.parent / f"{name.name}.sig").read_bytes()
                except OSError:
                    continue
                self._add(kind, payload, signature, save=False)
        if directory:
            for kind in LIST_KINDS:
                folder = os.path.join(directory, kind)
                try:
                    names = sorted(os.listdir(folder))
                except OSError:
                    continue
                for n in names:
                    if not n.endswith(".json"):
                        continue
                    try:
                        with open(os.path.join(folder, n), "rb") as f:
                            payload = f.read(mr.MAX_LIST_BYTES + 1)
                        with open(os.path.join(folder, n + ".sig"), "rb") as f:
                            signature = f.read(mr.MAX_SIGNATURE_BYTES + 1)
                    except OSError:
                        continue
                    self._add(kind, payload, signature, save=False)

    def _add(self, kind: str, payload: bytes, signature: bytes, *, save: bool):
        try:
            got = mr.verify_signed_list(payload, signature, kind=kind, held_serial=0,
                                        signers=self.signers)
        except (mr.ListRefused, ValueError) as e:
            logger.error("A %s offered to this node does not verify (%s); not held", kind, e)
            return None
        self._held[(kind, got.sha256)] = (got, payload, signature)
        if save and self.directory:
            folder = os.path.join(self.directory, kind)
            try:
                os.makedirs(folder, exist_ok=True)
                for name, data in ((f"{got.sha256}.json", payload),
                                   (f"{got.sha256}.json.sig", signature)):
                    tmp = os.path.join(folder, name + ".tmp")
                    with open(tmp, "wb") as f:
                        f.write(data)
                    os.replace(tmp, os.path.join(folder, name))
            except OSError as e:
                logger.warning("Could not keep a copy of %s serial %d (%s)", kind, got.serial,
                               type(e).__name__)
        return got

    def add(self, kind: str, payload: bytes, signature: bytes):
        """Verify and hold a list; the list, or None when it does not verify."""
        return self._add(kind, payload, signature, save=True)

    def get(self, kind: str, ref: ms.AllowlistRef):
        """The verified list `ref` names, fetched when not held; None when the
        reference names none, or no source serves a list that verifies and
        is that list."""
        if ref.serial == 0:
            return None
        held = self._held.get((kind, ref.sha256))
        if held is None and self.fetch is not None:
            try:
                offered = self.fetch(kind, ref.sha256)
            except Exception as e:  # noqa: BLE001 - a source that fails offers nothing
                logger.warning("Fetching %s serial %d failed (%s)", kind, ref.serial,
                               type(e).__name__)
                offered = None
            if offered is not None:
                self._add(kind, offered[0], offered[1], save=True)
            held = self._held.get((kind, ref.sha256))
        if held is None or held[0].serial != ref.serial:
            return None
        return held[0]

    def raw(self, kind: str, sha256: str) -> tuple[bytes, bytes] | None:
        held = self._held.get((kind, sha256))
        return None if held is None else (held[1], held[2])

    def newest(self, kind: str):
        """The held list of `kind` with the highest serial: what a node hands
        the step (the one in force or a newer one)."""
        lists = [v[0] for (k, _), v in self._held.items() if k == kind]
        return max(lists, key=lambda x: x.serial) if lists else None


def _named_lists(previous: ms.MenuState, raw: bytes | None, lists: ListBook):
    """The lists the menu being read names: the lab and host list of the
    offered bytes when they parse (else the one in force), and a device list
    only when the bytes name one other than the one in force."""
    labs, devices = previous.labs, None
    if raw is not None and len(raw) <= ms.MENU_MAX_BYTES:      # larger bytes are refused unread
        try:
            parsed = ms.from_json(bytes(raw))
        except ValueError:
            parsed = None
        if parsed is not None:
            labs = parsed.labs
            if parsed.devices != previous.devices:
                devices = parsed.devices
    return (lists.get(mr.KIND_MENU_ALLOWLIST, labs),
            lists.get(mr.KIND_DEVICE_LIST, devices) if devices is not None else None)


# --- reading an epoch's menu ------------------------------------------------------


@dataclass(frozen=True)
class MenuRead:
    """What a node holds for one epoch.

    status: "seed" (the first v4 epoch), "carried" (no commitment for the
    epoch), "committed" (the committed menu, changed or not), "refused" (a
    committed menu every node refuses: the carried one runs, with an alarm),
    "unavailable" (no menu: the node cannot check the committed one, or
    cannot reach the menu before it; it cannot verify the epoch's proofs).

    `base` is the menu before the epoch's joins; `join` the root's join
    pointer read for the epoch (None: none); `join_status` "none" (no
    pointer, or a pointer that admits nothing), "joined", "stale" or
    "refused" (an alarm: no entry joins, on every node)."""

    epoch: int
    state: ms.MenuState | None
    status: str
    committed_hash: str | None = None
    detail: str = ""
    base: ms.MenuState | None = None
    join: object = None
    join_status: str = "none"

    @property
    def ok(self) -> bool:
        return self.state is not None

    @property
    def menu_hash(self) -> str | None:
        return None if self.state is None else ms.menu_hash(self.state)


def with_joins(base: MenuRead, join, menu_bytes: Callable[[str], bytes | None],
               lists: ListBook) -> MenuRead:
    """`base` (the epoch's base menu, as read) with the epoch's joins: the
    root's join pointer `join` (chain_proposals.JoinPointer, or None) and the
    delta's bytes from `menu_bytes(hash)` (this node's copy, the store, a
    mirror). No pointer, or one that admits nothing: the base. A delta with
    no bytes: no menu ("unavailable"). A delta `chain_proposals.check_delta`
    refuses or finds stale: the base, every node alike, logged (an alarm for
    a refusal). Never raises."""
    from fugal_subnet import chain_proposals as cp

    out = dataclasses.replace(base, base=base.state, join=join)
    if base.state is None or join is None or join.delta_hash is None:
        return out
    try:
        raw = menu_bytes(join.delta_hash)
    except Exception:  # noqa: BLE001 - a source that fails offers nothing
        raw = None
    if raw is None or hashlib.sha256(raw).hexdigest() != join.delta_hash:
        return dataclasses.replace(out, state=None, status="unavailable",
                                   detail=f"no bytes for the root's join delta "
                                          f"{join.delta_hash[:16]}")
    allowlist = lists.get(mr.KIND_MENU_ALLOWLIST, base.state.labs) if base.state.labs.serial \
        else None
    try:
        entries = cp.check_delta(base.state, raw, allowlist)
    except cp.DeltaStale as e:
        logger.warning("The root's join delta for %s is stale (%s); no entry joins",
                       epoch_id_for_block(base.epoch), e)
        return dataclasses.replace(out, join_status="stale", detail=str(e))
    except cp.DeltaRefused as e:
        logger.error("ALARM: the root's join delta for %s is refused (%s); no entry joins, as "
                     "on every node", epoch_id_for_block(base.epoch), e)
        return dataclasses.replace(out, join_status="refused", detail=str(e))
    if not entries:
        return out
    return dataclasses.replace(out, state=ms.apply_joins(base.state, entries),
                               join_status="joined")


def read_epoch_menu(epoch: int, previous: ms.MenuState, committed_hash: str | None, *,
                    menu_bytes: Callable[[str], bytes | None], lists: ListBook,
                    params: ms.MenuParams | None = None, join=None,
                    join_bytes: Callable[[str], bytes | None] | None = None) -> MenuRead:
    """MenuState(epoch) from MenuState(epoch - 1), the hash committed for
    `epoch` and whatever bytes and lists this node can find, through the one
    reader, `menu_state.menu_for_epoch`, then the epoch's joins (`join`, the
    root's join pointer for the epoch: `with_joins`). Bytes are asked for
    only when the hash is not the carried menu's. Never raises."""
    return with_joins(_read_base_menu(epoch, previous, committed_hash, menu_bytes=menu_bytes,
                                      lists=lists, params=params), join,
                      join_bytes or menu_bytes, lists)


def _read_base_menu(epoch: int, previous: ms.MenuState, committed_hash: str | None, *,
                    menu_bytes: Callable[[str], bytes | None], lists: ListBook,
                    params: ms.MenuParams | None = None) -> MenuRead:
    epoch = int(epoch)
    try:
        carried = ms.carry(previous)
        raw = None
        if committed_hash is not None and committed_hash != ms.menu_hash(carried):
            raw = menu_bytes(committed_hash)
        allowlist, device_list = _named_lists(previous, raw, lists)
        state = ms.menu_for_epoch(previous, committed_hash, raw, allowlist=allowlist,
                                  device_list=device_list, params=params)
    except ms.MenuRefused as e:
        logger.error("ALARM: the root's menu for epoch %s is refused (%s); the epoch runs on "
                     "the carried menu, as on every node", epoch_id_for_block(epoch), e)
        return MenuRead(epoch, e.carried, "refused", committed_hash, str(e))
    except ms.MenuUnavailable as e:
        return MenuRead(epoch, None, "unavailable", committed_hash, str(e))
    except Exception as e:  # noqa: BLE001 - a menu this node cannot read is no menu
        return MenuRead(epoch, None, "unavailable", committed_hash,
                        f"the menu could not be read ({type(e).__name__}: {e})")
    return MenuRead(epoch, state, "carried" if committed_hash is None else "committed",
                    committed_hash)


@dataclass(frozen=True)
class Rebuilt:
    state: ms.MenuState             # the newest menu reached
    stopped_at: int | None          # the first epoch with no menu, or None
    reason: str = ""
    links: int = 0
    error: BaseException | None = None   # the chain read that stopped it, if one did


def rebuild(*, start: ms.MenuState, target: int, read_commitment: Callable[[int], str | None],
            menu_bytes: Callable[[str], bytes | None], lists: ListBook,
            params: ms.MenuParams | None = None, max_links: int = REBUILD_MAX_LINKS,
            hold: Callable[[ms.MenuState], None] | None = None,
            read_join: Callable[[int], object] | None = None,
            join_bytes: Callable[[str], bytes | None] | None = None) -> Rebuilt:
    """MenuState(target), walked forward from `start` (the last menu this node
    holds, or the pinned seed at the first v4 epoch) one epoch at a time:
    each link is the root's commitment for that epoch (a pinned read at its
    boundary block) through `read_epoch_menu`, exactly as a node that was
    there read it. `hold` keeps each menu reached.

    A link this node cannot verify (the chain read fails or no root hotkey
    covers its epoch, the committed bytes or the list they name are not to
    be had, or the link budget is spent) stops the walk there and says why:
    the node holds every menu before it and none from it on, because every
    later menu follows from that one and none can be skipped, and it never
    goes on from a menu it could not verify. A link every node refuses is
    verified (the carried menu, as on every node that was there). Each
    epoch's commitment must be read under the root hotkey of that epoch
    (`read_menu_commitment` with the table of `config.root_hotkeys_v4`),
    and its joins by the root's join pointer read at the same block
    (`read_join`, `read_join_pointer`). The next call goes on from the last
    menu held, so bytes that appear later (in the store or any mirror) end
    the stop."""
    state, links = start, 0
    for epoch in range(start.epoch + 1, int(target) + 1):
        if links >= max_links:
            return Rebuilt(state, epoch, f"walked {links} epochs this time; goes on next epoch",
                           links)
        try:
            committed = read_commitment(epoch)
            join = read_join(epoch) if read_join is not None else None
        except Exception as e:  # noqa: BLE001 - an unreadable link stops the walk
            return Rebuilt(state, epoch, f"the root's commitment for {epoch_id_for_block(epoch)} "
                                         f"could not be read ({type(e).__name__}: {str(e)[:200]})",
                           links, e)
        got = read_epoch_menu(epoch, state, committed, menu_bytes=menu_bytes, lists=lists,
                              params=params, join=join, join_bytes=join_bytes)
        links += 1
        if not got.ok:
            return Rebuilt(state, epoch, got.detail, links)
        state = got.state
        if hold is not None:
            hold(state)
    return Rebuilt(state, None, "", links)


# --- what a node keeps -----------------------------------------------------------------


class MenuBook:
    """The menus this node ran each epoch on, and the bytes of every menu it
    has seen, which also make it a mirror: `directory/menus/<hash>.json` and
    `directory/held.json` ({epoch index: hash}). `directory` None keeps them
    in memory only. Anything read back from disk is checked against its hash."""

    def __init__(self, directory: str | None = None):
        self.directory = directory
        self._bytes: dict[str, bytes] = {}
        self._held: dict[int, str] = {}
        # {epoch: (source epoch, source hash)}: the last published menu each
        # held menu was carried from (`source_of`), in `directory/sources.json`.
        self._sources: dict[int, tuple[int, str]] = {}
        self._summary_docs: dict[int, dict] | None = None
        self._root_epochs: set[int] | None = None
        if directory:
            try:
                with open(os.path.join(directory, "held.json"), encoding="utf-8") as f:
                    raw = json.load(f)
                self._held = {int(k): str(v) for k, v in raw.items()
                              if str(k).isdigit() and isinstance(v, str)}
            except (OSError, ValueError, AttributeError):
                self._held = {}
            try:
                with open(os.path.join(directory, "sources.json"), encoding="utf-8") as f:
                    raw = json.load(f)
                self._sources = {int(k): (int(v[0]), str(v[1])) for k, v in raw.items()
                                 if str(k).isdigit() and isinstance(v, list) and len(v) == 2
                                 and type(v[0]) is int and isinstance(v[1], str)}
            except (OSError, ValueError, AttributeError, TypeError):
                self._sources = {}

    def bytes_of(self, menu_hash: str) -> bytes | None:
        raw = self._bytes.get(menu_hash)
        if raw is None and self.directory:
            try:
                with open(os.path.join(self.directory, "menus", f"{menu_hash}.json"), "rb") as f:
                    raw = f.read(reveal_sync.MAX_MENU_FETCH_BYTES + 1)
            except (OSError, ValueError):
                raw = None
        if raw is None or hashlib.sha256(raw).hexdigest() != menu_hash:
            return None
        self._bytes[menu_hash] = raw
        return raw

    def keep_bytes(self, raw: bytes) -> str:
        h = hashlib.sha256(raw).hexdigest()
        self._bytes[h] = bytes(raw)
        if self.directory:
            folder = os.path.join(self.directory, "menus")
            try:
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(folder, f"{h}.json")
                if not os.path.exists(path):
                    with open(path + ".tmp", "wb") as f:
                        f.write(raw)
                    os.replace(path + ".tmp", path)
            except OSError as e:
                logger.warning("Could not keep menu %s on disk (%s)", h[:16], type(e).__name__)
        return h

    def hold(self, state: ms.MenuState, source: tuple[int, str] | None = None) -> None:
        """Hold `state` as the menu its epoch runs on, and record the last
        published menu it was carried from (`source_of`; `source` when the
        caller knows it, as an anchor on the root's reveal does). Menus are
        held in epoch order: the menu before is held first (a rebuild holds
        every link it walks)."""
        before = self.held(state.epoch - 1)
        h = self.keep_bytes(ms.canonical_json(state))
        self._held[state.epoch] = h
        if source is not None:
            self._sources[state.epoch] = (int(source[0]), str(source[1]))
        elif (before is not None and dataclasses.replace(before, epoch=state.epoch) == state
                and state.epoch - 1 in self._sources):
            self._sources[state.epoch] = self._sources[state.epoch - 1]
        else:
            self._sources[state.epoch] = (state.epoch, h)
        for e in sorted(self._held)[:-MAX_HELD_EPOCHS]:
            del self._held[e]
        for e in sorted(self._sources)[:-MAX_HELD_EPOCHS]:
            del self._sources[e]
        if self.directory:
            try:
                os.makedirs(self.directory, exist_ok=True)
                for name, doc in (("sources.json", {str(k): [v[0], v[1]] for k, v
                                                    in sorted(self._sources.items())}),
                                  ("held.json", {str(k): v for k, v in sorted(self._held.items())})):
                    path = os.path.join(self.directory, name)
                    with open(path + ".tmp", "w", encoding="utf-8") as f:
                        json.dump(doc, f)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(path + ".tmp", path)
            except OSError as e:
                logger.warning("Could not record the menu held for epoch %d (%s)", state.epoch,
                               type(e).__name__)

    def source_of(self, epoch: int) -> tuple[int, str] | None:
        """(epoch, hash) of the last published menu the menu held for `epoch`
        was carried from: the epoch whose menu has the same content, epoch
        aside, and at which that content was first held (a committed menu,
        which the root put in the store before committing it, or the pinned
        seed at the first v4 epoch); the held menu's own (epoch, hash) when
        its menu was committed at that epoch. A carried menu differs from
        its source only in `epoch` (menu_state.carry), so anyone holding the
        source's bytes rebuilds it (`menu_from_field`). None when no menu is
        held for `epoch`."""
        if int(epoch) not in self._held:
            return None
        return self._sources.get(int(epoch))

    def held(self, epoch: int) -> ms.MenuState | None:
        h = self._held.get(int(epoch))
        raw = None if h is None else self.bytes_of(h)
        if raw is None:
            return None
        try:
            state = ms.from_json(raw)
        except ValueError:
            return None
        return state if state.epoch == int(epoch) else None

    # The window of epoch summaries the step's inputs read
    # (menu_inputs.summarize_epoch, kept in summary_to_dict's form; this
    # module does not read them). One file, `directory/summaries.json`, of
    # {epoch index: summary}, the newest MAX_HELD_EPOCHS.

    def _summaries(self) -> dict[int, dict]:
        if self._summary_docs is None:
            self._summary_docs = {}
            if self.directory:
                try:
                    with open(os.path.join(self.directory, "summaries.json"),
                              encoding="utf-8") as f:
                        raw = json.load(f)
                    self._summary_docs = {int(k): v for k, v in raw.items()
                                          if str(k).isdigit() and isinstance(v, dict)}
                except (OSError, ValueError, AttributeError):
                    self._summary_docs = {}
        return self._summary_docs

    def _from_root(self) -> set[int]:
        if self._root_epochs is None:
            self._root_epochs = set()
            if self.directory:
                try:
                    with open(os.path.join(self.directory, "summaries_from_root.json"),
                              encoding="utf-8") as f:
                        self._root_epochs = {int(e) for e in json.load(f) if isinstance(e, int)}
                except (OSError, ValueError, TypeError):
                    self._root_epochs = set()
        return self._root_epochs

    def _save_from_root(self) -> None:
        if self.directory:
            path = os.path.join(self.directory, "summaries_from_root.json")
            try:
                os.makedirs(self.directory, exist_ok=True)
                with open(path + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(sorted(self._from_root()), f)
                os.replace(path + ".tmp", path)
            except OSError as e:
                logger.warning("Could not record which summaries came from the root (%s)",
                               type(e).__name__)

    def fill_from_root(self, epochs, source: Callable[[int], dict | None]) -> list[int]:
        """Fill the epochs of `epochs` this node holds no summary of from the
        root's published ones (`root_summaries`), marking them as the root's.
        Its own summaries are never replaced. Returns the epochs filled."""
        docs = self._summaries()
        filled = []
        for e in sorted({int(e) for e in epochs}):
            if e in docs:
                continue
            doc = source(e)
            if doc is not None:
                self.keep_summary(e, doc, from_root=True)
                filled.append(e)
        return filled

    def from_root(self, *, through: int, size: int) -> list[int]:
        """The epochs of the window ending at `through` whose summary is the root's."""
        docs, root = self._summaries(), self._from_root()
        return sorted(e for e in root if e in docs and through - size < e <= through)

    def keep_summary(self, epoch: int, doc: dict, *, from_root: bool = False) -> None:
        """Keep epoch `epoch`'s summary (replacing one kept for it: a retried
        epoch summarises the same proofs). One of its own replaces the
        root's; `from_root` marks it as the root's (`fill_from_root`)."""
        docs = self._summaries()
        docs[int(epoch)] = doc
        root = self._from_root()
        if from_root != (int(epoch) in root):
            (root.add if from_root else root.discard)(int(epoch))
            self._save_from_root()
        for e in sorted(docs)[:-MAX_HELD_EPOCHS]:
            del docs[e]
            root.discard(e)
        if self.directory:
            path = os.path.join(self.directory, "summaries.json")
            try:
                os.makedirs(self.directory, exist_ok=True)
                with open(path + ".tmp", "w", encoding="utf-8") as f:
                    json.dump({str(k): v for k, v in sorted(docs.items())}, f,
                              sort_keys=True, separators=(",", ":"))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(path + ".tmp", path)
            except OSError as e:
                logger.warning("Could not keep the summary of epoch %d (%s)", int(epoch),
                               type(e).__name__)

    def summaries(self, *, through: int, size: int) -> list[dict]:
        """The kept summaries of the `size` epochs ending at `through`, oldest
        first. Epochs this node did not observe are simply absent."""
        docs = self._summaries()
        return [docs[e] for e in sorted(docs) if through - size < e <= through]

    def newest_held(self, at_most: int) -> ms.MenuState | None:
        for e in sorted((e for e in self._held if e <= at_most), reverse=True):
            state = self.held(e)
            if state is not None:
                return state
        return None


# --- differences, for the audit --------------------------------------------------------


def _keyed(items) -> dict | None:
    if isinstance(items, list) and all(isinstance(x, dict) for x in items):
        keys = [x.get("proposal_id", x.get("id")) for x in items]
        if all(isinstance(k, str) for k in keys) and len(set(keys)) == len(keys):
            return dict(zip(keys, items))
    return None


def _diff(a, b, path: str, out: list[str]) -> None:
    if a == b:
        return
    ka, kb = _keyed(a), _keyed(b)
    if ka is not None and kb is not None:
        a, b = ka, kb
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            where = f"{path}/{k}" if path else str(k)
            if k not in b:
                out.append(f"{where}: only in this validator's menu")
            elif k not in a:
                out.append(f"{where}: only in the root's menu")
            else:
                _diff(a[k], b[k], where, out)
        return
    out.append(path or "the whole menu")


def menu_differences(own: ms.MenuState, root: ms.MenuState,
                     limit: int = MAX_AUDIT_DIFFERENCES) -> list[str]:
    """Where `own` differs from `root`, as paths into the published document
    (`entries/<id>/price_in`, `pending/<proposal id>: only in the root's
    menu`, `serial`), sorted, at most `limit` and then a count. [] when they
    are the same menu. A menu holds no secret, so neither does this."""
    out: list[str] = []
    _diff(ms.to_dict(own), ms.to_dict(root), "", out)
    if len(out) > limit:
        out = out[:limit] + [f"... and {len(out) - limit} more"]
    return out


# --- the reveal's fields ------------------------------------------------------------


MENU_FIELD_KEYS = ("epoch", "hash", "source")
MENU_SOURCE_KEYS = ("epoch", "hash")


def menu_field(state: ms.MenuState, source: tuple[int, str]) -> dict:
    """`menu`: the menu the epoch ran on, by epoch and hash, and `source`,
    the last published menu it was carried from, by epoch and hash
    (`MenuBook.source_of`; for a menu committed at its own epoch, itself).
    The source's bytes are in the root's store under its hash
    (`<netuid>/menus/<hash>.json`; the pinned seed's at the first v4 epoch
    are also built from the release, `menu_state.seed_state`), and the menu
    is those bytes with `epoch` set to the epoch's (`menu_from_field`)."""
    src_epoch, src_hash = int(source[0]), str(source[1])
    if src_epoch > state.epoch:
        raise ValueError(f"the source of the menu of epoch {state.epoch} is from {src_epoch}")
    return {"epoch": epoch_id_for_block(state.epoch), "hash": ms.menu_hash(state),
            "source": {"epoch": epoch_id_for_block(src_epoch), "hash": src_hash}}


def _epoch_of(epoch_id) -> int:
    """The index of a canonical epoch id; ValueError for anything else."""
    if not isinstance(epoch_id, str):
        raise ValueError("an epoch id is a string")
    k = epoch_index_of(epoch_id)
    if epoch_id_for_block(k) != epoch_id:
        raise ValueError(f"{epoch_id[:24]!r} is not an epoch id")
    return k


def check_menu_field(field, epoch_id: str) -> tuple[int, str, int, str]:
    """(k, hash, source epoch, source hash) of a v4 reveal's `menu` field
    for reveal `epoch_id`, checked in its shape: exactly the keys `epoch`,
    `hash`, `source`; `source` exactly `epoch` and `hash`; both epochs
    canonical epoch ids, the first the reveal's own and the source not
    after it; both hashes 64 lowercase hex. ValueError otherwise."""
    if not isinstance(field, Mapping) or sorted(field) != sorted(MENU_FIELD_KEYS):
        raise ValueError(f"a reveal's menu field is an object of {list(MENU_FIELD_KEYS)}")
    source = field["source"]
    if not isinstance(source, Mapping) or sorted(source) != sorted(MENU_SOURCE_KEYS):
        raise ValueError(f"a menu field's source is an object of {list(MENU_SOURCE_KEYS)}")
    if field["epoch"] != epoch_id:
        raise ValueError("the menu field is not for the reveal's epoch")
    k, src_k = _epoch_of(field["epoch"]), _epoch_of(source["epoch"])
    if src_k > k:
        raise ValueError("the menu's source is from a later epoch than the menu")
    return (k, ms._hex64(field["hash"], "the menu hash"), src_k,
            ms._hex64(source["hash"], "the menu source's hash"))


def menu_from_field(field, epoch_id: str, menu_bytes: Callable[[str], bytes | None]
                    ) -> ms.MenuState:
    """The menu reveal `epoch_id` ran on, from its `menu` field and the bytes
    `menu_bytes(hash)` offers (the root's store: reveal_sync.fetch_menu_bytes),
    trusting neither: the source's bytes must hash to the source hash, be
    within MENU_MAX_BYTES, parse as a menu, and be the menu of the source
    epoch; that menu with `epoch` set to the reveal's is the menu, and its
    canonical bytes must hash to the field's `hash`. ValueError otherwise.

    The menu checks a node runs at its boundary (the compiled caps and rules,
    and the chain's commitment) are not repeated here: this rebuilds and
    checks the menu a reveal names, as a reader with no history needs."""
    k, menu_hash, src_k, src_hash = check_menu_field(field, epoch_id)
    raw = menu_bytes(src_hash)
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > ms.MENU_MAX_BYTES:
        raise ValueError(f"no bytes for the menu source {src_hash[:16]}")
    if hashlib.sha256(bytes(raw)).hexdigest() != src_hash:
        raise ValueError("the bytes offered for the menu source are not the source")
    state = ms.from_json(bytes(raw))
    if state.epoch != src_k:
        raise ValueError(f"the menu source is the menu of epoch {state.epoch}, not of {src_k}")
    rebuilt = dataclasses.replace(state, epoch=k)
    if ms.menu_hash(rebuilt) != menu_hash:
        raise ValueError("the menu rebuilt from its source is not the menu the reveal names")
    return rebuilt


def root_reveal_fields(ran_on: ms.MenuState, result: ms.StepResult, start: ms.MenuState,
                       lists: ListBook, source: tuple[int, str]) -> dict:
    """What the root adds to reveal k: the menu epoch k ran on, `menu_next`
    (MenuState(k + MENU_LAG) as its canonical document, ASCII, exactly the
    bytes that are hashed and committed), its hash, the step's events in the
    step's order, and every signed list the new menu names that `start`
    (the menu it was stepped from) did not, with its bytes and signature, so
    the reveal is one more source for them."""
    raw = ms.canonical_json(result.state)
    named = []
    for kind, new, old in ((mr.KIND_MENU_ALLOWLIST, result.state.labs, start.labs),
                           (mr.KIND_DEVICE_LIST, result.state.devices, start.devices)):
        if new == old or new.serial == 0:
            continue
        item = {"kind": kind, "serial": new.serial, "sha256": new.sha256}
        got = lists.raw(kind, new.sha256)
        if got is not None:
            item.update(payload=got[0].decode("ascii"), signature=got[1].decode("ascii"))
        named.append(item)
    return {
        MENU_FIELD: menu_field(ran_on, source),
        REVEAL_NEXT: raw.decode("ascii"),
        REVEAL_NEXT_HASH: hashlib.sha256(raw).hexdigest(),
        REVEAL_EVENTS: [[e.kind, e.subject, e.detail] for e in result.events],
        REVEAL_LISTS: named,
    }


def follower_reveal_fields(ran_on: ms.MenuState, audit: dict | None,
                           source: tuple[int, str]) -> dict:
    """What another validator adds to its reveal: the menu the epoch ran on
    and its audit of the root's menu for it."""
    out = {MENU_FIELD: menu_field(ran_on, source)}
    if audit is not None:
        out[REVEAL_AUDIT] = audit
    return out


# --- the public store -----------------------------------------------------------------


class GcsStore:
    """The root's public bucket. Writes are create-only (an object, once
    there, is never replaced: every name is a hash of its bytes); reads go
    over public HTTPS, the way any reader sees the object, so a read-back
    confirms what readers will get. Every call is bounded by `timeout`."""

    def __init__(self, bucket: str, timeout: float = 60.0, opener: Callable | None = None):
        if not reveal_sync.valid_bucket(bucket):
            raise ValueError("not a bucket name")
        self.bucket, self.timeout, self.opener = bucket, float(timeout), opener

    def put(self, key: str, data: bytes) -> None:
        try:
            from google.cloud import storage  # type: ignore
        except ImportError:
            storage = None
        if storage is not None:
            from google.api_core.exceptions import PreconditionFailed  # type: ignore

            blob = storage.Client().bucket(self.bucket).blob(key)
            try:
                blob.upload_from_string(bytes(data), if_generation_match=0, timeout=self.timeout)
            except PreconditionFailed:
                pass          # already there; the read-back says whether it is these bytes
            return
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            f.write(bytes(data))
            tmp = f.name
        try:
            subprocess.run(["gcloud", "storage", "cp", "--no-clobber", tmp,
                            f"gs://{self.bucket}/{key}"],
                           check=True, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.CalledProcessError as e:
            # Its text names the local temporary file; this says only what failed.
            raise OSError(f"gcloud storage cp of {key} exited {e.returncode}") from None
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"gcloud storage cp of {key} timed out") from None
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def get(self, key: str, limit: int) -> bytes:
        return reveal_sync._get(f"{reveal_sync.GCS}/{self.bucket}/{key}", int(limit),
                                self.timeout, self.opener)


# --- mirrors --------------------------------------------------------------------------


class Mirror:
    """A validator's own bucket as a mirror of the root's (4.3, the root's
    availability): each menu it accepted at a boundary and each signed list
    it holds for that menu, put under the root's content-addressed names, so
    a reader that lists this bucket in FUGAL_MENU_MIRRORS can read them when
    the root's store does not serve them. Create-only puts, each object once.
    Never raises: nothing depends on a mirror."""

    def __init__(self, *, netuid: int, store):
        self.netuid, self.store = int(netuid), store
        self._done: set[str] = set()

    def _put(self, key: str, data: bytes) -> None:
        if key in self._done:
            return
        try:
            self.store.put(key, data)
            self._done.add(key)
        except Exception as e:  # noqa: BLE001 - a mirror is a convenience
            logger.warning("Could not mirror %s (%s)", key.rsplit("/", 1)[-1][:16],
                           type(e).__name__)

    def mirror(self, read: MenuRead, lists: ListBook) -> None:
        if read is None or read.state is None:
            return
        try:
            raw = ms.canonical_json(read.state)
            h = hashlib.sha256(raw).hexdigest()
            self._put(reveal_sync.menu_object(self.netuid, h), raw)
            for kind, ref in ((mr.KIND_MENU_ALLOWLIST, read.state.labs),
                              (mr.KIND_DEVICE_LIST, read.state.devices)):
                got = lists.raw(kind, ref.sha256) if ref.serial else None
                if got is None:
                    continue
                self._put(reveal_sync.list_object(self.netuid, kind, ref.sha256), got[0])
                self._put(reveal_sync.signature_object(self.netuid, kind, ref.sha256), got[1])
        except Exception as e:  # noqa: BLE001 - a mirror is a convenience
            logger.warning("Could not mirror the menu (%s)", type(e).__name__)


# --- the root's publication -----------------------------------------------------------


class MenuPublisher:
    """The `menu` hook of reveal_publisher.publish_due_reveals, for the root.

    Called with the newest reveal being committed, it returns the hash of
    that reveal's `menu_next` only when all of these hold, in order:

    1. the bytes are the canonical document the reveal names by hash, for
       epoch k + lag;
    2. the menu follows, through `menu_for_epoch`, from the menu this node
       read from the chain for epoch k + lag - 1 (so the root never commits a
       menu its own reader would refuse or could not check, including one
       stepped from a menu the chain did not end up holding);
    3. the menu bytes, and every signed list it names that the menu before
       did not, are put in the store, create-only;
    4. each object is read back from the public store and is exactly the
       bytes put there (a list's signature: one that verifies the list);

    within `budget_s` for steps 3 and 4 together. Otherwise None, and the
    commitment goes out without a menu hash: the previous menu carries and
    the changes wait. It never raises."""

    def __init__(self, *, netuid: int, store, held: Callable[[int], ms.MenuState | None],
                 lists: ListBook, params: ms.MenuParams | None = None,
                 budget_s: float = PUBLISH_BUDGET_S, clock: Callable[[], float] = time.monotonic):
        self.netuid, self.store, self.held, self.lists = int(netuid), store, held, lists
        self.params, self.budget_s, self.clock = params, float(budget_s), clock
        self._sources_put: set[str] = set()

    def __call__(self, epoch_id: str, reveal: Mapping) -> str | None:
        try:
            got = self._publish(str(epoch_id), reveal)
        except Exception as e:  # noqa: BLE001 - a failed publication only delays the menu
            logger.warning("Menu of reveal %s not published (%s: %s); its commitment carries no "
                           "menu hash", epoch_id, type(e).__name__, e)
            got = None
        self._publish_ran_on(str(epoch_id), reveal)
        return got

    def _publish_ran_on(self, epoch_id: str, reveal: Mapping) -> None:
        """Put the bytes of the menu the reveal's epoch ran on (its `menu`
        field) in the store under their content-addressed name, so a TD that
        anchors on this reveal can fetch them (td_v4.V4Session). Most epochs
        run on a carried menu, whose bytes no commitment named. Best effort:
        never raises, and the commitment never waits on it."""
        try:
            field = reveal.get(MENU_FIELD)
            if not isinstance(field, Mapping) or field.get("epoch") != epoch_id:
                return
            state = self.held(epoch_index_of(epoch_id))
            if state is None:
                return
            raw = ms.canonical_json(state)
            h = hashlib.sha256(raw).hexdigest()
            if field.get("hash") != h:
                return
            self.store.put(reveal_sync.menu_object(self.netuid, h), raw)
            # The source the field names (menu_field): a committed menu is
            # in the store already (put and read back before its commitment);
            # the pinned seed's bytes at the first v4 epoch are put here.
            _k, _h, src_k, src_h = check_menu_field(field, epoch_id)
            if src_h != h and src_h not in self._sources_put:
                src_raw = ms.canonical_json(dataclasses.replace(state, epoch=src_k))
                if hashlib.sha256(src_raw).hexdigest() == src_h:
                    self.store.put(reveal_sync.menu_object(self.netuid, src_h), src_raw)
                    self._sources_put.add(src_h)
        except Exception as e:  # noqa: BLE001 - only an anchor's convenience
            logger.warning("The menu reveal %s ran on was not put in the store (%s)", epoch_id,
                           type(e).__name__)

    def _publish(self, epoch_id: str, reveal: Mapping) -> str | None:
        lag = (self.params or ms.READER_PARAMS).menu_lag
        k = epoch_index_of(epoch_id)
        if epoch_id_for_block(k) != epoch_id:
            raise ValueError("the reveal's epoch id is not an epoch id")
        text, named = reveal.get(REVEAL_NEXT), reveal.get(REVEAL_NEXT_HASH)
        if not isinstance(text, str):
            raise ValueError("the reveal carries no menu document")
        raw = text.encode("ascii")
        menu_hash = hashlib.sha256(raw).hexdigest()
        if named != menu_hash:
            raise ValueError("the reveal's menu hash is not the hash of its menu document")
        state = ms.from_json(raw)
        if state.epoch != k + lag:
            raise ValueError(f"the reveal of {epoch_id} carries a menu for epoch {state.epoch}, "
                             f"not {k + lag}")
        previous = self.held(k + lag - 1)
        if previous is None:
            logger.warning("Menu of reveal %s not committed: this node holds no menu for %s, "
                           "the one it follows", epoch_id, epoch_id_for_block(k + lag - 1))
            return None
        got = read_epoch_menu(k + lag, previous, menu_hash, menu_bytes=lambda _h: raw,
                              lists=self.lists, params=self.params)
        if got.status != "committed" or got.menu_hash != menu_hash:
            logger.error("Menu of reveal %s not committed: it does not follow the menu the chain "
                         "gave %s (%s: %s)", epoch_id, epoch_id_for_block(k + lag - 1),
                         got.status, got.detail)
            return None
        def same(data: bytes):
            return lambda served: served == data

        # (name, bytes to put, what the store must serve back). A signature
        # is not named by its own hash: a store that already holds another
        # valid signature for the same list serves that one, and readers
        # verify whatever signature they get, so any that verifies will do.
        objects = [(reveal_sync.menu_object(self.netuid, menu_hash), raw, same(raw))]
        for kind, new, old in ((mr.KIND_MENU_ALLOWLIST, state.labs, previous.labs),
                               (mr.KIND_DEVICE_LIST, state.devices, previous.devices)):
            if new == old or new.serial == 0:
                continue
            held = self.lists.raw(kind, new.sha256)
            if held is None:
                logger.error("Menu of reveal %s not committed: it names %s serial %d, which "
                             "this node cannot publish", epoch_id, kind, new.serial)
                return None
            objects += [(reveal_sync.list_object(self.netuid, kind, new.sha256), held[0],
                         same(held[0])),
                        (reveal_sync.signature_object(self.netuid, kind, new.sha256), held[1],
                         self._verifies(kind, held[0]))]
        deadline = self.clock() + self.budget_s
        for key, data, _check in objects:
            if self.clock() > deadline:
                raise TimeoutError("the publication budget ran out while uploading")
            self.store.put(key, data)
        for key, data, check in objects:
            if self.clock() > deadline:
                raise TimeoutError("the publication budget ran out while reading back")
            if not check(self.store.get(key, max(len(data), mr.MAX_SIGNATURE_BYTES) + 1)):
                raise ValueError(f"the store did not serve back what was put under {key}")
        logger.info("Menu for %s published and read back (%s); committing it with reveal %s",
                    epoch_id_for_block(k + lag), menu_hash[:16], epoch_id)
        return menu_hash

    def _verifies(self, kind: str, payload: bytes):
        def check(signature: bytes) -> bool:
            try:
                mr.verify_signed_list(payload, signature, kind=kind, held_serial=0,
                                      signers=self.lists.signers)
            except (mr.ListRefused, ValueError):
                return False
            return True
        return check


def committed_menu(epoch_dir: str, epoch: int) -> str | None:
    """The menu hash this root committed with its reveal of `epoch`, as its
    publisher recorded it (reveal_publisher's marker), or None when that
    reveal was not committed with one."""
    marker = read_marker(os.path.join(epoch_dir, epoch_id_for_block(int(epoch))))
    if not marker or marker.get("committed") is not True:
        return None
    h = marker.get("menu_hash")
    return h if isinstance(h, str) and len(h) == 64 else None


# --- one node across epochs -----------------------------------------------------------


class MenuNode:
    """One validator's menu across epochs.

    `read_commitment(epoch)` is the root's menu hash for `epoch` read at its
    boundary block (`read_menu_commitment`), raising when the chain cannot
    be read; `fetch_menu(hash)` bytes from the store or a mirror (checked
    here against the hash); `step_params` the step's constants
    (menu_state.MenuParams with the pinned mix)."""

    def __init__(self, *, first_epoch: int, book: MenuBook, lists: ListBook,
                 read_commitment: Callable[[int], str | None],
                 fetch_menu: Callable[[str], bytes | None] | None = None,
                 step_params: ms.MenuParams | None = None,
                 reader_params: ms.MenuParams | None = None,
                 max_links: int = REBUILD_MAX_LINKS,
                 read_join: Callable[[int], object] | None = None,
                 fetch_join: Callable[[str], bytes | None] | None = None):
        self.first_epoch = int(first_epoch)
        self.book, self.lists = book, lists
        self.read_commitment, self.fetch_menu = read_commitment, fetch_menu
        # The root's join pointer of an epoch (read_join_pointer) and a
        # delta's bytes from the store or a mirror (reveal_sync.fetch_join_bytes).
        self.read_join, self.fetch_join = read_join, fetch_join
        self.step_params, self.reader_params = step_params, reader_params
        self.max_links = int(max_links)
        self.lag = (reader_params or ms.READER_PARAMS).menu_lag
        self._inputs: dict[int, Inputs] = {}
        self._rebuilt: dict[int, tuple[int, ...]] = {}
        # (k, menu hash[, source epoch, source hash]) of the newest root reveal
        # verified at an epoch's boundary (read_root_anchor), or None: where a
        # walk that met a pruned link re-anchors (`_anchor`). None: never.
        self.read_anchor: Callable[[int], tuple | None] | None = None

    def join_bytes(self, delta_hash: str) -> bytes | None:
        """A join delta's bytes: this node's copy, else `fetch_join`'s (kept)."""
        raw = self.book.bytes_of(delta_hash)
        if raw is None and self.fetch_join is not None:
            try:
                raw = self.fetch_join(delta_hash)
            except Exception:  # noqa: BLE001 - a source that fails offers nothing
                raw = None
            if raw is not None and hashlib.sha256(raw).hexdigest() == delta_hash:
                self.book.keep_bytes(raw)
            else:
                raw = None
        return raw

    def menu_bytes(self, menu_hash: str) -> bytes | None:
        raw = self.book.bytes_of(menu_hash)
        if raw is None and self.fetch_menu is not None:
            try:
                raw = self.fetch_menu(menu_hash)
            except Exception as e:  # noqa: BLE001 - a source that fails offers nothing
                logger.warning("Fetching menu %s failed (%s)", menu_hash[:16], type(e).__name__)
                raw = None
            if raw is not None and hashlib.sha256(raw).hexdigest() == menu_hash:
                self.book.keep_bytes(raw)
            else:
                raw = None
        return raw

    def _menu_before(self, epoch: int) -> tuple[ms.MenuState | None, str]:
        """MenuState(epoch - 1), from what this node holds or by a rebuild."""
        previous = self.book.held(epoch - 1)
        if previous is not None:
            return previous, ""
        start = self.book.newest_held(epoch - 1)
        if start is None or start.epoch < self.first_epoch:
            start = ms.seed_state(self.first_epoch)
            self.book.hold(start)
        if start.epoch == epoch - 1:
            return start, ""
        logger.warning("No menu held for %s: walking the root's commitments from %s",
                       epoch_id_for_block(epoch - 1), epoch_id_for_block(start.epoch))
        got = self._walk(start, epoch)
        if got.stopped_at is not None and got.error is not None and self._pruned(got.error):
            anchor = self._anchor(epoch, after=got.state.epoch)
            if anchor is not None:
                got = self._walk(anchor, epoch)
        if got.stopped_at is not None:
            return None, (f"the menus could be rebuilt only up to {epoch_id_for_block(got.state.epoch)}; "
                          f"{epoch_id_for_block(got.stopped_at)}: {got.reason}")
        return got.state, ""

    def _walk(self, start: ms.MenuState, epoch: int) -> Rebuilt:
        return rebuild(start=start, target=epoch - 1, read_commitment=self.read_commitment,
                       menu_bytes=self.menu_bytes, lists=self.lists, params=self.reader_params,
                       max_links=self.max_links, hold=self.book.hold, read_join=self.read_join,
                       join_bytes=self.join_bytes)

    @staticmethod
    def _pruned(e: BaseException) -> bool:
        from fugal_subnet.memo_sync import is_pruned

        return is_pruned(e)

    def _anchor(self, epoch: int, *, after: int) -> ms.MenuState | None:
        """A walk that met a link the chain endpoint no longer serves (a
        pruned block) re-anchors as a miner's TD does (td_v4.V4Session): on
        the menu the newest root reveal verified at `epoch`'s boundary names
        (`read_anchor`), its bytes checked against that hash (else the
        bytes of the menu it was carried from, rebuilt at its epoch, or a
        menu this node holds that differs only in its epoch), within the
        compiled caps and passing the compiled rules under the list it
        names (td_v4.check_menu_in_td). Only a menu after epoch `after` (the
        newest this node reached). Held, and returned; None when there is
        none."""
        import dataclasses

        if self.read_anchor is None:
            return None
        try:
            got = self.read_anchor(epoch)
        except Exception as e:  # noqa: BLE001 - no anchor: the walk's stop stands
            logger.warning("No root reveal to re-anchor the menu on (%s)", type(e).__name__)
            return None
        if got is None:
            return None
        try:
            from fugal_subnet.tee import td_v4

            k, menu_hash = int(got[0]), ms._hex64(got[1], "the reveal's menu hash")
            if not (max(self.first_epoch, int(after) + 1) <= k < epoch):
                return None
            state = None
            raw = self.menu_bytes(menu_hash)
            if raw is not None:
                state = ms.from_json(bytes(raw))
            elif len(got) >= 4:
                src_k, src_hash = int(got[2]), ms._hex64(got[3], "the menu source's hash")
                src = self.menu_bytes(src_hash) if self.first_epoch <= src_k <= k else None
                base = ms.from_json(bytes(src)) if src is not None else None
                if base is not None and base.epoch == src_k:
                    rebased = dataclasses.replace(base, epoch=k)
                    state = rebased if ms.menu_hash(rebased) == menu_hash else None
            if state is None:
                for held in (self.book.newest_held(k), ms.seed_state(self.first_epoch)):
                    if held is not None and held.epoch <= k:
                        rebased = dataclasses.replace(held, epoch=k)
                        if ms.menu_hash(rebased) == menu_hash:
                            state = rebased
                            break
            if state is None or state.epoch != k:
                return None
            why = ms.check_caps(state, self.reader_params or ms.READER_PARAMS)
            if why is not None:
                raise ValueError(why)
            td_v4.check_menu_in_td(state, self.lists.get(mr.KIND_MENU_ALLOWLIST, state.labs))
        except Exception as e:  # noqa: BLE001 - an anchor that does not check is not taken
            logger.error("The root's reveal names a menu this node does not take as an anchor "
                         "(%s: %s)", type(e).__name__, str(e)[:200])
            return None
        self.book.hold(state, source=(int(got[2]), str(got[3])) if len(got) >= 4 else None)
        logger.warning("Menu re-anchored on the menu of %s the root's newest reveal names (the "
                       "chain endpoint no longer serves the walk's older links)",
                       epoch_id_for_block(k))
        return state

    def at_boundary(self, epoch: int) -> MenuRead:
        """MenuState(epoch), read at its boundary (section 4.3), and held.
        Raises only when the root's commitment cannot be read from the
        chain: the caller retries the epoch, as for head commitments."""
        epoch = int(epoch)
        if epoch < self.first_epoch:
            raise ValueError(f"epoch {epoch} is before the first v4 epoch {self.first_epoch}")
        if epoch == self.first_epoch:
            seed = ms.seed_state(epoch)
            self.book.hold(seed)
            return MenuRead(epoch, seed, "seed")
        previous, why = self._menu_before(epoch)
        if previous is None:
            got = MenuRead(epoch, None, "unavailable", None, why)
        else:
            got = read_epoch_menu(epoch, previous, self.read_commitment(epoch),
                                  menu_bytes=self.menu_bytes, lists=self.lists,
                                  params=self.reader_params,
                                  join=self.read_join(epoch) if self.read_join else None,
                                  join_bytes=self.join_bytes)
        if got.state is not None:
            self.book.hold(got.state)
        else:
            logger.error("NO MENU for %s: this validator cannot verify any proof of the epoch "
                         "(%s). Menu bytes and signed lists come from the root's reveal store or "
                         "any mirror of it.", epoch_id_for_block(epoch), got.detail)
        return got

    # The root -------------------------------------------------------------------

    def predict_start(self, k: int, committed_menu_of: Callable[[int], str | None],
                      committed_join_of: Callable[[int], object] | None = None
                      ) -> ms.MenuState | None:
        """MenuState(k + lag - 1), the menu the step for scored epoch k starts
        from, as the chain will hold it at its boundary: from MenuState(k),
        which this node read, through the root's own commitments for epochs
        k - lag + 1 .. k - 1 (`committed_menu_of`, its publisher's record)
        and its own join pointers for epochs k + 1 .. k + lag - 1
        (`committed_join_of`: the root commits the joins of k + 1 before it
        steps for k).
        The menu the step derives is read back against the chain's own menu
        before it is committed (MenuPublisher, 2), so a prediction the chain
        did not bear out costs the menu change, never a refused menu."""
        state = self.book.held(k)
        if state is None:
            return None
        for epoch in range(k + 1, k + self.lag):
            got = read_epoch_menu(epoch, state, committed_menu_of(epoch - self.lag),
                                  menu_bytes=self.menu_bytes, lists=self.lists,
                                  params=self.reader_params,
                                  join=committed_join_of(epoch) if committed_join_of else None,
                                  join_bytes=self.join_bytes)
            if got.state is None:
                logger.error("The root cannot tell which menu %s will run on (%s)",
                             epoch_id_for_block(epoch), got.detail)
                return None
            state = got.state
        return state

    def root_step(self, k: int, inputs: Inputs,
                  committed_menu_of: Callable[[int], str | None],
                  committed_join_of: Callable[[int], object] | None = None
                  ) -> tuple[ms.StepResult, ms.MenuState] | None:
        """The root's step for scored epoch k, and the menu it started from;
        None (logged) when there is no step this epoch: the commitment of
        reveal k then names no menu and the menu carries."""
        start = self.predict_start(k, committed_menu_of, committed_join_of)
        if start is None:
            return None
        try:
            result = ms.step_detail(start, _inputs_for(inputs, start), self.step_params)
        except Exception as e:  # noqa: BLE001 - no step this epoch, never a failed epoch
            logger.error("The menu step for %s failed (%s); no menu change is published with "
                         "it", epoch_id_for_block(k), e)
            return None
        self.book.keep_bytes(ms.canonical_json(result.state))
        return result, start

    # Every other validator ------------------------------------------------------

    def keep_inputs(self, k: int, inputs: Inputs, rebuilt=()) -> None:
        """Hold the step inputs of scored epoch k (in a validator, the
        function of the start menu that menu_inputs.step_inputs is, holding
        epoch k's verified proofs and window) until the root's menu for
        k + lag can be audited. In memory: a restart in between leaves that
        epoch unaudited, which decides nothing. `rebuilt` are the epochs of
        the window those inputs read whose summary is the root's
        (MenuBook.fill_from_root); the audit names them."""
        self._inputs[int(k)] = inputs
        self._rebuilt[int(k)] = tuple(sorted(int(e) for e in rebuilt))
        for old in [e for e in self._inputs if e < int(k) - self.lag - 1]:
            del self._inputs[old]
            self._rebuilt.pop(old, None)

    def audit(self, read: MenuRead) -> dict:
        """`menu_audit` for the epoch `read` is of: this validator's own step
        for scored epoch E - lag, from MenuState(E - 1), against the menu the
        root committed for E. `differences` [] means they agree. An alarm is
        logged when they differ; the epoch runs on the root's menu either
        way (`read`).

        When the root committed no menu for E (it published no step for
        E - lag: it was down, or its publication failed), the epoch runs on
        the carried menu, which is the rule (4.3), and there is no root step
        to compare this validator's own step with. The audit then compares
        the menu the epoch runs on with the carry of MenuState(E - 1), and
        says why it did not compare its own step (`why`); its own step's hash
        is still published (`own`). Without its own step (no inputs kept,
        no menu held for E - 1) there is nothing of its own to audit, and
        `differences` is null with the reason, as before."""
        epoch, k = read.epoch, read.epoch - self.lag
        out = {"scored_epoch": epoch_id_for_block(k) if k >= 0 else None,
               "menu_epoch": epoch_id_for_block(epoch), "root": read.committed_hash,
               "status": read.status}
        inputs, start = self._inputs.get(k), self.book.held(epoch - 1)
        if inputs is None or start is None:
            out["differences"] = None
            out["why"] = ("this validator did not keep the step inputs of that epoch"
                          if inputs is None else "this validator held no menu for the epoch before")
            return out
        try:
            own = ms.step_detail(start, _inputs_for(inputs, start), self.step_params).state
        except Exception as e:  # noqa: BLE001 - an audit that cannot run decides nothing
            out.update(differences=None, why=f"this validator's own step failed ({e})")
            return out
        out["own"] = ms.menu_hash(own)
        if self._rebuilt.get(k):
            # Audited from a window partly rebuilt from the root's own
            # summaries: on those epochs it checks the root against itself.
            out["window_from_root"] = [epoch_id_for_block(e) for e in self._rebuilt[k]]
        # The step's output is the epoch's base menu; the joins on top of it
        # are audited apart (join_step.audit_join).
        base = read.base if read.base is not None else read.state
        if read.committed_hash is None:
            out["why"] = ("the root committed no menu for this epoch: it runs on the carried "
                          "menu, and this validator's own step is not compared with it")
            out["differences"] = (None if base is None
                                  else menu_differences(base, ms.carry(start)))
            if out["differences"]:
                logger.error("ALARM: the menu %s runs on is not the carry of the menu before "
                             "(%d difference(s), first: %s)", out["menu_epoch"],
                             len(out["differences"]), out["differences"][0])
            return out
        root = None
        if read.status == "committed":
            root = base
        else:
            raw = self.menu_bytes(read.committed_hash)
            try:
                root = ms.from_json(raw) if raw is not None else None
            except ValueError:
                root = None
        if root is None:
            out["differences"] = ["the root's committed menu could not be read"]
        else:
            out["differences"] = menu_differences(own, root)
        if out["differences"]:
            logger.error("ALARM: the root's menu for %s differs from this validator's own step "
                         "(%d difference(s), first: %s); running on the root's, as every node "
                         "does", out["menu_epoch"], len(out["differences"]),
                         out["differences"][0])
        return out


def epoch_reveal_fields(node: MenuNode, read: MenuRead, *, k: int, inputs: Inputs | None,
                        root: bool, committed_menu_of: Callable[[int], str | None],
                        summary: dict | None = None, rebuilt=(),
                        committed_join_of: Callable[[int], object] | None = None) -> dict:
    """The menu fields of reveal k (protocol v4).

    The root steps from the menu the chain will hold at k + lag - 1 and
    publishes the result (`root_reveal_fields`); it calls this after it has
    published reveal k - 1, so `committed_menu_of(k - 1)` is known. Every
    other validator keeps epoch k's inputs for its audit at k + lag and
    publishes its audit of the root's menu for k (`follower_reveal_fields`).
    `inputs` None (they could not be built) publishes the menu the epoch ran
    on and nothing more. The root's reveal also carries epoch k's
    `summary` (menu_inputs.summary_to_dict) when it has one, so a validator
    that lost its window can rebuild it; `rebuilt` are the window's epochs a
    follower took from the root (its audit names them). Never raises: the
    reveal is written whatever happens here."""
    if read.state is None:
        return {}
    source = node.book.source_of(read.epoch)
    if source is None or node.book.held(read.epoch) != read.state:
        # The book holds every menu a node runs on (at_boundary): one it does
        # not hold is published with no `menu` field rather than a guessed source.
        logger.error("The menu of %s is not the one this node holds; the reveal names no menu",
                     epoch_id_for_block(read.epoch))
        return {}
    plain = {MENU_FIELD: menu_field(read.state, source)}
    if root and summary is not None:
        plain[REVEAL_SUMMARY] = summary
    try:
        if root:
            got = (node.root_step(k, inputs, committed_menu_of, committed_join_of)
                   if inputs is not None else None)
            if got is None:
                return plain
            out = root_reveal_fields(read.state, got[0], got[1], node.lists, source)
            if summary is not None:
                out[REVEAL_SUMMARY] = summary
            return out
        if inputs is not None:
            node.keep_inputs(k, inputs, rebuilt)
        return follower_reveal_fields(read.state, node.audit(read), source)
    except Exception as e:  # noqa: BLE001 - the reveal goes out with the menu it ran on
        logger.error("The menu fields of %s could not be built (%s: %s)", epoch_id_for_block(k),
                     type(e).__name__, e)
        return plain
