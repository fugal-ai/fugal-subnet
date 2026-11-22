"""Protocol v4, the TD side (docs/decisions/LIVE_MENU_V4.md; work package W3).

What a miner's TD does in a v4 epoch, before and around its model calls:

1. **The menu its state names** (sections 4.3 and 15). The TD is handed
   MenuState bytes from any source (the operator, the public reveal store, a
   mirror) and trusts none of them: `epoch_menu` reads the epoch's menu
   through the one reader, `menu_state.menu_for_epoch`, against the hash the
   root committed on chain, and then applies the compiled rules to every
   entry itself before anything is called (`check_menu_in_td`). A menu the
   reader refuses is carried, as on every node. A menu the TD cannot check
   (`MenuUnavailable`) or that fails the rules inside the TD
   (`MenuRejected`) means no proof for the epoch: the harness raises before
   its first call.
2. **The wire request** of an entry or a proposal is built from its named
   fields only (`wire_request`): model, reasoning, max_tokens, and a provider
   order with fallbacks forbidden. Nothing else from the state reaches the
   provider.
3. **Endpoint reads** (section 7.1): `wanted_reads` names every model on the
   menu, provisional entries included, with each one's pinned hosts;
   `endpoint_reads.read_endpoints` reads them.
4. **Joins** (FAST_CROWN_V4.md 18): the root's join delta for the epoch,
   named by the root's second commitment field read at the boundary, is
   checked by hash and by `chain_proposals.check_delta` and applied to the
   base menu (`with_joins`): its entries are on the epoch's menu as
   provisional. A refused or stale delta adds nothing, as on every node. A
   provisional entry whose hosts this TD's own read prices above its menu
   price by more than the price step is not called (`overpriced`).
5. **The challenge** (FAST_CROWN_V4.md 2): `ChallengePolicy` decides each
   epoch whether the TD answers a challenge of the operator's size (500 or
   1,000 questions; 0, the default, none: an exploration-only proof), and
   with which holder: the seat holder the root's reveal of E - 2 names (the
   newest reveal at E's boundary, its bytes checked against the root's
   commitment), its head bytes from any source checked against that hash
   here. The holder itself answers nothing: it stands and is paid on its
   exploration.

Every function is reachable only from the harness's protocol v4 path, and no
v1-v3 code imports this module.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from fugal_subnet import config, menu_rules
from fugal_subnet import menu_state as ms
from fugal_subnet.reveal_sync import MAX_REVEAL_BYTES
from fugal_subnet.tee.endpoint_reads import EndpointRead, wanted_hosts
from fugal_subnet.tee.proof import BenchmarkProof

logger = logging.getLogger(__name__)

# The keys a v4 wire request may carry, and those of its provider preference.
# `messages` is added by the harness and is the only other key sent.
WIRE_KEYS = frozenset({"model", "reasoning", "max_tokens", "provider"})
WIRE_PROVIDER_KEYS = frozenset({"order", "allow_fallbacks"})


# --- the challenge: its size and its holder, each epoch -------------------------------

# The challenge sizes an operator may choose (config.CHALLENGE_SIZES_V4), and
# 0 for none.
CHALLENGE_CHOICES = (0,) + tuple(config.CHALLENGE_SIZES_V4)
# The most bytes read of a root reveal when looking for the seat holder: the
# bound every reader of a reveal uses (reveal_sync.MAX_REVEAL_BYTES), so the
# miner reads every reveal the root can publish. A v4 reveal at 256 UIDs is
# bounded below it (tests/test_v4_reveal_budget.py).
MAX_HOLDER_REVEAL_BYTES = MAX_REVEAL_BYTES


def seat_holder_from_reveal(raw: bytes, epoch_id: str, sha256: str) -> str | None:
    """The seat holder's head hash a root reveal names (`champion.champion`,
    v2_epoch.decision_for_reveal), or None when it names none. The bytes must
    hash to the `sha256` the root committed and be the reveal of `epoch_id`;
    otherwise ValueError (they are not that reveal)."""
    import json

    if not isinstance(raw, (bytes, bytearray)) or len(raw) > MAX_HOLDER_REVEAL_BYTES:
        raise ValueError("not a reveal's bytes")
    if hashlib.sha256(bytes(raw)).hexdigest() != sha256:
        raise ValueError("the bytes are not the reveal the root committed")
    doc = json.loads(bytes(raw))
    if not isinstance(doc, dict) or doc.get("epoch_id") != epoch_id:
        raise ValueError(f"the bytes are not the reveal of {epoch_id}")
    champion = doc.get("champion")
    holder = champion.get("champion") if isinstance(champion, dict) else None
    if holder is not None and not (isinstance(holder, str) and len(holder) == 64
                                   and all(c in "0123456789abcdef" for c in holder)):
        raise ValueError("the reveal's seat holder is not a head hash")
    return holder


@dataclass(frozen=True)
class ChallengeSpec:
    """One epoch's challenge as the TD makes it: `size` questions of the
    challenge list, paired with `holder` (a head hash, or None: there is no
    holder, the open form, and no holder-side call is made), whose head
    `holder_head` (head_eval.load_head_v4) was checked against that hash;
    `spend_usd` the operator's dollar ceiling on its realised spend (whole
    dollars, config.CHALLENGE_SPEND_USD_V4 by default; `challenge_estimate`
    is checked against it before the first call)."""

    size: int
    holder: str | None
    holder_head: object | None = None
    spend_usd: int = config.CHALLENGE_SPEND_USD_V4


def challenge_estimate(state: ms.MenuState, own, theirs) -> Fraction:
    """A challenge's estimated cost in exact dollars, before any call: each
    own call (`own`, the challenger's pick on each question) and each
    holder-side call (`theirs`, where it is not None and differs), at its
    entry's pick cost (menu_state.pick_costs: the epoch's pick-cost table,
    what a challenge's items are priced at). The TD makes no challenge whose
    estimate exceeds the operator's ceiling, and while running counts each
    call in flight at this cost until it is paid (harness._call_all_v4)."""
    own, theirs = list(own), list(theirs)
    if len(own) != len(theirs):
        raise ValueError("a challenge's own picks and the holder's do not align")
    cost = ms.pick_costs(state)
    total = Fraction(0)
    for mine, hp in zip(own, theirs):
        total += cost[mine]
        if hp is not None and hp != mine:
            total += cost[hp]
    return total


class ChallengePolicy:
    """Whether this miner's TD answers a challenge in epoch E, and against
    whom (FAST_CROWN_V4.md 2, 6; the miner's --v4-challenge,
    FUGAL_V4_CHALLENGE, carried over provisioning as `v4_challenge`).

    - `size` 0: never; every proof is exploration only (the odds-only miner,
      and the holder, which answers nothing).
    - `size` 500 or 1000: every epoch, the first `size` questions of the
      epoch's challenge list, paired with the holder after E - HOLDER_LAG
      (config.CHALLENGE_HOLDER_LAG_V4): the holder the root's reveal of that
      epoch names. `read_holder(E)` returns (k, holder) from the newest root
      reveal verified at E's boundary (its bytes checked against the root's
      commitment: `seat_holder_from_reveal`), k its epoch; it raises when
      nothing could be read. `holder_bytes(hash)` offers the holder's head
      bytes from any source (the root's head publication), or None; they are
      checked here against the hash and loaded strictly.

    No challenge this epoch (None), logged with the reason, when the newest
    reveal is not of E - HOLDER_LAG, or the reveal or the holder's bytes
    cannot be had or do not check: a challenge against a holder this TD
    cannot verify is never made. Before the v4 activation there is no holder
    (E - HOLDER_LAG earlier than `first_epoch`): the open form."""

    def __init__(self, size: int, *, first_epoch: int,
                 read_holder: Callable[[int], tuple[int, str | None] | None] | None = None,
                 holder_bytes: Callable[[str], bytes | None] | None = None,
                 spend_usd: int = config.CHALLENGE_SPEND_USD_V4):
        if type(size) is not int or size not in CHALLENGE_CHOICES:
            raise ValueError(f"the challenge size is one of {CHALLENGE_CHOICES}, not {size!r}")
        if (type(spend_usd) is not int
                or not 1 <= spend_usd <= int(config.CHALLENGE_SPEND_MAX_USD_V4)):
            raise ValueError("the challenge's dollar budget is a whole number of dollars from 1 "
                             f"to {config.CHALLENGE_SPEND_MAX_USD_V4}")
        self.spend_usd = spend_usd
        if size and (read_holder is None or holder_bytes is None):
            raise ValueError("a challenge reads the seat holder from the root's reveals and its "
                             "head from the published heads: it needs both readers")
        self.size, self.first_epoch = size, int(first_epoch)
        self.read_holder, self.holder_bytes = read_holder, holder_bytes

    def __call__(self, epoch: int) -> ChallengeSpec | None:
        from fugal_subnet import config as _cfg
        from fugal_subnet.head_eval import load_head_v4

        if not self.size:
            return None
        epoch = int(epoch)
        want = epoch - _cfg.CHALLENGE_HOLDER_LAG_V4
        if want < self.first_epoch:
            return ChallengeSpec(self.size, None, None, self.spend_usd)
        try:
            got = self.read_holder(epoch)
        except Exception as e:  # noqa: BLE001 - no challenge this epoch; logged
            logger.warning("Epoch %d: the seat holder could not be read (%s): no challenge this "
                           "epoch", epoch, type(e).__name__)
            return None
        if got is None or int(got[0]) != want:
            logger.warning("Epoch %d: the root's newest reveal at the boundary is not of epoch %d: "
                           "the holder to pair with is not known, no challenge this epoch",
                           epoch, want)
            return None
        holder = got[1]
        if holder is None:
            return ChallengeSpec(self.size, None, None, self.spend_usd)
        try:
            raw = self.holder_bytes(holder)
        except Exception as e:  # noqa: BLE001 - no challenge this epoch; logged
            logger.warning("Epoch %d: the holder's head could not be fetched (%s): no challenge "
                           "this epoch", epoch, type(e).__name__)
            return None
        if (not isinstance(raw, (bytes, bytearray))
                or hashlib.sha256(bytes(raw)).hexdigest() != holder):
            logger.warning("Epoch %d: no bytes of the holder's head %s that hash to it: no "
                           "challenge this epoch", epoch, holder[:16])
            return None
        try:
            head = load_head_v4(bytes(raw))
        except Exception as e:  # noqa: BLE001 - no challenge this epoch; logged
            logger.warning("Epoch %d: the holder's head does not load (%s): no challenge this "
                           "epoch", epoch, type(e).__name__)
            return None
        return ChallengeSpec(self.size, holder, head, self.spend_usd)


class MenuRejected(RuntimeError):
    """The menu this TD would run the epoch on fails the compiled rules in the
    TD, or is not the epoch's. No call is made and no proof is produced."""


# --- the epoch's menu --------------------------------------------------------------


def check_menu_in_td(state: ms.MenuState,
                     allowlist: menu_rules.MenuAllowlist | None) -> None:
    """The compiled rules on every entry and every proposal of `state`, in
    the TD, before anything is called (section 6; section 14, A8).

    A state that names a signed list is checked under that list
    (`menu_state.check_rules`), which this TD must hold: without it the menu
    cannot be checked, `MenuUnavailable`. A state that names none is a v4
    epoch before the first step adopted a list, so it must be the pinned seed
    at that epoch (`menu_state.seed_state`), whose entries the seed builder
    and the safety check hold to the rules; anything else is MenuRejected.
    Every request is then built (`wire_request`) to be sure it is one."""
    ms.validate(state)
    if state.labs.serial == 0:
        if ms.canonical_json(state) != ms.canonical_json(ms.seed_state(state.epoch)):
            raise MenuRejected(f"epoch {state.epoch}: the menu names no signed list and is not "
                               "the pinned seed")
    else:
        if allowlist is None or (allowlist.serial, allowlist.sha256) != (
                state.labs.serial, state.labs.sha256):
            raise ms.MenuUnavailable(
                f"epoch {state.epoch}: the menu names lab and host list serial "
                f"{state.labs.serial} and this TD does not hold it")
        try:
            refusal = ms.check_rules(state, allowlist)
        except ValueError as e:
            raise MenuRejected(f"epoch {state.epoch}: {e}") from None
        if refusal is not None:
            raise MenuRejected(f"epoch {state.epoch}: the compiled rules refuse the menu: "
                               f"{refusal}")
    for r in state.entries:
        wire_request(r)


def list_naming(state: ms.MenuState, *lists: menu_rules.MenuAllowlist | None
                ) -> menu_rules.MenuAllowlist | None:
    """Of the verified lab and host lists offered, the one `state` names, or
    None."""
    for lst in lists:
        if lst is not None and (lst.serial, lst.sha256) == (state.labs.serial, state.labs.sha256):
            return lst
    return None


def with_joins(state: ms.MenuState, join_pointer, join_published: bytes | None,
               allowlist: menu_rules.MenuAllowlist | None) -> ms.MenuState:
    """MenuState(E) from base(E) and the root's join pointer for E
    (chain_proposals.JoinPointer, or None when the root's second field names
    nothing for E): the base itself without a delta; with one, the delta's
    bytes must hash to the pointer (else MenuUnavailable: no menu, as for a
    committed menu with no bytes), and `chain_proposals.check_delta` must
    accept them under `allowlist` (the list the base names); a delta it
    refuses or finds stale adds nothing, on every node, and is logged."""
    from fugal_subnet import chain_proposals as cp

    if join_pointer is None or join_pointer.delta_hash is None:
        return state
    if join_pointer.epoch != state.epoch:
        raise ValueError(f"a join pointer for epoch {join_pointer.epoch}, not {state.epoch}")
    if (join_published is None
            or hashlib.sha256(bytes(join_published)).hexdigest() != join_pointer.delta_hash):
        raise ms.MenuUnavailable(f"epoch {state.epoch}: the root names join delta "
                                 f"{join_pointer.delta_hash[:16]} and no bytes with that hash "
                                 "were offered")
    try:
        entries = cp.check_delta(state, bytes(join_published), allowlist)
    except cp.DeltaStale as e:
        logger.warning("The join delta for epoch %d is stale (%s); no entry joins", state.epoch, e)
        return state
    except cp.DeltaRefused as e:
        logger.error("ALARM: the join delta for epoch %d is refused (%s); no entry joins, as on "
                     "every node", state.epoch, e)
        return state
    return ms.apply_joins(state, entries)


def epoch_menu(epoch: int, *, previous: ms.MenuState | None, committed_hash: str | None,
               published: bytes | None,
               allowlist: menu_rules.MenuAllowlist | None,
               device_list: menu_rules.DeviceList | None = None,
               held_allowlist: menu_rules.MenuAllowlist | None = None,
               join_pointer=None, join_published: bytes | None = None) -> ms.MenuState:
    """The menu the TD runs epoch `epoch` on.

    `previous` is MenuState(epoch - 1) as this TD holds it, or None for the
    first v4 epoch, whose menu is the pinned seed. Otherwise the menu is
    `menu_state.menu_for_epoch(previous, committed_hash, published,
    allowlist=, device_list=)`: `committed_hash` is the root's commitment read at the
    epoch's boundary block, `published` bytes from any source, `allowlist`
    the verified list those bytes name. A refused committed menu is carried,
    exactly as every other node carries it, and logged as an alarm. Then
    `check_menu_in_td`, under whichever list offered the menu it runs on
    names: `allowlist`, or `held_allowlist`, the verified list `previous`
    names. A carried menu names `previous`'s list whatever list the refused
    bytes named, so a refused menu costs the TD one epoch of menu changes
    and nothing else, as on every node.

    Then the epoch's joins (`with_joins`: `join_pointer`, the root's second
    field read at the boundary, and `join_published`, bytes from any source).

    Raises `MenuUnavailable` (no bytes for the committed hash or the join
    delta, or not the list the menu names) or `MenuRejected`: no proof for
    the epoch."""
    if previous is None:
        state = ms.seed_state(epoch)
    else:
        try:
            state = ms.menu_for_epoch(previous, committed_hash, published, allowlist=allowlist,
                                      device_list=device_list)
        except ms.MenuRefused as e:
            logger.error("ALARM: the menu committed for epoch %d is refused (%s); this epoch "
                         "runs on the carried menu, as on every node", epoch, e)
            state = e.carried
        except ms.MenuUnavailable:
            raise
        except ValueError as e:
            # A list that was not verified, or a malformed hash: the caller's
            # mistake, and still no menu this TD can run on.
            raise MenuRejected(f"epoch {epoch}: {e}") from None
    if state.epoch != epoch:
        raise MenuRejected(f"the menu read is for epoch {state.epoch}, not {epoch}")
    try:
        state = with_joins(state, join_pointer, join_published,
                           list_naming(state, allowlist, held_allowlist))
    except ms.MenuUnavailable:
        raise
    except ValueError as e:
        raise MenuRejected(f"epoch {epoch}: {e}") from None
    check_menu_in_td(state, list_naming(state, allowlist, held_allowlist))
    return state


# --- requests and prices -------------------------------------------------------------


def wire_request(r: ms.Request) -> dict:
    """What is sent for an entry or a proposal, from its named fields only:
    `model`, `reasoning` (when it has one), `max_tokens`, and `provider` with
    the pinned order and fallbacks forbidden. For an entry this is
    `menu_state.request_for`, the one reader of an entry's request."""
    out = {"model": r.model, "max_tokens": r.max_tokens}
    if r.reasoning is not None:
        out["reasoning"] = {r.reasoning[0]: r.reasoning[1]}
    out["provider"] = {"order": list(r.provider_order), "allow_fallbacks": False}
    check_wire_request(out)
    return out


def check_wire_request(req) -> None:
    """A v4 request carries the named keys and nothing else; MenuRejected otherwise."""
    if (not isinstance(req, dict) or set(req) - WIRE_KEYS
            or not {"model", "max_tokens", "provider"} <= set(req)
            or not isinstance(req["provider"], dict)
            or set(req["provider"]) != WIRE_PROVIDER_KEYS
            or req["provider"]["allow_fallbacks"] is not False):
        raise MenuRejected(f"not a v4 wire request: {sorted(req) if isinstance(req, dict) else req!r}")


def entry_request(state: ms.MenuState, entry_id: str) -> dict:
    req = ms.request_for(state, entry_id)
    if req != wire_request(state.entry(entry_id)):
        raise MenuRejected(f"{entry_id}: request_for and the entry's fields disagree")
    return req


def proxy_prices(state: ms.MenuState) -> dict[str, tuple[Fraction, Fraction]]:
    """The proxy's price table for the epoch: dollars per token as exact
    Fractions, keyed by entry id."""
    return {e.id: (Fraction(e.price_in, ms.PICO), Fraction(e.price_out, ms.PICO))
            for e in state.entries}


# --- endpoint reads ----------------------------------------------------------------------


def wanted_reads(state: ms.MenuState) -> dict[str, tuple[str, ...]]:
    """{model: its pinned hosts} for every model on the menu (section 7.1),
    provisional entries included."""
    return wanted_hosts([(r.model, r.provider_order) for r in state.entries])


def overpriced(state: ms.MenuState, reads: Iterable[EndpointRead]) -> frozenset:
    """The provisional entries this TD does not call this epoch: those whose
    hosts its own read of the listing prices (the dearer listed pinned host,
    as the step prices an entry) above the menu's price by more than
    menu_state.MENU_PRICE_STEP, in or out, or above the price ceiling
    (menu_rules.check_price). Their price came from the root's
    read at admission; this read is attested in the proof, and a call not
    made is reported failed, against the entry's answered share."""
    by_model = {r.model: r for r in reads}
    out = set()
    for e in state.entries:
        if not e.provisional:
            continue
        read = by_model.get(e.model)
        hosts = [h for h in (read.hosts if read is not None else ())
                 if h.listed and h.host in e.provider_order]
        if not hosts:
            continue
        p_in, p_out = max(h.price_in for h in hosts), max(h.price_out for h in hosts)
        if (p_in > e.price_in * ms.MENU_PRICE_STEP or p_out > e.price_out * ms.MENU_PRICE_STEP
                or menu_rules.check_price(p_in, p_out) is not None):
            out.add(e.id)
    return frozenset(out)


# --- the epoch's inputs, as the harness takes them ------------------------------------------


@dataclass(frozen=True)
class V4Plan:
    """Which questions a v4 epoch asks. An epoch's plan is built by
    `epoch_plan.plan_v4(pool)` and by nothing else, in the miner's TD and in
    every verifier, so the two draw the same questions (the fields are open
    for tests that run the harness on stand-ins).

    - `scored_slice(nonce, pool, size) -> [question]`: the pinned v4 mix
    - `exploration(nonce, pool, scored_ids, menu, hotkey) -> {question id: entry id}`:
      `exploration_v4`, W5's draw on the same mix
    - `pool_index`, `targets`: the pool by category and the scored category mix.
    """

    scored_slice: Callable[[bytes, list[dict], int], list[dict]]
    exploration: Callable[[bytes, list[dict], set[str], ms.MenuState, str], dict[str, str]]
    pool_index: object
    targets: Mapping[str, float]


    def challenge_list(self, nonce: bytes, pool: list[dict], slice_size: int) -> list[dict]:
        """The epoch's challenge list (epoch_plan.challenge_list_v4): this
        plan's scored slice first, then the rest drawn on this plan's pool
        index and targets. The TD's harness and every verifier call this, so
        a challenge's prefix is the same list on both sides."""
        from fugal_subnet import epoch_plan

        first = self.scored_slice(nonce, pool, slice_size)
        return epoch_plan.challenge_list_v4(nonce, first, self.pool_index, dict(self.targets))


def exploration_v4(nonce: bytes, pool: list[dict], scored_ids: set[str], state: ms.MenuState,
                   hotkey: str, index=None) -> dict[str, str]:
    """The epoch's exploration, {question id: entry id}: section 8's draw
    (epoch_plan.exploration_plan_v4), the shared set, the cost-aware part and
    the coverage part together, on the shares of the epoch's menu alone:
    its measured costs and the exploration-pair counts it publishes
    (epoch_plan.explore_shares_v4(state)). Every validator draws the same.
    `index` is `pool`'s epoch_plan.pool_index_v4, when the caller holds it."""
    from fugal_subnet import epoch_plan

    shares = epoch_plan.explore_shares_v4(state)
    shared, cost, coverage = epoch_plan.exploration_plan_v4(nonce, pool, set(scored_ids), shares,
                                                            hotkey, index)
    out = dict(shared)
    for part in (cost, coverage):
        if set(part) & set(out):
            raise ValueError("the v4 exploration parts overlap")
        out.update(part)
    return out


MAX_CATCH_UP_EPOCHS = 512
# A menu this TD accepted (held in memory or kept on disk) is walked forward
# from when it is at most this many epochs old; an older one is used only
# after the newest root reveal could not anchor the session. Walking needs one
# pinned chain read per epoch, at that epoch's boundary block, which a node
# that keeps ~300 blocks of state no longer serves.
RESUME_MAX_EPOCHS = 8
# The pinned commitment reads this session keeps, newest first: a read made at
# an epoch's boundary while it was recent serves a later walk across it.
MAX_HELD_COMMITMENTS = 64
# Where a miner's TD keeps its v4 state, under its state directory (on a
# miner's TD, the encrypted data volume): the last menu it accepted, as its
# canonical bytes. It holds nothing secret.
MENU_FILE = "menu.json"


class V4Session:
    """A miner's v4 state across epochs: the menu it last accepted, the
    signed lists it has verified, and the root's commitments it has read.

    A full menu state is self-contained, so a TD that starts, or has fallen
    behind, does not replay history. For epoch E it starts from, in order,
    the first of these that reaches MenuState(E - 1) through every reader
    check:

    1. the last menu it accepted (in memory, or `state_dir`'s `menu.json`
       after a restart), when it is at most RESUME_MAX_EPOCHS old;
    2. the newest root reveal published and verified against the root's
       on-chain commitment at E's boundary block (`read_anchor`): the menu
       its epoch k ran on (the reveal's `menu` field, {epoch, hash}), its
       bytes checked against that hash, against the compiled caps and
       against the compiled rules in this TD (`check_menu_in_td`);
    3. the operator's `anchor`, a last resort, checked the same way;
    4. the last menu it accepted, however old (up to MAX_CATCH_UP_EPOCHS);
    5. without an operator's anchor, the pinned seed at `first_epoch` (up
       to MAX_CATCH_UP_EPOCHS).

    From there it walks forward to E - 1 with the full reader checks
    (`epoch_menu`), at most a few epochs from an anchor. Each pinned read
    of a commitment is kept, so a read made while its block was recent
    serves a later walk; a read that fails costs the epoch (no proof), and
    the next epoch starts again from the newest of the above, never from a
    menu it could not verify. Every menu reached is kept, in memory and in
    `menu.json`, whether or not the epoch's proof is produced.

    - `read_commitment(epoch)`: the menu hash the root committed for `epoch`
      (its reveal commitment for epoch - MENU_LAG, read at `epoch`'s boundary
      block), or None when there is none.
    - `read_join(epoch)`: the root's join pointer for `epoch`
      (chain_proposals.parse_join_field of its second field read at the
      boundary block), or None; `join_bytes(hash)` bytes offered for a
      delta, never trusted (`with_joins` checks them).
    - `read_anchor(epoch)`: (k, the hash of the menu epoch k ran on), and
      optionally the epoch and hash of the last published menu it was
      carried from (menu_publish.anchor_from_reveal), from the newest root
      reveal verified against the root's commitment read at `epoch`'s
      boundary block, or None when there is none. It raises when the chain
      cannot be read.
    - `menu_bytes(hash)`: bytes offered for that hash from any source (the
      operator, the reveal store, a mirror), or None. Never trusted: the
      reader checks them against the hash.
    - `list_bytes(serial, sha256)`: (list, signature) bytes for the signed lab
      and host list a menu names, or None. Verified here with
      `menu_rules.verify_signed_list` before use.
    - `device_list_bytes(serial, sha256)`: the same for the signed device
      list a changed menu newly names (section 13.4), or None; without it a
      menu that adopts a device list cannot be read (MenuUnavailable).
    - `pool`: the benchmark pool; the session's plan is
      `epoch_plan.plan_v4(pool)` (a `plan` may be handed in instead, in tests).
    - `published`: the published set this TD advances from the root's
      commitments read pinned at each boundary (memo_sync.ChainSet); with it
      each epoch's plan is `epoch_plan.plan_v4(pool, published=<the set as
      of E>, epoch=E)` (`plan_for`), and without it the epoch gets no proof.
    - `signers`: the allowed-signers bytes lists are verified against
      (default: the shipped, pinned file).
    - `state_dir`: where the TD keeps its v4 state (on a miner's TD, its
      encrypted data volume): `menu.json` (MENU_FILE). Without one it is in
      memory, and a restart starts from an anchor.
    - `anchor`: a menu the operator hands in (any source), a last resort: it
      is checked by the caps and the rules like every menu.
    - `challenge(epoch)`: the epoch's challenge (`ChallengePolicy`), or
      None for none; without it no epoch answers a challenge.
    """

    def __init__(self, *, first_epoch: int, fetch_listing: Callable[[str], bytes],
                 read_commitment: Callable[[int], str | None],
                 menu_bytes: Callable[[str], bytes | None],
                 list_bytes: Callable[[int, str], tuple[bytes, bytes] | None],
                 anchor: ms.MenuState | None = None,
                 device_list_bytes: Callable[[int, str], tuple[bytes, bytes] | None] | None = None,
                 pool: list[dict] | None = None, plan: V4Plan | None = None,
                 signers: bytes | None = None, state_dir=None,
                 read_anchor: Callable[[int], tuple | None] | None = None,
                 challenge: Callable[[int], ChallengeSpec | None] | None = None,
                 published=None, read_join: Callable[[int], object] | None = None,
                 join_bytes: Callable[[str], bytes | None] | None = None):
        import os

        self.first_epoch = int(first_epoch)
        self.signers = signers
        # The epoch's plan is epoch_plan.plan_v4, the one way to build it,
        # which every verifier builds too: with `published` (memo_sync.
        # ChainSet, advanced by the root's commitments read pinned at each
        # boundary) one per epoch, on the set as of E. `plan` is for tests
        # that run the harness on stand-ins.
        self.pool = pool
        self.published = published
        if published is not None and pool is None:
            raise ValueError("a v4 session with the published set is built on the benchmark pool")
        if plan is None and published is None:
            if pool is None:
                raise ValueError("a v4 session is built on the benchmark pool (or a plan)")
            from fugal_subnet import epoch_plan
            plan = epoch_plan.plan_v4(pool)
        self.plan = plan
        self.fetch_listing = fetch_listing
        self.read_commitment = read_commitment
        self.menu_bytes = menu_bytes
        self.list_bytes = list_bytes
        self.device_list_bytes = device_list_bytes
        self.read_join = read_join
        self.join_bytes = join_bytes
        self.read_anchor = read_anchor
        # Epoch E's challenge, if any (ChallengePolicy).
        self.challenge = challenge
        self.state_dir = None if state_dir is None else os.fspath(state_dir)
        if self.state_dir is not None:
            os.makedirs(self.state_dir, exist_ok=True)
        self.menu_path = None if self.state_dir is None else os.path.join(self.state_dir, MENU_FILE)
        self.operator_anchor = anchor
        # The last menu this TD accepted: the one kept on disk, if any; and
        # the few before it, in memory, so an epoch run twice (a restart in
        # the same epoch, a retry) finds the menu it follows.
        self.previous: ms.MenuState | None = self._load_menu()
        self._recent: dict[int, ms.MenuState] = (
            {} if self.previous is None else {self.previous.epoch: self.previous})
        self._commitments: dict[int, str | None] = {}
        self._joins: dict[int, object] = {}
        self._lists: dict[tuple[str, int, str], menu_rules.MenuAllowlist | menu_rules.DeviceList] = {}

    def allowlist(self, ref: ms.AllowlistRef, kind: str = menu_rules.KIND_MENU_ALLOWLIST):
        """The verified signed list of `kind` that `ref` names, or None when
        none is offered or what is offered does not verify or is another list."""
        if ref.serial == 0:
            return None
        source = self.list_bytes if kind == menu_rules.KIND_MENU_ALLOWLIST else self.device_list_bytes
        if source is None:
            return None
        key = (kind, ref.serial, ref.sha256)
        if key not in self._lists:
            got = source(ref.serial, ref.sha256)
            if got is None:
                return None
            try:
                lst = menu_rules.verify_signed_list(got[0], got[1], kind=kind, held_serial=0,
                                                    signers=self.signers)
            except (menu_rules.ListRefused, ValueError) as e:
                logger.error("The signed list offered for serial %d does not verify (%s)",
                             ref.serial, e)
                return None
            if (lst.serial, lst.sha256) != key[1:]:
                logger.error("The list offered for serial %d is another list", ref.serial)
                return None
            self._lists[key] = lst
        return self._lists[key]

    def _named_list(self, previous: ms.MenuState | None, published: bytes | None):
        """The list the menu being read names: the one in the offered bytes
        when they parse, else the one in force. Bytes over MENU_MAX_BYTES are
        not parsed here either (the reader refuses them unread)."""
        ref = previous.labs if previous is not None else ms.AllowlistRef(0, "")
        if published is not None and len(published) <= ms.MENU_MAX_BYTES:
            try:
                ref = ms.from_json(published).labs
            except ValueError:
                pass
        return self.allowlist(ref)

    def _named_device_list(self, previous: ms.MenuState | None, published: bytes | None):
        """The device list the offered bytes newly name, when they parse and
        name one other than the one in force; else None (none is needed)."""
        if previous is None or published is None or len(published) > ms.MENU_MAX_BYTES:
            return None
        try:
            ref = ms.from_json(published).devices
        except ValueError:
            return None
        if ref == previous.devices:
            return None
        return self.allowlist(ref, menu_rules.KIND_DEVICE_LIST)

    # --- the state kept across restarts ------------------------------------------------

    def _load_menu(self) -> ms.MenuState | None:
        """The menu `menu.json` holds, or None. A file that is not a menu is
        set aside (logged), never trusted: the session then anchors anew."""
        import os

        if self.menu_path is None or not os.path.exists(self.menu_path):
            return None
        try:
            with open(self.menu_path, "rb") as f:
                raw = f.read(ms.MENU_MAX_BYTES + 1)
            state = ms.from_json(raw)
        except (OSError, ValueError) as e:
            logger.error("The kept menu %s is not a menu (%s); starting from an anchor",
                         self.menu_path, type(e).__name__)
            return None
        if state.epoch < self.first_epoch:
            return None
        return state

    def _keep(self, state: ms.MenuState) -> None:
        """Hold `state` as the newest menu accepted, and write it to
        `menu.json` (atomically). Older menus are not kept over newer ones."""
        import os

        self._recent[state.epoch] = state
        for old in sorted(self._recent)[:-RESUME_MAX_EPOCHS]:
            del self._recent[old]
        if self.previous is not None and state.epoch <= self.previous.epoch:
            return
        self.previous = state
        if self.menu_path is None:
            return
        try:
            tmp = f"{self.menu_path}.tmp"
            with open(tmp, "wb") as f:
                f.write(ms.canonical_json(state))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.menu_path)
        except OSError as e:
            logger.warning("Could not keep the menu of epoch %d on disk (%s)", state.epoch,
                           type(e).__name__)

    def _commitment(self, epoch: int) -> str | None:
        """The root's commitment for `epoch`, read once and kept: the read is
        pinned at the epoch's boundary block, and a node that keeps only
        recent state serves it only while that block is recent."""
        epoch = int(epoch)
        if epoch not in self._commitments:
            self._commitments[epoch] = self.read_commitment(epoch)
            for old in sorted(self._commitments)[:-MAX_HELD_COMMITMENTS]:
                del self._commitments[old]
        return self._commitments[epoch]

    def _join(self, epoch: int):
        """The root's join pointer for `epoch`, read once and kept, as
        `_commitment` keeps its reads."""
        epoch = int(epoch)
        if self.read_join is None:
            return None
        if epoch not in self._joins:
            self._joins[epoch] = self.read_join(epoch)
            for old in sorted(self._joins)[:-MAX_HELD_COMMITMENTS]:
                del self._joins[old]
        return self._joins[epoch]

    # --- where an epoch's walk starts --------------------------------------------------

    def _checked_start(self, state: ms.MenuState, what: str) -> ms.MenuState:
        """`state` as a start, after the compiled caps and the rules in this
        TD (`check_menu_in_td`, under the list it names)."""
        why = ms.check_caps(state, ms.READER_PARAMS)
        if why is not None:
            raise MenuRejected(f"{what} for epoch {state.epoch}: {why}")
        check_menu_in_td(state, self.allowlist(state.labs))
        return state

    def _reveal_anchor(self, epoch: int) -> ms.MenuState:
        """MenuState(k) from the newest root reveal verified at `epoch`'s
        boundary (`read_anchor`): the bytes of the menu its `menu` field
        names, from any source; else the bytes of the last published menu it
        was carried from (the field's `source`, when `read_anchor` gives it)
        rebuilt at k; else a menu this TD holds that differs from it only in
        its epoch (a carried menu). Each is checked against that hash."""
        got = self.read_anchor(epoch) if self.read_anchor is not None else None
        if got is None:
            raise ms.MenuUnavailable(f"epoch {epoch}: no root reveal to anchor on")
        k, menu_hash = int(got[0]), ms._hex64(got[1], "the reveal's menu hash")
        if not self.first_epoch <= k < epoch:
            raise ms.MenuUnavailable(f"epoch {epoch}: the root's newest reveal is for epoch {k}")
        state = None
        raw = self.menu_bytes(menu_hash)
        if isinstance(raw, (bytes, bytearray)) and hashlib.sha256(bytes(raw)).hexdigest() == menu_hash:
            try:
                state = ms.from_json(bytes(raw))
            except ValueError as e:
                raise MenuRejected(f"the menu the root's reveal {k} names is not a menu ({e})") from None
        elif len(got) >= 4:
            # The last published menu it was carried from (the reveal's
            # `menu.source`): its bytes by their hash, the menu of the source
            # epoch, rebuilt at k and checked against the menu hash.
            import dataclasses

            src_k, src_hash = int(got[2]), ms._hex64(got[3], "the menu source's hash")
            src = self.menu_bytes(src_hash) if self.first_epoch <= src_k <= k else None
            if isinstance(src, (bytes, bytearray)) and len(src) <= ms.MENU_MAX_BYTES and \
                    hashlib.sha256(bytes(src)).hexdigest() == src_hash:
                try:
                    base = ms.from_json(bytes(src))
                except ValueError:
                    base = None
                if base is not None and base.epoch == src_k:
                    rebased = dataclasses.replace(base, epoch=k)
                    if ms.menu_hash(rebased) == menu_hash:
                        state = rebased
        if state is None:
            import dataclasses

            for held in (self.previous, self.operator_anchor, ms.seed_state(self.first_epoch)):
                if held is not None and held.epoch <= k:
                    rebased = dataclasses.replace(held, epoch=k)
                    if ms.menu_hash(rebased) == menu_hash:
                        state = rebased
                        break
        if state is None:
            raise ms.MenuUnavailable(f"epoch {epoch}: no bytes for the menu of epoch {k} the "
                                     f"root's reveal names ({menu_hash[:16]})")
        if state.epoch != k:
            raise MenuRejected(f"the menu the root's reveal {k} names is for epoch {state.epoch}")
        return self._checked_start(state, "the menu the root's reveal names")

    def _starts(self, epoch: int):
        """(what, a function giving the start) in the order they are tried."""
        prev, anchor = self.previous, self.operator_anchor
        out = []
        if prev is not None and prev.epoch < epoch and epoch - prev.epoch <= RESUME_MAX_EPOCHS:
            out.append(("the last menu accepted", lambda: prev))
        out.append(("the newest root reveal", lambda: self._reveal_anchor(epoch)))
        if anchor is not None and anchor.epoch < epoch:
            out.append(("the operator's anchor",
                        lambda: self._checked_start(anchor, "the operator's anchor")))
        if prev is not None and prev.epoch < epoch and epoch - prev.epoch > RESUME_MAX_EPOCHS:
            out.append(("the last menu accepted", lambda: prev))
        if epoch > self.first_epoch and anchor is None:
            out.append(("the pinned seed", lambda: ms.seed_state(self.first_epoch)))
        return out

    def _walk(self, start: ms.MenuState, epoch: int) -> ms.MenuState:
        """MenuState(epoch - 1), read link by link from `start`, each menu
        reached kept. Raises when a link cannot be read."""
        if epoch - start.epoch > MAX_CATCH_UP_EPOCHS:
            raise ms.MenuUnavailable(f"{epoch - start.epoch} epochs since the menu of epoch "
                                     f"{start.epoch}; hand the TD a recent menu to start from")
        prev = start
        for e in range(start.epoch + 1, epoch):
            prev = epoch_menu(e, **self._menu_inputs(e, prev))
            self._keep(prev)
        return prev

    def menu_before(self, epoch: int) -> ms.MenuState | None:
        """MenuState(epoch - 1), from the first start that reaches it (the
        class docstring); None for the first v4 epoch. Raises MenuUnavailable
        (or MenuRejected) when none does: no proof for the epoch."""
        epoch = int(epoch)
        if epoch == self.first_epoch:
            return None
        if epoch - 1 in self._recent:
            return self._recent[epoch - 1]
        failures = []
        for what, start in self._starts(epoch):
            try:
                return self._walk(start(), epoch)
            except Exception as e:  # noqa: BLE001 - this start fails; the next is tried
                failures.append(f"{what}: {type(e).__name__}: {str(e)[:160]}")
        raise ms.MenuUnavailable(f"epoch {epoch}: no menu to start from reaches it ("
                                 + "; ".join(failures) + ")")

    def _menu_inputs(self, epoch: int, previous: ms.MenuState | None) -> dict:
        if previous is None:
            return dict(previous=None, committed_hash=None, published=None, allowlist=None,
                        device_list=None, held_allowlist=None, join_pointer=None,
                        join_published=None)
        committed = self._commitment(epoch)
        published = self.menu_bytes(committed) if committed is not None else None
        # Bytes from an untrusted source are looked at only when they carry
        # the committed hash; any others are as good as none (the reader then
        # says MenuUnavailable). Bytes that carry it are the committed menu
        # whatever their size: the reader refuses those over MENU_MAX_BYTES
        # unparsed, and the epoch is carried as on every node. Dropping them
        # here instead would stop this TD's proofs while every validator ran.
        if published is not None and (
                not isinstance(published, (bytes, bytearray))
                or hashlib.sha256(bytes(published)).hexdigest() != committed):
            logger.error("The menu bytes offered for epoch %d are not the committed menu", epoch)
            published = None
        join = self._join(epoch)
        join_published = None
        if join is not None and join.delta_hash is not None and self.join_bytes is not None:
            join_published = self.join_bytes(join.delta_hash)
        return dict(previous=previous, committed_hash=committed, published=published,
                    allowlist=self._named_list(previous, published),
                    device_list=self._named_device_list(previous, published),
                    held_allowlist=self.allowlist(previous.labs),
                    join_pointer=join, join_published=join_published)

    def plan_for(self, epoch: int) -> V4Plan:
        """Epoch E's plan: `epoch_plan.plan_v4` on the pool with the
        published set as of E (`published.view_for`), the plan every
        validator builds for E; the session's one plan without a published
        set (tests). Raises memo_sync.PublishedUnavailable when a reveal the
        chain names cannot be had or contradicts the TD's state: no proof."""
        if self.published is None:
            return self.plan
        from fugal_subnet import epoch_plan

        view = self.published.view_for(int(epoch))
        return epoch_plan.plan_v4(self.pool, published=view, epoch=int(epoch))

    def inputs_for(self, epoch: int) -> V4EpochInputs:
        """The harness's inputs for `epoch`, from MenuState(epoch - 1) as
        `menu_before` reaches it. Raises MenuUnavailable or MenuRejected
        when it cannot: no proof. The epoch's own menu, when it can be read
        here, is kept as the newest accepted whatever becomes of the proof,
        so the next epoch needs only its own boundary read."""
        epoch = int(epoch)
        if epoch < self.first_epoch:
            raise ValueError(f"epoch {epoch} is before the first v4 epoch {self.first_epoch}")
        if epoch > self.first_epoch:
            self._commitment(epoch)             # read while its block is recent, and kept
            self._join(epoch)
        prev = self.menu_before(epoch)
        challenge = None if self.challenge is None else self.challenge(epoch)
        inputs = V4EpochInputs(epoch=epoch, plan=self.plan_for(epoch),
                               fetch_listing=self.fetch_listing,
                               challenge=challenge, **self._menu_inputs(epoch, prev))
        try:
            self._keep(inputs.menu())
        except (ms.MenuUnavailable, MenuRejected):
            pass                                 # the harness raises it again: no proof
        return inputs

    def ran(self, inputs: V4EpochInputs, proof: BenchmarkProof) -> ms.MenuState:
        """Record the menu `proof` ran on as the one the next epoch follows."""
        state = inputs.menu()
        if ms.menu_hash(state) != proof.menu_hash:
            raise MenuRejected("the proof names a menu other than the one its inputs give")
        self._keep(state)
        return state


@dataclass(frozen=True)
class V4EpochInputs:
    """Everything a v4 epoch needs beyond what every protocol takes."""

    epoch: int
    previous: ms.MenuState | None
    committed_hash: str | None
    published: bytes | None
    allowlist: menu_rules.MenuAllowlist | None
    plan: V4Plan
    fetch_listing: Callable[[str], bytes]
    # The verified device list a changed menu newly names, or None.
    device_list: menu_rules.DeviceList | None = None
    # The verified lab and host list `previous` names: what a carried menu
    # is checked under when the committed one is refused.
    held_allowlist: menu_rules.MenuAllowlist | None = None
    # The epoch's challenge (FAST_CROWN_V4.md 2), or None: an exploration-only
    # proof. The miner decides its size (V4Session's `challenge`,
    # neurons/miner's --v4-challenge); the holder is the epoch's.
    challenge: ChallengeSpec | None = None
    # The TD's time budget for the challenge's calls (config.CHALLENGE_BUDGET_S_V4).
    challenge_budget_s: float = float(config.CHALLENGE_BUDGET_S_V4)
    # The root's join pointer for the epoch and the delta's bytes, if any.
    join_pointer: object = None
    join_published: bytes | None = None

    def menu(self) -> ms.MenuState:
        """The epoch's menu, `epoch_menu` on these inputs: what the harness
        runs on and what the session records."""
        return epoch_menu(self.epoch, previous=self.previous, committed_hash=self.committed_hash,
                          published=self.published, allowlist=self.allowlist,
                          device_list=self.device_list, held_allowlist=self.held_allowlist,
                          join_pointer=self.join_pointer, join_published=self.join_published)

    def lab_list(self, state: ms.MenuState) -> menu_rules.MenuAllowlist | None:
        """The verified lab and host list `state` (the epoch's menu) names."""
        return list_naming(state, self.allowlist, self.held_allowlist)
