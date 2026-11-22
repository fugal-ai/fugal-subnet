"""Protocol v4: proposals filed on chain, and the root's join delta
(docs/decisions/FAST_CROWN_V4.md, section 18; docs/INVARIANTS.md).

A miner proposes a model entry by committing the sha256 of a PROPOSAL
DOCUMENT on chain, in a second field of the same commitment that carries its
head pointer, and by serving the document from its own box. At a block B a
compiled margin before each boundary the root reads the chain, fetches the
documents named there, reads each proposed model's public endpoints listing,
checks each proposal by hash and against every entry rule, the signed list
in force and the listing, admits at most one or two (the ADMISSION ORDER is
HMAC(hash of block B, proposal hash)), and publishes the result as a JOIN
DELTA, a document named by hash in a second field of its own commitment
before the boundary. At the boundary every node applies the delta to the
menu, and the admitted entries join as provisional (menu_state.apply_joins).

Nobody but the root fetches from miners. A miner's TD reads only the
root's delta, checked by hash and by `check_delta`, which re-runs the
admission on the delta's own contents: every verdict the delta states must
be the one the rules give. Every other validator recomputes the delta from
its own pinned read of the chain at B and the documents and listings inside
the root's delta (`admit`) and publishes how the two differ (an alarm when
they do).

A proposal's price, answer limit and whether it may name a reasoning effort
come from the listing the ROOT reads at admission, stated in the delta,
never from the proposer: an effort is allowed only when every pinned host
lists `reasoning_effort` among its supported parameters. From the join epoch
the TDs read the entry's hosts in every proof (attested), and the step that
decides the entry prices it from their votes.

THE COMMITMENT FIELDS. The SDK pinned in uv.lock (bittensor 10.5.0) writes a
commitment as `{"fields": [[{Raw<n>: bytes}, ...]]}` and reads it back with
`decode_metadata`, which returns the FIRST field only; its own two-field
commitment is `[Raw<n>, ResetBondsFlag]`. So a reader that predates this
module reads the head pointer (a miner's) and the reveal pointer (the
root's) exactly as before, whatever a second field holds, and every code
reader of commitments goes through `decode_metadata`
(commitments.get_commitments_with_blocks). This module reads the second
field itself (`commitment_fields`) and writes both together
(`set_commitment_fields`), one transaction, so a head and its proposal, or a
reveal pointer and a join pointer, change atomically:

    miner:  [<head sha256, 64 hex>, "proposal:<document sha256, 64 hex>"]
    root:   [<reveal pointer, as before>, "join:<epoch id>" or "join:<epoch id>:<delta sha256>"]

`join:<epoch id>` with no hash says the root read the chain for that
epoch's boundary and admits nothing; with a hash it names the delta. No
second field for the epoch means the root was late or absent: no joins, and
the proposals wait on chain for the next epoch.

Exact: everything compared across nodes is integers, strings and canonical
JSON; the order is an HMAC. Nothing here runs before a v4 epoch, and no
v1-v3 code imports this module.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from fugal_subnet import menu_rules
from fugal_subnet import menu_state as ms
from fugal_subnet.benchmarks.slicer import epoch_id_for_block

logger = logging.getLogger(__name__)

# The chain read for epoch E's joins is pinned at E's boundary block minus
# this many blocks (FAST_CROWN_V4.md 18(b)). 30 blocks of 12 s are 360 s for
# the root to read the chain (one pinned read of the commitments and the
# metagraph), fetch the documents and the listings (`JOIN_FETCH_BUDGET_S`),
# publish the delta and read it back (`JOIN_PUBLISH_BUDGET_S`) and have its
# commitment included `JOIN_COMMIT_SLACK_BLOCKS` before the boundary:
#   reads 12 s + fetch 60 s + publish 90 s + commit 48 s (inclusion and
#   finality, four blocks) + 5 blocks of slack (60 s) = 270 s < 360 s.
# The root starts the step only while that much is left (`join_time_left`).
JOIN_READ_MARGIN_BLOCKS = 30
JOIN_FETCH_BUDGET_S = 60.0
JOIN_FETCH_TIMEOUT_S = 12.0
JOIN_PUBLISH_BUDGET_S = 90.0
JOIN_COMMIT_SLACK_BLOCKS = 5

PROPOSAL_DOMAIN = "fugal-chain-proposal-v1"
DELTA_DOMAIN = "fugal-join-delta-v1"
# A proposal document's bytes, at most: a request with two pinned hosts is
# under 300 bytes.
PROPOSAL_MAX_BYTES = 1024
# Proposals one delta considers, at most: one per registered hotkey of a
# subnet of 256 UIDs, the most a chain read at B can name.
MAX_CONSIDERED = 256
# A delta's bytes, at most: MAX_CONSIDERED documents at their largest with
# their listings and verdicts fit with room to spare.
DELTA_MAX_BYTES = 1_048_576
# The serving hotkeys a delta may state, at most: far above any subnet's
# registered hotkeys (256 on finney), and small enough that the check every
# node and TD makes of a delta (`claimed_read`, which stands in for each
# serving hotkey) stays a few megabytes.
MAX_SERVING = 65_536
MAX_HOTKEY_LEN = 64
MAX_SUPPORTED_PARAMETERS = 64

PROPOSAL_FIELD_PREFIX = "proposal:"
JOIN_FIELD_PREFIX = "join:"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_PROPOSAL_FIELD = re.compile(r"proposal:([0-9a-f]{64})")
_JOIN_FIELD = re.compile(r"join:(e\d{8,12})(?::([0-9a-f]{64}))?")

# The verdicts a delta states for each proposal it considers.
JOINED = "joined"
NOT_SERVED = "not_served"        # its hotkey has no verified proof in the reveal the root names
UNFETCHED = "unfetched"          # no bytes with its hash came back from the proposer's box
UNREAD = "unread"                # the root could not read the model's listing
MALFORMED = "malformed"          # the bytes are not a proposal document of this hotkey
COLDKEY_ONCE = "coldkey_once"    # its coldkey has an earlier proposal admitted this epoch
NO_PLACE = "no_place"            # the epoch's joins, the cap or the provisional entries are full
REFUSED = "refused:"             # the rules refuse it: refused:<code>
_VERDICTS = frozenset({JOINED, NOT_SERVED, UNFETCHED, UNREAD, MALFORMED, COLDKEY_ONCE, NO_PLACE})

# The field of a v4 reveal listing the hotkeys whose proofs the validator
# verified that epoch, sorted: what "served the epoch" is read from.
SERVED_FIELD = "served_v4"


class DeltaRefused(ValueError):
    """Bytes with the committed hash that are not a join delta every node
    accepts for the epoch: malformed, for another epoch, or stating a verdict
    the rules do not give. Every node runs the epoch on the base menu (no
    joins) and logs an alarm."""


class DeltaStale(ValueError):
    """A well-formed delta for the epoch that was computed against another
    base menu than the one the chain gave at the boundary: no joins."""


def _hex64(v, what: str) -> str:
    if not isinstance(v, str) or not _HEX64.fullmatch(v):
        raise ValueError(f"{what} must be 64 lowercase hex characters")
    return v


def _hotkey(v, what: str = "a hotkey") -> str:
    if (not isinstance(v, str) or not 1 <= len(v) <= MAX_HOTKEY_LEN
            or any(not 0x21 <= ord(c) <= 0x7E for c in v)):
        raise ValueError(f"{what} must be 1..{MAX_HOTKEY_LEN} printable ASCII characters")
    return v


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def canonical(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _no_float(text: str):
    raise ValueError(f"no fractional number ({text[:20]})")


def _no_repeat(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"field {k!r} appears twice")
        out[k] = v
    return out


def _parse_canonical(raw, limit: int, what: str):
    """The JSON object of canonical bytes at most `limit` long; ValueError
    for anything else (another layout, a float, a repeated field)."""
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError(f"{what} is bytes")
    raw = bytes(raw)
    if len(raw) > limit:
        raise ValueError(f"{what} is {len(raw)} bytes, more than {limit}")
    try:
        doc = json.loads(raw.decode("ascii"), parse_float=_no_float, parse_constant=_no_float,
                         object_pairs_hook=_no_repeat)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as e:
        raise ValueError(f"{what} is not JSON: {e}") from None
    if not isinstance(doc, dict):
        raise ValueError(f"{what} is a JSON object")
    try:
        again = canonical(doc)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what} cannot be written canonically ({e})") from None
    if again != raw:
        raise ValueError(f"{what} is not in canonical form")
    return doc


# --- the proposal document -------------------------------------------------------------


@dataclass(frozen=True)
class ProposalDoc:
    """A proposal document: which entry, and whose. Its prices, answer limit
    and whether its hosts take a reasoning effort come from the listing the
    root reads at admission (`HostListing`), never from the proposer."""

    proposer: str
    model: str
    reasoning: tuple[str, str | int | bool] | None
    provider_order: tuple[str, ...]
    quantization: str

    @property
    def id(self) -> str:
        return menu_rules.entry_id(self.model, dict([self.reasoning]) if self.reasoning else {})

    def request_dict(self, max_tokens: int) -> dict:
        """The request as menu_rules reads it."""
        out = {"model": self.model,
               "provider": {"order": list(self.provider_order), "allow_fallbacks": False},
               "quantization": self.quantization, "max_tokens": int(max_tokens)}
        if self.reasoning is not None:
            out["reasoning"] = {self.reasoning[0]: self.reasoning[1]}
        return out

    def request(self, max_tokens: int = menu_rules.MAX_ANSWER_TOKENS) -> ms.Request:
        return ms.Request(id=self.id, rev=1, model=self.model, reasoning=self.reasoning,
                          max_tokens=int(max_tokens), provider_order=self.provider_order,
                          quantization=self.quantization)


def document_dict(doc: ProposalDoc) -> dict:
    out = {"domain": PROPOSAL_DOMAIN, "proposer": doc.proposer, "model": doc.model,
           "provider": {"order": list(doc.provider_order), "allow_fallbacks": False},
           "quantization": doc.quantization}
    if doc.reasoning is not None:
        out["reasoning"] = {doc.reasoning[0]: doc.reasoning[1]}
    return out


def document_bytes(doc: ProposalDoc) -> bytes:
    """The one canonical form of a proposal document, checked."""
    raw = canonical(document_dict(doc))
    parse_document(raw)
    return raw


def proposal_hash(raw: bytes) -> str:
    return hashlib.sha256(bytes(raw)).hexdigest()


def parse_document(raw) -> ProposalDoc:
    """The proposal document of canonical bytes, strictly: at most
    PROPOSAL_MAX_BYTES, exactly the keys domain, proposer, model, provider,
    quantization (and reasoning when set); a provider order of distinct host
    tags with fallbacks forbidden; and a request whose shape the menu reads
    (menu_state's request checks). ValueError otherwise. Whether the rules,
    the signed list and the listing allow it is `admit`'s question."""
    d = _parse_canonical(raw, PROPOSAL_MAX_BYTES, "a proposal document")
    keys = {"domain", "proposer", "model", "provider", "quantization"}
    if set(d) - {"reasoning"} != keys:
        raise ValueError(f"a proposal document has exactly {sorted(keys)} and maybe reasoning")
    if d["domain"] != PROPOSAL_DOMAIN:
        raise ValueError("not a chain proposal document")
    provider = d["provider"]
    if (not isinstance(provider, dict) or set(provider) != {"order", "allow_fallbacks"}
            or provider["allow_fallbacks"] is not False or not isinstance(provider["order"], list)
            or not provider["order"] or not all(isinstance(h, str) for h in provider["order"])
            or len(set(provider["order"])) != len(provider["order"])
            or len(provider["order"]) > menu_rules.MENU_MAX_HOSTS):
        raise ValueError(f"a proposal pins an order of 1..{menu_rules.MENU_MAX_HOSTS} distinct "
                         "hosts and forbids fallbacks")
    reasoning = None
    if "reasoning" in d:
        r = d["reasoning"]
        if not isinstance(r, dict) or len(r) != 1:
            raise ValueError("a proposal's reasoning sets exactly one field")
        reasoning = next(iter(r.items()))
    if not isinstance(d["model"], str) or not isinstance(d["quantization"], str):
        raise ValueError("a proposal's model and quantization are strings")
    doc = ProposalDoc(proposer=_hotkey(d["proposer"], "the proposer"), model=d["model"],
                      reasoning=reasoning, provider_order=tuple(provider["order"]),
                      quantization=d["quantization"])
    try:
        ms._check_request(doc.request(), "proposal")
    except (ValueError, TypeError) as e:
        raise ValueError(f"not a request the menu reads: {e}") from None
    return doc


# --- the root's read of the listing ----------------------------------------------------


@dataclass(frozen=True)
class HostListing:
    """A pinned host of a proposed model as the root read the public endpoints
    listing at admission: listed or not, its prices (picodollars per token),
    its stated answer limit (0: none), and whether it takes a reasoning
    effort (`reasoning_effort` among its supported parameters)."""

    host: str
    listed: bool
    price_in: int = 0
    price_out: int = 0
    limit: int = 0
    efforts: bool = False


def listing_from_page(model: str, raw: bytes, hosts) -> tuple[HostListing, ...]:
    """The root's listing of `model`'s pinned `hosts` from one public
    endpoints page: prices, limit and listed-ness as the TDs parse them
    (endpoint_reads.parse_listing, every bound of it), and each listed host's
    `reasoning_effort` support from its `supported_parameters` (a list of at
    most MAX_SUPPORTED_PARAMETERS short strings). ValueError for a page that
    is not one."""
    from fugal_subnet.tee import endpoint_reads as er

    try:
        read = er.parse_listing(model, raw, hosts)
    except er.ListingRefused as e:
        raise ValueError(f"the listing of {model[:60]} is refused ({e})") from None
    eps = {ep["tag"]: ep for ep in json.loads(bytes(raw))["data"]["endpoints"]}
    out = []
    for h in read.hosts:
        efforts = False
        if h.listed:
            params = eps[h.host].get("supported_parameters")
            if params is not None and (not isinstance(params, list)
                                       or len(params) > MAX_SUPPORTED_PARAMETERS
                                       or not all(isinstance(x, str) and len(x) <= 64
                                                  for x in params)):
                raise ValueError(f"{h.host}: supported_parameters is not a short list of names")
            efforts = "reasoning_effort" in (params or [])
        out.append(HostListing(h.host, h.listed, h.price_in, h.price_out, h.limit, efforts))
    return tuple(sorted(out, key=lambda x: x.host))


def _listing_dict(listing: tuple[HostListing, ...]) -> dict:
    return {h.host: {"listed": h.listed, "price_in": h.price_in, "price_out": h.price_out,
                     "limit": h.limit, "efforts": h.efforts} for h in listing}


def _listing_from_dict(d) -> tuple[HostListing, ...]:
    from fugal_subnet.tee.endpoint_reads import MAX_ANSWER_LIMIT, MAX_PRICE_PICO

    if not isinstance(d, dict) or not 1 <= len(d) <= menu_rules.MENU_MAX_HOSTS:
        raise ValueError("a listing is an object of the pinned hosts")
    out = []
    for host in sorted(d):
        h = d[host]
        if not isinstance(host, str) or not 1 <= len(host) <= menu_rules.MAX_HOST_TAG_LEN:
            raise ValueError("a listed host is a host tag")
        if not isinstance(h, dict) or set(h) != {"listed", "price_in", "price_out", "limit",
                                                 "efforts"}:
            raise ValueError("a host's listing is listed, price_in, price_out, limit, efforts")
        if not isinstance(h["listed"], bool) or not isinstance(h["efforts"], bool):
            raise ValueError("listed and efforts are booleans")
        if not all(_is_int(h[k]) and 0 <= h[k] <= MAX_PRICE_PICO for k in ("price_in", "price_out")):
            raise ValueError("listed prices are integer picodollars")
        if not _is_int(h["limit"]) or not 0 <= h["limit"] <= MAX_ANSWER_LIMIT:
            raise ValueError("a listed limit is an integer token count")
        if not h["listed"] and (h["price_in"], h["price_out"], h["limit"], h["efforts"]) != (
                0, 0, 0, False):
            raise ValueError("a host not listed carries no value")
        out.append(HostListing(host, h["listed"], h["price_in"], h["price_out"], h["limit"],
                               h["efforts"]))
    return tuple(out)


# --- the commitment fields --------------------------------------------------------------


def proposal_field(proposal: str) -> str:
    """A miner's second commitment field: `proposal:<64 hex>`."""
    return PROPOSAL_FIELD_PREFIX + _hex64(proposal, "a proposal hash")


def parse_proposal_field(value) -> str | None:
    """The proposal hash a miner's second field names, or None for anything
    else (no field, another shape, upper case, padding)."""
    if not isinstance(value, str):
        return None
    m = _PROPOSAL_FIELD.fullmatch(value)
    return m.group(1) if m else None


def join_field(epoch: int, delta_hash: str | None) -> str:
    """The root's second commitment field for epoch `epoch`'s joins:
    `join:<epoch id>` (nothing admitted) or `join:<epoch id>:<delta sha256>`."""
    eid = epoch_id_for_block(int(epoch))
    return f"{JOIN_FIELD_PREFIX}{eid}" + ("" if delta_hash is None
                                          else ":" + _hex64(delta_hash, "a delta hash"))


@dataclass(frozen=True)
class JoinPointer:
    """What the root's second field says about one epoch's joins."""

    epoch: int
    delta_hash: str | None           # None: the root admits nothing for the epoch


def parse_join_field(value, epoch: int) -> JoinPointer | None:
    """The root's join pointer for `epoch`, or None when its second field is
    absent, malformed or for another epoch (a late or absent root: no
    joins). The epoch id must be the canonical one of `epoch`."""
    if not isinstance(value, str):
        return None
    m = _JOIN_FIELD.fullmatch(value)
    if m is None or m.group(1) != epoch_id_for_block(int(epoch)):
        return None
    return JoinPointer(int(epoch), m.group(2))


def commitment_fields(raw) -> tuple[str, ...]:
    """Every field of one `Commitments.CommitmentOf` value as text, in order:
    a `Raw<n>` field's bytes as ASCII, any other kind (or bytes that are not
    ASCII) as "". The first is what the SDK's `decode_metadata` reads; the
    second is a proposal's or a join's pointer. () when the value is not a
    commitment. (A fake chain in the tests holds a text, one field, or a
    tuple of texts.)"""
    raw = getattr(raw, "value", raw)
    if isinstance(raw, str):
        return (raw,) if raw else ()
    if isinstance(raw, (list, tuple)) and all(isinstance(x, str) for x in raw):
        return tuple(raw)
    try:
        fields = raw["info"]["fields"]
    except (KeyError, TypeError, IndexError):
        return ()
    out = []
    for f in fields if isinstance(fields, (list, tuple)) else ():
        text = ""
        if isinstance(f, Mapping) and len(f) == 1:
            kind, value = next(iter(f.items()))
            if isinstance(kind, str) and kind.startswith("Raw") and isinstance(value, str):
                try:
                    text = bytes.fromhex(value.removeprefix("0x")).decode("ascii")
                except (ValueError, UnicodeDecodeError):
                    text = ""
        out.append(text)
    return tuple(out)


def read_fields_at(subtensor, netuid: int, block: int) -> dict[str, tuple[str, ...]]:
    """{hotkey: its commitment's fields} for every hotkey of the subnet, read
    pinned at `block` (as commitments.get_commitments_with_blocks reads the
    first field). Raises when the chain cannot be read."""
    from fugal_subnet import chain_reads

    out: dict[str, tuple[str, ...]] = {}
    rows = chain_reads.pinned(
        subtensor, int(block), lambda s: list(s.query_map(module="Commitments",
                                                          name="CommitmentOf", params=[netuid],
                                                          block=int(block))),
        what="Commitment fields")
    for hotkey, value in rows:
        out[str(getattr(hotkey, "value", hotkey))] = commitment_fields(value)
    return out


def commitment_info(fields: Sequence[str]) -> dict:
    """The `info` argument of `Commitments.set_commitment` for these text
    fields: the shape the SDK's publish_metadata_extrinsic builds (one
    `Raw<n>` per field), at most three fields (the pallet's MaxFields) of
    1..128 bytes. ValueError otherwise."""
    data = [str(f).encode("ascii") for f in fields]
    if not 1 <= len(data) <= 3 or any(not 1 <= len(d) <= 128 for d in data):
        raise ValueError("a commitment is one to three fields of 1..128 bytes")
    return {"fields": [[{f"Raw{len(d)}": d} for d in data]]}


def set_commitment_fields(subtensor, wallet, netuid: int, fields: Sequence[str]) -> bool:
    """Write `fields` as this hotkey's commitment in one transaction, with
    the SDK's own call. Returns whether the chain took it; never raises.
    A tests' fake subtensor that offers `set_commitment_fields` is used
    instead."""
    try:
        fake = getattr(subtensor, "set_commitment_fields", None)
        if callable(fake):
            return bool(fake(wallet, netuid, list(fields)))
        info = commitment_info(fields)
        from bittensor.core.extrinsics.pallets import Commitments
        from bittensor.core.settings import DEFAULT_PERIOD

        call = Commitments(subtensor).set_commitment(netuid=int(netuid), info=info)
        response = subtensor.sign_and_send_extrinsic(
            call=call, wallet=wallet, sign_with="hotkey", wait_for_inclusion=True,
            wait_for_finalization=True, period=DEFAULT_PERIOD)
        ok = bool(getattr(response, "success", response))
        if not ok:
            logger.error("Commitment of %d field(s) refused: %s", len(info["fields"][0]),
                         str(getattr(response, "message", ""))[:200])
        return ok
    except Exception as e:  # noqa: BLE001 - reported; the caller decides
        logger.error("Commitment of fields failed (%s)", type(e).__name__)
        return False


# --- the chain read and the delta ----------------------------------------------------------


@dataclass(frozen=True)
class ChainRead:
    """What the join step reads at block B for epoch `epoch`'s joins: the
    proposals miners' second fields name, each hotkey's coldkey, the hash of
    block B (the order's key), and the hotkeys with a verified proof in the
    root's reveal its first field names at B (`reveal`: (epoch id, sha256))."""

    epoch: int
    block: int
    block_hash: str
    proposals: tuple[tuple[str, str], ...]     # (hotkey, proposal hash), sorted by hotkey
    coldkey_of: Mapping[str, str]
    serving: frozenset
    reveal: tuple[str, str] | None


def proposals_at(fields_of: Mapping[str, Sequence[str]],
                 eligible=None) -> tuple[tuple[str, str], ...]:
    """(hotkey, proposal hash) of every commitment whose second field is a
    proposal pointer, of a well-formed hotkey for which `eligible(hotkey)`
    holds (None: every hotkey), sorted by hotkey; at most MAX_CONSIDERED
    (the lowest hotkeys, so the cut is the same on every node). The cut
    comes after the filter: a commitment that cannot be admitted never
    takes another's place."""
    out = []
    for hk in sorted(fields_of):
        f = fields_of[hk]
        p = parse_proposal_field(f[1]) if len(f) >= 2 else None
        if p is None or (eligible is not None and not eligible(hk)):
            continue
        try:
            _hotkey(hk)
        except ValueError:
            continue
        out.append((hk, p))
    return tuple(out[:MAX_CONSIDERED])


def join_margin(blocks_per_epoch: int) -> int:
    """The pinned read's margin before the boundary, in blocks:
    JOIN_READ_MARGIN_BLOCKS on finney's 360-block epoch, and on a shorter
    epoch (a local chain, the tests' worlds) a fifth of it, at least two."""
    return min(JOIN_READ_MARGIN_BLOCKS, max(2, int(blocks_per_epoch) // 5))


def commit_slack(blocks_per_epoch: int) -> int:
    """How many blocks before the boundary the root's join commitment is
    made at the latest: JOIN_COMMIT_SLACK_BLOCKS, or half the margin on a
    shorter epoch."""
    return min(JOIN_COMMIT_SLACK_BLOCKS, join_margin(blocks_per_epoch) // 2)


def read_block(epoch: int, blocks_per_epoch: int) -> int:
    """Block B of epoch `epoch`'s joins: its boundary less the margin."""
    return int(epoch) * int(blocks_per_epoch) - join_margin(blocks_per_epoch)


def place(block_hash: str, proposal: str) -> bytes:
    """A proposal's place in the admission order: HMAC(hash of block B,
    proposal hash). Not known when the proposal is committed."""
    key = bytes.fromhex(_hex64(block_hash.removeprefix("0x").lower(), "block B's hash"))
    return hmac.new(key, b"join-order:" + proposal.encode("ascii"), hashlib.sha256).digest()


def join_limit(serving: int, params: ms.MenuParams | None = None) -> int:
    """Provisional joins one boundary takes: one while fewer than
    `joins_field_hotkeys` hotkeys served, else `max_joins_per_epoch`."""
    params = ms.READER_PARAMS if params is None else params
    return (params.joins_small_field if int(serving) < params.joins_field_hotkeys
            else params.max_joins_per_epoch)


@dataclass(frozen=True)
class Considered:
    """One proposal a delta considers: whose, which, its coldkey (from the
    chain at B), the verdict, the document when the root fetched it, and the
    root's read of its hosts' listing when it read one."""

    hotkey: str
    proposal: str
    coldkey: str
    verdict: str
    document: bytes | None = None
    listing: tuple[HostListing, ...] | None = None


@dataclass(frozen=True)
class JoinDelta:
    epoch: int
    base: str                       # menu_hash of the base menu of `epoch`
    block: int
    block_hash: str
    reveal: tuple[str, str] | None  # the root's reveal whose verified hotkeys served
    serving: int                    # how many hotkeys served
    considered: tuple[Considered, ...]

    def joined(self) -> tuple[Considered, ...]:
        return tuple(c for c in self.considered if c.verdict == JOINED)


def priced(doc: ProposalDoc, listing: Sequence[HostListing]) -> tuple[int, int, int]:
    """(price_in, price_out, max_tokens) of a proposal from the root's
    listing: the dearer pinned host's prices and the rules' answer limit
    (`menu_rules.answer_limit`: every pinned host listed with a limit).
    ValueError when a pinned host is not listed or states no limit."""
    by_host = {h.host: h for h in listing}
    hosts = [by_host.get(t) for t in doc.provider_order]
    if any(h is None or not h.listed for h in hosts):
        raise ValueError("a pinned host is not listed")
    limit = menu_rules.answer_limit(doc.provider_order, {h.host: h.limit for h in hosts})
    return max(h.price_in for h in hosts), max(h.price_out for h in hosts), limit


def is_served(read: ChainRead, hotkey: str) -> bool:
    """Its hotkey has a verified proof in the epoch the root's reveal names."""
    return hotkey in read.serving


def occupied(base: ms.MenuState, ids_joined: set) -> set:
    """The ids a proposal cannot take: on the base menu, or admitted earlier
    in this epoch's order (one entry is never proposed into twice)."""
    return {e.id for e in base.entries} | set(ids_joined)


def coldkey_admitted(taken: Mapping[str, str], coldkey: str) -> bool:
    """Its coldkey has a proposal admitted earlier this epoch (one per coldkey)."""
    return coldkey in taken


def _verdict(base: ms.MenuState, hotkey: str, proposal: str, coldkey: str | None,
             served: bool, raw: bytes | None, listing, allowlist, taken: dict,
             ids_joined: set, params: ms.MenuParams, limit: int):
    """(verdict, document kept, listing kept) of one proposal in its place."""
    if not served or coldkey is None:
        return NOT_SERVED, None, None
    if raw is None or proposal_hash(raw) != proposal:
        return UNFETCHED, None, None
    raw = bytes(raw)
    try:
        doc = parse_document(raw)
    except ValueError:
        return MALFORMED, raw, None
    if doc.proposer != hotkey:
        return MALFORMED, raw, None
    if listing is None:
        return UNREAD, raw, None
    listing = tuple(sorted(listing, key=lambda h: h.host))
    if allowlist is None:
        return REFUSED + "no_list", raw, listing
    try:
        price_in, price_out, max_tokens = priced(doc, listing)
    except ValueError:
        return REFUSED + "not_listed", raw, listing
    ceiling = menu_rules.check_price(price_in, price_out)
    if ceiling is not None:
        return REFUSED + ceiling.code, raw, listing
    if (doc.reasoning is not None and doc.reasoning[0] == "effort"
            and not all(h.efforts for h in listing if h.host in doc.provider_order)):
        return REFUSED + "effort_unsupported", raw, listing
    refusal = menu_rules.check_request(doc.request_dict(max_tokens), allowlist,
                                       host_limits={h.host: h.limit for h in listing})
    if refusal is None and doc.id in occupied(base, ids_joined):
        refusal = menu_rules.Refusal("already_on_menu", doc.id)
    if refusal is None:
        refusal = menu_rules.check_cooldown(
            doc.id, base.epoch,
            retired={r.id: {"cooldown_until": r.cooldown_until} for r in base.retired},
            proposer_coldkey=coldkey, coldkey_cooldowns=base.cooldowns())
    if refusal is None and ms._cost(price_in, price_out, *ms.join_means(base)) <= 0:
        refusal = menu_rules.Refusal("no_cost", "its listed price makes a call cost nothing")
    if refusal is not None:
        return REFUSED + refusal.code, raw, listing
    if coldkey_admitted(taken, coldkey):
        return COLDKEY_ONCE, raw, listing
    joined = len(ids_joined)
    places = ({e.id for e in base.entries} | ids_joined
              | {r.id for r in base.retired if r.cooldown_until > base.epoch})
    if (joined >= limit or len(base.entries) + joined + 1 > params.menu_cap
            or len(base.provisional) + joined + 1 > params.max_provisional
            or len(places | {doc.id}) > params.retired_keep):
        return NO_PLACE, raw, listing
    taken[coldkey] = proposal
    ids_joined.add(doc.id)
    return JOINED, raw, listing


def admit(base: ms.MenuState, read: ChainRead, documents: Mapping[str, bytes | None],
          listings: Mapping[str, Sequence[HostListing] | None],
          allowlist: menu_rules.MenuAllowlist | None,
          params: ms.MenuParams | None = None) -> JoinDelta:
    """The join delta of `read.epoch` on base menu `base`: every proposal of
    the chain read, in the admission order (`place`), with its verdict.

    In that order: its hotkey served the epoch the root's reveal names, and
    has a coldkey at B; a document with its hash came back
    (`documents[proposal]`), and it is a proposal document of that hotkey;
    the root read its model's listing (`listings[proposal]`); every pinned
    host is listed with an answer limit, its listed price is within the
    ceiling (`menu_rules.check_price`), and when it names an effort every
    pinned host takes one; the rules pass it under the list in force
    (`menu_rules.check_request`); it is not on the menu or admitted earlier
    in the order; neither its id nor its coldkey is in cooldown; its listed
    price makes a call cost something; its coldkey has no proposal admitted
    earlier this epoch; and there is a place: fewer than
    `join_limit(serving)` admitted, the cap and `max_provisional` not
    reached, and a place in `retired` for its id. Pure; the same on every
    node given the same read, documents and listings."""
    params = ms.READER_PARAMS if params is None else params
    if base.epoch != read.epoch:
        raise ValueError(f"the base menu is of epoch {base.epoch}, the read for {read.epoch}")
    if allowlist is not None and (allowlist.serial, allowlist.sha256) != (
            base.labs.serial, base.labs.sha256):
        raise ValueError("the joins are judged under the list the base menu names")
    limit = join_limit(len(read.serving), params)
    order = sorted(read.proposals, key=lambda hp: (place(read.block_hash, hp[1]), hp[0]))
    taken: dict = {}
    ids_joined: set = set()
    out = []
    for hk, p in order:
        ck = read.coldkey_of.get(hk)
        verdict, raw, listing = _verdict(base, hk, p, ck, is_served(read, hk), documents.get(p),
                                         listings.get(p), allowlist, taken, ids_joined, params,
                                         limit)
        out.append(Considered(hk, p, ck or "", verdict, raw, listing))
    return JoinDelta(epoch=read.epoch, base=ms.menu_hash(base), block=read.block,
                     block_hash=read.block_hash.removeprefix("0x").lower(), reveal=read.reveal,
                     serving=len(read.serving), considered=tuple(out))


def new_entries(delta: JoinDelta) -> tuple[ms.NewEntry, ...]:
    """The entries the delta admits, in its order, as `apply_joins` takes them."""
    out = []
    for c in delta.joined():
        doc = parse_document(c.document)
        price_in, price_out, max_tokens = priced(doc, c.listing)
        out.append(ms.NewEntry(request=doc.request(max_tokens), price_in=price_in,
                               price_out=price_out, proposal=c.proposal, proposer=c.hotkey,
                               coldkey=c.coldkey))
    return tuple(out)


# --- the delta's one encoding -------------------------------------------------------------


def delta_dict(delta: JoinDelta) -> dict:
    items = []
    for c in delta.considered:
        item = {"hotkey": c.hotkey, "proposal": c.proposal, "coldkey": c.coldkey,
                "verdict": c.verdict}
        if c.document is not None:
            item["document"] = c.document.decode("ascii")
        if c.listing is not None:
            item["listing"] = _listing_dict(c.listing)
        items.append(item)
    return {"domain": DELTA_DOMAIN, "epoch": delta.epoch, "base": delta.base,
            "block": delta.block, "block_hash": delta.block_hash,
            "reveal": (None if delta.reveal is None
                       else {"epoch": delta.reveal[0], "sha256": delta.reveal[1]}),
            "serving": delta.serving, "considered": items}


def delta_bytes(delta: JoinDelta) -> bytes:
    raw = canonical(delta_dict(delta))
    if len(raw) > DELTA_MAX_BYTES:
        raise ValueError(f"a join delta of {len(raw)} bytes is over {DELTA_MAX_BYTES}")
    return raw


def delta_hash(raw: bytes) -> str:
    return hashlib.sha256(bytes(raw)).hexdigest()


def parse_delta(raw) -> JoinDelta:
    """A join delta of canonical bytes, strictly (sizes, keys, types, every
    document's hash its proposal's). ValueError otherwise."""
    d = _parse_canonical(raw, DELTA_MAX_BYTES, "a join delta")
    keys = {"domain", "epoch", "base", "block", "block_hash", "reveal", "serving", "considered"}
    if set(d) != keys or d["domain"] != DELTA_DOMAIN:
        raise ValueError(f"a join delta has exactly {sorted(keys)}")
    for k in ("epoch", "block", "serving"):
        if not _is_int(d[k]) or not 0 <= d[k] <= ms.MAX_INT:
            raise ValueError(f"a join delta's {k} is an integer in range")
    if d["serving"] > MAX_SERVING:
        raise ValueError(f"a join delta states at most {MAX_SERVING} serving hotkeys")
    _hex64(d["base"], "the base menu hash")
    _hex64(d["block_hash"], "block B's hash")
    reveal = d["reveal"]
    if reveal is not None:
        if not isinstance(reveal, dict) or set(reveal) != {"epoch", "sha256"}:
            raise ValueError("a join delta's reveal is {epoch, sha256} or null")
        if not isinstance(reveal["epoch"], str) or not re.fullmatch(r"e\d{8,12}", reveal["epoch"]):
            raise ValueError("a join delta's reveal epoch is an epoch id")
        reveal = (reveal["epoch"], _hex64(reveal["sha256"], "the reveal's sha256"))
    items = d["considered"]
    if not isinstance(items, list) or len(items) > MAX_CONSIDERED:
        raise ValueError(f"a join delta considers at most {MAX_CONSIDERED} proposals")
    out = []
    base_keys = {"hotkey", "proposal", "coldkey", "verdict"}
    for c in items:
        if (not isinstance(c, dict) or not base_keys <= set(c)
                or set(c) - base_keys - {"document", "listing"}):
            raise ValueError("a considered proposal is hotkey, proposal, coldkey, verdict, "
                             "maybe document and listing")
        verdict = c["verdict"]
        if not isinstance(verdict, str) or not (
                verdict in _VERDICTS or (verdict.startswith(REFUSED) and len(verdict) <= 64)):
            raise ValueError("an unknown verdict")
        proposal = _hex64(c["proposal"], "a proposal hash")
        doc = None
        if "document" in c:
            if not isinstance(c["document"], str) or len(c["document"]) > PROPOSAL_MAX_BYTES:
                raise ValueError("a considered proposal's document is text of bounded size")
            doc = c["document"].encode("ascii")
            if proposal_hash(doc) != proposal:
                raise ValueError("a document in the delta does not hash to its proposal")
        listing = _listing_from_dict(c["listing"]) if "listing" in c else None
        coldkey = c["coldkey"]
        if not isinstance(coldkey, str) or (coldkey and _hotkey(coldkey, "a coldkey") != coldkey):
            raise ValueError("a coldkey is printable ASCII")
        out.append(Considered(_hotkey(c["hotkey"]), proposal, coldkey, verdict, doc, listing))
    hotkeys = [c.hotkey for c in out]
    if len(set(hotkeys)) != len(hotkeys):
        raise ValueError("a join delta considers one proposal per hotkey")
    delta = JoinDelta(epoch=d["epoch"], base=d["base"], block=d["block"],
                      block_hash=d["block_hash"], reveal=reveal, serving=d["serving"],
                      considered=tuple(out))
    if delta_bytes(delta) != bytes(raw):
        raise ValueError("the bytes are not the canonical delta they describe")
    return delta


def claimed_read(delta: JoinDelta):
    """(chain read, documents, listings) a delta states, as `admit` takes
    them: what a node that does not read the chain (a TD) re-runs the
    admission on. A proposal stated as not served is not served; every other
    one's coldkey is the one stated. Serving hotkeys that filed no proposal
    are stood in for, so the count is the one stated."""
    served = {c.hotkey for c in delta.considered if c.verdict != NOT_SERVED}
    if len(served) > delta.serving:
        raise ValueError("the delta states fewer serving hotkeys than proposals it served")
    filler = {f"\x00serving-{i}" for i in range(delta.serving - len(served))}
    read = ChainRead(epoch=delta.epoch, block=delta.block, block_hash=delta.block_hash,
                     proposals=tuple(sorted((c.hotkey, c.proposal) for c in delta.considered)),
                     coldkey_of={c.hotkey: c.coldkey for c in delta.considered if c.coldkey},
                     serving=frozenset(served | filler), reveal=delta.reveal)
    return (read, {c.proposal: c.document for c in delta.considered},
            {c.proposal: c.listing for c in delta.considered})


def check_delta(base: ms.MenuState, raw, allowlist: menu_rules.MenuAllowlist | None, *,
                block: int | None = None,
                params: ms.MenuParams | None = None) -> tuple[ms.NewEntry, ...]:
    """The entries a committed delta admits on `base`, as every node checks
    it without reading the chain: bytes within DELTA_MAX_BYTES, canonical and
    well-formed; for base's epoch (and pinned at `block` when given); and the
    admission re-run on the delta's own contents (`claimed_read`) gives
    exactly the delta, every verdict and its order. DeltaStale when it names
    another base menu; DeltaRefused for anything else. `allowlist` is the
    list `base` names (None when it names none: nothing can join)."""
    params = ms.READER_PARAMS if params is None else params
    try:
        delta = parse_delta(raw)
    except ValueError as e:
        raise DeltaRefused(f"the join delta is malformed: {e}") from None
    if delta.epoch != base.epoch:
        raise DeltaRefused(f"the join delta is for epoch {delta.epoch}, not {base.epoch}")
    if block is not None and delta.block != int(block):
        raise DeltaRefused(f"the join delta read the chain at block {delta.block}, not {block}")
    if delta.base != ms.menu_hash(base):
        raise DeltaStale(f"the join delta was computed on menu {delta.base[:16]}, not on the "
                         f"base menu of epoch {base.epoch}")
    try:
        again = admit(base, *claimed_read(delta), allowlist, params)
    except ValueError as e:
        raise DeltaRefused(f"the join delta cannot be re-run ({e})") from None
    if delta_bytes(again) != delta_bytes(delta):
        diff = [f"{a.hotkey[:12]}: {a.verdict} vs {b.verdict}"
                for a, b in zip(delta.considered, again.considered) if a != b]
        raise DeltaRefused("the join delta states verdicts the rules do not give "
                           f"({'; '.join(diff[:3]) or 'its order or contents'})")
    try:
        entries = new_entries(delta)
        ms.apply_joins(base, entries, params)
    except ValueError as e:
        raise DeltaRefused(f"the menu cannot take the delta's joins ({e})") from None
    return entries


def audit(expected: JoinDelta, got: JoinDelta | None) -> list[str]:
    """Where the root's delta (`got`; None when it committed none for the
    epoch) differs from the delta this node recomputed from its own pinned
    read (`expected`): [] when they are the same delta."""
    if got is None:
        return ([] if not expected.joined()
                else [f"the root admitted nothing; {len(expected.joined())} proposal(s) join here"])
    out = []
    for name in ("epoch", "base", "block", "block_hash", "reveal", "serving"):
        if getattr(expected, name) != getattr(got, name):
            out.append(f"{name}: {getattr(got, name)!r} (root) vs {getattr(expected, name)!r}")
    mine = {c.hotkey: c for c in expected.considered}
    theirs = {c.hotkey: c for c in got.considered}
    for hk in sorted(set(mine) | set(theirs)):
        a, b = mine.get(hk), theirs.get(hk)
        if a is None:
            out.append(f"{hk[:16]}: only in the root's delta")
        elif b is None:
            out.append(f"{hk[:16]}: a proposal on chain the root's delta leaves out")
        elif (a.proposal, a.coldkey, a.verdict) != (b.proposal, b.coldkey, b.verdict):
            out.append(f"{hk[:16]}: {b.verdict} (root) vs {a.verdict}")
    if not out and [c.hotkey for c in expected.considered] != [c.hotkey for c in got.considered]:
        out.append("the order of the proposals")
    return out


def contents_of(delta: JoinDelta | None) -> tuple[dict, dict]:
    """({proposal: document}, {proposal: listing}) of everything a delta
    carries: what a follower re-runs the admission with (it fetches nothing
    from miners and takes the root's read of the listing)."""
    if delta is None:
        return {}, {}
    return ({c.proposal: c.document for c in delta.considered if c.document is not None},
            {c.proposal: c.listing for c in delta.considered if c.listing is not None})


def served_hotkeys(reveal) -> frozenset:
    """The hotkeys a root reveal lists as having a verified proof
    (`served_v4`), or the empty set."""
    got = reveal.get(SERVED_FIELD) if isinstance(reveal, Mapping) else None
    if not isinstance(got, list) or len(got) > 4096 or not all(isinstance(h, str) for h in got):
        return frozenset()
    return frozenset(got)


def reveal_for_joins(epoch: int) -> str:
    """The epoch id of the reveal whose verified hotkeys may propose for
    `epoch`'s joins: the epoch before the one they file in (epoch - 2)."""
    return epoch_id_for_block(int(epoch) - 2)


def candidates(fields_of: Mapping[str, Sequence[str]], coldkey_of: Mapping[str, str], *,
               epoch: int, block: int, block_hash: str, serving: Iterable[str],
               reveal: tuple[str, str] | None) -> ChainRead:
    """The chain read of `epoch`'s joins from a pinned read at `block`: the
    proposals of hotkeys registered at `block` (a coldkey in `coldkey_of`)
    that served the epoch the root's reveal names (`serving`), then cut to
    MAX_CONSIDERED (`proposals_at`)."""
    serving = frozenset(serving)
    coldkey_of = dict(coldkey_of)
    return ChainRead(epoch=int(epoch), block=int(block),
                     block_hash=str(block_hash).removeprefix("0x").lower(),
                     proposals=proposals_at(fields_of,
                                            lambda hk: hk in serving and hk in coldkey_of),
                     coldkey_of=coldkey_of, serving=serving, reveal=reveal)
