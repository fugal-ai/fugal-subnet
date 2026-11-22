"""Protocol v4: the rules a menu entry must pass, and the signed lists they read.

Design: docs/decisions/LIVE_MENU_V4.md, sections 6 (rules compiled into the
image) and 13.4 (the device list). **Nothing on the v1-v3 path imports this
module**; it takes effect only from a v4 activation block that no release has
set.

Two parts.

**Rules.** In v4 the menu is published state, so the image carries the rules
an entry must pass instead of the entries. The miner's TD applies them to
every entry of a menu it is handed and to every proposal it files; validators
apply them to every proposal and to the next menu. Each check is a pure
function of its arguments and returns `None` (passes) or a `Refusal` naming
the rule: no clock, no network, no environment, so every node reaches the
same verdict on the same bytes. The constants here are compiled: they change
by release only, and none is read from the environment.

**Signed lists.** Which labs and hosts an entry may name is data, signed by a
release key, so the owner can add a lab without a release. The device list of
section 13.4 uses the same format and the same signature. A list is canonical
JSON with a serial number; the signature is an OpenSSH signature
(`ssh-keygen -Y sign`) under the namespace `LIST_NAMESPACE`, made by a key
that `signed_lists/allowed_signers` lists for that namespace. Verification
happens in this process (`verify_signed_list`), offline, and refuses a
signature that does not verify, one made under another namespace, a signer
that is not listed for lists, a serial not above the one held, and anything
malformed, repeated or oversized.

Why the signature is checked here and not by running `ssh-keygen -Y verify`:
the verdict is consensus input, and an external program's verdict depends on
which OpenSSH each node has, on whether one is installed at all (the miner
image has none), and on a child process starting. `cryptography` is already a
pinned dependency of every `--live` node and of the image. The format read
here is a strict subset of what OpenSSH writes, and the tests hold the two to
the same verdicts on real `ssh-keygen` output.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from fugal_subnet.routing_protocol import validate_menu_v3

# ════════════════════════════════════════════════════════════════════════════
# Rules
# ════════════════════════════════════════════════════════════════════════════

# An entry may pin at most this many hosts, in order, with no fallback.
MENU_MAX_HOSTS = 2
# The answer budget of an entry is the smallest answer limit among its pinned
# hosts, capped here (v3's rule and v3's cap).
MAX_ANSWER_TOKENS = 131072
# The dearest price an entry may have, in picodollars per token: $25 per
# million prompt tokens and $100 per million completion tokens. Chosen from
# the real menu (2026-11-22): the pinned seed's dearest entries
# (anthropic/claude-fable-5.2, openai/gpt-6.2-astra) list $8 / $40, and the
# 19-lab draft's flagships are at or below that, so the ceiling leaves 3.1x
# headroom on input and 2.5x on output. A proposal above it is refused at
# admission (`check_price`, every node's and TD's `check_delta` included),
# and an entry whose refreshed price would rise above it leaves the menu
# (menu_state, reason `price_ceiling`). With MAX_ANSWER_TOKENS it bounds one
# call: at most $13.11 of completion tokens.
MAX_PRICE_IN_PICO = 25_000_000
MAX_PRICE_OUT_PICO = 100_000_000
# The reasoning settings an entry may carry: one of these efforts, or
# `{"enabled": true}` for a model with a switch and no levels. The set is the
# settings the v3 menu uses.
REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
# The quantization names of OpenRouter's provider routing (its documented
# `quantizations` values, read 2026-11-18). An entry pins exactly one.
QUANTIZATIONS = ("int4", "int8", "fp4", "mxfp4", "nvfp4", "fp6", "fp8", "mxfp8", "fp16", "bf16",
                 "fp32", "unknown")
UNKNOWN_QUANTIZATION = "unknown"
# Service tiers of a host. Each is a different product at a different price,
# so a tier is never a pinned host and never a listed one. The names are the
# ones OpenRouter's listing has shown as a host tag with a service tier
# (data/observations/openrouter_endpoints_2026-11-18.json).
HOST_TIER_SUFFIXES = ("flex", "fast", "priority", "spot", "ultrafast", "turbo", "highspeed")

# The refusals that depend on the signed list. An entry that passed under one
# list and fails under a newer one with one of these lost its lab or a host
# (section 4.4, step 1).
ALLOWLIST_REFUSALS = frozenset({"lab_not_allowed", "host_not_allowed", "unknown_quantization"})

MAX_MODEL_ID_LEN = 200
MAX_LAB_LEN = 64
MAX_HOST_LEN = 64
MAX_HOST_TAG_LEN = 96

_NAME = r"[a-z0-9][a-z0-9._-]*"
_LAB_RE = re.compile(_NAME)
_MODEL_RE = re.compile(rf"{_NAME}/{_NAME}")
# A listed host: `provider`, or `provider/variant[/region]` when the owner
# lists that variant by name. A host tag in an entry may carry one more
# segment, the entry's quantization (`deepinfra/fp8`).
_HOST_RE = re.compile(rf"{_NAME}(?:/{_NAME}){{0,2}}")
_HOST_TAG_RE = re.compile(rf"{_NAME}(?:/{_NAME}){{0,3}}")


@dataclass(frozen=True)
class Refusal:
    """Why something was refused: a stable `code` naming the rule, and a
    `detail` naming the value. Both are functions of the input alone."""
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True)
class MenuAllowlist:
    """The labs and hosts an entry may name. `first_party` maps a lab to the
    hosts that serve its models first-party (its own, or under its contract:
    `first_party_map`), sorted; such a host may be pinned with no
    quantization for that lab's models only. `signer` is the fingerprint of
    the key whose signature was verified, and None for a list that was only
    parsed (a draft, or a list built in a test)."""
    serial: int
    labs: frozenset[str]
    hosts: frozenset[str]
    first_party: tuple[tuple[str, frozenset[str]], ...] = ()
    sha256: str = ""
    signer: str | None = None

    def first_party_of(self, lab: str) -> frozenset[str]:
        """The hosts first-party for `lab`'s models."""
        return next((hs for lb, hs in self.first_party if lb == lab), frozenset())

    @property
    def first_party_hosts(self) -> frozenset[str]:
        """Every host first-party for some lab."""
        return frozenset(h for _lb, hs in self.first_party for h in hs)

    def first_party_dict(self) -> dict[str, list[str]]:
        return {lb: sorted(hs) for lb, hs in self.first_party}


def first_party_map(mapping) -> tuple[tuple[str, frozenset[str]], ...]:
    """`MenuAllowlist.first_party` of {lab: hosts}: sorted by lab, labs with
    no host left out."""
    return tuple((lab, frozenset(mapping[lab])) for lab in sorted(mapping) if mapping[lab])


@dataclass(frozen=True)
class DeviceList:
    """The device ids (sha256 of the platform's PPID, hex) of section 13.4."""
    serial: int
    devices: frozenset[str]
    sha256: str = ""
    signer: str | None = None


def _show(value) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _is_int(value) -> bool:
    return type(value) is int


def entry_id(model: str, reasoning: Mapping) -> str:
    """The id of the entry that calls `model` with `reasoning`:
    `<model>@<effort>`, or the model id for a setting without an effort. One
    (model, reasoning) has one id, so two entries for the same call cannot
    sit on the menu under two names."""
    effort = reasoning.get("effort") if isinstance(reasoning, Mapping) else None
    return f"{model}@{effort}" if effort is not None else model


def host_identity(tag: str) -> str:
    """The host a tag names: the tag without its quantization segment
    (`deepinfra/fp8` -> `deepinfra`, `google-vertex/global` unchanged). This
    is what the signed list holds."""
    head, *rest = str(tag).split("/")
    return "/".join([head] + [s for s in rest if s not in QUANTIZATIONS])


def pinned_quantization(order: Iterable[str]) -> str | None:
    """The one quantization a host order pins: the segment every tag carries,
    `unknown` if none carries one, None if the tags disagree."""
    found = set()
    for tag in order:
        segments = [s for s in str(tag).split("/")[1:] if s in QUANTIZATIONS]
        found.add(segments[0] if len(segments) == 1 else
                  UNKNOWN_QUANTIZATION if not segments else None)
    return found.pop() if len(found) == 1 else None


def answer_limit(order: Iterable[str], host_limits: Mapping[str, int], *,
                 on_menu: bool = False) -> int:
    """min(MAX_ANSWER_TOKENS, the smallest answer limit among the pinned
    hosts). `host_limits` maps each host tag the listing shows to the answer
    limit it states, 0 when it states none; a tag that is not a key is a host
    the listing does not show.

    A new proposal (the default): every pinned host must be listed and state
    a limit. What is proposed is what will be called.

    An entry already on the menu (`on_menu`): a pinned host the listing does
    not show is skipped, and so is a listed one that states no limit,
    provided at least one pinned host is listed. With no stated limit the
    answer is limited by the cap alone. An entry whose hosts have all left
    the listing is retired by the health rule (section 9.2), not here.

    Raises ValueError when the rule cannot be met or a limit is not a
    non-negative integer.
    """
    order = list(order)
    if not order:
        raise ValueError("no pinned host")
    known = host_limits if isinstance(host_limits, Mapping) else {}
    limits, listed = [], 0
    for tag in order:
        if tag not in known:
            if on_menu:
                continue
            raise ValueError(f"no answer limit for pinned host {_show(tag)}")
        limit = known[tag]
        if not _is_int(limit) or limit < 0 or (limit == 0 and not on_menu):
            raise ValueError(f"no answer limit for pinned host {_show(tag)}")
        listed += 1
        if limit:
            limits.append(limit)
    if not listed:
        raise ValueError(f"none of the pinned hosts {_show(order)} is listed")
    return min([MAX_ANSWER_TOKENS] + limits)


def _check_reasoning(reasoning) -> Refusal | None:
    if isinstance(reasoning, Mapping) and len(reasoning) == 1:
        (key, value), = reasoning.items()
        if key == "effort" and type(value) is str and value in REASONING_EFFORTS:
            return None
        if key == "enabled" and value is True:
            return None
    return Refusal("reasoning_not_allowed",
                   f"reasoning {_show(reasoning)} is not one effort of {list(REASONING_EFFORTS)} "
                   "or {'enabled': True}")


def _check_host(tag, quantization: str, lab: str, allowlist: MenuAllowlist) -> Refusal | None:
    if type(tag) is not str or len(tag) > MAX_HOST_TAG_LEN or not _HOST_TAG_RE.fullmatch(tag):
        return Refusal("host_shape", f"host {_show(tag)} is not provider[/variant][/quantization]")
    segments = tag.split("/")
    tiers = [s for s in segments[1:] if s in HOST_TIER_SUFFIXES]
    if tiers:
        return Refusal("host_tier", f"host {tag!r} names the service tier {tiers[0]!r}")
    carried = [s for s in segments[1:] if s in QUANTIZATIONS]
    expected = [] if quantization == UNKNOWN_QUANTIZATION else [quantization]
    if carried != expected:
        return Refusal("quantization_mismatch",
                       f"host {tag!r} carries quantization {carried or 'none'}, the entry "
                       f"pins {quantization!r}")
    identity = host_identity(tag)
    if identity not in allowlist.hosts:
        return Refusal("host_not_allowed", f"host {identity!r} is not on the signed list")
    if (quantization == UNKNOWN_QUANTIZATION and identity != lab
            and identity not in allowlist.first_party_of(lab)):
        return Refusal("unknown_quantization",
                       f"host {identity!r} is neither {lab!r}'s own nor listed first-party for "
                       f"{lab!r}, so the entry must pin a quantization")
    return None


def check_request(request, allowlist: MenuAllowlist, *,
                  host_limits: Mapping[str, int] | None = None,
                  on_menu: bool = False) -> Refusal | None:
    """The rules on what an entry calls: `model`, `reasoning`, `provider`,
    `quantization`, `max_tokens`. Other keys are not read. `on_menu` says the
    request is an entry's already on the menu and not a new proposal's: the
    two differ only in how `answer_limit` treats a pinned host the listing
    does not show.

    - the model is `<lab>/<name>` in lower case with no `:variant`, and its
      lab is on the signed list;
    - the reasoning setting is one of the fixed set;
    - the provider preference is exactly a pinned `order` of at most
      `MENU_MAX_HOSTS` distinct hosts and `allow_fallbacks: false`;
    - one quantization across the hosts: every host tag carries it, or, for
      `unknown`, none carries any and every host is the lab's own or listed
      first-party;
    - every host is on the signed list; no host is a service tier;
    - `max_tokens` is a positive integer up to `MAX_ANSWER_TOKENS`; with
      `host_limits` (the answer limits read from the listing) it must equal
      `answer_limit(order, host_limits, on_menu=on_menu)`.
    """
    if not isinstance(request, Mapping):
        return Refusal("malformed", f"an entry is an object, not {_show(request)}")
    model = request.get("model")
    if type(model) is not str:
        return Refusal("malformed", f"`model` must be a string, not {_show(model)}")
    if ":" in model:
        return Refusal("model_variant", f"model {_show(model)} names a variant after ':'")
    if len(model) > MAX_MODEL_ID_LEN or not _MODEL_RE.fullmatch(model):
        return Refusal("model_shape", f"model {_show(model)} is not <lab>/<name> in lower case")
    lab = model.split("/", 1)[0]
    if lab not in allowlist.labs:
        return Refusal("lab_not_allowed", f"lab {lab!r} of {model!r} is not on the signed list")

    reasoning = request.get("reasoning")
    refusal = _check_reasoning(reasoning)
    if refusal is not None:
        return refusal

    provider = request.get("provider")
    if not isinstance(provider, Mapping) or "order" not in provider:
        return Refusal("provider_shape", "`provider` must be {order, allow_fallbacks: false}")
    extra = sorted(str(k) for k in provider if k not in ("order", "allow_fallbacks"))
    if extra:
        return Refusal("provider_shape", f"`provider` carries {_show(extra)} beyond order and "
                                         "allow_fallbacks")
    if "allow_fallbacks" not in provider:
        return Refusal("provider_shape", "`provider` must set allow_fallbacks to false")
    if provider["allow_fallbacks"] is not False:
        return Refusal("fallbacks_allowed", "`provider.allow_fallbacks` must be false")
    order = provider["order"]
    if (type(order) is not list or not order
            or not all(type(tag) is str for tag in order)):
        return Refusal("provider_shape", "`provider.order` must be a non-empty list of hosts")
    if len(order) > MENU_MAX_HOSTS:
        return Refusal("too_many_hosts", f"{len(order)} hosts pinned, at most {MENU_MAX_HOSTS}")
    if len(set(order)) != len(order):
        return Refusal("duplicate_host", f"`provider.order` {_show(order)} names a host twice")

    quantization = request.get("quantization")
    if type(quantization) is not str or quantization not in QUANTIZATIONS:
        return Refusal("quantization_not_allowed",
                       f"quantization {_show(quantization)} is not one of {list(QUANTIZATIONS)}")
    for tag in order:
        refusal = _check_host(tag, quantization, lab, allowlist)
        if refusal is not None:
            return refusal

    max_tokens = request.get("max_tokens")
    if not _is_int(max_tokens) or not 1 <= max_tokens <= MAX_ANSWER_TOKENS:
        return Refusal("answer_limit", f"`max_tokens` {_show(max_tokens)} is not an integer in "
                                       f"1..{MAX_ANSWER_TOKENS}")
    if host_limits is not None:
        try:
            expected = answer_limit(order, host_limits, on_menu=on_menu)
        except ValueError as e:
            return Refusal("answer_limit_unknown", str(e))
        if max_tokens != expected:
            return Refusal("answer_limit", f"`max_tokens` {max_tokens} is not the smallest "
                                           f"pinned answer limit, capped: {expected}")
    return None


def check_price(price_in, price_out) -> Refusal | None:
    """The price ceiling: picodollars per prompt and completion token at
    most MAX_PRICE_IN_PICO and MAX_PRICE_OUT_PICO. None, or the refusal
    `price_above_ceiling`."""
    if not _is_int(price_in) or not _is_int(price_out):
        raise ValueError("a price is an integer number of picodollars per token")
    if price_in > MAX_PRICE_IN_PICO or price_out > MAX_PRICE_OUT_PICO:
        return Refusal("price_above_ceiling",
                       f"{price_in}/{price_out} picodollars a token, above the ceiling of "
                       f"{MAX_PRICE_IN_PICO}/{MAX_PRICE_OUT_PICO}")
    return None


def check_entry(entry, allowlist: MenuAllowlist, *,
                host_limits: Mapping[str, int] | None = None) -> Refusal | None:
    """Every compiled rule on one entry of a menu: `check_request` as for an
    entry on the menu (a pinned host the listing no longer shows does not
    refuse it while another is listed), then

    - `id` is `entry_id(model, reasoning)`;
    - `price_in` and `price_out`, when present, are integers;
    - `routing_protocol.validate_menu_v3`'s shape rules, unchanged, on the
      entry's v3 projection (prices default to 0 for an entry without them).
    """
    refusal = check_request(entry, allowlist, host_limits=host_limits, on_menu=True)
    if refusal is not None:
        return refusal
    model, reasoning = entry["model"], entry["reasoning"]
    expected = entry_id(model, reasoning)
    if entry.get("id") != expected or type(entry.get("id")) is not str:
        return Refusal("id_mismatch", f"id {_show(entry.get('id'))} is not {expected!r}, the id "
                                      "of its model and reasoning setting")
    for key in ("price_in", "price_out"):
        if key in entry and not _is_int(entry[key]):
            return Refusal("price_type", f"`{key}` {_show(entry[key])} is not an integer")
    row = {"id": expected, "model": model, "in": entry.get("price_in", 0),
           "out": entry.get("price_out", 0), "reasoning": dict(reasoning),
           "max_tokens": entry["max_tokens"],
           "provider": {"order": list(entry["provider"]["order"]), "allow_fallbacks": False}}
    try:
        validate_menu_v3([row])
    except ValueError as e:
        return Refusal("shape", str(e))
    return None


def check_menu(entries, allowlist: MenuAllowlist) -> Refusal | None:
    """A whole menu: a non-empty list, every entry passes `check_entry`, no
    id twice. The first failing entry decides, in list order."""
    if type(entries) is not list or not entries:
        return Refusal("malformed", "a menu is a non-empty list of entries")
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        refusal = check_entry(entry, allowlist)
        if refusal is not None:
            name = entry.get("id") if isinstance(entry, Mapping) else None
            return Refusal(refusal.code, f"entry {index} ({_show(name)}): {refusal.detail}")
        if entry["id"] in seen:
            return Refusal("duplicate_entry", f"entry {index}: id {entry['id']!r} is on the menu "
                                              "twice")
        seen.add(entry["id"])
    return None


def _slot(item) -> tuple[str, str] | None:
    """(model, canonical reasoning) of an entry or proposal; None if it has none."""
    if not isinstance(item, Mapping):
        return None
    model, reasoning = item.get("model"), item.get("reasoning")
    if type(model) is not str or not isinstance(reasoning, Mapping):
        return None
    try:
        return model, json.dumps(dict(reasoning), sort_keys=True)
    except (TypeError, ValueError):
        return None


def _call(item) -> tuple | None:
    """The part of a request a host change alters: (order, quantization)."""
    provider = item.get("provider") if isinstance(item, Mapping) else None
    order = provider.get("order") if isinstance(provider, Mapping) else None
    if type(order) is not list:
        return None
    return tuple(order), item.get("quantization")


def is_request_update(request, entries: Iterable) -> bool:
    """True when `request` names a model and reasoning setting already on the
    menu with a different host order or quantization: section 4.4's request
    update, which goes through the proposal path and bumps the entry's `rev`."""
    slot = _slot(request)
    return slot is not None and any(
        _slot(e) == slot and _call(e) != _call(request) for e in entries)


def check_duplicate(request, *, entries: Iterable, pending: Iterable) -> Refusal | None:
    """No second proposal for a model and reasoning setting that is already
    being decided, and none identical to an entry on the menu.

    `pending` is every proposal not yet joined or dropped (the state's
    `pending` and `queue`); `entries` is the menu. Each item is an object
    with at least `model` and `reasoning`. The same model and setting with a
    different host order or quantization is a request update and passes.
    """
    slot = _slot(request)
    if slot is None:
        return Refusal("malformed", "a proposal names a model and a reasoning setting")
    name = entry_id(request["model"], request["reasoning"])
    if any(_slot(p) == slot for p in pending):
        return Refusal("already_pending", f"{name!r} already has a proposal being decided")
    if any(_slot(e) == slot and _call(e) == _call(request) for e in entries):
        return Refusal("already_on_menu", f"{name!r} is on the menu with this request")
    return None


def check_cooldown(eid: str, epoch: int, *, retired: Mapping,
                   proposer_coldkey: str | None = None,
                   coldkey_cooldowns: Mapping[str, int] | None = None) -> Refusal | None:
    """An entry, or a proposer, in cooldown cannot be proposed at `epoch`.

    `retired` maps an entry id to its record (`{..., cooldown_until}`), as in
    the menu state; `coldkey_cooldowns` maps a coldkey to the first epoch it
    may file again. A cooldown ends at its epoch: filing at `cooldown_until`
    passes. A record whose `cooldown_until` cannot be read refuses. The state
    views themselves are the caller's: one of the wrong type is a TypeError,
    never a pass.
    """
    if not _is_int(epoch):
        raise TypeError(f"epoch {epoch!r} is not an integer")
    if not isinstance(retired, Mapping):
        raise TypeError("`retired` is a mapping of entry id to its record")
    if coldkey_cooldowns is not None and not isinstance(coldkey_cooldowns, Mapping):
        raise TypeError("`coldkey_cooldowns` is a mapping of coldkey to an epoch")
    if eid in retired:
        record = retired[eid]
        until = record.get("cooldown_until") if isinstance(record, Mapping) else None
        if not _is_int(until):
            return Refusal("entry_cooldown", f"{eid!r} has a cooldown record that "
                                             "cannot be read")
        if epoch < until:
            return Refusal("entry_cooldown", f"{eid!r} is in cooldown until epoch {until}")
    if proposer_coldkey is not None and coldkey_cooldowns is not None \
            and proposer_coldkey in coldkey_cooldowns:
        until = coldkey_cooldowns[proposer_coldkey]
        if not _is_int(until):
            return Refusal("proposer_cooldown", "the proposer has a cooldown record that "
                                                "cannot be read")
        if epoch < until:
            return Refusal("proposer_cooldown", f"the proposer is in cooldown until epoch {until}")
    return None


def check_proposal(request, allowlist: MenuAllowlist, *, epoch: int, entries: Iterable,
                   pending: Iterable, retired: Mapping,
                   host_limits: Mapping[str, int] | None = None,
                   proposer_coldkey: str | None = None,
                   coldkey_cooldowns: Mapping[str, int] | None = None) -> Refusal | None:
    """A proposal filed at `epoch`, against the rules and a view of the menu
    state: `check_request`, then `check_duplicate`, then `check_cooldown`.
    The first refusal decides."""
    refusal = check_request(request, allowlist, host_limits=host_limits)
    if refusal is None:
        refusal = check_duplicate(request, entries=entries, pending=pending)
    if refusal is None:
        refusal = check_cooldown(entry_id(request["model"], request["reasoning"]), epoch,
                                 retired=retired, proposer_coldkey=proposer_coldkey,
                                 coldkey_cooldowns=coldkey_cooldowns)
    return refusal


# ════════════════════════════════════════════════════════════════════════════
# Signed lists
# ════════════════════════════════════════════════════════════════════════════

# The OpenSSH signature namespace of a signed list. A signature a release key
# made for anything else (a git tag is namespace `git`) is not a signature
# over a list, and a list signature is not one over a tag.
LIST_NAMESPACE = "fugal-list@fugalai.com"
# The format of each kind's bytes. The menu allowlist is format 2: its
# first-party hosts are listed per lab (`first_party`), where format 1 had
# one lab-agnostic `first_party_hosts` array. A format-1 allowlist is refused.
LIST_FORMATS = {"fugal-menu-allowlist": 2, "fugal-device-list": 1}
# Both kinds are signed under the one namespace, so the kind is inside the
# signed bytes: a device list cannot be presented as a menu allowlist.
KIND_MENU_ALLOWLIST = "fugal-menu-allowlist"
KIND_DEVICE_LIST = "fugal-device-list"
# Serial 0 marks a draft: it is refused whatever signs it. A signed list
# starts at 1 and a newer one must carry a higher serial.
DRAFT_SERIAL = 0
MAX_SERIAL = 2**53 - 1

MAX_LIST_BYTES = 262144
MAX_LIST_ITEMS = 4096
MAX_SIGNATURE_BYTES = 2048
MAX_SIGNERS_BYTES = 16384

SIGNED_LISTS_DIR = Path(__file__).resolve().parent / "signed_lists"
# The keys that may sign a list: a byte-identical copy of
# deploy/keys/allowed_signers (which is outside the image), pinned here and in
# scripts/check_safety_invariants.py like the TPM root. A change to the
# release signers is a change to both pins, in the same commit.
SIGNERS_PATH = SIGNED_LISTS_DIR / "allowed_signers"
LIST_SIGNERS_SHA256 = "5a6bc2ef0b3fd93d193f4162791b62364f10f2c48025a3b32c05903efde41ead"
SHIPPED_LISTS = {KIND_MENU_ALLOWLIST: "menu_allowlist.json", KIND_DEVICE_LIST: "device_list.json"}

_DEVICE_RE = re.compile(r"[0-9a-f]{64}")
_LIST_FIELDS = {KIND_MENU_ALLOWLIST: ("labs", "hosts", "first_party"),
                KIND_DEVICE_LIST: ("devices",)}


class ListRefused(Exception):
    """A signed list was refused. `code` names the reason:

    oversized, malformed, duplicate, not_canonical, wrong_kind, draft,
    rollback (the list), signature_malformed, wrong_namespace,
    signer_not_allowed, bad_signature (the signature)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail

    @property
    def refusal(self) -> Refusal:
        return Refusal(self.code, self.detail)


def _valid_lab(item) -> bool:
    return type(item) is str and len(item) <= MAX_LAB_LEN and bool(_LAB_RE.fullmatch(item))


def _valid_host(item) -> bool:
    if type(item) is not str or len(item) > MAX_HOST_LEN or not _HOST_RE.fullmatch(item):
        return False
    return not any(s in HOST_TIER_SUFFIXES or s in QUANTIZATIONS for s in item.split("/"))


def _valid_device(item) -> bool:
    return type(item) is str and bool(_DEVICE_RE.fullmatch(item))


_ITEM_RULES = {"labs": _valid_lab, "hosts": _valid_host,
               "devices": _valid_device}


def canonical_list_bytes(kind: str, serial: int, **fields) -> bytes:
    """The one byte string of a list: JSON with sorted keys, one sorted item
    per line, ASCII, a final newline. These are the bytes that are signed
    and hashed. A menu allowlist's `first_party` is {lab: hosts} (or
    `MenuAllowlist.first_party`): an object of sorted labs, each a sorted
    array. (Encoding only: `parse_unsigned_list` is what validates.)"""
    obj = {"format": LIST_FORMATS[kind], "kind": kind, "serial": serial}
    for name in _LIST_FIELDS[kind]:
        if name == "first_party":
            fp = fields[name]
            fp = dict(fp) if not isinstance(fp, dict) else fp
            obj[name] = {lab: sorted(fp[lab]) for lab in sorted(fp)}
        else:
            obj[name] = sorted(fields[name])
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


class _RepeatedKey(ValueError):
    pass


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise _RepeatedKey(key)
        out[key] = value
    return out


def _not_a_number(text):
    raise ValueError(f"{text} is not an integer")


def parse_unsigned_list(payload: bytes, kind: str) -> MenuAllowlist | DeviceList:
    """The format rules of a list, **with no signature check**: the result is
    not a signed list (`signer` is None), whatever its serial. For drafts and
    tools; the neurons use `verify_signed_list` and `shipped_list`, and
    scripts/check_safety_invariants.py holds that nothing else under
    fugal_subnet/ or neurons/ calls this.

    Refuses (ListRefused): bytes over `MAX_LIST_BYTES` or an array over
    `MAX_LIST_ITEMS` (oversized); anything that is not ASCII JSON for an
    object with exactly `format`, `kind`, `serial` and the kind's arrays, an
    item of the wrong shape, an empty `labs` or `hosts`, a `first_party`
    that is not an object of listed labs each with a non-empty array of
    listed hosts, a format other than the kind's (malformed); another kind
    (wrong_kind); a repeated key
    or item (duplicate); bytes other than `canonical_list_bytes` of the
    content, which includes unsorted items (not_canonical).
    """
    if kind not in _LIST_FIELDS:
        raise ValueError(f"unknown list kind {kind!r}")
    if type(payload) is not bytes:
        raise ListRefused("malformed", "a list is bytes")
    if len(payload) > MAX_LIST_BYTES:
        raise ListRefused("oversized", f"list of {len(payload)} bytes, at most {MAX_LIST_BYTES}")
    try:
        obj = json.loads(payload.decode("ascii"), object_pairs_hook=_object,
                         parse_float=_not_a_number, parse_constant=_not_a_number)
    except _RepeatedKey as e:
        raise ListRefused("duplicate", f"key {_show(str(e))} appears twice") from None
    except (ValueError, RecursionError):
        raise ListRefused("malformed", "the list is not ASCII JSON with integer numbers") from None
    if type(obj) is not dict:
        raise ListRefused("malformed", "the list is not a JSON object")
    if obj.get("kind") != kind:
        raise ListRefused("wrong_kind", f"kind {_show(obj.get('kind'))}, expected {kind!r}")
    names = _LIST_FIELDS[kind]
    expected = {"format", "kind", "serial", *names}
    if set(obj) != expected:
        raise ListRefused("malformed", f"fields {sorted(obj)}, expected {sorted(expected)}")
    if not _is_int(obj["format"]) or obj["format"] != LIST_FORMATS[kind]:
        raise ListRefused("malformed", f"format {_show(obj['format'])}, expected "
                                       f"{LIST_FORMATS[kind]}")
    serial = obj["serial"]
    if not _is_int(serial) or not 0 <= serial <= MAX_SERIAL:
        raise ListRefused("malformed", f"serial {_show(serial)} is not an integer in "
                                       f"0..{MAX_SERIAL}")
    for name in names:
        if name == "first_party":
            continue
        items = obj[name]
        if type(items) is not list:
            raise ListRefused("malformed", f"`{name}` is not an array")
        if len(items) > MAX_LIST_ITEMS:
            raise ListRefused("oversized", f"`{name}` has {len(items)} items, at most "
                                           f"{MAX_LIST_ITEMS}")
        for item in items:
            if not _ITEM_RULES[name](item):
                raise ListRefused("malformed", f"`{name}` item {_show(item)} has the wrong shape")
        if len(set(items)) != len(items):
            raise ListRefused("duplicate", f"`{name}` names an item twice")
    if kind == KIND_MENU_ALLOWLIST:
        fp = obj["first_party"]
        if type(fp) is not dict:
            raise ListRefused("malformed", "`first_party` is not an object")
        if len(fp) > MAX_LIST_ITEMS or sum(len(v) for v in fp.values()
                                           if type(v) is list) > MAX_LIST_ITEMS:
            raise ListRefused("oversized", f"`first_party` names more than {MAX_LIST_ITEMS} items")
        for lab, hosts in fp.items():
            if not _valid_lab(lab) or type(hosts) is not list or not hosts or not all(
                    _valid_host(h) for h in hosts):
                raise ListRefused("malformed", f"`first_party` item {_show(lab)} is not a lab "
                                               "with a non-empty array of hosts")
            if len(set(hosts)) != len(hosts):
                raise ListRefused("duplicate", f"`first_party` names a host twice for {lab!r}")
    if payload != canonical_list_bytes(kind, serial, **{n: obj[n] for n in names}):
        raise ListRefused("not_canonical", "the bytes are not the canonical encoding of the list")
    digest = hashlib.sha256(payload).hexdigest()
    if kind == KIND_DEVICE_LIST:
        return DeviceList(serial=serial, devices=frozenset(obj["devices"]), sha256=digest)
    if not obj["labs"] or not obj["hosts"]:
        raise ListRefused("malformed", "`labs` and `hosts` must each name at least one item")
    fp = obj["first_party"]
    stray_lab = sorted(set(fp) - set(obj["labs"]))
    if stray_lab:
        raise ListRefused("malformed", f"first-party lab {stray_lab[0]!r} is not in `labs`")
    stray = sorted({h for hs in fp.values() for h in hs} - set(obj["hosts"]))
    if stray:
        raise ListRefused("malformed", f"first-party host {stray[0]!r} is not in `hosts`")
    return MenuAllowlist(serial=serial, labs=frozenset(obj["labs"]), hosts=frozenset(obj["hosts"]),
                         first_party=first_party_map(fp), sha256=digest)


# ── the signers ─────────────────────────────────────────────────────────────

_KEY_TYPE = b"ssh-ed25519"
# One line of the signers file, in the only form this module reads:
#   <principal> namespaces="<name>[,<name>...]" ssh-ed25519 <base64> [comment]
# with single spaces and literal namespace names (no patterns). OpenSSH reads
# more forms than this; a line in any other form grants nothing here, and the
# safety check refuses a shipped file that has one.
_SIGNERS_LINE = re.compile(
    r'(?P<principal>[^\s"#,]+) namespaces="(?P<namespaces>[A-Za-z0-9@._-]+(?:,[A-Za-z0-9@._-]+)*)"'
    r" ssh-ed25519 (?P<key>[A-Za-z0-9+/]+={0,2})(?: [\x20-\x7e]*)?")


class _Reader:
    """SSH wire strings (RFC 4251): a big-endian uint32 length, then the bytes."""

    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if n > len(self.data) - self.pos:
            raise ValueError("short read")
        self.pos += n
        return self.data[self.pos - n:self.pos]

    def u32(self) -> int:
        return int.from_bytes(self.take(4), "big")

    def string(self) -> bytes:
        return self.take(self.u32())

    def done(self) -> bool:
        return self.pos == len(self.data)


def _ed25519_key(blob: bytes) -> bytes | None:
    """The 32 raw bytes of an `ssh-ed25519` public key blob, or None."""
    try:
        r = _Reader(blob)
        if r.string() != _KEY_TYPE:
            return None
        key = r.string()
    except ValueError:
        return None
    return key if len(key) == 32 and r.done() else None


def _b64(text: str) -> bytes | None:
    """Strict base64: the canonical encoding of what it decodes to, or None."""
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return None
    return raw if base64.b64encode(raw).decode("ascii") == text else None


def _signers_lines(signers: bytes) -> list[tuple[int, str]]:
    try:
        text = signers.decode("ascii")
    except UnicodeDecodeError:
        return [(0, "")]
    return [(n, line) for n, line in enumerate(text.split("\n"), 1)
            if line.strip() and not line.startswith("#")]


def _read_signers_line(line: str) -> tuple[str, tuple[str, ...], bytes] | None:
    m = _SIGNERS_LINE.fullmatch(line)
    blob = _b64(m["key"]) if m else None
    key = _ed25519_key(blob) if blob is not None else None
    if key is None:
        return None
    return m["principal"], tuple(m["namespaces"].split(",")), key


def unread_signers_lines(signers: bytes) -> list[int]:
    """Line numbers of the signers file that are neither blank, a comment,
    nor in the one form this module reads. A file with such a line could mean
    one thing to OpenSSH and another here."""
    return [n for n, line in _signers_lines(signers) if _read_signers_line(line) is None]


def permitted_signers(signers: bytes, namespace: str = LIST_NAMESPACE) -> dict[bytes, str]:
    """{raw Ed25519 public key: principal} for every key the signers file
    lists for `namespace` by name. A line without a `namespaces` option, with
    a pattern, with any other option or of another key type grants nothing
    (OpenSSH would let a line without the option sign anything)."""
    if type(signers) is not bytes:
        raise ValueError("the signers file is bytes")
    if len(signers) > MAX_SIGNERS_BYTES:
        raise ListRefused("oversized", f"signers file of {len(signers)} bytes, at most "
                                       f"{MAX_SIGNERS_BYTES}")
    out: dict[bytes, str] = {}
    for _, line in _signers_lines(signers):
        read = _read_signers_line(line)
        if read is not None and namespace in read[1]:
            out.setdefault(read[2], read[0])
    return out


def list_signers() -> bytes:
    """The shipped signers file, checked against its compiled pin."""
    data = SIGNERS_PATH.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != LIST_SIGNERS_SHA256:
        raise RuntimeError(f"{SIGNERS_PATH} has hash {got}, which is not the pin "
                           f"{LIST_SIGNERS_SHA256}: the list signers are a trust root and "
                           "change only with the pin")
    return data


# ── the signature ───────────────────────────────────────────────────────────

_ARMOR_BEGIN = "-----BEGIN SSH SIGNATURE-----"
_ARMOR_END = "-----END SSH SIGNATURE-----"
_MAGIC = b"SSHSIG"
_HASH = b"sha512"


def _malformed(why: str) -> ListRefused:
    return ListRefused("signature_malformed", why)


def _dearmor(signature: bytes) -> bytes:
    try:
        text = signature.decode("ascii")
    except UnicodeDecodeError:
        raise _malformed("the signature is not ASCII") from None
    lines = (text[:-1] if text.endswith("\n") else text).split("\n")
    if len(lines) < 3 or lines[0] != _ARMOR_BEGIN or lines[-1] != _ARMOR_END:
        raise _malformed("not one SSH SIGNATURE block and nothing else")
    body = lines[1:-1]
    if any(not line or len(line) > 76 for line in body):
        raise _malformed("an empty or over-long line in the signature")
    blob = _b64("".join(body))
    if blob is None:
        raise _malformed("the signature is not canonical base64")
    return blob


def _verify_signature(payload: bytes, signature: bytes, signers: bytes) -> str:
    """Verify an OpenSSH signature (PROTOCOL.sshsig, version 1, Ed25519,
    SHA-512) over `payload` under LIST_NAMESPACE. Returns the signer's key
    fingerprint; raises ListRefused."""
    # Imported here so that the rules above need nothing but the standard
    # library. A node without the library fails loudly (ImportError) instead
    # of quietly refusing a list every other node accepts.
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    blob = _dearmor(signature)
    try:
        r = _Reader(blob)
        magic, version = r.take(len(_MAGIC)), r.u32()
        public, namespace, reserved, hash_name, sig = (r.string() for _ in range(5))
        complete = r.done()
        s = _Reader(sig)
        sig_type, raw_sig = s.string(), s.string()
        complete = complete and s.done()
    except ValueError:
        raise _malformed("the signature is cut short") from None
    key = _ed25519_key(public)
    if (magic != _MAGIC or version != 1 or not complete or reserved != b"" or hash_name != _HASH
            or key is None or sig_type != _KEY_TYPE or len(raw_sig) != 64):
        raise _malformed("not a version 1 Ed25519 signature over SHA-512 with nothing extra")
    if namespace != LIST_NAMESPACE.encode("ascii"):
        raise ListRefused("wrong_namespace", f"signed under namespace {_show(namespace)}, not "
                                             f"{LIST_NAMESPACE!r}")
    if key not in permitted_signers(signers):
        raise ListRefused("signer_not_allowed",
                          "the signing key is not listed for the list namespace")

    def wire(b: bytes) -> bytes:
        return len(b).to_bytes(4, "big") + b

    signed = (_MAGIC + wire(namespace) + wire(reserved) + wire(hash_name)
              + wire(hashlib.sha512(payload).digest()))
    try:
        Ed25519PublicKey.from_public_bytes(key).verify(raw_sig, signed)
    except (InvalidSignature, ValueError):
        raise ListRefused("bad_signature", "the signature does not verify over these bytes") \
            from None
    return "SHA256:" + base64.b64encode(hashlib.sha256(public).digest()).decode("ascii").rstrip("=")


def verify_signed_list(payload: bytes, signature: bytes, *, kind: str, held_serial: int,
                       signers: bytes | None = None) -> MenuAllowlist | DeviceList:
    """The one way a list becomes trusted. Returns the list, with `signer`
    set to the verified key's fingerprint, or raises ListRefused.

    `payload` is the list's bytes, `signature` the armored OpenSSH signature
    over them (`ssh-keygen -Y sign -n fugal-list@fugalai.com`), `kind` the
    list the caller expects, `held_serial` the serial of the list it holds
    (0 when it holds none), `signers` the allowed-signers bytes (default: the
    shipped, pinned file).

    Order: sizes, then the signature (well-formed, this namespace, a listed
    key, verifies over exactly these bytes), and only then is the list
    parsed (`parse_unsigned_list`'s rules), its serial required to be at
    least 1 (draft) and above `held_serial` (rollback). The verdict depends
    on the arguments alone.
    """
    if kind not in _LIST_FIELDS:
        raise ValueError(f"unknown list kind {kind!r}")
    if not _is_int(held_serial) or held_serial < 0:
        raise ValueError(f"held_serial {held_serial!r} is not a non-negative integer")
    if type(payload) is not bytes:
        raise ListRefused("malformed", "a list is bytes")
    if type(signature) is not bytes:
        raise _malformed("a signature is bytes")
    if len(payload) > MAX_LIST_BYTES:
        raise ListRefused("oversized", f"list of {len(payload)} bytes, at most {MAX_LIST_BYTES}")
    if len(signature) > MAX_SIGNATURE_BYTES:
        raise ListRefused("oversized", f"signature of {len(signature)} bytes, at most "
                                       f"{MAX_SIGNATURE_BYTES}")
    if signers is None:
        signers = list_signers()
    elif type(signers) is not bytes:
        raise ValueError("the signers file is bytes")
    if len(signers) > MAX_SIGNERS_BYTES:
        raise ListRefused("oversized", f"signers file of {len(signers)} bytes, at most "
                                       f"{MAX_SIGNERS_BYTES}")
    signer = _verify_signature(payload, signature, signers)
    parsed = parse_unsigned_list(payload, kind)
    if parsed.serial <= DRAFT_SERIAL:
        raise ListRefused("draft", f"serial {parsed.serial} marks a draft, not a list")
    if parsed.serial <= held_serial:
        raise ListRefused("rollback", f"serial {parsed.serial} is not above the serial held, "
                                      f"{held_serial}")
    return replace(parsed, signer=signer)


def shipped_list(kind: str, *, directory: Path | None = None,
                 signers: bytes | None = None) -> MenuAllowlist | DeviceList:
    """The list of `kind` this release ships, verified: `<name>` and
    `<name>.sig` under `signed_lists/`. FileNotFoundError when the release
    ships none (or ships one without its signature); ListRefused when what
    it ships does not verify."""
    directory = SIGNED_LISTS_DIR if directory is None else Path(directory)
    name = SHIPPED_LISTS[kind]
    payload = (directory / name).read_bytes()
    signature = (directory / f"{name}.sig").read_bytes()
    return verify_signed_list(payload, signature, kind=kind, held_serial=0, signers=signers)
