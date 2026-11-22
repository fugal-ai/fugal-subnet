"""Protocol v4: the TD's attested reads of OpenRouter's endpoints listing.

Design: docs/decisions/LIVE_MENU_V4.md, sections 7.1 and 12. Each v4 epoch the
miner's TD reads the public `GET /models/{model}/endpoints` page of every
model on the menu, in measurement or waiting (and of a model it proposes),
through its one permitted base URL, and puts a compact record in its proof:
per model, per pinned host, whether the host is listed, its prices, its
answer limit, its quantization and its status. Validators take the median of
these across coldkeys (W4); nobody's single read moves a price.

Two parsers live here, and both are strict:

- `parse_listing`: a listing page from the network, inside the TD. It is a
  third party's bytes, so it is bounded (size, endpoint count, string
  lengths, number ranges) before anything in it is used, and a page that
  fails any check is a failed read: no record, which validators count as no
  vote. Prices are decimal strings converted to integer picodollars per token
  with integer arithmetic only (never a float), and a price that is not a
  whole number of picodollars, negative, absurd or written any other way
  fails the read.
- `reads_from_json`: the `endpoint_reads` field of a proof, on a validator.
  It is miner-supplied bytes, bounded the same way (I2), and accepts one
  encoding of each record so that a record has exactly one hash.

Nothing here imports the v4 menu modules: a validator that parses a proof
carrying this field on any protocol stays off the v4 code.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PICO = 10**12                     # picodollars in a dollar (menu_state.PICO)

# --- bounds (I2) -------------------------------------------------------------
# A real page is under 60 KB with at most 51 endpoints (the 25 pages of
# data/observations/openrouter_endpoints_2026-11-18.json). The caps leave an
# order of magnitude of room and no more.
MAX_LISTING_BYTES = 512 * 1024
MAX_LISTING_ENDPOINTS = 256
# Models read in one epoch: the menu cap (64) and the proposals in
# measurement or waiting (8), the TD's own proposal, with room to spare.
MAX_READ_MODELS = 80
# Pinned hosts of one model across every entry and proposal that names it.
MAX_READ_HOSTS = 32
# The dearest entry on the 2026-11-18 menu lists $40 per million output
# tokens (4e7 picodollars per token); a price above $1,000 per million
# tokens is not a price this subnet calls at.
MAX_PRICE_PICO = 10**9
# An answer limit is a token count; the listing states up to ~10^6 today.
MAX_ANSWER_LIMIT = 10**9
# OpenRouter's endpoint status is a small integer (0, -2, -5 on 2026-11-18).
MAX_ABS_STATUS = 1000
MAX_MODEL_LEN = 200
MAX_HOST_LEN = 96
MAX_QUANTIZATION_LEN = 16
MAX_DECIMAL_LEN = 40

_NAME = r"[a-z0-9][a-z0-9._-]*"
MODEL_RE = re.compile(rf"{_NAME}/{_NAME}")
HOST_RE = re.compile(rf"{_NAME}(?:/{_NAME}){{0,3}}")
_QUANT_RE = re.compile(r"[a-z0-9]{1,16}")
# A plain non-negative decimal, as the listing writes prices ("0.00000004368").
_DECIMAL_RE = re.compile(r"(0|[1-9][0-9]{0,8})(?:\.([0-9]{1,30}))?")


class ListingRefused(ValueError):
    """A listing page that is not one this TD will take a price from."""


@dataclass(frozen=True)
class HostListing:
    """One pinned host of one model as the listing showed it.

    A host the listing does not show is `listed=False` with every other field
    at its zero value, so a record has one encoding. `limit` is the stated
    answer limit (`max_completion_tokens`), 0 when the listing states none.
    """

    host: str
    listed: bool
    price_in: int = 0            # picodollars per prompt token
    price_out: int = 0           # picodollars per completion token
    limit: int = 0
    quantization: str = ""
    status: int = 0


@dataclass(frozen=True)
class EndpointRead:
    """One model's pinned hosts, sorted by host, each once."""

    model: str
    hosts: tuple[HostListing, ...]


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def decimal_to_pico(text) -> int:
    """A price per token written as a decimal string, in integer picodollars.

    Integer arithmetic only: the digits after the twelfth decimal place must
    be zeros (OpenRouter's prices are whole picodollars), and anything that is
    not a plain non-negative decimal (a sign, an exponent, spaces, a number
    rather than a string) is refused."""
    if not isinstance(text, str) or len(text) > MAX_DECIMAL_LEN:
        raise ListingRefused(f"price {text!r:.60} is not a decimal string")
    m = _DECIMAL_RE.fullmatch(text)
    if m is None:
        raise ListingRefused(f"price {text!r:.60} is not a plain non-negative decimal")
    whole, frac = int(m.group(1)), m.group(2) or ""
    if frac[12:].strip("0"):
        raise ListingRefused(f"price {text!r} is not a whole number of picodollars")
    value = whole * PICO + int((frac[:12] or "0").ljust(12, "0"))
    if value > MAX_PRICE_PICO:
        raise ListingRefused(f"price {text!r} is above {MAX_PRICE_PICO} picodollars a token")
    return value


def _host_from_listing(tag: str, endpoint) -> HostListing:
    if not isinstance(endpoint, Mapping):
        raise ListingRefused(f"host {tag}: the endpoint is not an object")
    pricing = endpoint.get("pricing")
    if not isinstance(pricing, Mapping):
        raise ListingRefused(f"host {tag}: no pricing object")
    price_in = decimal_to_pico(pricing.get("prompt"))
    price_out = decimal_to_pico(pricing.get("completion"))
    limit = endpoint.get("max_completion_tokens")
    if limit is None:
        limit = 0
    if not _is_int(limit) or not 0 <= limit <= MAX_ANSWER_LIMIT:
        raise ListingRefused(f"host {tag}: answer limit {limit!r:.40} is not a count")
    quant = endpoint.get("quantization")
    if not isinstance(quant, str) or not _QUANT_RE.fullmatch(quant):
        raise ListingRefused(f"host {tag}: quantization {quant!r:.40} is not a name")
    status = endpoint.get("status")
    if not _is_int(status) or abs(status) > MAX_ABS_STATUS:
        raise ListingRefused(f"host {tag}: status {status!r:.40} is not a small integer")
    return HostListing(host=tag, listed=True, price_in=price_in, price_out=price_out,
                       limit=limit, quantization=quant, status=status)


def _no_repeat(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ListingRefused(f"field {k!r:.40} appears twice")
        out[k] = v
    return out


def _check_wanted(model, hosts) -> tuple[str, tuple[str, ...]]:
    if not isinstance(model, str) or len(model) > MAX_MODEL_LEN or not MODEL_RE.fullmatch(model):
        raise ValueError(f"model {model!r:.80} is not <lab>/<name>")
    hosts = tuple(sorted(set(hosts)))
    if not hosts or len(hosts) > MAX_READ_HOSTS:
        raise ValueError(f"{model}: 1..{MAX_READ_HOSTS} pinned hosts are read, not {len(hosts)}")
    for h in hosts:
        if not isinstance(h, str) or len(h) > MAX_HOST_LEN or not HOST_RE.fullmatch(h):
            raise ValueError(f"{model}: pinned host {h!r:.80} is not a host tag")
    return model, hosts


def parse_listing(model: str, raw: bytes, hosts: Iterable[str]) -> EndpointRead:
    """The record of `model`'s pinned `hosts` from one listing page's bytes.

    The page must be at most MAX_LISTING_BYTES of JSON with no repeated key,
    `{"data": {"id": model, "endpoints": [...]}}`, at most
    MAX_LISTING_ENDPOINTS endpoints, each an object whose `tag` is a host tag
    that appears once. Every pinned host the page lists must have well-formed
    prices, answer limit, quantization and status; other hosts' fields are not
    read. Any failure is ListingRefused: the read is no vote."""
    model, hosts = _check_wanted(model, hosts)
    if not isinstance(raw, (bytes, bytearray)):
        raise ListingRefused("a listing page is bytes")
    if len(raw) > MAX_LISTING_BYTES:
        raise ListingRefused(f"a page of {len(raw)} bytes, at most {MAX_LISTING_BYTES}")
    try:
        doc = json.loads(bytes(raw).decode("utf-8"), object_pairs_hook=_no_repeat)
    except ListingRefused:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise ListingRefused(f"not a JSON page: {type(e).__name__}") from None
    data = doc.get("data") if isinstance(doc, Mapping) else None
    if not isinstance(data, Mapping):
        raise ListingRefused("the page has no `data` object")
    if data.get("id") != model:
        raise ListingRefused(f"the page is for {data.get('id')!r:.80}, not {model}")
    endpoints = data.get("endpoints")
    if not isinstance(endpoints, list):
        raise ListingRefused("`data.endpoints` is not a list")
    if len(endpoints) > MAX_LISTING_ENDPOINTS:
        raise ListingRefused(f"{len(endpoints)} endpoints, at most {MAX_LISTING_ENDPOINTS}")
    by_tag: dict[str, object] = {}
    for ep in endpoints:
        tag = ep.get("tag") if isinstance(ep, Mapping) else None
        if not isinstance(tag, str) or len(tag) > MAX_HOST_LEN or not HOST_RE.fullmatch(tag):
            raise ListingRefused(f"an endpoint's tag {tag!r:.80} is not a host tag")
        if tag in by_tag:
            raise ListingRefused(f"host {tag} is listed twice")
        by_tag[tag] = ep
    out = tuple(_host_from_listing(h, by_tag[h]) if h in by_tag else HostListing(h, False)
                for h in hosts)
    return EndpointRead(model=model, hosts=out)


def wanted_hosts(requests: Iterable[tuple[str, Iterable[str]]]) -> dict[str, tuple[str, ...]]:
    """{model: its pinned hosts, sorted}, merged over (model, host order) pairs."""
    out: dict[str, set[str]] = {}
    for model, order in requests:
        out.setdefault(model, set()).update(order)
    if len(out) > MAX_READ_MODELS:
        raise ValueError(f"{len(out)} models to read, at most {MAX_READ_MODELS}")
    return {m: tuple(sorted(h)) for m, h in sorted(out.items())}


def read_endpoints(wanted: Mapping[str, Iterable[str]], fetch: Callable[[str], bytes], *,
                   workers: int = 16) -> tuple[EndpointRead, ...]:
    """Read every wanted model's page with `fetch(model) -> bytes` (the TD's
    bounded GET through its one base URL, `runtime.fetch_endpoints_page`) and
    parse it. A fetch or a page that fails is logged and leaves no record:
    nothing waits on it, and a failed read is no vote. Sorted by model."""
    items = [_check_wanted(m, h) for m, h in sorted(wanted.items())]
    if len(items) > MAX_READ_MODELS:
        raise ValueError(f"{len(items)} models to read, at most {MAX_READ_MODELS}")

    def one(item):
        model, hosts = item
        try:
            return parse_listing(model, fetch(model), hosts)
        except Exception as e:  # noqa: BLE001 - a failed read is no vote, never a failed epoch
            logger.warning("Endpoint read for %s failed (%s: %s); no vote for it this epoch",
                           model, type(e).__name__, str(e)[:200])
            return None

    if not items:
        return ()
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as pool:
        got = list(pool.map(one, items))
    return tuple(r for r in got if r is not None)


# --- the proof field -----------------------------------------------------------

_HOST_KEYS = ("host", "listed", "price_in", "price_out", "limit", "quantization", "status")


def host_to_json(h: HostListing) -> dict:
    return {"host": h.host, "listed": h.listed, "price_in": h.price_in,
            "price_out": h.price_out, "limit": h.limit, "quantization": h.quantization,
            "status": h.status}


def reads_to_json(reads: Iterable[EndpointRead]) -> list[dict]:
    """The proof's `endpoint_reads`: a list sorted by model."""
    return [{"model": r.model, "hosts": [host_to_json(h) for h in r.hosts]}
            for r in sorted(reads, key=lambda r: r.model)]


def _exact_keys(d, keys, what: str) -> None:
    if not isinstance(d, Mapping) or set(d) != set(keys) or not all(isinstance(k, str) for k in d):
        raise ValueError(f"{what} must be an object with exactly {sorted(keys)}")


def _int_in(v, lo: int, hi: int, what: str) -> int:
    if not _is_int(v) or not lo <= v <= hi:
        raise ValueError(f"{what} must be an integer in {lo}..{hi}")
    return v


def _host_from_json(d) -> HostListing:
    _exact_keys(d, _HOST_KEYS, "an endpoint read host")
    host = d["host"]
    if not isinstance(host, str) or len(host) > MAX_HOST_LEN or not HOST_RE.fullmatch(host):
        raise ValueError("an endpoint read host is not a host tag")
    listed = d["listed"]
    if not isinstance(listed, bool):
        raise ValueError(f"{host}: `listed` must be a boolean")
    price_in = _int_in(d["price_in"], 0, MAX_PRICE_PICO, f"{host}: price_in")
    price_out = _int_in(d["price_out"], 0, MAX_PRICE_PICO, f"{host}: price_out")
    limit = _int_in(d["limit"], 0, MAX_ANSWER_LIMIT, f"{host}: limit")
    status = _int_in(d["status"], -MAX_ABS_STATUS, MAX_ABS_STATUS, f"{host}: status")
    quant = d["quantization"]
    if not isinstance(quant, str):
        raise ValueError(f"{host}: quantization must be a string")
    if listed and not _QUANT_RE.fullmatch(quant):
        raise ValueError(f"{host}: quantization must be a name")
    h = HostListing(host, listed, price_in, price_out, limit, quant, status)
    if not listed and h != HostListing(host, False):
        raise ValueError(f"{host}: a host that is not listed carries no other value")
    return h


def reads_from_json(value) -> tuple[EndpointRead, ...]:
    """The proof's `endpoint_reads`, bounded and in its one encoding: at most
    MAX_READ_MODELS models sorted with none repeated, each with 1..
    MAX_READ_HOSTS hosts sorted with none repeated. ValueError otherwise."""
    if not isinstance(value, list):
        raise ValueError("endpoint_reads must be a list")
    if len(value) > MAX_READ_MODELS:
        raise ValueError(f"endpoint_reads names {len(value)} models, at most {MAX_READ_MODELS}")
    out = []
    for d in value:
        _exact_keys(d, ("model", "hosts"), "an endpoint read")
        model = d["model"]
        if not isinstance(model, str) or len(model) > MAX_MODEL_LEN or not MODEL_RE.fullmatch(model):
            raise ValueError("an endpoint read's model is not <lab>/<name>")
        hosts = d["hosts"]
        if not isinstance(hosts, list) or not 1 <= len(hosts) <= MAX_READ_HOSTS:
            raise ValueError(f"{model}: 1..{MAX_READ_HOSTS} hosts per read")
        parsed = tuple(_host_from_json(h) for h in hosts)
        names = [h.host for h in parsed]
        if names != sorted(set(names)):
            raise ValueError(f"{model}: hosts must be sorted with none repeated")
        out.append(EndpointRead(model, parsed))
    models = [r.model for r in out]
    if models != sorted(set(models)):
        raise ValueError("endpoint_reads must be sorted by model with none repeated")
    return tuple(out)
