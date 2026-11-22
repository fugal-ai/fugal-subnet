"""Protocol v4: the TD reads OpenRouter's endpoints listing (LIVE_MENU_V4.md, 7.1).

Recorded pages only (data/observations/openrouter_endpoints_2026-11-18.json,
a committed snapshot of the real listing) and hand-made hostile pages: no
network, no key, no spend.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fugal_subnet.tee import endpoint_reads as er

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data" / "observations" / "openrouter_endpoints_2026-11-18.json"


def _pages() -> dict[str, bytes]:
    doc = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    out = {}
    for page in doc["pages"]:
        raw = page["raw"].encode("utf-8")
        assert hashlib.sha256(raw).hexdigest() == page["sha256"]
        out[page["model"]] = raw
    return out


PAGES = _pages()
FABLE = "anthropic/claude-fable-5.2"


def _page(model: str, endpoints: list, **data) -> bytes:
    return json.dumps({"data": {"id": model, "endpoints": endpoints, **data}}).encode()


def _ep(tag: str, prompt="0.000001", completion="0.000004", limit=65536, quant="fp8",
        status=0, **over) -> dict:
    d = {"tag": tag, "pricing": {"prompt": prompt, "completion": completion},
         "max_completion_tokens": limit, "quantization": quant, "status": status}
    d.update(over)
    return d


# --- prices: exact, no float -----------------------------------------------------


@pytest.mark.parametrize("text, pico", [
    ("0", 0), ("0.000008", 8_000_000), ("0.00004", 40_000_000),
    ("0.00000004368", 43_680), ("0.000001727124", 1_727_124),
    ("0.000000000001", 1), ("0.0000010000000000000", 1_000_000),
])
def test_a_listed_price_is_exact_integer_picodollars(text, pico):
    assert er.decimal_to_pico(text) == pico


@pytest.mark.parametrize("text", [
    "-0.000001", "+0.000001", "1e-6", "0.0000000000001", ".5", "0.", "00.1", " 0.1", "0.1 ",
    "0,1", "NaN", "Infinity", "", "0x10", "١", "0.000000000000５",
    0.000001, 1, None, True, ["0.1"], "1" + "0" * 40,
    "0.002",     # $2,000 per million tokens: above the cap
])
def test_a_price_in_any_other_form_or_out_of_range_is_refused(text):
    with pytest.raises(er.ListingRefused):
        er.decimal_to_pico(text)


def test_every_price_in_the_recorded_listing_converts_exactly():
    n = 0
    for model, raw in PAGES.items():
        for ep in json.loads(raw)["data"]["endpoints"]:
            for key in ("prompt", "completion"):
                text = ep["pricing"][key]
                pico = er.decimal_to_pico(text)
                # The same value through Python's exact decimal reader.
                from decimal import Decimal
                assert Decimal(text) * 10**12 == pico
                n += 1
    assert n > 400


# --- a recorded page -------------------------------------------------------------


def test_a_recorded_page_gives_each_pinned_host_its_listed_values():
    read = er.parse_listing(FABLE, PAGES[FABLE], ["azure", "google-vertex/global", "nobody"])
    assert read.model == FABLE
    by = {h.host: h for h in read.hosts}
    assert [h.host for h in read.hosts] == ["azure", "google-vertex/global", "nobody"]
    assert by["azure"] == er.HostListing("azure", True, 8_000_000, 40_000_000, 128000,
                                         "unknown", 0)
    assert by["nobody"] == er.HostListing("nobody", False)


def test_every_recorded_page_parses_for_its_listed_hosts():
    for model, raw in PAGES.items():
        tags = [ep["tag"] for ep in json.loads(raw)["data"]["endpoints"]]
        read = er.parse_listing(model, raw, tags[:er.MAX_READ_HOSTS])
        assert all(h.listed for h in read.hosts)


def test_the_seed_hosts_read_from_the_snapshot_give_the_seed_prices():
    """The TD's parser and the seed builder agree: each seed entry's price is
    the dearer listed pinned host's, read from the same recorded pages."""
    seed = json.loads((ROOT / "fugal_subnet" / "routing_data" / "menu_v4_seed.json").read_text())
    for e in seed["entries"]:
        read = er.parse_listing(e["model"], PAGES[e["model"]], e["provider"]["order"])
        listed = [h for h in read.hosts if h.listed]
        assert listed, e["id"]
        assert max(h.price_in for h in listed) == e["price_in"], e["id"]
        assert max(h.price_out for h in listed) == e["price_out"], e["id"]


# --- hostile pages ---------------------------------------------------------------

M = "lab-a/model-one"
GOOD = _page(M, [_ep("host-a/fp8"), _ep("host-z")])


def test_the_control_page_parses():
    read = er.parse_listing(M, GOOD, ["host-a/fp8"])
    assert read.hosts == (er.HostListing("host-a/fp8", True, 1_000_000, 4_000_000, 65536,
                                         "fp8", 0),)


@pytest.mark.parametrize("raw", [
    b"",
    b"not json",
    b"\xff\xfe",
    b"[]",
    b'{"data": []}',
    json.dumps({"data": {"id": "lab-b/other", "endpoints": [_ep("host-a/fp8")]}}).encode(),
    json.dumps({"data": {"id": M, "endpoints": {"host-a/fp8": _ep("host-a/fp8")}}}).encode(),
    _page(M, [_ep("host-a/fp8"), _ep("host-a/fp8", prompt="0.000002")]),        # duplicate host
    _page(M, [_ep("host-a/fp8"), _ep("host-z"), _ep("host-z")]),                # duplicate, unpinned
    _page(M, [_ep("host-a/fp8"), "host-z"]),                                     # not an object
    _page(M, [_ep("host-a/fp8"), _ep("HOST-Z")]),                                # tag shape
    _page(M, [_ep("host-a/fp8"), {"tag": 7}]),
    _page(M, [_ep("host-a/fp8", prompt="-0.000001")]),                           # negative
    _page(M, [_ep("host-a/fp8", completion="5")]),                               # absurd
    _page(M, [_ep("host-a/fp8", completion=0.000004)]),                          # a number
    _page(M, [_ep("host-a/fp8", prompt="1e-6")]),
    _page(M, [_ep("host-a/fp8", limit=-1)]),
    _page(M, [_ep("host-a/fp8", limit="65536")]),
    _page(M, [_ep("host-a/fp8", limit=True)]),
    _page(M, [_ep("host-a/fp8", limit=10**12)]),
    _page(M, [_ep("host-a/fp8", quant=None)]),
    _page(M, [_ep("host-a/fp8", quant="FP8")]),
    _page(M, [_ep("host-a/fp8", status=None)]),
    _page(M, [_ep("host-a/fp8", status=10**9)]),
    _page(M, [_ep("host-a/fp8", pricing=None)]),
    _page(M, [_ep("host-a/fp8")] + [_ep(f"h{i}") for i in range(er.MAX_LISTING_ENDPOINTS)]),
    b'{"data": {"id": "lab-a/model-one", "id": "lab-a/model-one", "endpoints": []}}',
    b"[" * 100000 + b"]" * 100000,
    _page(M, [_ep("host-a/fp8")], description="x" * er.MAX_LISTING_BYTES),        # oversized
])
def test_a_hostile_page_is_a_failed_read(raw):
    with pytest.raises(er.ListingRefused):
        er.parse_listing(M, raw, ["host-a/fp8"])


def test_an_unpinned_host_with_odd_values_does_not_spoil_the_read():
    """Only the pinned hosts' values are read. A listed host nobody pins may
    carry a variable price (OpenRouter writes "-1" for some routers)."""
    raw = _page(M, [_ep("host-a/fp8"), _ep("host-z", prompt="-1", limit=None, quant="weird!")])
    assert er.parse_listing(M, raw, ["host-a/fp8"]).hosts[0].listed


def test_a_stated_limit_of_null_is_no_limit():
    raw = _page(M, [_ep("host-a/fp8", limit=None)])
    assert er.parse_listing(M, raw, ["host-a/fp8"]).hosts[0].limit == 0


@pytest.mark.parametrize("model, hosts", [
    ("lab-a/../../keys", ["host-a"]),
    ("lab-a/model?x=1", ["host-a"]),
    ("lab-a/model-one", []),
    ("lab-a/model-one", ["host a"]),
    ("lab-a/model-one", [f"h{i}" for i in range(er.MAX_READ_HOSTS + 1)]),
])
def test_what_is_read_is_checked_before_anything_is_fetched(model, hosts):
    with pytest.raises(ValueError):
        er.read_endpoints({model: hosts}, fetch=lambda m: pytest.fail("fetched"))


# --- read_endpoints --------------------------------------------------------------


def test_a_failed_fetch_or_page_leaves_no_record_and_the_others_stand():
    def fetch(model):
        if model == "lab-b/broken":
            raise TimeoutError("deadline")
        if model == "lab-c/bad-page":
            return b"<html>"
        return _page(model, [_ep("host-a/fp8")])

    got = er.read_endpoints({"lab-c/bad-page": ["host-a/fp8"], "lab-b/broken": ["host-a/fp8"],
                             M: ["host-a/fp8", "host-b"]}, fetch)
    assert [r.model for r in got] == [M]
    assert [h.host for h in got[0].hosts] == ["host-a/fp8", "host-b"]
    assert [h.listed for h in got[0].hosts] == [True, False]


def test_reads_round_trip_through_the_proof_encoding():
    reads = er.read_endpoints({FABLE: ["azure", "anthropic", "gone"]}, PAGES.__getitem__)
    assert er.reads_from_json(json.loads(json.dumps(er.reads_to_json(reads)))) == reads


def test_wanted_hosts_merges_and_bounds():
    assert er.wanted_hosts([(M, ["b", "a"]), (M, ["a", "c"]), ("lab-b/x", ["z"])]) == {
        M: ("a", "b", "c"), "lab-b/x": ("z",)}
    with pytest.raises(ValueError):
        er.wanted_hosts([(f"lab/m{i}", ["h"]) for i in range(er.MAX_READ_MODELS + 1)])
