"""The v4 entry rules against OpenRouter's recorded public listing
(data/observations/openrouter_endpoints_2026-11-18.json) and the reviewed
draft lab and host list (data/menu_allowlist.draft.json, list format 2):

- a host tag with a service tier is refused, every tier name the listing has
  shown included;
- a host is first-party only for the labs the list names it for: an entry
  may leave its quantization open on its own lab's host or on one marked
  first-party for its lab, and nowhere else;
- a reasoning effort is admitted only on hosts whose listing (as the root
  read it) takes one, and a second setting of a model already on the menu
  that does no differently is refused by the gates, while a setting that
  does differ (or costs clearly less) joins.

No network: the listing is the file recorded on 2026-11-18."""
from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest

from fugal_subnet import chain_proposals as cp
from fugal_subnet import menu_inputs as mi
from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from tests import menu_fixtures as fx

ROOT = pathlib.Path(__file__).resolve().parents[1]
LISTING = ROOT / "data" / "observations" / "openrouter_endpoints_2026-11-18.json"


def _pages():
    doc = json.loads(LISTING.read_text())
    return {p["model"]: p["raw"].encode() for p in doc["pages"]}


def _endpoints():
    for raw in _pages().values():
        data = json.loads(raw)["data"]
        for ep in data["endpoints"]:
            yield data["id"], ep


def _draft() -> mr.MenuAllowlist:
    return mr.parse_unsigned_list((ROOT / "data" / "menu_allowlist.draft.json").read_bytes(),
                                  mr.KIND_MENU_ALLOWLIST)


def _request(model, tag, quantization=None, reasoning=None):
    q = mr.pinned_quantization([tag]) if quantization is None else quantization
    return {"model": model, "reasoning": reasoning or {"effort": "high"},
            "provider": {"order": [tag], "allow_fallbacks": False},
            "quantization": q if q is not None else "unknown", "max_tokens": 16384}


# --- service tiers ------------------------------------------------------------------------


def test_every_tier_name_the_listing_has_shown_is_a_compiled_tier_suffix():
    shown = {ep["tag"] for _m, ep in _endpoints() if ep.get("service_tier")}
    assert shown, "the recorded listing shows no tiered endpoint"
    for tag in sorted(shown):
        assert any(s in mr.HOST_TIER_SUFFIXES for s in tag.split("/")[1:]), tag
    assert {"ultrafast", "turbo", "highspeed"} <= set(mr.HOST_TIER_SUFFIXES)


def test_a_tiered_endpoint_is_refused_even_under_a_list_that_names_its_tag():
    """A list built in process that names each tiered tag as a host (the
    signed-list parser refuses such an item; this checks the entry rule
    alone)."""
    tiered = [(m, ep["tag"]) for m, ep in _endpoints() if ep.get("service_tier")]
    labs = frozenset(m.split("/")[0] for m, _t in tiered)
    hosts = frozenset(mr.host_identity(t) for _m, t in tiered)
    anything = mr.MenuAllowlist(serial=1, labs=labs, hosts=hosts, first_party=mr.first_party_map(
        {lab: hosts for lab in labs}))
    for model, tag in tiered:
        refusal = mr.check_request(_request(model, tag), anything)
        assert refusal is not None and refusal.code == "host_tier", (model, tag, refusal)
    with pytest.raises(mr.ListRefused):
        mr.parse_unsigned_list(mr.canonical_list_bytes(
            mr.KIND_MENU_ALLOWLIST, 1, labs=["openai"], hosts=["openai/ultrafast"], first_party={}),
            mr.KIND_MENU_ALLOWLIST)


# --- first-party, per lab -----------------------------------------------------------------


def test_unknown_quantization_passes_only_on_a_host_first_party_for_the_entrys_own_lab():
    allow = _draft()
    for model, ep in _endpoints():
        lab, tag = model.split("/", 1)[0], ep["tag"]
        req = _request(model, tag)
        if req["quantization"] != "unknown" or mr.check_request(req, allow) is not None:
            continue
        ident = mr.host_identity(tag)
        assert ident == lab or ident in allow.first_party_of(lab), (model, tag)


@pytest.mark.parametrize("model, tag", [
    ("anthropic/claude-opus-5.5", "amazon-bedrock"),
    ("anthropic/claude-sonnet-5.5", "google-vertex/global"),
    ("meta/muse-ember", "google-vertex/global"),
    ("google/gemini-4-argon", "google-ai-studio"),
    ("openai/gpt-6.2-sol", "azure"),
    ("qwen/qwen4-max", "alibaba"),
])
def test_the_entries_the_list_marks_hosts_for_still_pass(model, tag):
    assert mr.check_request(_request(model, tag), _draft()) is None


@pytest.mark.parametrize("model, tag", [
    ("deepseek/deepseek-v4.1-flash-1017", "alibaba"),      # alibaba is first-party for qwen
    ("moonshotai/kimi-k3.1-pro", "alibaba"),
    ("anthropic/claude-fable-5.2", "azure"),               # azure is first-party for openai
])
def test_another_labs_model_on_a_marked_host_must_pin_a_quantization(model, tag):
    refusal = mr.check_request(_request(model, tag), _draft())
    assert refusal is not None and refusal.code == "unknown_quantization", refusal


def test_the_draft_marks_each_host_for_the_labs_it_serves_first_party():
    allow = _draft()
    assert allow.first_party_dict() == {
        "amazon": ["amazon-bedrock"],
        "anthropic": ["amazon-bedrock", "google-vertex/global"],
        "google": ["google-ai-studio", "google-vertex/global", "google-vertex/us"],
        "meta": ["google-vertex/global"],
        "mistralai": ["mistral/zdr"],
        "openai": ["azure"],
        "qwen": ["alibaba"],
        "x-ai": ["xai"],
    }
    assert {lab for lab, _h in allow.first_party} <= allow.labs
    assert allow.first_party_hosts <= allow.hosts


# --- reasoning efforts ----------------------------------------------------------------------


def _admit(model, tag, quantization, reasoning, allow):
    """The root's admission of one proposal, its listing read from the
    recorded page as the root reads it (chain_proposals.listing_from_page)."""
    base = dataclasses.replace(fx.wide_state(epoch=100), labs=ms.AllowlistRef(1, allow.sha256))
    allow = dataclasses.replace(allow, serial=1)
    raw = cp.document_bytes(cp.ProposalDoc(proposer="hk", model=model, reasoning=reasoning,
                                           provider_order=(tag,), quantization=quantization))
    h = cp.proposal_hash(raw)
    listing = cp.listing_from_page(model, _pages()[model], (tag,))
    read = cp.ChainRead(epoch=100, block=0, block_hash="ab" * 32, proposals=(("hk", h),),
                        coldkey_of={"hk": "ck"}, serving=frozenset({"hk"}), reveal=None)
    (c,) = cp.admit(base, read, {h: raw}, {h: listing}, allow).considered
    return c.verdict, listing


def test_an_effort_is_admitted_only_on_a_host_whose_listing_takes_one():
    allow = _draft()
    verdict, listing = _admit("google/gemma-4-31b-it", "coreweave/fp4", "fp4",
                              ("effort", "high"), allow)
    assert not listing[0].efforts and verdict == cp.REFUSED + "effort_unsupported"
    # The same host with no effort named is judged on the other rules.
    verdict, _ = _admit("google/gemma-4-31b-it", "coreweave/fp4", "fp4", ("enabled", True), allow)
    assert verdict != cp.REFUSED + "effort_unsupported"
    # A host that takes reasoning_effort is not refused for it.
    verdict, listing = _admit("openai/gpt-6.2-sol", "azure", "unknown", ("effort", "high"), allow)
    assert listing[0].efforts and verdict != cp.REFUSED + "effort_unsupported"


def _bounds(cells):
    """(rate, lower, upper) in ppm of {category: (labels, correct)}, by the
    validators' own function (menu_inputs.weighted_bounds)."""
    rate, lower, upper = mi.weighted_bounds(cells, fx.MIX)
    return mi._ppm_floor(rate), mi._ppm_floor(lower), mi._ppm_ceil(upper)


def test_a_second_effort_that_does_no_differently_is_refused_and_one_that_does_joins():
    """lab-a/model-one@high is on the menu with 500 labels at 80%. The same
    model at xhigh, priced the same: with the entry's own outcome rate it is
    refused (gate_sibling, SIBLING_MARGIN_PPM); with outcomes beyond the
    margin, or a call more than SIBLING_CHEAPER_BP cheaper, it joins."""
    assert (ms.SIBLING_MARGIN_PPM, ms.SIBLING_CHEAPER_BP) == (20_000, 2000)
    wide = fx.wide_state(epoch=100)
    on = wide.entry("lab-a/model-one@high")
    base = fx.provisional(wide, "lab-a/model-one", "xhigh", hosts=on.provider_order,
                          price_in=on.price_in, price_out=on.price_out)
    eid = "lab-a/model-one@xhigh"
    rate, lower, _u = _bounds({"code": (75, 60), "hard": (100, 80), "knowledge": (100, 80),
                               "math": (225, 180)})
    acc = (ms.AccuracyStat(on.id, on.rev, 500, rate, lower),)

    def decide(correct, tokens_out):
        _r, lo, up = _bounds({c: (60, correct // 4) for c in ("code", "hard", "knowledge", "math")})
        g = fx.gate(eid, correct=correct, tokens_in=250, tokens_out=tokens_out, lower_ppm=lo,
                    upper_ppm=up)
        after = ms.step(ms.carry(base), fx.inputs(epoch=base.epoch, gates=(g,), accuracy=acc),
                        fx.PARAMS)
        return {r.id: r.reason for r in after.retired}.get(eid) or (
            "full" if not after.entry(eid).provisional else "provisional")

    assert decide(192, 900) == ms.REASON_GATE_SIBLING           # 80%, as the entry does
    assert decide(236, 900) == "full"                           # 98%: clearly another setting
    assert decide(120, 900) in (ms.REASON_GATE_USEFUL, ms.REASON_GATE_CATEGORY)  # worse: not a sibling question
    assert decide(192, 200) == "full"                           # the same outcomes, far cheaper
