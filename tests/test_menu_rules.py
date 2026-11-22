"""Protocol v4's compiled menu rules (docs/decisions/LIVE_MENU_V4.md, section 6).

Every rule is a pure function of an entry, the signed allowlist and (for the
proposal checks) a view of the menu state, and names the rule it refused
under. The cases marked A1 are section 14's first attack: a model that looks
perfect and cheap, proposed from a lab or through a host nobody vouched for.
"""
import copy
import importlib.util
import json
import pathlib

import pytest

from fugal_subnet import menu_rules as mr
from fugal_subnet.routing_protocol import validate_menu_v3

ROOT = pathlib.Path(__file__).resolve().parents[1]

ALLOW = mr.MenuAllowlist(
    serial=1,
    labs=frozenset({"openai", "deepseek", "google", "xiaomi"}),
    hosts=frozenset({"openai", "azure", "baidu", "deepinfra", "xiaomi", "google-vertex/global"}),
    first_party=mr.first_party_map({"openai": {"azure"}, "google": {"google-vertex/global"}}),
)


def closed(**over):
    """A closed-weights entry: unknown quantization on the lab's own host and a
    host the list marks first-party."""
    e = {"id": "openai/gpt-x@high", "model": "openai/gpt-x", "reasoning": {"effort": "high"},
         "max_tokens": 128000, "quantization": "unknown",
         "provider": {"order": ["openai", "azure"], "allow_fallbacks": False}}
    e.update(over)
    return e


def open_weights(**over):
    """An open-weights entry: one pinned quantization on two listed hosts."""
    e = {"id": "deepseek/deepseek-x@low", "model": "deepseek/deepseek-x",
         "reasoning": {"effort": "low"}, "max_tokens": 131072, "quantization": "fp8",
         "provider": {"order": ["baidu/fp8", "deepinfra/fp8"], "allow_fallbacks": False}}
    e.update(over)
    return e


def code(refusal):
    assert refusal is None or isinstance(refusal, mr.Refusal)
    return None if refusal is None else refusal.code


# ── entries the rules accept ────────────────────────────────────────────────

def test_a_closed_and_an_open_weights_entry_pass():
    assert mr.check_entry(closed(), ALLOW) is None
    assert mr.check_entry(open_weights(), ALLOW) is None


def test_an_entry_without_an_effort_is_named_by_its_model():
    e = closed(id="xiaomi/mimo-x", model="xiaomi/mimo-x", reasoning={"enabled": True},
               quantization="fp8",
               provider={"order": ["xiaomi/fp8", "deepinfra/fp8"], "allow_fallbacks": False})
    assert mr.check_entry(e, ALLOW) is None
    assert mr.entry_id("xiaomi/mimo-x", {"enabled": True}) == "xiaomi/mimo-x"
    assert mr.entry_id("openai/gpt-x", {"effort": "max"}) == "openai/gpt-x@max"


def test_a_refusal_names_its_rule_and_the_value():
    refusal = mr.check_entry(closed(model="nolab/gpt-x", id="nolab/gpt-x@high"), ALLOW)
    assert refusal.code == "lab_not_allowed"
    assert "nolab" in refusal.detail
    assert str(refusal).startswith("lab_not_allowed: ")


def test_an_entry_that_passes_also_passes_the_v3_shape_rules():
    for e in (closed(price_in=2000, price_out=10000), open_weights()):
        assert mr.check_entry(e, ALLOW) is None
        validate_menu_v3([{"id": e["id"], "model": e["model"], "in": 0, "out": 0,
                           "reasoning": e["reasoning"], "max_tokens": e["max_tokens"],
                           "provider": e["provider"]}])


# ── A1: the lab allowlist ───────────────────────────────────────────────────

@pytest.mark.parametrize("model", [
    "pool-fitted/perfect-and-cheap",      # a lab nobody listed
    "openai-community/gpt-x",             # a listed lab's name as a prefix
    "openai.evil/gpt-x",
    "xopenai/gpt-x",
])
def test_a1_a_model_from_an_unlisted_lab_is_refused(model):
    e = closed(model=model, id=f"{model}@high")
    assert code(mr.check_entry(e, ALLOW)) == "lab_not_allowed"


@pytest.mark.parametrize("model", [
    "OpenAI/gpt-x",                       # case is not folded: the id is the lab's, byte for byte
    "openai/GPT-x",
    " openai/gpt-x",
    "openai/gpt-x ",
    "openai//gpt-x",
    "openai/gpt-x/extra",
    "openai",
    "/gpt-x",
    "openai/",
    "оpenai/gpt-x",                  # a Cyrillic first letter
    "openai/gpt-x\n",
    "openai/gpt-x@high",                  # the effort belongs to the id, not the model
    "openai/" + "x" * 300,
])
def test_a1_a_model_id_that_is_not_lab_slash_name_is_refused(model):
    assert code(mr.check_entry(closed(model=model, id=f"{model}@high"), ALLOW)) == "model_shape"


@pytest.mark.parametrize("suffix", [":free", ":nitro", ":floor", ":extended", ":anything"])
def test_a1_a_model_variant_is_refused(suffix):
    """`:free`, `:nitro` and the like are other products under a listed lab's name."""
    e = closed(model=f"openai/gpt-x{suffix}", id=f"openai/gpt-x{suffix}@high")
    assert code(mr.check_entry(e, ALLOW)) == "model_variant"


def test_a1_the_id_cannot_name_another_model_than_the_one_called():
    """A row keyed `openai/...` that calls an unlisted lab, and the reverse."""
    assert code(mr.check_entry(closed(model="pool-fitted/m"), ALLOW)) == "lab_not_allowed"
    assert code(mr.check_entry(closed(id="pool-fitted/m@high"), ALLOW)) == "id_mismatch"
    assert code(mr.check_entry(closed(id="openai/gpt-x@low"), ALLOW)) == "id_mismatch"
    assert code(mr.check_entry(closed(id="openai/gpt-x"), ALLOW)) == "id_mismatch"
    assert code(mr.check_entry(closed(id=None), ALLOW)) == "id_mismatch"


def test_a1_an_empty_allowlist_admits_nothing():
    empty = mr.MenuAllowlist(serial=1, labs=frozenset(), hosts=frozenset())
    assert code(mr.check_entry(closed(), empty)) == "lab_not_allowed"


# ── A1: the host allowlist and the provider rules ───────────────────────────

def provider(*order, **extra):
    return {"order": list(order), "allow_fallbacks": False, **extra}


def test_a1_a_host_that_is_not_listed_is_refused():
    e = open_weights(provider=provider("baidu/fp8", "pool-fitted-host/fp8"))
    assert code(mr.check_entry(e, ALLOW)) == "host_not_allowed"
    e = closed(provider=provider("openai", "somewhere-else"))
    assert code(mr.check_entry(e, ALLOW)) == "host_not_allowed"


@pytest.mark.parametrize("tier", ["flex", "fast", "priority", "spot", "ultrafast", "turbo",
                                  "highspeed"])
def test_a_tier_of_a_listed_host_is_refused(tier):
    assert code(mr.check_entry(closed(provider=provider("openai", f"azure/{tier}")),
                               ALLOW)) == "host_tier"
    e = open_weights(provider=provider("baidu/fp8", f"deepinfra/fp8/{tier}"))
    assert code(mr.check_entry(e, ALLOW)) == "host_tier"


def test_a_regional_or_other_variant_passes_only_when_the_list_names_it():
    assert mr.check_entry(closed(model="google/gem-x", id="google/gem-x@high",
                                 provider=provider("google-vertex/global")), ALLOW) is None
    for tag in ("google-vertex/europe", "google-vertex", "azure/eastus"):
        e = closed(provider=provider("openai", tag))
        assert code(mr.check_entry(e, ALLOW)) == "host_not_allowed", tag


def test_fallbacks_must_be_switched_off():
    for value in (True, None, 0, "false"):
        e = closed(provider={"order": ["openai"], "allow_fallbacks": value})
        assert code(mr.check_entry(e, ALLOW)) == "fallbacks_allowed", value
    assert code(mr.check_entry(closed(provider={"order": ["openai"]}), ALLOW)) == "provider_shape"


@pytest.mark.parametrize("extra", [
    {"only": ["openai"]}, {"ignore": ["azure"]}, {"sort": "price"},
    {"quantizations": ["fp8"]}, {"require_parameters": True}, {"service_tier": "flex"},
])
def test_a_provider_preference_beyond_order_and_no_fallbacks_is_refused(extra):
    assert code(mr.check_entry(closed(provider=provider("openai", **extra)),
                               ALLOW)) == "provider_shape"


@pytest.mark.parametrize("order", [[], "openai", None, ["openai", 7], [""], [["openai"]]])
def test_the_order_is_a_non_empty_list_of_host_names(order):
    e = closed(provider={"order": order, "allow_fallbacks": False})
    assert code(mr.check_entry(e, ALLOW)) in ("provider_shape", "host_shape")


def test_at_most_two_hosts_and_no_host_twice():
    assert mr.MENU_MAX_HOSTS == 2
    e = open_weights(provider=provider("baidu/fp8", "deepinfra/fp8", "xiaomi/fp8"))
    assert code(mr.check_entry(e, ALLOW)) == "too_many_hosts"
    e = open_weights(provider=provider("baidu/fp8", "baidu/fp8"))
    assert code(mr.check_entry(e, ALLOW)) == "duplicate_host"


@pytest.mark.parametrize("tag", ["Azure", "azure ", "azure/", "/azure", "azure//fp8", "a/b/c/d/e",
                                 "azure\n", "azure:fast", "x" * 200])
def test_a_host_tag_that_is_not_a_plain_slug_is_refused(tag):
    assert code(mr.check_entry(closed(provider=provider(tag)), ALLOW)) == "host_shape"


# ── one pinned quantization ─────────────────────────────────────────────────

def test_the_quantization_is_one_of_a_fixed_set():
    for q in ("fp8", "bf16", "mxfp4", "unknown"):
        assert q in mr.QUANTIZATIONS
    for q in ("fp7", "FP8", "", None, 8, "flex", ["fp8"]):
        assert code(mr.check_entry(open_weights(quantization=q), ALLOW)) == \
            "quantization_not_allowed", q


def test_every_host_carries_the_entry_s_one_quantization():
    mixed = open_weights(provider=provider("baidu/fp8", "deepinfra/bf16"))
    assert code(mr.check_entry(mixed, ALLOW)) == "quantization_mismatch"
    other = open_weights(quantization="bf16")
    assert code(mr.check_entry(other, ALLOW)) == "quantization_mismatch"
    unpinned = open_weights(provider=provider("baidu/fp8", "deepinfra"))
    assert code(mr.check_entry(unpinned, ALLOW)) == "quantization_mismatch"
    twice = open_weights(provider=provider("baidu/fp8/fp8"))
    assert code(mr.check_entry(twice, ALLOW)) == "quantization_mismatch"
    stated_unknown = open_weights(quantization="unknown")
    assert code(mr.check_entry(stated_unknown, ALLOW)) == "quantization_mismatch"
    assert code(mr.check_entry(closed(provider=provider("openai/unknown")),
                               ALLOW)) == "quantization_mismatch"


def test_unknown_quantization_only_on_the_lab_s_own_or_a_first_party_host():
    """`deepinfra` and `baidu` are listed, but not as first-party: an entry
    cannot leave its quantization open on them."""
    e = open_weights(quantization="unknown", provider=provider("baidu", "deepinfra"))
    assert code(mr.check_entry(e, ALLOW)) == "unknown_quantization"
    own = closed(model="xiaomi/mimo-x", id="xiaomi/mimo-x@high", provider=provider("xiaomi"))
    assert mr.check_entry(own, ALLOW) is None
    first_party = closed(model="openai/gpt-x", id="openai/gpt-x@high",
                         provider=provider("azure"))
    assert mr.check_entry(first_party, ALLOW) is None
    not_its_own = closed(model="deepseek/deepseek-x", id="deepseek/deepseek-x@high",
                         provider=provider("openai"))
    assert code(mr.check_entry(not_its_own, ALLOW)) == "unknown_quantization"
    # A first-party mark is for the labs the list names it for (list format 2):
    # azure is openai's, so another lab's model on it must pin a quantization.
    on_anothers = closed(model="deepseek/deepseek-x", id="deepseek/deepseek-x@high",
                         provider=provider("azure"))
    assert code(mr.check_entry(on_anothers, ALLOW)) == "unknown_quantization"
    assert mr.check_entry(dict(on_anothers, quantization="fp8",
                               provider=provider("azure/fp8")), ALLOW) is None


def test_pinned_quantization_reads_the_one_quantization_of_a_host_order():
    assert mr.pinned_quantization(["baidu/fp8", "deepinfra/fp8"]) == "fp8"
    assert mr.pinned_quantization(["openai", "azure"]) == "unknown"
    assert mr.pinned_quantization(["google-vertex/global"]) == "unknown"
    assert mr.pinned_quantization(["baidu/fp8", "deepinfra/bf16"]) is None
    assert mr.pinned_quantization(["baidu/fp8", "deepinfra"]) is None
    assert mr.host_identity("deepinfra/fp8") == "deepinfra"
    assert mr.host_identity("google-vertex/global") == "google-vertex/global"


# ── the fixed reasoning settings ────────────────────────────────────────────

def test_the_reasoning_settings_are_a_fixed_set():
    assert mr.REASONING_EFFORTS == ("low", "medium", "high", "xhigh", "max")
    for effort in mr.REASONING_EFFORTS:
        assert mr.check_entry(closed(id=f"openai/gpt-x@{effort}", reasoning={"effort": effort}),
                              ALLOW) is None


@pytest.mark.parametrize("reasoning", [
    None, {}, "high", ["high"], {"effort": "minimal"}, {"effort": "none"}, {"effort": "HIGH"},
    {"effort": "high", "exclude": True}, {"effort": "high", "enabled": True},
    {"max_tokens": 2000}, {"enabled": False}, {"enabled": 1}, {"enabled": "true"},
    {"effort": None}, {"effort": 3},
])
def test_any_other_reasoning_setting_is_refused(reasoning):
    assert code(mr.check_entry(closed(reasoning=reasoning), ALLOW)) == "reasoning_not_allowed"


# ── the answer limit ────────────────────────────────────────────────────────

def test_the_answer_limit_is_a_positive_integer_up_to_the_cap():
    assert mr.MAX_ANSWER_TOKENS == 131072
    assert mr.check_entry(closed(max_tokens=131072), ALLOW) is None
    for bad in (0, -1, 131073, 10**9, None, "128000", 128000.0, True):
        assert code(mr.check_entry(closed(max_tokens=bad), ALLOW)) == "answer_limit", bad


def test_the_answer_limit_is_the_smallest_among_the_pinned_hosts_capped():
    limits = {"openai": 128000, "azure": 200000, "other": 5}
    assert mr.answer_limit(["openai", "azure"], limits) == 128000
    assert mr.answer_limit(["azure"], limits) == 131072
    assert mr.check_entry(closed(), ALLOW, host_limits=limits) is None
    assert code(mr.check_entry(closed(max_tokens=131072), ALLOW,
                               host_limits=limits)) == "answer_limit"
    assert code(mr.check_entry(closed(max_tokens=100), ALLOW,
                               host_limits=limits)) == "answer_limit"


@pytest.mark.parametrize("limits", [{"openai": 128000}, {"openai": 128000, "azure": 0},
                                    {"openai": 128000, "azure": "200000"},
                                    {"openai": 128000, "azure": True}, {}])
def test_a_new_proposal_needs_every_pinned_host_listed_with_a_limit(limits):
    """What is proposed is what will be called: a proposal pins only hosts the
    listing shows, each stating an answer limit."""
    state = dict(epoch=10, entries=[], pending=[], retired={})
    assert code(mr.check_request(closed(), ALLOW, host_limits=limits)) == "answer_limit_unknown"
    assert code(mr.check_proposal(closed(), ALLOW, host_limits=limits,
                                  **state)) == "answer_limit_unknown"
    with pytest.raises(ValueError):
        mr.answer_limit(["openai", "azure"], limits)


def test_an_entry_on_the_menu_skips_a_pinned_host_the_listing_does_not_show():
    """Hosts come and go on a timescale of days. An entry stays while one of
    its pinned hosts is listed, limited by the hosts that are: the listed
    ones, and of those the ones that state a limit."""
    only_openai = {"openai": 128000}
    assert mr.answer_limit(["openai", "azure"], only_openai, on_menu=True) == 128000
    assert mr.check_entry(closed(), ALLOW, host_limits=only_openai) is None
    only_azure = {"azure": 200000, "other": 5}
    assert mr.answer_limit(["openai", "azure"], only_azure, on_menu=True) == 131072
    assert mr.check_entry(closed(max_tokens=131072), ALLOW, host_limits=only_azure) is None
    assert code(mr.check_entry(closed(), ALLOW, host_limits=only_azure)) == "answer_limit"
    # A listed host that states no limit sets none; with no stated limit, the cap.
    assert mr.answer_limit(["openai", "azure"], {"openai": 0, "azure": 64000}, on_menu=True) == 64000
    assert mr.answer_limit(["openai", "azure"], {"openai": 0, "azure": 0}, on_menu=True) == 131072
    assert mr.check_entry(closed(max_tokens=64000), ALLOW,
                          host_limits={"openai": 0, "azure": 64000}) is None


@pytest.mark.parametrize("limits", [{}, {"other": 5}, {"openai": "128000"}, {"openai": True},
                                    {"openai": -1}, {"openai": None}, {"azure": 1.5}])
def test_an_entry_on_the_menu_needs_one_pinned_host_listed_and_readable_limits(limits):
    assert code(mr.check_entry(closed(), ALLOW, host_limits=limits)) == "answer_limit_unknown"
    with pytest.raises(ValueError):
        mr.answer_limit(["openai", "azure"], limits, on_menu=True)


# ── shape: hostile input never raises, prices are integers ──────────────────

@pytest.mark.parametrize("entry", [None, [], "entry", 7, {}, {"model": 7}, {"model": None},
                                   {"model": ["openai/gpt-x"]}])
def test_something_that_is_not_an_entry_is_refused_not_raised(entry):
    assert code(mr.check_entry(entry, ALLOW)) == "malformed"


def test_prices_are_non_negative_integers_when_present():
    assert mr.check_entry(closed(price_in=0, price_out=43680), ALLOW) is None
    for bad in (1.5, "2", None, True, float("nan")):
        assert code(mr.check_entry(closed(price_in=bad, price_out=1), ALLOW)) == "price_type", bad
    assert code(mr.check_entry(closed(price_in=1, price_out=-1), ALLOW)) == "shape"


def test_checking_changes_neither_the_entry_nor_the_verdict():
    e = open_weights()
    before = copy.deepcopy(e)
    assert mr.check_entry(e, ALLOW) == mr.check_entry(e, ALLOW)
    bad = open_weights(quantization="bf16")
    assert mr.check_entry(bad, ALLOW) == mr.check_entry(copy.deepcopy(bad), ALLOW)
    assert e == before


# ── a whole menu ────────────────────────────────────────────────────────────

def test_a_menu_passes_when_every_entry_does_and_no_id_repeats():
    assert mr.check_menu([closed(), open_weights()], ALLOW) is None
    refusal = mr.check_menu([closed(), open_weights(quantization="int4")], ALLOW)
    assert refusal.code == "quantization_mismatch"
    assert "deepseek/deepseek-x@low" in refusal.detail
    assert code(mr.check_menu([closed(), closed()], ALLOW)) == "duplicate_entry"
    for not_a_menu in ([], None, {}, "menu", (closed(),)):
        assert code(mr.check_menu(not_a_menu, ALLOW)) == "malformed"
    assert code(mr.check_menu([closed(), None], ALLOW)) == "malformed"


def test_a8_one_forbidden_entry_refuses_the_menu_it_is_in():
    """Section 4.3: a menu carrying an entry the rules forbid is refused whole."""
    menu = [closed(), open_weights(),
            closed(model="pool-fitted/m", id="pool-fitted/m@high")]
    refusal = mr.check_menu(menu, ALLOW)
    assert refusal.code == "lab_not_allowed" and "pool-fitted" in refusal.detail


def test_a_lab_or_host_leaving_the_list_fails_its_entries():
    """Section 4.4 step 1: the same entry against a newer list without its lab or host."""
    assert mr.check_entry(open_weights(), ALLOW) is None
    no_lab = mr.MenuAllowlist(serial=2, labs=ALLOW.labs - {"deepseek"}, hosts=ALLOW.hosts,
                              first_party=ALLOW.first_party)
    no_host = mr.MenuAllowlist(serial=2, labs=ALLOW.labs, hosts=ALLOW.hosts - {"baidu"},
                               first_party=ALLOW.first_party)
    no_mark = mr.MenuAllowlist(serial=2, labs=ALLOW.labs, hosts=ALLOW.hosts)
    assert code(mr.check_entry(open_weights(), no_lab)) == "lab_not_allowed"
    assert code(mr.check_entry(open_weights(), no_host)) == "host_not_allowed"
    assert code(mr.check_entry(closed(), no_mark)) == "unknown_quantization"
    assert mr.ALLOWLIST_REFUSALS == {"lab_not_allowed", "host_not_allowed", "unknown_quantization"}


# ── proposals against the menu state: duplicates and cooldowns ──────────────

def test_a_proposal_for_a_model_and_setting_already_pending_is_refused():
    pending = [{"model": "openai/gpt-x", "reasoning": {"effort": "high"}}]
    assert code(mr.check_duplicate(closed(), entries=[], pending=pending)) == "already_pending"
    other_effort = closed(id="openai/gpt-x@low", reasoning={"effort": "low"})
    assert mr.check_duplicate(other_effort, entries=[], pending=pending) is None
    assert mr.check_duplicate(closed(), entries=[], pending=[]) is None


def test_a_proposal_identical_to_a_menu_entry_is_refused():
    assert code(mr.check_duplicate(closed(), entries=[closed()], pending=[])) == "already_on_menu"
    assert mr.check_duplicate(closed(), entries=[open_weights()], pending=[]) is None


def test_the_same_model_and_setting_with_another_request_is_a_request_update():
    """Section 4.4: same id, new provider order goes through the proposal path."""
    for update in (closed(provider=provider("azure", "openai")),
                   closed(provider=provider("openai"))):
        assert mr.check_duplicate(update, entries=[closed()], pending=[]) is None
        assert mr.is_request_update(update, [closed()]) is True
    quant = open_weights(quantization="bf16", provider=provider("baidu/bf16", "deepinfra/bf16"))
    assert mr.is_request_update(quant, [open_weights()]) is True
    assert mr.is_request_update(closed(), [closed()]) is False
    assert mr.is_request_update(closed(), [open_weights()]) is False
    # an update is still refused while one for the same model and setting is pending
    assert code(mr.check_duplicate(closed(provider=provider("openai")), entries=[closed()],
                                   pending=[closed(provider=provider("azure"))])) == \
        "already_pending"


def test_a_retired_entry_cannot_be_proposed_before_its_cooldown_ends():
    retired = {"openai/gpt-x@high": {"epoch": 100, "reason": "failing", "cooldown_until": 148}}
    eid = "openai/gpt-x@high"
    assert code(mr.check_cooldown(eid, 100, retired=retired)) == "entry_cooldown"
    assert code(mr.check_cooldown(eid, 147, retired=retired)) == "entry_cooldown"
    assert mr.check_cooldown(eid, 148, retired=retired) is None
    assert mr.check_cooldown("openai/gpt-x@low", 100, retired=retired) is None
    assert mr.check_cooldown(eid, 100, retired={}) is None


def test_a_cooldown_record_that_cannot_be_read_refuses():
    for record in ({}, {"cooldown_until": "148"}, {"cooldown_until": None}, None, 148,
                   {"cooldown_until": True}):
        assert code(mr.check_cooldown("m/x", 1, retired={"m/x": record})) == "entry_cooldown"


def test_a_proposer_in_cooldown_cannot_file():
    cooldowns = {"coldkey-a": 60}
    kw = dict(retired={}, coldkey_cooldowns=cooldowns)
    assert code(mr.check_cooldown("m/x", 59, proposer_coldkey="coldkey-a", **kw)) == \
        "proposer_cooldown"
    assert mr.check_cooldown("m/x", 60, proposer_coldkey="coldkey-a", **kw) is None
    assert mr.check_cooldown("m/x", 59, proposer_coldkey="coldkey-b", **kw) is None


def test_a_state_view_of_the_wrong_type_raises_instead_of_passing():
    for kw in (dict(retired=None), dict(retired=[]), dict(retired={}, coldkey_cooldowns=[]),):
        with pytest.raises(TypeError):
            mr.check_cooldown("m/x", 1, proposer_coldkey="ck", **kw)
    for epoch in (None, "1", 1.0, True):
        with pytest.raises(TypeError):
            mr.check_cooldown("m/x", epoch, retired={})


def test_a2_a_proposal_is_checked_against_rules_then_duplicates_then_cooldowns():
    state = dict(entries=[open_weights()], pending=[], retired={}, epoch=10)
    assert mr.check_proposal(closed(), ALLOW, **state) is None
    assert code(mr.check_proposal(closed(model="nolab/m", id="nolab/m@high"), ALLOW,
                                  **state)) == "lab_not_allowed"
    assert code(mr.check_proposal(open_weights(), ALLOW, **state)) == "already_on_menu"
    state["pending"] = [closed()]
    assert code(mr.check_proposal(closed(), ALLOW, **state)) == "already_pending"
    state["pending"] = []
    state["retired"] = {"openai/gpt-x@high": {"cooldown_until": 58}}
    assert code(mr.check_proposal(closed(), ALLOW, **state)) == "entry_cooldown"
    state["retired"] = {}
    assert code(mr.check_proposal(closed(), ALLOW, proposer_coldkey="ck",
                                  coldkey_cooldowns={"ck": 58}, **state)) == "proposer_cooldown"
    limits = {"openai": 64000, "azure": 128000}
    assert code(mr.check_proposal(closed(), ALLOW, host_limits=limits, **state)) == "answer_limit"
    assert mr.check_proposal(closed(max_tokens=64000), ALLOW, host_limits=limits,
                             **state) is None


# ── the first draft list, and the v3 menu it was derived from ───────────────

def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test",
                                                  ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _v3b_rows():
    return json.loads((ROOT / "data" / "models_v3b.json").read_text())


def test_the_draft_list_holds_everything_the_v3_menu_pins():
    """The first draft was derived by scripts/signed_list.py from the labs and
    hosts data/models_v3b.json pins, and the owner's review has only added to
    it (more labs, the hosts their models need): nothing the v3 menu pins was
    dropped, and no first-party mark was lost. (The owner's reviewed, signed
    list replaces the draft, and this test goes with it.)"""
    tool = _script("signed_list")
    draft = (ROOT / "data" / "menu_allowlist.draft.json").read_bytes()
    derived = mr.parse_unsigned_list(tool.draft_from_menu(_v3b_rows()), mr.KIND_MENU_ALLOWLIST)
    parsed = mr.parse_unsigned_list(draft, mr.KIND_MENU_ALLOWLIST)
    assert parsed.serial == mr.DRAFT_SERIAL == 0 and parsed.signer is None
    rows = _v3b_rows()
    assert parsed.labs >= {r["model"].split("/")[0] for r in rows}
    assert parsed.hosts >= {mr.host_identity(h) for r in rows for h in r["provider"]["order"]}
    marks = {(lab, h) for lab, hs in parsed.first_party for h in hs}
    assert (parsed.labs >= derived.labs and parsed.hosts >= derived.hosts
            and marks >= {(lab, h) for lab, hs in derived.first_party for h in hs})
    assert parsed.first_party_hosts <= parsed.hosts
    assert {lab for lab, _hs in parsed.first_party} <= parsed.labs


def test_the_draft_admits_a_current_model_of_every_lab_the_review_added():
    """One real call shape per added lab (OpenRouter's public listing, read
    2026-11-21): xAI and Mistral are served by hosts whose slug is not the
    lab's (`xai`, `mistral/zdr`), so only the first-party mark lets them pin
    no quantization; Cohere's own host is `cohere`; MiniMax, NVIDIA and
    Amazon models ride hosts that were already on the list."""
    allow = mr.parse_unsigned_list((ROOT / "data" / "menu_allowlist.draft.json").read_bytes(),
                                   mr.KIND_MENU_ALLOWLIST)
    calls = [("x-ai/grok-4.7", ["xai"], "unknown"),
             ("mistralai/mistral-large-4-0", ["mistral/zdr"], "unknown"),
             ("cohere/command-a-plus", ["cohere/fp8"], "fp8"),
             ("minimax/minimax-m3", ["deepinfra/fp8"], "fp8"),
             ("nvidia/nemotron-3.5-lightning", ["coreweave/bf16", "deepinfra/bf16"], "bf16"),
             ("amazon/nova-2-lite-v1", ["amazon-bedrock"], "unknown"),
             # a host tag that carries its quantization (`tencent/fp4`, `seed/fp8`) is
             # listed by its host (`tencent`, `seed`) and pins that quantization; Baidu's
             # own host is `baidu`, the lab's slug, so it may pin none
             ("tencent/hy4", ["tencent/fp4"], "fp4"),
             ("bytedance-seed/seed-2-1-turbo", ["seed/fp8"], "fp8"),
             ("baidu/ernie-light-dt-1.0-intl", ["baidu"], "unknown")]
    for model, order, quantization in calls:
        request = {"model": model, "reasoning": {"effort": "high"}, "quantization": quantization,
                   "provider": {"order": order, "allow_fallbacks": False}, "max_tokens": 16384}
        assert mr.check_request(request, allow) is None, (model, mr.check_request(request, allow))
    assert "xai" in allow.first_party_of("x-ai")
    assert "mistral/zdr" in allow.first_party_of("mistralai")
    # no mark where the entry pins a quantization anyway: a mark would let any lab's
    # model skip pinning one on that host
    assert not {"tencent", "seed", "cohere"} & allow.first_party_hosts


def test_every_v3_menu_entry_passes_the_rules_against_the_draft_list():
    """The seed of the first v4 menu is v3's entries (section 4.5): each one,
    with the quantization its pinned hosts carry, passes the compiled rules."""
    allow = mr.parse_unsigned_list((ROOT / "data" / "menu_allowlist.draft.json").read_bytes(),
                                   mr.KIND_MENU_ALLOWLIST)
    rows = _v3b_rows()
    entries = []
    for r in rows:
        quantization = mr.pinned_quantization(r["provider"]["order"])
        assert quantization is not None, r["id"]
        entries.append({"id": r["id"], "model": r["model"], "reasoning": r["reasoning"],
                        "max_tokens": r["max_tokens"], "provider": r["provider"],
                        "quantization": quantization})
        assert mr.check_entry(entries[-1], allow) is None, (r["id"], mr.check_entry(entries[-1], allow))
    assert len(entries) == 60
    assert mr.check_menu(entries, allow) is None
