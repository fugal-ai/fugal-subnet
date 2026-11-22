"""The protocol v4 menu seed (docs/decisions/LIVE_MENU_V4.md, section 4.5).

The seed is the menu of the first v4 epoch, so it is consensus state before
any step has run: pinned by hash, mirrored under routing_data/, and built by
scripts/build_menu_v4_seed.py from files alone. These tests hold that the
committed inputs rebuild it byte for byte, that each rule of the builder does
what it says on hand-made inputs, and that the safety check notices a seed
that moved. Nothing here reads the network.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import pathlib
from fractions import Fraction

import pytest

from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from fugal_subnet import routing_protocol

ROOT = pathlib.Path(__file__).resolve().parents[1]
LISTING = ROOT / "data" / "observations" / "openrouter_endpoints_2026-11-18.json"
PAIRS = ROOT / "data" / "observations" / "menu_v4_seed_pairs_e00026466-e00026484.json"
SEED = ROOT / "data" / "menu_v4_seed.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load("build_menu_v4_seed")
csi = _load("check_safety_invariants")


# --- the pinned seed ---------------------------------------------------------------


def test_the_committed_inputs_rebuild_the_seed_byte_for_byte():
    text, _ = builder.build(str(LISTING), str(PAIRS))
    assert text.encode("utf-8") == SEED.read_bytes()


def test_the_builder_checks_both_copies_from_the_committed_inputs_by_default(tmp_path,
                                                                             monkeypatch, capsys):
    """`build --check` with no arguments rebuilds from the two committed
    inputs and compares the result with the reviewed copy in data/ and with
    the copy the package ships: the builder, the pin and the package agree."""
    assert builder.main(["build", "--check"]) == 0
    out = capsys.readouterr().out
    assert "data/menu_v4_seed.json rebuilds byte for byte" in out
    assert "fugal_subnet/routing_data/menu_v4_seed.json rebuilds byte for byte" in out
    stale = tmp_path / "menu_v4_seed.json"
    stale.write_bytes(SEED.read_bytes().replace(b'"serial":0', b'"serial":1', 1))
    monkeypatch.setattr(builder, "PACKAGE_SEED_PATH", str(stale))
    with pytest.raises(SystemExit, match="is not what these inputs build"):
        builder.main(["build", "--check"])


def test_the_builder_reads_the_menus_constants_from_menu_state():
    """The token rule the seed is built by is the step's, defined once: the
    builder names menu_state's constant instead of restating its value. Its
    health filter is its own, applied once to the named reveals (the pairs
    file keeps per-operator totals, not the per-epoch counts the live rule
    of section 9.2 reads), and the seed it built is pinned."""
    import ast

    source = (ROOT / "scripts" / "build_menu_v4_seed.py").read_text(encoding="utf-8")
    assigned = {t.id: ast.get_source_segment(source, node.value)
                for node in ast.parse(source).body if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Name)}
    assert "menu_state." in assigned["TOKEN_MIN_PAIRS"]
    assert builder.TOKEN_MIN_PAIRS is ms.TOKEN_MIN_PAIRS
    assert (builder.HEALTH_MIN_OPERATORS, builder.HEALTH_FAIL_RATE, builder.HEALTH_EPOCHS,
            builder.HEALTH_MIN_CALLS) == (3, Fraction(1, 5), 12, 10)


def test_the_seed_is_pinned_and_mirrored_where_the_neurons_read_it():
    data = SEED.read_bytes()
    assert hashlib.sha256(data).hexdigest() == csi.PINNED_MENU_V4_SEED_SHA256
    assert ms.SEED_PATH.read_bytes() == data
    assert "menu_v4_seed.json" in csi.ROUTING_DATA_REQUIRED


def test_the_seed_is_a_valid_state_with_nothing_in_flight():
    seed = ms.load_seed()
    ms.validate(seed)
    assert (seed.epoch, seed.serial, seed.provisional, seed.retired) == (0, 0, (), ())
    assert seed.coldkey_cooldowns == ()
    assert seed.labs == seed.devices == ms.AllowlistRef(0, "")
    assert all((e.rev, e.joined_epoch, e.scored_from, e.proposer) == (1, 0, 0, "")
               for e in seed.entries)
    assert all(e.price_in > 0 and e.price_out > 0 and e.mean_in > 0 and e.mean_out > 0
               and e.token_pairs > 0 for e in seed.entries)


def test_the_first_v4_epoch_runs_on_the_seed_at_that_epoch():
    first = ms.seed_state(26_900)
    assert first.epoch == 26_900 and first.entries == ms.load_seed().entries
    assert not any(e.in_allowance(26_900) for e in first.entries)
    assert ms.menu_hash(first) != ms.menu_hash(ms.load_seed())
    assert ms.menu_for_epoch(first, None) == ms.carry(first)


def test_every_seed_entry_is_a_v3_entry_with_its_model_reasoning_and_hosts_unchanged():
    """What is called is v3's. The answer limit is the listing's, by the rule
    the step applies to an entry every epoch, and differs from v3's pin only
    where the listing moved."""
    seed = ms.load_seed()
    v3 = set(routing_protocol.models_for_protocol(3, routing_protocol.MENU_B))
    assert set(seed.ids()) <= v3
    moved = {}
    for eid in seed.ids():
        ours = ms.request_for(seed, eid)
        theirs = routing_protocol.request_for(3, eid, 0, menu=routing_protocol.MENU_B)
        if ours["max_tokens"] != theirs["max_tokens"]:
            moved[eid] = (theirs["max_tokens"], ours.pop("max_tokens"))
            theirs.pop("max_tokens")
        assert ours == theirs
    assert moved == {"google/gemini-4-carbon@high": (65536, 131072),
                     "google/gemini-4-carbon@low": (65536, 131072)}
    # The one v3 entry left out has no pinned host in the listing.
    assert v3 - set(seed.ids()) == {"xiaomi/mimo-v2.5-pro"}


def _seed_against_its_listing():
    """(entry document, the answer limits its model's listed hosts state)."""
    listing = builder.read_listing(str(LISTING))
    doc = ms.to_dict(ms.load_seed())
    return [(e, builder.host_limits(listing[e["model"]], e["id"])) for e in doc["entries"]]


def test_every_seed_entry_passes_the_compiled_rules_against_the_listing_it_was_built_from():
    """All 59, as entries on the menu: under the lab and host list, with the
    answer limit the rule gives over the pinned hosts the listing shows."""
    allow = builder.lab_list()
    checked = _seed_against_its_listing()
    assert len(checked) == 59
    for entry, limits in checked:
        assert mr.check_entry(entry, allow, host_limits=limits) is None, entry["id"]
        assert entry["max_tokens"] == mr.answer_limit(entry["provider"]["order"], limits,
                                                      on_menu=True)
    assert ms.check_rules(dataclasses.replace(
        ms.load_seed(), labs=ms.AllowlistRef(1, _signed(allow).sha256)), _signed(allow)) is None


def _signed(allow: mr.MenuAllowlist) -> mr.MenuAllowlist:
    """The same labs and hosts as a list with serial 1 that names a signer:
    what the draft becomes once the owner signs it. No key is involved."""
    raw = mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, 1, labs=allow.labs, hosts=allow.hosts,
                                  first_party=allow.first_party)
    return dataclasses.replace(allow, serial=1, sha256=hashlib.sha256(raw).hexdigest(),
                               signer="SHA256:not-a-real-key")


def test_the_seed_entries_with_a_host_the_listing_does_not_show_pass_as_entries_only():
    """Fifty-three entries would also pass as new proposals: every pinned
    host is listed and states a limit. Six pin a second host the listing no
    longer shows; they stay on the menu on the host that is listed, and could
    not be proposed afresh with that order."""
    allow = builder.lab_list()
    as_proposals = {e["id"]: mr.check_request(e, allow, host_limits=limits)
                    for e, limits in _seed_against_its_listing()}
    refused = {eid: r.code for eid, r in as_proposals.items() if r is not None}
    assert refused == {eid: "answer_limit_unknown" for eid in (
        "anthropic/claude-fable-5.2@high", "anthropic/claude-fable-5.2@low",
        "anthropic/claude-fable-5.2@max", "anthropic/claude-fable-5.2@xhigh",
        "google/gemini-4-carbon@high", "google/gemini-4-carbon@low")}
    assert len(as_proposals) - len(refused) == 53


def test_the_first_step_from_the_seed_adopts_a_list_and_keeps_every_entry():
    """The seed names no list. Handed the list its entries were checked
    under, the first step adopts it and retires nothing."""
    allow = _signed(builder.lab_list())
    first = ms.seed_state(26_900)
    mix = builder.scored_mix()
    out = ms.step_detail(ms.carry(first), ms.StepInputs(
        epoch=26_900, nonce=b"\x01" * 32, lab_list=allow),
        ms.MenuParams(mix=mix, min_category_accuracy_bp=5000))
    assert out.state.labs == ms.AllowlistRef(1, allow.sha256)
    assert out.state.entries == first.entries and out.state.retired == ()
    assert [e.kind for e in out.events] == ["lab_list"]


def test_seed_prices_are_the_listings_in_exact_picodollars():
    seed = ms.load_seed()
    listing = builder.read_listing(str(LISTING))
    e = seed.entry("deepseek/deepseek-v4-flash@low")
    hosts = [listing[e.model][h] for h in e.provider_order]
    assert [h["pricing"]["prompt"] for h in hosts] == ["0.00000004998", "0.00000007"]
    assert (e.price_in, e.price_out) == (70_000, 140_000)          # the dearer of the two
    # Every price is some listed pinned host's, to the picodollar.
    for e in seed.entries:
        listed = [listing[e.model][h] for h in e.provider_order if h in listing[e.model]]
        assert Fraction(e.price_in, ms.PICO) in {Fraction(h["pricing"]["prompt"]) for h in listed}
        assert Fraction(e.price_out, ms.PICO) in {Fraction(h["pricing"]["completion"]) for h in listed}
        assert {h["quantization"] for h in listed} == {e.quantization}


def test_the_safety_check_accepts_the_seed_and_names_what_moved(tmp_path, monkeypatch):
    errors: list[str] = []
    csi.check_menu_v4_seed_pinned(errors)
    csi.check_routing_data_mirrors_data(errors)
    assert errors == []

    (tmp_path / "data").mkdir()
    (tmp_path / "fugal_subnet" / "routing_data").mkdir(parents=True)
    moved = SEED.read_bytes().replace(b'"price_in":70000,', b'"price_in":70001,', 1)
    assert moved != SEED.read_bytes()
    (tmp_path / "data" / "menu_v4_seed.json").write_bytes(moved)
    monkeypatch.setattr(csi, "ROOT", tmp_path)
    errors = []
    csi.check_menu_v4_seed_pinned(errors)
    assert len(errors) == 1 and "does not match pin" in errors[0]
    (tmp_path / "data" / "menu_v4_seed.json").unlink()
    errors = []
    csi.check_menu_v4_seed_pinned(errors)
    assert errors == ["data/menu_v4_seed.json is missing"]


def test_no_menu_other_than_the_pinned_seed_ships_in_the_package(tmp_path):
    """The image carries the rules and the seed. A second v4 menu document in
    the package would be a menu no hash pins and no step derived."""
    errors: list[str] = []
    csi.check_no_other_v4_menu_ships(errors)
    assert errors == []

    package = tmp_path / "fugal_subnet"
    (package / "routing_data").mkdir(parents=True)
    (package / "routing_data" / "menu_v4_seed.json").write_bytes(SEED.read_bytes())
    (package / "routing_data" / "models_v3b.json").write_text('[{"id": "a/b@low"}]')
    csi.check_no_other_v4_menu_ships(errors, root=tmp_path)
    assert errors == []
    (package / "tee").mkdir()
    (package / "tee" / "menu_now.json").write_bytes(ms.canonical_json(ms.seed_state(7)))
    (package / "tee" / "not_json.json").write_text("{")
    csi.check_no_other_v4_menu_ships(errors, root=tmp_path)
    assert len(errors) == 1 and "fugal_subnet/tee/menu_now.json" in errors[0]


# --- the builder's rules, on hand-made inputs ----------------------------------------


def _page(model: str, hosts: dict) -> dict:
    raw = json.dumps({"data": {"id": model, "endpoints": [
        {"tag": tag, "quantization": q, "max_completion_tokens": 100_000,
         "pricing": {"prompt": p_in, "completion": p_out}} for tag, (p_in, p_out, q) in hosts.items()]}})
    return {"model": model, "sha256": hashlib.sha256(raw.encode()).hexdigest(), "raw": raw}


def _row(model: str, effort: str, order: list[str]) -> dict:
    return {"id": f"{model}@{effort}", "model": model, "in": 1.0, "out": 2.0,
            "reasoning": {"effort": effort}, "max_tokens": 64_000,
            "provider": {"order": order, "allow_fallbacks": False}}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A two-model menu, its manifest, a listing and a pairs file, all hand-made.
    Returns a function that builds the seed after applying edits."""
    menu = [_row("lab-a/one", "low", ["h1/fp8", "h2/fp8"]), _row("lab-a/one", "high", ["h1/fp8", "h2/fp8"]),
            _row("lab-b/two", "low", ["h3"])]
    manifest = {"models": [{"id": r["id"], "samples": 10, "mean_in_tokens": 248.8,
                            "mean_out_tokens": 1006.6} for r in menu]}
    pages = [_page("lab-a/one", {"h1/fp8": ("0.0000001", "0.0000004", "fp8"),
                                 "h2/fp8": ("0.00000015", "0.0000003", "fp8"),
                                 "other/fp4": ("0.00000001", "0.00000001", "fp4")}),
             _page("lab-b/two", {"h3": ("0.000002", "0.00001", "unknown")})]
    measured = {r["id"]: {"id": r["id"], "cells": {"arc": [20, 4000, 20000], "mmlu": [20, 8000, 60000]},
                          "calls": [[40, 1], [30, 0], [12, 2]]} for r in menu}
    # The lab and host list of this world; `h3` serves a lab that pins no quantization.
    allow = mr.MenuAllowlist(serial=1, labs=frozenset({"lab-a", "lab-b"}),
                             hosts=frozenset({"h1", "h2", "h3", "h8"}),
                             first_party=mr.first_party_map({"lab-b": {"h3"}}))

    def build(edit=lambda menu, manifest, pages, measured: None):
        edit(menu, manifest, pages, measured)
        for name, doc in (("menu.json", menu), ("manifest.json", manifest),
                          ("listing.json", {"format": builder.LISTING_FORMAT, "pages": pages}),
                          ("pairs.json", {"format": builder.PAIRS_FORMAT,
                                          "entries": list(measured.values())})):
            (tmp_path / name).write_text(json.dumps(doc))
        monkeypatch.setattr(builder, "MENU_PATH", str(tmp_path / "menu.json"))
        monkeypatch.setattr(builder, "MANIFEST_PATH", str(tmp_path / "manifest.json"))
        text, notes = builder.build(str(tmp_path / "listing.json"), str(tmp_path / "pairs.json"),
                                    allowlist=allow)
        return ms.from_dict(json.loads(text)), notes

    return build


def test_an_entry_takes_the_higher_input_and_the_higher_output_price_of_its_listed_hosts(world):
    seed, _ = world()
    e = seed.entry("lab-a/one@low")
    assert (e.price_in, e.price_out, e.quantization) == (150_000, 400_000, "fp8")
    # The answer limit is the listing's (100,000 on every host), not the v3 row's 64,000.
    assert (e.max_tokens, e.provider_order, e.rev) == (100_000, ("h1/fp8", "h2/fp8"), 1)
    assert seed.entry("lab-b/two@low").price_out == 10_000_000


def test_the_answer_limit_is_the_smallest_among_the_listed_pinned_hosts_capped(world):
    def limits(*values):
        def edit(menu, manifest, pages, measured):
            page = json.loads(pages[0]["raw"])
            for endpoint, value in zip(page["data"]["endpoints"], values):
                endpoint["max_completion_tokens"] = value
            raw = json.dumps(page)
            pages[0] = {"model": "lab-a/one", "raw": raw,
                        "sha256": hashlib.sha256(raw.encode()).hexdigest()}
        return edit

    assert world(limits(90_000, 70_000, 5))[0].entry("lab-a/one@low").max_tokens == 70_000
    assert world(limits(900_000, 700_000))[0].entry("lab-a/one@low").max_tokens == 131072
    assert world(limits(None, 80_000))[0].entry("lab-a/one@low").max_tokens == 80_000
    with pytest.raises(SystemExit, match="not a count"):
        world(limits("90000", 70_000))


def test_an_entry_the_compiled_rules_refuse_stops_the_build(world):
    def unlisted_lab(menu, manifest, pages, measured):
        menu.append(_row("lab-z/three", "low", ["h1/fp8"]))
        manifest["models"].append(dict(manifest["models"][0], id="lab-z/three@low"))
        pages.append(_page("lab-z/three", {"h1/fp8": ("0.0000001", "0.0000004", "fp8")}))

    with pytest.raises(SystemExit, match="lab_not_allowed"):
        world(unlisted_lab)


def test_token_means_come_from_the_pooled_pairs_weighted_to_the_scored_mix(world):
    seed, _ = world()
    e = seed.entry("lab-a/one@low")
    # Weighted to the pinned v4 mix: arc is everyday (5), mmlu knowledge (15):
    # in = (5 * 200 + 15 * 400) / 20 tokens.
    assert dict(builder.scored_mix())["everyday"] == 5
    assert dict(builder.scored_mix())["knowledge"] == 15
    assert e.token_pairs == 40
    assert e.mean_in == int((2 * Fraction(5 * 200 + 15 * 400, 20) * 1000 + 1) // 2)
    assert e.mean_out == int((2 * Fraction(5 * 1000 + 15 * 3000, 20) * 1000 + 1) // 2)


def test_a_seed_entrys_exploration_pairs_are_its_pooled_pairs_in_the_named_reveals(world):
    seed, _ = world()
    assert seed.entry("lab-a/one@low").explore_pairs == 40


def test_the_committed_seed_counts_every_pooled_pair_of_the_pairs_file():
    measured = builder.read_pairs(str(PAIRS))
    seed = ms.load_seed()
    for e in seed.entries:
        cells = measured.get(e.id, {"cells": {}})["cells"]
        assert e.explore_pairs == sum(n for n, _, _ in cells.values()), e.id
    # Unlike token_pairs, it is never the manifest's sample count.
    assert any(e.explore_pairs < ms.TOKEN_MIN_PAIRS for e in seed.entries)
    assert any(e.explore_pairs != e.token_pairs for e in seed.entries)


def test_with_too_few_pairs_the_manifest_row_stands(world):
    def few(menu, manifest, pages, measured):
        measured["lab-b/two@low"]["cells"] = {"arc": [29, 1, 1]}

    seed, notes = world(few)
    e = seed.entry("lab-b/two@low")
    assert (e.mean_in, e.mean_out, e.token_pairs) == (248_800, 1_006_600, 10)
    assert any("lab-b/two@low" in n and "manifest row" in n for n in notes)


def test_an_entry_with_no_listed_pinned_host_is_left_out_and_one_with_some_is_kept(world):
    def gone(menu, manifest, pages, measured):
        menu[2]["provider"]["order"] = ["h9"]
        menu[0]["provider"]["order"] = ["h1/fp8", "h8/fp8"]

    seed, notes = world(gone)
    assert seed.ids() == ("lab-a/one@high", "lab-a/one@low")
    e = seed.entry("lab-a/one@low")
    assert (e.price_in, e.price_out, e.provider_order) == (100_000, 400_000, ("h1/fp8", "h8/fp8"))
    assert any(n.startswith("LEFT OUT lab-b/two@low") for n in notes)


@pytest.mark.parametrize("calls, left_out", [
    ([[10, 3], [10, 3], [10, 0]], True),            # median 30%
    ([[10, 2], [10, 2], [10, 9]], False),           # median exactly 20%
    ([[10, 9], [10, 9], [9, 9]], False),            # only two operators with enough calls
    ([[10, 0], [10, 1], [10, 4], [10, 10]], True),  # median of four: (10% + 40%) / 2
    ([[10, 0], [10, 1], [10, 3], [10, 10]], False),
])
def test_an_entry_failing_for_the_median_operator_is_left_out(world, calls, left_out):
    def edit(menu, manifest, pages, measured):
        measured["lab-b/two@low"]["calls"] = calls

    seed, _ = world(edit)
    assert ("lab-b/two@low" not in seed.ids()) is left_out
    assert builder.failing(calls) is left_out


def test_listed_hosts_that_differ_in_quantization_stop_the_build(world):
    def edit(menu, manifest, pages, measured):
        pages[0] = _page("lab-a/one", {"h1/fp8": ("0.0000001", "0.0000004", "fp8"),
                                       "h2/fp8": ("0.0000001", "0.0000004", "bf16")})

    with pytest.raises(SystemExit, match="quantization"):
        world(edit)


def test_a_price_that_is_not_a_whole_number_of_picodollars_stops_the_build(world):
    def edit(menu, manifest, pages, measured):
        pages[1] = _page("lab-b/two", {"h3": ("0.0000000000015", "0.00001", "unknown")})

    with pytest.raises(SystemExit, match="picodollars"):
        world(edit)


def test_a_listing_page_that_does_not_match_its_sha256_stops_the_build(world):
    def edit(menu, manifest, pages, measured):
        pages[1]["raw"] = pages[1]["raw"].replace("0.000002", "0.000001")

    with pytest.raises(SystemExit, match="sha256"):
        world(edit)


def test_the_committed_listing_holds_every_page_as_served():
    doc = json.loads(LISTING.read_text(encoding="utf-8"))
    assert doc["format"] == builder.LISTING_FORMAT and len(doc["pages"]) == 25
    for page in doc["pages"]:
        assert hashlib.sha256(page["raw"].encode("utf-8")).hexdigest() == page["sha256"]


# --- the pairs file, from reveals ------------------------------------------------------


def _reveal(n: int, pool: list, calls: list) -> bytes:
    return json.dumps({
        "epoch_id": f"e{n:08d}", "protocol_version": 3,
        "exploration_pool": [{"question_id": q, "model": m, "benchmark": b, "y": 10000,
                              "prompt_tokens": t_in, "completion_tokens": t_out}
                             for q, m, b, t_in, t_out in pool],
        "exploration": [{"question_id": "q", "model": m, "coldkey": ck, "failed": failed}
                        for m, ck, failed in calls]}).encode()


def test_the_pairs_file_sums_tokens_by_benchmark_and_calls_by_operator(tmp_path):
    first = 500
    for i in range(14):
        (tmp_path / f"reveal_e{first + i:08d}.json").write_bytes(_reveal(
            first + i,
            pool=[(f"q{i}", "lab-a/one@low", "arc", 100, 1000), (f"r{i}", "lab-a/one@low", "mmlu", 10, 20)],
            calls=[("lab-a/one@low", "ck-a", i % 2 == 0), ("lab-a/one@low", "ck-b", False),
                   ("lab-b/two@low", "ck-a", True)]))
    text = builder.pairs(str(tmp_path), f"e{first:08d}", f"e{first + 13:08d}")
    doc = json.loads(text)
    assert doc["format"] == builder.PAIRS_FORMAT and len(doc["reveals"]) == 14
    assert doc["reveals"][0]["sha256"] == hashlib.sha256(
        (tmp_path / f"reveal_e{first:08d}.json").read_bytes()).hexdigest()
    # Health reads the last twelve epochs only; tokens every named epoch.
    assert doc["health_epochs"] == [f"e{first + 2:08d}", f"e{first + 13:08d}"]
    rows = {r["id"]: r for r in doc["entries"]}
    assert rows["lab-a/one@low"]["cells"] == {"arc": [14, 1400, 14000], "mmlu": [14, 140, 280]}
    assert rows["lab-a/one@low"]["calls"] == [[12, 0], [12, 6]]       # no operator is named
    assert rows["lab-b/two@low"] == {"id": "lab-b/two@low", "cells": {}, "calls": [[12, 12]]}
    assert "ck-a" not in text
    # The same files give the same bytes, and a missing epoch is refused.
    assert builder.pairs(str(tmp_path), f"e{first:08d}", f"e{first + 13:08d}") == text
    (tmp_path / f"reveal_e{first + 5:08d}.json").unlink()
    with pytest.raises(SystemExit, match="no reveal file"):
        builder.pairs(str(tmp_path), f"e{first:08d}", f"e{first + 13:08d}")


def test_the_committed_pairs_file_names_its_nineteen_reveals():
    doc = json.loads(PAIRS.read_text(encoding="utf-8"))
    ids = [r["epoch_id"] for r in doc["reveals"]]
    assert ids[0] == "e00026466" and ids[-1] == "e00026484" and len(ids) == 19
    assert doc["health_epochs"] == ["e00026473", "e00026484"]
    assert all(len(r["sha256"]) == 64 for r in doc["reveals"])


# --- the pick-cost tables (W13) ----------------------------------------------------------

PICKS = ROOT / "data" / "observations" / "menu_v4_seed_picks_e00026466-e00026509.json"


def test_the_seeds_pick_tables_are_the_step_rule_on_the_committed_graded_cells():
    """Each seed entry's pick-cost table is what the step's shrinkage gives
    on its graded cells in the committed picks file, by v4 category, with
    the prompt side at the entry's call mean (a v3 reveal publishes no
    prompt tokens for a cell); an entry no head routed to has none and is
    picked at its call cost."""
    seed = ms.load_seed()
    graded = builder.graded_sums(str(PICKS))
    params = ms.MenuParams(mix=builder.scored_mix(), min_category_accuracy_bp=0)
    assert (params.pick_prior_cells_entry, params.pick_prior_cells_category) == (30, 20)
    tables = 0
    for e in seed.entries:
        cells = {c: (n, n * e.mean_in, ms.MILLI * out)
                 for c, (n, out) in graded.get(e.id, {}).items()}
        assert e.pick == ms.pick_table_from_sums(cells, dataclasses.replace(e, pick=None),
                                                 params), e.id
        if e.pick is None:
            assert not graded.get(e.id)
            assert ms.pick_costs(seed)[e.id] == ms.costs(seed)[e.id]
            continue
        tables += 1
        assert e.pick.cells == sum(n for n, _ in graded[e.id].values())
        assert e.pick.mean_in == e.mean_in
        assert all(c.mean_in == e.mean_in for c in e.pick.categories)
        assert {c.category for c in e.pick.categories} <= {c for c, _ in builder.scored_mix()}
    assert tables >= 50
    assert ms.check_caps(seed) is None


def test_the_picks_file_names_the_same_reveals_as_the_pairs_file_where_they_overlap():
    picks = json.loads(PICKS.read_text(encoding="utf-8"))
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    ids = [r["epoch_id"] for r in picks["reveals"]]
    assert ids[0] == "e00026466" and ids[-1] == "e00026509" and len(ids) == 44
    assert ids == sorted(set(ids))
    named = {r["epoch_id"]: r["sha256"] for r in picks["reveals"]}
    assert all(named[r["epoch_id"]] == r["sha256"] for r in pairs["reveals"])
    assert picks["clamped_cells"] == 16


def test_the_picks_file_is_built_from_the_reveals_by_its_rule(tmp_path):
    """The picks subcommand on a hand-made reveal: a cell counts once with
    its published completion tokens rounded half up and held to the answer
    limit; a head's failed call and a pick with no cell count nothing; the
    champion's picks are also every head's."""
    limit = {r["id"]: r["max_tokens"] for r in json.loads(
        (ROOT / "data" / "models_v3b.json").read_text())}
    a, b = sorted(limit)[:2]
    reveal = {
        "epoch_id": "e00000007", "protocol_version": 3,
        "questions": [{"question_id": "q1", "benchmark": "aime"},
                      {"question_id": "q2", "benchmark": "mmlupro"}],
        "models": [a, b],
        "shared_grades": [
            {"question_id": "q1", "model": a, "grade": 1.0, "completion_tokens": 10.5},
            {"question_id": "q2", "model": a, "grade": 0.0,
             "completion_tokens": float(limit[a] + 100)},
            {"question_id": "q2", "model": b, "grade": 1.0, "completion_tokens": 3.0}],
        "champion": {"champion": "h1"},
        "v2_state": {"serving": {"h1": 11, "h2": 12}},
        "routing_decisions": {"11": [0, 0], "12": [1, 1]},
        "failed": {"12": ["q2"]},
        "exploration_pool": [{"question_id": "q9", "model": b, "benchmark": "arc",
                              "completion_tokens": 40}],
    }
    (tmp_path / "reveal_e00000007.json").write_text(json.dumps(reveal))
    text = builder.picks(str(tmp_path), "e00000007", "e00000007")
    out = tmp_path / "picks.json"
    out.write_text(text)
    epochs, rows = builder.read_picks(str(out))
    assert epochs == ["e00000007"]
    got = {(e, bm): v for _r, e, bm, v in rows}
    assert got[(a, "aime")] == (1, 11, 1, 11, 1, 11, 0, 0)
    assert got[(a, "mmlupro")] == (1, limit[a], 1, limit[a], 1, limit[a], 0, 0)
    assert got[(b, "mmlupro")] == (1, 3, 0, 0, 0, 0, 0, 0)     # h2's pick failed
    assert (b, "aime") not in got                               # h2 picked b: no such cell
    assert got[(b, "arc")] == (0, 0, 0, 0, 0, 0, 1, 40)
    assert json.loads(text)["clamped_cells"] == 1
