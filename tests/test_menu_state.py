"""Protocol v4: the menu as published state (docs/decisions/LIVE_MENU_V4.md, section 4).

The menu of an epoch is a value every node must hold byte for byte: its hash
is in every proof's namespace. These tests hold the data model to that: one
canonical encoding, integers and strings only, the same hash however the
state was built or parsed, and one rule for which state an epoch runs on.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from fugal_subnet import routing_protocol
from fugal_subnet.vendor import success_contract as contract
from tests import menu_fixtures as fx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _full_state() -> ms.MenuState:
    """A state with something in every field."""
    base = fx.provisional(fx.state())
    retired = (ms.Retired(id="lab-z/old@low", epoch=90, reason="failing", cooldown_until=138),
               ms.Retired(id="lab-y/gated@high", epoch=95, reason="gate_useful",
                          cooldown_until=143))
    out = dataclasses.replace(base, retired=tuple(sorted(retired, key=lambda r: r.id)),
                              coldkey_cooldowns=(ms.ColdkeyCooldown("5Fa-coldkey", 150),
                                                 ms.ColdkeyCooldown("5Fsome-coldkey", 143)))
    ms.validate(out)
    return out


# --- canonical JSON and the hash ---------------------------------------------


def test_the_hash_is_the_sha256_of_sorted_compact_ascii_json():
    s = _full_state()
    raw = ms.canonical_json(s)
    assert raw == json.dumps(ms.to_dict(s), sort_keys=True, separators=(",", ":")).encode("ascii")
    assert b" " not in raw and b"\n" not in raw
    assert ms.menu_hash(s) == hashlib.sha256(raw).hexdigest()
    # The same digest rule the proof namespace already uses.
    assert ms.menu_hash(s) == contract.digest(ms.to_dict(s))


def test_a_state_survives_a_round_trip_through_its_bytes():
    s = _full_state()
    assert ms.from_json(ms.canonical_json(s)) == s
    assert ms.from_dict(json.loads(ms.canonical_json(s))) == s


def test_the_document_holds_integers_strings_and_the_two_fixed_booleans_only():
    def leaves(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                assert isinstance(k, str)
                yield from leaves(v, f"{path}.{k}")
        elif isinstance(x, list):
            for v in x:
                yield from leaves(v, path + "[]")
        else:
            yield path, x

    for path, v in leaves(ms.to_dict(_full_state())):
        assert not isinstance(v, float) and v is not None, path
        if isinstance(v, bool):
            assert path.endswith(".allow_fallbacks") or path.endswith(".reasoning.enabled"), path
        else:
            assert isinstance(v, (int, str)), path


def test_key_order_of_a_parsed_document_does_not_move_the_hash():
    s = _full_state()

    def reverse_keys(x):
        if isinstance(x, dict):
            return {k: reverse_keys(x[k]) for k in reversed(list(x))}
        if isinstance(x, list):
            return [reverse_keys(v) for v in x]
        return x

    shuffled = reverse_keys(json.loads(ms.canonical_json(s)))
    assert list(shuffled) != list(ms.to_dict(s))
    assert ms.menu_hash(ms.from_dict(shuffled)) == ms.menu_hash(s)


def test_a_second_process_with_another_hash_seed_computes_the_same_bytes(tmp_path):
    s = _full_state()
    doc = tmp_path / "state.json"
    doc.write_bytes(json.dumps(ms.to_dict(s), indent=3).encode())   # not the canonical layout
    code = ("import sys, json; from fugal_subnet import menu_state as ms; "
            "s = ms.from_dict(json.load(open(sys.argv[1]))); "
            "sys.stdout.write(ms.menu_hash(s) + ' ' + ms.canonical_json(s).hex())")
    env = dict(os.environ, PYTHONHASHSEED="12345", PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, "-c", code, str(doc)], capture_output=True, text=True,
                         env=env, timeout=120, check=True).stdout.split()
    assert out == [ms.menu_hash(s), ms.canonical_json(s).hex()]


@pytest.mark.parametrize("change", [
    lambda e: dataclasses.replace(e, price_in=e.price_in + 1),
    lambda e: dataclasses.replace(e, mean_out=e.mean_out + 1),
    lambda e: dataclasses.replace(e, rev=e.rev + 1),
    lambda e: dataclasses.replace(e, provider_order=tuple(reversed(e.provider_order))),
    lambda e: dataclasses.replace(e, scored_from=e.scored_from + 1),
    lambda e: dataclasses.replace(e, max_tokens=e.max_tokens - 1),
    lambda e: dataclasses.replace(e, explore_pairs=e.explore_pairs + 1),
])
def test_any_change_to_an_entry_moves_the_hash(change):
    s = fx.state()
    entries = (change(s.entries[0]),) + s.entries[1:]
    assert ms.menu_hash(dataclasses.replace(s, entries=entries)) != ms.menu_hash(s)


def test_each_entrys_exploration_pair_count_is_published_state():
    s = fx.state(fx.entry("lab-a/model-one", "high", explore_pairs=140),
                 fx.entry("lab-b/model-two", None, provider_order=("host-c",), explore_pairs=0))
    assert s.explore_pairs() == {"lab-a/model-one@high": 140, "lab-b/model-two": 0}
    doc = ms.to_dict(s)
    assert [e["explore_pairs"] for e in doc["entries"]] == [140, 0]
    assert ms.from_json(ms.canonical_json(s)) == s
    missing = json.loads(ms.canonical_json(s))
    del missing["entries"][0]["explore_pairs"]
    with pytest.raises(ValueError):
        ms.from_dict(missing)
    with pytest.raises(ValueError):
        ms.validate(fx.state(fx.entry("lab-a/model-one", "high", explore_pairs=-1)))


def test_the_epoch_and_the_serial_are_in_the_hash():
    s = fx.state()
    assert ms.menu_hash(dataclasses.replace(s, epoch=s.epoch + 1)) != ms.menu_hash(s)
    assert ms.menu_hash(dataclasses.replace(s, serial=s.serial + 1)) != ms.menu_hash(s)


def test_a_proposers_cooldown_is_part_of_the_hashed_state():
    """Who may file is consensus: the coldkeys in cooldown are in the document,
    keyed by coldkey, beside the ids in cooldown."""
    s = _full_state()
    doc = ms.to_dict(s)
    assert doc["coldkey_cooldowns"] == {"5Fa-coldkey": 150, "5Fsome-coldkey": 143}
    assert s.cooldowns() == doc["coldkey_cooldowns"]
    assert set(doc["retired"]["lab-y/gated@high"]) == {"epoch", "reason", "cooldown_until"}
    moved = dataclasses.replace(s, coldkey_cooldowns=(ms.ColdkeyCooldown("5Fa-coldkey", 151),
                                                      s.coldkey_cooldowns[1]))
    assert ms.menu_hash(moved) != ms.menu_hash(s)
    assert ms.to_dict(fx.state())["coldkey_cooldowns"] == {}


# --- what a state may hold -----------------------------------------------------


@pytest.mark.parametrize("bad", [
    lambda d: d["entries"][0].__setitem__("price_in", 1.5),
    lambda d: d["entries"][0].__setitem__("price_in", -1),
    lambda d: d["entries"][0].__setitem__("price_in", 2**53),
    lambda d: d["entries"][0].__setitem__("price_in", True),
    lambda d: d["entries"][0].__setitem__("mean_in", None),
    lambda d: d["entries"][0].__setitem__("extra", 1),
    lambda d: d["entries"][0].pop("quantization"),
    lambda d: d["entries"][0].__setitem__("id", "café/model@high"),
    lambda d: d["entries"][0].__setitem__("model", "another/model"),
    lambda d: d["entries"][0]["provider"].__setitem__("allow_fallbacks", True),
    lambda d: d["entries"][0]["provider"].__setitem__("order", []),
    lambda d: d["entries"][0].__setitem__("reasoning", {"effort": "high", "enabled": True}),
    lambda d: d["entries"][0].__setitem__("scored_from", -1),
    lambda d: d["entries"].reverse(),
    lambda d: d["entries"].append(dict(d["entries"][0])),
    lambda d: d.__setitem__("serial", "3"),
    lambda d: d.pop("provisional"),
    lambda d: d.__setitem__("unknown", []),
    lambda d: d["allowlists"]["labs"].__setitem__("sha256", "xyz"),
    lambda d: d["retired"].__setitem__("lab-q/x", {"epoch": 1, "reason": "gone"}),
    lambda d: d["retired"]["lab-z/old@low"].__setitem__("coldkey", "5Fa-coldkey"),
    lambda d: d["retired"]["lab-z/old@low"].__setitem__("reason", "unpopular"),
    lambda d: d.pop("coldkey_cooldowns"),
    lambda d: d.__setitem__("coldkey_cooldowns", [["5Fa-coldkey", 150]]),
    lambda d: d["coldkey_cooldowns"].__setitem__("5Fa-coldkey", "150"),
    lambda d: d["coldkey_cooldowns"].__setitem__("", 150),
    lambda d: d["provisional"][0].__setitem__("proposal", "nothex"),
    lambda d: d["provisional"][0].__setitem__("extra", 1),
    lambda d: d["provisional"].append(dict(d["provisional"][0])),
    lambda d: d.__setitem__("provisional", []),          # a provisional entry with no record
    lambda d: d["provisional"][0].__setitem__("proposer", "5Fsomeone-else"),
])
def test_a_malformed_document_is_refused(bad):
    d = json.loads(ms.canonical_json(_full_state()))
    bad(d)
    with pytest.raises(ValueError):
        ms.from_dict(d)


@pytest.mark.parametrize("raw", [
    b'{"epoch": 1e2}', b'{"epoch": NaN}', b'{"epoch": 1, "epoch": 2}', b"[]", b"\xff\xfe", b"",
])
def test_bytes_that_are_not_a_menu_document_are_refused(raw):
    with pytest.raises(ValueError):
        ms.from_json(raw)


def test_bytes_in_another_layout_are_not_the_canonical_document():
    s = _full_state()
    pretty = json.dumps(ms.to_dict(s), indent=1).encode()
    with pytest.raises(ValueError, match="canonical"):
        ms.from_json(pretty)


@settings(max_examples=60, deadline=None)
@given(st.binary(max_size=200))
def test_arbitrary_bytes_never_raise_anything_but_value_error(raw):
    try:
        ms.from_json(raw)
    except ValueError:
        pass


_INT = st.integers(min_value=0, max_value=2**53 - 1)
_HOST = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-/", min_size=1, max_size=12)


@st.composite
def _entries(draw):
    n = draw(st.integers(min_value=0, max_value=5))
    out = []
    for i in range(n):
        effort = draw(st.sampled_from(["low", "medium", "high", "max"]))
        model = f"lab-{draw(st.integers(0, 3))}/model-{i}"
        joined = draw(st.integers(0, 10**6))
        out.append(ms.Entry(
            id=f"{model}@{effort}", rev=draw(st.integers(1, 9)), model=model,
            reasoning=draw(st.sampled_from([("effort", effort), ("max_tokens", 4096),
                                            ("enabled", True), None])),
            max_tokens=draw(st.integers(1, 131072)),
            provider_order=tuple(draw(st.lists(_HOST, min_size=1, max_size=2, unique=True))),
            quantization=draw(st.sampled_from(["fp8", "bf16", "unknown"])),
            price_in=draw(_INT), price_out=draw(_INT), mean_in=draw(_INT), mean_out=draw(_INT),
            token_pairs=draw(_INT), joined_epoch=joined,
            scored_from=joined + draw(st.integers(0, 48)),
            measured_rate=draw(st.integers(0, ms.RATE_SCALE)),
            proposer=draw(st.sampled_from(["", "5Fhk"]))))
    return tuple(sorted(out, key=lambda e: e.id))


@settings(max_examples=80, deadline=None)
@given(_entries(), _INT, _INT, st.randoms(use_true_random=False))
def test_every_valid_state_has_one_encoding(entries, epoch, serial, rnd):
    s = ms.MenuState(epoch=epoch, serial=serial, entries=entries, retired=(),
                     labs=ms.AllowlistRef(0, ""), devices=ms.AllowlistRef(0, ""))
    ms.validate(s)
    raw = ms.canonical_json(s)
    assert ms.from_json(raw) == s
    assert ms.canonical_json(ms.from_json(raw)) == raw
    # Parsed from a document whose keys arrive in another order: same bytes.
    d = json.loads(raw)
    keys = list(d)
    rnd.shuffle(keys)
    assert ms.canonical_json(ms.from_dict({k: d[k] for k in keys})) == raw


# --- which state an epoch runs on ----------------------------------------------


def test_with_no_commitment_an_epoch_carries_the_previous_menu():
    prev = _full_state()
    nxt = ms.menu_for_epoch(prev, None)
    assert nxt == dataclasses.replace(prev, epoch=prev.epoch + 1)
    assert nxt.serial == prev.serial


def test_a_commitment_to_the_carried_menu_needs_no_bytes():
    prev = fx.state()
    carried = ms.carry(prev)
    assert ms.menu_for_epoch(prev, ms.menu_hash(carried)) == carried


def _changed(prev: ms.MenuState, **over) -> ms.MenuState:
    """A menu for the epoch after `prev` that differs from the carried one."""
    fields = dict(serial=prev.serial + 1, entries=(
        dataclasses.replace(prev.entries[0], price_in=123_456),) + prev.entries[1:])
    fields.update(over)
    return dataclasses.replace(ms.carry(prev), **fields)


def _offer(prev: ms.MenuState, state: ms.MenuState, allowlist=fx.ALLOW) -> ms.MenuState:
    # fx.PARAMS: the price `_changed` moves may move in any epoch
    # (tests/test_menu_refresh.py holds the cadence).
    return ms.menu_for_epoch(prev, ms.menu_hash(state), ms.canonical_json(state),
                             allowlist=allowlist, params=fx.PARAMS)


def test_a_committed_change_is_accepted_from_any_source_that_matches_the_hash():
    prev = fx.state()
    changed = _changed(prev)
    assert _offer(prev, changed) == changed


def test_a_committed_change_without_its_bytes_is_unavailable_not_guessed():
    prev = fx.state()
    with pytest.raises(ms.MenuUnavailable):
        ms.menu_for_epoch(prev, "ab" * 32, allowlist=fx.ALLOW)
    with pytest.raises(ms.MenuUnavailable):
        ms.menu_for_epoch(prev, "ab" * 32, ms.canonical_json(ms.carry(prev)), allowlist=fx.ALLOW)


def test_a_node_that_cannot_check_a_changed_menu_has_no_menu_to_run_the_epoch_on(monkeypatch):
    """Without the bytes of the committed hash, or the list the menu names, a
    node gets no menu, and nothing it holds is a stand-in: the menu it does
    hold names another proof namespace than the epoch's, so a TD that ran on
    it would produce no proof any validator accepts for the epoch."""
    fx.allow_small_menus(monkeypatch)
    prev = fx.state()
    newer = fx.allowlist(3, labs=fx.LABS + ("lab-x",))
    for committed, published, held in (
            (_changed(prev), None, fx.ALLOW),                                  # no bytes
            (_changed(prev, labs=ms.AllowlistRef(3, newer.sha256), entries=prev.entries),
             True, fx.ALLOW)):                                                  # no list
        raw = ms.canonical_json(committed) if published else None
        with pytest.raises(ms.MenuUnavailable) as unavailable:
            ms.menu_for_epoch(prev, ms.menu_hash(committed), raw, allowlist=held)
        assert not isinstance(unavailable.value, ms.MenuRefused)
        assert not hasattr(unavailable.value, "carried")
        epoch_namespace = routing_protocol.identity_v4(ms.menu_hash(committed))
        assert routing_protocol.identity_v4(ms.menu_hash(ms.carry(prev))) != epoch_namespace
        assert routing_protocol.identity_v4(ms.menu_hash(prev)) != epoch_namespace


def test_committed_bytes_for_another_epoch_or_out_of_sequence_are_refused():
    prev = fx.state()
    wrong_epoch = dataclasses.replace(prev, epoch=prev.epoch + 2, serial=prev.serial + 1)
    with pytest.raises(ms.MenuRefused) as refused:
        _offer(prev, wrong_epoch)
    assert refused.value.carried == ms.carry(prev)
    skipped = dataclasses.replace(ms.carry(prev), serial=prev.serial + 5)
    with pytest.raises(ms.MenuRefused):
        _offer(prev, skipped)


def _refused(prev: ms.MenuState, state: ms.MenuState, allowlist=fx.ALLOW) -> str:
    """Offer `state` as the committed menu; it must be refused, with the
    carried menu named as what the epoch runs on. Returns the refusal."""
    with pytest.raises(ms.MenuRefused) as refused:
        _offer(prev, state, allowlist)
    assert refused.value.carried == ms.carry(prev)
    return str(refused.value)


def test_a_committed_menu_that_fails_the_compiled_rules_is_refused_and_the_previous_one_carried():
    """The root's commitment names the menu, and every node checks it: an
    entry or a proposal the compiled rules refuse makes the whole menu one no
    node runs on. Each of them runs the epoch on the previous menu instead."""
    prev = fx.state()
    cases = {
        "lab_not_allowed": fx.entry("lab-x/model", "high"),
        "host_not_allowed": fx.entry("lab-a/other", "high", provider_order=("host-x/fp8",)),
        "host_tier": fx.entry("lab-a/other", "high", provider_order=("host-a/fp8/flex",)),
        "too_many_hosts": fx.entry(
            "lab-a/other", "high", provider_order=("host-a/fp8", "host-b/fp8", "host-c/fp8")),
        "unknown_quantization": fx.entry("lab-a/other", "high", provider_order=("host-a",)),
        "quantization_mismatch": fx.entry("lab-a/other", "high", quantization="bf16"),
        "reasoning_not_allowed": fx.entry(
            "lab-a/other", "high", id="lab-a/other", reasoning=("max_tokens", 2048)),
        "id_mismatch": fx.entry("lab-a/other", "high", reasoning=("effort", "low")),
        "answer_limit": fx.entry("lab-a/other", "high", max_tokens=131073),
    }
    for code, e in cases.items():
        # An entry on the menu (however it got there) that the rules refuse
        # makes any changed menu one no node runs on.
        before, state = _holding(prev, e)
        assert _refused(before, state).split(": ")[1] == code, code


def _holding(prev: ms.MenuState, e: ms.Entry) -> tuple[ms.MenuState, ms.MenuState]:
    """`prev` holding `e`, and the menu after it with a price moved."""
    before = dataclasses.replace(prev, entries=tuple(sorted(prev.entries + (e,),
                                                            key=lambda x: x.id)))
    return before, _changed(before)


def test_a_changed_menu_is_checked_under_the_signed_list_it_names():
    prev = fx.state()                                          # names list 2
    newer = fx.allowlist(3, labs=fx.LABS + ("lab-x",))
    # An entry of a lab only the newer list names, on the menu that adopts it.
    held = dataclasses.replace(prev, entries=tuple(sorted(
        prev.entries + (fx.entry("lab-x/model", "high"),), key=lambda e: e.id)))
    prev = dataclasses.replace(prev, entries=held.entries)
    adopts = _changed(prev, labs=ms.AllowlistRef(3, newer.sha256))
    assert _offer(prev, adopts, newer) == adopts               # passes under the list it names
    # A node that does not hold that list cannot check the menu: it is not
    # refused (others accept it), and nothing is guessed.
    for held in (None, fx.ALLOW, fx.allowlist(3)):
        with pytest.raises(ms.MenuUnavailable):
            _offer(prev, adopts, held)
    with pytest.raises(ms.MenuUnavailable):
        _offer(prev, _changed(prev), None)
    # A list nobody verified is the caller's mistake, never a verdict.
    with pytest.raises(ValueError, match="verified"):
        _offer(prev, _changed(prev), fx.allowlist(2, signer=None))


def test_a_changed_menu_cannot_go_back_to_an_older_list_or_swap_the_one_in_force():
    prev = fx.state()                                          # names list 2
    older, other = fx.allowlist(1), fx.allowlist(2, labs=fx.LABS + ("lab-x",))
    for held in (older, other):
        assert "list" in _refused(prev, _changed(prev, labs=ms.AllowlistRef(
            held.serial, held.sha256)), held)
    back = fx.state(devices=ms.AllowlistRef(4, "ee" * 32))
    assert "list" in _refused(back, _changed(back, devices=ms.AllowlistRef(3, "ee" * 32)))
    # A menu changes only under a signed list: one that names none is refused.
    unlisted = fx.state(labs=ms.AllowlistRef(0, ""))
    assert "names no signed list" in _refused(unlisted, _changed(unlisted), None)
    # Carrying it needs no list, and neither does the first list's adoption
    # being checked: the menu that adopts list 1 is checked under list 1.
    assert ms.menu_for_epoch(unlisted, None) == ms.carry(unlisted)
    first = fx.allowlist(1)
    adopted = _changed(unlisted, labs=ms.AllowlistRef(1, first.sha256))
    assert _offer(unlisted, adopted, first) == adopted


def test_every_menu_the_step_derives_is_accepted_by_the_reader():
    """What an honest root computes, every node accepts: the step's output,
    offered as the committed menu, follows the menu the step started from."""
    prev = fx.state()
    newer = fx.allowlist(3, labs=tuple(lab for lab in fx.LABS if lab != "lab-b"))
    derived = ms.step(prev, fx.inputs(lab_list=newer, host_reads=(
        fx.read("lab-a/model-one", "host-a/fp8", price_in=150_000),
        fx.read("lab-a/model-one", "host-b/fp8", listed=False))), fx.PARAMS)
    assert derived.serial == prev.serial + 1
    assert _offer(prev, derived, newer) == derived


# --- reading an entry ----------------------------------------------------------


def test_request_for_reads_the_request_from_the_entry():
    s = fx.state()
    assert ms.request_for(s, "lab-a/model-one@high") == {
        "model": "lab-a/model-one", "max_tokens": 131072, "reasoning": {"effort": "high"},
        "provider": {"order": ["host-a/fp8", "host-b/fp8"], "allow_fallbacks": False}}
    assert ms.request_for(s, "lab-b/model-two")["reasoning"] == {"enabled": True}
    with pytest.raises(KeyError):
        ms.request_for(s, "lab-a/absent@low")


def test_an_entry_costs_its_prices_times_its_token_means_exactly():
    e = fx.entry(price_in=49_980, price_out=99_960, mean_in=248_800, mean_out=1_234_567)
    assert ms.entry_cost(e) == (Fraction(49_980, 10**12) * Fraction(248_800, 1000)
                                + Fraction(99_960, 10**12) * Fraction(1_234_567, 1000))
    assert isinstance(ms.entry_cost(e), Fraction)
    assert ms.costs(fx.state())["lab-b/model-two"] == Fraction(
        2_000_000 * 250_000 + 10_000_000 * 900_000, 10**15)


def test_token_means_are_weighted_to_the_scored_mix_in_milli_tokens():
    cells = (ms.TokenCell("code", 10, 1000, 5000),       # 100 in, 500 out a pair
             ms.TokenCell("math", 30, 6000, 30001),      # 200 in, 1000.03.. out
             ms.TokenCell("unlisted", 5, 999, 999))      # no weight in the mix: left out
    mean_in, mean_out, pairs = ms.weighted_token_means(cells, fx.MIX)
    assert pairs == 45
    assert mean_in == 175_000                            # (15*100 + 45*200) / 60
    exact_out = (15 * Fraction(5000, 10) + 45 * Fraction(30001, 30)) / 60
    assert mean_out == int((2 * exact_out * 1000 + 1) // 2)
    # No category of the mix measured: the plain mean over what there is.
    assert ms.weighted_token_means((ms.TokenCell("unlisted", 4, 10, 30),), fx.MIX) == (2500, 7500, 4)
    assert ms.weighted_token_means((), fx.MIX) == (0, 0, 0)
    # The order the cells arrive in does not matter.
    assert ms.weighted_token_means(tuple(reversed(cells)), fx.MIX) == (mean_in, mean_out, pairs)


@pytest.mark.parametrize("pairs, tokens, milli", [
    (3, 1, 333),         # 333.33..: down
    (3, 2, 667),         # 666.66..: up
    (16, 1, 63),         # 62.5 exactly: a half goes up
    (16, 3, 188),        # 187.5 exactly
    (7, 22, 3143),       # 3142.857..
])
def test_a_token_mean_is_rounded_half_up_to_the_milli_token(pairs, tokens, milli):
    cells = (ms.TokenCell("code", pairs, tokens, tokens),)
    assert ms.weighted_token_means(cells, fx.MIX) == (milli, milli, pairs)


# --- the proof namespace -------------------------------------------------------


def test_identity_v4_names_the_epochs_menu():
    s = fx.state()
    ident = routing_protocol.identity_v4(ms.menu_hash(s))
    assert ident == contract.digest({
        "protocol": "fugal-success-v4", "contract": contract.CONTRACT,
        "profile": contract.PROFILE_ID, "lambda": routing_protocol.BENCHMARK_LAMBDA,
        "menu": ms.menu_hash(s)})
    other = dataclasses.replace(s, serial=s.serial + 1)
    assert routing_protocol.identity_v4(ms.menu_hash(other)) != ident
    assert ident not in {routing_protocol.identity_for(v) for v in (1, 2, 3)}


@pytest.mark.parametrize("bad", ["", "abc", "AB" * 32, "g" * 64, None, 5, "ab" * 33])
def test_identity_v4_refuses_anything_that_is_not_a_menu_hash(bad):
    with pytest.raises(ValueError):
        routing_protocol.identity_v4(bad)


def test_the_earlier_namespaces_are_unchanged():
    """identity_v4 is new code beside the pinned identities, not a change to them."""
    assert [routing_protocol.identity_for(v) for v in (1, 2, 3)] == [
        "49390166e93fbf5c4bdb7f29e813a8015df4a2affaae64e421eed1d3ce5f0a1a",
        "ad0dadc8e47e7ae571e74d6d71b36f6a5437d42d4c208f810f3fbe95da8e7522",
        "3b7bada761de74f86fb7fdd3887af4252e24d375ca56584aa7d3e74c01ef7b8a"]
    with pytest.raises(KeyError):
        routing_protocol.identity_for(4)


# --- one definition of each constant the rules and the state share -----------------


def _module_constants(module) -> set[str]:
    """The upper-case names a module assigns at its top level."""
    import ast
    import inspect

    names: set[str] = set()
    for node in ast.parse(inspect.getsource(module)).body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        names |= {t.id for t in targets if isinstance(t, ast.Name) and t.id.isupper()}
    return names


def test_each_constant_the_rules_and_the_state_share_is_defined_once():
    """The answer cap is a compiled rule and lives with the rules; the menu's
    sizes and timings live with the state. Neither module restates a constant
    of the other, so the two cannot come to disagree on one."""
    rules, state = _module_constants(mr), _module_constants(ms)
    assert rules & state == set()
    assert "MAX_ANSWER_TOKENS" in rules
    assert {"MENU_CAP", "MENU_MAX_PROVISIONAL", "MENU_LAG", "ENTRY_SHIELD_EPOCHS"} <= state
    # The cap is not a parameter of the step either. The step limits an answer
    # with the rules' own function, so the state module holds the rules'
    # value or none.
    assert "max_tokens_cap" not in {f.name for f in dataclasses.fields(ms.MenuParams)}
    assert getattr(ms, "MAX_ANSWER_TOKENS", mr.MAX_ANSWER_TOKENS) is mr.MAX_ANSWER_TOKENS
    reads = (fx.read("lab-b/model-two", "host-c", limit=mr.MAX_ANSWER_TOKENS + 1),)
    stepped = ms.step(fx.state(), fx.inputs(host_reads=reads), fx.PARAMS)
    assert stepped.entry("lab-b/model-two").max_tokens == mr.MAX_ANSWER_TOKENS
