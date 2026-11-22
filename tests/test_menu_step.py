"""Protocol v4: the menu derivation step (docs/decisions/LIVE_MENU_V4.md, 4.4).

    MenuState(k + 2) = step(MenuState(k + 1), inputs(k))

Every test hands the step a hand-made state and hand-made inputs: the
aggregations that produce real inputs (endpoint votes, token sums, health
counts, measurement verification) belong to fugal_subnet/menu_inputs.py. What
is held here is what the step does with them, in the order the design gives,
and that it is a pure function: the same state and inputs give the same bytes
whatever order the inputs arrive in, in any process.
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from tests import menu_fixtures as fx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIGH, LOW, TWO = "lab-a/model-one@high", "lab-a/model-one@low", "lab-b/model-two"


def run(state: ms.MenuState, params: ms.MenuParams = fx.PARAMS, **inputs) -> ms.StepResult:
    """One step from `state`, on the inputs of the scored epoch it follows."""
    return ms.step_detail(state, fx.inputs(epoch=state.epoch - 1, **inputs), params)


def kinds(result: ms.StepResult) -> list[tuple[str, str]]:
    return [(e.kind, e.subject) for e in result.events]


# The make-up rule is the same for a measurement of any length. v4 compiles
# two epochs of 150 questions (SEAT_MINIMAL_V4.md 4.6), where a skipped step
# can only fall on the first or the last epoch; the tests of how skipped steps
# are made up also walk a five-epoch measurement of 60 a epoch (v4's earlier
# setting), where one can fall in the middle and several can be made up.
# --- the shape of a step ---------------------------------------------------------


def test_with_nothing_to_apply_the_menu_is_carried_and_the_serial_stands():
    s = fx.state()
    out = run(s)
    assert out.state == ms.carry(s) and out.state.serial == s.serial and out.events == ()


def test_the_serial_rises_by_one_when_anything_changes_and_only_then():
    s = fx.state()
    reads = (fx.read("lab-b/model-two", "host-c", price_in=2_100_000, price_out=10_000_000),)
    changed = run(s, host_reads=reads).state
    assert changed.serial == s.serial + 1 and changed.epoch == s.epoch + 1
    # The same reads again change nothing more.
    again = run(changed, host_reads=reads).state
    assert again.serial == changed.serial and again == ms.carry(changed)


def test_a_step_starts_from_the_menu_one_lag_after_the_scored_epoch():
    s = fx.state(epoch=101)
    for wrong in (99, 101, 102):
        with pytest.raises(ValueError, match="starts from the menu of epoch"):
            ms.step(s, fx.inputs(epoch=wrong), fx.PARAMS)
    lag3 = dataclasses.replace(fx.PARAMS, menu_lag=3)
    assert ms.step(s, fx.inputs(epoch=99), lag3).epoch == 102


def test_a_step_needs_the_epochs_nonce():
    with pytest.raises(ValueError, match="nonce"):
        ms.step(fx.state(), ms.StepInputs(epoch=100, nonce=b""), fx.PARAMS)


def test_the_same_key_given_twice_is_refused_not_resolved_by_order():
    s = fx.state()
    r = fx.read("lab-b/model-two", "host-c")
    with pytest.raises(ValueError, match="twice"):
        run(s, host_reads=(r, dataclasses.replace(r, price_in=1)))
    h = ms.HealthStat(TWO, 1)
    with pytest.raises(ValueError, match="twice"):
        run(s, health=(h, h))


# --- 1. signed lists: the step applies the compiled rules itself -----------------------
#
# The step is handed the signed list a node holds, never a verdict about it:
# which entries and proposals a newer list no longer covers is decided here,
# by menu_rules, so two nodes cannot be told different things.


def _without(**dropped) -> mr.MenuAllowlist:
    """The list after `fx.ALLOW` (serial 3), without the named labs, hosts or
    first-party marks."""
    fields = dict(labs=fx.LABS, hosts=fx.HOSTS)
    marks = dropped.pop("first_party", ())
    for name, gone in dropped.items():
        fields[name] = tuple(x for x in fields[name] if x not in gone)
    fields["first_party"] = {lab: tuple(h for h in hs if h in fields["hosts"] and h not in marks)
                             for lab, hs in fx.FIRST_PARTY.items()}
    return fx.allowlist(3, **fields)


def test_a_newer_list_is_adopted_and_an_entry_whose_lab_left_it_retires():
    s = fx.state()
    newer = _without(labs={"lab-b"})
    out = run(s, lab_list=newer)
    assert out.state.labs == ms.AllowlistRef(3, newer.sha256)
    assert out.state.ids() == (HIGH, LOW)
    assert out.state.retired == (ms.Retired(TWO, 102, ms.REASON_LAB_REMOVED, 102 + 48),)
    assert ("retired", TWO) in kinds(out) and ("lab_list", "3") in kinds(out)


def test_an_entry_with_a_pinned_host_that_left_the_list_retires():
    out = run(fx.state(), lab_list=_without(hosts={"host-b"})).state
    assert out.ids() == (TWO,)
    assert {r.id: r.reason for r in out.retired} == {
        HIGH: ms.REASON_HOST_REMOVED, LOW: ms.REASON_HOST_REMOVED}


def test_an_entry_with_no_pinned_quantization_retires_when_its_host_loses_the_first_party_mark():
    out = run(fx.state(), lab_list=_without(first_party={"host-c"})).state
    assert out.ids() == (HIGH, LOW)
    assert out.retired == (ms.Retired(TWO, 102, ms.REASON_HOST_REMOVED, 150),)


def test_the_reasons_a_list_retires_for_are_the_rules_list_refusals():
    assert set(ms.LIST_REFUSAL_REASONS) == mr.ALLOWLIST_REFUSALS
    assert set(ms.LIST_REFUSAL_REASONS.values()) == {ms.REASON_LAB_REMOVED, ms.REASON_HOST_REMOVED}


def test_a_newer_list_that_covers_everything_changes_only_the_list_in_force():
    s = fx.state()
    newer = fx.allowlist(3, labs=fx.LABS + ("lab-new",))
    out = run(s, lab_list=newer).state
    assert out.entries == s.entries and out.labs == ms.AllowlistRef(3, newer.sha256)
    assert out.serial == s.serial + 1


def test_the_list_in_force_changes_nothing_and_any_other_list_is_refused():
    s = fx.state()                                   # holds serial 2
    assert run(s, lab_list=fx.ALLOW).state == ms.carry(s)
    for wrong, why in (
            (None, "needs the signed"),                                        # no list at all
            (fx.allowlist(1), "in force"),                                     # an older one
            (fx.allowlist(2, labs=fx.LABS[:-1]), "in force"),                  # another list, same serial
            (fx.allowlist(3, signer=None), "verified"),                        # one nobody verified
            (dataclasses.replace(fx.allowlist(3), sha256="cd" * 32), "hash"),  # content under another hash
            (dataclasses.replace(fx.allowlist(3), labs=frozenset()), "hash"),
            (ms.AllowlistRef(3, "cd" * 32), "needs the signed")):              # a reference is not a list
        with pytest.raises(ValueError, match=why):
            run(s, lab_list=wrong)


def test_a_state_that_names_no_list_adopts_one_in_its_first_step():
    """The seed names no list. The first step is handed the list the release
    ships and adopts it; no step runs without one."""
    s = fx.state(labs=ms.AllowlistRef(0, ""))
    with pytest.raises(ValueError, match="needs the signed"):
        run(s, lab_list=None)
    first = fx.allowlist(1)
    out = run(s, lab_list=first).state
    assert out.labs == ms.AllowlistRef(1, first.sha256) and out.entries == s.entries
def test_a_newer_device_list_is_adopted_and_an_older_one_is_not():
    held = fx.device_list(4)
    s = fx.state(devices=ms.AllowlistRef(4, held.sha256))
    assert run(s, device_list=fx.device_list(5)).state.devices.serial == 5
    assert run(s, device_list=held).state == ms.carry(s)
    with pytest.raises(ValueError, match="in force"):
        run(s, device_list=fx.device_list(3))


# --- 2. prices and answer limits ---------------------------------------------------


def test_an_entry_is_priced_at_its_dearer_pinned_host_and_limited_by_the_smaller():
    reads = (fx.read("lab-a/model-one", "host-a/fp8", price_in=150_000, price_out=600_000,
                     limit=393_216),
             fx.read("lab-a/model-one", "host-b/fp8", price_in=98_000, price_out=294_000,
                     limit=100_000))
    out = run(fx.state(), host_reads=reads).state
    for eid in (HIGH, LOW):
        e = out.entry(eid)
        assert (e.price_in, e.price_out, e.max_tokens) == (150_000, 600_000, 100_000)
        assert e.rev == 1                              # a price or limit move is not a request update
    assert out.entry(TWO) == fx.state().entry(TWO)


def test_the_answer_budget_never_exceeds_the_cap_and_a_host_with_no_stated_limit_sets_none():
    reads = (fx.read("lab-b/model-two", "host-c", limit=943_718),)
    assert run(fx.state(), host_reads=reads).state.entry(TWO).max_tokens == 131072
    s = fx.state(fx.entry("lab-b/model-two", None, provider_order=("host-c",), max_tokens=4096))
    reads = (fx.read("lab-b/model-two", "host-c", limit=0),)
    assert run(s, host_reads=reads).state.entry(TWO).max_tokens == 131072


@pytest.mark.parametrize("voters, moved", [(0, False), (2, False), (3, True), (40, True)])
def test_a_price_moves_only_on_a_read_with_enough_voters(voters, moved):
    reads = (fx.read("lab-b/model-two", "host-c", price_in=3_000_000, price_out=1_000_000,
                     voters=voters),)
    out = run(fx.state(), host_reads=reads).state
    assert (out.entry(TWO).price_in == 3_000_000) is moved


def test_a_price_does_not_move_while_any_pinned_host_lacks_a_read():
    reads = (fx.read("lab-a/model-one", "host-a/fp8", price_in=1, price_out=1),)
    assert run(fx.state(), host_reads=reads).state == ms.carry(fx.state())
    reads += (fx.read("lab-a/model-one", "host-b/fp8", price_in=1, price_out=1, voters=2),)
    assert run(fx.state(), host_reads=reads).state == ms.carry(fx.state())


def test_an_unlisted_host_does_not_price_an_entry_and_no_listed_host_changes_nothing():
    reads = (fx.read("lab-a/model-one", "host-a/fp8", listed=False, price_in=0, price_out=0, limit=0),
             fx.read("lab-a/model-one", "host-b/fp8", price_in=77_000, price_out=188_000, limit=50_000))
    e = run(fx.state(), host_reads=reads).state.entry(HIGH)
    assert (e.price_in, e.price_out, e.max_tokens) == (77_000, 188_000, 50_000)
    none = tuple(dataclasses.replace(r, listed=False) for r in reads)
    assert run(fx.state(), host_reads=none).state == ms.carry(fx.state())


def test_the_step_and_the_rules_agree_on_an_entry_whose_pinned_host_left_the_listing():
    """What the step derives from the reads is what menu_rules says of an
    entry on the menu under the same listing; the same request filed afresh
    is refused, since a new proposal needs every pinned host listed."""
    reads = (fx.read("lab-a/model-one", "host-a/fp8", listed=False, price_in=0, price_out=0, limit=0),
             fx.read("lab-a/model-one", "host-b/fp8", price_in=77_000, price_out=88_000, limit=50_000))
    out = run(fx.state(), host_reads=reads).state
    listing = {r.host: r.limit for r in reads if r.listed}
    for eid in (HIGH, LOW):
        doc = ms.to_dict(out)["entries"][out.ids().index(eid)]
        assert mr.check_entry(doc, fx.ALLOW, host_limits=listing) is None
        assert doc["max_tokens"] == mr.answer_limit(doc["provider"]["order"], listing, on_menu=True)
        assert mr.check_request(doc, fx.ALLOW, host_limits=listing).code == "answer_limit_unknown"
    assert ms.check_rules(out, fx.ALLOW) is None
# --- 3. token means ----------------------------------------------------------------


def _stat(eid, rev=1, pairs=40, tokens_in=300, tokens_out=2000):
    """`pairs` pairs split over two categories of the mix."""
    half = pairs // 2
    return ms.TokenStat(eid, rev, (
        ms.TokenCell("code", half, tokens_in * half, tokens_out * half),
        ms.TokenCell("math", pairs - half, tokens_in * (pairs - half), 2 * tokens_out * (pairs - half))))


def test_token_means_follow_the_window_weighted_to_the_mix():
    out = run(fx.state(), token_stats=(_stat(TWO),)).state
    e = out.entry(TWO)
    # code 15, math 45 of the mix: out = (15 * 2000 + 45 * 4000) / 60 tokens.
    assert (e.mean_in, e.mean_out, e.token_pairs) == (300_000, 3_500_000, 40)
    assert out.entry(HIGH) == fx.state().entry(HIGH)


@pytest.mark.parametrize("pairs, moved", [(0, False), (29, False), (30, True)])
def test_below_the_minimum_pairs_the_previous_means_stand(pairs, moved):
    out = run(fx.state(), token_stats=(_stat(TWO, pairs=pairs),)).state
    assert (out.entry(TWO).token_pairs == pairs) is moved
    assert (out == ms.carry(fx.state())) is not moved


def test_pairs_measured_under_another_revision_do_not_move_the_means():
    s = fx.state(fx.entry("lab-b/model-two", None, rev=2))
    assert run(s, token_stats=(_stat(TWO, rev=1),)).state == ms.carry(s)
    assert run(s, token_stats=(_stat(TWO, rev=2),)).state.entry(TWO).token_pairs == 40


# --- 3, continued: each entry's exploration pairs ------------------------------------
#
# What exploration's coverage part reads (section 8): an entry's pooled
# exploration pairs in the window, for its current revision. The step
# publishes the count it is handed; measurement pairs are never in it.


def test_an_entrys_exploration_pair_count_is_the_count_it_is_handed_for_its_revision():
    s = fx.state()
    out = run(s, explore_counts=(ms.ExploreCount(HIGH, 1, 140), ms.ExploreCount(LOW, 2, 9),
                                 ms.ExploreCount("lab-z/gone@low", 1, 77))).state
    assert out.explore_pairs() == {HIGH: 140, LOW: 0, TWO: 0}         # LOW's count is of rev 2
    assert out.serial == s.serial + 1
    # No count handed in: the published one stands.
    assert run(out).state.explore_pairs() == out.explore_pairs()
    with pytest.raises(ValueError, match="twice"):
        run(s, explore_counts=(ms.ExploreCount(HIGH, 1, 1), ms.ExploreCount(HIGH, 1, 2)))
    with pytest.raises(ValueError):
        run(s, explore_counts=(ms.ExploreCount(HIGH, 1, -1),))
# --- 4. health ---------------------------------------------------------------------


@pytest.mark.parametrize("eligible, failing, retires", [
    (4, 3, True), (5, 3, True), (9, 5, True), (4, 2, False), (9, 4, False),
    (3, 3, False), (1, 1, False), (0, 0, False), (30, 0, False),
])
def test_an_entry_retires_when_more_than_half_of_at_least_four_eligible_coldkeys_fail_on_it(
        eligible, failing, retires):
    """Section 9.2 as the replay fixed it: the aggregation hands in how many
    coldkeys are eligible on the entry (enough calls in the window) and how
    many of them are failing on it (failure rate and distinct epochs); the
    step retires it when at least four are eligible and more than half fail."""
    h = ms.HealthStat(TWO, 1, eligible=eligible, failing=failing)
    out = run(fx.state(), health=(h,)).state
    assert (TWO not in out.ids()) is retires
    if retires:
        assert out.retired == (ms.Retired(TWO, 102, ms.REASON_FAILING, 150),)


@pytest.mark.parametrize("epochs, retires", [(5, False), (6, True), (60, True)])
def test_an_entry_retires_when_none_of_its_hosts_has_been_listed_for_long_enough(epochs, retires):
    out = run(fx.state(), health=(ms.HealthStat(TWO, 1, unlisted_epochs=epochs),)).state
    assert (TWO not in out.ids()) is retires
    if retires:
        assert out.retired[0].reason == ms.REASON_GONE


def test_health_counts_that_contradict_themselves_are_refused():
    for bad in (dict(eligible=3, failing=4), dict(eligible=-1), dict(failing=-1),
                dict(eligible=True), dict(unlisted_epochs=1.0)):
        with pytest.raises(ValueError, match="health"):
            run(fx.state(), health=(ms.HealthStat(TWO, 1, **bad),))


def test_the_health_constants_are_the_replays():
    """docs/decisions/LIVE_MENU_V4.md 9.2, the thresholds W0's replay fixed."""
    assert (ms.RETIRE_WINDOW_EPOCHS, ms.RETIRE_MIN_CALLS, ms.RETIRE_FAIL_RATE_BP,
            ms.RETIRE_FAIL_MIN_EPOCHS, ms.RETIRE_FAIL_MIN_COLDKEYS,
            ms.RETIRE_OUTAGE_RATE_BP) == (12, 5, 3000, 3, 4, 3000)
    assert fx.PARAMS.retire_fail_min_coldkeys == 4


def test_health_of_another_revision_retires_nothing():
    h = ms.HealthStat(TWO, 7, eligible=9, failing=9, unlisted_epochs=99)
    assert run(fx.state(), health=(h,)).state == ms.carry(fx.state())
# --- 6. a place kept free for a join: displacement -----------------------------------------
def _full_menu(n=4, **entry):
    return tuple(fx.entry(f"lab-a/m{i}", "low", provider_order=("host-c",), **entry)
                 for i in range(n))


def test_under_the_cap_nothing_is_displaced():
    params = dataclasses.replace(fx.PARAMS, menu_cap=5)
    s = fx.state(*_full_menu(4))
    out = run(s, params, use_total=1000).state
    assert len(out.entries) == 4 and out.retired == ()


def test_a_full_menu_gives_up_its_least_used_entry_so_a_join_has_a_place():
    params = dataclasses.replace(fx.PARAMS, menu_cap=4)
    s = fx.state(*_full_menu(4))
    use = (ms.EntryUse("lab-a/m0@low", 500), ms.EntryUse("lab-a/m1@low", 3),
           ms.EntryUse("lab-a/m2@low", 9), ms.EntryUse("lab-a/m3@low", 488))
    out = run(s, params, use=use, use_total=1000)
    assert "lab-a/m1@low" not in out.state.ids() and len(out.state.entries) == 3
    assert out.state.retired == (ms.Retired("lab-a/m1@low", 102, ms.REASON_DISPLACED, 150),)
    assert ms.check_transition(s, out.state, params) is None


def test_ties_in_use_go_to_the_lower_measured_accuracy_then_the_id():
    params = dataclasses.replace(fx.PARAMS, menu_cap=4)
    s = fx.state(*_full_menu(4))
    acc = (ms.AccuracyStat("lab-a/m0@low", 1, 200, 700_000, 0),
           ms.AccuracyStat("lab-a/m1@low", 1, 200, 650_000, 0),
           ms.AccuracyStat("lab-a/m2@low", 1, 200, 650_000, 0),
           ms.AccuracyStat("lab-a/m3@low", 1, 200, 990_000, 0))
    out = run(s, params, accuracy=acc, use_total=1000).state           # nobody picks any of them
    assert out.retired[0].id == "lab-a/m1@low"
    # An entry nobody measured goes before any measured one.
    out = run(s, params, accuracy=acc[:3], use_total=1000).state
    assert out.retired[0].id == "lab-a/m3@low"


def test_a_shielded_entry_or_one_above_the_use_floor_is_never_displaced():
    params = dataclasses.replace(fx.PARAMS, menu_cap=4)
    fresh = fx.entry("lab-a/m0", "low", provider_order=("host-c",), joined_epoch=90, scored_from=95)
    s = fx.state(fresh, *_full_menu(4)[1:])
    use = tuple(ms.EntryUse(f"lab-a/m{i}@low", 11) for i in (1, 2, 3))    # 1.1% each: above 1%
    out = run(s, params, use=use, use_total=1000)
    assert out.state.ids() == s.ids() and ("menu_full", "4") in kinds(out)
    # At exactly the floor an entry can be displaced; the shielded one never.
    at_floor = (ms.EntryUse("lab-a/m2@low", 10),) + use[:1] + use[2:]
    assert run(s, params, use=at_floor, use_total=1000).state.retired[0].id == "lab-a/m2@low"
    shield_over = dataclasses.replace(s, epoch=95 + ms.ENTRY_SHIELD_EPOCHS - 1)
    got = ms.step(shield_over, fx.inputs(epoch=shield_over.epoch - 1, use_total=1000), params)
    assert got.retired[0].id == "lab-a/m0@low"


def test_displacement_never_leaves_fewer_than_the_minimum_of_full_entries():
    """Found by test_any_step_yields_a_valid_state_one_epoch_on: three full
    entries and two provisional ones (which the gates decide one step later),
    on a menu with room for three. Health retires two full entries ("gone",
    "failing"), which leaves the one full entry the minimum (1 here) asks
    for, and the menu at its cap. Displacing that last full entry, as the
    step did when it counted provisional entries toward the minimum, left
    none; the next step's gates retired both provisional entries, and the
    menu was empty."""
    params = dataclasses.replace(fx.PARAMS, menu_cap=3)
    s = fx.provisional(fx.provisional(fx.state(), "lab-c/model-x"), "lab-c/model-y")
    health = (ms.HealthStat(HIGH, 1, eligible=6, unlisted_epochs=6),
              ms.HealthStat(TWO, 1, eligible=6, failing=4))
    out = run(s, params, health=health, use=(), use_total=3)
    assert [(x.kind, x.subject) for x in out.events if x.kind == "retired"] == [
        ("retired", HIGH), ("retired", TWO)]
    assert ("menu_full", "3") in kinds(out)
    nxt = out.state
    assert ms.full_count(nxt.entries) == 1 and nxt.ids()[0] == LOW
    reader = dataclasses.replace(params, menu_cap=5)
    assert ms.menu_for_epoch(s, ms.menu_hash(nxt), ms.canonical_json(nxt), allowlist=fx.ALLOW,
                             params=reader) == nxt
    # The gates retire both provisional entries; the full one stays.
    last = run(nxt, dataclasses.replace(params, menu_cap=1)).state
    assert last.ids() == (LOW,) and not last.provisional
    # The menu the old step derived (no full entry) is refused by the reader.
    empty = dataclasses.replace(nxt, entries=tuple(e for e in nxt.entries if e.provisional))
    assert "0 full entries" in ms.check_caps(empty, reader)


def test_with_no_counted_pick_there_is_no_evidence_of_use_and_nothing_is_displaced():
    params = dataclasses.replace(fx.PARAMS, menu_cap=4)
    s = fx.state(*_full_menu(4))
    out = run(s, params, use_total=0).state
    assert out.ids() == s.ids()


def test_the_reader_holds_a_displacement_to_one_on_a_full_menu():
    params = dataclasses.replace(fx.PARAMS, menu_cap=5)
    s = fx.state(*_full_menu(4))
    gone = dataclasses.replace(ms.carry(s), serial=s.serial + 1, entries=s.entries[1:],
                               retired=(ms.Retired(s.entries[0].id, 102, ms.REASON_DISPLACED,
                                                   150),))
    assert "keep one place free" in ms.check_transition(s, gone, params)


def test_retired_holds_its_newest_records():
    params = dataclasses.replace(fx.PARAMS, retired_keep=3)
    old = tuple(ms.Retired(f"lab-z/old-{i}@low", 10 * i, ms.REASON_GONE, 10 * i + 48)
                for i in range(1, 4))
    s = fx.state(retired=old)
    out = run(s, params, health=(ms.HealthStat(TWO, 1, unlisted_epochs=6),)).state
    assert [r.id for r in out.retired] == ["lab-b/model-two", "lab-z/old-2@low", "lab-z/old-3@low"]


def test_a_cooldown_still_running_is_never_dropped_from_the_retired_record():
    """A record makes room only once its cooldown can refuse nothing more. An
    older record whose cooldown runs stays; a newer one that has run out goes."""
    params = dataclasses.replace(fx.PARAMS, retired_keep=3)
    running = ms.Retired("lab-z/long@low", 10, ms.REASON_GONE, 500)
    over = (ms.Retired("lab-z/over-1@low", 50, ms.REASON_GONE, 98),
            ms.Retired("lab-z/over-2@low", 53, ms.REASON_GONE, 101))   # ends as scored epoch 101 files
    s = fx.state(retired=(running,) + over)
    out = run(s, params, health=(ms.HealthStat(TWO, 1, unlisted_epochs=6),)).state
    assert [r.id for r in out.retired] == ["lab-b/model-two", "lab-z/long@low", "lab-z/over-2@low"]
    # With nothing run out, nothing is dropped, whatever the record's size.
    all_running = tuple(ms.Retired(f"lab-c/run-{i}@low", 90 + i, ms.REASON_GONE, 138 + i)
                        for i in range(3))
    out = run(fx.state(retired=all_running), params,
              health=(ms.HealthStat(TWO, 1, unlisted_epochs=6),)).state
    assert {r.id for r in out.retired} == {r.id for r in all_running} | {TWO}
# --- purity ------------------------------------------------------------------------


def _busy_inputs(state: ms.MenuState, rnd) -> dict:
    """Inputs that touch every part of the step, in an order `rnd` chooses."""
    reads = [fx.read("lab-a/model-one", "host-a/fp8", price_in=150_000, limit=90_000),
             fx.read("lab-a/model-one", "host-b/fp8", price_out=500_000),
             fx.read("lab-b/model-two", "host-c", voters=2)]
    stats = [_stat(HIGH, pairs=64), _stat(LOW, pairs=12), _stat(TWO, pairs=31)]
    health = [ms.HealthStat(LOW, 1, eligible=4, failing=3),
              ms.HealthStat(HIGH, 1, eligible=4, failing=1)]
    gates = [fx.gate(p.id, correct=200) for p in state.provisional]
    for seq in (reads, stats, health, gates):
        rnd.shuffle(seq)
    return dict(host_reads=tuple(reads), token_stats=tuple(stats), health=tuple(health),
                gates=tuple(gates), use=(ms.EntryUse(HIGH, 5),), use_total=100)


@settings(max_examples=40, deadline=None)
@given(st.randoms(use_true_random=False), st.integers(1, 6))
def test_the_order_inputs_arrive_in_never_changes_the_result(rnd, steps):
    import random

    a = b = fx.provisional(fx.state())
    for _ in range(steps):
        a = run(a, **_busy_inputs(a, random.Random(0))).state
        b = run(b, **_busy_inputs(b, rnd)).state
        assert ms.canonical_json(a) == ms.canonical_json(b)
    assert ms.from_json(ms.canonical_json(a)) == a


def test_a_step_reads_its_state_and_inputs_and_nothing_else():
    """The same call twice gives the same bytes, and leaves its arguments alone."""
    import random

    s = fx.provisional(fx.state())
    kwargs = _busy_inputs(s, random.Random(3))
    before = ms.canonical_json(s)
    first, second = run(s, **kwargs), run(s, **kwargs)
    assert first == second and ms.canonical_json(s) == before
    assert first.state.serial == s.serial + 1 and first.events


def test_a_second_process_derives_the_same_menu_hashes():
    code = (
        "import random, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from fugal_subnet import menu_state as ms\n"
        "from tests import menu_fixtures as fx\n"
        "from tests.test_menu_step import _busy_inputs, run\n"
        "s, out = fx.provisional(fx.state()), []\n"
        "for i in range(9):\n"
        "    s = run(s, **_busy_inputs(s, random.Random(int(sys.argv[2]) + i))).state\n"
        "    out.append(ms.menu_hash(s))\n"
        "print(' '.join(out))\n")

    def hashes(hash_seed: str, order_seed: str) -> str:
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        return subprocess.run([sys.executable, "-c", code, ROOT, order_seed], capture_output=True,
                              text=True, env=env, timeout=300, check=True).stdout.strip()

    a, b = hashes("0", "1"), hashes("4242", "77")
    assert a == b and len(set(a.split())) == 9


@settings(max_examples=60, deadline=None)
@given(st.data())
def test_any_step_yields_a_valid_state_one_epoch_on(data):
    """Whatever well-typed inputs arrive, the result is a state that encodes,
    for the next epoch, with the serial moved by one or not at all."""
    s = fx.provisional(fx.provisional(fx.state(), "lab-c/model-x"), "lab-c/model-y")
    keep = data.draw(st.integers(5, 9))                  # the three entries have their places
    for _ in range(data.draw(st.integers(1, 4))):
        ids = list(s.ids())
        draw_int = lambda hi: data.draw(st.integers(0, hi))                     # noqa: E731
        reads = tuple(
            ms.HostRead(e.model, h, voters=draw_int(5), listed=data.draw(st.booleans()),
                        price_in=draw_int(10**7), price_out=draw_int(10**8), limit=draw_int(10**6))
            for e in {e.model: e for e in s.entries}.values() for h in e.provider_order)
        stats = tuple(ms.TokenStat(i, draw_int(2), (ms.TokenCell("code", draw_int(80), draw_int(10**6),
                                                                 draw_int(10**7)),))
                      for i in ids if data.draw(st.booleans()))
        health = tuple(ms.HealthStat(i, 1, eligible=6 + draw_int(4), failing=draw_int(6),
                                     unlisted_epochs=draw_int(8))
                       for i in ids if data.draw(st.booleans()))
        gates = tuple(fx.gate(p.id, correct=draw_int(240), upper_ppm=draw_int(10**6))
                      for p in s.provisional if data.draw(st.booleans()))
        accuracy = tuple(ms.AccuracyStat(i, 1, draw_int(300), draw_int(10**6), draw_int(10**6))
                         for i in ids)
        params = dataclasses.replace(fx.PARAMS, menu_cap=data.draw(st.integers(1, 6)),
                                     retired_keep=keep)
        out = run(s, params, host_reads=reads, token_stats=stats, health=health, gates=gates,
                  accuracy=accuracy,
                  use=tuple(ms.EntryUse(i, draw_int(3)) for i in ids), use_total=draw_int(50) + 3)
        nxt = out.state
        ms.validate(nxt)
        # The step never leaves fewer full entries than the minimum.
        assert ms.full_count(nxt.entries) >= params.min_entries
        # What the step derives from a menu that passes the compiled rules
        # passes them: no node refuses a menu an honest root computed.
        assert ms.check_rules(nxt, fx.ALLOW) is None
        # The retired record stays within its size, and no cooldown that still
        # runs was dropped to keep it there.
        assert len(nxt.retired) <= keep
        still = {r.id for r in nxt.retired}
        assert all(r.id in still for r in s.retired if r.cooldown_until > s.epoch)
        assert nxt.epoch == s.epoch + 1 and nxt.serial - s.serial in (0, 1)
        assert (nxt.serial == s.serial) == (nxt == ms.carry(s))
        assert ms.from_json(ms.canonical_json(nxt)) == nxt
        assert len(nxt.provisional) <= len(s.provisional)        # a step adds none
        assert len(nxt.entries) <= max(params.menu_cap, len(s.entries))
        # Every node that holds `s` reads the derived menu back from its
        # bytes: it passes the caps and the transition checks. (The cap is
        # drawn anew each step here, below the three entries the run starts
        # with, which no release does; the reader is held to the larger.)
        if nxt != ms.carry(s):
            reader = dataclasses.replace(params, menu_cap=max(params.menu_cap, len(nxt.entries)))
            assert ms.menu_for_epoch(s, ms.menu_hash(nxt), ms.canonical_json(nxt),
                                     allowlist=fx.ALLOW, params=reader) == nxt
        s = nxt
