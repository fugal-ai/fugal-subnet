"""Protocol v4: what every node checks of a committed menu before running an
epoch on it (docs/decisions/LIVE_MENU_V4.md, 4.3; `menu_state.menu_for_epoch`).

Beyond the sequence and the compiled rules, a committed menu that differs
from the carried one must

- stay within the compiled caps: entries, provisional entries, and the size
  of the document;
- name a newer signed list (lab and host, or device) only together with the
  list itself, which the node holds;
- follow from the menu the node holds by one step: no entry joins in a step
  (entries join at a boundary, by the root's join delta:
  tests/test_v4_chain_proposals.py), an entry keeps its request,
  `joined_epoch` and `scored_from` (a provisional one becomes full only in the
  step that decides it), and no cooldown that is still running is shortened
  or dropped.

A menu that fails any of these is refused on every node, and the epoch runs
on the carried menu, exactly as for a menu the compiled rules refuse.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random

import pytest

from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from tests import menu_fixtures as fx
from tests.signed_list_fixtures import Signer, device_bytes

HIGH, LOW, TWO = "lab-a/model-one@high", "lab-a/model-one@low", "lab-b/model-two"
K, NEW = 100, 102          # fx.state() is the menu of epoch 101: the step for scored epoch 100
P5 = fx.PARAMS


def offer(prev, state, *, allowlist=fx.ALLOW, params=P5, **kw) -> ms.MenuState:
    return ms.menu_for_epoch(prev, ms.menu_hash(state), ms.canonical_json(state),
                             allowlist=allowlist, params=params, **kw)


def refused(prev, state, **kw) -> str:
    """`state`, committed for the epoch after `prev`, must be refused, and the
    carried menu named as what the epoch runs on. Returns why."""
    with pytest.raises(ms.MenuRefused) as r:
        offer(prev, state, **kw)
    assert r.value.carried == ms.carry(prev)
    return str(r.value)


def changed(prev, **over) -> ms.MenuState:
    """The menu after `prev` with a price moved (a change any step may make)."""
    entries = over.pop("entries", None)
    if entries is None:
        entries = tuple(dataclasses.replace(e, price_in=e.price_in + 1) if e.id == HIGH else e
                        for e in prev.entries)
    return dataclasses.replace(ms.carry(prev), serial=prev.serial + 1,
                               entries=tuple(sorted(entries, key=lambda e: e.id)), **over)
# --- 1. the compiled caps ----------------------------------------------------------------


def _wide(n: int, epoch: int = 101) -> ms.MenuState:
    entries = [fx.entry(f"lab-a/m{i:03d}", "high") for i in range(n)]
    return fx.state(*entries, epoch=epoch)


def test_a_committed_menu_with_more_entries_than_the_cap_is_refused():
    at_cap, over_cap = _wide(ms.MENU_CAP), _wide(ms.MENU_CAP + 1)
    first = at_cap.entries[0].id
    bump = lambda s: tuple(dataclasses.replace(e, price_in=e.price_in + 7) if e.id == first else e    # noqa: E731
                           for e in s.entries)
    # The compiled cap applies (fx.PARAMS holds it; its price may move in any epoch).
    assert ms.menu_for_epoch(at_cap, *_bytes(changed(at_cap, entries=bump(at_cap))),
                             allowlist=fx.ALLOW, params=fx.PARAMS) == changed(at_cap, entries=bump(at_cap))
    with pytest.raises(ms.MenuRefused, match="cap"):
        ms.menu_for_epoch(over_cap, *_bytes(changed(over_cap, entries=bump(over_cap))),
                          allowlist=fx.ALLOW, params=fx.PARAMS)
    # And a menu with a provisional entry counts it against the cap.
    prev = fx.provisional(fx.state())
    state = changed(prev)
    assert offer(prev, state) == state
    assert "cap" in refused(prev, state, params=dataclasses.replace(fx.PARAMS, menu_cap=3))


def _bytes(state):
    return ms.menu_hash(state), ms.canonical_json(state)
def test_a_committed_document_over_the_byte_limit_is_refused_unread():
    prev = fx.state()
    raw = b" " * (ms.MENU_MAX_BYTES + 1)
    import hashlib

    with pytest.raises(ms.MenuRefused, match="bytes") as r:
        ms.menu_for_epoch(prev, hashlib.sha256(raw).hexdigest(), raw, allowlist=fx.ALLOW)
    assert r.value.carried == ms.carry(prev)


def test_the_largest_menu_the_caps_allow_fits_under_the_byte_limit():
    """Every string at its longest, every list at its cap: 64 entries, each
    with a pick-cost table at its category cap, eight provisional records,
    a full retired record
    (RETIRED_KEEP) and the coldkey cap's coldkeys in cooldown
    (`coldkey_cooldown_cap`): every list a reader accepts, at its cap."""
    s = ms.MAX_STRING

    def name(prefix: str, i: int, room: int = s) -> str:
        head = f"{prefix}{i:04d}-"
        return head + "x" * (room - len(head))

    big = 2**53 - 1
    # A pick-cost table on every entry, at its category cap, each category
    # named at its longest.
    pick = ms.PickTable(cells=big, mean_in=big, mean_out=big, categories=tuple(
        ms.PickCategory(name(f"c{i}", 0, ms.PICK_CATEGORY_MAX_CHARS), 1 if i else big - 7,
                        big, big) for i in range(ms.PICK_MAX_CATEGORIES)))
    entries = []
    for i in range(ms.MENU_CAP):
        model = name("lab-a/m", i, s - len("@xhigh"))
        entries.append(ms.Entry(
            id=f"{model}@xhigh", rev=big, model=model, reasoning=("effort", "xhigh"),
            max_tokens=big, provider_order=(name("h", i) + "/fp8"[:0], name("g", i)),
            quantization=name("q", i), price_in=big, price_out=big, mean_in=big, mean_out=big,
            token_pairs=big, joined_epoch=big - 1, scored_from=big - 1,
            measured_rate=ms.RATE_SCALE,
            proposer=name("p", i),
            pick=pick))
    # The provisional entries at their cap, each with its record.
    prov = []
    for i in range(ms.MENU_MAX_PROVISIONAL):
        e = entries[i]
        entries[i] = ms.Entry(**{**{f: getattr(e, f) for f in e.__dataclass_fields__},
                                 "rev": 1, "scored_from": ms.PROVISIONAL,
                                 "proposer": name("p", i)})
        prov.append(ms.Provisional(id=e.id, proposal=f"{i:064x}", proposer=name("p", i),
                                   coldkey=name("c", i)))
    retired = tuple(ms.Retired(name("lab-r/m", i), big, ms.REASON_GATE_CATEGORY, big)
                    for i in range(ms.RETIRED_KEEP))
    coldkeys = tuple(ms.ColdkeyCooldown(name("ck", i), big)
                     for i in range(ms.coldkey_cooldown_cap()))
    state = ms.MenuState(epoch=big, serial=big, entries=tuple(sorted(entries, key=lambda e: e.id)),
                         provisional=tuple(sorted(prov, key=lambda p: p.id)),
                         retired=retired, labs=ms.AllowlistRef(big, "ab" * 32),
                         devices=ms.AllowlistRef(big, "cd" * 32), coldkey_cooldowns=coldkeys)
    size = len(ms.canonical_json(state))
    assert size <= ms.MENU_MAX_BYTES // 2, size
    # The seed is a small fraction of it.
    assert len(ms.canonical_json(ms.load_seed())) < ms.MENU_MAX_BYTES // 16


# --- 2. a newer list is adopted only with the list itself -----------------------------------


def test_a_committed_menu_adopts_a_newer_device_list_only_with_that_list():
    prev = fx.state()                                       # names no device list yet
    held = fx.device_list(1)
    adopts = changed(prev, devices=ms.AllowlistRef(1, held.sha256))
    assert offer(prev, adopts, device_list=held) == adopts
    # Without the list, or with another one, the node cannot check the menu.
    for other in (None, fx.device_list(1, ("ee" * 32,)), fx.device_list(2)):
        with pytest.raises(ms.MenuUnavailable):
            offer(prev, adopts, device_list=other)
    # A bare reference with a higher serial and an arbitrary hash names no list
    # any node holds, so no node adopts it.
    bare = changed(prev, devices=ms.AllowlistRef(9, "ab" * 32))
    for other in (None, held, fx.device_list(9)):
        with pytest.raises(ms.MenuUnavailable):
            offer(prev, bare, device_list=other)
    # A list that was not verified, or whose content is not what its hash
    # says, or of the other kind, is the caller's mistake, never a verdict.
    for wrong in (fx.device_list(1, signer=None),
                  dataclasses.replace(held, devices=frozenset({"ee" * 32})),
                  fx.allowlist(1)):
        with pytest.raises(ValueError):
            offer(prev, adopts, device_list=wrong)
    # A menu that keeps the device list in force needs none.
    keeps = fx.state(devices=ms.AllowlistRef(1, held.sha256))
    assert offer(keeps, changed(keeps)) == changed(keeps)


def test_a_committed_menu_adopts_a_newer_lab_list_only_with_that_list():
    prev = fx.state()                                       # names lab list 2
    bare = changed(prev, labs=ms.AllowlistRef(3, "ab" * 32))
    for held in (None, fx.ALLOW, fx.allowlist(3)):
        with pytest.raises(ms.MenuUnavailable):
            offer(prev, bare, allowlist=held)


def test_the_step_adopts_a_device_list_only_as_the_signed_list_itself():
    s = fx.state(devices=ms.AllowlistRef(4, fx.device_list(4).sha256))
    newer = fx.device_list(5)
    out = ms.step(s, fx.inputs(device_list=newer), fx.PARAMS)
    assert out.devices == ms.AllowlistRef(5, newer.sha256)
    assert ms.step(s, fx.inputs(device_list=fx.device_list(4)), fx.PARAMS) == ms.carry(s)
    for wrong, why in (
            (ms.AllowlistRef(5, "ff" * 32), "signed device list"),          # a reference
            (fx.device_list(5, signer=None), "verified"),
            (dataclasses.replace(newer, sha256="ff" * 32), "hash"),
            (dataclasses.replace(newer, devices=frozenset()), "hash"),
            (fx.allowlist(5), "signed device list"),                        # the other kind
            (fx.device_list(3), "in force"),                                 # an older one
            (fx.device_list(4, ("ee" * 32,)), "in force")):                  # another at the serial held
        with pytest.raises(ValueError, match=why):
            ms.step(s, fx.inputs(device_list=wrong), fx.PARAMS)


def test_a_signed_device_list_reaches_the_menu_through_verify_signed_list(tmp_path):
    signer = Signer(tmp_path, "release")
    signers = signer.signers_line().encode()
    payload = device_bytes(serial=1)
    held = mr.verify_signed_list(payload, signer.sign(payload), kind=mr.KIND_DEVICE_LIST,
                                 held_serial=0, signers=signers)
    prev = fx.state()
    nxt = ms.step(prev, fx.inputs(device_list=held), fx.PARAMS)
    assert nxt.devices == ms.AllowlistRef(1, held.sha256)
    assert offer(prev, nxt, device_list=held) == nxt
    # A list another key signed, or another namespace, never becomes a held list.
    for bad in (Signer(tmp_path, "stranger").sign(payload), signer.sign(payload, namespace="git")):
        with pytest.raises(mr.ListRefused):
            mr.verify_signed_list(payload, bad, kind=mr.KIND_DEVICE_LIST, held_serial=0,
                                  signers=signers)


# --- 3. a committed menu follows the held one by one step ------------------------------------
def test_a_cooldown_that_is_still_running_is_not_shortened_or_dropped():
    # Scored epoch 100: a cooldown until 102 or later can still refuse a filing.
    running = ms.Retired("lab-z/old@low", 90, ms.REASON_FAILING, 140)
    spent = ms.Retired("lab-y/spent@low", 40, ms.REASON_FAILING, 101)
    prev = fx.state(retired=(spent, running))
    for retired in ((spent,), (spent, dataclasses.replace(running, cooldown_until=120)), ()):
        assert "cooldown" in refused(prev, changed(prev, retired=retired))
    # The record is the step's: one that ran out stays while there is room
    # (tests/test_menu_root_limits.py), and none is lengthened.
    assert offer(prev, changed(prev, retired=(spent, running))) == changed(
        prev, retired=(spent, running))
    for retired in ((running,), (dataclasses.replace(running, epoch=NEW, cooldown_until=NEW + 48),)):
        assert "record" in refused(prev, changed(prev, retired=retired))
    ck = (ms.ColdkeyCooldown("5Fck-gone", 101), ms.ColdkeyCooldown("5Fck-held", 130))
    prev = fx.state(coldkey_cooldowns=ck)
    for kept in ((ck[0],), (ck[0], ms.ColdkeyCooldown("5Fck-held", 129)), ()):
        assert "cooldown" in refused(prev, changed(prev, coldkey_cooldowns=kept))
    # A coldkey's cooldown that ran out leaves in that step, and none is lengthened.
    assert offer(prev, changed(prev, coldkey_cooldowns=(ck[1],))) == changed(
        prev, coldkey_cooldowns=(ck[1],))
    for kept in (ck, (ms.ColdkeyCooldown("5Fck-held", 150),)):
        assert "cooldown" in refused(prev, changed(prev, coldkey_cooldowns=kept))


def test_an_entrys_joined_epoch_and_scored_from_do_not_move():
    fresh = fx.entry("lab-d/fresh", "low", joined_epoch=95, scored_from=119)
    prev = fx.state(*fx.state().entries, fresh)
    for moved in (dataclasses.replace(fresh, scored_from=95),          # grace cut short
                  dataclasses.replace(fresh, scored_from=130),         # or lengthened
                  dataclasses.replace(fresh, joined_epoch=96)):
        entries = tuple(moved if e.id == fresh.id else e for e in prev.entries)
        why = refused(prev, changed(prev, entries=entries))
        assert "scored_from" in why or "joined_epoch" in why
def test_exploration_pair_counts_are_checked_only_where_every_node_knows_them():
    """A node that is not the root cannot recompute another node's count
    from its own proofs, so a count on an entry already on the menu is never
    a reason to refuse (an entry joins with no exploration pairs: the join
    delta, tests/test_v4_chain_proposals.py)."""
    prev = fx.state()
    recounted = tuple(dataclasses.replace(e, explore_pairs=10_000) for e in prev.entries)
    assert offer(prev, changed(prev, entries=recounted)) == changed(prev, entries=recounted)


# --- an honest root ------------------------------------------------------------------------


def _walk(seed: int, params: ms.MenuParams = fx.PARAMS):
    """Twenty steps of an honest root from fx.state(), with joins at the
    boundaries: provisional entries joined, decided by their gates (some
    pass, some fail), a newer list, prices, device lists; each step's menu,
    as the node that follows it reads it, is the menu the step derived."""
    rnd = random.Random(seed)
    s = fx.state()
    newer = fx.allowlist(3, labs=fx.LABS + ("lab-x",))
    for n in range(20):
        k = s.epoch - 1
        lab_list = newer if n >= 6 else fx.ALLOW
        gates = tuple(fx.gate(p.id, correct=240 if rnd.random() < 0.6 else 60,
                              coldkeys=(("ck-1", 60, 0), ("ck-2", 60, 0),
                                        ("ck-3", 60, 0 if rnd.random() < 0.8 else 30)))
                      for p in s.provisional)
        reads = (fx.read("lab-a/model-one", "host-a/fp8", price_in=100_000 + n),
                 fx.read("lab-a/model-one", "host-b/fp8"))
        dev = fx.device_list(1 + n // 5) if n % 5 == 0 else None
        nxt = ms.step(s, fx.inputs(epoch=k, lab_list=lab_list, gates=gates, host_reads=reads,
                                   device_list=dev, explore_counts=tuple(
                                       ms.ExploreCount(e.id, e.rev, 7 * n + len(e.id))
                                       for e in s.entries),
                                   # Graded cells for every entry but the
                                   # last, so the pick-cost tables refresh too.
                                   pick_stats=tuple(
                                       ms.PickStat(e.id, e.rev, (
                                           ms.PickCell("code", 3 + n, (3 + n) * 300,
                                                       (3 + n) * (900 + 40 * i)),
                                           ms.PickCell("math", 1, 250, 4_000)))
                                       for i, e in enumerate(s.entries[:-1]))), params)
        if nxt != ms.carry(s):
            got = offer(s, nxt, allowlist=lab_list, params=params,
                        device_list=dev if nxt.devices != s.devices else None)
            assert got == nxt, n
        yield s, nxt
        s = nxt
        if n in (0, 1, 3, 7, 9):
            # The boundary's joins on the step's menu (a delta's entries).
            s = fx.provisional(s, f"lab-c/model-{n}", "high", proposer=f"5Fhk{n}",
                               coldkey=f"5Fck{n % 3}")


@pytest.mark.parametrize("seed, params", [(1, fx.PARAMS), (2, fx.PARAMS), (3, fx.PARAMS)])
def test_every_menu_an_honest_step_derives_passes_every_check(seed, params):
    kinds = set()
    for prev, nxt in _walk(seed, params):
        if any(e.provisional and not nxt.entry(e.id).provisional
               for e in prev.entries if e.id in nxt.ids()):
            kinds.add("full")
        if any(r.reason in ms._GATE_REASONS for r in nxt.retired):
            kinds.add("gate")
        if nxt.devices != prev.devices:
            kinds.add("devices")
        if nxt.labs != prev.labs:
            kinds.add("labs")
        if nxt.coldkey_cooldowns:
            kinds.add("lapse")
    assert {"full", "gate", "devices", "labs"} <= kinds, kinds


# --- a proposal's measurement: its window is the step's, and it ends when it ends ------
# --- any bytes: the menu, MenuRefused or MenuUnavailable, and nothing else --------

def _fuzz_prev() -> ms.MenuState:
    return fx.provisional(fx.state(epoch=101,
                                   retired=(ms.Retired("lab-z/x@high", 90, "gate_category", 200),),
                                   coldkey_cooldowns=(ms.ColdkeyCooldown("ck", 300),)))


_ODD_VALUES = (None, True, False, 0, -1, 2**60, "", "x", [], [1], [{}], {}, {"a": 1}, [[]],
               "lab-a/model-one", 10**400)


def _paths(d, p=()):
    yield p
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _paths(v, p + (k,))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _paths(v, p + (i,))


def one_value_mutations(base: dict, n: int, seed: int):
    """`n` documents (fewer when one does not encode), each `base` with one
    value replaced by another JSON value, as canonical bytes."""
    rnd = random.Random(seed)
    every = [p for p in _paths(base) if p]
    for _ in range(n):
        doc = json.loads(json.dumps(base))
        p = rnd.choice(every)
        cur = doc
        for k in p[:-1]:
            cur = cur[k]
        cur[p[-1]] = rnd.choice(_ODD_VALUES)
        try:
            yield json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        except ValueError:
            continue


def _next_doc(prev: ms.MenuState) -> dict:
    return json.loads(ms.canonical_json(dataclasses.replace(ms.carry(prev),
                                                            serial=prev.serial + 1)))


def test_bytes_with_one_value_changed_are_the_menu_or_refused_never_an_error():
    """Committed bytes are the next menu or are refused (carried on every
    node); no other exception leaves the reader, whatever value sits where."""
    prev = _fuzz_prev()
    refusals = 0
    for raw in one_value_mutations(_next_doc(prev), 1500, seed=11):
        try:
            ms.menu_for_epoch(prev, hashlib.sha256(raw).hexdigest(), raw, allowlist=fx.ALLOW,
                              params=fx.PARAMS)
        except ms.MenuRefused as e:
            assert e.carried == ms.carry(prev)
            refusals += 1
    assert refusals > 1000


@pytest.mark.parametrize("where, value", [
    (("entries", 0, "provider", "order"), [{}]),
    (("entries", 0, "provider", "order"), [[1]]),
    (("entries", 0, "provider", "order"), [7]),
    (("entries", 0, "reasoning", "effort"), ["high"]),
    (("retired", "lab-z/x@high", "reason"), []),
    (("retired", "lab-z/x@high", "reason"), {}),
    (("provisional", 0, "coldkey"), {"a": 1}),
    (("allowlists", "labs", "sha256"), ["ab"]),
])
def test_a_value_of_another_type_anywhere_is_refused(where, value):
    prev = _fuzz_prev()
    doc = _next_doc(prev)
    cur = doc
    for k in where[:-1]:
        cur = cur[k]
    cur[where[-1]] = value
    raw = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    with pytest.raises(ms.MenuRefused):
        ms.menu_for_epoch(prev, hashlib.sha256(raw).hexdigest(), raw, allowlist=fx.ALLOW,
                          params=fx.PARAMS)
    # and from_json names such a document a ValueError, never another exception
    with pytest.raises(ValueError):
        ms.from_json(raw)
