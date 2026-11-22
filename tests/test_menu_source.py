"""A v4 reveal's `menu` field names a menu anyone can fetch: besides the
menu's epoch and hash it names `source`, the last published menu the epoch's
menu was carried from (its epoch and hash). The source's bytes are in the
root's store under that hash, and the epoch's menu is those bytes with its
`epoch` set, so a reader with no history rebuilds and checks it from one
reveal and the bucket (fugal_subnet/menu_publish.menu_field, menu_from_field,
MenuBook.source_of; docs/decisions/LIVE_MENU_V4.md 12)."""
from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from fugal_subnet import menu_publish as mp
from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from fugal_subnet import reveal_sync, v2_sync
from fugal_subnet.benchmarks.slicer import epoch_id_for_block
from tests import menu_fixtures as fx
from tests.signed_list_fixtures import Signer
from tests.test_menu_publish import BPE, BUCKET, LIST_BYTES, NETUID, World, changing, quiet


@pytest.fixture(autouse=True)
def _small_menus(monkeypatch):
    fx.allow_small_menus(monkeypatch)


@pytest.fixture
def signer(tmp_path, monkeypatch):
    s = Signer(tmp_path, "release")
    monkeypatch.setattr(mr, "list_signers", lambda: s.signers_line().encode())
    return s


def _run(w, f, epochs=range(100, 110), change_at=(100, 102)):
    """The root and a follower over `epochs`; the root's step changes the
    menu (an entry retires) for scored epochs in `change_at`. Returns
    {k: (root's menu field, follower's menu field)}."""
    fields = {}
    for k in epochs:
        w.chain.block = k * BPE
        inputs = changing(k) if k in change_at else quiet(k)
        _, ff = w.follower_epoch(f, k, inputs)
        _, rf = w.root_epoch(k, inputs)
        fields[k] = (rf["menu"], ff["menu"])
    return fields


def _expected_sources(book, first, last):
    """The last published menu each epoch's menu was carried from, derived
    on its own from the menus held: the first epoch of the run of epochs
    whose menus have the same serial (a changed menu always takes the next
    serial; a carried one keeps it)."""
    out, start = {}, first
    for k in range(first, last + 1):
        if k > first and book.held(k).serial != book.held(k - 1).serial:
            start = k
        out[k] = (start, ms.menu_hash(dataclasses.replace(book.held(k), epoch=start)))
    return out


def _store_reader(w):
    return lambda h: reveal_sync.fetch_menu_bytes(h, bucket=BUCKET, netuid=NETUID,
                                                  opener=w.store.opener)


def _root_reveal(w, k) -> tuple[bytes, dict]:
    raw = (w.epoch_dir / epoch_id_for_block(k) / "reveal.json").read_bytes()
    return raw, json.loads(raw)


def test_every_reveal_names_its_menus_source_and_root_and_follower_agree_byte_for_byte(
        tmp_path, signer):
    w = World(tmp_path, signer)
    f = w.node("follower")
    fields = _run(w, f)
    want = _expected_sources(w.root.book, 99, 109)
    # The world changes the menu at 102 and 104 and carries it otherwise
    # (from 105 on with commitments that name the carried menu itself).
    assert sorted({src for src, _h in want.values()}) == [99, 102, 104]
    for k, (root, follower) in fields.items():
        assert json.dumps(root, sort_keys=True) == json.dumps(follower, sort_keys=True), k
        assert sorted(root) == ["epoch", "hash", "source"]
        assert root["hash"] == ms.menu_hash(w.root.book.held(k))
        assert root["source"] == {"epoch": epoch_id_for_block(want[k][0]), "hash": want[k][1]}, k
    # The field joins the view validators are compared on, source and all.
    for k in fields:
        assert v2_sync.consensus_view({"menu": fields[k][0]}) == \
            v2_sync.consensus_view({"menu": fields[k][1]})


def test_a_carried_epochs_reveal_and_the_bucket_rebuild_and_verify_its_menu(tmp_path, signer):
    w = World(tmp_path, signer)
    f = w.node("follower")
    _run(w, f)
    raw, reveal = _root_reveal(w, 107)
    field = reveal["menu"]
    assert field["source"]["epoch"] == "e00000104"
    # The store keeps only what the root published before committing (the
    # committed menus), and not even the epoch's own bytes (a root commits
    # the carried menu's hash when its step changes nothing, here from 105):
    # the reveal's source alone is enough.
    committed = {reveal_sync.menu_object(NETUID, w.chain.read(k)) for k in range(100, 110)
                 if w.chain.read(k) is not None}
    committed.discard(reveal_sync.menu_object(NETUID, field["hash"]))
    assert reveal_sync.menu_object(NETUID, field["source"]["hash"]) in committed
    for key in [k for k in w.store.objects if "/menus/" in k and k not in committed]:
        del w.store.objects[key]
    assert reveal_sync.fetch_menu_bytes(field["hash"], bucket=BUCKET, netuid=NETUID,
                                        opener=w.store.opener) is None
    state = mp.menu_from_field(field, "e00000107", _store_reader(w))
    assert state == w.root.book.held(107) == f.book.held(107)
    assert ms.menu_hash(state) == field["hash"]
    # A committed epoch's menu is its own source.
    _, reveal102 = _root_reveal(w, 102)
    assert reveal102["menu"]["source"] == {"epoch": "e00000102", "hash": reveal102["menu"]["hash"]}
    assert mp.menu_from_field(reveal102["menu"], "e00000102", _store_reader(w)) == \
        w.root.book.held(102)


@pytest.mark.parametrize("tamper, why", [
    (lambda fld, h102: {**fld, "source": {**fld["source"], "hash": "ab" * 32}}, "no bytes"),
    (lambda fld, h102: {**fld, "source": {**fld["source"], "hash": h102}}, "not of 104"),
    (lambda fld, h102: {**fld, "source": {**fld["source"], "epoch": "e00000105"}}, "not of 105"),
    (lambda fld, h102: {**fld, "source": {"epoch": "e00000102", "hash": h102}}, "rebuilt"),
    (lambda fld, h102: {**fld, "source": {**fld["source"], "epoch": "e00000108"}}, "later epoch"),
    (lambda fld, h102: {**fld, "source": {**fld["source"], "epoch": "e104"}}, "not an epoch id"),
    (lambda fld, h102: {**fld, "hash": "cd" * 32}, "rebuilt"),
    (lambda fld, h102: {k: v for k, v in fld.items() if k != "source"}, "an object of"),
    (lambda fld, h102: {**fld, "extra": 1}, "an object of"),
    (lambda fld, h102: {**fld, "source": {**fld["source"], "extra": 1}}, "an object of"),
    (lambda fld, h102: {**fld, "epoch": "e00000106"}, "not for the reveal's epoch"),
    (lambda fld, h102: {**fld, "source": {**fld["source"], "hash": "AB" * 32}}, "hex"),
])
def test_a_wrong_source_is_refused(tmp_path, signer, tamper, why):
    w = World(tmp_path, signer)
    f = w.node("follower")
    _run(w, f)
    _, reveal = _root_reveal(w, 107)
    h102 = ms.menu_hash(w.root.book.held(102))
    assert reveal_sync.menu_object(NETUID, h102) in w.store.objects    # published, another menu
    bad = tamper(reveal["menu"], h102)
    with pytest.raises(ValueError, match=why):
        mp.menu_from_field(bad, "e00000107", _store_reader(w))


def test_a_root_reveal_naming_a_wrong_source_differs_from_every_follower_and_is_not_anchored_on(
        tmp_path, signer):
    w = World(tmp_path, signer)
    f = w.node("follower")
    fields = _run(w, f)
    raw, reveal = _root_reveal(w, 107)
    wrong = dict(reveal, menu={**reveal["menu"], "source": {"epoch": "e00000105",
                                                            "hash": "cd" * 32}})
    own = {"menu": fields[107][1]}
    # The follower's check of the root's reveal (v2_sync.check, the view
    # validators are compared on) sees it.
    assert v2_sync.consensus_view(wrong) != v2_sync.consensus_view(own)
    assert v2_sync.consensus_view(reveal) == v2_sync.consensus_view(own)
    # A TD anchors on a reveal only when its menu field is well formed.
    sha = hashlib.sha256(raw).hexdigest()
    got = mp.anchor_from_reveal(raw, "e00000107", sha)
    assert got == (107, reveal["menu"]["hash"], 104, reveal["menu"]["source"]["hash"])
    for field in ({**reveal["menu"], "source": {"epoch": "e00000108", "hash": "cd" * 32}},
                  {"epoch": "e00000107", "hash": reveal["menu"]["hash"]}):
        bad = json.dumps(dict(reveal, menu=field)).encode()
        assert mp.anchor_from_reveal(bad, "e00000107", hashlib.sha256(bad).hexdigest()) is None


def test_a_node_that_rebuilt_or_restarted_names_the_same_sources(tmp_path, signer):
    w = World(tmp_path, signer)
    f = w.node("follower")
    _run(w, f, epochs=range(100, 110), change_at=(100, 103))
    late = w.node("late")                       # holds the base, walks the chain
    w.chain.block = 110 * BPE
    assert late.at_boundary(110).ok
    want = _expected_sources(w.root.book, 99, 109)
    for k in range(100, 110):
        assert late.book.source_of(k) == w.root.book.source_of(k) == f.book.source_of(k) \
            == want[k], k
    assert any(src < k for k, (src, _h) in want.items())    # carried epochs among them
    again = mp.MenuBook(str(tmp_path / "follower"))
    assert [again.source_of(k) for k in range(100, 110)] == \
        [f.book.source_of(k) for k in range(100, 110)]
    assert again.source_of(200) is None


def test_the_root_puts_a_source_that_no_commitment_published(tmp_path, signer):
    """The first menu (the pinned seed at the first v4 epoch, here the base)
    was never committed: the root puts its bytes with the reveal that names
    it, so it can be fetched like a committed one."""
    w = World(tmp_path, signer)
    f = w.node("follower")
    _run(w, f, epochs=range(100, 102))
    key = reveal_sync.menu_object(NETUID, ms.menu_hash(w.base))
    assert w.store.objects[key] == ms.canonical_json(w.base)
    _, reveal100 = _root_reveal(w, 100)
    assert mp.menu_from_field(reveal100["menu"], "e00000100", _store_reader(w)) == \
        ms.carry(w.base)


def test_a_td_anchors_on_a_carried_menu_through_its_source(tmp_path, signer, monkeypatch):
    from fugal_subnet.tee import td_v4
    from tests.test_harness_v4 import Provider, _plan
    from tests.test_tee_e2e import _pool

    w = World(tmp_path, signer)
    f = w.node("follower")
    _run(w, f)
    raw, reveal = _root_reveal(w, 107)
    source_key = reveal_sync.menu_object(NETUID, reveal["menu"]["source"]["hash"])
    only_source = {k: v for k, v in w.store.objects.items() if "/menus/" not in k or k == source_key}
    w.store.objects = only_source
    sha = hashlib.sha256(raw).hexdigest()
    pool = _pool()
    session = td_v4.V4Session(
        first_epoch=90, plan=_plan(pool), fetch_listing=Provider().fetch,
        read_commitment=w.chain.read, menu_bytes=_store_reader(w),
        list_bytes=lambda serial, h: (LIST_BYTES, signer.sign(LIST_BYTES)),
        read_anchor=lambda e: mp.anchor_from_reveal(raw, "e00000107", sha))
    assert session._reveal_anchor(109) == w.root.book.held(107)
    # A source that does not rebuild to the hash is not used.
    bad = dict(reveal, menu={**reveal["menu"], "source": {"epoch": "e00000102",
                                                          "hash": ms.menu_hash(
                                                              w.root.book.held(102))}})
    bad_raw = json.dumps(bad).encode()
    session.read_anchor = lambda e: mp.anchor_from_reveal(
        bad_raw, "e00000107", hashlib.sha256(bad_raw).hexdigest())
    with pytest.raises(ms.MenuUnavailable):
        session._reveal_anchor(109)


def test_a_menu_field_names_no_source_after_its_menu():
    state = dataclasses.replace(fx.state(), epoch=10)
    with pytest.raises(ValueError):
        mp.menu_field(state, (11, "ab" * 32))
    assert mp.menu_field(state, (8, "ab" * 32))["source"] == {"epoch": "e00000008", "hash": "ab" * 32}
