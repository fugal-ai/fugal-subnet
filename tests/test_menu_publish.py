"""Protocol v4: the root publishes the menu, every node reads it at the
boundary, the others audit it, and a node that missed epochs rebuilds it
(docs/decisions/LIVE_MENU_V4.md, 4.3 and 12; fugal_subnet/menu_publish.py).

Everything runs against a fake public store and a fake chain: no network, no
chain, no spend. The chain keeps one commitment per hotkey, as the
Commitments pallet does, and every read of it is pinned to a boundary block.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import types
import urllib.error

import pytest

from fugal_subnet import menu_publish as mp
from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from fugal_subnet import reveal_publisher as rp
from fugal_subnet import reveal_sync, v2_sync
from fugal_subnet.benchmarks.slicer import epoch_id_for_block
from fugal_subnet.tee import td_v4
from tests import menu_fixtures as fx
from tests.signed_list_fixtures import Signer

NETUID = 7
BUCKET = "fugal-test-reveals"
ROOT_HK = "5Froot-hotkey"
BPE = 10                                       # blocks an epoch
LIST_BYTES = mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, fx.ALLOW.serial, labs=fx.LABS,
                                     hosts=fx.HOSTS, first_party=fx.FIRST_PARTY)
CADENCE = dataclasses.replace(fx.PARAMS, refresh_epochs=ms.MENU_REFRESH_EPOCHS)


@pytest.fixture(autouse=True)
def _small_menus(monkeypatch):
    fx.allow_small_menus(monkeypatch)


@pytest.fixture
def signer(tmp_path, monkeypatch):
    s = Signer(tmp_path, "release")
    monkeypatch.setattr(mr, "list_signers", lambda: s.signers_line().encode())
    return s


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Store:
    """The root's public bucket: create-only puts, and reads exactly as any
    reader over HTTPS sees them (`opener` stands in for urlopen)."""

    def __init__(self, log, clock):
        self.objects: dict[str, bytes] = {}
        self.log, self.clock = log, clock
        self.fail_put = self.fail_get = self.serve_other = False
        self.slow = 0.0
        self.reads: list[str] = []

    def put(self, key, data):
        self.log.append(("put", key))
        self.clock.t += self.slow
        if self.fail_put:
            raise OSError("bucket unreachable")
        self.objects.setdefault(key, bytes(data))

    def get(self, key, limit):
        self.log.append(("get", key))
        self.clock.t += self.slow
        if self.fail_get:
            raise TimeoutError("read timed out")
        data = self.objects[key]
        return data + b" " if self.serve_other else data

    def opener(self, req, timeout=None, record=True):
        key = req.full_url.split("/", 4)[4]
        if record:
            self.reads.append(key)
        if key not in self.objects:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)
        return io.BytesIO(self.objects[key])


class Uploader:
    """reveal_publisher's uploader, into the same store."""

    def __init__(self, store):
        self.store = store

    def upload(self, path, key):
        with open(path, "rb") as f:
            self.store.objects[key] = f.read()
        self.store.log.append(("upload", key))
        return f"gs://{BUCKET}/{key}"


class Chain:
    """The root hotkey's commitments: the pallet keeps the newest one, and a
    read pinned at a block sees the one made at or before it."""

    def __init__(self, log, store):
        self.log, self.store = log, store
        self.block = 0
        self.commits: list[tuple[int, str]] = []

    def commit(self, data: str) -> bool:
        assert len(data.encode()) <= rp.COMMITMENT_MAX_BYTES
        got = reveal_sync.parse_commitment(data)
        assert got is not None
        if got[2] is not None:
            # Publish before commit: whatever hash a commitment names, a
            # reader can fetch its bytes from the public store right now.
            assert reveal_sync.fetch_menu_bytes(
                got[2], bucket=BUCKET, netuid=NETUID,
                opener=lambda req, timeout=None: self.store.opener(req, timeout, record=False)
            ) is not None
        self.log.append(("commit", data))
        self.commits.append((self.block, data))
        return True

    def at(self, block: int) -> str:
        older = [d for b, d in self.commits if b <= block]
        return older[-1] if older else ""

    def read(self, epoch: int):
        return mp.menu_hash_in_commitment(self.at(epoch * BPE), epoch)

    # The part of a subtensor that commitments.get_commitments_with_blocks reads.
    def query_map(self, module, name, params, block=None):
        data = self.at(block)
        if not data:
            return []
        raw = data.encode()
        return [(ROOT_HK, {"block": block, "info": {"fields": [{f"Raw{len(raw)}": "0x" + raw.hex()}]}})]


class World:
    """One root and any number of other validators on one store and chain."""

    def __init__(self, tmp_path, signer, *, base=None, step_params=fx.PARAMS,
                 reader_params=None):
        self.tmp, self.signer = tmp_path, signer
        self.log, self.clock = [], Clock()
        self.store = Store(self.log, self.clock)
        self.chain = Chain(self.log, self.store)
        self.base = base if base is not None else fx.state(epoch=99)
        self.step_params, self.reader_params = step_params, reader_params
        self.epoch_dir = tmp_path / "root-epochs"
        self.epoch_dir.mkdir()
        self.root = self.node("root")
        self.publisher = mp.MenuPublisher(netuid=NETUID, store=self.store, held=self.root.book.held,
                                          lists=self.root.lists, params=reader_params,
                                          clock=self.clock)

    def lists(self, fetch=True) -> mp.ListBook:
        book = mp.ListBook(None, shipped=False, fetch=(lambda kind, sha: reveal_sync.fetch_signed_list(
            kind, sha, bucket=BUCKET, netuid=NETUID, opener=self.store.opener)) if fetch else None)
        assert book.add(mr.KIND_MENU_ALLOWLIST, LIST_BYTES, self.signer.sign(LIST_BYTES))
        return book

    def node(self, name, *, hold_base=True) -> mp.MenuNode:
        book = mp.MenuBook(str(self.tmp / name))
        if hold_base:
            book.hold(self.base)
        return mp.MenuNode(
            first_epoch=90, book=book, lists=self.lists(), read_commitment=self.chain.read,
            fetch_menu=lambda h: reveal_sync.fetch_menu_bytes(h, bucket=BUCKET, netuid=NETUID,
                                                              opener=self.store.opener),
            step_params=self.step_params, reader_params=self.reader_params)

    def committed_menu_of(self, j):
        return mp.committed_menu(str(self.epoch_dir), j)

    def root_epoch(self, k, inputs):
        """What the validator loop does for the root in a v4 epoch."""
        self.chain.block = k * BPE
        read = self.root.at_boundary(k)
        self.chain.block = k * BPE + 5
        rp.publish_due_reveals(str(self.epoch_dir), k, 1, NETUID, Uploader(self.store),
                               commit=self.chain.commit, menu=self.publisher)
        fields = mp.epoch_reveal_fields(self.root, read, k=k, inputs=inputs, root=True,
                                        committed_menu_of=self.committed_menu_of)
        folder = self.epoch_dir / epoch_id_for_block(k)
        folder.mkdir()
        (folder / "reveal.json").write_text(json.dumps(
            {"epoch_id": epoch_id_for_block(k), "protocol_version": 4, **fields}, indent=2))
        return read, fields

    def follower_epoch(self, node, k, inputs):
        read = node.at_boundary(k)
        return read, mp.epoch_reveal_fields(node, read, k=k, inputs=inputs, root=False,
                                            committed_menu_of=lambda j: None)


TWO = "lab-b/model-two"


def changing(k):
    """Inputs of scored epoch k that change the menu: an entry retires (gone),
    another one in each epoch it is used in."""
    gone = TWO if k <= 100 else "lab-a/model-one@low"
    return fx.inputs(epoch=k, health=(ms.HealthStat(gone, 1, unlisted_epochs=6),))


def quiet(k):
    return fx.inputs(epoch=k)


def menu_keys(log, kind):
    return [key for what, key in log if what == kind and "/menus/" in key]


# --- the commitment ---------------------------------------------------------------


def test_the_menu_commitment_fits_one_commitments_field_and_hex_would_not():
    eid, sha, h = "e00026489", "ab" * 32, "cd" * 32
    plain, menu = rp.commitment_for(eid, sha), rp.commitment_for(eid, sha, h)
    assert plain == f"reveal:{eid}:{sha}"                    # protocol v1-v3, unchanged
    assert menu.startswith(plain + ":") and len(menu.encode()) == 125
    assert len(f"{plain}:{h}".encode()) == 146 > rp.COMMITMENT_MAX_BYTES
    assert reveal_sync.parse_commitment(menu) == (eid, sha, h)
    assert reveal_sync.parse_pointer(menu) == (eid, sha)     # a follower reads either form
    assert reveal_sync.parse_commitment(plain) == (eid, sha, None)


def test_a_menu_hash_has_one_token_and_nothing_else_reads_as_one():
    h = hashlib.sha256(b"menu").hexdigest()
    token = rp.encode_menu_hash(h)
    assert len(token) == 43 and rp.decode_menu_hash(token) == h
    other_tail = token[:-1] + ("B" if token[-1] != "B" else "C")   # nonzero padding bits
    for bad in (other_tail, token + "=", token[:-1], token + "A", "", None, 7, h):
        assert rp.decode_menu_hash(bad) is None
    sha = "ab" * 32
    assert reveal_sync.parse_commitment(f"reveal:e00000100:{sha}:{other_tail}") is None
    assert reveal_sync.parse_pointer(f"reveal:e00000100:{sha}:{other_tail}") is None


def test_the_commitment_is_read_for_its_own_epoch_only():
    h, sha = "cd" * 32, "ab" * 32
    data = rp.commitment_for(epoch_id_for_block(100), sha, h)
    assert mp.menu_hash_in_commitment(data, 102) == h
    for epoch in (101, 103):                                  # another epoch's: no commitment
        assert mp.menu_hash_in_commitment(data, epoch) is None
    assert mp.menu_hash_in_commitment(rp.commitment_for(epoch_id_for_block(100), sha), 102) is None
    assert mp.menu_hash_in_commitment("garbage", 102) is None
    assert mp.menu_hash_in_commitment(data, 1) is None


def test_the_chain_read_is_pinned_to_the_boundary_block(tmp_path, signer):
    w = World(tmp_path, signer)
    h, sha = "cd" * 32, "ab" * 32
    w.chain.block = 1015
    w.chain.commits.append((1015, rp.commitment_for(epoch_id_for_block(100), sha, h)))
    w.chain.commits.append((1025, rp.commitment_for(epoch_id_for_block(101), sha, h)))
    # At the boundary of 102 the commitment for 100 holds, whatever came after.
    assert mp.read_menu_commitment(w.chain, NETUID, ROOT_HK, 102, 1020) == h
    assert mp.read_menu_commitment(w.chain, NETUID, "5Fsomeone-else", 102, 1020) is None
    assert mp.read_menu_commitment(w.chain, NETUID, ROOT_HK, 101, 1010) is None


@pytest.mark.parametrize("hold,delay,root,hotkey,refused", [
    (1, 1, True, ROOT_HK, 0), (1, 1, False, ROOT_HK, 0), (2, 1, False, ROOT_HK, 1),
    (1, 2, True, ROOT_HK, 1), (1, 2, False, ROOT_HK, 0), (1, 1, False, None, 1),
    (3, 3, True, None, 3)])
def test_a_validator_that_cannot_read_or_publish_the_menu_in_time_does_not_start(
        hold, delay, root, hotkey, refused):
    got = mp.startup_refusals(chain_hold=hold, reveal_delay=delay, root=root, root_hotkey=hotkey,
                              epoch_id=epoch_id_for_block(26489))
    assert len(got) == refused, got


# --- the root publishes before it commits -------------------------------------------


def test_the_root_commits_a_menu_hash_only_after_the_store_serves_the_bytes_back(tmp_path, signer):
    w = World(tmp_path, signer)
    w.root_epoch(100, changing(100))
    _, fields = w.root_epoch(101, quiet(101))
    commit = [d for what, d in w.log if what == "commit"]
    assert len(commit) == 1
    eid, _sha, h = reveal_sync.parse_commitment(commit[0])
    assert eid == epoch_id_for_block(100)
    reveal100 = json.loads((w.epoch_dir / eid / "reveal.json").read_text())
    assert h == reveal100[mp.REVEAL_NEXT_HASH]
    assert h == hashlib.sha256(reveal100[mp.REVEAL_NEXT].encode()).hexdigest()
    # The order: reveal, then the menu put and read back, then the commitment.
    # The menu the epoch ran on is put as well (for a TD that anchors on the
    # reveal), and so is the last published menu it was carried from (the
    # reveal's menu source; here the base menu, which no commitment named),
    # and nothing waits on either.
    order = [what for what, key in w.log if what in ("upload", "commit") or "/menus/" in key]
    assert order == ["upload", "put", "get", "put", "put", "commit"]
    ran_on = ms.menu_hash(ms.carry(w.base))
    assert ms.from_json(reveal_sync.fetch_menu_bytes(
        ran_on, bucket=BUCKET, netuid=NETUID, opener=w.store.opener)) == ms.carry(w.base)
    assert rp.read_marker(str(w.epoch_dir / eid))["menu_hash"] == h
    # Any reader gets exactly those bytes from the store by the hash alone.
    raw = reveal_sync.fetch_menu_bytes(h, bucket=BUCKET, netuid=NETUID, opener=w.store.opener)
    assert ms.from_json(raw).epoch == 102 and TWO not in ms.from_json(raw).ids()
    # The step's events and the menu the epoch ran on are in the reveal.
    assert ["retired", TWO] == reveal100[mp.REVEAL_EVENTS][0][:2]
    assert reveal100["menu"] == {"epoch": "e00000100", "hash": ms.menu_hash(ms.carry(w.base)),
                                 "source": {"epoch": "e00000099", "hash": ms.menu_hash(w.base)}}
    assert fields[mp.REVEAL_NEXT_HASH] != h                  # reveal 101 names the next one


@pytest.mark.parametrize("failure", ["put fails", "read fails", "store serves other bytes",
                                     "store hangs", "list cannot be published"])
def test_a_store_that_fails_costs_the_menu_change_and_nothing_else(tmp_path, signer, failure,
                                                                    caplog):
    w = World(tmp_path, signer)
    if failure == "list cannot be published":
        # The menu adopts a newer list this root does not hold the bytes of.
        newer = fx.allowlist(3)
        w.root_epoch(100, fx.inputs(epoch=100, lab_list=newer))
    else:
        w.root_epoch(100, changing(100))
    w.store.fail_put = failure == "put fails"
    w.store.fail_get = failure == "read fails"
    w.store.serve_other = failure == "store serves other bytes"
    w.store.slow = 200.0 if failure == "store hangs" else 0.0
    with caplog.at_level(logging.WARNING):
        w.root_epoch(101, quiet(101))                       # never raises
    commit = [d for what, d in w.log if what == "commit"]
    digest = rp.reveal_digest((w.epoch_dir / "e00000100" / "reveal.json").read_bytes())
    assert commit == [rp.commitment_for("e00000100", digest)]   # the reveal is still committed
    assert rp.read_marker(str(w.epoch_dir / "e00000100"))["menu_hash"] is None
    # Every node carries the menu at 102; the root's own next step starts from it.
    w.store.fail_put = w.store.fail_get = w.store.serve_other = False
    f = w.node("follower")
    for k in (100, 101):
        w.follower_epoch(f, k, quiet(k))
    w.chain.block = 1020
    read = f.at_boundary(102)
    assert read.status == "carried" and read.state == ms.carry(f.book.held(101))
    assert w.root.at_boundary(102).state == read.state


def test_a_newer_signed_list_is_published_with_the_menu_and_followers_take_it_from_the_store(
        tmp_path, signer, monkeypatch):
    second = Signer(tmp_path, "second-release")
    monkeypatch.setattr(mr, "list_signers",
                        lambda: (signer.signers_line() + second.signers_line()).encode())
    w = World(tmp_path, signer)
    newer = mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, 3, labs=fx.LABS, hosts=fx.HOSTS,
                                    first_party=fx.FIRST_PARTY)
    assert w.root.lists.add(mr.KIND_MENU_ALLOWLIST, newer, signer.sign(newer)).serial == 3
    f = w.node("follower")                                 # holds list 2 only
    for k in range(100, 103):
        w.chain.block = k * BPE
        read, _ = w.follower_epoch(f, k, quiet(k))
        w.root_epoch(k, fx.inputs(epoch=k, lab_list=w.root.lists.newest(mr.KIND_MENU_ALLOWLIST)))
    sha3 = hashlib.sha256(newer).hexdigest()
    puts = [key for what, key in w.log if what == "put"]
    assert reveal_sync.list_object(NETUID, mr.KIND_MENU_ALLOWLIST, sha3) in puts
    assert reveal_sync.signature_object(NETUID, mr.KIND_MENU_ALLOWLIST, sha3) in puts
    assert read.status == "committed" and read.state.labs == ms.AllowlistRef(3, sha3)
    assert f.lists.raw(mr.KIND_MENU_ALLOWLIST, sha3) == (
        newer, w.store.objects[reveal_sync.signature_object(NETUID, mr.KIND_MENU_ALLOWLIST, sha3)])
    reveal100 = json.loads((w.epoch_dir / "e00000100" / "reveal.json").read_text())
    assert [(x["serial"], x["payload"].encode()) for x in reveal100[mp.REVEAL_LISTS]] == [(3, newer)]
    # A store holding another valid signature for the same list serves that
    # one; it verifies, so the menu is still committed.
    (tmp_path / "again").mkdir()
    w2 = World(tmp_path / "again", signer)
    w2.root.lists.add(mr.KIND_MENU_ALLOWLIST, newer, signer.sign(newer))
    other_signature = second.sign(newer)
    assert other_signature != signer.sign(newer)
    w2.store.objects[reveal_sync.signature_object(NETUID, mr.KIND_MENU_ALLOWLIST, sha3)] = (
        other_signature)
    for k in (100, 101):
        w2.root_epoch(k, fx.inputs(epoch=k, lab_list=w2.root.lists.newest(mr.KIND_MENU_ALLOWLIST)))
    assert w2.chain.read(102) is not None
    # One that does not verify is not a confirmed publication.
    (tmp_path / "bad").mkdir()
    w3 = World(tmp_path / "bad", signer)
    w3.root.lists.add(mr.KIND_MENU_ALLOWLIST, newer, signer.sign(newer))
    w3.store.objects[reveal_sync.signature_object(NETUID, mr.KIND_MENU_ALLOWLIST, sha3)] = (
        signer.sign(newer + b" "))
    for k in (100, 101):
        w3.root_epoch(k, fx.inputs(epoch=k, lab_list=w3.root.lists.newest(mr.KIND_MENU_ALLOWLIST)))
    assert w3.chain.read(102) is None


def test_the_root_never_commits_a_menu_its_own_reader_refuses(tmp_path, signer):
    w = World(tmp_path, signer)
    w.chain.block = 1000
    w.root.at_boundary(100)
    w.chain.block = 1010
    held = w.root.at_boundary(101).state
    # A menu that skips a new entry's grace does not follow from the one held.
    bad = dataclasses.replace(ms.carry(held), serial=held.serial + 1, entries=held.entries + (
        fx.entry("lab-c/model-new", "high", joined_epoch=102, scored_from=102),))
    raw = ms.canonical_json(bad)
    reveal = {"epoch_id": "e00000100", mp.REVEAL_NEXT: raw.decode(),
              mp.REVEAL_NEXT_HASH: hashlib.sha256(raw).hexdigest()}
    assert w.publisher("e00000100", reveal) is None
    assert not menu_keys(w.log, "put")
    # And a menu document that is not the one the reveal names by hash.
    reveal[mp.REVEAL_NEXT_HASH] = "0" * 64
    assert w.publisher("e00000100", reveal) is None


# --- every node reads the root's menu at the boundary ---------------------------------


def test_every_node_runs_the_root_menu_from_the_epoch_it_was_committed_for(tmp_path, signer):
    w = World(tmp_path, signer)
    f = w.node("follower")
    reads = {}
    for k in range(100, 104):
        w.chain.block = k * BPE
        reads[k] = w.follower_epoch(f, k, changing(k) if k == 100 else quiet(k))
        w.root_epoch(k, changing(k) if k == 100 else quiet(k))
    assert [reads[k][0].status for k in range(100, 104)] == ["carried", "carried", "committed",
                                                               "committed"]
    for k in range(100, 104):
        assert reads[k][0].state == w.root.book.held(k)
    assert TWO not in reads[102][0].state.ids() and reads[102][0].state.serial == w.base.serial + 1
    # The TD's reader, given the same commitment and the store's bytes, runs
    # the same menu (td_v4.V4Session's menu_bytes calls reveal_sync).
    h = w.chain.read(102)
    td = td_v4.epoch_menu(102, previous=f.book.held(101), committed_hash=h,
                          published=reveal_sync.fetch_menu_bytes(h, bucket=BUCKET, netuid=NETUID,
                                                                 opener=w.store.opener),
                          allowlist=f.lists.get(mr.KIND_MENU_ALLOWLIST, f.book.held(101).labs))
    assert td == reads[102][0].state


def test_an_absent_root_carries_the_menu_on_every_node_and_fetches_nothing(tmp_path, signer):
    w = World(tmp_path, signer)
    f = w.node("follower")
    for k in range(100, 106):
        w.chain.block = k * BPE
        read, fields = w.follower_epoch(f, k, changing(k))
        assert read.status == "carried" and read.state == ms.carry(f.book.held(k - 1))
        assert fields["menu"]["hash"] == ms.menu_hash(read.state)
    assert f.book.held(105).serial == w.base.serial and w.store.reads == []
    # A late commitment (the root's reveal of 100 committed during 103) is
    # nobody's menu commitment: at 104 the chain is asked for 102's.
    w.chain.block = 1035
    w.chain.commit(rp.commitment_for("e00000100", "ab" * 32))
    assert w.chain.read(104) is None and w.chain.read(102) is None


def test_a_menu_the_reader_refuses_is_carried_by_every_node_with_an_alarm(tmp_path, signer,
                                                                           caplog):
    w = World(tmp_path, signer)
    f, g = w.node("follower"), w.node("other")
    for k in (100, 101):
        w.chain.block = k * BPE
        for n in (f, g):
            w.follower_epoch(n, k, quiet(k))
    # A root that commits, with its bytes published, a menu that moves an
    # entry's scored_from: the transition checks refuse it on every node.
    held = f.book.held(101)
    bad = dataclasses.replace(ms.carry(held), serial=held.serial + 1, entries=tuple(
        dataclasses.replace(e, scored_from=e.scored_from + 1) for e in held.entries))
    raw = ms.canonical_json(bad)
    h = hashlib.sha256(raw).hexdigest()
    w.store.objects[reveal_sync.menu_object(NETUID, h)] = raw
    w.chain.block = 1015
    w.chain.commit(rp.commitment_for("e00000100", "ab" * 32, h))
    w.chain.block = 1020
    with caplog.at_level(logging.ERROR):
        reads = [n.at_boundary(102) for n in (f, g)]
    assert [r.status for r in reads] == ["refused", "refused"]
    assert reads[0].state == reads[1].state == ms.carry(held)
    assert "ALARM" in caplog.text
    # And a commitment whose bytes nobody serves leaves a node with no menu.
    held102 = f.book.held(102)
    other = dataclasses.replace(ms.carry(held102), serial=held102.serial + 1,
                                entries=held102.entries[:2])
    w.chain.block = 1025
    w.chain.commits.append((1025, rp.commitment_for("e00000101", "ab" * 32, ms.menu_hash(other))))
    w.chain.block = 1030
    with caplog.at_level(logging.ERROR):
        none = f.at_boundary(103)
    assert none.status == "unavailable" and not none.ok
    assert "cannot verify any proof" in caplog.text


# --- the others audit the root --------------------------------------------------------


def test_a_follower_with_the_roots_proofs_finds_nothing_to_report(tmp_path, signer):
    w = World(tmp_path, signer)
    f = w.node("follower")
    audits = {}
    for k in range(100, 104):
        w.chain.block = k * BPE
        inputs = changing(k) if k == 100 else quiet(k)
        _, fields = w.follower_epoch(f, k, inputs)
        audits[k] = fields.get(mp.REVEAL_AUDIT)
        w.root_epoch(k, inputs)
    assert audits[102]["differences"] == [] and audits[103]["differences"] == []
    assert audits[102]["root"] == w.chain.read(102) and audits[102]["own"] == audits[102]["root"]
    assert audits[102]["scored_epoch"] == "e00000100"


def test_a_follower_with_another_proof_set_publishes_the_difference_and_follows_the_root(
        tmp_path, signer, caplog):
    w = World(tmp_path, signer)
    f = w.node("follower")
    for k in range(100, 103):
        w.chain.block = k * BPE
        # The follower did not see the retirement the root's proofs gave.
        with caplog.at_level(logging.ERROR):
            read, fields = w.follower_epoch(f, k, quiet(k))
        w.root_epoch(k, changing(k) if k == 100 else quiet(k))
    audit = fields[mp.REVEAL_AUDIT]
    assert "serial" in audit["differences"]
    assert any(d.startswith(f"entries/{TWO}") and d.endswith("only in this validator's menu")
               for d in audit["differences"])
    assert read.state == w.root.book.held(102) and TWO not in read.state.ids()   # the root's
    assert "ALARM" in caplog.text
    # The role fields stay out of the view validators are compared on.
    assert set(v2_sync.MENU_ROLE_FIELDS) <= set(v2_sync.PER_VALIDATOR_FIELDS)
    assert {mp.REVEAL_AUDIT, mp.REVEAL_NEXT, mp.REVEAL_EVENTS} <= set(v2_sync.PER_VALIDATOR_FIELDS)


def test_a_follower_that_restarted_says_it_could_not_audit(tmp_path, signer):
    w = World(tmp_path, signer)
    f = w.node("follower")
    for k in range(100, 102):
        w.chain.block = k * BPE
        w.follower_epoch(f, k, changing(k) if k == 100 else quiet(k))
        w.root_epoch(k, changing(k) if k == 100 else quiet(k))
    restarted = mp.MenuNode(first_epoch=90, book=mp.MenuBook(str(tmp_path / "follower")),
                            lists=w.lists(), read_commitment=w.chain.read,
                            fetch_menu=f.fetch_menu, step_params=fx.PARAMS)
    w.chain.block = 1020
    _, fields = w.follower_epoch(restarted, 102, quiet(102))
    assert fields[mp.REVEAL_AUDIT]["differences"] is None
    assert "did not keep" in fields[mp.REVEAL_AUDIT]["why"]
    assert fields["menu"]["hash"] == ms.menu_hash(w.root.at_boundary(102).state)


# --- a node that missed epochs ----------------------------------------------------------


def test_a_node_that_missed_epochs_walks_the_roots_commitments_to_the_same_menu(tmp_path, signer):
    w = World(tmp_path, signer)
    for k in range(100, 106):
        w.root_epoch(k, changing(k) if k in (100, 102) else quiet(k))
    late = w.node("late")                                 # holds the menu of 99 and nothing since
    w.chain.block = 1060
    read = late.at_boundary(106)
    assert read.ok and read.state == w.root.at_boundary(106).state
    for k in range(100, 106):
        assert late.book.held(k) == w.root.book.held(k)
    # A node with nothing at all starts from the pinned seed at the first v4
    # epoch, and its walk is bounded per call.
    seedling = w.node("seedling", hold_base=False)
    seedling.max_links = 3
    first = seedling.at_boundary(100)
    assert not first.ok and "goes on next epoch" in first.detail
    assert seedling.book.held(93) is not None and seedling.book.held(94) is None


def test_a_link_whose_bytes_cannot_be_had_stops_the_walk_there_until_they_appear(tmp_path, signer):
    w = World(tmp_path, signer)
    for k in range(100, 106):
        w.root_epoch(k, changing(k) if k in (100, 102) else quiet(k))
    h104 = w.chain.read(104)
    assert h104 != ms.menu_hash(ms.carry(w.root.book.held(103)))       # a changed menu
    key = reveal_sync.menu_object(NETUID, h104)
    kept = w.store.objects.pop(key)
    late = w.node("late")
    w.chain.block = 1060
    read = late.at_boundary(106)
    assert not read.ok and "e00000104" in read.detail
    assert late.book.held(103) == w.root.book.held(103) and late.book.held(104) is None
    w.store.objects[key] = kept                            # a mirror, or the store, has them again
    assert late.at_boundary(106).state == w.root.at_boundary(106).state


def _live_menus_across_a_root_key_change(change_at: int):
    """Twelve epochs of honest steps from fx.state(epoch=101), each changed
    menu committed by the root hotkey of its epoch: OLD before `change_at`,
    NEW from it. Returns (start, live menus by epoch, commitments by hotkey,
    bytes by hash)."""
    start = fx.state(epoch=101)
    menus, commits, raw = {101: start}, {"5Fold-root": {}, "5Fnew-root": {}}, {}
    s = start
    for k in range(100, 112):
        reads = (fx.read("lab-a/model-one", "host-a/fp8", price_in=100_000 + k),
                 fx.read("lab-a/model-one", "host-b/fp8", price_in=100_000 + k))
        nxt = ms.step(s, fx.inputs(epoch=k, host_reads=reads if k % 3 == 0 else ()), fx.PARAMS)
        b = ms.canonical_json(nxt)
        raw[ms.menu_hash(nxt)] = b
        commits["5Fold-root" if nxt.epoch < change_at else "5Fnew-root"][nxt.epoch] = ms.menu_hash(nxt)
        s = menus[nxt.epoch] = nxt
    return start, menus, commits, raw


def test_a_rebuild_reads_each_epochs_commitment_under_the_root_hotkey_in_force_then():
    """After the root's hotkey changes, a node that rebuilds history reads
    every past epoch under the hotkey that was the root's for it, from the
    compiled table, and reaches the menu the live nodes ran on."""
    table = ((0, "5Fold-root"), (108, "5Fnew-root"))
    assert [mp.hotkey_for_epoch(table, e) for e in (0, 107, 108, 10**6)] == [
        "5Fold-root", "5Fold-root", "5Fnew-root", "5Fnew-root"]
    assert mp.hotkey_for_epoch("5Fone", 5) == "5Fone"
    start, menus, commits, raw = _live_menus_across_a_root_key_change(108)
    lists = mp.ListBook(shipped=False)
    lists._held[(mr.KIND_MENU_ALLOWLIST, fx.ALLOW.sha256)] = (fx.ALLOW, b"", b"")

    def read(e):
        return commits[mp.hotkey_for_epoch(table, e)].get(e)

    got = mp.rebuild(start=start, target=113, read_commitment=read, menu_bytes=raw.get,
                     lists=lists, params=fx.PARAMS)
    assert got.stopped_at is None and got.state == menus[113]
    # Read under the newest hotkey alone, the walk would end on another menu.
    wrong = mp.rebuild(start=start, target=113, read_commitment=commits["5Fnew-root"].get,
                       menu_bytes=raw.get, lists=lists, params=fx.PARAMS)
    assert wrong.state != menus[113]


def test_a_rebuild_stops_and_says_so_at_an_epoch_no_root_hotkey_covers():
    start, menus, commits, raw = _live_menus_across_a_root_key_change(108)
    lists = mp.ListBook(shipped=False)
    lists._held[(mr.KIND_MENU_ALLOWLIST, fx.ALLOW.sha256)] = (fx.ALLOW, b"", b"")
    table = ((105, "5Fold-root"),)
    with pytest.raises(ValueError, match="no root hotkey"):
        mp.hotkey_for_epoch(table, 104)
    got = mp.rebuild(start=start, target=113,
                     read_commitment=lambda e: commits["5Fold-root"].get(e) if mp.hotkey_for_epoch(
                         table, e) else None,
                     menu_bytes=raw.get, lists=lists, params=fx.PARAMS)
    assert got.stopped_at == 102 and got.state == start and "no root hotkey" in got.reason


def test_the_root_hotkeys_are_a_compiled_table_on_finney():
    from fugal_subnet import config

    table = config.root_hotkeys_v4(True, env="")
    assert table == config.FINNEY_ROOT_HOTKEYS_V4 and table[-1][1] == config.FINNEY_ROOT_HOTKEY
    assert [e for e, _ in table] == sorted({e for e, _ in table}) and table[0][0] == 0
    assert config.root_hotkeys_v4(False, env="5Fsomeone") == ((0, "5Fsomeone"),)
    assert config.root_hotkeys_v4(False, env="") == ()
    with pytest.raises(ValueError):
        config.root_hotkeys_v4(True, env="5Fsomeone-else")


# --- the refresh cadence ---------------------------------------------------------------


def test_across_a_refresh_epoch_the_bytes_are_needed_only_where_the_menu_moved(tmp_path, signer):
    w = World(tmp_path, signer, base=fx.state(epoch=115), step_params=CADENCE)
    f = w.node("follower")

    def moving(k):
        return fx.inputs(epoch=k, host_reads=(
            fx.read("lab-a/model-one", "host-a/fp8", price_in=150_000 + k),
            fx.read("lab-a/model-one", "host-b/fp8", price_in=150_000 + k)))

    for k in range(116, 124):
        w.chain.block = k * BPE
        w.follower_epoch(f, k, moving(k))
        w.root_epoch(k, moving(k))
    held = {k: f.book.held(k) for k in range(116, 124)}
    assert [held[k] == w.root.book.held(k) for k in held] == [True] * len(held)
    prices = {k: held[k].entry("lab-a/model-one@high").price_in for k in held}
    assert prices[119] == 100_000 and prices[120] == 150_118 and prices[123] == 150_118
    # The root committed a hash every epoch; the follower fetched bytes for
    # one of them, the refresh epoch's.
    assert all(w.chain.read(k) is not None for k in range(118, 124))
    assert [r for r in w.store.reads if "/menus/" in r] == [
        reveal_sync.menu_object(NETUID, ms.menu_hash(held[120]))]


# --- protocol v3 is untouched ----------------------------------------------------------


def test_a_v3_reveal_publishes_and_commits_exactly_as_before(tmp_path, signer):
    w = World(tmp_path, signer)
    folder = tmp_path / "v3" / "e00000100"
    folder.mkdir(parents=True)
    reveal = {"epoch_id": "e00000100", "protocol_version": 3, "weights": {"1": 1.0},
              "v2_state": {"seat": {}}}
    (folder / "reveal.json").write_text(json.dumps(reveal, indent=2))

    def never(*_a):
        raise AssertionError("the menu hook is reached by a v3 reveal")

    commits = []
    out = rp.publish_due_reveals(str(tmp_path / "v3"), 101, 1, NETUID, Uploader(w.store),
                                 commit=lambda d: commits.append(d) or True, menu=never)
    digest = rp.reveal_digest((folder / "reveal.json").read_bytes())
    assert commits == [f"reveal:e00000100:{digest}"]
    assert json.loads((folder / rp.MARKER).read_text()) == {
        "epoch_id": "e00000100", "sha256": digest, "url": f"gs://{BUCKET}/{NETUID}/e00000100/reveal.json"}
    assert out[0]["committed"] is True and "menu_hash" not in out[0]
    # The view validators are compared on, for a reveal with no menu: as before.
    assert v2_sync.consensus_view(reveal) == json.dumps(
        {"v2_state": reveal["v2_state"], "weights": reveal["weights"]},
        sort_keys=True, separators=(",", ":"))
    assert v2_sync.comparable_reveal(dict(reveal, environment={})) == reveal


def test_nothing_on_the_v1_to_v3_path_imports_the_v4_menu():
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    for rel in ("fugal_subnet/reveal_publisher.py", "fugal_subnet/reveal_sync.py",
                "fugal_subnet/v2_sync.py", "fugal_subnet/commit_reveal.py"):
        tree = ast.parse((root / rel).read_text())
        names = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                 for a in n.names} | {n.module for n in ast.walk(tree)
                                      if isinstance(n, ast.ImportFrom) and n.module}
        assert not any("menu_state" in x or "menu_publish" in x or "menu_rules" in x
                       for x in names), rel
    # The validator reaches the menu only inside its v4 helpers.
    tree = ast.parse((root / "neurons/validator.py").read_text())
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        uses = {a.name for n in ast.walk(fn) if isinstance(n, ast.ImportFrom) for a in n.names}
        if "menu_publish" in uses or "menu_inputs" in uses:
            assert fn.name.startswith("_v4_"), fn.name


# --- the read side for miners and the product ----------------------------------------


def test_menu_bytes_are_taken_only_with_the_hash_and_a_bad_source_never_hides_a_good_one(tmp_path):
    raw = b'{"a":1}'
    h = hashlib.sha256(raw).hexdigest()
    key = reveal_sync.menu_object(NETUID, h)
    served = {f"https://storage.googleapis.com/{BUCKET}/{key}": raw + b"x",
              f"https://mirror.example/{key}": raw}

    def opener(req, timeout=None):
        if req.full_url not in served:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)
        return io.BytesIO(served[req.full_url])

    assert reveal_sync.fetch_menu_bytes(h, bucket=BUCKET, netuid=NETUID, opener=opener) is None
    assert reveal_sync.fetch_menu_bytes(h, bucket=BUCKET, netuid=NETUID, opener=opener,
                                        mirrors=("http://plain.example", "https://mirror.example/")
                                        ) == raw
    # Bytes over the reader's 1 MiB limit still come back when they carry the
    # hash (every node then refuses them alike and carries the menu).
    big = b" " * (ms.MENU_MAX_BYTES + 10)
    hb = hashlib.sha256(big).hexdigest()
    served[f"https://storage.googleapis.com/{BUCKET}/{reveal_sync.menu_object(NETUID, hb)}"] = big
    assert reveal_sync.fetch_menu_bytes(hb, bucket=BUCKET, netuid=NETUID, opener=opener) == big
    assert reveal_sync.MAX_MENU_FETCH_BYTES >= 4 * ms.MENU_MAX_BYTES
    assert reveal_sync.MAX_LIST_FETCH_BYTES == mr.MAX_LIST_BYTES
    assert reveal_sync.MAX_SIGNATURE_FETCH_BYTES == mr.MAX_SIGNATURE_BYTES
    with pytest.raises(ValueError):
        reveal_sync.menu_object(NETUID, "../x")


# --- the validator's v4 branch --------------------------------------------------------


class _HP:
    def __init__(self, period):
        self.commit_reveal_weights_enabled = True
        self.commit_reveal_period = period


class _Sub:
    def __init__(self, period):
        self.period = period

    def get_subnet_hyperparameters(self, netuid):
        return _HP(self.period)

    def get_current_block(self):
        return 26489 * 360


class _Cfg:
    PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4 = 5000
    REFERENCE_BUCKET = ""


def _setup(tmp_path, *, period=1, root=False, bucket=BUCKET, cfg=_Cfg, live=True):
    from neurons import validator

    role = reveal_sync.Role(root, None if root else str(tmp_path / "ref"), None if root else BUCKET,
                            "test", root_hotkey=None if root else ROOT_HK)
    return validator._v4_menu_setup(
        cfg=cfg, subtensor=_Sub(period), netuid=NETUID, my_hotkey=ROOT_HK if root else "5Fme",
        role=role, root_hotkey=ROOT_HK, reveal_delay=1, live=live, blocks_per_epoch=360,
        v4_block=26400 * 360, state_path=str(tmp_path / "state.json"), reveal_bucket=bucket)


def test_no_release_schedules_v4_so_the_validator_sets_up_no_menu():
    from fugal_subnet import config
    from neurons import validator

    assert validator._v4_activation_block(config, config.FINNEY_GENESIS_HASH) is None


def test_a_live_validator_whose_chain_holds_reveals_too_long_refuses_to_start(tmp_path):
    import click

    node, publisher = _setup(tmp_path)
    assert node.first_epoch == 26400 and publisher is None
    # The production step parameters: the pinned v4 mix and the named floor.
    from fugal_subnet import epoch_plan
    assert node.step_params == epoch_plan.menu_step_params_v4(5000)
    _root_node, root_publisher = _setup(tmp_path / "r", root=True)
    assert isinstance(root_publisher, mp.MenuPublisher)
    with pytest.raises(click.ClickException, match="holds reveals for 2 epochs"):
        _setup(tmp_path, period=2)
    with pytest.raises(click.ClickException, match="publishes no reveals"):
        _setup(tmp_path, root=True, bucket="")

    class Unpinned(_Cfg):
        PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4 = None

    with pytest.raises(click.ClickException, match="not pinned"):
        _setup(tmp_path, cfg=Unpinned)
    _setup(tmp_path, period=2, live=False)                  # mock: a warning


def _fake_menu_inputs(calls):
    """A stand-in with menu_inputs' names and signatures (W4a): what the
    validator's v4 helper calls, in the order it calls them."""
    import types

    fake = types.ModuleType("fugal_subnet.menu_inputs")

    @dataclasses.dataclass(frozen=True)
    class VerifiedProof:
        proof: object
        coldkey: str

    def summarize_epoch(epoch, state, proofs, *, pooled, category_of, counted_heads):
        calls.append(("summarize", epoch, state.epoch, [p.coldkey for p in proofs],
                      list(pooled), sorted(category_of), list(counted_heads)))
        return {"epoch": epoch}

    def step_inputs(start, epoch_state, window, proofs, *, epoch, nonce, lab_list, params,
                    category_of, device_list=None, lab_list_in_force=None):
        calls.append(("step_inputs", start.epoch, epoch_state.epoch,
                      [s["epoch"] for s in window], epoch, nonce, lab_list.serial,
                      params is fx.PARAMS, device_list))
        return changing(epoch) if epoch == 100 else quiet(epoch)

    fake.VerifiedProof = VerifiedProof
    fake.summarize_epoch = summarize_epoch
    fake.summary_to_dict = lambda s: dict(s)
    fake.summary_from_dict = lambda d: dict(d)
    fake.step_inputs = step_inputs
    return fake


def test_the_v4_epoch_calls_the_inputs_in_order_and_the_root_publishes_first(
        tmp_path, signer, monkeypatch):
    import sys

    import fugal_subnet
    from neurons import validator

    w = World(tmp_path, signer)
    f = w.node("follower")
    monkeypatch.setattr("fugal_subnet.commit_reveal.EPOCH_DIR", str(w.epoch_dir))
    calls = []
    fake = _fake_menu_inputs(calls)
    # Both routes an import can take: the module table, and the package's
    # attribute once the real module was imported by an earlier test.
    monkeypatch.setitem(sys.modules, "fugal_subnet.menu_inputs", fake)
    monkeypatch.setattr(fugal_subnet, "menu_inputs", fake, raising=False)
    nonce = bytes(range(32))

    def epoch(node, k, *, root):
        return validator._v4_menu_epoch(
            node, w.publisher if root else None, node.at_boundary(k), epoch_index=k, nonce=nonce,
            verified_proofs={3: "proof-3", 1: "proof-1"}, coldkey_of={1: "5Fck-a", 3: "5Fck-b"},
            pooled=["pair"], category_of={"q1": "math"}, counted_heads=["h1"], window_epochs=48,
            root=root, uploader=Uploader(w.store) if root else None, reveal_delay=1,
            netuid=NETUID, commit=w.chain.commit)

    for k in (100, 101, 102):
        w.chain.block = k * BPE
        calls.clear()
        follower_fields, published, _deferred = epoch(f, k, root=False)
        assert not published
        # Kept, not yet run: the follower's step for k runs at k + 2.
        assert [c[0] for c in calls] == ["summarize"] + (["step_inputs"] if k == 102 else [])
        calls.clear()
        w.chain.block = k * BPE + 5                           # past the boundary
        fields, published, _deferred = epoch(w.root, k, root=True)
        assert published
        # Proofs in uid order with their chain coldkeys; the step from
        # MenuState(k + 1) with the proofs of MenuState(k), the window
        # holding epoch k's summary.
        assert calls == [
            ("summarize", k, k, ["5Fck-a", "5Fck-b"], ["pair"], ["q1"], ["h1"]),
            ("step_inputs", k + 1, k, list(range(100, k + 1)), k, nonce, fx.ALLOW.serial, True,
             None)]
        assert ms.from_json(fields[mp.REVEAL_NEXT].encode()).epoch == k + 2
        folder = w.epoch_dir / epoch_id_for_block(k)          # the loop's reveal_epoch
        folder.mkdir()
        (folder / "reveal.json").write_text(json.dumps(
            {"epoch_id": epoch_id_for_block(k), "protocol_version": 4, **fields}))
    # The root's reveal of 100 (published during 101) committed the menu of
    # 102, and the follower's audit of it, from its own step, is empty.
    assert follower_fields[mp.REVEAL_AUDIT]["differences"] == []
    assert follower_fields[mp.REVEAL_AUDIT]["root"] == w.chain.read(102)
    # Inputs that cannot be built: the reveal still names the menu it ran on.
    w.chain.block = 1030
    fields, _, _ = validator._v4_menu_epoch(
        f, None, f.at_boundary(103), epoch_index=103, nonce=nonce, verified_proofs={1: "p"},
        coldkey_of={}, pooled=[], category_of={}, counted_heads=[], window_epochs=48,
        root=False, uploader=None, reveal_delay=1, netuid=NETUID, commit=None)
    assert fields["menu"] == mp.menu_field(f.book.held(103), f.book.source_of(103))


def test_the_counted_heads_are_the_seat_holders_and_the_senior_bars():
    """Section 9.2's use: picks by heads that held the seat or were senior
    bars. A senior bar can be a head that holds no seat now (a retired
    champion still barring through its retire window, a ghost)."""
    from fugal_subnet.champion import SeatState
    from neurons import validator

    assert validator._v4_counted_heads(SeatState(), {}) == []
    seat = SeatState(champion="c" * 64, history=("a" * 64,))
    assert validator._v4_counted_heads(seat, {}) == ["a" * 64, "c" * 64]
    bars = {"x" * 64: ("g" * 64, "c" * 64), "y" * 64: ()}
    assert validator._v4_counted_heads(seat, bars) == ["a" * 64, "c" * 64, "g" * 64]


def test_the_epoch_result_carries_the_senior_bars_the_seat_was_decided_with(monkeypatch):
    """run_v2_epoch exposes the bars it handed seat_step (V2EpochResult.senior_bars),
    so the loop reads them rather than computing them a second way; the
    reveal is unchanged."""
    from fugal_subnet import copy_rule, v2_epoch
    from tests import test_v2_epoch as t

    plain = t.run(0, {1: t.oracle, 2: t.always("m/mid")})
    seen = {}
    real = copy_rule.senior_bars

    def spy(*a, **k):
        seen["bars"] = real(*a, **k)
        return seen["bars"]

    monkeypatch.setattr(copy_rule, "senior_bars", spy)
    r = t.run(0, {1: t.oracle, 2: t.always("m/mid")})
    assert r.senior_bars is seen["bars"] and r.reveal == plain.reveal
    assert isinstance(v2_epoch.V2EpochResult(r.window, r.decision, r.copy_state, r.uids,
                                             r.weights, r.reveal).senior_bars, dict)


def test_the_loop_counts_the_bars_of_its_own_epoch_result():
    import ast
    import inspect

    from neurons import validator

    calls = [n for n in ast.walk(ast.parse(inspect.getsource(validator)))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_v4_counted_heads"]
    assert [[ast.unparse(a) for a in c.args] for c in calls] == [
        ["v2_result.decision.state", "v2_result.senior_bars"]]


def test_the_window_of_summaries_is_kept_beside_the_state_and_bounded(tmp_path):
    book = mp.MenuBook(str(tmp_path / "m"))
    for e in range(100, 110):
        book.keep_summary(e, {"epoch": e})
    book.keep_summary(105, {"epoch": 105, "again": True})
    again = mp.MenuBook(str(tmp_path / "m"))
    assert [d["epoch"] for d in again.summaries(through=107, size=4)] == [104, 105, 106, 107]
    assert again.summaries(through=105, size=1) == [{"epoch": 105, "again": True}]


# --- the coldkey map under v4 (integration decision 2g) --------------------------------


def test_under_v4_an_unpinned_metagraph_fails_the_epoch_for_a_retry():
    """The v4 menu step's votes and counts are per coldkey, read from the
    metagraph at the collection block. Under v1-v3 a failed pinned read falls
    back to the current metagraph and says so; under v4 that map would feed
    a consensus input (the root's step, a follower's audit), so the epoch
    fails and is retried, as a failed read of the menu commitment does. The
    retry in the running loop is tests/test_v4_loop.py's (a follower's pinned
    read fails once, before the query, and the epoch is run again)."""
    from neurons import validator

    validator._v4_require_pinned_metagraph(3, False)            # v1-v3: the fallback stands
    validator._v4_require_pinned_metagraph(4, True)
    with pytest.raises(RuntimeError, match="metagraph"):
        validator._v4_require_pinned_metagraph(4, False)


@pytest.mark.parametrize("missed", [2, 10, 60])
def test_a_node_whose_endpoint_no_longer_serves_its_walk_re_anchors_on_the_roots_reveal(
        tmp_path, signer, missed):
    """A walk that meets a link the chain endpoint has pruned re-anchors on
    the menu the root's newest reveal names (verified against the root's
    commitment at the current boundary, its bytes against that hash) and
    walks the few links after it: the same menus as the root. With every
    block served, nothing re-anchors."""
    w = World(tmp_path, signer)
    last = 100 + missed + 3
    for k in range(100, last + 1):
        w.root_epoch(k, changing(k) if k in (100, 102) else quiet(k))
    keep = 3 * BPE + 3                                   # three epochs of state

    def read(epoch):
        if w.chain.block - epoch * BPE > keep:
            raise ConnectionError(f"State discarded for block {epoch * BPE}")
        return w.chain.read(epoch)

    def fetch_reveal(epoch_id, sha):
        path = w.epoch_dir / epoch_id / "reveal.json"
        raw = path.read_bytes() if path.exists() else None
        return raw if raw is not None and hashlib.sha256(raw).hexdigest() == sha else None

    late = w.node("late")                                # holds the menu of 99 only
    late.read_commitment = read
    late.read_anchor = lambda e: mp.read_root_anchor(w.chain, NETUID, ROOT_HK, e, e * BPE,
                                                     fetch_reveal=fetch_reveal)
    w.chain.block = (last + 1) * BPE
    got = late.at_boundary(last + 1)
    assert got.ok and got.state == w.root.at_boundary(last + 1).state
    # Without the anchor the walk stops at the first pruned link, every epoch.
    stuck = w.node("stuck")
    stuck.read_commitment = read
    assert not stuck.at_boundary(last + 1).ok
    # With every block served the walk is the plain one, and the same.
    full = w.node("full")
    full.read_anchor = late.read_anchor
    full.max_links = 100
    assert full.at_boundary(last + 1).state == got.state
    assert all(full.book.held(k) == w.root.book.held(k) for k in range(100, last + 1))


def test_the_window_fill_takes_the_sync_ledger_and_a_pruned_read_only_leaves_an_epoch_unfilled(
        tmp_path, monkeypatch):
    """The window's fill from the root's reveals reads each reveal's sha from
    the follower sync's pointer ledger first, and the chain only for an
    epoch the ledger lacks: an endpoint that no longer serves that read
    leaves the epoch unfilled (never an error, never a stop)."""
    from neurons import validator as V

    ref = tmp_path / "reference"
    for k in (100, 101, 102):
        eid = epoch_id_for_block(k)
        (ref / eid).mkdir(parents=True)
        (ref / eid / "reveal.json").write_text(json.dumps({"epoch_id": eid}))
    ledger = reveal_sync.pointer_ledger_path(str(ref))
    shas = {k: hashlib.sha256((ref / epoch_id_for_block(k) / "reveal.json").read_bytes()
                              ).hexdigest() for k in (100, 101, 102)}
    for k in (100, 101):
        reveal_sync.record_pointer(ledger, epoch_id_for_block(k), shas[k])

    class Node:
        def read_reveal_sha(self, k):
            raise ConnectionError(f"State discarded for block {(k + ms.MENU_LAG) * BPE}")

    seen = {}
    real = mp.root_summaries

    def capture(directory, sha_of, *, check):
        seen["sha_of"] = sha_of
        return real(directory, sha_of, check=lambda doc: types.SimpleNamespace(
            epoch=doc.get("epoch")))
    monkeypatch.setattr(mp, "root_summaries", capture)
    source = V._v4_root_summaries(Node(), root=False, reference_dir=str(ref))
    assert seen["sha_of"](100) == shas[100] and seen["sha_of"](101) == shas[101]
    with pytest.raises(ConnectionError):
        seen["sha_of"](102)
    assert source(102) is None                              # unfilled, not an error
