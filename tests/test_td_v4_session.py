"""Protocol v4: the miner's TD across epochs (td_v4.V4Session) and the miner's epoch.

The session reads every menu from the last one the TD ran on to the epoch
being run, as every node read them, takes menu bytes and signed lists from
any source and verifies both, and a menu it cannot read ends the epoch with
no proof. No network, no spend.
"""
from __future__ import annotations

import dataclasses
import hashlib
import threading

import numpy as np
import pytest

from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from fugal_subnet.tee import harness, td_v4
from tests import menu_fixtures as mf
from tests.signed_list_fixtures import Signer, device_bytes
from tests.test_harness_v4 import FIXTURE_IDS, HOTKEY, NONCE, SLICE, Provider, _plan
from tests.test_tee_e2e import _head, _pool

LIST_BYTES = mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, mf.ALLOW.serial, labs=mf.LABS,
                                     hosts=mf.HOSTS, first_party=mf.FIRST_PARTY)
assert hashlib.sha256(LIST_BYTES).hexdigest() == mf.ALLOW.sha256


@pytest.fixture(autouse=True)
def _small_menus(monkeypatch):
    mf.allow_small_menus(monkeypatch)


@pytest.fixture
def signer(tmp_path, monkeypatch):
    s = Signer(tmp_path, "release")
    monkeypatch.setattr(mr, "list_signers", lambda: s.signers_line().encode())
    return s


class Chain:
    """The root's commitments and the bytes offered for them."""

    def __init__(self, signer, *, offer_list=True):
        self.commitments: dict[int, str] = {}
        self.menus: dict[str, bytes] = {}
        self.signer = signer
        self.offer_list = offer_list

    def commit(self, state: ms.MenuState, *, offer=True):
        raw = ms.canonical_json(state)
        h = hashlib.sha256(raw).hexdigest()
        self.commitments[state.epoch] = h
        if offer:
            self.menus[h] = raw

    def list_bytes(self, serial, sha):
        if not self.offer_list or sha != mf.ALLOW.sha256:
            return None
        return LIST_BYTES, self.signer.sign(LIST_BYTES)


def _session(chain, pool, provider, **over):
    fields = dict(first_epoch=100, plan=_plan(pool), fetch_listing=provider.fetch,
                  read_commitment=chain.commitments.get, menu_bytes=chain.menus.get,
                  list_bytes=chain.list_bytes, anchor=mf.state())
    fields.update(over)
    return td_v4.V4Session(**fields)


def _epoch(monkeypatch, session, epoch, provider, pool):
    monkeypatch.setattr(harness, "_call_model", provider.call)
    hidden = np.random.RandomState(0).randn(len(pool), 1024).astype(np.float32)
    inputs = session.inputs_for(epoch)
    proof = harness.run_benchmark(
        nonce=NONCE, head_bytes=_head(FIXTURE_IDS), benchmark_pool=pool,
        proxy=td_v4_proxy(), hidden_states=hidden, slice_size=SLICE,
        epoch_id=f"e{epoch:08d}", source_hash="img", hotkey=HOTKEY, protocol_version=4,
        v4=inputs)
    session.ran(inputs, proof)
    return proof


def td_v4_proxy():
    from tests.test_harness_v4 import _Proxy
    return _Proxy(port=0)


def test_the_session_carries_reads_and_follows_committed_menus(monkeypatch, signer):
    chain, pool, provider = Chain(signer), _pool(), Provider()
    s = _session(chain, pool, provider)
    p102 = _epoch(monkeypatch, s, 102, provider, pool)
    assert p102.menu_hash == ms.menu_hash(ms.carry(mf.state()))
    # Epoch 103 is skipped by this TD; 104's menu is a committed change.
    at103 = ms.carry(ms.carry(mf.state()))
    # An entry retires (a structural change), leaving its record.
    gone = at103.entries[0]
    changed = dataclasses.replace(
        ms.carry(at103), serial=at103.serial + 1, entries=at103.entries[1:],
        retired=(ms.Retired(gone.id, 104, ms.REASON_GONE, 104 + ms.PROPOSAL_COOLDOWN_EPOCHS),))
    chain.commit(changed)
    p104 = _epoch(monkeypatch, s, 104, provider, pool)
    assert p104.menu_hash == ms.menu_hash(changed)
    assert s.previous == changed


def test_without_the_committed_bytes_the_epoch_has_no_proof(monkeypatch, signer):
    chain, pool, provider = Chain(signer), _pool(), Provider()
    s = _session(chain, pool, provider)
    changed = dataclasses.replace(ms.carry(mf.state()), serial=mf.state().serial + 1,
                                  entries=mf.state().entries[:2])
    chain.commit(changed, offer=False)
    with pytest.raises(ms.MenuUnavailable):
        _epoch(monkeypatch, s, 102, provider, pool)
    assert provider.calls == [] and provider.fetched == []


@pytest.mark.parametrize("why", ["no list offered", "bad signature"])
def test_without_a_verified_list_the_epoch_has_no_proof(monkeypatch, signer, tmp_path, why):
    chain, pool, provider = Chain(signer, offer_list=why != "no list offered"), _pool(), Provider()
    if why == "bad signature":
        other = Signer(tmp_path, "other")
        chain.list_bytes = lambda serial, sha: (LIST_BYTES, other.sign(LIST_BYTES))
    s = _session(chain, pool, provider)
    with pytest.raises(ms.MenuUnavailable):
        _epoch(monkeypatch, s, 102, provider, pool)
    assert provider.calls == []


def _adopting_devices(serial=1):
    raw = device_bytes(serial)
    at = ms.carry(mf.state())
    return raw, dataclasses.replace(at, serial=at.serial + 1, devices=ms.AllowlistRef(
        serial, hashlib.sha256(raw).hexdigest()))


def test_a_menu_that_adopts_a_device_list_runs_with_that_list(monkeypatch, signer):
    """A committed menu naming a newer device list is read only with the
    list itself (menu_for_epoch, 4.3). The session takes the device list from
    any source, verifies it, and hands it to the reader with the menu."""
    chain, pool, provider = Chain(signer), _pool(), Provider()
    raw, adopting = _adopting_devices()
    chain.commit(adopting)
    offered = []

    def device_list_bytes(serial, sha):
        offered.append((serial, sha))
        return (raw, signer.sign(raw)) if sha == adopting.devices.sha256 else None

    s = _session(chain, pool, provider, device_list_bytes=device_list_bytes)
    proof = _epoch(monkeypatch, s, 102, provider, pool)
    assert proof.menu_hash == ms.menu_hash(adopting) and s.previous == adopting
    assert offered == [(1, adopting.devices.sha256)]


@pytest.mark.parametrize("why", ["no source", "not offered", "bad signature"])
def test_without_the_device_list_a_menu_names_the_epoch_has_no_proof(monkeypatch, signer,
                                                                      tmp_path, why):
    chain, pool, provider = Chain(signer), _pool(), Provider()
    raw, adopting = _adopting_devices()
    chain.commit(adopting)
    over = {}
    if why == "not offered":
        over["device_list_bytes"] = lambda serial, sha: None
    if why == "bad signature":
        other = Signer(tmp_path, "other")
        over["device_list_bytes"] = lambda serial, sha: (raw, other.sign(raw))
    s = _session(chain, pool, provider, **over)
    with pytest.raises(ms.MenuUnavailable):
        _epoch(monkeypatch, s, 102, provider, pool)
    assert provider.calls == []


def test_a_menu_skipped_while_catching_up_is_read_too(monkeypatch, signer):
    """A refused menu in an epoch the TD did not run is carried there, as on
    every node, so the next epoch follows the carried one."""
    chain, pool, provider = Chain(signer), _pool(), Provider()
    s = _session(chain, pool, provider)
    bad = dataclasses.replace(ms.carry(mf.state()), serial=99)
    chain.commit(bad)
    proof = _epoch(monkeypatch, s, 103, provider, pool)
    assert proof.menu_hash == ms.menu_hash(ms.carry(ms.carry(mf.state())))


def test_the_first_v4_epoch_needs_nothing_but_the_seed(monkeypatch, signer):
    from tests.test_endpoint_reads import PAGES
    chain, pool, provider = Chain(signer), _pool(), Provider(pages=PAGES)
    s = _session(chain, pool, provider, first_epoch=102, anchor=None)
    inputs = s.inputs_for(102)
    assert inputs.previous is None and inputs.allowlist is None
    seed_ids = list(ms.seed_state(102).ids())[:3]
    monkeypatch.setattr(harness, "_call_model", provider.call)
    hidden = np.random.RandomState(0).randn(len(pool), 1024).astype(np.float32)
    proof = harness.run_benchmark(
        nonce=NONCE, head_bytes=_head(seed_ids), benchmark_pool=pool, proxy=td_v4_proxy(),
        hidden_states=hidden, slice_size=SLICE, epoch_id="e00000102", source_hash="img",
        hotkey=HOTKEY, protocol_version=4, v4=inputs)
    assert s.ran(inputs, proof) == ms.seed_state(102)


@pytest.mark.parametrize("offered", [b"{" * 10, b" " * (ms.MENU_MAX_BYTES + 1), "text"])
def test_bytes_that_are_not_the_committed_menu_are_never_parsed(monkeypatch, signer, offered):
    chain, pool, provider = Chain(signer), _pool(), Provider()
    changed = dataclasses.replace(ms.carry(mf.state()), serial=mf.state().serial + 1,
                                  entries=mf.state().entries[:2])
    chain.commit(changed, offer=False)
    chain.menus[chain.commitments[102]] = offered
    monkeypatch.setattr(ms, "from_json", lambda raw: pytest.fail("parsed untrusted bytes"))
    s = _session(chain, pool, provider)
    inputs = s.inputs_for(102)
    assert inputs.published is None
    with pytest.raises(ms.MenuUnavailable):
        td_v4.epoch_menu(102, previous=inputs.previous, committed_hash=inputs.committed_hash,
                         published=inputs.published, allowlist=inputs.allowlist)


def test_committed_bytes_over_the_readers_limit_are_carried_as_on_every_node(signer):
    """Bytes that carry the committed hash and are larger than the reader's
    limit (menu_state.MENU_MAX_BYTES) are refused unparsed by the reader on
    every node, which carries the previous menu. The TD does the same; it
    does not treat them as missing, which would stop its proofs while every
    validator ran the epoch on the carried menu."""
    chain = Chain(signer)
    for size in (ms.MENU_MAX_BYTES + 1, 4 * ms.MENU_MAX_BYTES + 1):
        _carried_when_offered(chain, signer, b" " * size)


def _carried_when_offered(chain, signer, big):
    chain.commitments[102] = hashlib.sha256(big).hexdigest()
    chain.menus[chain.commitments[102]] = big
    s = _session(chain, _pool(), Provider())
    inputs = s.inputs_for(102)
    assert inputs.published == big
    assert td_v4.epoch_menu(102, previous=inputs.previous, committed_hash=inputs.committed_hash,
                            published=inputs.published,
                            allowlist=inputs.allowlist) == ms.carry(mf.state())


def _refused_bytes(kind: str) -> bytes:
    import json

    prev = mf.state()
    doc = ms.to_dict(dataclasses.replace(ms.carry(prev), serial=prev.serial + 1))
    if kind == "wrong serial, names a list nobody holds":
        doc["serial"] = prev.serial + 7
        doc["allowlists"]["labs"] = {"serial": 99, "sha256": "ab" * 32}
    elif kind == "rolled-back list":
        doc["allowlists"]["labs"] = {"serial": 0, "sha256": ""}
    elif kind == "a host of another type":
        doc["entries"][0]["provider"]["order"] = [{}]
    elif kind == "a moved grace, same list":
        doc["entries"][0]["joined_epoch"] = doc["entries"][0]["scored_from"] = 5
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


@pytest.mark.parametrize("kind", ["wrong serial, names a list nobody holds", "rolled-back list",
                                  "a host of another type", "a moved grace, same list"])
def test_a_refused_menu_costs_the_td_one_epoch_of_menu_changes_and_nothing_else(
        monkeypatch, signer, kind):
    """Bytes every node refuses: the TD runs the carried menu, checked under
    the list the menu it holds names (not whatever list the refused bytes
    name), in that epoch and every later one, as every validator does."""
    chain, pool, provider = Chain(signer), _pool(), Provider()
    raw = _refused_bytes(kind)
    h = hashlib.sha256(raw).hexdigest()
    chain.commitments[102] = h
    chain.menus[h] = raw
    s = _session(chain, pool, provider)
    expect = ms.carry(mf.state())
    for epoch in (102, 103, 104):
        proof = _epoch(monkeypatch, s, epoch, provider, pool)
        assert proof.menu_hash == ms.menu_hash(expect)
        expect = ms.carry(expect)


def test_the_tds_menu_is_the_committed_one_or_the_carried_one_whatever_the_bytes(signer):
    """The one-value mutation fuzz of the reader, through the TD's session:
    the epoch runs on the committed menu or on the carried one, or (a list
    nobody offered) has no menu; no other exception leaves `epoch_menu`."""
    import json

    from tests.test_menu_reader_checks import one_value_mutations

    prev = mf.state()
    base = json.loads(ms.canonical_json(dataclasses.replace(ms.carry(prev),
                                                            serial=prev.serial + 1)))
    outcomes = {"carried": 0, "committed": 0, "unavailable": 0}
    for raw in one_value_mutations(base, 300, seed=12):
        chain = Chain(signer)
        h = hashlib.sha256(raw).hexdigest()
        chain.commitments[102] = h
        chain.menus[h] = raw
        try:
            got = _session(chain, _pool(40), Provider()).inputs_for(102).menu()
        except ms.MenuUnavailable:
            outcomes["unavailable"] += 1
            continue
        if got == ms.carry(prev):
            outcomes["carried"] += 1
        else:
            assert ms.canonical_json(got) == raw
            outcomes["committed"] += 1
    assert outcomes["carried"] > 200, outcomes


def test_a_long_gap_needs_a_recent_menu(signer):
    chain = Chain(signer)
    s = _session(chain, _pool(), Provider())
    with pytest.raises(ms.MenuUnavailable):
        s.inputs_for(101 + td_v4.MAX_CATCH_UP_EPOCHS + 1)


# --- a TD that starts late, or falls behind, does not replay history -------------------


class LiteChain(Chain):
    """A node that serves pinned reads only near the chain's head: a
    commitment read at an epoch's boundary fails once that epoch is past
    (a non-archive node keeps ~300 blocks, an epoch is 360). `now` is the
    epoch the chain is in."""

    def __init__(self, signer, now):
        super().__init__(signer)
        self.now = now
        self.reads = []

    def read(self, epoch):
        self.reads.append(epoch)
        if epoch < self.now:
            raise RuntimeError(f"State already discarded for the boundary of epoch {epoch}")
        return self.commitments.get(epoch)


def _history(chain, first, last):
    """The menus every node ran on from `first` to `last`: mf.state() from
    `first`, carried, with one committed change at `last - 1` (an entry
    retires). Returns {epoch: menu}. The root's store holds the bytes of the
    menus its reveals name (its publisher puts them)."""
    menus = {first: dataclasses.replace(mf.state(), epoch=first)}
    for e in range(first + 1, last + 1):
        menus[e] = ms.carry(menus[e - 1])
        if e == last - 1:
            gone = menus[e - 1].entries[0]
            menus[e] = dataclasses.replace(
                menus[e], serial=menus[e - 1].serial + 1, entries=menus[e - 1].entries[1:],
                retired=(ms.Retired(gone.id, e, ms.REASON_GONE, e + ms.PROPOSAL_COOLDOWN_EPOCHS),))
            chain.commit(menus[e])
    return menus


def _reveal_anchor(chain, menus):
    """read_anchor: the root's newest reveal at `epoch`'s boundary is reveal
    epoch - MENU_LAG; its `menu` field names the menu that epoch ran on, and
    the root's store holds its bytes."""
    def read(epoch):
        k = epoch - ms.MENU_LAG
        raw = ms.canonical_json(menus[k])
        chain.menus[hashlib.sha256(raw).hexdigest()] = raw
        return k, hashlib.sha256(raw).hexdigest()
    return read


def test_a_td_started_long_after_activation_proves_on_its_first_full_epoch(monkeypatch, signer,
                                                                           tmp_path):
    """2,000 epochs into v4, with nothing kept and no anchor of its own: the
    newest root reveal anchors it, and a few links later the epoch is proved."""
    first, now = 100, 2100
    chain, pool, provider = Chain(signer), _pool(), Provider()
    menus = _history(chain, now - 30, now)
    reads = []

    def read_commitment(e):
        reads.append(e)
        return chain.commitments.get(e)

    s = _session(chain, pool, provider, first_epoch=first, anchor=None,
                 read_commitment=read_commitment, read_anchor=_reveal_anchor(chain, menus),
                 state_dir=tmp_path / "v4")
    proof = _epoch(monkeypatch, s, now, provider, pool)
    assert proof.menu_hash == ms.menu_hash(menus[now])
    assert sorted(set(reads)) == [now - 1, now]            # two links, no history
    # Kept on the data volume: a restart resumes from it, with no anchor at all.
    assert (tmp_path / "v4" / td_v4.MENU_FILE).read_bytes() == ms.canonical_json(menus[now])
    again = _session(chain, pool, provider, first_epoch=first, anchor=None,
                     state_dir=tmp_path / "v4")
    assert again.previous == menus[now]
    nxt = _epoch(monkeypatch, again, now + 1, provider, pool)
    assert nxt.menu_hash == ms.menu_hash(ms.carry(menus[now]))


def test_a_td_on_a_node_that_keeps_only_recent_state_recovers_after_a_skipped_epoch(
        monkeypatch, signer):
    """It ran epoch 1000, skipped 1001 (no read at its boundary while it was
    recent), and its node no longer serves that boundary: epoch 1002 costs
    no more than itself, and 1003 is proved from the newest root reveal."""
    chain, pool, provider = LiteChain(signer, 1000), _pool(), Provider()
    menus = _history(chain, 990, 1004)
    s = _session(chain, pool, provider, first_epoch=100, anchor=menus[999],
                 read_commitment=chain.read, read_anchor=_reveal_anchor(chain, menus))
    assert _epoch(monkeypatch, s, 1000, provider, pool).menu_hash == ms.menu_hash(menus[1000])
    chain.now = 1002
    with pytest.raises(ms.MenuUnavailable):
        _epoch(monkeypatch, s, 1002, provider, pool)
    assert s.previous == menus[1000]                         # nothing unverified was kept
    chain.now = 1003
    proof = _epoch(monkeypatch, s, 1003, provider, pool)
    assert proof.menu_hash == ms.menu_hash(menus[1003])     # the change at 1003 included


def test_a_running_td_whose_epoch_failed_late_needs_only_its_next_boundary(monkeypatch, signer):
    """The menu of an epoch is kept when it is read, before any call, so an
    epoch whose proof failed later still leaves the next one a one-link walk."""
    chain, pool, provider = LiteChain(signer, 500), _pool(), Provider()
    menus = _history(chain, 490, 503)
    s = _session(chain, pool, provider, anchor=menus[499], read_commitment=chain.read)
    s.inputs_for(500)                                        # the harness then raised
    assert s.previous == menus[500]
    chain.now = 501
    assert _epoch(monkeypatch, s, 501, provider, pool).menu_hash == ms.menu_hash(menus[501])


class _RootChain:
    """The part of a subtensor commitments.get_commitments_with_blocks reads:
    the root's commitment as of each block."""

    def __init__(self, hotkey, at):
        self.hotkey, self.at, self.blocks = hotkey, at, []

    def query_map(self, module, name, params, block=None):
        self.blocks.append(block)
        data = self.at.get(block, "")
        if not data:
            return []
        raw = data.encode()
        return [(self.hotkey, {"block": block, "info": {"fields": [{f"Raw{len(raw)}":
                                                                    "0x" + raw.hex()}]}})]


def test_the_miners_session_starts_from_the_newest_root_reveal_2000_epochs_in(
        monkeypatch, signer, tmp_path):
    """The session as the miner process builds it (neurons.miner._v4_session),
    restarted 2,000 epochs into v4 with nothing kept: the root's commitment
    read at the epoch's boundary names its newest reveal, the reveal names
    the menu its epoch ran on, and the epoch is proved after two links. The
    menu is kept in the state directory."""
    import json

    import neurons.miner as miner
    from fugal_subnet import config, epoch_plan, reveal_sync
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block
    from fugal_subnet.reveal_publisher import commitment_for

    first, now, bpe, root = 100, 2100, 10, "5Froot-hotkey"
    chain, pool, provider = Chain(signer), _pool(), Provider()
    menus = _history(chain, now - 30, now)
    reveals = {}
    at = {}
    for e in (now - 1, now):                       # the root's commitment for e: its reveal e - 2
        eid = epoch_id_for_block(e - 2)
        # The reveal's menu field names the last published menu it was
        # carried from (menu_publish.menu_field): the first epoch of its serial.
        src = e - 2
        while src - 1 in menus and menus[src - 1].serial == menus[e - 2].serial:
            src -= 1
        source = {"epoch": epoch_id_for_block(src),
                  "hash": ms.menu_hash(dataclasses.replace(menus[e - 2], epoch=src))}
        raw = json.dumps({"epoch_id": eid, "menu": {"epoch": eid, "source": source,
                                                    "hash": ms.menu_hash(menus[e - 2])}}).encode()
        reveals[eid] = raw
        at[e * bpe] = commitment_for(eid, hashlib.sha256(raw).hexdigest(), chain.commitments.get(e))
        chain.menus[ms.menu_hash(menus[e - 2])] = ms.canonical_json(menus[e - 2])
    sub = _RootChain(root, at)
    monkeypatch.setattr(config, "ROOT_HOTKEY_ENV", root)
    monkeypatch.setattr(config, "REFERENCE_BUCKET", "")
    monkeypatch.setattr(epoch_plan, "plan_v4", lambda p: _plan(p))
    # This test is the menu's start; the TD's published set (and the reads
    # it makes) is tests/test_memo_sync.py's and the loop tests'.
    monkeypatch.setattr(miner, "_v4_published", lambda *a, **k: None)
    monkeypatch.setattr(reveal_sync, "fetch_menu_bytes", lambda h, **kw: chain.menus.get(h))
    monkeypatch.setattr(reveal_sync, "fetch_signed_list",
                        lambda kind, sha, **kw: chain.list_bytes(2, sha))
    s = miner._v4_session(sub, netuid=7, pool=pool, first_epoch=first, blocks_per_epoch=bpe,
                          coldkey=None, genesis="0xlocal", fetch_listing=provider.fetch,
                          state_dir=str(tmp_path / "v4"),
                          fetch_reveal=lambda eid, sha: reveals.get(eid))
    proof = _epoch(monkeypatch, s, now, provider, pool)
    assert proof.menu_hash == ms.menu_hash(menus[now])
    assert sorted(set(sub.blocks)) == [(now - 1) * bpe, now * bpe]
    assert (tmp_path / "v4" / td_v4.MENU_FILE).exists()


def test_an_epoch_run_again_after_a_restart_in_it_finds_the_menu_it_follows(
        monkeypatch, signer, tmp_path):
    """The kept menu is the epoch's own (it is kept when read, before any
    call); a TD restarted in that epoch runs it again from the root's reveal,
    and in one process an epoch run twice finds the menu before it in memory."""
    chain, pool, provider = Chain(signer), _pool(), Provider()
    menus = _history(chain, 280, 300)
    kw = dict(first_epoch=100, anchor=None, read_anchor=_reveal_anchor(chain, menus),
              state_dir=tmp_path)
    s = _session(chain, pool, provider, **kw)
    s.inputs_for(300)                                        # then the TD restarted
    assert s.previous == menus[300]
    again = _session(chain, pool, provider, **kw)
    assert _epoch(monkeypatch, again, 300, provider, pool).menu_hash == ms.menu_hash(menus[300])
    assert _epoch(monkeypatch, again, 300, provider, pool).menu_hash == ms.menu_hash(menus[300])


def test_a_kept_menu_that_is_not_a_menu_is_set_aside(signer, tmp_path):
    (tmp_path / td_v4.MENU_FILE).write_bytes(b"{not a menu")
    s = _session(Chain(signer), _pool(), Provider(), state_dir=tmp_path)
    assert s.previous is None


def test_the_session_applies_the_roots_join_delta_and_a_restarted_td_keeps_the_menu(
        monkeypatch, signer, tmp_path):
    """The root's join pointer for 102, read at its boundary, names a delta
    the session fetches by hash and applies: the epoch runs on the base menu
    with the entry provisional. A TD restarted in the same epoch resumes on
    that menu from its state directory; one that is offered no bytes for the
    delta has no menu (no proof)."""
    from tests.test_harness_v4 import _joined

    chain, pool, provider = Chain(signer), _pool(), Provider()
    pointer, data = _joined()
    joins = {102: pointer}
    s = _session(chain, pool, provider, read_join=joins.get, join_bytes={
        pointer.delta_hash: data}.get, state_dir=tmp_path / "v4")
    proof = _epoch(monkeypatch, s, 102, provider, pool)
    menu = s.previous
    assert menu.epoch == 102 and menu.entry("lab-c/model-new@high").provisional
    assert proof.menu_hash == ms.menu_hash(menu)
    again = _session(chain, pool, provider, read_join=joins.get, join_bytes={}.get,
                     state_dir=tmp_path / "v4")
    assert again.menu_before(103) == menu                    # kept: needs no delta bytes
    blind = _session(chain, pool, provider, read_join=joins.get, join_bytes={}.get)
    with pytest.raises((ms.MenuUnavailable, td_v4.MenuRejected)):
        blind.inputs_for(102).menu()


# --- the miner's epoch -------------------------------------------------------------------


def test_the_miners_v4_epoch_runs_through_the_session(monkeypatch, signer):
    import neurons.miner as miner
    from fugal_subnet.tee.runtime import TEERuntime

    chain, pool, provider = Chain(signer), _pool(), Provider()
    s = _session(chain, pool, provider)
    monkeypatch.setattr(harness, "_call_model", provider.call)
    current = {"proof": None, "proof_json": "", "epoch_id": "", "lock": threading.Lock()}
    hidden = np.random.RandomState(0).randn(len(pool), 1024).astype(np.float32)
    kw = dict(blocks_per_epoch=360, head_data=_head(FIXTURE_IDS), weights_hash="w", pool=pool,
              hidden_states=hidden, block_hash="0x" + "11" * 32, tee_runtime=TEERuntime(mock=True),
              current_proof=current, slice_size=SLICE, proxy_port=0, explore_models=None,
              explore_size=0, hotkey=HOTKEY, protocol_version=4, v4_session=s)
    miner._run_epoch(epoch_index=102, **kw)
    assert current["epoch_id"] == "e00000102"
    assert current["proof"].menu_hash == ms.menu_hash(ms.carry(mf.state()))
    assert s.previous.epoch == 102

    # A committed menu with no bytes anywhere: no proof for 103, nothing called.
    chain.commit(dataclasses.replace(ms.carry(ms.carry(mf.state())), serial=9), offer=False)
    before = len(provider.calls)
    with pytest.raises(ms.MenuUnavailable):
        miner._run_epoch(epoch_index=103, **kw)
    assert current["epoch_id"] == "e00000102" and len(provider.calls) == before


def test_the_miners_v4_epoch_needs_a_session():
    import neurons.miner as miner
    with pytest.raises(ValueError):
        miner._run_epoch(blocks_per_epoch=360, head_data=b"", weights_hash="w", pool=[],
                         hidden_states=None, epoch_index=1, block_hash="0x00",
                         tee_runtime=None, current_proof={}, slice_size=1, proxy_port=0,
                         explore_models=None, explore_size=0, hotkey=HOTKEY,
                         protocol_version=4)
