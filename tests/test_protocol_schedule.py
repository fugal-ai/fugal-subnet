"""The signed protocol schedule (docs/decisions/SIGNED_SCHEDULE.md).

Throwaway Ed25519 keys sign every document here (tests/signed_list_fixtures:
an OpenSSH signature writer independent of the verifier), and one test signs
with the real `ssh-keygen -Y sign` the owner runs. A fake chain holds the
root's commitment fields per boundary block; the fold reads it through the
same `read_pointer` the neurons use.
"""
import json
import shutil
import subprocess

import pytest

from fugal_subnet import config, join_step
from fugal_subnet import protocol_schedule as ps
from fugal_subnet.chain_proposals import parse_join_field
from tests.signed_list_fixtures import Signer

CHAIN = "0x" + "ab" * 32
OTHER_CHAIN = "0x" + "cd" * 32
BPE = 360
LEAD = 2
ROOT = "5RootHotkeyForTheTests"


@pytest.fixture
def keys(tmp_path):
    owner = Signer(tmp_path, "owner")
    stranger = Signer(tmp_path, "stranger")
    gitonly = Signer(tmp_path, "gitonly")
    signers = (owner.signers_line() + gitonly.signers_line(namespaces="git",
                                                           principal="git@example.test")).encode()
    return owner, stranger, gitonly, signers


def doc(serial, acts, chain=CHAIN):
    return ps.canonical_bytes(serial, chain, acts)


def signed(signer, serial, acts, chain=CHAIN, **kw):
    payload = doc(serial, acts, chain)
    return payload, signer.sign(payload, **kw)


# ── the document ─────────────────────────────────────────────────────────────


def test_canonical_bytes_round_trip(keys):
    owner, _, _, signers = keys
    payload, sig = signed(owner, 3, {"v6": 7200, "v5_1": 3600})
    got = ps.verify(payload, sig, signers=signers)
    assert got.serial == 3 and got.chain == CHAIN
    assert got.activations == (("v5_1", 3600), ("v6", 7200))
    assert got.digest == ps.bundle_digest(payload, sig) and got.signer.startswith("SHA256:")
    assert got.pointer == ps.Pointer(3, got.digest)
    assert json.loads(payload)["kind"] == ps.KIND


@pytest.mark.parametrize("mutate, code", [
    (lambda b: b.replace(b'"format": 1', b'"format": 2'), "malformed"),
    (lambda b: b.replace(b"\n", b"", 1), "not_canonical"),
    (lambda b: b.replace(b'"v6": 7200', b'"v6": 7200.0'), "malformed"),
    (lambda b: b.replace(b'"v6": 7200', b'"V6": 7200'), "malformed"),
    (lambda b: b.replace(b'"v6": 7200', b'"v6": 0'), "malformed"),
    (lambda b: b.replace(b'"v6": 7200', b'"v6": 7200, "v6": 7560'), "duplicate"),
    (lambda b: b.replace(ps.KIND.encode(), b"fugal-menu-allowlist"), "wrong_kind"),
    (lambda b: b.replace(CHAIN.encode(), b"finney"), "malformed"),
    (lambda b: b + b" " * ps.MAX_DOCUMENT_BYTES, "oversized"),
])
def test_format_refusals(mutate, code):
    with pytest.raises(ps.ScheduleRefused) as e:
        ps.parse_unsigned(mutate(doc(1, {"v6": 7200})))
    assert e.value.code == code


def test_a_draft_is_refused_whatever_signs_it(keys):
    owner, _, _, signers = keys
    payload, sig = signed(owner, 0, {"v6": 7200})
    with pytest.raises(ps.ScheduleRefused) as e:
        ps.verify(payload, sig, signers=signers)
    assert e.value.code == "draft"


@pytest.mark.parametrize("case, code", [
    ("stranger", "signer_not_allowed"),
    ("git_namespace_key", "signer_not_allowed"),
    ("signed_as_git", "wrong_namespace"),
    ("tampered", "bad_signature"),
    ("unsigned", "signature_malformed"),
])
def test_signature_refusals(keys, case, code):
    owner, stranger, gitonly, signers = keys
    payload = doc(1, {"v6": 7200})
    sig = {
        "stranger": lambda: stranger.sign(payload),
        "git_namespace_key": lambda: gitonly.sign(payload),
        "signed_as_git": lambda: owner.sign(payload, namespace="git"),
        "tampered": lambda: owner.sign(doc(1, {"v6": 7560})),
        "unsigned": lambda: b"",
    }[case]()
    with pytest.raises(ps.ScheduleRefused) as e:
        ps.verify(payload, sig, signers=signers)
    assert e.value.code == code


def test_a_list_signature_is_not_a_schedule(keys):
    """Same namespace, other kind: the kind inside the signed bytes decides."""
    from fugal_subnet import menu_rules as mr
    from tests.signed_list_fixtures import allowlist_bytes

    owner, _, _, signers = keys
    payload = allowlist_bytes()
    sig = owner.sign(payload)
    mr.verify_signed_list(payload, sig, kind=mr.KIND_MENU_ALLOWLIST, held_serial=0,
                          signers=signers)
    with pytest.raises(ps.ScheduleRefused) as e:
        ps.verify(payload, sig, signers=signers)
    assert e.value.code == "wrong_kind"
    sched = doc(1, {"v6": 7200})
    with pytest.raises(mr.ListRefused):
        mr.verify_signed_list(sched, owner.sign(sched), kind=mr.KIND_MENU_ALLOWLIST,
                              held_serial=0, signers=signers)


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="needs OpenSSH")
def test_the_owners_ssh_keygen_command_verifies(tmp_path):
    key = tmp_path / "release"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "t", "-f", str(key)],
                   check=True)
    pub = (tmp_path / "release.pub").read_text().split()
    signers = f'release@example.test namespaces="git,fugal-list@fugalai.com" {pub[0]} {pub[1]}\n'
    path = tmp_path / "schedule.json"
    path.write_bytes(doc(1, {"v6": 7200}))
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(key), "-n", "fugal-list@fugalai.com",
                    str(path)], check=True, capture_output=True)
    got = ps.verify(path.read_bytes(), (tmp_path / "schedule.json.sig").read_bytes(),
                    signers=signers.encode())
    assert got.as_dict() == {"v6": 7200}


# ── the pointer ──────────────────────────────────────────────────────────────


def test_pointer_field_fits_and_parses():
    p = ps.Pointer(ps.MAX_SERIAL, "f" * 64)
    assert len(p.field()) <= 128 and ps.parse_pointer(p.field()) == p
    for bad in ("sched:0:" + "f" * 64, "sched:01:" + "f" * 64, "sched:1:" + "F" * 64,
                "sched:1:" + "f" * 63, " sched:1:" + "f" * 64, None, 5):
        assert ps.parse_pointer(bad) is None
    assert ps.pointer_in_fields(["sched:1:" + "f" * 64]) is None   # field 0 is the reveal's
    assert ps.pointer_in_fields(["reveal:x", "join:y", "sched:2:" + "a" * 64]) == ps.Pointer(
        2, "a" * 64)


def test_root_fields_keep_the_reveal_and_join_where_old_readers_look():
    writes = []
    rf = join_step.RootFields(lambda f: writes.append(list(f)) or True, reveal="R0")
    rf.sched = "sched:1:" + "a" * 64
    assert rf.commit_reveal("R1")
    assert writes[-1] == ["R1", rf.sched]
    # v0.8.1's join reader treats a second field that is not a join as no join.
    assert parse_join_field(writes[-1][1], 5) is None
    assert rf.commit_join("join:e00000005")
    assert writes[-1] == ["R1", "join:e00000005", rf.sched]
    rf.sched = None
    assert rf.commit_reveal("R2") and writes[-1] == ["R2", "join:e00000005"]
    assert ps.with_pointer(["R", "J", "sched:1:" + "a" * 64], ps.Pointer(2, "b" * 64)) == [
        "R", "J", "sched:2:" + "b" * 64]


# ── the transition rule ──────────────────────────────────────────────────────


def _sched(serial, acts, chain=CHAIN, digest=None):
    return ps.Schedule(serial=serial, chain=chain, activations=tuple(sorted(acts.items())),
                       digest=digest or f"{serial:064x}")


@pytest.mark.parametrize("held, new, code", [
    (_sched(2, {}), _sched(2, {"v6": 360 * 50}), "rollback"),
    (_sched(3, {}), _sched(2, {"v6": 360 * 50}), "rollback"),
    (_sched(1, {}), _sched(2, {"v6": 360 * 50 + 1}), "not_boundary"),
    (_sched(1, {}), _sched(2, {"v6": 360 * 50}, chain=OTHER_CHAIN), "wrong_chain"),
    # epoch 10, lead 2: the floor is block 360 x 12
    (_sched(1, {}), _sched(2, {"v6": 360 * 11}), "too_soon"),
    (_sched(1, {"v6": 360 * 11}), _sched(2, {"v6": 360 * 40}), "frozen"),     # move a near one
    (_sched(1, {"v6": 360 * 5}), _sched(2, {}), "frozen"),                    # remove a passed one
    (_sched(1, {"v6": 360 * 40}), _sched(2, {"v6": 360 * 11}), "too_soon"),   # pull one in
])
def test_transition_refusals(held, new, code):
    with pytest.raises(ps.ScheduleRefused) as e:
        ps.check_transition(held, new, epoch=10, chain=CHAIN, blocks_per_epoch=BPE,
                            lead_epochs=LEAD)
    assert e.value.code == code


def test_transition_allows_set_move_and_remove_beyond_the_lead():
    held = _sched(1, {"v5_1": 360 * 3, "v6": 360 * 40, "v7": 360 * 60})
    new = _sched(2, {"v5_1": 360 * 3, "v6": 360 * 12, "v8": 360 * 90})   # passed one repeated
    ps.check_transition(held, new, epoch=10, chain=CHAIN, blocks_per_epoch=BPE, lead_epochs=LEAD)


# ── the fold, on a fake chain ────────────────────────────────────────────────


class FakeChain:
    """The root's commitment fields as of each boundary block. `name(e, p)`
    makes the root's commitment name pointer p from epoch e on (None: write
    it without a schedule field)."""

    def __init__(self):
        self.changes: dict[int, tuple] = {}
        self.reads = 0
        self.down: set[int] = set()

    def name(self, epoch, pointer, *, join=True):
        fields = ["reveal:e%08d:%s" % (epoch, "0" * 64)] + (["join:x"] if join else [])
        if pointer is not None:
            fields.append(pointer.field() if isinstance(pointer, ps.Pointer) else pointer)
        self.changes[epoch * BPE] = tuple(fields)

    def query_module(self, module, name, params=None, block=None):
        assert (module, name) == ("Commitments", "CommitmentOf") and params[1] == ROOT
        if block in self.down:
            raise ConnectionError("node unreachable")
        self.reads += 1
        at = [b for b in self.changes if b <= block]
        return self.changes[max(at)] if at else ("reveal:e00000000:" + "0" * 64,)


class Store:
    """The public store: {digest: (bytes, signature)}, switchable off."""

    def __init__(self):
        self.objects, self.up = {}, True

    def put(self, payload, sig):
        d = ps.bundle_digest(payload, sig)
        self.objects[d] = (payload, sig)
        return d

    def fetch(self, digest):
        return self.objects.get(digest) if self.up else None


def tracker(chain, store, signers, *, start=10, state_path=None, shipped=(), cache=None):
    src = ps.DocumentSource(directories=shipped, cache_dir=cache, fetch=store.fetch)
    return ps.ScheduleTracker(
        chain=CHAIN, start_epoch=start, blocks_per_epoch=BPE, lead_epochs=LEAD,
        pointer_at=lambda e: ps.read_pointer(chain, 70, ROOT, e, BPE), document=src,
        state_path=state_path, signers=signers)


def publish(store, signer, serial, acts, chain=CHAIN):
    payload, sig = signed(signer, serial, acts, chain)
    return ps.Pointer(serial, store.put(payload, sig)), payload, sig


def test_adopted_at_the_first_boundary_that_shows_it(keys):
    owner, _, _, signers = keys
    chain, store = FakeChain(), Store()
    p1, _, _ = publish(store, owner, 1, {"v6": 360 * 30})
    chain.name(12, p1)
    t = tracker(chain, store, signers)
    view = t.advance(11)
    assert view.in_force is ps.EMPTY and view.certain_through == 13
    view = t.advance(28)
    assert view.in_force.pointer == p1 and view.adopted_epoch == 12
    assert not view.active("v6", 29) and view.active("v6", 30)
    assert [ev[2] for ev in t.state.events] == ["adopted"]


def test_a_late_named_document_cannot_reach_back(keys):
    """The lead counts from the chain's first sight, not from anything signed:
    the same document named one epoch later than it allowed is refused."""
    owner, _, _, signers = keys
    chain, store = FakeChain(), Store()
    p1, _, _ = publish(store, owner, 1, {"v6": 360 * 30})
    chain.name(29, p1)
    view = tracker(chain, store, signers).advance(40)
    assert view.in_force is ps.EMPTY
    assert not view.active("v6", 30)


def test_replay_of_an_older_serial_is_refused_while_it_stays_named(keys):
    owner, _, _, signers = keys
    chain, store = FakeChain(), Store()
    p1, _, _ = publish(store, owner, 1, {"v6": 360 * 30})
    p2, _, _ = publish(store, owner, 2, {"v6": 360 * 50})
    chain.name(12, p1)
    chain.name(15, p2)
    chain.name(60, p1)            # the root's hotkey names serial 1 again, after v6 at 50
    t = tracker(chain, store, signers)
    view = t.advance(80)
    assert view.in_force.pointer == p2
    assert not view.active("v6", 49) and view.active("v6", 50) and view.active("v6", 80)
    assert [ev[2] for ev in t.state.events] == ["adopted", "adopted", "refused:rollback"]


def test_absence_carries_and_a_passed_switch_never_moves(keys):
    owner, _, _, signers = keys
    chain, store = FakeChain(), Store()
    p1, _, _ = publish(store, owner, 1, {"v6": 360 * 30})
    p2, _, _ = publish(store, owner, 2, {"v6": 360 * 90})   # moves v6 after it passed
    p3, _, _ = publish(store, owner, 3, {"v6": 360 * 30, "v7": 360 * 90})
    chain.name(12, p1)
    chain.name(20, None)                  # an old root writes no schedule field
    chain.name(40, p2)
    chain.name(50, p3)
    t = tracker(chain, store, signers)
    view = t.advance(100)
    assert view.in_force.pointer == p3
    assert all(view.active("v6", e) for e in range(30, 101))
    assert [ev[2] for ev in t.state.events] == ["adopted", "refused:frozen", "adopted"]


def test_unserved_bytes_stop_the_fold_then_it_resumes_identically(keys, tmp_path):
    owner, _, _, signers = keys
    chain, store = FakeChain(), Store()
    p1, _, _ = publish(store, owner, 1, {"v6": 360 * 30})
    chain.name(12, p1)
    store.up = False
    t = tracker(chain, store, signers, state_path=str(tmp_path / "s.json"))
    with pytest.raises(ps.ScheduleUnresolved) as e:
        t.advance(20)
    assert e.value.epoch == 12
    view = t.view
    assert view.through == 11 and view.certain_through == 13
    assert view.active("v6", 13) is False
    with pytest.raises(ps.ScheduleUnresolved):
        view.active("v6", 14)
    store.up = True
    again = tracker(chain, store, signers, state_path=str(tmp_path / "s.json"))
    assert again.state.through == 11              # resumed from disk
    assert again.advance(20).in_force.pointer == p1


def test_another_signature_from_a_source_is_unserved_not_refused(keys):
    """A source that serves the document with a different signature does not
    serve what the pointer names: the fold waits, it never refuses on it."""
    owner, stranger, _, signers = keys
    chain, store = FakeChain(), Store()
    p1, payload, _ = publish(store, owner, 1, {"v6": 360 * 30})
    store.objects[p1.digest] = (payload, stranger.sign(payload))
    chain.name(12, p1)
    t = tracker(chain, store, signers)
    with pytest.raises(ps.ScheduleUnresolved):
        t.advance(20)
    assert t.state.events == []


@pytest.mark.parametrize("how", ["unsigned", "wrong_signer", "draft", "other_chain"])
def test_named_but_invalid_documents_are_refused_and_carried(keys, how):
    owner, stranger, _, signers = keys
    chain, store = FakeChain(), Store()
    payload = doc(0 if how == "draft" else 1, {"v6": 360 * 30},
                  chain=OTHER_CHAIN if how == "other_chain" else CHAIN)
    sig = {"unsigned": b"not a signature", "wrong_signer": stranger.sign(payload)}.get(
        how, owner.sign(payload))
    d = store.put(payload, sig)
    chain.name(12, ps.Pointer(1, d) if how != "draft" else f"sched:1:{d}")
    t = tracker(chain, store, signers)
    view = t.advance(40)
    assert view.in_force is ps.EMPTY and len(t.state.events) == 1
    assert t.state.events[0][2].startswith("refused:")


def test_a_pointer_whose_serial_is_not_the_documents_is_refused(keys):
    owner, _, _, signers = keys
    chain, store = FakeChain(), Store()
    p, _, _ = publish(store, owner, 1, {"v6": 360 * 30})
    chain.name(12, ps.Pointer(7, p.digest))
    t = tracker(chain, store, signers)
    assert t.advance(20).in_force is ps.EMPTY
    assert t.state.events[0][2] == "refused:serial_mismatch"


def test_no_commitment_at_all_is_unresolved_not_absent(keys):
    """A pruned endpoint can answer the storage default: the root always has
    a commitment, so none at all is an unreadable block, never a carry."""
    owner, _, _, signers = keys
    chain, store = FakeChain(), Store()
    chain.changes[14 * BPE] = ()
    t = tracker(chain, store, signers)
    with pytest.raises(ps.ScheduleUnresolved):
        t.advance(20)
    assert t.state.through == 13


def test_an_unreadable_chain_is_unresolved_not_absent(keys):
    owner, _, _, signers = keys
    chain, store = FakeChain(), Store()
    chain.down = {14 * BPE}
    t = tracker(chain, store, signers)
    with pytest.raises(ps.ScheduleUnresolved):
        t.advance(20)
    assert t.state.through == 13


# ── two nodes: a validator and a TD derive the same schedule ─────────────────


def _history(owner, store):
    """Every case at once: adoption, replay, absence, a refused move of a
    passed switch, an unsigned pointer, then a legitimate later move."""
    chain = FakeChain()
    p1, _, _ = publish(store, owner, 1, {"v6": 360 * 30})
    p2, _, _ = publish(store, owner, 2, {"v6": 360 * 26})        # moved earlier, in time
    p3, _, _ = publish(store, owner, 3, {"v6": 360 * 70})        # moves a passed one: refused
    p4, _, _ = publish(store, owner, 4, {"v6": 360 * 26, "v7": 360 * 80})
    p5, _, _ = publish(store, owner, 5, {"v6": 360 * 26, "v7": 360 * 95})
    junk = store.put(doc(9, {"v6": 360 * 200}), b"garbage")
    for epoch, p in ((12, p1), (18, p2), (22, None), (24, p1), (33, p3), (40, p4),
                     (45, ps.Pointer(9, junk)), (50, p4), (60, p5), (70, None)):
        chain.name(epoch, p)
    return chain


def test_validator_and_td_derive_the_same_schedule_at_every_boundary(keys, tmp_path):
    owner, _, _, signers = keys
    store = Store()
    chain = _history(owner, store)
    shipped = tmp_path / "shipped"
    shipped.mkdir()
    for digest, (payload, sig) in list(store.objects.items())[:2]:   # the release has two
        (shipped / f"{digest}.json").write_bytes(payload)
        (shipped / f"{digest}.json.sig").write_bytes(sig)

    validator = tracker(chain, store, signers, shipped=[shipped],
                        state_path=str(tmp_path / "v.json"), cache=str(tmp_path / "v.d"))
    td_store = Store()
    td_store.objects = store.objects
    seen = {}
    for e in range(10, 101):
        seen[e] = validator.advance(e)
    # The TD starts late, restarts twice, and its store is down for a while:
    # it catches up and must agree at every boundary.
    td_state = str(tmp_path / "td" / "schedule.json")
    td = None
    for e in range(10, 101):
        if e in (10, 47, 73):
            td = tracker(chain, td_store, signers, state_path=td_state,
                         cache=str(tmp_path / "td" / "schedule.d"))
        td_store.up = not 41 <= e <= 44
        try:
            view = td.advance(e)
        except ps.ScheduleUnresolved:
            view = td.view
            assert view.certain_through >= e - 1     # it still decides the epoch before
            continue
        v = seen[e]
        assert view.in_force.digest == v.in_force.digest, e
        for name in ("v6", "v7"):
            for at in range(10, view.certain_through + 1):
                assert view.active(name, at) == v.active(name, at), (e, name, at)
    final = td.advance(100)
    assert final.in_force.pointer == seen[100].in_force.pointer
    assert final.in_force.as_dict() == {"v6": 360 * 26, "v7": 360 * 95}
    outcomes = [ev[2] for ev in validator.state.events]
    assert outcomes == ["adopted", "adopted", "refused:rollback", "refused:frozen", "adopted",
                        "refused:signature_malformed", "adopted"]


def test_a_fresh_node_folds_the_same_as_one_that_ran_throughout(keys, tmp_path):
    owner, _, _, signers = keys
    store = Store()
    chain = _history(owner, store)
    a = tracker(chain, store, signers)
    for e in range(10, 101):
        a.advance(e)
    b = tracker(chain, store, signers)
    b.advance(100)
    assert a.state.in_force == b.state.in_force and a.state.events == b.state.events
    assert a.state.adopted_epoch == b.state.adopted_epoch == 60


def test_kept_state_for_other_parameters_or_tampered_is_refolded(keys, tmp_path):
    owner, _, _, signers = keys
    store = Store()
    chain = _history(owner, store)
    path = tmp_path / "s.json"
    tracker(chain, store, signers, state_path=str(path)).advance(100)
    d = json.loads(path.read_text())
    d["payload"] = ps._b64(doc(5, {"v6": 360 * 999}))           # not what was signed
    path.write_text(json.dumps(d))
    t = tracker(chain, store, signers, state_path=str(path))
    assert t.state.through == 9                                  # back to the start
    assert t.advance(100).in_force.as_dict() == {"v6": 360 * 26, "v7": 360 * 95}
    other = tracker(chain, store, signers, state_path=str(path), start=20)
    assert other.state.through == 19


# ── the root's choice ────────────────────────────────────────────────────────


def test_the_root_names_the_newest_valid_served_document(keys):
    owner, _, _, signers = keys
    chain, store = FakeChain(), Store()
    p1, a1, s1 = publish(store, owner, 1, {"v6": 360 * 30})
    _, a2, s2 = publish(store, owner, 2, {"v6": 360 * 13})       # too soon at epoch 12
    held = {p1.digest: (a1, s1), ps.bundle_digest(a2, s2): (a2, s2)}
    t = tracker(chain, store, signers)
    t.advance(10)
    assert ps.root_pointer(t, epoch=10, documents=held, confirm=lambda d: True) == p1
    assert ps.root_pointer(t, epoch=10, documents=held, confirm=lambda d: False) is None
    chain.name(11, p1)
    t.advance(11)
    assert ps.root_pointer(t, epoch=11, documents=held, confirm=lambda d: True) == p1
    assert ps.root_pointer(t, epoch=5, documents=held, confirm=lambda d: True) is None


# ── config: compiled blocks take precedence ──────────────────────────────────


def test_switch_block_compiled_precedence_and_conflict(monkeypatch):
    monkeypatch.setattr(config, "SCHEDULE_SWITCHES", {"v6": 7200, "v7": None})
    view = ps.ScheduleView(in_force=_sched(1, {"v6": 7200, "v7": 9000}), through=10,
                           lead_epochs=LEAD, blocks_per_epoch=BPE)
    fin = config.FINNEY_GENESIS_HASH
    assert config.switch_block("v6", fin, view) == 7200
    assert config.switch_block("v7", fin, view) == 9000
    assert config.switch_block("v7", fin, None) is None
    assert config.switch_block("v6", CHAIN, view) == 7200       # off finney: the schedule's
    clash = ps.ScheduleView(in_force=_sched(1, {"v6": 7560}), through=10, lead_epochs=LEAD,
                            blocks_per_epoch=BPE)
    with pytest.raises(ps.ScheduleConflict):
        config.switch_block("v6", fin, clash)
    with pytest.raises(KeyError):
        config.switch_block("v9", fin, view)


def test_schedule_from_block_follows_the_activation_rule(monkeypatch):
    fin = config.FINNEY_GENESIS_HASH
    monkeypatch.setattr(config, "SCHEDULE_FROM_BLOCK_ENV", None)
    monkeypatch.setattr(config, "SCHEDULE_FROM_BLOCK_MALFORMED", None)
    assert config.schedule_from_block(fin) == config.FINNEY_SCHEDULE_FROM_BLOCK
    monkeypatch.setattr(config, "SCHEDULE_FROM_BLOCK_ENV", 3600)
    assert config.schedule_from_block(CHAIN) == 3600
    if config.FINNEY_SCHEDULE_FROM_BLOCK != 3600:
        with pytest.raises(ValueError):
            config.schedule_from_block(fin)
    monkeypatch.setattr(config, "SCHEDULE_FROM_BLOCK_ENV", None)
    monkeypatch.setattr(config, "SCHEDULE_FROM_BLOCK_MALFORMED", "12x")
    with pytest.raises(ValueError):
        config.schedule_from_block(CHAIN)


def test_the_schedule_is_off_on_finney_in_this_release():
    assert config.FINNEY_SCHEDULE_FROM_BLOCK is None
    assert config.SCHEDULE_LEAD_EPOCHS == 2


def test_shipped_schedules_verify_against_the_pinned_signers():
    for digest, (payload, sig) in ps.shipped_documents().items():
        got = ps.verify(payload, sig)
        assert got.digest == digest and got.chain == config.FINNEY_GENESIS_HASH


# ── the release check ────────────────────────────────────────────────────────


def _release_tree(tmp_path, files=(), compose_env=""):
    root = tmp_path / "repo"
    (root / "fugal_subnet" / "signed_schedule").mkdir(parents=True)
    (root / "deploy" / "dstack").mkdir(parents=True)
    (root / "deploy" / "dstack" / "docker-compose.yaml").write_text(
        "services:\n  miner:\n    environment:\n      - FUGAL_NETWORK=finney\n" + compose_env)
    for name, data in files:
        (root / "fugal_subnet" / "signed_schedule" / name).write_bytes(data)
    return root


def _check(root, signers, switches=None):
    import importlib.util
    import types
    from pathlib import Path

    path = Path(config.__file__).resolve().parents[1] / "scripts" / "check_safety_invariants.py"
    spec = importlib.util.spec_from_file_location("check_safety_invariants_for_tests", path)
    csi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(csi)
    cfg = types.SimpleNamespace(FINNEY_GENESIS_HASH=config.FINNEY_GENESIS_HASH,
                                SCHEDULE_SWITCHES=switches or {})
    errors = []
    csi.check_protocol_schedule(errors, root, cfg, signers=signers)
    return errors


def test_release_check_accepts_a_good_shipped_schedule(keys, tmp_path):
    owner, _, _, signers = keys
    payload, sig = signed(owner, 1, {"v6": 7200}, chain=config.FINNEY_GENESIS_HASH)
    d = ps.bundle_digest(payload, sig)
    root = _release_tree(tmp_path, [(f"{d}.json", payload), (f"{d}.json.sig", sig)])
    assert _check(root, signers, {"v6": None}) == []
    assert _check(root, signers, {"v6": 7200}) == []
    assert any("ScheduleConflict" in e for e in _check(root, signers, {"v6": 7560}))


@pytest.mark.parametrize("case", ["misnamed", "lonely", "wrong_signer", "testnet", "compose"])
def test_release_check_refusals(keys, tmp_path, case):
    owner, stranger, _, signers = keys
    chain = CHAIN if case == "testnet" else config.FINNEY_GENESIS_HASH
    payload, sig = signed(stranger if case == "wrong_signer" else owner, 1, {"v6": 7200},
                          chain=chain)
    d = ps.bundle_digest(payload, sig)
    files = {"misnamed": [("a" * 64 + ".json", payload), ("a" * 64 + ".json.sig", sig)],
             "lonely": [(f"{d}.json", payload)]}.get(
        case, [(f"{d}.json", payload), (f"{d}.json.sig", sig)])
    root = _release_tree(tmp_path, files, compose_env=(
        "      - FUGAL_SCHEDULE_FROM_BLOCK=3600\n" if case == "compose" else ""))
    assert _check(root, signers), case
