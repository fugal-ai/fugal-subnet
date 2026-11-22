"""Protocol v4: how the published set moves between nodes (fugal_subnet/memo_sync.py).

The pieces the loops (tests/test_v4_loop.py, tests/test_v4_published_loop.py,
tests/test_v4_flip.py, tests/test_v4_seat_loop.py) run, each on its own: the
snapshot's bytes and their strict reading, the root's publication of a
snapshot before its commitment, and the rule every node's set follows
(memo_sync.ChainSet): the root's commitments read at each boundary, nothing
moving while the root is down, a backlog folded when first named, a stray
older commitment ignored, the state kept across a restart, a snapshot anchor
for a node without state, an ALARM (a TD stops) on a reveal that contradicts
the chain, and the TD's plan per epoch.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from fugal_subnet import memo_sync, memo_v4, reveal_publisher
from fugal_subnet.benchmarks.slicer import epoch_id_for_block
from tests import v4_world

SEED_AT = 8


@pytest.fixture
def world(tmp_path, monkeypatch):
    pool = v4_world.fixture_pool()
    seed = v4_world.fixture_seed(pool, SEED_AT, v4_world.fixture_unpublished(pool, 4))
    v4_world.install_seed(tmp_path, monkeypatch.setattr, seed)
    return pool, seed


def _reserve(pool, seed):
    return sorted(q for ids in seed["unpublished"].values() for q in ids)


def _reveal(k, ids, *, named=None):
    r = {"epoch_id": epoch_id_for_block(k), "protocol_version": 4,
         "questions": [{"question_id": q} for q in sorted(ids)]}
    if named is not None:
        r["memo_v4"] = {"through": named.through, "published_digest": named.digest}
    return r


def _chain(pool, seed, *, until, empty=()):
    """Root reveals SEED_AT + 1 .. until, each publishing two reserve
    questions (none for an epoch in `empty`) and naming the view through
    k - 2 the reveals before it fold to."""
    reserve = _reserve(pool, seed)
    pub = memo_v4.seed_published(pool, seed)
    out = {}
    for k in range(SEED_AT + 1, until + 1):
        named = pub.view(k - 2) if k - 2 >= SEED_AT else None
        if k in empty:
            pub = memo_v4.fold(pub, k, ())
            continue
        ids = reserve[2 * (k - SEED_AT - 1): 2 * (k - SEED_AT)]
        out[k] = _reveal(k, ids, named=named)
        pub = memo_v4.fold_reveal(pub, out[k])
    return out, pub


# --- the snapshot ---------------------------------------------------------------------------


def test_a_snapshot_is_the_canonical_document_its_digest_names(world):
    pool, seed = world
    pub = memo_v4.seed_published(pool, seed)
    raw = memo_v4.snapshot_bytes(pub)
    assert hashlib.sha256(raw).hexdigest() == pub.digest
    got = memo_v4.published_from_snapshot(raw, pub.digest, through=SEED_AT,
                                          pool_ids={q["question_id"] for q in pool})
    assert got == pub and got.digest == pub.digest
    assert memo_sync.snapshot_object(7, pub.digest) == f"7/published/{pub.digest}.json"
    with pytest.raises(ValueError):
        memo_sync.snapshot_object(7, "not-a-digest")


def _refused(raw, digest, **kw):
    with pytest.raises(ValueError) as e:
        memo_v4.published_from_snapshot(raw, digest, **kw)
    return str(e.value)


def test_hostile_snapshot_bytes_are_refused(world):
    pool, seed = world
    pub = memo_v4.seed_published(pool, seed)
    raw = memo_v4.snapshot_bytes(pub)
    ids = {q["question_id"] for q in pool}
    # Other bytes than the digest names.
    assert "do not hash" in _refused(raw + b" ", pub.digest)
    # Too large to read, whatever they hash to.
    big = b"x" * (memo_v4.MAX_SNAPSHOT_BYTES + 1)
    assert "at most" in _refused(big, hashlib.sha256(big).hexdigest())
    # Not JSON, and JSON that is not the document.
    for bad in (b"{not json", b"[]", json.dumps({"format": "x"}).encode()):
        assert _refused(bad, hashlib.sha256(bad).hexdigest())
    # The document, but not in canonical form.
    loose = json.dumps(pub.to_doc(), indent=1).encode()
    assert "canonical" in _refused(loose, hashlib.sha256(loose).hexdigest())
    # A question under two epochs, and a list with a question twice.
    doc = pub.to_doc()
    q = doc["first"][str(SEED_AT)][0]
    twice = dict(doc, first={**doc["first"], str(SEED_AT - 1): [q]})
    raw2 = json.dumps(twice, sort_keys=True, separators=(",", ":")).encode()
    assert "once" in _refused(raw2, hashlib.sha256(raw2).hexdigest())
    dup = dict(doc, first={str(SEED_AT): [q, q]})
    raw3 = json.dumps(dup, sort_keys=True, separators=(",", ":")).encode()
    assert "once" in _refused(raw3, hashlib.sha256(raw3).hexdigest())
    # A question this pool does not hold.
    alien = memo_v4.Published({**pub.first, "zz-not-in-pool": SEED_AT}, SEED_AT)
    raw4 = memo_v4.snapshot_bytes(alien)
    assert "not in the pool" in _refused(raw4, alien.digest, pool_ids=ids)
    # Another view than the reveal names.
    assert "folded through" in _refused(raw, pub.digest, through=SEED_AT + 1)


def _wrong_types(doc, q):
    """Snapshot documents with a value of the wrong type where an id list,
    an id, an epoch key or the integer `through` is expected."""
    e = str(SEED_AT)
    for bad in (None, 1, 1.5, [], {}, [q], True):
        yield dict(doc, first={e: [q, bad]})
        yield dict(doc, first={e: [bad]})
        if not isinstance(bad, list):                            # a list of ids is an id list
            yield dict(doc, first={e: bad})
        if type(bad) is not int:                                 # an int is a `through`
            yield dict(doc, through=bad)
    yield dict(doc, first={"١٢": [q]})                 # digits, not ASCII ones


def test_a_snapshot_with_a_value_of_the_wrong_type_is_refused_as_a_value_error(world):
    pool, seed = world
    pub = memo_v4.seed_published(pool, seed)
    doc = pub.to_doc()
    q = doc["first"][str(SEED_AT)][0]
    cases = [json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
             for d in _wrong_types(doc, q)]
    cases += [b"[" * 5000 + b"]" * 5000, b'{"first":' * 3000]
    for raw in cases:
        _refused(raw, hashlib.sha256(raw).hexdigest())             # ValueError, nothing else
    for d in _wrong_types(doc, q):
        with pytest.raises(ValueError):
            memo_v4.published_from_doc(d)


def test_a_named_snapshot_or_a_kept_state_of_the_wrong_types_is_set_aside(tmp_path, world):
    pool, seed = world
    epochs = range(SEED_AT + 1, SEED_AT + 9)
    root = _Root(pool, seed, {c + 2: c for c in epochs}, _published(pool, seed, epochs))
    e, c = SEED_AT + 10, SEED_AT + 8
    q = sorted(memo_v4.seed_published(pool, seed).first)[0]
    for bad in (None, [q], 3):
        doc = _rule(root, c).view(c - 2).to_doc()
        doc["first"][str(SEED_AT)] = [q, bad]
        raw = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        r = dict(root.reveals[c], memo_v4={"through": c - 2, "published_digest": digest})
        root.reveals[c], root.raw[c] = r, json.dumps(r, sort_keys=True).encode()
        root.sha[c] = hashlib.sha256(root.raw[c]).hexdigest()
        fresh = _set(root, snapshots={digest: raw})
        fresh.view_for(e)                                        # the seed, replayed: no TypeError
        assert fresh.view(e).view(e - 2).digest == _rule(root, e).view(e - 2).digest
        # A kept state holding the same: set aside, never trusted.
        path = tmp_path / "held.json"
        path.write_text(json.dumps({"format": memo_sync.HELD_FORMAT, "set": doc, "named": c,
                                    "at": c - 2 + memo_v4.READ_LAG}))
        assert _set(root, tmp_path).at is None


def test_a_reveal_names_its_view_strictly():
    assert memo_sync.named_view({"memo_v4": {"through": 3, "published_digest": "a" * 64}}) == (
        3, "a" * 64)
    for bad in ({}, {"memo_v4": {}}, {"memo_v4": {"through": "3", "published_digest": "a" * 64}},
                {"memo_v4": {"through": 3, "published_digest": "A" * 64}}, None):
        assert memo_sync.named_view(bad) is None


# --- the root's publication -------------------------------------------------------------------


class _Store:
    def __init__(self, serve=None):
        self.objects, self.serve = {}, serve

    def put(self, key, data):
        self.objects.setdefault(key, bytes(data))

    def get(self, key, limit):
        return (self.serve or self.objects)[key][:limit + 1]


def test_the_root_publishes_the_view_a_reveal_names_and_only_that(world):
    pool, seed = world
    reveals, pub = _chain(pool, seed, until=SEED_AT + 4)
    store = _Store()
    hook = memo_sync.SnapshotPublisher(netuid=7, store=store, held=lambda: pub)
    r = reveals[SEED_AT + 4]
    assert hook(r["epoch_id"], r)
    key = memo_sync.snapshot_object(7, r["memo_v4"]["published_digest"])
    assert hashlib.sha256(store.objects[key]).hexdigest() == r["memo_v4"]["published_digest"]
    # A reveal naming a set this node does not hold (a digest of a set that
    # leaves out a published question): nothing is published.
    short = memo_v4.Published({q: e for q, e in pub.view(SEED_AT + 2).first.items()
                               if q != sorted(pub.first)[0]}, SEED_AT + 2)
    lying = dict(r, memo_v4={"through": SEED_AT + 2, "published_digest": short.digest})
    assert not hook(r["epoch_id"], lying)
    assert memo_sync.snapshot_object(7, short.digest) not in store.objects
    # No named view (a v3 reveal), or a store that serves other bytes: none.
    assert not hook("e00000001", {"epoch_id": "e00000001"})
    bad = memo_sync.SnapshotPublisher(netuid=7, store=_Store(serve={key: b"other"}),
                                      held=lambda: pub)
    assert not bad(r["epoch_id"], r)


def test_publish_due_reveals_puts_the_snapshot_before_the_commitment(tmp_path):
    epochs = tmp_path / "epochs"
    events = []

    class Up:
        def upload(self, path, key):
            events.append(("upload", key))
            return "gs://b/" + key

    def write(k, body):
        d = epochs / epoch_id_for_block(k)
        d.mkdir(parents=True)
        (d / "reveal.json").write_text(json.dumps({"epoch_id": epoch_id_for_block(k), **body}))

    write(10, {"memo_v4": {"through": 8, "published_digest": "c" * 64}})
    write(11, {})
    reveal_publisher.publish_due_reveals(
        str(epochs), 12, 1, 7, Up(), commit=lambda d: events.append(("commit", d)) or True,
        snapshot=lambda eid, r: events.append(("snapshot", eid)) or True)
    assert [e[0] for e in events] == ["upload", "upload", "snapshot", "commit"]
    assert events[2] == ("snapshot", "e00000010")          # only the reveal that names a view
    assert events[3][1].startswith("reveal:e00000011:")
    # Nothing due: nothing is committed (no reveal names a newer epoch).
    events.clear()
    reveal_publisher.publish_due_reveals(
        str(epochs), 12, 1, 7, Up(), commit=lambda d: events.append(("commit", d)) or True)
    assert events == []


# --- the rule: the set advances by the root's commitments read at boundaries ------------------


class _Root:
    """The root as the chain and its store show it: reveal c's bytes, and
    which reveal its commitment names at each boundary (`at`, boundary ->
    reveal epoch; absent: no commitment newer than the last). Each reveal
    names the view its epoch used, by the rule (`_rule`)."""

    def __init__(self, pool, seed, at: dict, published: dict):
        self.pool, self.seed, self.at = pool, seed, dict(at)
        self.reveals, self.raw = {}, {}
        for c in sorted(published):
            view = _rule(self, c).view(c - 2)
            r = _reveal(c, published[c], named=view)
            self.reveals[c] = r
            self.raw[c] = json.dumps(r, sort_keys=True).encode()
        self.sha = {c: hashlib.sha256(raw).hexdigest() for c, raw in self.raw.items()}
        self.reads, self.fetched = [], []

    def commitment(self, b):
        """The newest commitment at boundary b (the last named at or before it)."""
        named = [c for bb, c in sorted(self.at.items()) if bb <= b]
        if not named:
            return None
        c = named[-1]
        return reveal_publisher.commitment_for(epoch_id_for_block(c), self.sha[c])

    def read_at(self, b):
        self.reads.append(b)
        return self.commitment(b)

    def reveal_bytes(self, eid, sha):
        self.fetched.append(eid)
        c = memo_v4.epoch_index(eid)
        return self.raw.get(c)


def _rule(root, epoch):
    """The set as of `epoch`, by the rule, computed directly."""
    pub = memo_v4.seed_published(root.pool, root.seed)
    named = pub.through
    for b in range(pub.through + 3, epoch + 1):
        cs = [c for bb, c in sorted(root.at.items()) if bb <= b]
        ids = set()
        if cs and cs[-1] > named and cs[-1] in root.reveals:
            named = cs[-1]
            ids = memo_v4.published_ids(root.reveals[named])
        pub = memo_v4.fold(pub, b - 2, ids)
    return memo_v4.fold(pub.view(epoch - 2), epoch - 1, ())


def _published(pool, seed, epochs):
    reserve = _reserve(pool, seed)
    return {c: reserve[2 * (c - SEED_AT - 1): 2 * (c - SEED_AT)] for c in epochs}


def _set(root, tmp_path=None, *, snapshots=None, refuse=False):
    return memo_sync.ChainSet(
        pool=root.pool, read_at=root.read_at, reveal_bytes=root.reveal_bytes,
        snapshot=(lambda d: (snapshots or {}).get(d)) if snapshots is not None else None,
        path=None if tmp_path is None else tmp_path / "held.json", refuse=refuse)


def test_in_steady_state_the_set_is_the_roots_reveals_through_e_minus_2(world):
    pool, seed = world
    epochs = range(SEED_AT + 1, SEED_AT + 9)
    root = _Root(pool, seed, {c + 2: c for c in epochs}, _published(pool, seed, epochs))
    cs = _set(root)
    for e in range(SEED_AT + 3, SEED_AT + 11):
        got = cs.view_for(e)
        plain = memo_v4.seed_published(pool, seed)
        for c in range(SEED_AT + 1, e - 1):
            plain = memo_v4.fold_reveal(plain, root.reveals[c])
        assert got.view(e - 2).digest == plain.view(e - 2).digest == _rule(root, e).view(
            e - 2).digest, e
        assert got.through == e - 1
    # The view of an epoch is the same whatever has been read since.
    assert cs.view(SEED_AT + 5).digest == _rule(root, SEED_AT + 5).digest


def test_while_the_root_is_down_the_set_holds_and_its_backlog_is_folded_when_named(world):
    pool, seed = world
    s = SEED_AT
    # The root commits on time to s + 4, is down for s + 5 .. s + 8 (no
    # commitment), and on its return its reveal of s + 4 (written before it
    # went down) is committed during s + 9 and named at s + 10; s + 9's at
    # s + 11. Its reveal of s + 3 is committed only behind s + 4's (never
    # at a boundary of its own) and is never folded.
    at = {s + 3: s + 1, s + 4: s + 2, s + 10: s + 4, s + 11: s + 9}
    root = _Root(pool, seed, at, _published(pool, seed, [s + 1, s + 2, s + 3, s + 4, s + 9]))
    cs = _set(root)
    views = {e: cs.view_for(e) for e in range(s + 3, s + 13)}
    for e in views:
        assert views[e].digest == _rule(root, e).digest, e
    held = [dict(views[e].view(e - 2).first) for e in range(s + 5, s + 10)]
    assert all(h == held[0] for h in held)                    # nothing moves while down
    late = set(memo_v4.published_ids(root.reveals[s + 4]))
    assert late <= set(views[s + 10].view(s + 8).first)        # folded when first named
    assert {views[s + 10].view(s + 8).first[q] for q in late} == {s + 8}
    never = set(memo_v4.published_ids(root.reveals[s + 3])) - late
    assert never and not never & set(views[s + 12].view(s + 10).first)


def test_an_older_commitment_is_ignored_and_a_named_reveal_that_cannot_be_had_stops_the_set(world):
    pool, seed = world
    s = SEED_AT
    root = _Root(pool, seed, {s + 3: s + 1, s + 4: s + 2, s + 5: s + 1, s + 6: s + 4},
                 _published(pool, seed, [s + 1, s + 2, s + 4]))
    cs = _set(root)
    assert cs.view_for(s + 6).digest == _rule(root, s + 6).digest
    assert root.fetched.count(epoch_id_for_block(s + 1)) == 1   # the stray is not even read
    gone = _Root(pool, seed, {s + 3: s + 1}, _published(pool, seed, [s + 1]))
    gone.raw[s + 1] = b"other bytes"
    with pytest.raises(memo_sync.PublishedUnavailable, match="not to be had"):
        _set(gone).view_for(s + 3)

    def broken(b):
        raise ConnectionError("pinned read failed")

    bad = memo_sync.ChainSet(pool=pool, read_at=broken, reveal_bytes=root.reveal_bytes)
    with pytest.raises(ConnectionError):
        bad.advance_to(s + 4)


def test_the_state_is_kept_and_a_restart_reads_only_the_boundaries_since(tmp_path, world):
    pool, seed = world
    epochs = range(SEED_AT + 1, SEED_AT + 9)
    root = _Root(pool, seed, {c + 2: c for c in epochs}, _published(pool, seed, epochs))
    cs = _set(root, tmp_path)
    cs.view_for(SEED_AT + 6)
    cs.save()
    root.reads.clear()
    again = _set(root, tmp_path)
    assert again.at == SEED_AT + 6 and again.view(SEED_AT + 6).digest == cs.view(SEED_AT + 6).digest
    assert again.view_for(SEED_AT + 9).digest == _rule(root, SEED_AT + 9).digest
    assert root.reads == [SEED_AT + 7, SEED_AT + 8, SEED_AT + 9]
    (tmp_path / "held.json").write_text("{}")
    assert _set(root, tmp_path).at is None                     # set aside, never trusted


def test_a_node_without_state_anchors_on_the_named_snapshot_and_refuses_bad_bytes(world):
    pool, seed = world
    epochs = range(SEED_AT + 1, SEED_AT + 9)
    root = _Root(pool, seed, {c + 2: c for c in epochs}, _published(pool, seed, epochs))
    e = SEED_AT + 10
    c = SEED_AT + 8
    named = root.reveals[c]["memo_v4"]
    good = memo_v4.snapshot_bytes(_rule(root, c).view(c - 2))
    assert hashlib.sha256(good).hexdigest() == named["published_digest"]
    fresh = _set(root, snapshots={named["published_digest"]: good})
    root.reads.clear()
    assert fresh.view_for(e).digest == _rule(root, e).digest
    assert sorted(set(root.reads)) == [c, c + 1, c + 2]        # not every boundary since the seed
    # Tampered bytes are not taken: the node replays from the seed instead.
    bad = _set(root, snapshots={named["published_digest"]: good.replace(b'"first"', b'"First"')})
    assert bad.view_for(e).digest == _rule(root, e).digest


def test_a_reveal_that_contradicts_the_chain_is_an_alarm_and_stops_a_td(world):
    pool, seed = world
    epochs = range(SEED_AT + 1, SEED_AT + 7)
    root = _Root(pool, seed, {c + 2: c for c in epochs}, _published(pool, seed, epochs))
    c = SEED_AT + 6
    lying = dict(root.reveals[c], memo_v4={"through": c - 2, "published_digest": "e" * 64})
    root.reveals[c], root.raw[c] = lying, json.dumps(lying, sort_keys=True).encode()
    root.sha[c] = hashlib.sha256(root.raw[c]).hexdigest()
    follower = _set(root)
    follower.view_for(c + 2)
    assert len(follower.findings) == 1 and "names the published set" in follower.findings[0]
    with pytest.raises(memo_sync.PublishedUnavailable, match="names the published set"):
        _set(root, refuse=True).view_for(c + 2)


def test_the_tds_plan_is_built_per_epoch_on_the_chain_view(world):
    from fugal_subnet.tee import td_v4

    pool, seed = world
    epochs = range(SEED_AT + 1, SEED_AT + 7)
    root = _Root(pool, seed, {c + 2: c for c in epochs}, _published(pool, seed, epochs))
    s = td_v4.V4Session(first_epoch=SEED_AT + 1, pool=pool, fetch_listing=lambda m: b"",
                        read_commitment=lambda e: None, menu_bytes=lambda h: None,
                        list_bytes=lambda s, h: None, published=_set(root, refuse=True))
    plan = s.plan_for(SEED_AT + 8)
    assert plan.pool_index.epoch == SEED_AT + 8
    assert plan.pool_index.digest == _rule(root, SEED_AT + 8).view(SEED_AT + 6).digest
    gone = _Root(pool, seed, {SEED_AT + 3: SEED_AT + 1}, _published(pool, seed, [SEED_AT + 1]))
    gone.raw[SEED_AT + 1] = b"x"
    s2 = td_v4.V4Session(first_epoch=SEED_AT + 1, pool=pool, fetch_listing=lambda m: b"",
                         read_commitment=lambda e: None, menu_bytes=lambda h: None,
                         list_bytes=lambda s, h: None, published=_set(gone, refuse=True))
    with pytest.raises(memo_sync.PublishedUnavailable):
        s2.plan_for(SEED_AT + 8)


# --- the attack suite's cases (fugal_subnet/attacks/run_miner_attacks.py) -----------------------


def test_the_published_set_attack_cases_are_held_and_each_rule_is_what_holds_it(monkeypatch):
    from fugal_subnet.attacks import run_miner_attacks as atk

    for name, fn, _gain, _inv in atk.PUBLISHED_ATTACKS:
        held, detail = fn()
        assert held, (name, detail)
    # Each held by its rule: without the digest check another set's bytes
    # read; without the pool check foreign ids do; without the audit a
    # reveal that leaves out published questions is drawn on.
    real = memo_v4.published_from_snapshot

    def no_digest(raw, digest, **kw):
        return real(raw, hashlib.sha256(raw).hexdigest(), **kw)

    monkeypatch.setattr(memo_v4, "published_from_snapshot", no_digest)
    assert not atk.atk_published_wrong_digest()[0]
    monkeypatch.setattr(memo_v4, "published_from_snapshot",
                        lambda raw, digest, pool_ids=None, through=None: real(
                            raw, digest, through=through))
    assert not atk.atk_published_ids_outside_the_pool()[0]
    monkeypatch.setattr(memo_v4, "published_from_snapshot", real)
    monkeypatch.setattr(memo_sync, "check_named", lambda published, reveal: None)
    assert not atk.atk_published_reveal_omits_published_questions()[0]


# --- an endpoint that keeps only recent state --------------------------------------------------


BPE = 360


class _Pruning:
    """`root.read_at` through an endpoint that keeps the state of the last
    `keep` blocks (None: every block): an older pinned read raises as a
    pruning node answers. `now` is the epoch the chain is in (30 blocks past
    its boundary)."""

    def __init__(self, root, keep):
        self.root, self.keep, self.now, self.pruned = root, keep, 0, []

    def read_at(self, b):
        age = self.now * BPE + 30 - b * BPE
        if self.keep is not None and age > self.keep:
            self.pruned.append(b)
            raise ConnectionError(f"State discarded for block {b * BPE}: the block is too old")
        return self.root.read_at(b)


def _snapshots(root, epochs):
    out = {}
    for c in epochs:
        named = root.reveals[c]["memo_v4"]
        out[named["published_digest"]] = memo_v4.snapshot_bytes(_rule(root, c).view(c - 2))
    return out


def _outage(tmp_path, world, missed, keep):
    """A node in step with the chain through `first`, its state kept, back
    after `missed` epochs on an endpoint keeping `keep` blocks: the views it
    reaches in its first three epochs back (each a fresh process), and the
    views the rule gives."""
    pool, seed = world
    epochs = range(SEED_AT + 1, SEED_AT + missed + 16)
    root = _Root(pool, seed, {c + 2: c for c in epochs}, _published(pool, seed, epochs))
    snaps = _snapshots(root, epochs)
    chain = _Pruning(root, keep)
    path = tmp_path / f"held-{missed}-{keep}.json"

    def node(snapshot=snaps.get):
        return memo_sync.ChainSet(pool=pool, read_at=chain.read_at, reveal_bytes=root.reveal_bytes,
                                  snapshot=snapshot, path=path)

    first = SEED_AT + 5
    cs = node(None)                     # from the seed (the world's first reveals predate it)
    for e in range(SEED_AT + 3, first + 1):
        chain.now = e
        cs.view_for(e)
    cs.save()
    got, want = {}, {}
    for e in range(first + missed + 1, first + missed + 4):
        chain.now = e
        n = node()
        got[e] = n.view_for(e).digest
        n.save()
        want[e] = _rule(root, e).digest
    return got, want, chain


@pytest.mark.parametrize("missed", [2, 10, 60])
def test_a_node_back_after_missed_epochs_on_a_pruning_endpoint_re_anchors_and_agrees(
        tmp_path, world, missed):
    keep = memo_sync.ANCHOR_HISTORY_EPOCHS * BPE + 60
    got, want, chain = _outage(tmp_path, world, missed, keep)
    assert got == want
    # Two missed epochs are within what the endpoint keeps (a plain replay);
    # past that the node meets a pruned read and re-anchors.
    assert bool(chain.pruned) == (missed > 2)


def test_with_full_history_nothing_re_anchors_and_the_views_are_the_same(tmp_path, world):
    full, want, chain = _outage(tmp_path, world, 10, None)
    assert full == want and chain.pruned == []
    pruned, _want, _chain = _outage(tmp_path, world, 10, memo_sync.ANCHOR_HISTORY_EPOCHS * BPE + 60)
    assert pruned == full


def test_an_endpoint_without_the_history_to_anchor_says_so_and_a_transient_error_still_raises(
        tmp_path, world):
    with pytest.raises(memo_sync.PublishedUnavailable, match="archive endpoint"):
        _outage(tmp_path, world, 10, 256)
    assert memo_sync.is_pruned(ConnectionError("State discarded for 0xab"))
    assert not memo_sync.is_pruned(ConnectionError("timed out"))

    class StateDiscardedError(Exception):
        pass
    assert memo_sync.is_pruned(StateDiscardedError("0xab"))
    pool, seed = world
    root = _Root(pool, seed, {SEED_AT + 3: SEED_AT + 1}, _published(pool, seed, [SEED_AT + 1]))

    def flaky(b):
        raise ConnectionError("connection reset")
    cs = memo_sync.ChainSet(pool=pool, read_at=flaky, reveal_bytes=root.reveal_bytes,
                            snapshot=lambda d: None)
    with pytest.raises(ConnectionError, match="reset"):
        cs.advance_to(SEED_AT + 4)
