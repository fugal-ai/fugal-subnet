"""Verified heads are published by hash, with the reveal that scores them.

The root uploads every head it verified in an epoch to the public store once
that epoch's reveal is published, under the head's sha256. Operational: no
score reads it. These tests pin the properties the feature rests on:

- a head never becomes public before the reveal that lists it;
- only heads that verified, within the size cap, whose bytes still match
  their name, are ever uploaded;
- the reveal's bytes, its commit hash and the consensus digest are the same
  with the feature on and off (the head locations go in a sidecar);
- a failing or a hanging store never raises into the epoch and never takes
  more than the time budget, and what was not sent is sent on a later epoch;
- no local path reaches a log line or the sidecar.

Every uploader here is a fake. Nothing touches a network, a bucket or a
credential.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import fugal_subnet.config as config
from fugal_subnet import head_publisher as hp
from fugal_subnet.reveal_publisher import GcsUploader, publish_due_reveals

ROOT = Path(__file__).resolve().parents[1]
NETUID = 7
HEAD_A = b"PK\x03\x04 head A bytes"
HEAD_B = b"PK\x03\x04 head B bytes"
HEAD_C = b"PK\x03\x04 head C bytes"
SHA_A = hashlib.sha256(HEAD_A).hexdigest()
SHA_B = hashlib.sha256(HEAD_B).hexdigest()
SHA_C = hashlib.sha256(HEAD_C).hexdigest()
CAP = 1 << 20


class Store:
    """A bucket, as the publisher sees it. Records every object it was sent."""

    def __init__(self, fail_on=(), existing=()):
        self.objects: dict[str, bytes] = {}
        self.sent: list[str] = []
        self.fail_on = set(fail_on)
        self.existing = set(existing)

    def _check(self, path, key):
        if any(part in key for part in self.fail_on):
            # A real failure names the local file (subprocess.CalledProcessError
            # carries the whole command line), so the fake does too.
            raise OSError(f"could not send {path}: bucket unreachable")

    def upload(self, path, key):
        self._check(path, key)
        self.objects[key] = Path(path).read_bytes()
        self.sent.append(key)
        return f"gs://b/{key}"

    def upload_if_absent(self, path, key):
        self._check(path, key)
        if key not in self.existing and key not in self.objects:
            self.objects[key] = Path(path).read_bytes()
            self.sent.append(key)
        return f"gs://b/{key}"


def _epoch(epoch_dir, index, head_hashes, champion=None, published=True):
    """One epoch directory holding a reveal, optionally already published."""
    eid = f"e{index:08d}"
    d = Path(epoch_dir) / eid
    d.mkdir(parents=True)
    reveal = {"epoch_id": eid, "head_hashes": {str(u): h for u, h in head_hashes.items()}}
    if champion:
        reveal["champion"] = {"event": "hold", "champion": champion}
    data = json.dumps(reveal).encode()
    (d / "reveal.json").write_bytes(data)
    if published:
        (d / ".published").write_text(json.dumps({
            "epoch_id": eid, "sha256": hashlib.sha256(data).hexdigest(),
            "url": f"gs://b/{NETUID}/{eid}/reveal.json"}))
    return eid


def _dirs(tmp_path):
    epochs = tmp_path / "epochs"
    epochs.mkdir()
    return str(epochs), str(tmp_path / "outbox")


def _publish(epochs, spool, store, now, delay=1, **kw):
    return hp.publish_due_heads(epochs, spool, now, delay, NETUID, store, max_bytes=CAP, **kw)


# --- the layout ----------------------------------------------------------------


def test_a_head_is_named_by_its_sha256_outside_the_prefix_followers_list():
    assert hp.head_key(70, SHA_A) == f"heads/70/{SHA_A}.npz"
    assert hp.sidecar_key(70, "e00026483") == "70/e00026483/heads.json"
    # Followers list `<netuid>/` (reveal_sync.list_reveals); head objects, whose
    # number miners decide, must not be in that listing.
    assert not hp.head_key(70, SHA_A).startswith("70/")
    for bad in ("", "A" * 64, "../" + "a" * 61, "a" * 63, "a" * 64 + "/x"):
        with pytest.raises(ValueError):
            hp.head_key(70, bad)


# --- the outbox ----------------------------------------------------------------


def test_verified_heads_are_kept_once_per_hash_and_private(tmp_path):
    spool = tmp_path / "outbox"
    counts = hp.spool_verified_heads(str(spool), {4: HEAD_A, 9: HEAD_B, 11: HEAD_A},
                                     {4: SHA_A, 9: SHA_B, 11: SHA_A}, CAP)
    assert counts == {"kept": 2, "present": 1, "sent": 0, "skipped": 0}
    assert (spool / f"{SHA_A}.npz").read_bytes() == HEAD_A
    assert stat.S_IMODE(os.stat(spool).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(spool / f"{SHA_A}.npz").st_mode) == 0o600
    assert sorted(p.name for p in spool.iterdir()) == sorted([f"{SHA_A}.npz", f"{SHA_B}.npz"])


def test_bytes_that_do_not_match_their_hash_or_exceed_the_cap_are_never_kept(tmp_path):
    spool = tmp_path / "outbox"
    counts = hp.spool_verified_heads(
        str(spool), {1: HEAD_A, 2: HEAD_B, 3: b"x" * 65, 4: b""},
        {1: SHA_B, 2: "../../etc/passwd", 3: hashlib.sha256(b"x" * 65).hexdigest(),
         4: hashlib.sha256(b"").hexdigest()}, 64)
    assert counts["skipped"] == 4 and counts["kept"] == 0
    assert not spool.exists() or list(spool.iterdir()) == []


def test_the_outbox_is_bounded_and_drops_the_heads_seen_longest_ago(tmp_path, monkeypatch):
    spool = tmp_path / "outbox"
    monkeypatch.setattr(hp, "SPOOL_MAX_HEADS", 2)
    blobs = [bytes([i]) * 8 for i in range(4)]
    for i, b in enumerate(blobs):
        hp.spool_verified_heads(str(spool), {i: b}, {i: hashlib.sha256(b).hexdigest()}, CAP)
        os.utime(spool / f"{hashlib.sha256(b).hexdigest()}.npz", (1000 + i, 1000 + i))
    hp.spool_verified_heads(str(spool), {}, {}, CAP)
    last = hashlib.sha256(blobs[3]).hexdigest()
    hp.spool_verified_heads(str(spool), {3: blobs[3]}, {3: last}, CAP)
    kept = {p.name for p in spool.iterdir()}
    assert len(kept) == 2 and f"{last}.npz" in kept


def test_the_publisher_never_parses_a_head():
    tree = ast.parse((ROOT / "fugal_subnet" / "head_publisher.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    # Hashing, paths, JSON, a thread and a clock: nothing that can open an npz.
    assert imported <= {"__future__", "hashlib", "json", "logging", "os", "re",
                        "threading", "time"}


# --- when a head becomes public --------------------------------------------------


def test_heads_are_uploaded_with_the_published_reveal_that_lists_them(tmp_path):
    epochs, spool = _dirs(tmp_path)
    hp.spool_verified_heads(spool, {4: HEAD_A, 9: HEAD_B}, {4: SHA_A, 9: SHA_B}, CAP)
    eid = _epoch(epochs, 10, {4: SHA_A, 9: SHA_B})
    reveal_before = (Path(epochs) / eid / "reveal.json").read_bytes()
    store = Store()
    out = _publish(epochs, spool, store, now=11)
    assert store.objects[f"heads/{NETUID}/{SHA_A}.npz"] == HEAD_A
    assert store.objects[f"heads/{NETUID}/{SHA_B}.npz"] == HEAD_B
    assert out["uploaded"] == 2 and out["failed"] == 0 and out["epochs"] == [eid]
    # The sidecar beside the reveal says where each hash is.
    sidecar = json.loads(store.objects[f"{NETUID}/{eid}/heads.json"])
    assert sidecar["epoch_id"] == eid and sidecar["netuid"] == NETUID
    assert sidecar["reveal_sha256"] == hashlib.sha256(reveal_before).hexdigest()
    assert sidecar["heads"] == {
        SHA_A: {"key": f"heads/{NETUID}/{SHA_A}.npz", "url": f"gs://b/heads/{NETUID}/{SHA_A}.npz",
                "bytes": len(HEAD_A)},
        SHA_B: {"key": f"heads/{NETUID}/{SHA_B}.npz", "url": f"gs://b/heads/{NETUID}/{SHA_B}.npz",
                "bytes": len(HEAD_B)},
    }
    assert sidecar["unavailable"] == []
    assert "allow_pickle=False" in sidecar["verify"] and "sha256" in sidecar["verify"]
    # The reveal is not rewritten, and the sent bytes leave the outbox.
    assert (Path(epochs) / eid / "reveal.json").read_bytes() == reveal_before
    assert not (Path(spool) / f"{SHA_A}.npz").exists()
    # Next epoch: settled, nothing is sent again.
    sent = list(store.sent)
    assert _publish(epochs, spool, store, now=12)["epochs"] == []
    assert store.sent == sent


def test_a_head_is_not_public_before_its_reveal_is(tmp_path):
    epochs, spool = _dirs(tmp_path)
    hp.spool_verified_heads(spool, {4: HEAD_A}, {4: SHA_A}, CAP)
    store = Store()
    # The reveal exists but has not been published (inside the hold, or its
    # upload failed): no marker, no head.
    eid = _epoch(epochs, 10, {4: SHA_A}, published=False)
    assert _publish(epochs, spool, store, now=10)["uploaded"] == 0
    assert _publish(epochs, spool, store, now=11)["uploaded"] == 0
    assert _publish(epochs, spool, store, now=50)["uploaded"] == 0
    assert store.objects == {}
    # A marker inside the hold is not enough either: the hold is checked here
    # too, not only by the reveal publisher.
    data = (Path(epochs) / eid / "reveal.json").read_bytes()
    (Path(epochs) / eid / ".published").write_text(json.dumps(
        {"epoch_id": eid, "sha256": hashlib.sha256(data).hexdigest(), "url": "gs://b/x"}))
    assert _publish(epochs, spool, store, now=10, delay=1)["uploaded"] == 0
    assert _publish(epochs, spool, store, now=12, delay=3)["uploaded"] == 0
    assert store.objects == {}
    assert _publish(epochs, spool, store, now=13, delay=3)["uploaded"] == 1


def test_the_reveal_publisher_then_the_heads_end_to_end(tmp_path):
    epochs, spool = _dirs(tmp_path)
    hp.spool_verified_heads(spool, {4: HEAD_A}, {4: SHA_A}, CAP)
    eid = _epoch(epochs, 10, {4: SHA_A}, published=False)
    store = Store(fail_on=["reveal.json"])
    assert publish_due_reveals(epochs, 11, 1, NETUID, store) == []
    assert _publish(epochs, spool, store, now=11)["uploaded"] == 0 and store.objects == {}
    store.fail_on.clear()
    assert [r["epoch_id"] for r in publish_due_reveals(epochs, 12, 1, NETUID, store)] == [eid]
    assert _publish(epochs, spool, store, now=12)["uploaded"] == 1
    assert store.sent == [f"{NETUID}/{eid}/reveal.json", f"heads/{NETUID}/{SHA_A}.npz",
                          f"{NETUID}/{eid}/heads.json"]


def test_only_hashes_a_published_reveal_lists_are_uploaded(tmp_path):
    epochs, spool = _dirs(tmp_path)
    # A is in the published epoch; B was verified in a later epoch whose
    # reveal is still held; C was never in any reveal.
    hp.spool_verified_heads(spool, {1: HEAD_A, 2: HEAD_B, 3: HEAD_C},
                            {1: SHA_A, 2: SHA_B, 3: SHA_C}, CAP)
    _epoch(epochs, 10, {1: SHA_A})
    _epoch(epochs, 11, {1: SHA_A, 2: SHA_B}, published=False)
    store = Store()
    _publish(epochs, spool, store, now=11)
    assert [k for k in store.objects if k.startswith("heads/")] == [f"heads/{NETUID}/{SHA_A}.npz"]
    assert (Path(spool) / f"{SHA_B}.npz").exists() and (Path(spool) / f"{SHA_C}.npz").exists()


def test_a_hash_the_reveal_lists_but_this_validator_did_not_keep_is_reported_not_invented(tmp_path):
    epochs, spool = _dirs(tmp_path)
    hp.spool_verified_heads(spool, {1: HEAD_A}, {1: SHA_A}, CAP)
    eid = _epoch(epochs, 10, {1: SHA_A, 2: SHA_B, 3: "not-a-hash"})
    store = Store()
    out = _publish(epochs, spool, store, now=11)
    sidecar = json.loads(store.objects[f"{NETUID}/{eid}/heads.json"])
    assert list(sidecar["heads"]) == [SHA_A] and sidecar["unavailable"] == [SHA_B]
    assert out["unavailable"] == 1
    assert (Path(epochs) / eid / hp.EPOCH_MARKER).exists()


def test_a_file_whose_bytes_changed_or_grew_past_the_cap_is_removed_not_uploaded(tmp_path):
    epochs, spool = _dirs(tmp_path)
    hp.spool_verified_heads(spool, {1: HEAD_A, 2: HEAD_B}, {1: SHA_A, 2: SHA_B}, CAP)
    (Path(spool) / f"{SHA_A}.npz").write_bytes(b"something else")
    (Path(spool) / f"{SHA_B}.npz").write_bytes(b"y" * (CAP + 1))
    eid = _epoch(epochs, 10, {1: SHA_A, 2: SHA_B})
    store = Store()
    _publish(epochs, spool, store, now=11)
    assert [k for k in store.objects if k.startswith("heads/")] == []
    assert sorted(json.loads(store.objects[f"{NETUID}/{eid}/heads.json"])["unavailable"]) == \
        sorted([SHA_A, SHA_B])
    assert not (Path(spool) / f"{SHA_A}.npz").exists()


def test_a_reveal_that_is_not_the_one_published_is_left_alone(tmp_path):
    epochs, spool = _dirs(tmp_path)
    hp.spool_verified_heads(spool, {1: HEAD_A}, {1: SHA_A}, CAP)
    eid = _epoch(epochs, 10, {1: SHA_B})
    # The file changed after it was published: its hashes are not the
    # published reveal's, so nothing is uploaded on its word.
    (Path(epochs) / eid / "reveal.json").write_text(
        json.dumps({"epoch_id": eid, "head_hashes": {"1": SHA_A}}))
    store = Store()
    assert _publish(epochs, spool, store, now=11)["uploaded"] == 0
    assert store.objects == {}


def test_objects_already_in_the_store_are_not_sent_again(tmp_path):
    epochs, spool = _dirs(tmp_path)
    hp.spool_verified_heads(spool, {1: HEAD_A, 2: HEAD_B}, {1: SHA_A, 2: SHA_B}, CAP)
    e10 = _epoch(epochs, 10, {1: SHA_A})
    store = Store(existing=[f"heads/{NETUID}/{SHA_A}.npz"])
    _publish(epochs, spool, store, now=11)
    # In the store already (this validator lost its record, or a peer sent
    # it): not overwritten, and listed in the sidecar as fetchable.
    assert f"heads/{NETUID}/{SHA_A}.npz" not in store.sent
    assert SHA_A in json.loads(store.objects[f"{NETUID}/{e10}/heads.json"])["heads"]
    # The same head served again next epoch is neither kept nor sent again.
    assert hp.spool_verified_heads(spool, {1: HEAD_A}, {1: SHA_A}, CAP)["sent"] == 1
    e11 = _epoch(epochs, 11, {1: SHA_A, 2: SHA_B})
    _publish(epochs, spool, store, now=12)
    assert [k for k in store.sent if k.startswith("heads/")] == [f"heads/{NETUID}/{SHA_B}.npz"]
    side = json.loads(store.objects[f"{NETUID}/{e11}/heads.json"])
    assert sorted(side["heads"]) == sorted([SHA_A, SHA_B])
    assert side["heads"][SHA_A]["bytes"] == len(HEAD_A)


def test_the_champion_goes_first(tmp_path):
    epochs, spool = _dirs(tmp_path)
    hp.spool_verified_heads(spool, {1: HEAD_A, 2: HEAD_B, 3: HEAD_C},
                            {1: SHA_A, 2: SHA_B, 3: SHA_C}, CAP)
    champion = max(SHA_A, SHA_B, SHA_C)     # last in sorted order
    _epoch(epochs, 10, {1: SHA_A, 2: SHA_B, 3: SHA_C}, champion=champion)
    store = Store()
    _publish(epochs, spool, store, now=11)
    assert store.sent[0] == f"heads/{NETUID}/{champion}.npz"


# --- a store that fails, or hangs --------------------------------------------------


def test_a_failing_store_costs_nothing_and_is_retried_on_a_later_epoch(tmp_path, caplog):
    epochs, spool = _dirs(tmp_path)
    hp.spool_verified_heads(spool, {1: HEAD_A, 2: HEAD_B}, {1: SHA_A, 2: SHA_B}, CAP)
    eid = _epoch(epochs, 10, {1: SHA_A, 2: SHA_B})
    store = Store(fail_on=["heads/"])
    with caplog.at_level(logging.DEBUG):
        out = _publish(epochs, spool, store, now=11)     # must not raise
    assert out["uploaded"] == 0 and out["failed"] >= 1 and out["epochs"] == []
    assert store.objects == {}
    assert not (Path(epochs) / eid / hp.EPOCH_MARKER).exists()
    assert (Path(spool) / f"{SHA_A}.npz").exists()
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    # The store's error names the local file; the log must not.
    assert str(tmp_path) not in caplog.text
    # Later: the store is back, and everything unpublished goes up.
    store.fail_on.clear()
    out = _publish(epochs, spool, store, now=12)
    assert out["uploaded"] == 2 and out["epochs"] == [eid]
    assert sorted(json.loads(store.objects[f"{NETUID}/{eid}/heads.json"])["heads"]) == \
        sorted([SHA_A, SHA_B])


def test_a_failed_sidecar_is_retried_without_sending_the_heads_again(tmp_path):
    epochs, spool = _dirs(tmp_path)
    hp.spool_verified_heads(spool, {1: HEAD_A}, {1: SHA_A}, CAP)
    eid = _epoch(epochs, 10, {1: SHA_A})
    store = Store(fail_on=["heads.json"])
    assert _publish(epochs, spool, store, now=11)["epochs"] == []
    store.fail_on.clear()
    assert _publish(epochs, spool, store, now=12)["epochs"] == [eid]
    assert store.sent == [f"heads/{NETUID}/{SHA_A}.npz", f"{NETUID}/{eid}/heads.json"]


def test_a_failing_store_is_given_up_on_after_a_few_tries_in_one_run(tmp_path):
    epochs, spool = _dirs(tmp_path)
    blobs = {i: bytes([i]) * 16 for i in range(12)}
    hashes = {i: hashlib.sha256(b).hexdigest() for i, b in blobs.items()}
    hp.spool_verified_heads(spool, blobs, hashes, CAP)
    _epoch(epochs, 10, hashes)

    class Counting(Store):
        tries = 0

        def upload_if_absent(self, path, key):
            Counting.tries += 1
            raise TimeoutError("slow bucket")

    out = _publish(epochs, spool, Counting(), now=11)
    assert out["failed"] == Counting.tries == hp.MAX_FAILURES_PER_RUN


def test_an_epoch_older_than_the_retry_window_is_dropped(tmp_path):
    epochs, spool = _dirs(tmp_path)
    hp.spool_verified_heads(spool, {1: HEAD_A}, {1: SHA_A}, CAP)
    _epoch(epochs, 10, {1: SHA_A})
    store = Store()
    assert _publish(epochs, spool, store, now=11 + hp.RETRY_EPOCHS)["uploaded"] == 0
    assert _publish(epochs, spool, store, now=10 + hp.RETRY_EPOCHS)["uploaded"] == 1


def test_the_time_budget_stops_uploads_and_the_rest_go_next_time(tmp_path):
    epochs, spool = _dirs(tmp_path)
    hp.spool_verified_heads(spool, {1: HEAD_A, 2: HEAD_B}, {1: SHA_A, 2: SHA_B}, CAP)
    eid = _epoch(epochs, 10, {1: SHA_A, 2: SHA_B})
    store = Store()
    out = _publish(epochs, spool, store, now=11, deadline=time.monotonic() - 1)
    assert out["uploaded"] == 0 and out["out_of_time"] is True and store.objects == {}
    out = _publish(epochs, spool, store, now=12, deadline=time.monotonic() + 60)
    assert out["uploaded"] == 2 and out["epochs"] == [eid]


class Hanging(Store):
    """A store whose connection never answers, until the test releases it."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.entered = threading.Event()

    def upload_if_absent(self, path, key):
        self.entered.set()
        self.release.wait(30)
        return super().upload_if_absent(path, key)


def test_a_hanging_store_never_holds_the_epoch_longer_than_the_budget(tmp_path, caplog):
    epochs, spool = _dirs(tmp_path)
    store = Hanging()
    pub = hp.HeadPublisher(epochs, NETUID, store, delay_epochs=1, max_bytes=CAP,
                           spool_dir=spool, budget_s=0.3)
    pub.spool({1: HEAD_A}, {1: SHA_A})
    eid = _epoch(epochs, 10, {1: SHA_A})
    try:
        t0 = time.monotonic()
        with caplog.at_level(logging.WARNING):
            assert pub.publish(11) is None
        assert time.monotonic() - t0 < 5.0
        assert store.entered.is_set() and store.objects == {}
        assert "still running" in caplog.text or "time budget" in caplog.text
        # The next epoch does not stack a second upload on the stuck one.
        t0 = time.monotonic()
        assert pub.publish(12) is None
        assert time.monotonic() - t0 < 1.0
    finally:
        store.release.set()
    # Once the store answers again the head and its sidecar go up.
    deadline = time.monotonic() + 10
    while pub.busy and time.monotonic() < deadline:
        time.sleep(0.02)
    out = pub.publish(13)
    assert out is not None and f"heads/{NETUID}/{SHA_A}.npz" in store.objects
    assert f"{NETUID}/{eid}/heads.json" in store.objects


def test_the_publisher_object_never_raises_into_the_epoch(tmp_path, monkeypatch):
    epochs, spool = _dirs(tmp_path)
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x")
    pub = hp.HeadPublisher(epochs, NETUID, Store(), delay_epochs=1, max_bytes=CAP,
                           spool_dir=str(not_a_dir / "outbox"), budget_s=5)
    pub.spool({1: HEAD_A}, {1: SHA_A})          # a disk error: logged, not raised

    def boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(hp, "publish_due_heads", boom)
    assert pub.publish(11) is None


def test_the_default_outbox_is_beside_the_epoch_directory_not_inside_it(tmp_path):
    epochs = tmp_path / "results" / "epochs"
    d = hp.spool_dir_for(str(epochs))
    assert Path(d).parent == epochs.parent and Path(d) != epochs
    assert hp.spool_dir_for(str(epochs) + "/") == d


# --- nothing consensus moves --------------------------------------------------------


def _one_epoch(tmp_path, monkeypatch, name, publish):
    """Commit, reveal and publish one epoch; return what consensus compares."""
    from fugal_subnet import commit_reveal
    from fugal_subnet.fingerprint import consensus_digest

    epochs = tmp_path / name / "epochs"
    spool = str(tmp_path / name / "outbox")
    monkeypatch.setattr(commit_reveal, "EPOCH_DIR", str(epochs))
    eid = "e00000010"
    questions = [{"id": "q1", "prompt": "2+2?"}, {"id": "q2", "prompt": "3+3?"}]
    heads, hashes = {4: HEAD_A, 9: HEAD_B}, {4: SHA_A, 9: SHA_B}
    commitment = commit_reveal.commit_epoch(eid, questions, "0x" + "ab" * 32)
    store = Store()
    pub = hp.HeadPublisher(str(epochs), NETUID, store, delay_epochs=1, max_bytes=CAP,
                           spool_dir=spool, budget_s=10) if publish else None
    if pub is not None:
        pub.spool(heads, hashes)
    assert commit_reveal.reveal_epoch(
        eid, questions, [[1, 0], [0, 1]], ["m1", "m2"], {"m1": 0.1, "m2": 0.2},
        hashes, {4: [0, 1], 9: [1, 0]}, {4: {"score": 1.0}, 9: {"score": 0.5}},
        {4: 0.7, 9: 0.3}, protocol_version=2,
        v2_fields={"champion": {"event": "hold", "champion": SHA_A}},
    )
    published = publish_due_reveals(str(epochs), 11, 1, NETUID, store)
    if pub is not None:
        assert pub.publish(11)["uploaded"] == 2
    reveal = json.loads((epochs / eid / "reveal.json").read_bytes())
    return {
        "commit_hash": commitment.commit_hash,
        "reveal_commit_hash": reveal["commit_hash"],
        "consensus_digest": reveal["consensus_digest"],
        "live_digest": consensus_digest(),
        "reveal_bytes": (epochs / eid / "reveal.json").read_bytes(),
        "published_sha256": published[0]["sha256"],
        "reveal_object": store.objects[f"{NETUID}/{eid}/reveal.json"],
        "verified": commit_reveal.verify_epoch(str(epochs / eid)),
    }, store


def test_the_commit_hash_the_digest_and_the_reveal_are_identical_with_the_feature_on_and_off(
        tmp_path, monkeypatch):
    on, store_on = _one_epoch(tmp_path, monkeypatch, "on", publish=True)
    off, store_off = _one_epoch(tmp_path, monkeypatch, "off", publish=False)
    # Every consensus value, and the reveal file byte for byte (so the sha256
    # the validator commits on chain for it is the same too).
    assert on == off
    assert on["verified"] is True
    # What differs is only the extra objects: the heads and the sidecar.
    extra = set(store_on.objects) - set(store_off.objects)
    assert extra == {f"heads/{NETUID}/{SHA_A}.npz", f"heads/{NETUID}/{SHA_B}.npz",
                     f"{NETUID}/e00000010/heads.json"}


def test_the_switch_does_not_move_the_consensus_digest_or_the_commit_hash():
    """The same two values from a fresh interpreter with the switch 1, 0 and unset."""
    code = (
        "from fugal_subnet.fingerprint import consensus_digest, grader_hash\n"
        "from fugal_subnet.commit_reveal import compute_commit_hash\n"
        "q = [{'id': 'q1', 'prompt': '2+2?'}]\n"
        "print(consensus_digest(), compute_commit_hash(q, grader_hash()))\n"
    )
    seen = {}
    for value in ("1", "0", None):
        env = {k: v for k, v in os.environ.items() if k != "FUGAL_PUBLISH_HEADS"}
        if value is not None:
            env["FUGAL_PUBLISH_HEADS"] = value
        out = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(ROOT),
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        seen[value] = out.stdout.split()
        assert len(seen[value]) == 2
    assert seen["1"] == seen["0"] == seen[None]


def test_the_switch_is_operational_and_on_by_default_for_the_root_only():
    assert "FUGAL_PUBLISH_HEADS" in config.OPERATIONAL_ENV_VARS
    assert "FUGAL_PUBLISH_HEADS" not in config.CONSENSUS_ENV_VARS
    assert "FUGAL_PUBLISH_HEADS" not in config.CONSENSUS_ENV_INPUTS
    # Unset: the root publishes, nobody else does.
    assert config.publish_heads_enabled(root=True, flag=None) is True
    assert config.publish_heads_enabled(root=False, flag=None) is False
    # Any validator may publish; the root may decline.
    assert config.publish_heads_enabled(root=False, flag=True) is True
    assert config.publish_heads_enabled(root=True, flag=False) is False
    assert config.parse_publish_heads_flag("") is None
    assert config.parse_publish_heads_flag("1") is True
    assert config.parse_publish_heads_flag("no") is False
    with pytest.raises(ValueError, match="FUGAL_PUBLISH_HEADS"):
        config.parse_publish_heads_flag("off")


# --- the wiring ---------------------------------------------------------------------


def test_heads_are_published_after_weights_and_after_the_reveal_and_feed_nothing_back():
    src = (ROOT / "neurons" / "validator.py").read_text()
    spool = src.index("head_publisher.spool(verified_heads, head_hashes)")
    weights = src.index("set_weights_with_retry(\n                subtensor")
    reveals = src.index("publish_due_reveals(\n                        EPOCH_DIR")
    heads = src.index("head_publisher.publish(epoch_index)")
    assert spool < weights < reveals < heads
    # Neither call's result is used, so no score, hash, reveal or fingerprint
    # input can depend on them.
    for call in ("head_publisher.spool(verified_heads, head_hashes)",
                 "head_publisher.publish(epoch_index)"):
        line = next(ln for ln in src.splitlines() if call in ln)
        assert line.strip() == call
    for name in ("reveal_epoch(", "update_scores(", "compute_weights("):
        i = src.index(name)
        assert "head_publisher" not in src[i:src.index(")", i)]


# --- the store adapter --------------------------------------------------------------


def test_the_cli_upload_never_overwrites_and_adds_no_metadata(tmp_path, monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    # Force the CLI path whether or not the client library is installed.
    monkeypatch.setitem(sys.modules, "google.cloud", None)
    monkeypatch.setattr(subprocess, "run", run)
    f = tmp_path / f"{SHA_A}.npz"
    f.write_bytes(HEAD_A)
    url = GcsUploader("bkt", timeout=7).upload_if_absent(str(f), hp.head_key(70, SHA_A))
    assert url == f"gs://bkt/heads/70/{SHA_A}.npz"
    (cmd, kw), = calls
    assert cmd[:3] == ["gcloud", "storage", "cp"] and "--no-clobber" in cmd
    assert cmd[-2:] == [str(f), url]
    assert kw["timeout"] == 7 and kw["check"] is True
    assert not any("metadata" in part or "preserve" in part for part in cmd)


# --- the reader ---------------------------------------------------------------------


class _Bucket:
    def __init__(self, objects):
        self.objects, self.calls = objects, []

    def __call__(self, req, timeout=None):
        self.calls.append(req.full_url)
        key = req.full_url.split("/", 4)[4]
        if key not in self.objects:
            import urllib.error
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)
        return io.BytesIO(self.objects[key])


def _npz(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def test_a_reader_gets_the_bytes_only_if_they_match_the_hash(tmp_path):
    from fugal_subnet import reveal_sync

    good = _npz(W=np.zeros((2, 3), dtype=np.float32))
    sha = hashlib.sha256(good).hexdigest()
    bucket = _Bucket({f"heads/70/{sha}.npz": good, f"heads/70/{SHA_A}.npz": b"other bytes"})
    data = reveal_sync.fetch_head("fugal-reveals", 70, sha, opener=bucket)
    assert data == good
    assert bucket.calls == [f"https://storage.googleapis.com/fugal-reveals/heads/70/{sha}.npz"]
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        assert z["W"].shape == (2, 3)
    # Wrong bytes under the right name, a name that is not a hash, a bad
    # bucket and an object over the cap are all refused.
    with pytest.raises(ValueError, match="sha256"):
        reveal_sync.fetch_head("fugal-reveals", 70, SHA_A, opener=bucket)
    with pytest.raises(ValueError):
        reveal_sync.fetch_head("fugal-reveals", 70, "../70/e00000001/reveal.json", opener=bucket)
    with pytest.raises(ValueError):
        reveal_sync.fetch_head("Not A Bucket", 70, sha, opener=bucket)
    with pytest.raises(ValueError, match="larger"):
        reveal_sync.fetch_head("fugal-reveals", 70, sha, opener=bucket, max_bytes=16)


def test_a_fetched_head_holding_a_pickle_does_not_load_with_pickle_disabled():
    hostile = _npz(W=np.array([{"a": 1}], dtype=object))
    with np.load(io.BytesIO(hostile), allow_pickle=False) as z, pytest.raises(ValueError):
        z["W"]
