"""A validator that is not the root follows the root with nothing to set up,
and a finney validator that would silently score differently is refused.

Each refusal and each default here is a mistake a mainnet operator made with
the guide open: the v1 entry only, no reference reveals, no published IP. The
validator ran, looked healthy, and burned its weight.
"""
import base64
import hashlib
import io
import json
import logging
import urllib.error

import click
import pytest

import fugal_subnet
from fugal_subnet import config, reveal_sync
from neurons import validator
from neurons.validator import _check_chain_configuration as check_chain

FINNEY = config.FINNEY_GENESIS_HASH
OTHER = "0x" + "cd" * 32
BASE = "a" * 64
V2_BLOCK = config.FINNEY_V2_ACTIVATION_BLOCK
IP = "34.120.8.9"


# --- the fetch -----------------------------------------------------------------


class Bucket:
    """A public GCS bucket, as `urlopen` sees it."""

    def __init__(self, netuid=70):
        self.netuid, self.objects, self.calls, self.page_size = netuid, {}, [], 1000
        self.fail_listing = self.listing_override = None

    def put(self, epoch, reveal=None, raw=None):
        eid = f"e{epoch:08d}"
        data = raw if raw is not None else json.dumps(reveal or {
            "epoch_id": eid, "protocol_version": 2,
            "v2_state": {"seat": {"champion": "c" * 64}, "serving": {"c" * 64: 12}},
            "environment": {"fugal_version": "0.3.2"}, "weights": {"12": 1.0}}).encode()
        self.objects[f"{self.netuid}/{eid}/reveal.json"] = data
        return eid

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.calls.append(url)
        if "/storage/v1/b/" in url:
            if self.fail_listing:
                raise self.fail_listing
            if self.listing_override is not None:
                return io.BytesIO(self.listing_override)
            names = sorted(self.objects)
            start = int(url.split("pageToken=")[1].split("&")[0]) if "pageToken=" in url else 0
            page = names[start:start + self.page_size]
            body = {"items": [{"name": n, "size": str(len(self.objects[n])),
                               "md5Hash": base64.b64encode(
                                   hashlib.md5(self.objects[n]).digest()).decode()}
                              for n in page]}
            if start + self.page_size < len(names):
                body["nextPageToken"] = str(start + self.page_size)
            return io.BytesIO(json.dumps(body).encode())
        key = url.split("/", 4)[4]
        if key not in self.objects:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return io.BytesIO(self.objects[key])


def fetches(bucket):
    return [u for u in bucket.calls if "/storage/v1/b/" not in u]


def test_a_sync_fetches_what_is_missing_and_nothing_twice(tmp_path):
    bucket = Bucket()
    for e in (10, 11, 12):
        bucket.put(e)
    r = reveal_sync.sync("fugal-reveals", 70, str(tmp_path), opener=bucket)
    assert r.ok and (r.listed, r.downloaded, r.present, r.newest) == (3, 3, 0, "e00000012")
    assert json.loads((tmp_path / "e00000011" / "reveal.json").read_text())["epoch_id"] == "e00000011"
    bucket.calls.clear()
    bucket.put(13)
    r = reveal_sync.sync("fugal-reveals", 70, str(tmp_path), opener=bucket)
    assert (r.downloaded, r.present, r.newest) == (1, 3, "e00000013")
    assert len(fetches(bucket)) == 1 and fetches(bucket)[0].endswith("/70/e00000013/reveal.json")


def test_since_bounds_what_is_fetched_and_listings_are_paged(tmp_path):
    bucket = Bucket()
    bucket.page_size = 2
    for e in range(1, 8):
        bucket.put(e)
    r = reveal_sync.sync("fugal-reveals", 70, str(tmp_path), since=5, opener=bucket)
    assert r.ok and r.listed == 3 and sorted(p.name for p in tmp_path.iterdir()) == [
        "e00000005", "e00000006", "e00000007"]


def test_a_file_that_was_damaged_on_disk_is_fetched_again(tmp_path):
    bucket = Bucket()
    eid = bucket.put(3)
    reveal_sync.sync("fugal-reveals", 70, str(tmp_path), opener=bucket)
    path = tmp_path / eid / "reveal.json"
    good = path.read_bytes()
    path.write_bytes(good[:-1] + b" ")           # same size, different bytes
    r = reveal_sync.sync("fugal-reveals", 70, str(tmp_path), opener=bucket)
    assert r.downloaded == 1 and path.read_bytes() == good


@pytest.mark.parametrize("raw, why", [
    (b"not json", "JSONDecodeError"),
    (json.dumps({"epoch_id": "e00000099"}).encode(), "not a reveal for this epoch"),
    (json.dumps(["epoch_id"]).encode(), "not a reveal for this epoch"),
])
def test_an_object_that_is_not_this_epochs_reveal_never_reaches_disk(tmp_path, raw, why):
    bucket = Bucket()
    eid = bucket.put(4, raw=raw)
    r = reveal_sync.sync("fugal-reveals", 70, str(tmp_path), opener=bucket)
    assert not r.ok and why in r.failed[0] and r.newest is None
    assert not (tmp_path / eid / "reveal.json").exists()


def test_bytes_that_differ_from_the_listing_are_refused(tmp_path):
    bucket = Bucket()
    eid = bucket.put(4)
    listing = bucket(type("R", (), {"full_url": "https://x/storage/v1/b/b/o"})()).read()
    bucket.listing_override = listing
    bucket.objects[f"70/{eid}/reveal.json"] = bucket.objects[f"70/{eid}/reveal.json"][:-1] + b" "
    r = reveal_sync.sync("fugal-reveals", 70, str(tmp_path), opener=bucket)
    assert not r.ok and "MD5 differs" in r.failed[0]
    assert not (tmp_path / eid / "reveal.json").exists()


def test_object_names_are_not_trusted(tmp_path):
    bucket = Bucket()
    bucket.put(5)
    for name in ("70/../../etc/reveal.json", "70/e00000006/other.json", "71/e00000007/reveal.json",
                 "70/e6/reveal.json", "70/e00000008/x/reveal.json"):
        bucket.objects[name] = b"{}"
    r = reveal_sync.sync("fugal-reveals", 70, str(tmp_path), opener=bucket)
    assert r.listed == 1 and sorted(p.name for p in tmp_path.iterdir()) == ["e00000005"]


def test_an_oversized_object_is_refused_before_it_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(reveal_sync, "MAX_REVEAL_BYTES", 10)
    bucket = Bucket()
    bucket.put(5)
    r = reveal_sync.sync("fugal-reveals", 70, str(tmp_path), opener=bucket)
    assert not r.ok and "outside 1..10" in r.failed[0] and not fetches(bucket)


def test_a_failed_listing_is_a_result_not_an_exception(tmp_path):
    bucket = Bucket()
    bucket.put(5)
    reveal_sync.sync("fugal-reveals", 70, str(tmp_path), opener=bucket)
    bucket.fail_listing = urllib.error.URLError("no route to host")
    r = reveal_sync.sync("fugal-reveals", 70, str(tmp_path), opener=bucket)
    assert not r.ok and "no route to host" in r.error and r.newest == "e00000005"
    assert "listing failed" in r.summary()
    r = reveal_sync.sync("Not A Bucket!", 70, str(tmp_path), opener=bucket)
    assert "is not a bucket name" in r.error


def test_the_time_budget_stops_a_sync_and_the_next_one_continues(tmp_path):
    bucket = Bucket()
    for e in range(1, 6):
        bucket.put(e)
    ticks = iter(range(0, 1000, 10))
    r = reveal_sync.sync("fugal-reveals", 70, str(tmp_path), budget_s=25,
                         opener=bucket, clock=lambda: next(ticks))
    assert not r.ok and "time budget" in r.failed[-1] and 0 < r.downloaded < 5
    # Oldest first: what is on disk is a contiguous prefix.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"e{e:08d}" for e in range(1, r.downloaded + 1)]
    r2 = reveal_sync.sync("fugal-reveals", 70, str(tmp_path), opener=bucket)
    assert r2.ok and r2.present == r.downloaded and r2.newest == "e00000005"


# --- root or follower ----------------------------------------------------------------


def role(**kw):
    args = dict(finney=True, root_flag=None, own_bucket="", reference_bucket="",
                explicit_dir=None, state_path="/var/lib/fugal/validator_state.json", netuid=70)
    args.update(kw)
    return reveal_sync.resolve_role(**args)


def test_a_finney_validator_with_nothing_set_follows_the_roots_bucket():
    r = role()
    assert not r.root and r.bucket == "fugal-reveals"
    assert r.reference_dir == "/var/lib/fugal/reference-reveals/70"
    assert "fetched by this validator" in r.describe()
    # Publishing its own reveals elsewhere does not make it the root.
    assert not role(own_bucket="my-own-bucket").root


def test_the_root_is_named_or_recognised_by_its_bucket():
    for r in (role(root_flag=True), role(own_bucket="fugal-reveals"),
              role(finney=False, own_bucket="rehearsal", reference_bucket="rehearsal")):
        assert r.root and r.reference_dir is None and r.bucket is None
    # FUGAL_ROOT wins over a directory (a root may name one to rebuild from).
    assert role(root_flag=True, explicit_dir="/x").root


def test_fugal_root_0_makes_the_root_a_follower_while_it_recovers():
    """A root that lost its state follows another validator's reveals until
    it has caught up, although it publishes to the reference bucket."""
    r = role(root_flag=False, own_bucket="fugal-reveals", explicit_dir="/var/lib/fugal/peer")
    assert (r.root, r.reference_dir, r.bucket) == (False, "/var/lib/fugal/peer", None)


def test_an_operator_managed_directory_is_used_as_it_is():
    r = role(explicit_dir="/var/lib/fugal/root-reveals")
    assert (r.root, r.reference_dir, r.bucket) == (False, "/var/lib/fugal/root-reveals", None)


def test_off_finney_there_is_no_default_reference():
    assert role(finney=False) is None
    r = role(finney=False, reference_bucket="testnet-reveals", netuid=552)
    assert r.bucket == "testnet-reveals" and r.reference_dir.endswith("reference-reveals/552")


def test_a_follower_with_no_reveals_and_no_way_to_fetch_them_is_refused(tmp_path, monkeypatch):
    bucket = Bucket()
    bucket.fail_listing = urllib.error.URLError("blocked")
    monkeypatch.setattr(reveal_sync.urllib.request, "urlopen", bucket)
    r = reveal_sync.Role(False, str(tmp_path / "ref"), "fugal-reveals", "the default")
    with pytest.raises(click.ClickException, match="Could not fetch the root validator's reveals"):
        validator._sync_reference(r, 70, since=0, budget_s=5, required=True)
    # In the epoch loop the same failure is a log line (I6).
    assert not validator._sync_reference(r, 70, since=0, budget_s=5, required=False).ok
    # A root, or an operator-managed directory, fetches nothing.
    assert validator._sync_reference(reveal_sync.Role(True, None, None, "x"), 70, since=0,
                                     budget_s=5, required=True) is None
    assert not bucket.calls[2:]


def test_the_following_line_names_the_roots_seat_and_how_far_behind_it_is(tmp_path, caplog):
    ref, own = tmp_path / "ref", tmp_path / "own"

    def write(folder, epoch, champion):
        d = folder / f"e{epoch:08d}"
        d.mkdir(parents=True)
        (d / "reveal.json").write_text(json.dumps({
            "epoch_id": f"e{epoch:08d}", "protocol_version": 2,
            "environment": {"fugal_version": "0.3.2"},
            "v2_state": {"seat": {"champion": champion}, "serving": {champion: 55}}}))

    for e in range(100, 106):
        write(ref, e, "d" * 64)
        write(own, e, "d" * 64 if e < 103 else "e" * 64)
    r = reveal_sync.Role(False, str(ref), "fugal-reveals", "the default")
    seat = type("Seat", (), {"champion": "e" * 64})()
    with caplog.at_level(logging.INFO, logger="fugal.validator"):
        st = validator._log_following(r, 108, 2, seat, epoch_dir=str(own), differ_limit=5)
    assert (st["k"], st["lag"], st["uid"], st["seat_differs"]) == (105, 2, 55, 3)
    line = next(m for m in caplog.messages if m.startswith("Following root"))
    assert "seat uid 55 (head dddddddddddd) at e00000105" in line and "entries 2" in line
    assert f"fugal v{fugal_subnet.__version__} (root v0.3.2)" in line and "lag 2 epoch(s)" in line
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="fugal.validator"):
        validator._log_following(r, 108, 2, seat, epoch_dir=str(own), differ_limit=3)
    errors = [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert len(errors) == 1 and "has differed from the root's on the newest 3" in errors[0]

    caplog.clear()
    empty = reveal_sync.Role(False, str(tmp_path / "none"), "fugal-reveals", "the default")
    with caplog.at_level(logging.INFO, logger="fugal.validator"):
        validator._log_following(empty, 108, 2, seat, epoch_dir=str(own))
    assert any("NO reference reveal" in m for m in caplog.messages)
    assert validator._log_following(reveal_sync.Role(True, None, None, "x"), 1, 1, seat) is None


# --- finney refusals ---------------------------------------------------------------


@pytest.fixture
def v2_hash(monkeypatch):
    h = "b" * 64
    monkeypatch.setattr(validator, "v2_compose_hashes", lambda block, root=None: {h})
    return h


def test_a_correct_finney_configuration_starts(v2_hash):
    assert check_chain(FINNEY, True, [f"{BASE}:{v2_hash}"], V2_BLOCK, IP) is \
        config.FINNEY_CONFIRM_ACTIVATION_BLOCK


@pytest.mark.parametrize("block", [None, V2_BLOCK + 360])
def test_finney_refuses_a_missing_or_different_v2_block(v2_hash, block):
    with pytest.raises(click.ClickException, match=f"activated at block {V2_BLOCK}"):
        check_chain(FINNEY, True, [f"{BASE}:{v2_hash}"], block, IP)


def test_finney_refuses_a_validator_that_approves_no_v2_image(v2_hash):
    """The v1 entry only: every v2 proof fails, the seat stays empty, all weight burns."""
    with pytest.raises(click.ClickException, match="approves no protocol v2 miner image"):
        check_chain(FINNEY, True, [f"{BASE}:{'9' * 64}"], V2_BLOCK, IP)
    # Beside an old entry, the v2 one is enough.
    check_chain(FINNEY, True, [f"{BASE}:{'9' * 64}", f"{BASE}:{v2_hash}"], V2_BLOCK, IP)


@pytest.mark.parametrize("ip", ["", "10.0.0.7", "not-an-ip"])
def test_finney_refuses_a_validator_with_no_public_ip_to_publish(v2_hash, ip):
    with pytest.raises(click.ClickException, match="FUGAL_VALIDATOR_IP"):
        check_chain(FINNEY, True, [f"{BASE}:{v2_hash}"], V2_BLOCK, ip)


def test_other_chains_and_mock_keep_their_warnings(v2_hash):
    for genesis, live in ((OTHER, True), (FINNEY, False), (None, True)):
        check_chain(genesis, live, [f"{BASE}:{'9' * 64}"], None, "")


def test_a_checkout_without_frozen_composes_cannot_check_the_entries(monkeypatch, caplog):
    monkeypatch.setattr(validator, "v2_compose_hashes", lambda block, root=None: None)
    with caplog.at_level(logging.WARNING, logger="fugal.validator"):
        check_chain(FINNEY, True, [f"{BASE}:{'9' * 64}"], V2_BLOCK, IP)
    assert any("could not be checked" in m for m in caplog.messages)


def test_the_v2_compose_hashes_come_from_the_frozen_composes(tmp_path):
    d = tmp_path / "deploy" / "dstack"
    d.mkdir(parents=True)

    def freeze(name, compose):
        raw = json.dumps({"manifest_version": 2, "docker_compose_file": compose}).encode()
        (d / f"app-compose.frozen-{name}.json").write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    v1 = freeze("old", "environment:\n  - FUGAL_NETUID=70\n")
    v2 = freeze("new", "environment:\n  - FUGAL_V2_ACTIVATION_BLOCK=9485640\n")
    quoted = freeze("quoted", 'environment:\n  FUGAL_V2_ACTIVATION_BLOCK="9485640"\n')
    other = freeze("test", "environment:\n  - FUGAL_V2_ACTIVATION_BLOCK=94856400\n")
    (d / "app-compose.frozen-bad.json").write_text("{not json")
    got = validator.v2_compose_hashes(9485640, root=str(tmp_path))
    assert got == {v2, quoted} and v1 not in got and other not in got
    assert validator.v2_compose_hashes(9485640, root=str(tmp_path / "nowhere")) is None


def test_this_checkout_freezes_a_compose_for_finneys_flip_block():
    """The check above reads the real deploy/ tree on a validator: it must
    find the image finney validators approve."""
    hashes = validator.v2_compose_hashes(V2_BLOCK)
    assert hashes and "5408259ffb5e1971f7552fdef7011415737651eb5aa737996b222dd638c7e796" in hashes


# --- collection, per UID --------------------------------------------------------------


class Reply:
    def __init__(self, code=None, message=None, seconds=None):
        self.dendrite = type("T", (), {"status_code": code, "status_message": message,
                                       "process_time": seconds})()


def test_the_collection_report_says_who_was_reached_and_what_became_of_each_proof():
    hotkeys = [f"hk{i}" for i in range(6)]
    commitments = {"hk1": ("h", 5), "hk2": ("h", 5), "hk3": ("h", 5), "hk5": ("h", 5)}
    responses = [Reply(), Reply(200, "Success", 3.21), Reply(408, "Timeout", 120.004),
                 Reply(200, "Success", 9.0), Reply(503, "Service\nUnavailable \x1b[31m" + "x" * 200, 0.3),
                 Reply()]
    outcomes = {1: "valid", 2: "absent", 3: "invalid", 4: "absent", 5: "absent"}
    report, line = validator._collection_report(responses, hotkeys, commitments, outcomes)
    assert line.startswith("Collection: 6 queried; 1 valid, 1 invalid, 0 unverifiable, 4 absent")
    assert "Committed miners without a valid proof (3): uid 2 (absent; query 408 Timeout " \
           "after 120.0s), uid 3 (invalid; query 200 after 9.0s), uid 5 (absent; query no reply)" in line
    # uid 0: no commitment, no reply, nothing to say. uid 4 answered, so it is kept.
    assert sorted(report) == ["1", "2", "3", "4", "5"]
    assert report["1"] == {"query": "200", "seconds": 3.21, "result": "valid", "committed": True}
    assert report["4"]["committed"] is False
    # The far end's text is cut to printable ASCII and a short length.
    assert "\n" not in report["4"]["query"] and "\x1b" not in report["4"]["query"]
    assert len(report["4"]["query"]) <= 4 + 80
    json.dumps(report)


def test_the_collection_line_names_a_bounded_number_of_uids():
    n = 120
    hotkeys = [f"hk{i}" for i in range(n)]
    commitments = {hk: ("h", 5) for hk in hotkeys}
    _, line = validator._collection_report([Reply()] * n, hotkeys, commitments, {})
    assert line.count("uid ") == validator._COLLECTION_LINE_UIDS
    assert f"and {n - validator._COLLECTION_LINE_UIDS} more" in line


def test_the_collection_report_never_raises():
    report, line = validator._collection_report([object()], None, None, None)
    assert report == {} and line.startswith("Collection: 1 queried")


# --- version ------------------------------------------------------------------------


def test_the_package_version_is_the_projects():
    """A release that bumped pyproject.toml alone published reveals naming the
    previous version (v0.3.2 reported 0.3.1)."""
    import pathlib
    import tomllib

    with open(pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml", "rb") as f:
        assert tomllib.load(f)["project"]["version"] == fugal_subnet.__version__
