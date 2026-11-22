"""Protocol v4 mirrors (docs/decisions/LIVE_MENU_V4.md, 4.3, the root's
availability): a validator with a bucket re-publishes each menu it accepted,
and each signed list it holds for it, under the same content-addressed names
in its own bucket; every reader takes mirror locations from one environment
variable, FUGAL_MENU_MIRRORS, and checks whatever a mirror serves against the
hash it asked for. Nothing depends on a mirror existing.
"""
from __future__ import annotations

import hashlib

import pytest

from fugal_subnet import config, reveal_sync
from fugal_subnet import menu_publish as mp
from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from tests import menu_fixtures as fx
from tests.test_menu_publish import (
    BPE,
    LIST_BYTES,
    NETUID,
    Clock,
    Store,
    World,
    changing,
    signer,  # noqa: F401 - the fixture
)

MIRROR_URL = "https://storage.googleapis.com/fugal-mirror"


@pytest.fixture(autouse=True)
def _small_menus(monkeypatch):
    fx.allow_small_menus(monkeypatch)


def test_the_mirror_variable_is_operational_and_parsed_as_https_urls(monkeypatch):
    assert "FUGAL_MENU_MIRRORS" in config.OPERATIONAL_ENV_VARS
    assert config.parse_menu_mirrors("") == ()
    assert config.parse_menu_mirrors(f" {MIRROR_URL}/ ,https://m2.example/x,,") == (
        MIRROR_URL, "https://m2.example/x")


def test_a_validator_with_a_bucket_mirrors_the_menus_and_lists_it_accepted(tmp_path, signer):  # noqa: F811
    w = World(tmp_path, signer)
    f = w.node("follower")
    mirror = Store([], Clock())
    m = mp.Mirror(netuid=NETUID, store=mirror)
    for k in range(100, 103):
        w.chain.block = k * BPE
        read = w.follower_epoch(f, k, changing(k) if k == 100 else fx.inputs(epoch=k))[0]
        w.root_epoch(k, changing(k) if k == 100 else fx.inputs(epoch=k))
        m.mirror(read, f.lists)
    h = w.chain.read(102)
    assert mirror.objects[reveal_sync.menu_object(NETUID, h)] == w.store.objects[
        reveal_sync.menu_object(NETUID, h)]
    sha = fx.ALLOW.sha256
    assert mirror.objects[reveal_sync.list_object(NETUID, mr.KIND_MENU_ALLOWLIST, sha)] == LIST_BYTES
    assert reveal_sync.signature_object(NETUID, mr.KIND_MENU_ALLOWLIST, sha) in mirror.objects
    # Each object once: a menu already mirrored is not put again.
    puts = [key for what, key in mirror.log if what == "put"]
    assert len(puts) == len(set(puts))
    # A reader that has only the mirror gets the same bytes, checked by hash.
    got = reveal_sync.fetch_menu_bytes(h, bucket=None, netuid=NETUID, mirrors=(MIRROR_URL,),
                                       opener=mirror.opener)
    assert got is not None and hashlib.sha256(got).hexdigest() == h
    assert reveal_sync.fetch_signed_list(mr.KIND_MENU_ALLOWLIST, sha, bucket=None, netuid=NETUID,
                                         mirrors=(MIRROR_URL,), opener=mirror.opener)[0] == LIST_BYTES


def test_a_mirror_that_fails_costs_nothing(tmp_path, signer):  # noqa: F811
    w = World(tmp_path, signer)
    f = w.node("follower")
    broken = Store([], Clock())
    broken.fail_put = True
    w.chain.block = 100 * BPE
    read = w.follower_epoch(f, 100, changing(100))[0]
    mp.Mirror(netuid=NETUID, store=broken).mirror(read, f.lists)      # never raises
    assert read.ok and broken.objects == {}


def test_readers_take_the_mirrors_from_the_environment_by_default(monkeypatch):
    raw = ms.canonical_json(fx.state())
    h = hashlib.sha256(raw).hexdigest()
    mirror = Store([], Clock())
    mirror.objects[reveal_sync.menu_object(NETUID, h)] = raw
    monkeypatch.setattr(config, "MENU_MIRRORS", (MIRROR_URL,))
    assert reveal_sync.fetch_menu_bytes(h, bucket=None, netuid=NETUID,
                                        opener=mirror.opener) == raw
    monkeypatch.setattr(config, "MENU_MIRRORS", ())
    assert reveal_sync.fetch_menu_bytes(h, bucket=None, netuid=NETUID,
                                        opener=mirror.opener) is None
