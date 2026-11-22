"""A follower's audit of an epoch whose menu carried (the root committed no
menu for it) compares the menu the epoch runs on with the carried one, which
is what the rule says it runs on; it does not compare its own step with a
step the root never published. A menu the root did commit is still compared
with the follower's own step, and a difference still raises the alarm
(fugal_subnet/menu_publish.MenuNode.audit; LIVE_MENU_V4.md 4.3)."""
from __future__ import annotations

import dataclasses
import logging

import pytest

from fugal_subnet import menu_publish as mp
from fugal_subnet import menu_state as ms
from tests import menu_fixtures as fx
from tests.test_menu_publish import BPE, World, changing, quiet


@pytest.fixture(autouse=True)
def _small_menus(monkeypatch):
    fx.allow_small_menus(monkeypatch)


@pytest.fixture
def signer(tmp_path, monkeypatch):
    from fugal_subnet import menu_rules as mr
    from tests.signed_list_fixtures import Signer

    s = Signer(tmp_path, "release")
    monkeypatch.setattr(mr, "list_signers", lambda: s.signers_line().encode())
    return s


def test_a_carried_menu_after_the_root_missed_its_epoch_raises_no_alarm(tmp_path, signer, caplog):
    w = World(tmp_path, signer)
    f = w.node("follower")
    audits = {}
    with caplog.at_level(logging.ERROR):
        for k in range(100, 104):
            w.chain.block = k * BPE
            # The follower's own step would change the menu every epoch (a
            # filing); the root is down for scored epochs 100 and 101.
            read, fields = w.follower_epoch(f, k, changing(k))
            audits[k] = fields[mp.REVEAL_AUDIT]
    for k in (102, 103):
        audit = audits[k]
        assert audit["status"] == "carried" and audit["root"] is None
        assert audit["differences"] == [], audit
        assert "committed no menu" in audit["why"]
        assert audit["own"] != ms.menu_hash(f.book.held(k))       # its step did move
    assert "ALARM" not in caplog.text


def test_a_menu_the_root_committed_unchanged_while_the_followers_step_moves_still_raises(
        tmp_path, signer, caplog):
    w = World(tmp_path, signer)
    f = w.node("follower")
    for k in (100, 101):
        w.chain.block = k * BPE
        w.follower_epoch(f, k, changing(k) if k == 100 else quiet(k))
        w.root_epoch(k, quiet(k))                    # the root's step saw no filing
    w.chain.block = 102 * BPE
    with caplog.at_level(logging.ERROR):
        read, fields = w.follower_epoch(f, 102, quiet(102))
    audit = fields[mp.REVEAL_AUDIT]
    assert read.status == "committed" and audit["differences"]
    assert "ALARM" in caplog.text


def test_a_carried_menu_that_is_not_the_carry_of_the_menu_before_raises(tmp_path, signer, caplog):
    """The audit's comparison for a carried epoch: the menu the epoch runs on
    against the carry of the one before. A node whose record of the two is
    inconsistent says so."""
    w = World(tmp_path, signer)
    f = w.node("follower")
    for k in (100, 101):
        w.chain.block = k * BPE
        w.follower_epoch(f, k, quiet(k))
    held = f.book.held(101)
    wrong = mp.MenuRead(102, dataclasses.replace(ms.carry(held), serial=held.serial + 5),
                        "carried")
    f.keep_inputs(100, quiet(100))
    with caplog.at_level(logging.ERROR):
        audit = f.audit(wrong)
    assert audit["differences"] == ["serial"]
    assert "ALARM" in caplog.text


def test_the_rehearsal_root_outage_shape_audits_clean(tmp_path, signer, caplog):
    """The root publishes for 100, misses 101 and 102, and is back for 103:
    every follower audit from 102 on is clean, and the menu carries exactly
    where no commitment names one."""
    w = World(tmp_path, signer)
    f = w.node("follower")
    with caplog.at_level(logging.ERROR):
        for k in range(100, 107):
            w.chain.block = k * BPE
            inputs = changing(k) if k in (101, 102) else quiet(k)
            read, fields = w.follower_epoch(f, k, inputs)
            if k not in (101, 102):
                w.root_epoch(k, inputs)
            if k >= 102:
                assert fields[mp.REVEAL_AUDIT]["differences"] == [], (k, fields[mp.REVEAL_AUDIT])
    assert "ALARM" not in caplog.text
