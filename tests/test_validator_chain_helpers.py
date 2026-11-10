"""The validator's small chain helpers: where unassigned weight burns, whether
the weights it set are really on chain, the v2 flip tie-break salt, each
miner's registration block, and the set-weights retry.

Four of these had no test at all, and 12 of 14 planted bugs survived the whole
suite (2026-11-08), among them a live validator burning to a guessed UID 0
(which pays whoever holds it) and a weight readback that confirms a mismatch.
"""
from types import SimpleNamespace

import pytest

from neurons import validator
from neurons.validator import (
    _activation_block_hash,
    confirm_weights_on_chain,
    registration_block_of,
    resolve_burn_uid,
    set_weights_with_retry,
)

OWNER = "5Owner"


# ---------------------------------------------------------- where weight burns --

class _Owners:
    def __init__(self, owner=OWNER, raises=False):
        self.owner, self.raises, self.asked = owner, raises, []

    def get_subnet_owner_hotkey(self, netuid, block=None):
        self.asked.append((netuid, block))
        if self.raises:
            raise ConnectionError("socket closed")
        return self.owner


META = SimpleNamespace(hotkeys=["5A", "5B", OWNER, "5C"])


@pytest.mark.parametrize("live", [True, False])
def test_the_burn_goes_to_the_owner_read_at_the_boundary(live):
    chain = _Owners()
    assert resolve_burn_uid(chain, 70, 4320, META, live) == 2
    assert chain.asked == [(70, 4320)]


def test_live_never_guesses_when_the_owner_cannot_be_read():
    # The specific reason, so the operator looks at the endpoint, not the owner.
    with pytest.raises(RuntimeError, match="Could not read the subnet owner hotkey at block 4320"):
        resolve_burn_uid(_Owners(raises=True), 70, 4320, META, live=True)


@pytest.mark.parametrize("owner", ["5NotRegistered", None, ""])
def test_live_never_guesses_when_the_owner_is_not_registered(owner):
    with pytest.raises(RuntimeError, match="refusing to burn to a guessed UID"):
        resolve_burn_uid(_Owners(owner=owner), 70, 4320, META, live=True)


def test_mock_falls_back_to_uid_0():
    assert resolve_burn_uid(_Owners(raises=True), 1, 10, META, live=False) == 0
    assert resolve_burn_uid(_Owners(owner="5Gone"), 1, 10, META, live=False) == 0


# --------------------------------------------------- weights really on chain --

class _Readback:
    def __init__(self, cr=False, last_update=100, head=110, rows=None, cr_raises=False,
                 weights_raise=False):
        self.cr, self.last_update, self.head = cr, last_update, head
        self.rows, self.cr_raises, self.weights_raise = rows or [], cr_raises, weights_raise

    def commit_reveal_enabled(self, netuid):
        if self.cr_raises:
            raise ConnectionError("probe failed")
        return self.cr

    def query_subtensor(self, name, params):
        assert name == "LastUpdate"
        return {7: self.last_update}

    def get_current_block(self):
        return self.head

    def weights(self, netuid):
        if self.weights_raise:
            raise ConnectionError("readback failed")
        return self.rows


def test_commit_reveal_confirms_the_commit_was_recorded():
    ok, detail = confirm_weights_on_chain(_Readback(cr=True, last_update=100, head=120), 1, 7, [0], [1.0])
    assert ok and "commit recorded at block 100" in detail


def test_commit_reveal_a_stale_last_update_is_not_confirmed():
    ok, detail = confirm_weights_on_chain(_Readback(cr=True, last_update=100, head=126), 1, 7, [0], [1.0])
    assert not ok and "not recorded" in detail


def test_readback_matching_weights_are_confirmed_after_normalising():
    # The chain stores u16-scaled weights; compared as proportions.
    rows = [(3, [(0, 16384), (2, 49151)]), (7, [(0, 16384), (2, 49151)])]
    ok, _ = confirm_weights_on_chain(_Readback(rows=rows), 1, 7, [0, 2], [0.25, 0.75])
    assert ok


def test_readback_a_mismatch_is_not_confirmed():
    rows = [(7, [(0, 49151), (2, 16384)])]
    ok, detail = confirm_weights_on_chain(_Readback(rows=rows), 1, 7, [0, 2], [0.25, 0.75])
    assert not ok and "uid 0" in detail


def test_readback_an_extra_uid_on_chain_is_a_mismatch():
    rows = [(7, [(0, 30000), (5, 30000)])]
    ok, _ = confirm_weights_on_chain(_Readback(rows=rows), 1, 7, [0], [1.0])
    assert not ok


def test_readback_no_row_for_this_validator_is_not_confirmed():
    ok, detail = confirm_weights_on_chain(_Readback(rows=[(3, [(0, 1)])]), 1, 7, [0], [1.0])
    assert not ok and "no weight row" in detail


def test_readback_failures_report_and_never_raise():
    ok, detail = confirm_weights_on_chain(_Readback(weights_raise=True), 1, 7, [0], [1.0])
    assert not ok and "readback failed" in detail
    # A failed commit-reveal probe falls back to the readback.
    ok, _ = confirm_weights_on_chain(_Readback(cr_raises=True, rows=[(7, [(0, 65535)])]),
                                     1, 7, [0], [1.0])
    assert ok


# ------------------------------------------------------- the v2 flip salt --

class _Hashes:
    def __init__(self, value="0xactivation"):
        self.value, self.asked = value, []

    def get_block_hash(self, block):
        self.asked.append(block)
        return self.value


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(validator, "_ACTIVATION_HASH", {})


def test_at_the_activation_boundary_the_salt_is_its_own_hash():
    chain = _Hashes()
    assert _activation_block_hash(chain, 3600, 3600, "0xboundary") == "0xboundary"
    assert chain.asked == []


def test_after_the_flip_the_salt_is_still_the_activation_blocks_hash():
    chain = _Hashes()
    assert _activation_block_hash(chain, 3600, 7200, "0xlater") == "0xactivation"
    assert chain.asked == [3600]
    # Cached: every later epoch uses the same salt without asking again.
    assert _activation_block_hash(chain, 3600, 10800, "0xlater2") == "0xactivation"
    assert chain.asked == [3600]


@pytest.mark.parametrize("value", [None, ""])
def test_an_unreadable_activation_hash_fails_the_epoch(value):
    with pytest.raises(RuntimeError, match="activation block 3600"):
        _activation_block_hash(_Hashes(value), 3600, 7200, "0xlater")


# ------------------------------------------------------- registration block --

def test_the_registration_block_comes_from_the_pinned_metagraph():
    meta = SimpleNamespace(block_at_registration=[10, 20, 30])
    assert registration_block_of(meta, 1) == 20


@pytest.mark.parametrize("meta,uid", [
    (SimpleNamespace(), 0),                                   # a mock metagraph
    (SimpleNamespace(block_at_registration=[10]), 5),         # uid beyond the list
    (SimpleNamespace(block_at_registration=["x"]), 0),        # unreadable value
])
def test_a_missing_registration_block_is_0_for_every_validator_alike(meta, uid):
    assert registration_block_of(meta, uid) == 0


# ------------------------------------------------------- set-weights retry --

class _Setter:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def blocks_since_last_update(self, netuid, uid):
        return 1000

    def weights_rate_limit(self, netuid):
        return 100

    def get_current_block(self):
        return 10

    def set_weights(self, **kw):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_a_raised_set_weights_is_a_failed_attempt_and_is_retried(monkeypatch):
    """The SDK's pre-flight reads raise on a dropped connection. That used to
    escape and fail the whole epoch after one call."""
    monkeypatch.setattr(validator.time, "sleep", lambda s: None)
    chain = _Setter([ConnectionError("websocket closed"), (True, "ok")])
    assert set_weights_with_retry(chain, None, 1, 0, [0], [1.0], deadline_block=100,
                                  attempts=3) == (True, "ok")
    assert chain.calls == 2


def test_set_weights_that_always_raises_fails_after_its_attempts(monkeypatch):
    monkeypatch.setattr(validator.time, "sleep", lambda s: None)
    chain = _Setter([ConnectionError("down")] * 5)
    ok, msg = set_weights_with_retry(chain, None, 1, 0, [0], [1.0], deadline_block=100, attempts=3)
    assert not ok and chain.calls == 3 and "raised ConnectionError" in msg


class _LandsThenDrops(_Setter):
    """The first attempt lands on chain and then the connection drops: the
    SDK raises, and the chain now reports this UID as just updated."""

    def __init__(self, on_chain_row, cr=False):
        super().__init__([ConnectionError("dropped while waiting for finalization")])
        self.row, self.cr, self.readbacks = on_chain_row, cr, 0

    def blocks_since_last_update(self, netuid, uid):
        return 2 if self.calls else 1000

    def commit_reveal_enabled(self, netuid):
        return self.cr

    def weights(self, netuid):
        self.readbacks += 1
        return [(0, self.row)]


def test_an_attempt_that_raised_but_landed_is_a_success(monkeypatch):
    """Without the readback, the rate limit made this a failure while the
    weights were on chain, and the cap baseline fell an epoch behind peers
    (independent review, 2026-11-08)."""
    monkeypatch.setattr(validator.time, "sleep", lambda s: None)
    chain = _LandsThenDrops(on_chain_row=[(0, 65535)])
    ok, msg = set_weights_with_retry(chain, None, 1, 0, [0], [1.0], deadline_block=100, attempts=3)
    assert ok and "had landed" in msg and chain.calls == 1 and chain.readbacks == 1


def test_an_attempt_that_raised_and_did_not_land_is_still_a_failure(monkeypatch):
    monkeypatch.setattr(validator.time, "sleep", lambda s: None)
    chain = _LandsThenDrops(on_chain_row=[(5, 65535)])          # someone else's weights
    ok, msg = set_weights_with_retry(chain, None, 1, 0, [0], [1.0], deadline_block=100, attempts=3)
    assert not ok and msg.startswith("rate limit")


def test_the_rate_limit_alone_reads_nothing_back(monkeypatch):
    class Limited(_LandsThenDrops):
        def blocks_since_last_update(self, netuid, uid):
            return 2
    chain = Limited(on_chain_row=[(0, 65535)])
    ok, msg = set_weights_with_retry(chain, None, 1, 0, [0], [1.0], deadline_block=100, attempts=3)
    assert not ok and msg.startswith("rate limit") and chain.calls == 0 and chain.readbacks == 0
