"""Three things the validator review of 2026-11-02 found the loop trusting
blindly: that the chain's commit-reveal version and weights_version match
what the SDK sends (findings 3/4), that a failed set_weights was worth one
try and no explanation (finding 3), and that the state file survives a power
loss between write and rename (finding 6). The helpers are module-level in
neurons/validator.py so they can be driven here with a fake chain."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

import pytest

VALIDATOR = pathlib.Path(__file__).resolve().parents[1] / "neurons" / "validator.py"


def _mod():
    spec = importlib.util.spec_from_file_location("validator_guards_under_test", VALIDATOR)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("validator_guards_under_test", mod)
    spec.loader.exec_module(mod)
    return mod


class _Chain:
    def __init__(self, *, crv=4, weights_version=0, cr_enabled=True, since=200, limit=100,
                 outcomes=None, block=1000):
        self.crv, self.wv, self.cr_enabled = crv, weights_version, cr_enabled
        self.since, self.limit = since, limit
        self.outcomes = list(outcomes or [(True, "ok")])
        self.block = block
        self.calls = 0

    # the surface assert_chain_weight_parameters touches
    def set_weights(self, wallet, netuid, uids, weights, wait_for_inclusion=True,
                    wait_for_finalization=True, commit_reveal_version: int = 4,
                    version_key: int = 10005000):
        self.calls += 1
        return self.outcomes.pop(0) if self.outcomes else (False, None)

    def query_subtensor(self, name, block=None, params=None):
        assert name == "CommitRevealWeightsVersion"
        return self.crv

    def get_subnet_hyperparameters(self, netuid):
        return types.SimpleNamespace(
            commit_reveal_weights_enabled=self.cr_enabled, commit_reveal_period=1,
            weights_rate_limit=self.limit, min_allowed_weights=1, max_weights_limit=65535,
            weights_version=self.wv)

    def blocks_since_last_update(self, netuid, uid):
        return self.since

    def weights_rate_limit(self, netuid):
        return self.limit

    def get_current_block(self):
        self.block += 1
        return self.block


# ------------------------------------------------ startup assertions --

def test_matching_chain_parameters_pass_and_are_reported():
    out = _mod().assert_chain_weight_parameters(_Chain(), 7, live=True)
    assert out["chain_commit_reveal_version"] == 4 == out["sdk_commit_reveal_version"]
    assert out["weights_rate_limit"] == 100


def test_a_commit_reveal_version_mismatch_refuses_live_and_warns_mock(caplog):
    mod = _mod()
    with pytest.raises(RuntimeError, match="CommitRevealWeightsVersion"):
        mod.assert_chain_weight_parameters(_Chain(crv=5), 7, live=True)
    mod.assert_chain_weight_parameters(_Chain(crv=5), 7, live=False)   # warns only
    assert any("every reveal would be rejected" in r.message for r in caplog.records)


def test_a_version_mismatch_is_ignored_when_commit_reveal_is_off():
    _mod().assert_chain_weight_parameters(_Chain(crv=9, cr_enabled=False), 7, live=True)


def test_a_weights_version_above_the_sdk_key_refuses_live():
    with pytest.raises(RuntimeError, match="weights_version"):
        _mod().assert_chain_weight_parameters(_Chain(weights_version=10_005_001), 7, live=True)


# ------------------------------------------------ set_weights retry --

def test_rate_limited_submit_is_named_and_never_attempted():
    chain = _Chain(since=40, limit=100)
    ok, msg = _mod().set_weights_with_retry(chain, None, 7, 3, [0], [1.0], deadline_block=2000)
    assert ok is False and "rate limit" in msg and "40 blocks" in msg
    assert chain.calls == 0


def test_a_transient_failure_is_retried_within_the_epoch(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    chain = _Chain(outcomes=[(False, "websocket closed"), (True, "Success")])
    ok, msg = mod.set_weights_with_retry(chain, None, 7, 3, [0], [1.0], deadline_block=2000)
    assert ok is True and chain.calls == 2


def test_retries_stop_at_the_epoch_deadline(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    chain = _Chain(outcomes=[(False, "boom")] * 5, block=1990)
    ok, msg = mod.set_weights_with_retry(chain, None, 7, 3, [0], [1.0], deadline_block=1991,
                                         attempts=5)
    assert ok is False and chain.calls == 1 and msg == "boom"


def test_retries_are_bounded(monkeypatch):
    mod = _mod()
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    chain = _Chain(outcomes=[(False, None)] * 10)
    ok, msg = mod.set_weights_with_retry(chain, None, 7, 3, [0], [1.0], deadline_block=10_000,
                                         attempts=3)
    assert ok is False and chain.calls == 3 and "no message" in msg


# ------------------------------------------------ durable state --

def _records(mod):
    from fugal_subnet.scoring import MinerRecord
    return {1: MinerRecord(uid=1, hotkey="hk1")}


def test_state_keeps_a_backup_and_falls_back_to_it_when_the_file_is_corrupt(tmp_path, caplog):
    mod = _mod()
    from fugal_subnet.scoring import MinerRecord
    path = str(tmp_path / "state.json")
    mod.save_state(_records(mod), [1], [1.0], 5, None, path=path)
    mod.save_state(_records(mod), [1], [1.0], 6, None, path=path)
    assert json.load(open(path))["last_epoch_index"] == 6
    assert json.load(open(path + ".bak"))["last_epoch_index"] == 5

    pathlib.Path(path).write_text("{\"truncated\": ")           # power loss mid-write
    state = mod.load_state(MinerRecord, path=path)
    assert state["last_epoch_index"] == 5
    assert any("resuming from the previous" in r.message for r in caplog.records)


def test_state_from_a_release_with_extra_fields_still_loads(tmp_path, caplog):
    mod = _mod()
    from fugal_subnet.scoring import MinerRecord
    path = str(tmp_path / "state.json")
    mod.save_state(_records(mod), [1], [1.0], 2, None, path=path)
    raw = json.load(open(path))
    raw["records"]["1"]["some_future_field"] = 42
    raw["first_commit_blocks"] = {"1": 5}
    pathlib.Path(path).write_text(json.dumps(raw))
    state = mod.load_state(MinerRecord, path=path)
    assert state["records"][1].hotkey == "hk1"
    assert any("some_future_field" in r.message for r in caplog.records)


def test_a_corrupt_file_with_no_backup_still_fails_loudly(tmp_path):
    mod = _mod()
    from fugal_subnet.scoring import MinerRecord
    path = str(tmp_path / "state.json")
    pathlib.Path(path).write_text("nope")
    with pytest.raises(ValueError, match="Cannot resume"):
        mod.load_state(MinerRecord, path=path)
