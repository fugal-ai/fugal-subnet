"""Diligence 2026-10, the six additional findings (not exploits; close before mainnet).

  A1  the reveal is written to the validator's disk and never published
  A2  score-affecting constants are env-overridable and outside the consensus digest
  A3  the validator defaults to mock, which accepts unattested proofs
      (the finney guard lives in fingerprint.py beside assert_environment)
  A4  on-chain commitments are read at the current block, not the boundary
  A5  burn goes to a hard-coded UID 0
  A6  docs contradict the code (no test; see the docs sweep PR)

Each test pins the required behaviour and fails until its PR lands; strict
xfail so the marker leaves with the fix.
"""
from __future__ import annotations

import pytest

XF = dict(strict=True, raises=(AssertionError, TypeError, AttributeError, ImportError))


def test_reveal_publisher_exists_and_hashes_what_it_uploads():
    from fugal_subnet.reveal_publisher import reveal_digest

    assert len(reveal_digest(b'{"epoch_id": "e1"}')) == 64


def test_scoring_constants_change_the_consensus_digest(monkeypatch):
    from fugal_subnet import config
    from fugal_subnet.fingerprint import consensus_digest

    before = consensus_digest()
    monkeypatch.setattr(config, "SCORE_QUALITY_FLOOR", 0.7)
    assert consensus_digest() != before, "SCORE_QUALITY_FLOOR moved and the digest did not"


def test_every_score_affecting_constant_is_classified_as_consensus():
    from fugal_subnet.config import CONSENSUS_ENV_VARS

    for name in ("FUGAL_SCORE_QUALITY_FLOOR", "FUGAL_FRONTIER_MIN_TRIALS",
                 "FUGAL_DEDUP_THRESHOLD", "FUGAL_FRAME_PRIOR_STRENGTH",
                 "FUGAL_FRAME_PRIOR_ACCURACY", "FUGAL_FRAME_HALF_LIFE",
                 "FUGAL_BURN_IN_QUESTIONS", "FUGAL_MAX_WEIGHT_DELTA"):
        assert name in CONSENSUS_ENV_VARS, name


def test_validator_defaults_to_live():
    from neurons.validator import main

    live = next(p for p in main.params if p.name == "live")
    assert live.default is True


def test_mock_is_refused_on_finney():
    from fugal_subnet.config import FINNEY_GENESIS_HASH
    from fugal_subnet.fingerprint import assert_mock_allowed

    with pytest.raises(RuntimeError, match="finney"):
        assert_mock_allowed(genesis_hash=FINNEY_GENESIS_HASH, mock=True)
    assert_mock_allowed(genesis_hash=FINNEY_GENESIS_HASH, mock=False)
    assert_mock_allowed(genesis_hash="0x" + "00" * 32, mock=True)


def test_commitments_are_read_at_the_requested_block():
    from fugal_subnet.commitments import get_commitments_with_blocks

    seen = {}

    class Sub:
        def query_map(self, module, name, params=None, block=None):
            seen.update(module=module, name=name, params=params, block=block)
            return []

    get_commitments_with_blocks(Sub(), netuid=7, block=12345)
    assert seen["block"] == 12345


def test_burn_goes_to_the_uid_the_caller_resolved():
    from fugal_subnet import rewards
    from fugal_subnet.scoring import MinerRecord

    assert not hasattr(rewards, "BURN_UID")
    records = {3: MinerRecord(uid=3, composite_score=0.1)}
    uids, weights = rewards.compute_weights(records, burn_uid=9, paid_fraction=0.5)
    assert dict(zip(uids, weights)) == {3: pytest.approx(0.5), 9: pytest.approx(0.5)}
    # A field that earns nothing burns everything, to the resolved UID.
    uids, weights = rewards.compute_weights({3: MinerRecord(uid=3)}, burn_uid=9)
    assert dict(zip(uids, weights)) == {9: pytest.approx(1.0)}


def test_a_live_validator_refuses_an_overridden_consensus_constant(monkeypatch):
    from fugal_subnet.fingerprint import check_environment

    monkeypatch.delenv("FUGAL_SCORE_QUALITY_FLOOR", raising=False)
    before = [p for p in check_environment() if "FUGAL_SCORE_QUALITY_FLOOR" in p]
    monkeypatch.setenv("FUGAL_SCORE_QUALITY_FLOOR", "0.5")
    after = [p for p in check_environment() if "FUGAL_SCORE_QUALITY_FLOOR" in p]
    assert not before and after and "cannot be overridden" in after[0]
    # The neurons defer this check to the genesis-aware one below.
    deferred = [p for p in check_environment(consensus_overrides=False)
                if "FUGAL_SCORE_QUALITY_FLOOR" in p]
    assert not deferred


def test_consensus_overrides_are_refused_on_finney_and_warned_elsewhere(monkeypatch, caplog):
    import logging

    from fugal_subnet.config import FINNEY_GENESIS_HASH
    from fugal_subnet.fingerprint import assert_consensus_overrides_allowed

    monkeypatch.delenv("FUGAL_EXPLORE_FRACTION", raising=False)
    assert_consensus_overrides_allowed(FINNEY_GENESIS_HASH)  # nothing set: nothing to refuse
    assert_consensus_overrides_allowed(None)

    monkeypatch.setenv("FUGAL_EXPLORE_FRACTION", "0.5")
    with pytest.raises(RuntimeError, match="finney"):
        assert_consensus_overrides_allowed(FINNEY_GENESIS_HASH)
    with pytest.raises(RuntimeError, match="genesis"):
        assert_consensus_overrides_allowed(None)
    # The drift flag downgrades library and dispatch problems; it does not lift this.
    monkeypatch.setenv("FUGAL_ALLOW_ENV_DRIFT", "1")
    with pytest.raises(RuntimeError, match="finney"):
        assert_consensus_overrides_allowed(FINNEY_GENESIS_HASH)
    # A test chain: allowed, named in a warning that carries the digest.
    with caplog.at_level(logging.WARNING, logger="fugal_subnet.fingerprint"):
        assert_consensus_overrides_allowed("0x" + "00" * 32)
    assert any("FUGAL_EXPLORE_FRACTION" in r.getMessage() for r in caplog.records)


def test_both_neurons_check_consensus_overrides_against_the_genesis():
    import inspect

    import neurons.miner
    import neurons.validator

    for mod in (neurons.validator, neurons.miner):
        assert "assert_consensus_overrides_allowed(genesis)" in inspect.getsource(mod), mod.__name__


def test_an_unreadable_genesis_does_not_pass_mock():
    from fugal_subnet.fingerprint import assert_mock_allowed, chain_genesis_hash

    class Dead:
        def get_block_hash(self, n):
            raise ConnectionError("no chain")

    assert chain_genesis_hash(Dead()) is None
    with pytest.raises(RuntimeError, match="could not be read"):
        assert_mock_allowed(None, mock=True)
    assert_mock_allowed(None, mock=False)


# --- review 2026-11-05, F-027/F-028: the whole package is classified; the code is named ---


def test_every_fugal_variable_read_anywhere_in_the_package_is_classified():
    import pathlib
    import re

    from fugal_subnet import config

    root = pathlib.Path(__file__).resolve().parents[1]
    pattern = re.compile(r'os\.(?:getenv|environ\.get)\(\s*"(FUGAL_[A-Z0-9_]+)"'
                         r'|os\.environ\[\s*"(FUGAL_[A-Z0-9_]+)"\s*\](?!\s*=)')
    read = set()
    for path in list((root / "fugal_subnet").rglob("*.py")) + list((root / "neurons").glob("*.py")):
        for a, b in pattern.findall(path.read_text(encoding="utf-8")):
            read.add(a or b)
    listed = set(config.CONSENSUS_ENV_VARS) | set(config.CONSENSUS_ENV_INPUTS) \
        | set(config.OPERATIONAL_ENV_VARS)
    assert read - listed == set(), read - listed
    for name in ("FUGAL_SKIP_BENCHMARKS", "FUGAL_POOL_UNPINNED", "FUGAL_OPENROUTER_BASE"):
        assert name in config.CONSENSUS_ENV_VARS, name


def test_the_pool_inputs_move_the_consensus_digest(monkeypatch):
    from fugal_subnet.fingerprint import consensus_digest, environment_fingerprint

    monkeypatch.delenv("FUGAL_SKIP_BENCHMARKS", raising=False)
    monkeypatch.delenv("FUGAL_POOL_UNPINNED", raising=False)
    base = consensus_digest()
    fp = environment_fingerprint()
    assert fp["epoch"]["pool_manifest_v1"] and fp["epoch"]["pool_manifest_v2"]
    assert fp["epoch"]["pool_unpinned"] is False
    monkeypatch.setenv("FUGAL_SKIP_BENCHMARKS", "mmlu")
    assert consensus_digest() != base, "the skip list moved and the digest did not"
    monkeypatch.delenv("FUGAL_SKIP_BENCHMARKS")
    monkeypatch.setenv("FUGAL_POOL_UNPINNED", "1")
    assert consensus_digest() != base, "verification was switched off and the digest did not move"


def test_an_unpinned_pool_or_a_skip_list_is_refused_on_finney(monkeypatch):
    from fugal_subnet.config import FINNEY_GENESIS_HASH
    from fugal_subnet.fingerprint import assert_consensus_overrides_allowed

    for name in ("FUGAL_POOL_UNPINNED", "FUGAL_SKIP_BENCHMARKS", "FUGAL_OPENROUTER_BASE"):
        monkeypatch.delenv(name, raising=False)
    assert_consensus_overrides_allowed(FINNEY_GENESIS_HASH)
    for name in ("FUGAL_POOL_UNPINNED", "FUGAL_SKIP_BENCHMARKS"):
        monkeypatch.setenv(name, "1")
        with pytest.raises(RuntimeError, match="finney"):
            assert_consensus_overrides_allowed(FINNEY_GENESIS_HASH)
        monkeypatch.delenv(name)


def test_the_reveal_names_the_code_but_the_digest_does_not():
    """Two validators on different commits whose score-affecting inputs agree
    are not a divergence; when they do diverge the reveal must say which code
    each ran (F-028)."""
    import neurons.miner as miner
    from fugal_subnet.fingerprint import consensus_digest, environment_fingerprint, source_sha256

    fp = environment_fingerprint()
    assert fp["code"]["source_sha256"] == source_sha256() == miner._get_source_hash()
    assert len(fp["code"]["source_sha256"]) == 64
    altered = dict(fp, code={"source_sha256": "0" * 64, "git_commit": ""})
    assert consensus_digest(altered) == consensus_digest(fp)
