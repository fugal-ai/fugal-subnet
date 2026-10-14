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


@pytest.mark.xfail(reason="A1: no reveal publisher exists", **XF)
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


def test_an_unreadable_genesis_does_not_pass_mock():
    from fugal_subnet.fingerprint import assert_mock_allowed, chain_genesis_hash

    class Dead:
        def get_block_hash(self, n):
            raise ConnectionError("no chain")

    assert chain_genesis_hash(Dead()) is None
    with pytest.raises(RuntimeError, match="could not be read"):
        assert_mock_allowed(None, mock=True)
    assert_mock_allowed(None, mock=False)
