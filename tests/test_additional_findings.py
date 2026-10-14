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


@pytest.mark.xfail(reason="A2: scoring constants are outside consensus_digest", **XF)
def test_scoring_constants_change_the_consensus_digest(monkeypatch):
    from fugal_subnet import config
    from fugal_subnet.fingerprint import consensus_digest

    before = consensus_digest()
    monkeypatch.setattr(config, "SCORE_QUALITY_FLOOR", 0.7)
    assert consensus_digest() != before, "SCORE_QUALITY_FLOOR moved and the digest did not"


@pytest.mark.xfail(reason="A2: no classification of FUGAL_ env constants", **XF)
def test_every_score_affecting_constant_is_classified_as_consensus():
    from fugal_subnet.config import CONSENSUS_ENV_VARS

    for name in ("FUGAL_SCORE_QUALITY_FLOOR", "FUGAL_FRONTIER_MIN_TRIALS",
                 "FUGAL_DEDUP_THRESHOLD", "FUGAL_FRAME_PRIOR_STRENGTH",
                 "FUGAL_FRAME_PRIOR_ACCURACY", "FUGAL_FRAME_HALF_LIFE",
                 "FUGAL_BURN_IN_QUESTIONS", "FUGAL_MAX_WEIGHT_DELTA"):
        assert name in CONSENSUS_ENV_VARS, name


@pytest.mark.xfail(reason="A3: validator defaults to --mock", **XF)
def test_validator_defaults_to_live():
    from neurons.validator import main

    live = next(p for p in main.params if p.name == "live")
    assert live.default is True


@pytest.mark.xfail(reason="A3: nothing refuses mock on finney", **XF)
def test_mock_is_refused_on_finney():
    from fugal_subnet.config import FINNEY_GENESIS_HASH
    from fugal_subnet.fingerprint import assert_mock_allowed

    with pytest.raises(RuntimeError, match="finney"):
        assert_mock_allowed(genesis_hash=FINNEY_GENESIS_HASH, mock=True)
    assert_mock_allowed(genesis_hash=FINNEY_GENESIS_HASH, mock=False)
    assert_mock_allowed(genesis_hash="0x" + "00" * 32, mock=True)


@pytest.mark.xfail(reason="A4: commitments read at chain head", **XF)
def test_commitments_are_read_at_the_requested_block():
    from fugal_subnet.commitments import get_commitments_with_blocks

    seen = {}

    class Sub:
        def query_map(self, module, name, params=None, block=None):
            seen.update(module=module, name=name, params=params, block=block)
            return []

    get_commitments_with_blocks(Sub(), netuid=7, block=12345)
    assert seen["block"] == 12345


@pytest.mark.xfail(reason="A5: BURN_UID is a hard-coded constant", **XF)
def test_burn_goes_to_the_uid_the_caller_resolved():
    from fugal_subnet import rewards
    from fugal_subnet.scoring import MinerRecord

    assert not hasattr(rewards, "BURN_UID")
    records = {3: MinerRecord(uid=3)}
    uids, weights = rewards.compute_weights(records, burn_uid=9, paid_fraction=0.5)
    assert dict(zip(uids, weights)).get(9) == pytest.approx(0.5)
