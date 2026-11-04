"""Protocol v2 activates at one named epoch boundary (I1).

Every validator must switch rules at the same epoch, and a retried or late
epoch must be scored under the same rules as on every peer. So the version is
a pure function of the epoch's boundary block and one configured block, the
block must itself be a boundary, and it is in the consensus digest.
docs/INVARIANTS.md, "I1 — protocol v2 activates at one named boundary".
"""
import pytest

from fugal_subnet.benchmarks.slicer import (
    blocks_per_epoch,
    epoch_index_for_block,
    protocol_version_for_boundary,
    validate_activation_block,
)

BPE = blocks_per_epoch(3600)  # 300 blocks
ACT = 1000 * BPE


def test_unset_activation_means_v1_forever():
    for boundary in (0, ACT, 10**9 * BPE):
        assert protocol_version_for_boundary(boundary, None) == 1
    validate_activation_block(None, BPE)  # valid: v2 never activates


def test_v2_applies_from_the_activation_boundary_itself():
    assert protocol_version_for_boundary(ACT - BPE, ACT) == 1
    assert protocol_version_for_boundary(ACT, ACT) == 2
    assert protocol_version_for_boundary(ACT + BPE, ACT) == 2


def test_every_block_of_an_epoch_gets_the_same_version():
    # The loop derives the boundary from whatever block it is at; any block in
    # the epoch, early or late, must give the same rules.
    for epoch in (999, 1000, 1001):
        versions = {
            protocol_version_for_boundary(epoch_index_for_block(b, BPE) * BPE, ACT)
            for b in range(epoch * BPE, (epoch + 1) * BPE)
        }
        assert len(versions) == 1, (epoch, versions)


@pytest.mark.parametrize("bad", [ACT + 1, ACT - 1, ACT + BPE // 2])
def test_an_activation_block_inside_an_epoch_is_refused(bad):
    with pytest.raises(ValueError, match="not an epoch boundary"):
        validate_activation_block(bad, BPE)


@pytest.mark.parametrize("bad", [-BPE, True, 1.0 * ACT, str(ACT)])
def test_a_malformed_activation_block_is_refused(bad):
    with pytest.raises(ValueError):
        validate_activation_block(bad, BPE)


def test_the_activation_block_is_read_from_the_environment(monkeypatch):
    import importlib

    from fugal_subnet import config

    monkeypatch.setenv("FUGAL_V2_ACTIVATION_BLOCK", str(ACT))
    try:
        assert importlib.reload(config).V2_ACTIVATION_BLOCK == ACT
        monkeypatch.setenv("FUGAL_V2_ACTIVATION_BLOCK", "")
        assert importlib.reload(config).V2_ACTIVATION_BLOCK is None
    finally:
        monkeypatch.delenv("FUGAL_V2_ACTIVATION_BLOCK", raising=False)
        importlib.reload(config)


def test_the_activation_block_is_in_the_consensus_digest(monkeypatch):
    from fugal_subnet import config
    from fugal_subnet.fingerprint import consensus_digest, environment_fingerprint

    monkeypatch.setattr(config, "V2_ACTIVATION_BLOCK", None)
    unset = consensus_digest()
    monkeypatch.setattr(config, "V2_ACTIVATION_BLOCK", ACT)
    assert environment_fingerprint()["epoch"]["v2_activation_block"] == ACT
    at_act = consensus_digest()
    monkeypatch.setattr(config, "V2_ACTIVATION_BLOCK", ACT + BPE)
    later = consensus_digest()
    assert len({unset, at_act, later}) == 3


def test_the_activation_block_is_a_consensus_input_not_refused():
    # Set per deployment, like the epoch length: recorded, never refused.
    from fugal_subnet import config

    assert "FUGAL_V2_ACTIVATION_BLOCK" in config.CONSENSUS_ENV_INPUTS
    assert "FUGAL_V2_ACTIVATION_BLOCK" not in config.CONSENSUS_ENV_VARS


V2_CONSTANTS = {
    "CHAMPION_ACCURACY_VALUE": ("FUGAL_CHAMPION_ACCURACY_VALUE", 30.0),
    "CHAMPION_HARD_WEIGHT": ("FUGAL_CHAMPION_HARD_WEIGHT", 2.0),
    "CHAMPION_ANCHOR_MODEL": ("FUGAL_CHAMPION_ANCHOR_MODEL", "anthropic/claude-fable-5.1"),
    "CHAMPION_MARGIN": ("FUGAL_CHAMPION_MARGIN", 0.2),
    "CHAMPION_CONFIDENCE": ("FUGAL_CHAMPION_CONFIDENCE", 0.90),
    "CHAMPION_BOOTSTRAP_RESAMPLES": ("FUGAL_CHAMPION_BOOTSTRAP_RESAMPLES", 2000),
    "CHAMPION_HARD_GATE_MIN_QUESTIONS": ("FUGAL_CHAMPION_HARD_GATE_MIN_QUESTIONS", 100),
    "CHAMPION_TIME_GATE_FACTOR": ("FUGAL_CHAMPION_TIME_GATE_FACTOR", 2.0),
    "CHAMPION_GHOST_OFF": ("FUGAL_CHAMPION_GHOST_OFF", 3),
    "CHAMPION_GHOST_RETIRE": ("FUGAL_CHAMPION_GHOST_RETIRE", 24),
    "CHAMPION_WINDOW_EPOCHS": ("FUGAL_CHAMPION_WINDOW_EPOCHS", 48),
    "CHAMPION_MIN_ACCURACY": ("FUGAL_CHAMPION_MIN_ACCURACY", 0.30),
    "V2_BURN_IN_QUESTIONS": ("FUGAL_V2_BURN_IN_QUESTIONS", 1500),
    "CHAMPION_MIN_PAIRED_QUESTIONS": ("FUGAL_CHAMPION_MIN_PAIRED_QUESTIONS", 1500),
    "PAYOUT_SEAT_SHARES": ("FUGAL_PAYOUT_SEAT_SHARES", (0.85, 0.10, 0.05)),
}


@pytest.mark.parametrize("attr", sorted(V2_CONSTANTS))
def test_v2_constants_have_the_approved_values_and_are_consensus(attr, monkeypatch):
    from fugal_subnet import config
    from fugal_subnet.fingerprint import consensus_digest, environment_fingerprint

    env, approved = V2_CONSTANTS[attr]
    assert getattr(config, attr) == approved
    assert env in config.CONSENSUS_ENV_VARS
    assert attr in environment_fingerprint()["scoring"]
    before = consensus_digest()
    moved = (0.8, 0.15, 0.05) if attr == "PAYOUT_SEAT_SHARES" else approved * 2
    monkeypatch.setattr(config, attr, moved)
    assert consensus_digest() != before, f"{attr} moved and the digest did not"


def test_seat_shares_sum_to_one():
    from fugal_subnet.config import PAYOUT_SEAT_SHARES

    assert len(PAYOUT_SEAT_SHARES) == 3
    assert abs(sum(PAYOUT_SEAT_SHARES) - 1.0) < 1e-12
