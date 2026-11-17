"""v3 review (2026-11-16), finding 6.

- The validator checked only the v2 block for epoch alignment; an unaligned
  v3 (or odds-seat) block was accepted by validators but refused by the v3
  miner, so validators would score a flip no v3 miner could start at.
- The cost-aware exploration constants decide every pair's sampling
  probability and weight, and were missing from the consensus digest.
"""
import click
import pytest

from fugal_subnet import config
from fugal_subnet.benchmarks.slicer import blocks_per_epoch


def test_the_validator_refuses_an_unaligned_v3_or_odds_seat_block():
    from neurons.validator import _check_v3_blocks_aligned

    bpe = blocks_per_epoch(config.EPOCH_INTERVAL)
    _check_v3_blocks_aligned(None, None, config.EPOCH_INTERVAL)
    _check_v3_blocks_aligned(10 * bpe, 18 * bpe, config.EPOCH_INTERVAL)
    with pytest.raises(click.ClickException, match="v3 activation block"):
        _check_v3_blocks_aligned(10 * bpe + 1, None, config.EPOCH_INTERVAL)
    with pytest.raises(click.ClickException, match="odds-seat block"):
        _check_v3_blocks_aligned(10 * bpe, 18 * bpe + 7, config.EPOCH_INTERVAL)


def test_the_validator_runs_the_check_on_the_blocks_it_read():
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "neurons" / "validator.py").read_text()
    read = src.index("odds_block = _cfg.odds_seat_block(genesis)")
    check = src.index("_check_v3_blocks_aligned(v3_block, odds_block, EPOCH_INTERVAL, menu_b_blk)")
    assert read < check < src.index("metagraph = subtensor.metagraph(netuid)", read)


@pytest.mark.parametrize("name, other", [
    ("EXPLORE_SPEND_CAP_USD", "0.30"),
    ("EXPLORE_COST_FLOOR", "1/3"),
    ("EXPLORE_COST_MAX_HALVES", 5),
])
def test_the_cost_aware_exploration_constants_are_in_the_digest(monkeypatch, name, other):
    from fugal_subnet.fingerprint import environment_fingerprint

    before = environment_fingerprint()
    assert before["scoring"][name] == getattr(config, name)
    monkeypatch.setattr(config, name, other)
    assert environment_fingerprint()["scoring"][name] == other
