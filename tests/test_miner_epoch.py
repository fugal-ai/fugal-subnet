"""The miner's epoch loop: start only what can finish, and say whether it did.

From the neurons/ review (2026-11-08): the late-start guard had no test
(planting a no-op survived everything; it is the rule a too-short rehearsal
epoch kept tripping), the miner stated its deadline but never whether it met
it, and v1 handed the proxy no prices even when main() had loaded others.
"""
import ast
import logging
import pathlib

import pytest

import neurons.miner as miner
from fugal_subnet.benchmarks.slicer import collect_block_for_epoch
from fugal_subnet.config import EPOCH_COLLECT_FRACTION, LATE_START_MARGIN_BLOCKS

SRC = pathlib.Path(miner.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("bpe", [300, 360])
def test_an_epoch_starts_only_with_the_margin_to_go(bpe):
    epoch = 40
    deadline = collect_block_for_epoch(epoch, bpe, EPOCH_COLLECT_FRACTION)
    assert miner._should_start_epoch(deadline - LATE_START_MARGIN_BLOCKS, epoch, bpe) == (True, deadline)
    assert miner._should_start_epoch(deadline - LATE_START_MARGIN_BLOCKS + 1, epoch, bpe) == (False, deadline)
    assert miner._should_start_epoch(epoch * bpe, epoch, bpe)[0]        # at the boundary
    assert not miner._should_start_epoch(deadline + 5, epoch, bpe)[0]   # past collection


def test_readiness_is_reported_both_ways(caplog):
    with caplog.at_level(logging.INFO, logger="fugal.miner"):
        miner._log_readiness("e00000040", 100, 150)
        miner._log_readiness("e00000041", 160, 150)
    msgs = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert any(lvl == logging.INFO and "50 blocks before collection" in m for lvl, m in msgs)
    assert any(lvl == logging.WARNING and "LATE by 10 blocks" in m for lvl, m in msgs)


def test_the_loop_uses_the_guard_and_hands_the_proxy_its_prices():
    fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.While))
    calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "_should_start_epoch" in {c.func.id for c in calls}
    run = next(c for c in calls if c.func.id == "_run_epoch")
    kw = {k.arg: ast.unparse(k.value) for k in run.keywords}
    assert kw["prices"] == "prices" and "block_now" in kw
