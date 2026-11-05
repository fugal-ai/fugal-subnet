"""Protocol v2 image wiring: per-model reasoning on every v2 call, and the miner's hold."""
import sys

from fugal_subnet.routing_protocol import models_for_protocol, reasoning_for
from fugal_subnet.tee import harness


class _Proxy:
    records = []


def test_v2_calls_carry_each_models_pinned_reasoning_and_v1_calls_carry_none(monkeypatch):
    seen = []

    def fake_call(proxy, model_id, question, reasoning=None):
        seen.append((model_id, reasoning))
        return "ok"

    monkeypatch.setattr(harness, "_call_model", fake_call)
    m2 = models_for_protocol(2)[:3]
    list(harness._call_all(_Proxy(), [({"question_id": f"q{i}"}, m, False) for i, m in enumerate(m2)], "e1", 2))
    assert sorted(seen) == sorted((m, reasoning_for(2, m)) for m in m2)    # calls run on a thread pool
    seen.clear()
    m1 = models_for_protocol(1)[:2]
    list(harness._call_all(_Proxy(), [({"question_id": f"q{i}"}, m, False) for i, m in enumerate(m1)], "e1", 1))
    assert sorted(seen) == sorted((m, None) for m in m1)


def test_the_miner_holds_until_the_target_block(monkeypatch):
    sys.path.insert(0, "neurons")
    import miner  # noqa: E402

    blocks = iter([100, 150, 199, 200, 201])

    class _Sub:
        def get_current_block(self):
            return next(blocks)

    monkeypatch.setattr(miner.time, "sleep", lambda s: None)
    miner._wait_for_block(_Sub(), 200, "serve the axon")
    assert next(blocks) == 201          # returned exactly when the chain reached 200


def test_the_miner_never_prices_a_head_with_the_v1_only_function():
    """Structural: the miner must price/validate heads with benchmark_costs_for(head, version).
    A bare v1 benchmark_costs() call refused every genuine v2 head at startup (caught by the v2
    flip rehearsal, missed by unit tests that never start a v2 miner with a v2 head)."""
    import ast
    import pathlib
    tree = ast.parse((pathlib.Path(__file__).resolve().parent.parent / "neurons" / "miner.py").read_text())
    names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "benchmark_costs" not in names
    assert "benchmark_costs_for" in names


class _FlakySub:
    """Fake subtensor: the chain advances one block per read; the first `fails` commits fail."""

    def __init__(self, start, fails, raise_on_fail=False):
        self.block, self.fails, self.raise_on_fail = start, fails, raise_on_fail
        self.attempts, self.committed = [], None

    def get_current_block(self):
        self.block += 1
        return self.block

    def set_commitment(self, wallet, netuid, data):
        self.attempts.append(self.block)
        if len(self.attempts) <= self.fails:
            if self.raise_on_fail:
                raise RuntimeError("transaction pool full")
            return False, "Priority is too low"
        self.committed = data
        return True, ""


class _Wallet:
    class hotkey:
        ss58_address = "5Fake"


def _miner(monkeypatch):
    sys.path.insert(0, "neurons")
    import miner  # noqa: E402

    from fugal_subnet import commitments
    monkeypatch.setattr(miner.time, "sleep", lambda s: None)
    monkeypatch.setattr(commitments, "get_commitments_with_blocks", lambda sub, netuid: {})
    return miner


def test_the_pre_flip_commit_retries_failed_transactions_until_it_lands(monkeypatch):
    """A single failed commit tx used to leave the head to the post-flip fallback, so
    validators reading commitments at the flip boundary rejected the first v2 epoch(s)."""
    miner = _miner(monkeypatch)
    for raise_on_fail in (False, True):
        sub = _FlakySub(start=1000, fails=3, raise_on_fail=raise_on_fail)
        assert miner._commit_before_deadline(sub, _Wallet(), 70, "ab" * 32, deadline=1100, retry_blocks=2)
        assert sub.committed == "ab" * 32 and len(sub.attempts) == 4
        assert all(b - a >= 2 for a, b in zip(sub.attempts, sub.attempts[1:]))   # spaced by >= 2 blocks
        assert sub.attempts[-1] < 1100


def test_the_pre_flip_commit_gives_up_loudly_at_the_deadline(monkeypatch, caplog):
    miner = _miner(monkeypatch)
    sub = _FlakySub(start=1000, fails=10**6)
    with caplog.at_level("WARNING"):
        assert miner._commit_before_deadline(sub, _Wallet(), 70, "ab" * 32, deadline=1030, retry_blocks=2) is False
    assert sub.committed is None and sub.attempts and max(sub.attempts) < 1030
    assert any(r.levelname == "ERROR" and "NOT committed by block 1030" in r.getMessage() for r in caplog.records)
    assert sum("commit attempt" in r.getMessage() for r in caplog.records) == len(sub.attempts)


def test_a_chain_read_error_does_not_kill_the_pre_flip_commit(monkeypatch):
    miner = _miner(monkeypatch)
    sub = _FlakySub(start=1000, fails=1)
    real, hiccups = sub.get_current_block, iter([True, False, False, True])

    def flaky_block():
        if next(hiccups, False):
            raise ConnectionError("websocket closed")
        return real()

    sub.get_current_block = flaky_block
    assert miner._commit_before_deadline(sub, _Wallet(), 70, "ab" * 32, deadline=1100, retry_blocks=2)
    assert sub.committed == "ab" * 32


def test_the_commit_deadline_sits_after_the_commit_block_and_before_the_flip():
    from fugal_subnet.benchmarks.slicer import blocks_per_epoch
    from fugal_subnet.config import (
        COMMIT_AFTER_BOUNDARY_BLOCKS,
        COMMIT_DEADLINE_BEFORE_FLIP_BLOCKS,
        COMMIT_RETRY_BLOCKS,
        EPOCH_INTERVAL,
    )
    bpe = blocks_per_epoch(EPOCH_INTERVAL)
    assert 0 < COMMIT_RETRY_BLOCKS < COMMIT_DEADLINE_BEFORE_FLIP_BLOCKS
    assert COMMIT_AFTER_BOUNDARY_BLOCKS + 10 * COMMIT_RETRY_BLOCKS < bpe - COMMIT_DEADLINE_BEFORE_FLIP_BLOCKS
