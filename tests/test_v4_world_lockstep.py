"""The loop harness (tests/v4_world.Lockstep) fails a run at once, with the
original exception, when a miner's epoch raises: the validators' loops catch
every Exception of an epoch, so without that the run went on (or waited on
the clock) and the cause surfaced, if at all, as a stall."""
from __future__ import annotations

import threading
import time
import types

import pytest

from tests.v4_world import BPE, Lockstep


def _loop(clock):
    """What validator.main does around its sleeps: an epoch's Exception is
    logged and the loop goes on; KeyboardInterrupt stops it."""
    def run():
        while True:
            try:
                clock.sleep(60)
            except KeyboardInterrupt:
                return
            except Exception:  # noqa: BLE001 - as the validator's epoch loop
                pass
    return run


def test_an_exception_in_a_miner_epoch_fails_the_run_at_once_with_it():
    boom = ValueError("a miner epoch failed")
    boundaries = []

    def on_boundary(epoch):
        boundaries.append(epoch)
        if epoch == 2:
            raise boom

    chain = types.SimpleNamespace(block=0)
    clock = Lockstep(chain, {"a": [], "b": []}, on_boundary, stop_block=10_000 * BPE)
    clock.start({"a": _loop(clock), "b": _loop(clock)})
    t0 = time.monotonic()
    with pytest.raises(ValueError) as got:
        clock.join(timeout=120)
    assert got.value is boom
    assert time.monotonic() - t0 < 30
    assert boundaries == [1, 2]                 # no boundary ran after the failure
    for t in clock.threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in clock.threads)


def test_a_failure_wakes_a_validator_waiting_for_its_turn():
    boom = RuntimeError("the miners' epoch raised")
    gate = threading.Event()

    def on_boundary(epoch):
        gate.wait(5)
        raise boom

    chain = types.SimpleNamespace(block=BPE - 1)
    clock = Lockstep(chain, {"a": [], "b": [], "c": []}, on_boundary, stop_block=100 * BPE)
    clock.start({n: _loop(clock) for n in ("a", "b", "c")})
    gate.set()
    with pytest.raises(RuntimeError) as got:
        clock.join(timeout=60)
    assert got.value is boom
    for t in clock.threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in clock.threads)


def test_a_run_without_a_failure_still_ends_at_its_stop_block():
    seen = []
    chain = types.SimpleNamespace(block=0)
    clock = Lockstep(chain, {"a": [], "b": []}, seen.append, stop_block=5 * BPE)
    clock.start({"a": _loop(clock), "b": _loop(clock)})
    clock.join(timeout=60)
    assert seen == [1, 2, 3, 4] and clock.errors == {}
