"""Scenario B of the dress rehearsal judges headroom only against a measured
frontier. The predicate is small; the reason it exists is not: with no frame, a
one-epoch frontier values any unexplored model at 1.0, so the 'real router beats
always-cheapest' comparison was decided by which models the local chain's block
hash happened to explore (failed 2026-11-02 at router=-0.26 vs cheap=+0.09 on
code that had passed hours earlier)."""
from __future__ import annotations

import importlib.util
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "dress_rehearsal.py"


def _fn():
    spec = importlib.util.spec_from_file_location("dress_rehearsal_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.frontier_measured


def test_unmeasured_or_missing_entries_are_not_measured():
    f = _fn()
    assert not f({"3": {"reference_accuracy": 1.0}, "5": {"reference_accuracy": 0.1}}, (3, 5))
    assert not f({"5": {"reference_accuracy": 0.1}}, (3, 5))          # 3 missing
    assert not f({"3": {}, "5": {"reference_accuracy": 0.1}}, (3, 5))  # field missing


def test_every_price_point_below_one_is_measured():
    f = _fn()
    assert f({"3": {"reference_accuracy": 0.874}, "5": {"reference_accuracy": 0.094}}, (3, 5))
    assert f({}, ())
