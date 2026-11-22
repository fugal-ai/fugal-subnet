"""The v4 seat's headline figures with the real seat step
(scripts/experiments/experiment_seat_v4.py; SEAT_MINIMAL_V4.md 3.9, 5.2,
10): no head with zero edge over R is seated, and a head 0.5 Brier points
better than the holder is seated in about eight epochs at the recommended
label budget. A short run in the quick loop, the design's scale under the
`slow` marker."""
from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "experiment_seat_v4.py"
_spec = importlib.util.spec_from_file_location("experiment_seat_v4", _PATH)
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)


def test_the_mixture_has_the_mean_it_is_built_for():
    for mean in (0.0, 2.0, 2.5):
        for s in (0.15, 0.2):
            f = sim.mix_for(mean, s)
            got = f * (40 * s - 16 * s * s) + (1 - f) * (-40 * s - 16 * s * s)
            assert got == pytest.approx(mean)


def test_heads_with_no_edge_are_never_seated_short_run():
    z = sim.zero_edge(heads=6, epochs=60, labels=500, seed=3)
    assert z["seated"] == []


def test_a_head_half_a_point_better_is_seated_in_about_eight_epochs_short_run():
    got = [sim.epochs_to_seat(seed=s) for s in range(3)]
    # Rule B's 1,000 questions take five epochs whatever the odds.
    assert all(x is not None and 5 <= x <= 20 for x in got), got


@pytest.mark.slow
def test_thirty_heads_with_no_edge_over_300_epochs_none_seated():
    z = sim.zero_edge(heads=30, epochs=300, labels=500, seed=1)
    assert z["seated"] == []


@pytest.mark.slow
def test_a_head_half_a_point_better_is_seated_in_about_eight_epochs():
    got = [sim.epochs_to_seat(seed=s) for s in range(50)]
    assert None not in got and min(got) >= 5
    assert 5 <= statistics.median(got) <= 10, got
