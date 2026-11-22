"""The fast-crowning simulation (scripts/experiments/experiment_fast_crown_v4.py;
docs/decisions/FAST_CROWN_V4.md): its wrapper is the real bound, its routing
items have the edge and spread they are built for, and its headline figures
hold on a short run. Single process, a few seconds."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fugal_subnet.seat_bound import lower_bound

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "experiment_fast_crown_v4.py"
_spec = importlib.util.spec_from_file_location("experiment_fast_crown_v4", _PATH)
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)


@pytest.fixture(scope="module")
def inp():
    return sim.load_inputs()


def test_the_wrapper_is_the_real_bound_on_the_same_items():
    rng = np.random.default_rng(0)
    items = {"a": rng.integers(-9000, 9000, 700).tolist(), "b": rng.integers(-5000, 7000, 300).tolist()}
    sums = {c: (len(v), sum(v), sum(x * x for x in v)) for c, v in items.items()}
    for margin in (-1.5, 0.0, 0.1):
        got = sim.bound_cp(sums, margin, 500, 100)
        want = lower_bound({c: [x * sim.CP for x in v] for c, v in items.items()},
                           margin_units=int(round(margin * 1_000_000)), n_min=500, n0=100)
        assert (got.passes, got.lower, got.n) == (want.passes, want.lower, want.n)


def test_the_guard_only_widens_the_bound():
    sums = {"a": (1000, 150_000, 400_000_000), "_e2": (1, 1000 ** 2, 0)}
    plain = sim.bound_cp(sums, -1.5, 500, 100)
    guarded = sim.bound_cp(sums, -1.5, 500, 100, tau0=0.5)
    assert guarded.lower < plain.lower


def test_the_routing_items_have_the_edge_and_spread_they_are_built_for(inp):
    model = sim.PicksModel(inp)
    rng = np.random.default_rng(1)
    for mu in (-2.0, 0.0, 1.0):
        b = model.batch(rng, 1000, mu, 200)
        x = np.concatenate([a for a in b.values()], axis=1) / 100.0
        assert x.mean() == pytest.approx(mu, abs=0.25)
        assert x.std() == pytest.approx(model.sd, rel=0.1)


def test_a_router_worse_than_the_margin_does_not_pass_and_a_much_better_one_does(inp):
    model = sim.PicksModel(inp)
    rng = np.random.default_rng(2)
    worse = model.batch(rng, 1000, -3.0, 100)
    better = model.batch(rng, 1000, 4.0, 100)
    assert not any(sim.bound_cp(model.sums(worse, r), sim.MARGIN_B, 1000, 100).passes
                   for r in range(100))
    assert sum(sim.bound_cp(model.sums(better, r), sim.MARGIN_B, 1000, 100).passes
               for r in range(100)) >= 95


def test_withholding_with_a_record_that_never_resets_beats_the_answered_run(inp):
    """The reason the run restarts on a gap: with a record that never resets,
    keeping only the epochs that help is a selection, and it passes a router
    a quarter point past the margin far more often than any restarting rule."""
    res = sim.sim_b(inp, runs=12, epochs=80, mus=(-1.75,), sizes=(500,), taus=(0.0,),
                    tau0s=(0.0,))
    keep = res[(500, -1.75, 0.0, 0.0, "keep")][1]
    run = res[(500, -1.75, 0.0, 0.0, "run")][1]
    assert run == 0
    assert keep >= run


def test_one_ticket_is_dear_beyond_the_margin(inp):
    t = sim.tickets(inp, sds=(20.0,), sizes=(1000,), mus=(-1.75, -2.0, -3.0))
    p = [t[(20.0, 1000, m)][0] for m in (-1.75, -2.0, -3.0)]
    assert p[0] > p[1] > p[2]
    assert t[(20.0, 1000, -2.0)][1] > 100_000      # dollars per expected false pass


def test_a_missing_row_costs_what_each_reading_says(inp):
    """Zero allowance (FAST_CROWN_V4.md 8): as built a missing row is
    certainly wrong (100 Brier points a label); read as odds 0 it costs
    100 y^2; a constant row at the entry's rate costs 100 (q - y)^2."""
    m = sim.g_entries(inp)["weak"]
    L = sim.entry_losses(inp, m)
    assert np.all(L["certainly_wrong"] == 100)
    assert np.all(L["constant"] <= L["certainly_wrong"])
    # y^2 <= y (a pooled outcome may be a fraction), so odds 0 costs at most 100 x the rate
    assert float((L["w"] * L["odds_zero"]).mean()) <= 100 * L["rate"] + 1e-9


def test_the_measurement_retry_removes_the_failure_refusals(inp):
    res = sim.sim_e(inp, runs=60, designs={k: sim.DESIGNS_RETRY[k] for k in ("300x1", "retry")})
    good = [m for m, r in res.items() if r["good_category"] and not r["dominated"]]
    plain = np.mean([res[m]["res"]["300x1"].get("answered", 0) for m in good])
    retried = np.mean([res[m]["res"]["retry"].get("answered", 0) for m in good])
    assert retried < plain / 10


def test_the_width_factor_matches_the_bound(inp):
    # At n = n0 the bound's half-width is the factor times the standard error.
    n = 1000
    rng = np.random.default_rng(3)
    x = rng.normal(0, 20, n)
    x = (x - x.mean()) / x.std() * 20
    cp = np.rint(x * 100).astype(np.int64)
    b = sim.bound_cp({"all": (n, int(cp.sum()), int((cp * cp).sum()))}, 0.0, 1, 100)
    half = -(b.lower / 1e6)
    assert half == pytest.approx(sim.width_factor(n, 100) * 20 / np.sqrt(n), rel=0.02)
