"""The protocol v4 question mix (docs/decisions/LIVE_MENU_V4.md, section 10).

The mix is derived once, from named published reveals, by
scripts/build_mix_v4.py, and pinned (data/pool_mix_v4.json, mirrored under
routing_data/). These tests hold that the committed observations rebuild it
byte for byte and reproduce the W0 replay's targets, that each step of the
rule does what the design says on hand-made inputs, that the scored slice,
exploration and the measurement draw follow the pinned mix under protocol
v4, and that there is one constructor of a v4 plan, so the TD and a verifier
draw the same questions. Nothing here reads the network.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
from collections import Counter
from fractions import Fraction

import numpy as np
import pytest

from fugal_subnet import config, epoch_plan
from fugal_subnet.benchmarks.slicer import select_private_explore
from fugal_subnet.tee import td_v4

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIX = ROOT / "data" / "pool_mix_v4.json"
PACKAGE_MIX = ROOT / "fugal_subnet" / "routing_data" / "pool_mix_v4.json"
OBSERVATIONS = ROOT / "data" / "observations" / "mix_v4_cells_e00026466-e00026489.json"
MANIFEST = ROOT / "fugal_subnet" / "benchmarks" / "pool_manifest_v3.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load("build_mix_v4")
csi = _load("check_safety_invariants")

# The W0 replay (2026-11-18, reveals e00026466..e00026489, DECISION.md section 3).
REPLAY_TARGETS = {"hard": 30, "code": 15, "math": 15, "knowledge": 15, "instruction": 15,
                  "easy_math": 5, "everyday": 5}
REPLAY_CELLS = {"hard": 1852, "code": 1491, "math": 867, "knowledge": 2008, "instruction": 1483,
                "easy_math": 511, "everyday": 1517}
REPLAY_ACCURACY = {"hard": 0.71940, "code": 0.90374, "math": 0.89375, "knowledge": 0.89583,
                   "instruction": 0.89811, "easy_math": 0.94429, "everyday": 0.99425}
REPLAY_EXACT_PCT = {"hard": 30.868, "code": 13.303, "math": 14.522, "knowledge": 14.270,
                    "instruction": 13.993, "easy_math": 8.044, "everyday": 5.000}


def _pinned() -> dict:
    return json.loads(MIX.read_bytes())


def _manifest_pool() -> list[dict]:
    """A pool of the v3 manifest's sets at their pinned sizes (ids only)."""
    per = json.loads(MANIFEST.read_bytes())["per_benchmark"]
    return [{"question_id": f"{b}_{i:05d}", "benchmark": b, "prompt": f"{b} {i}"}
            for b in sorted(per) for i in range(per[b]["n"])]


def _rows(cells: int, correct_rate: Fraction, weight: int = 1_000_000) -> list[list[int]]:
    """Observation rows for `cells` single-row cells at one weight."""
    k = int(correct_rate * cells)
    return [[1, cells, weight * cells, weight * k]]


# --- the pinned mix and its inputs ------------------------------------------------------


def test_the_committed_observations_rebuild_the_mix_byte_for_byte():
    text, _ = builder.build(str(OBSERVATIONS))
    assert text.encode("utf-8") == MIX.read_bytes() == PACKAGE_MIX.read_bytes()


def test_build_check_compares_both_copies_and_refuses_a_stale_one(tmp_path, capsys):
    assert builder.main(["build", "--check"]) == 0
    out = capsys.readouterr().out
    assert "data/pool_mix_v4.json rebuilds byte for byte" in out
    assert "fugal_subnet/routing_data/pool_mix_v4.json rebuilds byte for byte" in out
    stale = tmp_path / "pool_mix_v4.json"
    stale.write_bytes(MIX.read_bytes().replace(b'"hard": 30', b'"hard": 35', 1))
    with pytest.raises(SystemExit, match="is not what these inputs build"):
        builder.main(["build", "--check", "--out", str(stale)])


def test_the_mix_reproduces_the_replays_targets_from_the_same_epochs():
    doc = json.loads(OBSERVATIONS.read_bytes())
    assert [r["epoch_id"] for r in doc["reveals"]] == [f"e000{n}" for n in range(26466, 26490)]
    assert _pinned()["category_targets_pct"] == REPLAY_TARGETS
    pools, cats, _ = builder.pool_sizes()
    for size in (config.SLICE_SIZE_V4, 300):       # the targets do not depend on the slice
        mix = builder.derive(builder.read_observations(str(OBSERVATIONS)), pools, cats, size)
        assert mix["targets"] == REPLAY_TARGETS
    for c, t in mix["trace"]["categories"].items():
        assert t["cells"] == REPLAY_CELLS[c]
        assert abs(float(t["p"]) - REPLAY_ACCURACY[c]) < 5e-6, c
        assert abs(float(t["exact"]) * 100 - REPLAY_EXACT_PCT[c]) < 5e-4, c
    # The replay's own left-out counts: failed rows, the one entry not on the
    # seed menu, and omnimath ids no reveal placed.
    assert doc["left_out"]["unplaced_cells"] == 241
    assert sum(t["cells"] for t in mix["trace"]["categories"].values()) == 9970 - 241


def test_the_pinned_targets_sum_to_100_and_respect_the_floor_and_the_step():
    t = _pinned()["category_targets_pct"]
    assert sum(t.values()) == 100
    assert all(v >= config.MIX_FLOOR_PCT_V4 and v % config.MIX_STEP_PCT_V4 == 0 for v in t.values())
    assert set(t) == set(config.BENCHMARK_CATEGORY_V3.values())


def test_every_set_is_under_its_cycle_cap_and_each_category_sums_to_its_target():
    doc = _pinned()
    # SEAT_MINIMAL_V4.md section 8: 200 scored questions, the mix rebuilt for them.
    assert doc["slice_size"] == config.SLICE_SIZE_V4 == 200
    assert doc["cycle_epochs"] == config.MIX_CYCLE_EPOCHS_V4 == 96
    per_cat = Counter()
    for b, row in doc["benchmarks"].items():
        assert row["category"] == config.BENCHMARK_CATEGORY_V3[b]
        assert 0 < row["weight_ppm"]
        assert Fraction(row["weight_ppm"], 10**6) <= Fraction(row["pool"], 200 * 96), b
        per_cat[row["category"]] += row["weight_ppm"]
    assert dict(per_cat) == {c: v * 10**4 for c, v in doc["category_targets_pct"].items()}


def test_at_200_questions_no_cycle_cap_binds():
    """LIVE_MENU_V4.md 10: at a 300-question slice the cycle cap held every
    hard set but aime, and two code sets; at 200 nothing binds, and about 53
    supergpqa_hard questions are drawn an epoch (the whole set about every
    130 epochs)."""
    pools, cats, _ = builder.pool_sizes()
    observed = builder.read_observations(str(OBSERVATIONS))
    at_200 = builder.derive(observed, pools, cats, 200)["trace"]["sets"]
    assert not [b for b, t in at_200.items() if t["capped"]]
    at_300 = builder.derive(observed, pools, cats, 300)["trace"]["sets"]
    assert {"code_contests", "code_mbppplus"} <= {b for b, t in at_300.items() if t["capped"]}
    row = _pinned()["benchmarks"]["supergpqa_hard"]
    per_epoch = Fraction(200 * row["weight_ppm"], 10**6)
    assert 52 <= per_epoch <= 54
    assert 125 <= row["pool"] / per_epoch <= 135


def test_no_question_leaves_the_pool_for_being_answered_by_every_entry():
    """Every set of the v3 pool is in the mix at its whole size with a
    positive share, the saturated ones included (ARC at the floor; sets
    every observed entry answered keep their share), and the v4 index holds
    every question of the pool it is built on."""
    doc = _pinned()
    per = json.loads(MANIFEST.read_bytes())["per_benchmark"]
    assert {b: r["pool"] for b, r in doc["benchmarks"].items()} == {b: r["n"] for b, r in per.items()}
    assert doc["category_targets_pct"]["everyday"] == config.MIX_FLOOR_PCT_V4
    pools, cats, _ = builder.pool_sizes()
    trace = builder.derive(builder.read_observations(str(OBSERVATIONS)), pools, cats,
                           config.SLICE_SIZE_V4)["trace"]["sets"]
    answered_by_all = [b for b, t in trace.items() if t["p"] == 1]
    assert answered_by_all            # aime25 and code_humanevalplus: every observed cell right
    assert all(doc["benchmarks"][b]["weight_ppm"] > 0 for b in answered_by_all)
    pool = [{"question_id": f"{b}_{i}", "benchmark": b} for b in sorted(per) for i in range(7)]
    index = epoch_plan.pool_index_v4(pool)
    assert index.size == len(pool) == sum(len(v) for v in index.by_bench.values())


def test_a_build_that_would_give_a_set_no_share_stops():
    obs = {"a": _rows(400, Fraction(1)), "b": _rows(400, Fraction(1, 2)),
           "c": _rows(400, Fraction(1, 2))}
    with pytest.raises(SystemExit, match="no share"):
        builder.derive(obs, {"a": 50_000, "b": 50_000, "c": 50_000}, {"a": "x", "b": "x", "c": "y"}, 300)


# --- the rule, on hand-made inputs ---------------------------------------------------------


def test_targets_are_proportional_to_the_outcome_variance():
    obs = {"a": _rows(1000, Fraction(1, 2)),     # p(1-p) = 1/4
           "b": _rows(1000, Fraction(9, 10)),    # 9/100
           "c": _rows(1000, Fraction(4, 5))}     # 4/25
    mix = builder.derive(obs, {"a": 10**5, "b": 10**5, "c": 10**5}, {"a": "x", "b": "y", "c": "z"}, 300)
    exact = {c: t["exact"] for c, t in mix["trace"]["categories"].items()}
    total = Fraction(1, 4) + Fraction(9, 100) + Fraction(4, 25)
    assert exact == {"x": Fraction(1, 4) / total, "y": Fraction(9, 100) / total,
                     "z": Fraction(4, 25) / total}
    assert mix["targets"] == {"x": 50, "y": 20, "z": 30}       # 50.0 / 18.0 / 32.0


def test_the_floor_lifts_a_low_category_and_the_others_share_the_rest():
    values = {"a": Fraction(100), "b": Fraction(100), "c": Fraction(1)}
    t = builder.floored_targets(values, Fraction(5, 100))
    assert t == {"a": Fraction(95, 200), "b": Fraction(95, 200), "c": Fraction(1, 20)}
    # A second category pushed under the floor by the first lift is fixed too.
    values = {"a": Fraction(1000), "b": Fraction(55), "c": Fraction(1)}
    assert Fraction(55, 1056) > Fraction(5, 100)
    t = builder.floored_targets(values, Fraction(5, 100))
    assert t["c"] == t["b"] == Fraction(1, 20) and t["a"] == Fraction(9, 10)


def test_rounding_is_to_the_step_by_largest_remainder_and_ties_go_by_name():
    exact = {"a": Fraction(309, 1000), "b": Fraction(133, 1000), "c": Fraction(145, 1000),
             "d": Fraction(143, 1000), "e": Fraction(140, 1000), "f": Fraction(80, 1000),
             "g": Fraction(50, 1000)}
    assert builder.rounded_targets(exact, 5, 5) == {"a": 30, "b": 15, "c": 15, "d": 15, "e": 15,
                                                    "f": 5, "g": 5}
    near = {"a": Fraction(1, 2) - Fraction(1, 40), "b": Fraction(1, 2) + Fraction(1, 40)}
    assert builder.rounded_targets(near, 10, 5) == {"a": 50, "b": 50}
    tie = {"b": Fraction(475, 1000), "a": Fraction(475, 1000), "c": Fraction(50, 1000)}
    assert builder.rounded_targets(tie, 5, 5) == {"a": 50, "b": 45, "c": 5}


def test_inside_a_category_sets_split_by_own_variance_times_pool_and_thin_sets_take_the_categorys():
    obs = {"big": _rows(400, Fraction(1, 2)),           # own value 1/4
           "low": _rows(400, Fraction(9, 10)),          # own value 9/100
           "thin": _rows(50, Fraction(1))}              # under 100 cells: the category's value
    pools = {"big": 100_000, "low": 100_000, "thin": 100_000}        # no cycle cap binds
    mix = builder.derive(obs, pools, {b: "only" for b in obs}, 300)
    sets = mix["trace"]["sets"]
    category = mix["trace"]["categories"]["only"]["value"]
    assert sets["thin"]["value_from"] == "category" and sets["thin"]["value"] == category
    assert sets["big"]["value"] == Fraction(1, 4) and sets["low"]["value"] == Fraction(9, 100)
    total = Fraction(1, 4) + Fraction(9, 100) + category
    assert {b: s["share"] for b, s in sets.items()} == {
        "big": Fraction(1, 4) / total, "low": Fraction(9, 100) / total, "thin": category / total}
    # Pool size scales a set's share.
    mix2 = builder.derive(obs, {**pools, "low": 200_000}, {b: "only" for b in obs}, 300)
    s2 = mix2["trace"]["sets"]
    assert s2["low"]["share"] / s2["big"]["share"] == 2 * Fraction(9, 100) / Fraction(1, 4)


def test_the_cycle_cap_holds_a_small_set_and_its_category_keeps_its_share():
    """A small set with the highest variance would be drawn through in a
    few epochs; it is held at pool / (slice x 96) and the rest of its
    category's share goes to the others in proportion."""
    obs = {"small": _rows(400, Fraction(1, 2)), "a": _rows(400, Fraction(4, 5)),
           "b": _rows(400, Fraction(4, 5)), "other": _rows(400, Fraction(4, 5))}
    pools = {"small": 120, "a": 12_000, "b": 6_000, "other": 90_000}
    cats = {"small": "x", "a": "x", "b": "x", "other": "y"}
    mix = builder.derive(obs, pools, cats, 300)
    sets = mix["trace"]["sets"]
    cap = Fraction(120, 300 * 96)
    uncapped = (Fraction(mix["targets"]["x"], 100) * Fraction(1, 4) * 120
                / (Fraction(1, 4) * 120 + Fraction(4, 25) * 18_000))
    assert uncapped > cap                                   # it would be cycled too fast
    assert sets["small"]["share"] == cap and sets["small"]["capped"]
    rest = Fraction(mix["targets"]["x"], 100) - cap
    assert sets["a"]["share"] == rest * 2 / 3 and sets["b"]["share"] == rest / 3
    ppm = mix["weight_ppm"]
    assert Fraction(ppm["small"], 10**6) <= cap
    assert ppm["small"] + ppm["a"] + ppm["b"] == mix["targets"]["x"] * 10**4


def test_a_category_that_cannot_place_its_share_under_the_caps_stops_the_build():
    obs = {"s": _rows(400, Fraction(1, 2)), "t": _rows(400, Fraction(1, 2))}
    with pytest.raises(SystemExit, match="cannot place"):
        builder.derive(obs, {"s": 100, "t": 9000}, {"s": "x", "t": "y"}, 300)


def test_millionths_round_down_then_go_to_the_largest_remainders_never_over_a_cap():
    shares = {"a": Fraction(1, 3) * Fraction(15, 100), "b": Fraction(2, 3) * Fraction(15, 100)}
    assert builder.to_ppm(shares, 150_000, {"a": 10**6, "b": 10**6}, 300) == {"a": 50_000,
                                                                              "b": 100_000}
    cap = builder.cycle_cap(2500, 300)                       # 86,805.55... millionths
    shares = {"capped": cap, "free": Fraction(15, 100) - cap}
    got = builder.to_ppm(shares, 150_000, {"capped": 2500, "free": 10**6}, 300)
    assert got == {"capped": 86_805, "free": 63_195}


def test_the_observe_step_counts_published_cells_with_the_published_weights(tmp_path):
    """Two hand-made v3 reveals: failed rows, an entry off the seed menu and
    an omnimath id no reveal places are left out; a cell answered by two
    operators counts k / n; weights are the v3 rule's, checked against the
    published ones."""
    from fugal_subnet import explore_weights as xw
    from fugal_subnet import routing_protocol

    models = routing_protocol.models_for_protocol(3, routing_protocol.MENU_B)
    w = xw.weights(epoch_plan.explore_shares(3, models))
    seed = {e["id"] for e in json.loads((ROOT / "data" / "menu_v4_seed.json").read_bytes())["entries"]}
    a, b = sorted(seed)[:2]
    off = next(m for m in models if m not in seed)

    def row(q, m, correct, failed=False):
        return {"question_id": q, "model": m, "correct": correct, "failed": failed,
                "prompt_tokens": 1, "completion_tokens": 1, "coldkey": "ck"}

    for n, rows in ((1, [row("arc_easy_1", a, True), row("arc_easy_1", a, False),
                         row("mmlu_00001", b, True), row("omnimath_00001", a, True),
                         row("mmlu_00002", off, True), row("mmlu_00003", b, True, failed=True)]),
                    (2, [row("omnimath_00002", b, False), row("supergpqa_x", a, False)])):
        reveal = {"epoch_id": f"e0000000{n}", "protocol_version": 3, "models": models,
                  "questions": [], "explore_benchmarks": {},
                  "exploration_pool": [{"question_id": "mmlu_00001", "model": b, "benchmark": "mmlu",
                                        "weight": w[b]}] if n == 1 else
                  [{"question_id": "omnimath_00002", "model": b, "benchmark": "omnimath_mid",
                    "weight": w[b]}],
                  "exploration": rows}
        (tmp_path / f"reveal_e0000000{n}.json").write_text(json.dumps(reveal))
    text = builder.observe(str(tmp_path), "e00000001", "e00000002")
    out = tmp_path / "obs.json"
    out.write_text(text)
    got = builder.read_observations(str(out))
    assert got == {"arc": [[2, 1, w[a], w[a]]], "mmlu": [[1, 1, w[b], w[b]]],
                   "omnimath_mid": [[1, 1, w[b], 0]], "supergpqa_hard": [[1, 1, w[a], 0]]}
    assert json.loads(text)["left_out"] == {"failed_rows": 1, "off_menu_cells": 1,
                                            "unplaced_cells": 1}
    # A published weight that is not the rule's stops the step.
    bad = json.loads((tmp_path / "reveal_e00000001.json").read_text())
    bad["exploration_pool"][0]["weight"] += 1
    (tmp_path / "reveal_e00000001.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="published weight"):
        builder.observe(str(tmp_path), "e00000001", "e00000002")


# --- the pin -------------------------------------------------------------------------------


def test_the_safety_check_accepts_the_mix_and_names_what_moved(tmp_path, monkeypatch):
    errors: list[str] = []
    csi.check_pool_mix_v4_pinned(errors)
    csi.check_routing_data_mirrors_data(errors)
    assert errors == []
    assert hashlib.sha256(MIX.read_bytes()).hexdigest() == csi.PINNED_POOL_MIX_V4_SHA256
    assert "pool_mix_v4.json" in csi.ROUTING_DATA_REQUIRED
    (tmp_path / "data").mkdir()
    moved = MIX.read_bytes().replace(b'"hard": 30', b'"hard": 35', 1)
    (tmp_path / "data" / "pool_mix_v4.json").write_bytes(moved)
    monkeypatch.setattr(csi, "ROOT", tmp_path)
    errors = []
    csi.check_pool_mix_v4_pinned(errors)
    assert len(errors) == 1 and "does not match pin" in errors[0]


# --- the draws under protocol v4 ------------------------------------------------------------


def test_mix_v4_is_the_pinned_file():
    doc = _pinned()
    mix = epoch_plan.mix_v4()
    assert dict(mix.targets) == doc["category_targets_pct"]
    assert dict(mix.bench_weight) == {b: r["weight_ppm"] for b, r in doc["benchmarks"].items()}
    assert dict(mix.bench_category) == {b: r["category"] for b, r in doc["benchmarks"].items()}
    assert mix.slice_size == doc["slice_size"]
    assert mix.menu_mix() == tuple(sorted(doc["category_targets_pct"].items()))
    targets, cats, weights = epoch_plan._mix_v4()
    assert (targets, cats, weights) == (mix.targets, mix.bench_category, mix.bench_weight)


def test_the_v4_scored_slice_follows_the_pinned_mix():
    """Every epoch: each category's count is its target exactly, and each
    set's count is its pinned share of the slice, rounded down or up."""
    doc = _pinned()
    pool = _manifest_pool()
    for k in range(3):
        nonce = hashlib.sha256(f"v4-slice-{k}".encode()).digest()
        s = epoch_plan.scored_slice(4, nonce, pool, 200)
        assert len(s) == 200 and len({q["question_id"] for q in s}) == 200
        cats = Counter(config.BENCHMARK_CATEGORY_V3[q["benchmark"]] for q in s)
        assert dict(cats) == {c: 2 * v for c, v in doc["category_targets_pct"].items()}
        sets = Counter(q["benchmark"] for q in s)
        for b, row in doc["benchmarks"].items():
            want = Fraction(200 * row["weight_ppm"], 10**6)
            assert int(want) <= sets.get(b, 0) <= -(-want.numerator // want.denominator), b
    with pytest.raises(ValueError, match="slice"):
        epoch_plan.scored_slice(4, nonce, pool, 300)


def test_the_v4_exploration_draw_follows_the_pinned_mix():
    doc = _pinned()
    pool = _manifest_pool()
    index = epoch_plan.pool_index_v4(pool)
    for c, shares in index.shares.items():
        total = doc["category_targets_pct"][c] * 10**4
        assert shares == {b: doc["benchmarks"][b]["weight_ppm"] / total for b in shares}
    targets = epoch_plan.mix_v4().targets
    hard = Counter()
    size = config.EXPLORE_PRIVATE_SIZE_V4
    assert size == 112 and not hasattr(epoch_plan, "measurement_questions")
    for k in range(40):
        nonce = hashlib.sha256(f"v4-explore-{k}".encode()).digest()
        private = select_private_explore(nonce, index, set(), size, targets)
        _at_targets(Counter(config.BENCHMARK_CATEGORY_V3[q["benchmark"]] for q in private),
                    size, doc["category_targets_pct"])
        hard.update(q["benchmark"] for q in private
                    if config.BENCHMARK_CATEGORY_V3[q["benchmark"]] == "hard")
    # About 1,340 hard picks: supergpqa_hard's pinned share of the category.
    share = hard["supergpqa_hard"] / sum(hard.values())
    assert abs(share - doc["benchmarks"]["supergpqa_hard"]["weight_ppm"] / 300_000) < 0.04


def _at_targets(counts: Counter, size: int, targets_pct: dict) -> None:
    """`size` draws split over the categories at their per-cent targets, each
    count its exact share rounded down or up (the slicer's unbiased largest
    remainder: 112 and 150 are not whole multiples of every target)."""
    assert sum(counts.values()) == size
    for c, v in targets_pct.items():
        want = Fraction(size * v, 100)
        assert int(want) <= counts.get(c, 0) <= -(-want.numerator // want.denominator), c


def test_the_v4_exploration_plan_draws_its_shared_and_private_sets_on_the_v4_mix():
    from fugal_subnet import menu_state

    state = menu_state.seed_state(1000)
    shares = epoch_plan.explore_shares_v4(state)
    pool = _manifest_pool()
    nonce = hashlib.sha256(b"v4-plan").digest()
    scored = {q["question_id"] for q in epoch_plan.scored_slice(4, nonce, pool, 200)}
    shared, cost, cover = epoch_plan.exploration_plan_v4(nonce, pool, scored, shares, "5Fhotkey-a",
                                                         epoch_plan.pool_index_v4(pool))
    private = {**cost, **cover}
    _at_targets(Counter(config.BENCHMARK_CATEGORY_V3[q.rsplit("_", 1)[0]] for q in private),
                config.EXPLORE_PRIVATE_SIZE_V4, _pinned()["category_targets_pct"])
    assert (len(cost), len(cover)) == (config.EXPLORE_COST_AWARE_SIZE_V4,
                                       config.EXPLORE_COVERAGE_SIZE_V4)
    # No shared set under v4 (SEAT_MINIMAL_V4.md 2.3 item 33).
    assert shared == {} and config.EXPLORE_SHARED_SIZE_V4 == 0 and not set(private) & scored


def test_the_v3_draws_do_not_read_the_v4_mix(monkeypatch):
    pool = _manifest_pool()
    nonce = hashlib.sha256(b"v3-unchanged").digest()
    before = [q["question_id"] for q in epoch_plan.scored_slice(3, nonce, pool, 300)]
    index_before = epoch_plan.pool_index_v3(pool).shares

    def refuse():
        raise AssertionError("a v3 draw read the v4 mix")

    monkeypatch.setattr(epoch_plan, "mix_v4", refuse)
    monkeypatch.setattr(epoch_plan, "_mix_v4", refuse)
    assert [q["question_id"] for q in epoch_plan.scored_slice(3, nonce, pool, 300)] == before
    assert epoch_plan.pool_index_v3(pool).shares == index_before


# --- one constructor of a v4 plan ------------------------------------------------------------


def test_plan_v4_is_built_from_the_pool_and_the_pinned_mix_alone():
    pool = _manifest_pool()
    plan = epoch_plan.plan_v4(pool)
    other = epoch_plan.plan_v4(list(reversed(pool)))
    assert dict(plan.targets) == dict(other.targets) == _pinned()["category_targets_pct"]
    assert plan.pool_index.shares == other.pool_index.shares
    assert ({b: [q["question_id"] for q in qs] for b, qs in plan.pool_index.by_bench.items()}
            == {b: [q["question_id"] for q in qs] for b, qs in other.pool_index.by_bench.items()})
    nonce = hashlib.sha256(b"plan").digest()
    assert plan.scored_slice(nonce, pool, 200) == epoch_plan.scored_slice(4, nonce, pool, 200)
    with pytest.raises(ValueError, match="slice"):
        plan.scored_slice(nonce, pool, 300)
    with pytest.raises(ValueError, match="slice"):
        epoch_plan.plan_v4(pool, slice_size=300)
    excluded = {q["question_id"] for q in plan.scored_slice(nonce, pool, 200)}
    from fugal_subnet import menu_state as ms

    state = ms.seed_state(1000)
    assert (plan.exploration(nonce, pool, excluded, state, "5Fhk")
            == other.exploration(nonce, pool, excluded, state, "5Fhk"))


def test_the_td_and_the_verifier_draw_the_same_questions_in_a_join_epoch(monkeypatch):
    """The TD runs its epoch on plan_v4(pool) on a menu a join delta put a
    provisional entry on; a verifier builds its own plan_v4 from its own copy
    of the pool and draws the exploration it expects, coverage aimed first at
    the new entry. They are the same questions and entries."""
    from fugal_subnet.tee import harness
    from tests import test_harness_v4 as th
    from tests.test_tee_e2e import _head

    th.mf.allow_small_menus(monkeypatch)
    pool = th._v3_benchmarks(th._pool(1200))
    pointer, data = th._joined()
    provider = th.Provider()
    monkeypatch.setattr(harness, "_call_model", provider.call)
    inputs = td_v4.V4EpochInputs(epoch=102, previous=th.mf.state(), committed_hash=None,
                                 published=None, allowlist=th.mf.ALLOW,
                                 fetch_listing=provider.fetch, plan=epoch_plan.plan_v4(pool),
                                 join_pointer=pointer, join_published=data)
    hidden = np.random.RandomState(0).randn(len(pool), 1024).astype(np.float32)
    proof = harness.run_benchmark(
        nonce=th.NONCE, head_bytes=_head(th.FIXTURE_IDS), benchmark_pool=pool,
        proxy=th._Proxy(port=0), hidden_states=hidden, slice_size=200, epoch_id="e00000102",
        source_hash="img", hotkey=th.HOTKEY, protocol_version=4, v4=inputs)
    state = inputs.menu()
    assert state.entry("lab-c/model-new@high").provisional

    verifier = epoch_plan.plan_v4([dict(q) for q in pool])
    nonce = bytes.fromhex(th.NONCE)
    scored = verifier.scored_slice(nonce, pool, 200)
    assert proof.questions_hash == harness.compute_questions_hash([q["question_id"] for q in scored])
    explored = verifier.exploration(nonce, pool, {q["question_id"] for q in scored}, state,
                                    th.HOTKEY)
    assert {r.question_id: r.routed_model for r in proof.exploration_results} == explored
    assert "lab-c/model-new@high" in explored.values()
