"""Protocol v2 benchmark pool: open licences only, pinned categories and weights, verified manifest."""
import gzip
import json

import pytest

from fugal_subnet.benchmarks import loader
from fugal_subnet.benchmarks.slicer import select_slice_v2
from fugal_subnet.benchmarks.v2_sets import LOADERS_V2, V2_LICENCES, _integer_gold
from fugal_subnet.config import BENCHMARK_CATEGORY_V2, BENCHMARK_WEIGHT_V2, CATEGORY_TARGETS_V2

MANIFEST = json.load(open(loader.manifest_path_v2(), encoding="utf-8"))


def test_every_v2_benchmark_has_a_loader_a_category_and_an_open_licence():
    assert set(LOADERS_V2) == set(BENCHMARK_CATEGORY_V2) == set(V2_LICENCES)
    assert set(BENCHMARK_CATEGORY_V2.values()) == set(CATEGORY_TARGETS_V2)
    assert abs(sum(CATEGORY_TARGETS_V2.values()) - 1.0) < 1e-9
    for b, lic in V2_LICENCES.items():
        assert lic.split(" ")[0] in {"MIT", "Apache-2.0", "ODC-BY", "CC0-1.0"}, (b, lic)


def test_no_gated_dataset_is_in_the_v2_pool():
    assert MANIFEST["gated_datasets"] == []
    assert "Idavidrein/gpqa" not in MANIFEST["dataset_revisions"]
    assert not any("hle" in d.lower() for d in MANIFEST["dataset_revisions"])


def test_no_category_is_weighted_so_the_small_sets_stay_small():
    """FABLE REVIEW, item 4 (2026-11-09): the hard category was weighted 50%
    supergpqa_hard / 50% integer olympiad, so the olympiad and IFEval sets
    (2,118 questions, memorisable) carried 27% of the seat test's net value.
    Every category now splits by pool share."""
    assert BENCHMARK_WEIGHT_V2 == {}


def test_the_manifest_pins_the_same_categories_weights_and_licences_as_the_code():
    for b, row in MANIFEST["per_benchmark"].items():
        assert row["category"] == BENCHMARK_CATEGORY_V2[b]
        assert row["weight"] == BENCHMARK_WEIGHT_V2.get(b)
        assert row["licence"] == V2_LICENCES[b]
    assert MANIFEST["category_targets"] == CATEGORY_TARGETS_V2
    assert MANIFEST["n_questions"] == sum(r["n"] for r in MANIFEST["per_benchmark"].values())
    assert MANIFEST["n_questions"] >= 12000                       # the unmemorisable floor


def test_a_v2_slice_on_the_manifest_shape_hits_the_targets():
    pool = [{"question_id": f"{b}_{i:05d}", "benchmark": b}
            for b, r in MANIFEST["per_benchmark"].items() for i in range(r["n"])]
    s = select_slice_v2(b"n" * 32, pool, 300, CATEGORY_TARGETS_V2, BENCHMARK_CATEGORY_V2, bench_weight=BENCHMARK_WEIGHT_V2)
    by_cat = {}
    for q in s:
        by_cat[BENCHMARK_CATEGORY_V2[q["benchmark"]]] = by_cat.get(BENCHMARK_CATEGORY_V2[q["benchmark"]], 0) + 1
    assert by_cat == {"knowledge": 120, "math": 75, "hard": 90, "instruction": 15}
    n = MANIFEST["per_benchmark"]
    olympiad = ("aime", "aime25", "beyondaime", "omnimath_hard")
    share = n["supergpqa_hard"]["n"] / (n["supergpqa_hard"]["n"] + sum(n[b]["n"] for b in olympiad))
    assert abs(sum(1 for q in s if q["benchmark"] == "supergpqa_hard") - 90 * share) <= 1
    assert sum(1 for q in s if q["benchmark"] in olympiad) <= 20


def test_the_memorisable_sets_carry_at_most_a_seventh_of_the_net_value():
    """The olympiad sets and IFEval (about 2,100 questions, within what a head
    can memorise, and every cell's grade is published) must not decide the
    seat: at most 15% of the net-value weight (the hard weight counted),
    measured over 200 epochs of the shipped slicer on the manifest's sizes."""
    import hashlib

    from fugal_subnet.config import CHAMPION_HARD_WEIGHT

    pool = [{"question_id": f"{b}_{i:05d}", "benchmark": b}
            for b, r in MANIFEST["per_benchmark"].items() for i in range(r["n"])]
    small = {"aime", "aime25", "beyondaime", "omnimath_hard", "ifeval"}
    total = mem = 0.0
    for e in range(200):
        for q in select_slice_v2(hashlib.sha256(b"e%d" % e).digest(), pool, 300, CATEGORY_TARGETS_V2,
                                 BENCHMARK_CATEGORY_V2, bench_weight=BENCHMARK_WEIGHT_V2):
            w = CHAMPION_HARD_WEIGHT if BENCHMARK_CATEGORY_V2[q["benchmark"]] == "hard" else 1.0
            total += w
            mem += w if q["benchmark"] in small else 0.0
    assert mem / total <= 0.15, mem / total


def test_a_v2_pool_file_that_disagrees_with_the_manifest_is_refused(tmp_path, monkeypatch):
    p = tmp_path / "pool_v2.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump([{"question_id": "x_1", "benchmark": "mmlu", "prompt": "q", "gold": "A", "grader_id": "letter_mcq"}], f)
    monkeypatch.setenv("FUGAL_BENCHMARK_POOL_V2", str(p))
    monkeypatch.delenv("FUGAL_POOL_UNPINNED", raising=False)
    with pytest.raises(RuntimeError):
        loader.load_all_v2(strict=True)


def test_load_pool_dispatches_on_the_protocol_version():
    with pytest.raises(ValueError):
        loader.load_pool(3)


@pytest.mark.parametrize("raw,want", [("336^\\circ", "336"), (" 42 ", "42"), ("1,024", "1024"), ("-7", "-7")])
def test_integer_gold_tolerates_decoration(raw, want):
    assert _integer_gold(raw) == want


@pytest.mark.parametrize("raw", ["3 or 4", "x", ""])
def test_integer_gold_refuses_anything_that_is_not_one_integer(raw):
    with pytest.raises(ValueError):
        _integer_gold(raw)
