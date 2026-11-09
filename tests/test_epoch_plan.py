"""epoch_plan: the single question-selection entry point for both neurons."""
import hashlib
from collections import Counter

import pytest

from fugal_subnet import epoch_plan
from fugal_subnet.benchmarks.slicer import select_slice
from fugal_subnet.config import BENCHMARK_CATEGORY_V2, CATEGORY_TARGETS_V2
from fugal_subnet.exploration import expected_exploration

V1_POOL = [{"question_id": f"{b}_{i:05d}", "benchmark": b} for b, n in
           {"mmlu": 3000, "math": 1500, "gsm8k": 500, "aime": 200, "ifeval": 100}.items() for i in range(n)]
V2_SIZES = {"mmlupro": 6000, "mmlu": 1500, "math_l35": 1800, "omnimath_mid": 550, "supergpqa_hard": 3500,
            "aime": 450, "aime25": 30, "beyondaime": 100, "omnimath_hard": 420, "ifeval": 259}
V2_POOL = [{"question_id": f"{b}_{i:05d}", "benchmark": b} for b, n in V2_SIZES.items() for i in range(n)]
MODELS = [f"m/{i:02d}" for i in range(19)]


def _n(i):
    return hashlib.sha256(f"plan-{i}".encode()).digest()


def test_v1_is_byte_identical_to_the_functions_v1_always_used():
    for i in range(5):
        s = epoch_plan.scored_slice(1, _n(i), V1_POOL, 300)
        assert s == select_slice(_n(i), V1_POOL, 300)
        ids = {q["question_id"] for q in s}
        assert epoch_plan.exploration_plan(1, _n(i), V1_POOL, ids, MODELS[:17], 15) == \
            expected_exploration(_n(i), V1_POOL, ids, MODELS[:17], 15)


def test_v2_selects_by_category_and_explores_disjointly_with_distinct_models():
    s = epoch_plan.scored_slice(2, _n(1), V2_POOL, 300)
    cats = Counter(BENCHMARK_CATEGORY_V2[q["benchmark"]] for q in s)
    assert dict(cats) == {c: round(300 * t) for c, t in CATEGORY_TARGETS_V2.items()}
    ids = {q["question_id"] for q in s}
    plan = epoch_plan.exploration_plan(2, _n(1), V2_POOL, ids, MODELS, 15)
    assert len(plan) == 15 and not ids & set(plan)
    assert len(set(plan.values())) == 15                      # 15 slots over 19 models: all distinct
    assert set(plan.values()) <= set(MODELS)


def test_v2_plan_is_independent_of_pool_order_and_model_list_order_is_the_callers():
    s1 = epoch_plan.scored_slice(2, _n(2), V2_POOL, 300)
    s2 = epoch_plan.scored_slice(2, _n(2), list(reversed(V2_POOL)), 300)
    assert [q["question_id"] for q in s1] == [q["question_id"] for q in s2]
    ids = {q["question_id"] for q in s1}
    p1 = epoch_plan.exploration_plan(2, _n(2), V2_POOL, ids, MODELS, 15)
    p2 = epoch_plan.exploration_plan(2, _n(2), list(reversed(V2_POOL)), ids, MODELS, 15)
    assert p1 == p2


def test_an_unknown_version_is_refused():
    with pytest.raises(ValueError):
        epoch_plan.scored_slice(3, _n(0), V2_POOL, 300)
    with pytest.raises(ValueError):
        epoch_plan.exploration_plan(3, _n(0), V2_POOL, set(), MODELS, 15)


def test_v2_hard_category_follows_its_benchmark_weights_not_pool_size():
    """Within `hard`, BENCHMARK_WEIGHT_V2 sets the mix (supergpqa_hard 50%).
    Dropping the weights falls back to pool size (supergpqa_hard ~78% here),
    a different consensus question mix that no other test noticed (planted
    bug, 2026-11-08)."""
    from fugal_subnet.config import BENCHMARK_WEIGHT_V2

    hard = Counter()
    for i in range(20):
        hard.update(q["benchmark"] for q in epoch_plan.scored_slice(2, _n(100 + i), V2_POOL, 300)
                    if BENCHMARK_CATEGORY_V2[q["benchmark"]] == "hard")
    total = sum(hard.values())
    for bench, w in BENCHMARK_WEIGHT_V2.items():
        assert abs(hard[bench] / total - w) < 0.02, (bench, hard[bench] / total, w)


# Pinned outputs for one nonce. Both neurons must select exactly these; any
# change to selection, apportionment, the exploration nonce or model
# assignment is a consensus change and must update these on purpose.
GOLDEN_V2_SLICE_SHA256 = "4fc11fbddbcb506abff9395018f18bb72d9de4886961a76814c1496f80a02632"
GOLDEN_V2_PLAN_SHA256 = "c6cdaf83463d6ed770818e75aef80a0fa68da42b536b95fa4082c47cc0ea2720"


def test_v2_selection_is_pinned():
    import json

    ids = [q["question_id"] for q in epoch_plan.scored_slice(2, _n(7), V2_POOL, 300)]
    plan = epoch_plan.exploration_plan(2, _n(7), V2_POOL, set(ids), MODELS, 15)
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == GOLDEN_V2_SLICE_SHA256
    assert hashlib.sha256(json.dumps(sorted(plan.items())).encode()).hexdigest() == GOLDEN_V2_PLAN_SHA256
