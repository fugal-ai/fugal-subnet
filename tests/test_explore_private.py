"""Protocol v3 exploration: a shared set every miner answers, and a private
set drawn for each miner from the epoch nonce and its hotkey.

docs/INVARIANTS.md, "I4 — every miner measures the models (protocol v3)".
"""
import hashlib
from collections import Counter

import pytest

from fugal_subnet import config, epoch_plan
from fugal_subnet.benchmarks.slicer import PoolIndex, select_private_explore
from fugal_subnet.exploration import private_nonce

# Each category sized in proportion to its target (about 4,000 questions in all), split
# evenly across its benchmarks, so every question has about the same chance per draw and
# coverage measures the sampler, not the mix.
_PER_CAT = {c: sum(1 for v in config.BENCHMARK_CATEGORY_V3.values() if v == c)
            for c in config.CATEGORY_TARGETS_V3}
BENCHES = {b: max(20, round(4000 * config.CATEGORY_TARGETS_V3[c] / _PER_CAT[c]))
           for b, c in config.BENCHMARK_CATEGORY_V3.items()}
POOL = [{"question_id": f"{b}-{i:05d}", "benchmark": b, "prompt": f"q {b} {i}"}
        for b in sorted(BENCHES) for i in range(BENCHES[b])]
POOL_V2 = [{"question_id": f"{b}-{i:05d}", "benchmark": b, "prompt": f"q {b} {i}"}
           for b in sorted(config.BENCHMARK_CATEGORY_V2) for i in range(400)]
MODELS = [f"vendor/model-{i:02d}" for i in range(19)]
NONCE = hashlib.sha256(b"epoch-1").digest()
SCORED = {q["question_id"] for q in epoch_plan.scored_slice(3, NONCE, POOL, 300)}
INDEX = epoch_plan.pool_index_v3(POOL)


def plan(hotkey, nonce=NONCE, index=INDEX):
    return epoch_plan.exploration_plan_v3(nonce, POOL, SCORED, MODELS, hotkey, index)


def test_every_miner_gets_the_shared_set_and_its_own_private_set():
    shared_a, private_a = plan("hotkey-a")
    shared_b, private_b = plan("hotkey-b")
    assert shared_a == shared_b and len(shared_a) == config.EXPLORE_SHARED_SIZE == 8
    assert len(private_a) == len(private_b) == config.EXPLORE_PRIVATE_SIZE == 24
    assert private_a != private_b
    # Two miners' private sets barely overlap: that is what makes the pool wide.
    assert len(set(private_a) & set(private_b)) <= 2
    # The one entry point returns both halves, and is what the proof is checked against.
    merged = epoch_plan.exploration_plan(3, NONCE, POOL, SCORED, MODELS, 0, hotkey="hotkey-a")
    assert merged == {**shared_a, **private_a} and len(merged) == 32


def test_the_private_set_is_disjoint_from_the_scored_slice_and_the_shared_set():
    for hk in ("a", "b", "c", "d"):
        shared, private = plan(hk)
        assert not set(private) & SCORED and not set(private) & set(shared)
        assert not set(shared) & SCORED
        assert len(set(private)) == 24


def test_it_is_a_function_of_the_nonce_and_the_hotkey_only():
    assert plan("hk") == plan("hk")
    assert plan("hk") == plan("hk", index=None)            # the index is only a speed-up
    other = hashlib.sha256(b"epoch-2").digest()
    scored2 = {q["question_id"] for q in epoch_plan.scored_slice(3, other, POOL, 300)}
    assert epoch_plan.exploration_plan_v3(other, POOL, scored2, MODELS, "hk", INDEX)[1] != plan("hk")[1]
    # Pool order does not matter.
    shuffled = sorted(POOL, key=lambda q: hashlib.sha256(q["question_id"].encode()).digest())
    assert epoch_plan.exploration_plan_v3(NONCE, shuffled, SCORED, MODELS, "hk") == plan("hk")


def test_without_a_hotkey_only_the_shared_half_exists():
    shared, private = plan(None)
    assert len(shared) == 8 and private == {}
    with pytest.raises(ValueError):
        private_nonce(NONCE, "")


def test_each_miners_models_are_spread_round_robin_over_its_own_permutation():
    _, private = plan("hotkey-a")
    counts = Counter(private.values())
    # 24 slots over 19 models: every model once, five of them twice.
    assert set(counts) == set(MODELS) and sorted(counts.values()) == [1] * 14 + [2] * 5
    _, other = plan("hotkey-b")
    assert [m for m, n in sorted(counts.items()) if n == 2] != \
        [m for m, n in sorted(Counter(other.values()).items()) if n == 2]


def test_private_draws_follow_the_category_targets_and_cover_the_pool():
    """Over many miners the private questions land on categories at the
    published targets, and on models evenly: the pool measures every model on
    the mix the scored slice has."""
    cats, models, questions = Counter(), Counter(), set()
    n = 300
    for i in range(n):
        _, private = plan(f"hk-{i}")
        for qid, model in private.items():
            cats[config.BENCHMARK_CATEGORY_V3[qid.rsplit("-", 1)[0]]] += 1
            models[model] += 1
            questions.add(qid)
    total = sum(cats.values())
    for c, target in config.CATEGORY_TARGETS_V3.items():
        assert abs(cats[c] / total - target) < 0.01, (c, cats[c] / total)
    assert max(models.values()) - min(models.values()) <= 0.15 * total / len(MODELS)
    assert len(questions) > 0.6 * (len(POOL) - len(SCORED))


def test_a_nearly_exhausted_benchmark_still_fills_its_slots():
    small = [q for q in POOL if q["benchmark"] in ("ifeval", "mmlu")]
    index = PoolIndex(small, config.BENCHMARK_CATEGORY_V3)
    excluded = {q["question_id"] for q in small if q["benchmark"] == "ifeval"}
    excluded -= {"ifeval-00007", "ifeval-00031"}
    out = select_private_explore(private_nonce(NONCE, "hk"), index, excluded, 10,
                                 {"knowledge": 0.5, "instruction": 0.5})
    ids = [q["question_id"] for q in out]
    assert len(ids) == len(set(ids)) and not set(ids) & excluded
    assert {"ifeval-00007", "ifeval-00031"} <= set(ids)
    # Instruction ran out after two; its other slots are not filled from elsewhere.
    assert sum(1 for i in ids if i.startswith("ifeval")) == 2


def test_v2_plans_do_not_read_the_hotkey():
    scored = {q["question_id"] for q in epoch_plan.scored_slice(2, NONCE, POOL_V2, 300)}
    a = epoch_plan.exploration_plan(2, NONCE, POOL_V2, scored, MODELS, 15)
    b = epoch_plan.exploration_plan(2, NONCE, POOL_V2, scored, MODELS, 15, hotkey="anyone")
    assert a == b and len(a) == 15


def test_the_sizes_are_consensus_constants():
    assert {"FUGAL_EXPLORE_SHARED_SIZE", "FUGAL_EXPLORE_PRIVATE_SIZE"} <= set(config.CONSENSUS_ENV_VARS)
    assert config.EXPLORE_SHARED_SIZE == config.EXPLORE_DISAGREE_MIN_SAMPLES
