"""Protocol v2 slicer: category-first allocation, and exploration that matches the scored mix.

Measured on mainnet (v1): exploration over 15 slots split five benchmarks 1:1:1:5:7, so a
benchmark that was 1.7% of the scored slice was 6.7% of exploration, and the reference frame
over-weighted it ~4x. Each test pins one property that, if it failed, would put the frame on a
different question mix from the one miners are scored on.
"""
import hashlib
from collections import Counter

import pytest

from fugal_subnet.benchmarks.slicer import (
    category_quotas,
    mix_weights,
    select_explore_set_v2,
    select_slice,
    select_slice_v2,
)

TARGETS = {"knowledge": 0.30, "math": 0.25, "hard": 0.30, "instruction": 0.15}
# A v2-shaped pool: big and small benchmarks in every category, with small ones that
# a max(1, ...) floor would over-sample.
SIZES = {
    "mmlupro": 12000, "mmlu": 3000, "simpleqa": 4000,      # knowledge
    "math": 5000, "aime_hist": 900,                        # math
    "hle": 2150, "gpqa_d": 198, "aime25": 30, "aime26": 30,  # hard
    "ifeval": 541, "ifbench": 300,                         # instruction
}
CATEGORY = {"mmlupro": "knowledge", "mmlu": "knowledge", "simpleqa": "knowledge",
            "math": "math", "aime_hist": "math",
            "hle": "hard", "gpqa_d": "hard", "aime25": "hard", "aime26": "hard",
            "ifeval": "instruction", "ifbench": "instruction"}
POOL = [{"question_id": f"{b}_{i:05d}", "benchmark": b} for b, n in SIZES.items() for i in range(n)]


def _nonce(i: int) -> bytes:
    return hashlib.sha256(f"epoch-{i}".encode()).digest()


def _by_cat(items):
    return Counter(CATEGORY[q["benchmark"]] for q in items)


def test_scored_slice_hits_the_category_targets_exactly():
    s = select_slice_v2(_nonce(1), POOL, 300, TARGETS, CATEGORY)
    assert len(s) == 300 and len({q["question_id"] for q in s}) == 300
    assert _by_cat(s) == {"knowledge": 90, "math": 75, "hard": 90, "instruction": 45}


def test_exploration_categories_follow_the_targets_with_no_floor():
    """15 slots -> 4.5/3.75/4.5/2.25: every category gets its floor or its ceiling, never a
    per-benchmark floor, and the total is exact."""
    for i in range(200):
        q = category_quotas(15, TARGETS, set(TARGETS), _nonce(i), "explore")
        assert sum(q.values()) == 15
        assert q["knowledge"] in (4, 5) and q["hard"] in (4, 5)
        assert q["math"] in (3, 4) and q["instruction"] in (2, 3)


def test_category_quotas_are_unbiased_in_expectation_at_any_targets():
    """The regression: largest remainder gave leftover slots to the largest fractions EVERY
    epoch, so at 40/25/30/5 exploration always split 6/4/4/1 instead of 6/3.75/4.5/0.75 on
    average. Systematic sampling makes each expectation exact."""
    for targets in (TARGETS, {"knowledge": 0.40, "math": 0.25, "hard": 0.30, "instruction": 0.05}):
        acc = Counter()
        n = 4000
        for i in range(n):
            acc.update(category_quotas(15, targets, set(targets), _nonce(50_000 + i), "explore"))
        for c, t in targets.items():
            assert abs(acc[c] / n - 15 * t) < 0.03, (targets, c, acc[c] / n)


def test_exploration_tie_rotates_with_the_nonce():
    """The knowledge/hard tie must not always go the same way, or the frame drifts."""
    winners = Counter(max(("knowledge", "hard"),
                          key=lambda c: category_quotas(15, TARGETS, set(TARGETS), _nonce(i), "explore")[c])
                      for i in range(400))
    assert 0.4 < winners["knowledge"] / 400 < 0.6


@pytest.mark.slow
def test_expected_exploration_share_equals_expected_scored_share_per_benchmark():
    """The property v1 breaks: over many epochs each benchmark's share of exploration
    matches its share of the scored slice, including the 30-question ones."""
    scored, explored = Counter(), Counter()
    epochs = 300
    for i in range(epochs):
        n = _nonce(i)
        s = select_slice_v2(n, POOL, 300, TARGETS, CATEGORY)
        e = select_explore_set_v2(hashlib.sha256(b"x" + n).digest(), POOL, {q["question_id"] for q in s},
                                  15, TARGETS, CATEGORY)
        assert len(e) == 15
        scored.update(q["benchmark"] for q in s)
        explored.update(q["benchmark"] for q in e)
    for b in SIZES:
        ps, pe = scored[b] / (300 * epochs), explored[b] / (15 * epochs)
        # Binomial noise on 4,500 draws: ~0.45 pt at a 10% share; allow ~3 sd.
        assert abs(ps - pe) < max(0.012, 0.3 * ps), (b, ps, pe)


@pytest.mark.slow
def test_v1_sampler_over_samples_small_benchmarks_and_v2_does_not():
    """Pin the finding itself: v1 at 15 slots gives a 30-question benchmark a full slot
    (6.7%); v2 gives it about its scored share (well under 1%)."""
    v1 = select_slice(_nonce(7), POOL, 15)
    assert Counter(q["benchmark"] for q in v1)["aime25"] >= 1
    hits = sum(Counter(q["benchmark"] for q in select_explore_set_v2(_nonce(i), POOL, set(), 15, TARGETS, CATEGORY))["aime25"]
               for i in range(300))
    assert hits / (15 * 300) < 0.01


def test_exploration_is_disjoint_from_the_scored_slice_and_deterministic():
    n = _nonce(3)
    s = select_slice_v2(n, POOL, 300, TARGETS, CATEGORY)
    ids = {q["question_id"] for q in s}
    e1 = select_explore_set_v2(n, POOL, ids, 15, TARGETS, CATEGORY)
    e2 = select_explore_set_v2(n, list(reversed(POOL)), ids, 15, TARGETS, CATEGORY)
    assert not ids & {q["question_id"] for q in e1}
    assert [q["question_id"] for q in e1] == [q["question_id"] for q in e2]   # input order is irrelevant
    assert [q["question_id"] for q in s] == [q["question_id"] for q in select_slice_v2(n, list(reversed(POOL)), 300, TARGETS, CATEGORY)]


def test_the_finding6_cap_still_holds_inside_a_category():
    """A small benchmark cannot take more than share_factor x its share of its category."""
    s = select_slice_v2(_nonce(5), POOL, 300, TARGETS, CATEGORY)
    hard = [q for q in s if CATEGORY[q["benchmark"]] == "hard"]
    csize = sum(SIZES[b] for b in SIZES if CATEGORY[b] == "hard")
    for b in ("gpqa_d", "aime25", "aime26"):
        cap = max(1, int(1.5 * 90 * SIZES[b] / csize))
        assert Counter(q["benchmark"] for q in hard)[b] <= cap


def test_a_category_that_runs_out_passes_its_slots_on():
    small = [q for q in POOL if q["benchmark"] != "ifbench" and not (q["benchmark"] == "ifeval" and q["question_id"] >= "ifeval_00010")]
    s = select_slice_v2(_nonce(9), small, 300, TARGETS, CATEGORY)
    assert len(s) == 300 and _by_cat(s)["instruction"] == 10


def test_mix_weights_correct_an_epoch_that_rounds_away_from_the_scored_mix():
    scored = [{"benchmark": "a"}] * 270 + [{"benchmark": "b"}] * 30          # b is 10% scored
    explored = [{"benchmark": "a"}] * 12 + [{"benchmark": "b"}] * 3          # b is 20% explored
    w = mix_weights(scored, explored)
    assert w["b"] == pytest.approx(0.5) and w["a"] == pytest.approx(0.9 / 0.8)
    # Weighted exploration reproduces the scored mix.
    assert (3 * w["b"]) / (3 * w["b"] + 12 * w["a"]) == pytest.approx(0.10)


def test_unmapped_benchmark_is_an_error_not_a_silent_category():
    with pytest.raises(ValueError):
        select_slice_v2(_nonce(1), POOL + [{"question_id": "x_1", "benchmark": "x"}], 300, TARGETS, CATEGORY)


# --- explicit weights inside a category (approved v2 hard category: 50/50 MCQ vs integer math) ---

HARD_WEIGHTS = {"hle": 0.50, "gpqa_d": 0.30, "aime25": 0.10, "aime26": 0.10}   # sums to 1 within "hard"


def test_a_weighted_category_follows_its_weights_not_pool_size():
    """Pool sizes would give hle ~89% of the hard slots; the weights give it 50%."""
    s = select_slice_v2(_nonce(11), POOL, 300, TARGETS, CATEGORY, bench_weight=HARD_WEIGHTS)
    hard = Counter(q["benchmark"] for q in s if CATEGORY[q["benchmark"]] == "hard")
    assert sum(hard.values()) == 90
    assert hard["hle"] == 45                  # 0.50 x 90
    assert hard["gpqa_d"] == 27               # 0.30 x 90, within the 1.5x cap and the 198-item pool
    assert hard["aime25"] + hard["aime26"] == 18


@pytest.mark.slow
def test_weighted_exploration_still_matches_the_weighted_scored_share():
    scored, explored = Counter(), Counter()
    for i in range(300):
        n = _nonce(1000 + i)
        s = select_slice_v2(n, POOL, 300, TARGETS, CATEGORY, bench_weight=HARD_WEIGHTS)
        e = select_explore_set_v2(hashlib.sha256(b"x" + n).digest(), POOL, {q["question_id"] for q in s},
                                  15, TARGETS, CATEGORY, bench_weight=HARD_WEIGHTS)
        scored.update(q["benchmark"] for q in s)
        explored.update(q["benchmark"] for q in e)
    for b in ("hle", "gpqa_d", "aime25"):
        ps, pe = scored[b] / (300 * 300), explored[b] / (15 * 300)
        assert abs(ps - pe) < max(0.012, 0.3 * ps), (b, ps, pe)


def test_a_partial_weight_map_for_a_category_is_refused():
    with pytest.raises(ValueError):
        select_slice_v2(_nonce(1), POOL, 300, TARGETS, CATEGORY, bench_weight={"hle": 0.5, "gpqa_d": 0.5})


def test_weights_in_one_category_leave_the_others_on_pool_share():
    a = select_slice_v2(_nonce(12), POOL, 300, TARGETS, CATEGORY, bench_weight=HARD_WEIGHTS)
    b = select_slice_v2(_nonce(12), POOL, 300, TARGETS, CATEGORY)
    other = lambda s: [q["question_id"] for q in s if CATEGORY[q["benchmark"]] != "hard"]  # noqa: E731
    assert other(a) == other(b)
