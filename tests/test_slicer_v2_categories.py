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
    """15 slots -> 4.5/3.75/4.5/2.25: floors 4/3/4/2 and two remainder slots, never a
    per-benchmark floor."""
    for i in range(50):
        q = category_quotas(15, TARGETS, set(TARGETS), _nonce(i), "explore")
        assert sum(q.values()) == 15
        assert q["math"] == 4 and q["instruction"] == 2          # .75 always rounds up; .25 never does
        assert sorted((q["knowledge"], q["hard"])) == [4, 5]     # the .5 tie breaks on the nonce


def test_exploration_tie_rotates_with_the_nonce():
    """The knowledge/hard tie must not always go the same way, or the frame drifts."""
    winners = Counter(max(("knowledge", "hard"),
                          key=lambda c: category_quotas(15, TARGETS, set(TARGETS), _nonce(i), "explore")[c])
                      for i in range(400))
    assert 0.4 < winners["knowledge"] / 400 < 0.6


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
