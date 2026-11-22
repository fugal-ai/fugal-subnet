"""Protocol v4 exploration: a cost-aware part and a coverage part, each pair
weighted by the probability it was drawn with.

docs/decisions/LIVE_MENU_V4.md, section 8, at the sizes of
docs/decisions/SEAT_MINIMAL_V4.md, sections 5.3 and 8. There is no shared
set under v4. A miner answers a private set of 112: 68 drawn by v3's
cost-aware shares on the menu state's measured costs, within $0.55 of
expected spend, and 44 drawn toward entries with fewer than 100 exploration
pairs. Only the fraction min(1, deficits / 1,400) of the coverage part
follows the deficits; the part is then mixed toward uniform until its
expected spend is within $1.10. Every probability is a function of published
state, so a pair's weight is the exact inverse of the probability it was
drawn with, and the odds score stays unbiased.

The worked cases are the W0 replay's (24 published v3 epochs; its measured
cost per call and pooled pairs per entry are in
tests/fixtures/w0_replay_explore_inputs.json).
"""
import hashlib
import json
import random
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fugal_subnet import config, epoch_plan, routing_protocol
from fugal_subnet import explore_weights as xw

FIXTURES = Path(__file__).parent / "fixtures"
W0 = json.loads((FIXTURES / "w0_replay_explore_inputs.json").read_text())
W0_COSTS = {e: Fraction(v, 10**9) for e, v in W0["cost_nanousd"].items()}
W0_PAIRS = dict(W0["explore_pairs"])
DEAREST = max(W0_COSTS, key=W0_COSTS.get)          # anthropic/claude-opus-5.5@max, $0.129 a call
M = len(W0_COSTS)
FULL = dict.fromkeys(W0_COSTS, 150)                # every entry above the target

COV_CAP = Fraction(config.EXPLORE_COVERAGE_SPEND_CAP_USD_V4)
N_SHARED, N_COST, N_COV = (config.EXPLORE_SHARED_SIZE_V4, config.EXPLORE_COST_AWARE_SIZE_V4,
                           config.EXPLORE_COVERAGE_SIZE_V4)
DAMP, TARGET = config.EXPLORE_COVERAGE_DAMP_PAIRS_V4, config.EXPLORE_COVERAGE_TARGET_V4


def _probs(shares):
    total = sum(shares.values())
    return {e: Fraction(s, total) for e, s in shares.items()}


def _spend(shares, costs, n):
    return n * sum(p * costs[e] for e, p in _probs(shares).items())


def test_the_constants_are_the_designs():
    # SEAT_MINIMAL_V4.md section 8: 112 calls a hotkey-epoch, no shared set.
    assert (N_SHARED, N_COST, N_COV) == (0, 68, 44)
    assert config.EXPLORE_PRIVATE_SIZE_V4 == 112
    assert config.EXPLORE_COLDKEY_CAP_V4 == 168 == 3 * config.EXPLORE_PRIVATE_SIZE_V4 // 2
    assert Fraction(config.EXPLORE_SPEND_CAP_USD_V4) == Fraction(55, 100)
    assert COV_CAP == Fraction(110, 100)
    assert TARGET == 100
    assert DAMP == 1400
    # v3's constants are untouched.
    assert (config.EXPLORE_SHARED_SIZE, config.EXPLORE_PRIVATE_SIZE,
            config.EXPLORE_COLDKEY_CAP) == (8, 24, 36)
    assert config.EXPLORE_SPEND_CAP_USD == "0.20"


# --- every part's probabilities ------------------------------------------------


@pytest.mark.parametrize("pairs", [{}, W0_PAIRS, FULL, {**FULL, DEAREST: 0}])
def test_every_part_draws_with_probabilities_that_sum_to_one(pairs):
    plan = xw.plan_shares_v4(W0_COSTS, pairs)
    parts = plan.part_probabilities()
    assert set(parts) == {"shared", "cost_aware", "coverage", "private"}
    for name, p in parts.items():
        assert set(p) == set(W0_COSTS), name
        assert all(0 <= v <= 1 for v in p.values()), name
        assert sum(p.values()) == 1, name
    # The private set is its two parts in proportion to their calls.
    for e in W0_COSTS:
        assert parts["private"][e] == (N_COST * parts["cost_aware"][e]
                                       + N_COV * parts["coverage"][e]) / (N_COST + N_COV)
    # The shared set and the cost-aware part are drawn by the same shares.
    assert parts["shared"] == parts["cost_aware"]


def test_the_cost_aware_part_is_v3s_draw_on_the_measured_costs():
    plan = xw.plan_shares_v4(W0_COSTS, W0_PAIRS)
    floor = Fraction(config.EXPLORE_COST_FLOOR)
    h = xw.choose_exponent(W0_COSTS, n_calls=N_SHARED + N_COST,
                           cap_usd=Fraction(config.EXPLORE_SPEND_CAP_USD_V4), floor=floor,
                           max_halves=config.EXPLORE_COST_MAX_HALVES)
    assert plan.exponent_halves == h == 1               # W0: exponent 1/2 at measured costs
    assert plan.cost == xw.shares(W0_COSTS, exponent_halves=h, floor=floor)
    # W0: $0.163 for 32 calls drawn by these shares, so $0.346 for the 68
    # of the cost-aware part, under the $0.55 cap (design: "the measured
    # $0.35"). Uniform would cost $0.67 and does not fit.
    assert abs(float(_spend(plan.cost, W0_COSTS, N_COST)) - 0.1629 * 68 / 32) < 0.002
    assert float(N_COST * sum(W0_COSTS.values()) / M) > 0.55


# --- weights -------------------------------------------------------------------


def _random_state(rnd: random.Random):
    m = rnd.randint(2, 64)
    costs = {f"lab/e{i:02d}": Fraction(rnd.randint(1, 200_000), 10**6) for i in range(m)}
    if rnd.random() < 0.3:                                       # one very dear entry
        costs["lab/e00"] = Fraction(rnd.randint(1, 3000), 1000)
    pairs = {e: rnd.choice([0, 0, rnd.randint(0, 99), rnd.randint(100, 400)]) for e in costs}
    return costs, pairs


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(min_value=0, max_value=2**32), st.data())
def test_weights_are_exact_inverse_probabilities(seed, data):
    """For any indicator g over entries, the expectation of w * g under a
    part's draw is the uniform mean of g: E[w g] = sum_e pi_e w_e g_e =
    (1/M) sum_e g_e. Exactly with the rational weights, and within the
    rounding of WEIGHT_SCALE units with the integer weights pairs carry."""
    costs, pairs = _random_state(random.Random(seed))
    plan = xw.plan_shares_v4(costs, pairs)
    parts = plan.part_probabilities()
    m = len(costs)
    g = {e: data.draw(st.booleans()) for e in sorted(costs)}
    uniform_mean = Fraction(sum(g.values()), m)
    for part, weights in (("shared", plan.shared_weights()), ("private", plan.private_weights())):
        p = parts[part]
        exact = xw.inverse_probability(p)
        assert sum(p[e] * exact[e] * g[e] for e in p) == uniform_mean
        got = sum(p[e] * weights[e] * g[e] for e in p) / xw.WEIGHT_SCALE
        assert abs(got - uniform_mean) <= Fraction(1, 2 * xw.WEIGHT_SCALE)
        for e in p:
            assert abs(weights[e] - exact[e] * xw.WEIGHT_SCALE) <= Fraction(1, 2)
            assert plan.pair_weight(e, private=(part == "private")) == weights[e]
    # The epoch as a whole: every call of the shared set and of the private set.
    n = N_SHARED + N_COST + N_COV
    total = (N_SHARED * sum(parts["shared"][e] * xw.inverse_probability(parts["shared"])[e] * g[e]
                            for e in g)
             + (N_COST + N_COV) * sum(parts["private"][e]
                                      * xw.inverse_probability(parts["private"])[e] * g[e]
                                      for e in g))
    assert total / n == uniform_mean


def test_the_shared_weight_is_v3s_and_the_private_weight_is_the_mixtures():
    plan = xw.plan_shares_v4(W0_COSTS, W0_PAIRS)
    assert plan.shared_weights() == xw.weights(plan.cost)
    p = plan.part_probabilities()["private"]
    for e, w in plan.private_weights().items():
        assert w == xw._round_half_up(Fraction(1, M) / p[e] * xw.WEIGHT_SCALE)


def test_the_largest_pair_weight_is_bounded():
    """A private pair's probability is at least 68/112 of the cost-aware one,
    which is at least the floor's quarter of a uniform share, so no weight
    passes 112 / (68 * 1/4) = 6.6. At the replay's costs it is 4.8, for an
    entry with no deficit while others have one (section 8)."""
    bound = Fraction(N_COST + N_COV, N_COST) / Fraction(config.EXPLORE_COST_FLOOR)
    rnd = random.Random(11)
    for _ in range(60):
        costs, pairs = _random_state(rnd)
        plan = xw.plan_shares_v4(costs, pairs)
        assert max(plan.private_weights().values()) <= bound * xw.WEIGHT_SCALE + 1
        assert max(plan.shared_weights().values()) <= xw.WEIGHT_SCALE / Fraction(
            config.EXPLORE_COST_FLOOR) + 1
    # The replay's worst case: the dearest entry at target while every other
    # is short, so the coverage part never draws it.
    worst = xw.plan_shares_v4(W0_COSTS, {**dict.fromkeys(W0_COSTS, 0), DEAREST: 100})
    assert worst.part_probabilities()["coverage"][DEAREST] == 0
    top = max(worst.private_weights().values()) / xw.WEIGHT_SCALE
    assert 4.7 < top < 4.9
    assert max(worst.shared_weights().values()) / xw.WEIGHT_SCALE == pytest.approx(2.9, abs=0.05)


# --- the coverage part: damping, the spend cap, steady state --------------------


def test_one_entry_with_a_deficit_does_not_take_the_whole_coverage_part():
    """W0's worked case: the dearest entry joins (no pairs) while every other
    entry is at target. Undamped, it takes all 44 coverage calls of every
    miner: $5.68 of coverage, $6.02 an epoch with the cost-aware calls.
    Damped, it takes the fraction 100 / 1,400 = 1/14 of the part plus its
    uniform share of the rest: about 3.8 of the 44 calls, $0.81 of coverage,
    under the $1.10 cap, so at 112 calls the cap does not bind on this spike
    (it did at 40 calls and $0.40): about $1.15 an epoch at worst."""
    pairs = {**FULL, DEAREST: 0}
    undamped = xw.coverage_draw(W0_COSTS, pairs, n_calls=N_COV, cap_usd=None, target=TARGET,
                                damp_pairs=1)
    assert undamped.probabilities[DEAREST] == 1
    cost_aware = _spend(xw.plan_shares_v4(W0_COSTS, pairs).cost, W0_COSTS, N_SHARED + N_COST)
    assert float(N_COV * W0_COSTS[DEAREST] + cost_aware) == pytest.approx(6.02, abs=0.02)

    damped = xw.coverage_draw(W0_COSTS, pairs, n_calls=N_COV, cap_usd=None, target=TARGET,
                              damp_pairs=DAMP)
    assert damped.deficits[DEAREST] == 100 and sum(damped.deficits.values()) == 100
    assert damped.damping == Fraction(1, 14)
    assert damped.probabilities[DEAREST] == Fraction(1, 14) + Fraction(13, 14) / M
    assert all(damped.probabilities[e] == Fraction(13, 14) / M for e in W0_COSTS if e != DEAREST)

    plan = xw.plan_shares_v4(W0_COSTS, pairs)
    cov = plan.part_probabilities()["coverage"]
    assert plan.toward_deficits == 1                     # the damped draw fits the cap
    assert cov == damped.probabilities
    spend = N_COV * sum(cov[e] * W0_COSTS[e] for e in cov)
    assert float(spend) == pytest.approx(0.81, abs=0.01) and spend < COV_CAP
    assert 3 < N_COV * cov[DEAREST] < 4                  # a few calls, not forty-four
    worst_epoch = _spend(plan.cost, W0_COSTS, N_SHARED + N_COST) + spend
    assert float(worst_epoch) == pytest.approx(1.15, abs=0.01)


def test_the_damping_follows_the_sum_of_deficits():
    costs = dict.fromkeys((f"lab/e{i}" for i in range(10)), Fraction(1, 1000))
    for short, f in ((1, Fraction(100, 500)), (4, Fraction(400, 500)), (5, 1), (10, 1)):
        pairs = {e: (0 if i < short else 100) for i, e in enumerate(sorted(costs))}
        d = xw.coverage_draw(costs, pairs, n_calls=16, cap_usd=COV_CAP, target=100, damp_pairs=500)
        assert d.damping == f
        assert sum(d.probabilities.values()) == 1
        at_target = [d.probabilities[e] for e in sorted(costs)[short:]]
        assert all(p == (1 - f) / 10 for p in at_target)


@settings(max_examples=150, deadline=None)
@given(st.integers(min_value=0, max_value=2**32))
def test_expected_coverage_spend_never_exceeds_the_cap(seed):
    """The cap holds whenever any mixture can meet it, with t the largest
    such fraction: at the cap exactly when 0 < t < 1, and t = 1 when the
    deficit draw already fits."""
    costs, pairs = _random_state(random.Random(seed))
    plan = xw.plan_shares_v4(costs, pairs)
    cov = plan.part_probabilities()["coverage"]
    spend = N_COV * sum(cov[e] * costs[e] for e in cov)
    uniform = N_COV * sum(costs.values()) / len(costs)
    q = xw.coverage_draw(costs, pairs, n_calls=N_COV, cap_usd=None,
                         target=TARGET, damp_pairs=DAMP).probabilities  # damped, not yet capped
    q_spend = N_COV * sum(q[e] * costs[e] for e in costs)
    if min(uniform, q_spend) <= COV_CAP:
        assert spend <= COV_CAP
        if 0 < plan.toward_deficits < 1:
            assert spend == COV_CAP
        if q_spend <= COV_CAP:
            assert plan.toward_deficits == 1
    else:
        # No mixture fits: the cheaper of the two ends, never dearer.
        assert spend == min(uniform, q_spend)


def test_the_spend_cap_holds_with_one_very_dear_entry():
    """A dear entry short of pairs, on the replay's menu. The replay's
    dearest ($0.129 a call) is drawn on the damped deficits in full: their
    $0.81 is under the cap. One six times dearer ($0.75) would cost $3.19
    damped while uniform coverage is still under the cap, so the mix brings
    the part to the cap exactly."""
    plan = xw.plan_shares_v4(W0_COSTS, {**FULL, DEAREST: 0})
    cov = plan.part_probabilities()["coverage"]
    assert plan.toward_deficits == 1
    assert N_COV * sum(cov[e] * W0_COSTS[e] for e in cov) < COV_CAP
    costs = {**W0_COSTS, DEAREST: Fraction(3, 4)}
    plan = xw.plan_shares_v4(costs, {**FULL, DEAREST: 0})
    cov = plan.part_probabilities()["coverage"]
    assert 0 < plan.toward_deficits < 1
    assert N_COV * sum(cov[e] * costs[e] for e in cov) == COV_CAP
    assert cov[DEAREST] > Fraction(1, M)                 # still measured more than uniform


def test_a_menu_whose_uniform_coverage_exceeds_the_cap_draws_its_cheapest_mixture():
    """When even uniform coverage costs more than the cap, no mixture meets
    it. The part then takes whichever end is cheaper: uniform when the short
    entries are dear, the deficit draw when they are cheap."""
    costs = {f"lab/e{i}": Fraction(1, 10) for i in range(4)}          # 44 x $0.08 = $3.52 uniform
    costs["lab/cheap"] = Fraction(1, 1000)
    dear_short = xw.plan_shares_v4(costs, {**dict.fromkeys(costs, 100), "lab/e0": 0})
    assert dear_short.toward_deficits == 0
    assert len(set(dear_short.coverage.values())) == 1
    cheap_short = xw.plan_shares_v4(costs, {**dict.fromkeys(costs, 100), "lab/cheap": 0})
    assert cheap_short.toward_deficits == 1


def test_steady_state_is_uniform_coverage():
    plan = xw.plan_shares_v4(W0_COSTS, FULL)
    assert sum(plan.deficits.values()) == 0
    assert plan.damping == 0
    cov = plan.part_probabilities()["coverage"]
    assert set(cov.values()) == {Fraction(1, M)}
    # W0: 16 uniform coverage calls cost $0.155 (0.131-0.189) at measured
    # costs; 44 cost $0.435 (the design's "measured $0.43").
    assert float(N_COV * sum(cov[e] * W0_COSTS[e] for e in cov)) == pytest.approx(0.435, abs=0.005)
    # An entry above target is not short: exactly at target counts as covered.
    at_target = xw.plan_shares_v4(W0_COSTS, dict.fromkeys(W0_COSTS, 100))
    assert at_target.coverage == plan.coverage


def test_pair_counts_are_read_for_the_menu_only():
    """n_e counts an entry's exploration pairs; an id the menu does not hold
    is ignored, a menu entry with no count has none (a new entry), and a count
    that is not a non-negative integer is refused."""
    a = xw.plan_shares_v4(W0_COSTS, {**FULL, DEAREST: 0, "retired/x@max": 5})
    b = xw.plan_shares_v4(W0_COSTS, {e: n for e, n in FULL.items() if e != DEAREST})
    assert a == b
    for bad in (-1, 1.5, True, "3"):
        with pytest.raises(ValueError):
            xw.plan_shares_v4(W0_COSTS, {**FULL, DEAREST: bad})


def test_the_replays_window_today():
    """From the published window of the replay (43 of 59 entries short of 100
    pairs), the coverage part follows the deficits fully (they sum to more
    than 1,400 pairs) and the cap decides how far."""
    plan = xw.plan_shares_v4(W0_COSTS, W0_PAIRS)
    assert sum(plan.deficits.values()) == 2176 and plan.damping == 1
    cov = plan.part_probabilities()["coverage"]
    assert N_COV * sum(cov[e] * W0_COSTS[e] for e in cov) <= COV_CAP
    short = [e for e in W0_COSTS if W0_PAIRS[e] < 100]
    covered = [e for e in W0_COSTS if W0_PAIRS[e] >= 100]
    assert min(cov[e] for e in short) > max(cov[e] for e in covered)


def test_inputs_in_any_order_give_the_same_plan():
    rev_costs = dict(reversed(list(W0_COSTS.items())))
    rev_pairs = dict(reversed(list(W0_PAIRS.items())))
    assert xw.plan_shares_v4(rev_costs, rev_pairs) == xw.plan_shares_v4(W0_COSTS, W0_PAIRS)


def test_exact_shares_are_the_probabilities_exactly():
    probs = {"a": Fraction(1, 3), "b": Fraction(1, 6), "c": Fraction(1, 2), "d": Fraction(0)}
    s = xw.exact_shares(probs)
    assert s == {"a": 2, "b": 1, "c": 3, "d": 0}
    assert _probs(s) == probs
    with pytest.raises(ValueError):
        xw.exact_shares({"a": Fraction(1, 2), "b": Fraction(1, 3)})


# --- the draw --------------------------------------------------------------------

_PER_CAT = {c: sum(1 for v in config.BENCHMARK_CATEGORY_V3.values() if v == c)
            for c in config.CATEGORY_TARGETS_V3}
POOL = [{"question_id": f"{b}-{i:05d}", "benchmark": b}
        for b, c in sorted(config.BENCHMARK_CATEGORY_V3.items())
        for i in range(max(20, round(3000 * config.CATEGORY_TARGETS_V3[c] / _PER_CAT[c])))]
INDEX = epoch_plan.pool_index_v4(POOL)
NONCE = hashlib.sha256(b"v4-epoch-1").digest()
SCORED = {q["question_id"] for q in epoch_plan.scored_slice(3, NONCE, POOL, 300)}


def test_a_miner_answers_the_shared_set_and_both_private_parts():
    plan = xw.plan_shares_v4(W0_COSTS, W0_PAIRS)
    a = epoch_plan.exploration_plan_v4(NONCE, POOL, SCORED, plan, "hotkey-a", INDEX)
    b = epoch_plan.exploration_plan_v4(NONCE, POOL, SCORED, plan, "hotkey-b", INDEX)
    shared, cost_part, coverage = a
    assert shared == b[0] and len(shared) == N_SHARED
    assert len(cost_part) == N_COST and len(coverage) == N_COV
    assert (cost_part, coverage) != (b[1], b[2])
    ids = [set(shared), set(cost_part), set(coverage)]
    assert sum(map(len, ids)) == len(set().union(*ids))        # disjoint
    assert not set().union(*ids) & SCORED
    assert set(shared.values()) | set(cost_part.values()) | set(coverage.values()) <= set(W0_COSTS)
    # Deterministic, whatever order the pool comes in, with or without an index.
    assert epoch_plan.exploration_plan_v4(NONCE, list(reversed(POOL)), SCORED, plan, "hotkey-a") == a
    # The one entry point: the merged plan, and v4 needs its shares.
    merged = epoch_plan.exploration_plan(4, NONCE, POOL, SCORED, sorted(W0_COSTS), 0,
                                         hotkey="hotkey-a", index=INDEX, shares=plan)
    assert merged == {**shared, **cost_part, **coverage}
    with pytest.raises(ValueError):
        epoch_plan.exploration_plan(4, NONCE, POOL, SCORED, sorted(W0_COSTS), 0, hotkey="hotkey-a")
    with pytest.raises(ValueError):
        epoch_plan.exploration_plan(4, NONCE, POOL, SCORED, sorted(W0_COSTS)[1:], 0,
                                    hotkey="hotkey-a", shares=plan)
    # No hotkey: the shared set alone.
    assert epoch_plan.exploration_plan_v4(NONCE, POOL, SCORED, plan, None, INDEX) == (shared, {}, {})


def test_each_part_draws_its_entries_at_its_probabilities():
    """Over many epochs, the entries of each part come up at that part's
    probabilities, and the weighted share of an indicator over the private
    pairs is the uniform mean (the estimator is unbiased in the draw, not
    only on paper)."""
    costs = {f"lab/e{i}": Fraction(1 + 7 * i, 10_000) for i in range(6)}
    costs["lab/dear"] = Fraction(1, 5)
    pairs = {e: (0 if e in ("lab/dear", "lab/e1") else 300) for e in costs}
    plan = xw.plan_shares_v4(costs, pairs)
    parts = plan.part_probabilities()
    weights = plan.private_weights()
    g = {e: e in ("lab/dear", "lab/e3") for e in costs}
    counts = {"cost_aware": Counter(), "coverage": Counter()}
    weighted, n = Fraction(0), 0
    for k in range(400):
        nonce = hashlib.sha256(f"v4-draw-{k}".encode()).digest()
        _, cost_part, coverage = epoch_plan.exploration_plan_v4(
            nonce, POOL, set(), plan, f"hotkey-{k % 3}", INDEX)
        counts["cost_aware"].update(cost_part.values())
        counts["coverage"].update(coverage.values())
        for e in list(cost_part.values()) + list(coverage.values()):
            weighted += weights[e] * g[e]
            n += 1
    for part, c in counts.items():
        total = sum(c.values())
        for e, p in parts[part].items():
            sd = float(p * (1 - p) / total) ** 0.5
            assert abs(c[e] / total - float(p)) < 5 * sd + 1e-3, (part, e)
    uniform_mean = sum(g.values()) / len(g)
    assert abs(float(weighted / n / xw.WEIGHT_SCALE) - uniform_mean) < 0.03


def test_the_cap_is_on_expected_spend_not_on_realised_spend():
    """The draw never looks at what earlier calls cost: the coverage shares
    are fixed before the nonce exists and every miner's 44 coverage calls are
    drawn from them. So a miner's realised coverage spend can pass the cap in
    an epoch while its expectation stays within it; a rule that stopped at a
    realised total would make an entry's chance depend on the draws before
    it, and the weights would no longer be inverse probabilities."""
    pairs = {**FULL, DEAREST: 0}
    costs = {**W0_COSTS, DEAREST: Fraction(3, 4)}        # dear enough for the cap to bind
    plan = xw.plan_shares_v4(costs, pairs)
    cov = plan.part_probabilities()["coverage"]
    assert N_COV * sum(cov[e] * costs[e] for e in cov) == COV_CAP
    over = 0
    for k in range(200):
        nonce = hashlib.sha256(f"v4-spend-{k}".encode()).digest()
        _, _, coverage = epoch_plan.exploration_plan_v4(nonce, POOL, set(), plan, "hotkey-a", INDEX)
        assert len(coverage) == N_COV
        realised = sum(costs[e] for e in coverage.values())
        over += realised > COV_CAP
    assert over > 0
    # The shares, and so the weights, are the same for every miner and nonce.
    assert xw.plan_shares_v4(costs, pairs) == plan


def test_v4_shares_come_from_the_menu_state():
    from fugal_subnet import menu_state as ms

    state = ms.seed_state(1000)
    pairs = {e: 40 for e in state.ids()}
    plan = epoch_plan.explore_shares_v4(state, pairs)
    assert plan == xw.plan_shares_v4(ms.costs(state), pairs)
    assert sorted(plan.cost) == list(state.ids())


def test_v4_shares_read_the_exploration_pairs_the_state_publishes():
    """Under v4 n_e is what the epoch's menu publishes for each entry
    (menu_state.Entry.explore_pairs), so every TD and validator holding the
    menu draws the same coverage part without any other input."""
    import dataclasses

    from fugal_subnet import menu_state as ms

    seed = ms.seed_state(1000)
    ids = seed.ids()
    counted = dataclasses.replace(seed, entries=tuple(
        dataclasses.replace(e, explore_pairs=(0 if i % 7 == 0 else 60 + 13 * i))
        for i, e in enumerate(seed.entries)))
    plan = epoch_plan.explore_shares_v4(counted)
    assert plan == xw.plan_shares_v4(ms.costs(counted), counted.explore_pairs())
    assert plan.deficits == {e: max(0, config.EXPLORE_COVERAGE_TARGET_V4 - n)
                             for e, n in counted.explore_pairs().items()}
    # The seed publishes v3's pooled pairs (2,689 short of the target in
    # all): the coverage part starts fully on the deficits.
    first = epoch_plan.explore_shares_v4(seed)
    assert first.deficits == {e: max(0, config.EXPLORE_COVERAGE_TARGET_V4 - n)
                              for e, n in seed.explore_pairs().items()}
    assert sorted(first.deficits) == sorted(ids) and first.damping == 1


# --- v3 is unchanged ---------------------------------------------------------------


def test_v3_weights_are_the_published_ones():
    """Every exploration-pair weight published in the v3 reveals
    e00026466..e00026489 (4,657 pairs), against today's functions."""
    doc = json.loads((FIXTURES / "v3_published_explore_weights.json").read_text())
    assert doc["pairs"] == 4657
    for key, menu in (("menu_a", None), ("menu_b", routing_protocol.MENU_B)):
        models = routing_protocol.models_for_protocol(3, menu) if menu else \
            routing_protocol.models_for_protocol(3)
        assert sorted(models) == sorted(doc[key]["weights"])
        assert xw.weights(epoch_plan.explore_shares(3, models)) == doc[key]["weights"]


def test_v3_shares_draws_and_weights_are_byte_identical():
    """v3's shares, weights and every miner's plan on recorded inputs, hashed
    before the v4 code was added."""
    pool = [{"question_id": f"{b}-{i:05d}", "benchmark": b}
            for b in sorted(config.BENCHMARK_CATEGORY_V3) for i in range(150)]
    recorded = {"a": "105963efa019a632f63fdc9f2c0d5d4c57f2e8cb1be52d25f905d8f40665f6da",
                "b": "038bcc5727a3fb1626033879cbd29917fcfb004b237fd25a5be81a92184b14d8"}
    for label, menu in (("a", None), ("b", routing_protocol.MENU_B)):
        models = routing_protocol.models_for_protocol(3, menu) if menu else \
            routing_protocol.models_for_protocol(3)
        shares = epoch_plan.explore_shares(3, models)
        plans = {}
        for k in range(4):
            nonce = hashlib.sha256(f"v3-record-{k}".encode()).digest()
            scored = {q["question_id"] for q in epoch_plan.scored_slice(3, nonce, pool, 300)}
            for hk in ("5Fhotkey-a", "5Fhotkey-b"):
                p = epoch_plan.exploration_plan(3, nonce, pool, scored, models, 0, hotkey=hk,
                                                shares=shares)
                plans[f"{k}:{hk}"] = sorted(p.items())
        doc = {"shares": sorted(shares.items()), "weights": sorted(xw.weights(shares).items()),
               "plans": plans}
        digest = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode())
        assert digest.hexdigest() == recorded[label]
