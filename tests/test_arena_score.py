"""Protocol v5's per-question value (arena_score): RouterArena's score,
linearised per compared pair, in exact integers."""
from __future__ import annotations

import math
from fractions import Fraction

import pytest

from fugal_subnet import arena_score as a
from fugal_subnet import config
from fugal_subnet.benchmarks.slicer import protocol_version_for_boundary

ND_PER_1K = a.NANO_PER_QUERY_PER_USD_PER_1K   # $1 per 1K queries, in nanodollars per query


def test_arena_matches_routerarenas_published_scores():
    # Divyam, Sqwish and the v4 seat head, from RouterArena's own code on sub_10.
    assert a.arena(0.8354, 0.655) == pytest.approx(79.45, abs=0.01)
    assert a.arena(0.8614, 0.441) == pytest.approx(82.32, abs=0.01)
    assert a.arena(0.7954, 0.618) == pytest.approx(76.24, abs=0.01)
    assert a.arena(0.9, 1e9) == 0.0


@pytest.mark.parametrize("num,den", [(1, 1), (2, 1), (3, 1), (200, 1), (1, 3), (500000, 11),
                                     (7, 1024), (10**12, 7), (2**60 + 1, 3)])
def test_log2_fixed_is_the_logarithm_within_its_bits(num, den):
    got = a.log2_fixed(num, den)
    assert got.denominator <= 2 ** a.LOG2_BITS
    assert abs(float(got) - math.log2(num / den)) < 1e-12
    # truncation: never above the true value (2^got <= num/den)
    assert got <= Fraction(math.log2(num / den)) + Fraction(1, 2 ** 40)


def test_log2_fixed_powers_of_two_are_exact():
    for k in range(-20, 40):
        num, den = (2 ** k, 1) if k >= 0 else (1, 2 ** -k)
        assert a.log2_fixed(num, den) == k


@pytest.mark.parametrize("bad", [(0, 1), (1, 0), (-1, 1), (1.5, 1), (True, 1)])
def test_log2_fixed_refuses_non_positive_or_non_int(bad):
    with pytest.raises(ValueError):
        a.log2_fixed(*bad)


def test_log_mean_is_symmetric_between_the_means_and_floored():
    for x, y in [(1, 1), (440_000, 1_200_000), (5, 5_000_000), (10**9, 10**6)]:
        lm = a.log_mean_nd(x, y)
        assert lm == a.log_mean_nd(y, x)
        assert math.isqrt(x * y) <= lm <= (x + y) // 2 + 1
    assert a.log_mean_nd(0, 0) == 1
    with pytest.raises(ValueError):
        a.log_mean_nd(-1, 3)


def test_log_mean_is_within_a_percent_of_the_true_one_up_to_a_ratio_of_ten():
    for r in (2, 4, 10):
        x, y = 1_000_000, 1_000_000 * r
        true = (y - x) / math.log(y / x)
        assert abs(a.log_mean_nd(x, y) / true - 1) < 0.0085


def test_k_pair_is_routerarenas_trade_off_at_the_reference_accuracy():
    # k = C^2 ln2 log2(200/0.0044) / (b a^2), C at the pair's cost; 43-52 where routers run.
    for usd_per_1k in (0.44, 0.5, 0.62):
        c = round(usd_per_1k * ND_PER_1K)
        cs = math.log2(200 / usd_per_1k) / math.log2(200 / 0.0044)
        expect = cs * cs * math.log(2) * math.log2(200 / 0.0044) / (0.1 * 0.82 ** 2)
        assert float(a.k_pair(c, c)) == pytest.approx(expect, rel=1e-9)
        assert 43 < float(a.k_pair(c, c)) < 53
    assert a.k_pair(10**12, 10**12) == 0          # at or above the $200 per 1K ceiling
    assert a.k_pair(1, 1) == a.k_pair(4_400, 4_400)   # below the floor: clamped


def test_value_of_a_correct_answer_is_about_three_cents_where_routers_run():
    v = a.value_scale_nd(620_000, 620_000)      # $0.62 per 1K
    assert 0.025 < float(v) / a.NANO < 0.032
    assert a.value_scale_nd(10**12, 10**12) == 1   # floored at one nanodollar


def test_item_is_accuracy_minus_cost_over_v_in_bound_units():
    v = 30_000_000                                # $0.03
    y = config.Y_SCALE_V5
    assert a.item_units(y, 0, 0, 0, v) == 100 * a.UNITS_PER_POINT       # one right answer: 100 points
    assert a.item_units(0, y, 0, 0, v) == -100 * a.UNITS_PER_POINT
    assert a.item_units(y, y, 300_000, 0, v) == -1_000_000            # $0.0003 more = -1 point
    assert a.item_units(0, 0, 0, 0, v) == 0
    # a median of votes may be a half; the result is floored exactly
    assert a.item_units(Fraction(y, 2), 0, 0, 0, v) == 50 * a.UNITS_PER_POINT
    assert a.item_units(0, 0, 1, 0, Fraction(3)) == math.floor(-100 * a.UNITS_PER_POINT / 3)


def test_items_are_antisymmetric_up_to_the_floor():
    for args in [(10_000, 0, 5, 9, 31_000_000), (3_333, 7_777, 1_234_567, 7, 25_000_017)]:
        x = a.item_units(*args)
        yx, yy, tx, ty, v = args
        assert x + a.item_units(yy, yx, ty, tx, v) in (0, -1)


@pytest.mark.parametrize("bad", [
    dict(y_x=-1), dict(y_x=config.Y_SCALE_V5 + 1), dict(y_x=0.5), dict(y_x=True),
    dict(t_x_nd=-1), dict(t_x_nd=1.0), dict(v_nd=0), dict(v_nd=-5), dict(v_nd=1.5)])
def test_item_refuses_out_of_range_or_inexact_inputs(bad):
    args = dict(y_x=0, y_y=0, t_x_nd=0, t_y_nd=0, v_nd=1)
    args.update(bad)
    with pytest.raises(ValueError):
        a.item_units(**args)


def test_the_s_gap_tracks_the_arena_gap():
    # v4 seat (79.54%, $0.618/1K) vs the v3 champion (78.83%, $1.204/1K): Arena +1.84.
    ax, cx = 0.7954, 618_000
    ay, cy = 0.7883, 1_204_000
    v = a.value_scale_nd(cx, cy)
    s_gap = 100 * ((ax - ay) - (cx - cy) / float(v))
    arena_gap = a.arena(ax, cx / ND_PER_1K) - a.arena(ay, cy / ND_PER_1K)
    assert arena_gap == pytest.approx(1.84, abs=0.02)
    assert s_gap == pytest.approx(arena_gap * 1.2, rel=0.25)   # ~0.8 Arena per S-point


def test_protocol_five_follows_four_and_is_never_before_it():
    v2, v3, v4, v5 = 100, 200, 300, 400
    assert protocol_version_for_boundary(399, v2, v3, v4_activation_block=v4,
                                         v5_activation_block=v5) == 4
    assert protocol_version_for_boundary(400, v2, v3, v4_activation_block=v4,
                                         v5_activation_block=v5) == 5
    assert protocol_version_for_boundary(400, v2, v3, v4_activation_block=v4) == 4
    with pytest.raises(ValueError):
        protocol_version_for_boundary(400, v2, v3, v4_activation_block=None, v5_activation_block=v5)
    with pytest.raises(ValueError):
        protocol_version_for_boundary(400, v2, v3, v4_activation_block=v4, v5_activation_block=250)


def test_v5_activation_block_follows_the_v4_rule(monkeypatch):
    monkeypatch.setattr(config, "V5_ACTIVATION_BLOCK_MALFORMED", None)
    monkeypatch.setattr(config, "V5_ACTIVATION_BLOCK_ENV", None)
    assert config.v5_activation_block(config.FINNEY_GENESIS_HASH) is None
    monkeypatch.setattr(config, "V5_ACTIVATION_BLOCK_ENV", 9_600_000)
    with pytest.raises(ValueError):          # finney: the variable may only repeat the release
        config.v5_activation_block(config.FINNEY_GENESIS_HASH)
    monkeypatch.setattr(config, "V5_ACTIVATION_BLOCK_MALFORMED", "soon")
    monkeypatch.setattr(config, "V5_ACTIVATION_BLOCK_ENV", None)
    assert config.v5_activation_block(config.FINNEY_GENESIS_HASH) is None
    with pytest.raises(ValueError):
        config.v5_activation_block("0xlocal")
