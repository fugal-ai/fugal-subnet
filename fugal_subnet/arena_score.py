"""Protocol v5: a router's value the way RouterArena scores it (I1, I3).

RouterArena scores a router by Arena(a, c) = 100 (1 + b) a C / (b a + C),
with b = 0.1, a its accuracy and C = log2(200 / c) / log2(200 / 0.0044) a cost
score, c its mean cost per 1K queries clamped to [$0.0044, $200]. That is a
ratio of averages, so it cannot be split into per-question items; its local
trade-off can. To first order Arena moves with

    S = accuracy - ln(mean cost per query) / k        (one S-point = 0.01 of S)

    k = C^2 ln2 log2(200 / 0.0044) / (b a^2)

Comparing router X with router Y on the same questions, the ln term is
ln(c_X / c_Y) / k, and the dollar difference of their picks divided by
V_XY = k x LM(c_X, c_Y), LM the logarithmic mean, is exactly that term. So one
question's item is

    x_q = 100 x [(y_X - y_Y) - (t_X - t_Y) / V_XY]          (S-points)

with y the picked cell's label (0..1) and t what the call cost. `k_pair`
evaluates k per compared pair: C at the pair's logarithmic-mean PICK-TABLE
cost per query (c_X, c_Y read no label) and a at the pinned reference
accuracy config.ARENA_A_REF_V5. So V_XY and every item are a fixed function
of the labels and the bound over them (seat_bound.lower_bound) stays valid.

DETERMINISM. Integers and Fractions only on the consensus path: dollars as
integer nanodollars per query; the logarithmic mean by the Carlson
approximation (2 G + A) / 3 with G = isqrt(a b); log2 by integer
fixed-point squaring (`log2_fixed`, LOG2_BITS fractional bits, truncating);
ln 2 a pinned rational; items floored to integer units of
1 / BOUND_UNITS_PER_POINT_V4 of an S-point. `arena()` is a float diagnostic
for reports and decides nothing.
"""
from __future__ import annotations

import math
from fractions import Fraction
from math import isqrt
from numbers import Rational

from fugal_subnet import config

Y_SCALE: int = int(config.Y_SCALE_V5)
UNITS_PER_POINT: int = int(config.BOUND_UNITS_PER_POINT_V4)
A_REF: Fraction = Fraction(config.ARENA_A_REF_V5)
NANO: int = 1_000_000_000          # nanodollars per dollar

# RouterArena's published constants: b, and the cost clamp per 1K queries,
# as rationals; in nanodollars PER QUERY the clamp is [4,400, 200,000,000]
# (a dollar per 1K queries is 1,000,000 nanodollars per query).
ARENA_BETA = Fraction(1, 10)
ARENA_COST_MAX_PER_1K = Fraction(200)
ARENA_COST_MIN_PER_1K = Fraction(44, 10_000)
NANO_PER_QUERY_PER_USD_PER_1K = NANO // 1000
COST_MAX_ND = int(ARENA_COST_MAX_PER_1K * NANO_PER_QUERY_PER_USD_PER_1K)   # 200,000,000
COST_MIN_ND = int(ARENA_COST_MIN_PER_1K * NANO_PER_QUERY_PER_USD_PER_1K)   # 4,400

# Fractional bits of the fixed-point log2, and ln 2 to 30 digits (pinned).
LOG2_BITS = 48
LN2 = Fraction(693147180559945309417232121458, 10**30)


def log2_fixed(num: int, den: int = 1) -> Fraction:
    """log2(num / den), num, den positive ints, as a Fraction with
    denominator 2^LOG2_BITS: the integer part exactly, the fraction by
    repeated squaring of a fixed-point mantissa, truncated (floor of each
    step). Integer operations only, so every host computes the same value;
    it is within LOG2_BITS bits of the true logarithm."""
    if type(num) is not int or type(den) is not int or num <= 0 or den <= 0:
        raise ValueError(f"log2_fixed needs positive ints, got {num!r} / {den!r}")
    n = num.bit_length() - den.bit_length()
    # 2^n <= num / den < 2^(n + 1)
    if (num << max(0, -n)) < (den << max(0, n)):
        n -= 1
    one = 1 << LOG2_BITS
    # mantissa y = (num / den) / 2^n in [1, 2), fixed point
    if n >= 0:
        y = (num << LOG2_BITS) // (den << n)
    else:
        y = (num << (LOG2_BITS - n)) // den
    frac = 0
    for _ in range(LOG2_BITS):
        y = (y * y) >> LOG2_BITS
        frac <<= 1
        if y >= 2 * one:
            y >>= 1
            frac |= 1
    return Fraction(n * one + frac, one)


# log2(200 / 0.0044), the width of RouterArena's cost scale.
LOG2_SPAN: Fraction = log2_fixed(COST_MAX_ND, COST_MIN_ND)


def _nonneg_int(name: str, value) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    return value


def log_mean_nd(a_nd: int, b_nd: int) -> int:
    """The logarithmic mean of two costs in nanodollars, by the Carlson
    approximation (2 sqrt(ab) + (a + b) / 2) / 3, in exact integers (floor).
    It overstates (a - b) / (ln a - ln b) by 0.12% at a cost ratio of 4, 0.84%
    at 10 and 3.4% at 30 (only the cost term's weight moves), and is
    symmetric. A zero cost counts as one nanodollar."""
    a = max(1, _nonneg_int("a_nd", a_nd))
    b = max(1, _nonneg_int("b_nd", b_nd))
    return (2 * isqrt(a * b) + (a + b) // 2) // 3


def cost_score(c_nd: int) -> Fraction:
    """RouterArena's cost score C of a mean cost per query in nanodollars:
    log2(200 / c) / log2(200 / 0.0044), c per 1K queries clamped to
    [$0.0044, $200]; 1 at the cheap end, 0 at the dear end."""
    c = min(max(_nonneg_int("c_nd", c_nd), COST_MIN_ND), COST_MAX_ND)
    return log2_fixed(COST_MAX_ND, c) / LOG2_SPAN


def k_pair(cx_nd: int, cy_nd: int) -> Fraction:
    """k for the pair whose mean pick-table costs per query are cx, cy
    (nanodollars): C(LM(cx, cy))^2 ln2 log2(200 / 0.0044) / (b A_REF^2).
    Label-free. Zero only at or above RouterArena's $200 per 1K ceiling."""
    c = cost_score(log_mean_nd(cx_nd, cy_nd))
    return c * c * LN2 * LOG2_SPAN / (ARENA_BETA * A_REF * A_REF)


def value_scale_nd(cx_nd: int, cy_nd: int) -> Fraction:
    """V_XY in nanodollars: what a cost difference is divided by to be in
    accuracy units, k_pair x LM(c_X, c_Y). At least one nanodollar (a pair at
    RouterArena's cost ceiling, where k is 0, has every cost difference
    count at most that much)."""
    v = k_pair(cx_nd, cy_nd) * log_mean_nd(cx_nd, cy_nd)
    return max(v, Fraction(1))


def _label(name: str, y) -> Fraction:
    if isinstance(y, bool) or not isinstance(y, Rational):
        raise ValueError(f"{name} must be an int or Fraction label, got {y!r}")
    f = Fraction(y)
    if f < 0 or f > Y_SCALE:
        raise ValueError(f"{name} must be within 0..{Y_SCALE}, got {y!r}")
    return f


def _cost(name: str, t) -> Fraction:
    if isinstance(t, bool) or not isinstance(t, Rational) or t < 0:
        raise ValueError(f"{name} must be a non-negative int or Fraction of nanodollars, got {t!r}")
    return Fraction(t)


def item_units(y_x, y_y, t_x_nd, t_y_nd, v_nd) -> int:
    """One question's item for X against Y, in units of 1 / UNITS_PER_POINT of
    an S-point: floor(UNITS x 100 x ((y_X - y_Y) / Y_SCALE - (t_X - t_Y) / V)).
    Labels are 0..Y_SCALE (a median of votes may be a half: Fractions are
    accepted), costs nanodollars, V a positive int or Fraction of nanodollars
    (`value_scale_nd`)."""
    yx, yy = _label("y_x", y_x), _label("y_y", y_y)
    tx, ty = _cost("t_x_nd", t_x_nd), _cost("t_y_nd", t_y_nd)
    if isinstance(v_nd, bool) or not isinstance(v_nd, Rational) or v_nd <= 0:
        raise ValueError(f"v_nd must be a positive int or Fraction, got {v_nd!r}")
    value = (yx - yy) / Y_SCALE - (tx - ty) / Fraction(v_nd)
    return math.floor(UNITS_PER_POINT * 100 * value)


def arena(accuracy: float, cost_per_1k_usd: float) -> float:
    """RouterArena's score of a router with this accuracy (0..1) and mean cost
    per 1K queries in dollars. Diagnostic only: it decides nothing."""
    hi, lo = float(ARENA_COST_MAX_PER_1K), float(ARENA_COST_MIN_PER_1K)
    c = min(max(float(cost_per_1k_usd), lo), hi)
    cs = (math.log2(hi) - math.log2(c)) / (math.log2(hi) - math.log2(lo))
    a = float(accuracy)
    b = float(ARENA_BETA)
    if a <= 0 or cs <= 0:
        return 0.0
    return 100.0 * (1 + b) * a * cs / (b * a + cs)
