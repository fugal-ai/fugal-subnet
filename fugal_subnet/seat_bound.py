"""Protocol v4: the one bound every seat comparison uses (I1, I3).

docs/decisions/SEAT_MINIMAL_V4.md, section 3.1. Every v4 seat decision (the
odds test, the real-picks check, the memorisation check) compares a paired
mean with a margin and is re-run every epoch, so each needs a lower bound that
stays valid however many times it is looked at. This is that bound, and the
only one:

    items  x_i integers (one-millionth of a point), each in a category c
    n      the number of items;  S = sum x_i;  D = S - margin x n
    V      sum over categories of  sum_c x_i^2 - (sum_c x_i)^2 / n_c
           (exact Fraction: the within-category spread, so it is stratified)
    rho    V x n0 / n                          (n0: the rule's minimum evidence)
    passes n >= n_min  and  D > 0  and
           D^2 > (V + rho) x (A_ALPHA + LN2_UP x bit_length(ceil(1 + n / n0)))

It is the one-sided normal-mixture confidence sequence (Robbins; Howard,
Ramdas, McAuliffe and Sekhon, 2021) at alpha = 1% with the variance estimated
from the items: since (V + rho) / rho = 1 + n / n0, the true boundary is
D^2 > (V + rho)(2 ln(1 / alpha) + ln(1 + n / n0)), and both logarithms are
replaced by pinned rationals that are never below them (`config`), which only
widens it: from n = n0 on (every rule's minimum is its n0) by at most 8% in
the log term, 4% in the width. For a true mean at the margin the chance that it
EVER passes, at any number of looks, is at most about alpha.

DETERMINISM. Integers and Fractions only: no float, no random stream, no
library whose version could change a verdict. The sums are integer sums, so
neither the order of the items nor the order of the categories can matter,
and a window kept as per-epoch sums (`Stratum`, added and subtracted) gives
exactly the bound of its items.

`Bound.lower` is the bound itself, (S - sqrt((V + rho) x log term)) / n,
rounded down to an integer unit: what is published. The verdict is
`Bound.passes`, the design's inequality evaluated exactly; a margin at or
above `lower + 1` never passes, and one below `lower` always does once the
minimum is met.

Nothing on the v1-v3 path reads this module (`champion` re-exports
`lower_bound` for the v4 seat).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt

from fugal_subnet import config

A_ALPHA = Fraction(config.BOUND_A_ALPHA_V4)
LN2_UP = Fraction(config.BOUND_LN2_UP_V4)
UNITS_PER_POINT = config.BOUND_UNITS_PER_POINT_V4


def _int(name: str, value) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, not {type(value).__name__}")
    return value


@dataclass(frozen=True)
class Stratum:
    """One category's sufficient sums: the count, the sum and the sum of
    squares of its items. Sums of disjoint item sets add; an epoch leaves a
    window by subtraction."""

    n: int
    total: int
    squares: int

    def __add__(self, other: Stratum) -> Stratum:
        return Stratum(self.n + other.n, self.total + other.total,
                       self.squares + other.squares)

    def __sub__(self, other: Stratum) -> Stratum:
        return Stratum(self.n - other.n, self.total - other.total,
                       self.squares - other.squares)


@dataclass(frozen=True)
class Bound:
    """The bound on one set of items. `variance` is V; `lower` is the bound
    of the mean in units, rounded down (None with no items); `passes` is the
    verdict against the margin and the minimum it was computed with."""

    n: int
    total: int
    variance: Fraction
    lower: int | None
    passes: bool


def strata(items: Mapping[str, Iterable[int]]) -> dict[str, Stratum]:
    """Each category's sums. Categories are strings and items ints (never a
    bool, a float or a numpy scalar); anything else is a TypeError."""
    out: dict[str, Stratum] = {}
    for cat, values in items.items():
        if type(cat) is not str:
            raise TypeError(f"a category must be a str, not {type(cat).__name__}")
        n = total = squares = 0
        for x in values:
            _int("an item", x)
            n += 1
            total += x
            squares += x * x
        out[cat] = Stratum(n, total, squares)
    return out


INT64_LIMIT = 2**63            # every int64 value v has -INT64_LIMIT <= v < INT64_LIMIT


def _magnitude(a) -> int:
    """The largest |v| of an integer array, as a Python int (0 when empty).
    Computed from min and max as Python ints, so INT64_MIN is not folded."""
    if len(a) == 0:
        return 0
    return max(-int(a.min()), int(a.max()))


def int_sum(a) -> int:
    """The exact sum of an int64 array (or an object array of Python ints)
    as a Python int, whatever the order numpy adds in.

    No overflow: every partial sum of n values of magnitude at most m lies
    within n x m, so when n x m < 2^63 the int64 sum is exact in any order;
    otherwise the values are added as Python ints."""
    import numpy as np

    a = np.asarray(a)
    if a.dtype != np.int64:
        return sum(int(v) for v in a.tolist())
    if len(a) * _magnitude(a) < INT64_LIMIT:
        return int(a.sum())
    return sum(a.tolist())


def int_sum_squares(a) -> int:
    """The exact sum of squares of an int64 array (or an object array of
    Python ints) as a Python int, in int64 arithmetic that cannot overflow.

    With m the largest |v| (b = m.bit_length() bits) and s = ceil(b / 2),
    each v = h 2^s + l with h = v >> s (|h| <= 2^(b - s) <= 2^s) and
    0 <= l < 2^s, so v^2 = h^2 2^(2s) + 2 h l 2^s + l^2 and each of h^2,
    |h l| and l^2 is at most 2^(2s) <= 2^(b + 1). Their three int64 sums
    over n values stay within n x 2^(b + 1), exact in any order while that is
    below 2^63 (asserted); otherwise the squares are added as Python ints."""
    import numpy as np

    a = np.asarray(a)
    n = len(a)
    if n == 0:
        return 0
    if a.dtype != np.int64:
        return sum(int(v) * int(v) for v in a.tolist())
    b = _magnitude(a).bit_length()
    if n * (1 << (b + 1)) >= INT64_LIMIT:
        return sum(v * v for v in a.tolist())
    s = (b + 1) // 2
    assert 2 * s <= b + 1 and n * (1 << (2 * s)) < INT64_LIMIT
    h = a >> s
    lo = a & ((1 << s) - 1)
    return ((int((h * h).sum()) << (2 * s)) + (int((h * lo).sum()) << (s + 1))
            + int((lo * lo).sum()))


def log_steps(n: int, n0: int) -> int:
    """bit_length(ceil(1 + n / n0)): LN2_UP times it is never below
    ln(1 + n / n0)."""
    return (1 + -(-n // n0)).bit_length()


def _ceil_sqrt(w: Fraction) -> int:
    """The least integer r >= 0 with r^2 >= w (w >= 0)."""
    c = -(-w.numerator // w.denominator)  # r^2 is an integer, so r^2 >= w iff r^2 >= ceil(w)
    r = isqrt(c)
    return r if r * r >= c else r + 1


def bound_from_strata(by_category: Mapping[str, Stratum], *, margin_units: int,
                      n_min: int, n0: int) -> Bound:
    """The bound from each category's sums: `margin_units` the margin in
    units (it may be negative), `n_min` the rule's minimum number of items,
    `n0` the evidence the mixture is tuned to (the rule's minimum)."""
    _int("margin_units", margin_units)
    if _int("n_min", n_min) < 0:
        raise ValueError("n_min must be at least 0")
    if _int("n0", n0) < 1:
        raise ValueError("n0 must be at least 1")
    # V = sum over categories of (cq cn - ct^2) / cn, kept as one exact
    # quotient v_num / v_den of Python ints (v_den the lcm of the counts):
    # the same rational as a sum of Fractions, without normalising each term.
    n = total = 0
    v_num, v_den = 0, 1
    for cat, s in by_category.items():
        if type(cat) is not str:
            raise TypeError(f"a category must be a str, not {type(cat).__name__}")
        cn, ct, cq = _int("a count", s.n), _int("a sum", s.total), _int("a sum of squares", s.squares)
        if cn < 0 or cq < 0 or cq * cn < ct * ct or (cn == 0 and (ct or cq)):
            raise ValueError(f"category {cat!r}: no set of items has these sums")
        if cn:
            g = gcd(v_den, cn)
            v_num = v_num * (cn // g) + (cq * cn - ct * ct) * (v_den // g)
            v_den = v_den // g * cn
            n += cn
            total += ct
    variance = Fraction(v_num, v_den)
    if n == 0:
        return Bound(0, 0, variance, None, False)
    # (V + rho) x the log term, rho = V n0 / n: width = w_num / w_den, with
    # the log term (A_ALPHA + LN2_UP k) over the product of their denominators
    a_n, a_d, l_n, l_d = A_ALPHA.numerator, A_ALPHA.denominator, LN2_UP.numerator, LN2_UP.denominator
    w_num = v_num * (n + n0) * (a_n * l_d + l_n * log_steps(n, n0) * a_d)
    w_den = v_den * n * a_d * l_d
    root = _ceil_sqrt_of(w_num, w_den)
    lower = (total - root) // n
    d = total - margin_units * n
    passes = n >= n_min and d > 0 and d * d * w_den > w_num
    return Bound(n, total, variance, lower, passes)


def _ceil_sqrt_of(num: int, den: int) -> int:
    """`_ceil_sqrt(Fraction(num, den))` for num >= 0 < den, without the
    Fraction: r^2 is an integer, so r^2 >= num / den iff r^2 >= ceil(num / den)."""
    c = -(-num // den)
    r = isqrt(c)
    return r if r * r >= c else r + 1


def lower_bound(items: Mapping[str, Iterable[int]], *, margin_units: int, n_min: int,
                n0: int) -> Bound:
    """The one v4 bound (SEAT_MINIMAL_V4.md 3.1) on integer items grouped by
    category: `items` maps each category to its items, in units of
    1 / UNITS_PER_POINT of a point. Pure and exact."""
    return bound_from_strata(strata(items), margin_units=margin_units, n_min=n_min, n0=n0)
