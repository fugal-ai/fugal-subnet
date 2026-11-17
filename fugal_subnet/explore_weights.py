"""Cost-aware exploration and the weights that keep it unbiased (protocol v3).

WHY. Exploration sends a question to a menu entry nobody chose; pooled, those
pairs are what a head's odds are scored on (odds_score.py). Drawn uniformly over
the v3 menu (53 entries, answer budgets up to ~128k tokens), a miner pays as
much for each Fable 5.2 @max call (~$5 at 128k) as for hundreds of cheap
ones, and exploration is a cost every miner bears every epoch.

WHAT. Entry e is drawn with probability

    pi_e = (1 - floor) * r_e / sum(r) + floor / M,     r_e = (c_max / c_e) ^ alpha

where c_e is its expected dollars per call (pinned prices x token means), M
the menu size and alpha = exponent_halves / 2. The floor keeps every entry at
least `floor` of a uniform share, so an expensive entry is still measured and
no weight exceeds 1 / floor. alpha is the SMALLEST (most uniform) value in
0, 1/2, ..., max_halves/2 whose expected spend for a miner's exploration calls
is within the cap: as uniform as the budget allows.

UNBIASED. Uniform sampling counted every entry equally. Each pooled pair
carries the weight w_e = 1 / (M * pi_e), whose mean under the sampling
distribution is one, so a weighted mean of per-pair Brier gains estimates
exactly what uniform sampling would have (Horvitz-Thompson). Per-entry
statistics (the reference R's rate per entry and benchmark, single_model's
accuracy and tokens) are conditional on the entry and need no weight.

EXACT. Costs are Fractions of the pinned JSON numbers; the square root of a
cost ratio is an integer square root at a fixed scale; shares are integers;
weights are integers in WEIGHT_SCALE units. Every validator computes the same
numbers on any CPU (I1).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping
from fractions import Fraction

WEIGHT_SCALE = 1_000_000          # a pair's weight of 1.0
SHARE_SCALE = 1_000_000_000       # shares are integers summing to about this
_SQRT_SCALE = 10**12              # fixed-point scale of the integer square root


def _round_half_up(x: Fraction) -> int:
    return int((2 * x + 1) // 2)


def _sqrt(x: Fraction) -> Fraction:
    """floor(sqrt(x) * _SQRT_SCALE) / _SQRT_SCALE, exactly."""
    return Fraction(math.isqrt(int(x * _SQRT_SCALE * _SQRT_SCALE)), _SQRT_SCALE)


def entry_costs(models: list[str], version: int = 3) -> dict[str, Fraction]:
    """{entry: expected dollars per call} from the pinned menu and token manifest:
    mean prompt tokens x input price + mean completion tokens x output price.
    An entry missing from either table is a KeyError, never a default."""
    from fugal_subnet import routing_protocol

    table = {r["id"]: r for r in json.loads(routing_protocol.prices_path(version).read_text())}
    manifest = {r["id"]: r for r in
                json.loads(routing_protocol.manifest_path(version).read_text())["models"]}
    out = {}
    for e in models:
        price, tok = table[e], manifest[e]
        out[e] = (Fraction(tok["mean_in_tokens"]) * Fraction(price["in"])
                  + Fraction(tok["mean_out_tokens"]) * Fraction(price["out"])) / 1_000_000
    return out


def shares(costs: Mapping[str, Fraction], *, exponent_halves: int, floor: Fraction) -> dict[str, int]:
    """{entry: integer share}; entry e is drawn with probability share / sum(shares)."""
    if not costs:
        raise ValueError("cost-aware exploration needs a non-empty menu")
    if exponent_halves < 0 or not 0 < floor <= 1:
        raise ValueError(f"bad exploration sampling parameters ({exponent_halves}, {floor})")
    names = sorted(costs)
    if any(costs[e] <= 0 for e in names):
        raise ValueError("every entry needs a positive expected cost")
    top = max(costs[e] for e in names)
    raw = {}
    for e in names:
        t = top / costs[e]                                    # >= 1
        r = t ** (exponent_halves // 2)
        if exponent_halves % 2:
            r *= _sqrt(t)
        raw[e] = r
    total = sum(raw.values())
    m = len(names)
    return {e: max(1, _round_half_up(((1 - floor) * raw[e] / total + floor / m) * SHARE_SCALE))
            for e in names}


def weights(entry_shares: Mapping[str, int]) -> dict[str, int]:
    """{entry: inverse-probability weight in WEIGHT_SCALE units}: 1 / (M * pi_e)."""
    total, m = sum(entry_shares.values()), len(entry_shares)
    return {e: _round_half_up(Fraction(total, m * s) * WEIGHT_SCALE)
            for e, s in sorted(entry_shares.items())}


def expected_spend(entry_shares: Mapping[str, int], costs: Mapping[str, Fraction],
                   n_calls: int) -> Fraction:
    """A miner's expected exploration dollars for `n_calls` drawn by these shares."""
    total = sum(entry_shares.values())
    return n_calls * sum(Fraction(s, total) * costs[e] for e, s in entry_shares.items())


def choose_exponent(costs: Mapping[str, Fraction], *, n_calls: int, cap_usd: Fraction,
                    floor: Fraction, max_halves: int) -> int:
    """The smallest exponent (in halves) whose expected spend is within the cap;
    `max_halves` when none is."""
    for h in range(int(max_halves) + 1):
        if expected_spend(shares(costs, exponent_halves=h, floor=floor), costs, n_calls) <= cap_usd:
            return h
    return int(max_halves)


def plan_shares(models: list[str], costs: Mapping[str, Fraction] | None = None) -> dict[str, int]:
    """The epoch's sampling shares under the release's constants (config):
    EXPLORE_SPEND_CAP_USD over EXPLORE_SHARED_SIZE + EXPLORE_PRIVATE_SIZE calls,
    EXPLORE_COST_FLOOR, EXPLORE_COST_MAX_HALVES."""
    from fugal_subnet import config

    costs = entry_costs(list(models)) if costs is None else costs
    floor = Fraction(config.EXPLORE_COST_FLOOR)
    h = choose_exponent(costs, n_calls=config.EXPLORE_SHARED_SIZE + config.EXPLORE_PRIVATE_SIZE,
                        cap_usd=Fraction(config.EXPLORE_SPEND_CAP_USD), floor=floor,
                        max_halves=config.EXPLORE_COST_MAX_HALVES)
    return shares(costs, exponent_halves=h, floor=floor)


def assign_by_share(nonce: bytes, chosen: list[dict], entry_shares: Mapping[str, int]) -> dict[str, str]:
    """{question_id: entry} for an already chosen exploration set.

    Systematic sampling on a line of length sum(shares), the entries laid on it
    in a nonce-keyed order: question k (in id order) of n takes the entry under
    position (start + k * total // n) mod total, with a keyed start. Each
    question's entry is drawn with probability exactly share / total, and the
    entries come out as evenly spread as the shares allow (uniform shares give
    a round-robin). Deterministic in (nonce, question ids, shares)."""
    if not entry_shares:
        raise ValueError("exploration requires a non-empty model list")
    order = sorted(entry_shares, key=lambda m: hmac.new(
        nonce, f"model-order:{m}".encode("utf-8"), hashlib.sha256).digest())
    total = sum(int(entry_shares[m]) for m in order)
    bounds, acc = [], 0
    for m in order:
        acc += int(entry_shares[m])
        bounds.append(acc)
    ordered = sorted(chosen, key=lambda q: q["question_id"])
    n = len(ordered)
    start = int.from_bytes(hmac.new(nonce, b"explore-start", hashlib.sha256).digest(), "big") % total
    out = {}
    for k, q in enumerate(ordered):
        pos = (start + (k * total) // n) % total
        lo, hi = 0, len(bounds) - 1
        while lo < hi:                               # first bound above pos
            mid = (lo + hi) // 2
            if bounds[mid] > pos:
                hi = mid
            else:
                lo = mid + 1
        out[q["question_id"]] = order[lo]
    return out


def apply(units, pair_weights):
    """Per-pair loss units times their weights, rounded half up, int64."""
    import numpy as np

    u = np.asarray(units, dtype=np.int64)
    w = np.asarray(pair_weights, dtype=np.int64)
    return (2 * u * w + WEIGHT_SCALE) // (2 * WEIGHT_SCALE)
