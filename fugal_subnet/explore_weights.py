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

PROTOCOL V4 (`plan_shares_v4`; docs/decisions/LIVE_MENU_V4.md, section 8).
The costs are the menu state's measured ones (menu_state.costs), and a
miner's private set has two parts: the cost-aware part, drawn by the shares
above, and a coverage part drawn toward entries short of exploration pairs.
Each part's probabilities are a function of published state, so each pair
carries the exact inverse of the probability it was drawn with: a shared pair
the shared set's, a private pair the private set's (its two parts in
proportion to their calls). Nothing here runs before a v4 epoch.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
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
    if int(version) == 3 and any(e not in table for e in models):
        # v3 menu B: its added entries' rows (menu A's rows are the same in both
        # files, so an epoch on menu A draws exactly as before).
        b = routing_protocol.MENU_B
        table = {**{r["id"]: r for r in json.loads(routing_protocol.prices_path(3, b).read_text())},
                 **table}
        manifest = {**{r["id"]: r for r in json.loads(
            routing_protocol.manifest_path(3, b).read_text())["models"]}, **manifest}
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


# --- protocol v4: exploration that measures every entry ------------------------
#
# docs/decisions/LIVE_MENU_V4.md, section 8. Reached only from a v4 epoch
# (epoch_plan.explore_shares_v4 / exploration_plan_v4); nothing on the v1-v3
# path calls it.


def _exact(values, what: str) -> None:
    if any(isinstance(v, float) for v in values):
        raise ValueError(f"{what} must be exact (int or Fraction), never float")


def probabilities(entry_shares: Mapping[str, int]) -> dict[str, Fraction]:
    """{entry: share / sum(shares)}: the probability a draw by these shares
    takes each entry, exactly."""
    values = [int(s) for s in entry_shares.values()]
    if not values or any(s < 0 for s in values) or sum(values) <= 0:
        raise ValueError("shares must be non-negative integers with a positive sum")
    total = sum(values)
    return {e: Fraction(int(s), total) for e, s in sorted(entry_shares.items())}


def inverse_probability(probs: Mapping[str, Fraction]) -> dict[str, Fraction]:
    """{entry: 1 / (M * pi_e)} exactly, for each entry a draw can take
    (pi_e > 0); M is the number of entries. The weight that makes a pair
    drawn with pi count as a uniform draw over the M entries would."""
    m = len(probs)
    return {e: 1 / (m * Fraction(p)) for e, p in sorted(probs.items()) if p > 0}


def exact_shares(probs: Mapping[str, Fraction]) -> dict[str, int]:
    """Integer shares whose share / sum(shares) is each probability exactly:
    the probabilities over their least common denominator. They must be
    non-negative and sum to one."""
    if not probs:
        raise ValueError("no probabilities")
    _exact(probs.values(), "probabilities")
    fr = {e: Fraction(p) for e, p in probs.items()}
    if any(p < 0 for p in fr.values()) or sum(fr.values()) != 1:
        raise ValueError("probabilities must be non-negative and sum to one")
    den = 1
    for p in fr.values():
        den = math.lcm(den, p.denominator)
    return {e: int(p * den) for e, p in sorted(fr.items())}


def mixture_probabilities(parts) -> dict[str, Fraction]:
    """The probability a call of a set made of `parts` takes each entry.
    `parts` is ((calls, shares), ...), every part over the same entries; a
    call is in a part in proportion to the part's calls."""
    parts = [(int(n), s) for n, s in parts if int(n) > 0]
    if not parts:
        raise ValueError("a mixture needs a part with calls")
    entries = set(parts[0][1])
    if any(set(s) != entries for _, s in parts):
        raise ValueError("every part must draw over the same entries")
    calls = sum(n for n, _ in parts)
    probs = [(n, probabilities(s)) for n, s in parts]
    return {e: sum(n * p[e] for n, p in probs) / calls for e in sorted(entries)}


def mixture_weights(parts) -> dict[str, int]:
    """{entry: 1 / (M * pi_e)} in WEIGHT_SCALE units, rounded half up, with
    pi the mixture of `parts` (mixture_probabilities). Every entry must be
    one the mixture can draw."""
    p = mixture_probabilities(parts)
    if any(v <= 0 for v in p.values()):
        raise ValueError("an entry no part can draw has no weight")
    return {e: _round_half_up(w * WEIGHT_SCALE) for e, w in inverse_probability(p).items()}


def _explore_pairs(models, explore_pairs: Mapping[str, int]) -> dict[str, int]:
    """{entry: n_e} for the menu's entries. An entry with no count has none
    (it has just joined); a count for an id not on the menu is not read."""
    out = {}
    for e in models:
        n = explore_pairs.get(e, 0)
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"exploration pairs of {e!r} must be a non-negative integer, not {n!r}")
        out[e] = n
    return out


def coverage_deficits(models, explore_pairs: Mapping[str, int], *, target: int) -> dict[str, int]:
    """{entry: max(0, target - n_e)}, n_e the entry's exploration pairs."""
    return {e: max(0, int(target) - n)
            for e, n in _explore_pairs(sorted(models), explore_pairs).items()}


@dataclass(frozen=True)
class CoverageDraw:
    """The coverage part's probabilities and the two numbers that set them."""

    probabilities: dict[str, Fraction]
    deficits: dict[str, int]
    damping: Fraction          # f: the fraction of the part that follows the deficits
    toward_deficits: Fraction  # t: how far the capped part is from uniform, toward q


def _part_spend(probs: Mapping[str, Fraction], costs: Mapping[str, Fraction], n_calls: int) -> Fraction:
    return n_calls * sum(Fraction(p) * Fraction(costs[e]) for e, p in probs.items())


def coverage_draw(costs: Mapping[str, Fraction], explore_pairs: Mapping[str, int], *, n_calls: int,
                  cap_usd: Fraction | None, target: int, damp_pairs: int) -> CoverageDraw:
    """The coverage part's draw over the entries `costs` names (section 8):

    1. d_e = max(0, target - n_e); D = sum of d_e;
    2. f = min(1, D / damp_pairs); q_e = f * d_e / D + (1 - f) / M
       (q is uniform when D = 0);
    3. pi_e = t * q_e + (1 - t) / M, t the largest value in [0, 1] whose
       expected spend n_calls * sum(pi_e * c_e) is within `cap_usd`. Spend
       is linear in t, so t is an exact fraction. When no t meets the cap
       (uniform coverage alone costs more), t is the end with the lower
       expected spend, uniform on a tie. `cap_usd` None leaves t = 1.
    """
    models = sorted(costs)
    if not models:
        raise ValueError("coverage exploration needs a non-empty menu")
    if int(damp_pairs) <= 0 or int(target) < 0 or int(n_calls) < 0:
        raise ValueError(f"bad coverage parameters ({n_calls}, {target}, {damp_pairs})")
    _exact(costs.values(), "costs")
    m = len(models)
    uniform = Fraction(1, m)
    deficits = coverage_deficits(models, explore_pairs, target=target)
    total = sum(deficits.values())
    damping = min(Fraction(1), Fraction(total, int(damp_pairs)))
    q = {e: (damping * Fraction(deficits[e], total) if total else Fraction(0)) + (1 - damping) * uniform
         for e in models}
    t = Fraction(1)
    if cap_usd is not None:
        _exact((cap_usd,), "the cap")
        cap = Fraction(cap_usd)
        spend_q = _part_spend(q, costs, int(n_calls))
        spend_u = _part_spend(dict.fromkeys(models, uniform), costs, int(n_calls))
        if spend_q <= cap:
            t = Fraction(1)
        elif spend_u <= cap:
            t = (cap - spend_u) / (spend_q - spend_u)       # spend_q > cap >= spend_u
        else:
            t = Fraction(1) if spend_q < spend_u else Fraction(0)
    probs = {e: t * q[e] + (1 - t) * uniform for e in models}
    return CoverageDraw(probabilities=probs, deficits=deficits, damping=damping, toward_deficits=t)


@dataclass(frozen=True)
class SharesV4:
    """An epoch's exploration draw under protocol v4: the same for every
    miner, a function of the menu state and the published pair counts.

    `cost` draws the shared set and each miner's cost-aware part (v3's
    shares on the measured costs); `coverage` draws each miner's coverage
    part. A shared pair weighs 1 / (M * pi_shared). A private pair weighs
    1 / (M * pi_private), pi_private the two parts in proportion to their
    calls: a private question falls in each part with that chance
    (exploration.private_parts), so pi_private is the exact probability its
    entry was drawn with."""

    cost: dict[str, int]
    coverage: dict[str, int]
    exponent_halves: int
    deficits: dict[str, int]
    damping: Fraction
    toward_deficits: Fraction
    shared_calls: int
    cost_calls: int
    coverage_calls: int

    def models(self) -> list[str]:
        return sorted(self.cost)

    def _private_parts(self):
        return ((self.cost_calls, self.cost), (self.coverage_calls, self.coverage))

    def part_probabilities(self) -> dict[str, dict[str, Fraction]]:
        """{part: {entry: probability}} for the shared set, the cost-aware
        part, the coverage part and the private set as a whole."""
        cost = probabilities(self.cost)
        return {"shared": cost, "cost_aware": dict(cost), "coverage": probabilities(self.coverage),
                "private": mixture_probabilities(self._private_parts())}

    def shared_weights(self) -> dict[str, int]:
        """{entry: weight of a shared pair}: v3's weights of the cost shares."""
        return weights(self.cost)

    def private_weights(self) -> dict[str, int]:
        """{entry: weight of a private pair}, from the private set's mixture."""
        return mixture_weights(self._private_parts())

    def pair_weight(self, entry: str, *, private: bool) -> int:
        """The weight a pooled pair of `entry` carries (WEIGHT_SCALE units).
        A caller weighing many pairs takes the two dicts once instead."""
        return (self.private_weights() if private else self.shared_weights())[entry]

    def expected_spend(self, costs: Mapping[str, Fraction]) -> dict[str, Fraction]:
        """A miner's expected dollars: the calls drawn by the cost shares (the
        shared set and the cost-aware part, the calls their cap is on) and
        the coverage part."""
        return {"cost_aware": expected_spend(self.cost, costs, self.shared_calls + self.cost_calls),
                "coverage": expected_spend(self.coverage, costs, self.coverage_calls)}


def coverage_first(costs: Mapping[str, Fraction], explore_pairs: Mapping[str, int],
                   first, *, n_calls: int, cap_usd: Fraction, target: int,
                   damp_pairs: int) -> CoverageDraw:
    """The coverage part's draw aimed first at the entries `first` (the
    provisional entries in their join epoch, FAST_CROWN_V4.md 18(c)):

    1. they take a share a = min(1, cap / (n_calls x their mean cost)) of the
       part, split equally among them, undamped;
    2. the rest, 1 - a, is `coverage_draw` over the other entries with what
       is left of the cap per call of the rest, scaled by 1 - a.

    With `first` empty it is `coverage_draw`. Exact Fractions throughout."""
    first = tuple(sorted(set(first)))
    if not first:
        return coverage_draw(costs, explore_pairs, n_calls=n_calls, cap_usd=cap_usd,
                             target=target, damp_pairs=damp_pairs)
    models = sorted(costs)
    if any(e not in costs for e in first):
        raise ValueError("an entry coverage is aimed at is not on the menu")
    _exact(costs.values(), "costs")
    mean = sum(Fraction(costs[e]) for e in first) / len(first)
    share = Fraction(1) if cap_usd is None else min(Fraction(1), Fraction(cap_usd) / (int(n_calls) * mean))
    probs = dict.fromkeys(models, Fraction(0))
    for e in first:
        probs[e] = share / len(first)
    rest = [e for e in models if e not in first]
    deficits = coverage_deficits(models, explore_pairs, target=target)
    if share < 1 and rest:
        left = None if cap_usd is None else max(
            Fraction(0), Fraction(cap_usd) - int(n_calls) * share * mean) / (1 - share)
        inner = coverage_draw({e: costs[e] for e in rest}, explore_pairs, n_calls=n_calls,
                              cap_usd=left, target=target, damp_pairs=damp_pairs)
        for e in rest:
            probs[e] = (1 - share) * inner.probabilities[e]
        return CoverageDraw(probabilities=probs, deficits=deficits, damping=inner.damping,
                            toward_deficits=inner.toward_deficits)
    if share < 1:                         # every entry is first: the part is theirs
        probs = {e: (Fraction(1, len(first)) if e in first else Fraction(0)) for e in models}
    return CoverageDraw(probabilities=probs, deficits=deficits, damping=Fraction(1),
                        toward_deficits=Fraction(1))


def plan_shares_v4(costs: Mapping[str, Fraction], explore_pairs: Mapping[str, int],
                   first=()) -> SharesV4:
    """The epoch's v4 exploration draw under the release's constants (config):

    - cost shares: v3's rule (`choose_exponent`, `shares`) on `costs`, the
      menu state's measured dollars per call, with EXPLORE_SPEND_CAP_USD_V4
      over EXPLORE_SHARED_SIZE_V4 + EXPLORE_COST_AWARE_SIZE_V4 calls and v3's
      EXPLORE_COST_FLOOR and EXPLORE_COST_MAX_HALVES;
    - coverage shares: `coverage_draw` over EXPLORE_COVERAGE_SIZE_V4 calls
      with EXPLORE_COVERAGE_TARGET_V4, EXPLORE_COVERAGE_DAMP_PAIRS_V4 and
      EXPLORE_COVERAGE_SPEND_CAP_USD_V4, as exact integer shares.

    `explore_pairs` is {entry: n_e}: each entry's EXPLORATION pairs (never a
    proposer's measurement pairs) in the window as published MENU_LAG epochs
    back. `first`: the entries the coverage part is aimed at first
    (`coverage_first`)."""
    from fugal_subnet import config

    if not costs:
        raise ValueError("v4 exploration needs a non-empty menu")
    _exact(costs.values(), "costs")
    costs = {e: Fraction(costs[e]) for e in sorted(costs)}
    floor = Fraction(config.EXPLORE_COST_FLOOR)
    n_shared, n_cost = int(config.EXPLORE_SHARED_SIZE_V4), int(config.EXPLORE_COST_AWARE_SIZE_V4)
    n_cov = int(config.EXPLORE_COVERAGE_SIZE_V4)
    h = choose_exponent(costs, n_calls=n_shared + n_cost,
                        cap_usd=Fraction(config.EXPLORE_SPEND_CAP_USD_V4), floor=floor,
                        max_halves=config.EXPLORE_COST_MAX_HALVES)
    cover = coverage_first(costs, explore_pairs, first, n_calls=n_cov,
                           cap_usd=Fraction(config.EXPLORE_COVERAGE_SPEND_CAP_USD_V4),
                           target=config.EXPLORE_COVERAGE_TARGET_V4,
                           damp_pairs=config.EXPLORE_COVERAGE_DAMP_PAIRS_V4)
    return SharesV4(cost=shares(costs, exponent_halves=h, floor=floor),
                    coverage=exact_shares(cover.probabilities), exponent_halves=h,
                    deficits=cover.deficits, damping=cover.damping,
                    toward_deficits=cover.toward_deficits, shared_calls=n_shared,
                    cost_calls=n_cost, coverage_calls=n_cov)
