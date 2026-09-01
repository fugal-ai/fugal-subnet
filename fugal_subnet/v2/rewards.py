"""Exact v2 bounded-simplex weight projection with forced-zero overrides."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Mapping

MAX_WEIGHT_DELTA = Decimal("0.3")
WEIGHT_PRECISION = 12


class ProjectionInfeasible(RuntimeError):
    """No sum-one vector can satisfy the ordinary-miner change bounds."""


@dataclass(frozen=True)
class WeightResult:
    weights: dict[int, Decimal]
    precision: int

    def as_lists(self) -> tuple[list[int], list[float]]:
        uids = sorted(self.weights)
        return uids, [float(self.weights[uid]) for uid in uids]

    def serialized(self) -> dict[str, str]:
        return {
            str(uid): format(self.weights[uid], f".{self.precision}f")
            for uid in sorted(self.weights)
        }


def _decimal(value: int | float | str | Decimal, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def _box_simplex(
    target: dict[int, Decimal],
    lower: dict[int, Decimal],
    upper: dict[int, Decimal],
) -> dict[int, Decimal]:
    one = Decimal(1)
    if sum(lower.values()) > one or sum(upper.values()) < one:
        raise ProjectionInfeasible("ordinary-miner bounds cannot sum to one")

    free = set(target)
    fixed: dict[int, Decimal] = {}
    remaining = one
    while free:
        ordered = sorted(free)
        shift = (sum(target[uid] for uid in ordered) - remaining) / len(ordered)
        below = [uid for uid in ordered if target[uid] - shift < lower[uid]]
        above = [uid for uid in ordered if target[uid] - shift > upper[uid]]
        if not below and not above:
            for uid in ordered:
                fixed[uid] = target[uid] - shift
            break
        for uid in below:
            fixed[uid] = lower[uid]
            remaining -= lower[uid]
            free.remove(uid)
        for uid in above:
            if uid not in free:
                continue
            fixed[uid] = upper[uid]
            remaining -= upper[uid]
            free.remove(uid)
        if remaining < 0:
            raise ProjectionInfeasible("lower bounds exceed the remaining simplex mass")
    return fixed


def _quantize_exact(
    projected: dict[int, Decimal],
    lower: dict[int, Decimal],
    upper: dict[int, Decimal],
    precision: int,
) -> dict[int, Decimal]:
    scale = 10 ** precision
    scale_decimal = Decimal(scale)
    lower_units = {
        uid: int((lower[uid] * scale_decimal).to_integral_value(rounding=ROUND_CEILING))
        for uid in projected
    }
    upper_units = {
        uid: int((upper[uid] * scale_decimal).to_integral_value(rounding=ROUND_FLOOR))
        for uid in projected
    }
    raw_units = {uid: projected[uid] * scale_decimal for uid in projected}
    units = {
        uid: min(
            upper_units[uid],
            max(
                lower_units[uid],
                int(raw_units[uid].to_integral_value(rounding=ROUND_FLOOR)),
            ),
        )
        for uid in projected
    }

    difference = scale - sum(units.values())
    if difference > 0:
        order = sorted(
            units,
            key=lambda uid: (-(raw_units[uid] - int(raw_units[uid])), uid),
        )
        while difference:
            progressed = False
            for uid in order:
                if units[uid] < upper_units[uid]:
                    units[uid] += 1
                    difference -= 1
                    progressed = True
                    if difference == 0:
                        break
            if not progressed:
                raise ProjectionInfeasible("rounded upper bounds cannot sum to one")
    elif difference < 0:
        order = sorted(
            units,
            key=lambda uid: (raw_units[uid] - int(raw_units[uid]), -uid),
        )
        while difference:
            progressed = False
            for uid in order:
                if units[uid] > lower_units[uid]:
                    units[uid] -= 1
                    difference += 1
                    progressed = True
                    if difference == 0:
                        break
            if not progressed:
                raise ProjectionInfeasible("rounded lower bounds exceed one")

    return {uid: Decimal(units[uid]) / scale_decimal for uid in units}


def compute_bounded_weights(
    scores: Mapping[int, int | float | str | Decimal],
    previous_weights: Mapping[int, int | float | str | Decimal],
    eligible_uids: set[int],
    forced_zero_uids: set[int],
    *,
    max_delta: int | float | str | Decimal = MAX_WEIGHT_DELTA,
    precision: int = WEIGHT_PRECISION,
) -> WeightResult | None:
    """Project positive score targets into an exact, capped sum-one vector.

    Invalid, duplicate, liveness-disqualified, and otherwise ineligible UIDs are
    zeroed immediately and bypass the downward cap. If no positive eligible
    score exists, return ``None`` so the caller preserves prior chain weights.
    """
    if precision < 0 or precision > 18:
        raise ValueError("precision must be between 0 and 18")
    delta = _decimal(max_delta, "max_delta")
    if delta < 0 or delta > 1:
        raise ValueError("max_delta must be between zero and one")

    eligible = sorted(set(eligible_uids) - set(forced_zero_uids))
    score_values = {
        uid: max(Decimal(0), _decimal(scores.get(uid, 0), f"score[{uid}]"))
        for uid in eligible
    }
    score_total = sum(score_values.values())
    if not eligible or score_total <= 0:
        return None

    previous = {
        int(uid): _decimal(weight, f"previous_weights[{uid}]")
        for uid, weight in previous_weights.items()
    }
    if any(weight < 0 for weight in previous.values()):
        raise ValueError("previous weights cannot be negative")
    previous_total = sum(previous.values())
    if previous and previous_total <= 0:
        raise ValueError("non-empty previous weights must have positive mass")
    if previous:
        previous = {uid: weight / previous_total for uid, weight in previous.items()}

    target = {uid: score_values[uid] / score_total for uid in eligible}
    if previous:
        lower = {uid: max(Decimal(0), previous.get(uid, Decimal(0)) - delta) for uid in eligible}
        upper = {uid: min(Decimal(1), previous.get(uid, Decimal(0)) + delta) for uid in eligible}
    else:
        lower = {uid: Decimal(0) for uid in eligible}
        upper = {uid: Decimal(1) for uid in eligible}

    projected = _box_simplex(target, lower, upper)
    quantized = _quantize_exact(projected, lower, upper, precision)

    all_uids = set(previous) | set(scores) | set(eligible_uids) | set(forced_zero_uids)
    weights = {uid: Decimal(0) for uid in all_uids}
    weights.update(quantized)
    if sum(weights.values()) != Decimal(1):
        raise RuntimeError("internal error: projected weights do not sum exactly to one")
    for uid in set(forced_zero_uids) | (all_uids - set(eligible)):
        if weights[uid] != 0:
            raise RuntimeError("internal error: forced/ineligible UID retained weight")
    if previous:
        quantum = Decimal(1) / (10 ** precision)
        for uid in eligible:
            if abs(weights[uid] - previous.get(uid, Decimal(0))) > delta + quantum:
                raise RuntimeError("internal error: ordinary UID exceeded max_delta")
    return WeightResult(weights=weights, precision=precision)
