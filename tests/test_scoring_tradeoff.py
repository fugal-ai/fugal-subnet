"""The quality/cost tradeoff must hold across the range the caps permit.

`SCORE_QUALITY_EXPONENT` is derived from the widest cost ratio the thrift caps
allow, not from the ratio the product targets. That coupling is easy to break by
touching either constant alone, and the failure is invisible: scoring still
works, the numbers still look reasonable, and a cheap-and-wrong router quietly
starts winning. Two independent live runs produced exactly that before the
exponent was corrected — a 63% router beating a 93% one at 13x cheaper, and a
46% router beating a 62% one.
"""
import math

from fugal_subnet.config import (
    SCORE_QUALITY_CAP,
    SCORE_QUALITY_EXPONENT,
    SCORE_THRIFT_CAP,
)

# The product claim: a router that gives up this much quality has not delivered
# the product, however cheap it is.
DEGRADED_QUALITY = 0.6
# The floor thrift can reach: a miner routing to models this much more expensive
# than the reference. Symmetric with the cap, which is what makes the permitted
# cost ratio cap/floor.
THRIFT_FLOOR = 1.0 / SCORE_THRIFT_CAP


def composite(quality: float, thrift: float, w: float = SCORE_QUALITY_EXPONENT) -> float:
    q = min(max(quality, 0.0), SCORE_QUALITY_CAP)
    t = min(max(thrift, 0.0), SCORE_THRIFT_CAP)
    return (q ** w) * (t ** (1.0 - w))


def test_degraded_quality_never_wins_at_any_permitted_cost_ratio():
    """The whole point of the exponent. Checked at the widest ratio the caps
    allow, not at the ratio the product happens to target — the gap between
    those two is where the original derivation failed."""
    matched = composite(1.0, 1.0)
    max_ratio = SCORE_THRIFT_CAP / THRIFT_FLOOR
    for ratio in (2, 6, 7.7, 13, 20, 50, max_ratio):
        if ratio > max_ratio:
            continue
        cheap = composite(DEGRADED_QUALITY, THRIFT_FLOOR * ratio)
        assert cheap < matched, (
            f"a router at {DEGRADED_QUALITY:.0%} of best quality beats a full "
            f"quality match once it is {ratio}x cheaper "
            f"({cheap:.4f} >= {matched:.4f}). The exponent is derived from the "
            f"widest ratio the caps permit; if SCORE_THRIFT_CAP moved, "
            f"SCORE_QUALITY_EXPONENT must move with it."
        )


def test_exponent_satisfies_the_absolute_claim():
    """The documented claim, solved at the ratio the caps permit rather than
    the 6x the product targets. w=0.8 failed this at 1.0532 vs 1.000."""
    required = math.log(SCORE_THRIFT_CAP) / (
        math.log(SCORE_THRIFT_CAP) + math.log(1.0 / DEGRADED_QUALITY)
    )
    assert SCORE_QUALITY_EXPONENT >= required, (
        f"w={SCORE_QUALITY_EXPONENT} is below the {required:.4f} its own "
        f"derivation requires at the permitted thrift cap of {SCORE_THRIFT_CAP}x"
    )


def test_pairwise_bound_is_documented_as_a_knife_edge():
    """Weights come from comparing miners to each other, and a pairwise gap
    spans the whole band (cap^2), not half of it. w=0.9 sits fractionally BELOW
    that bound rather than above it — a deliberate, recorded choice. This test
    exists so that if the thrift cap is ever raised, the failure is loud."""
    band_ratio = SCORE_THRIFT_CAP / THRIFT_FLOOR
    required = math.log(band_ratio) / (
        math.log(band_ratio) + math.log(1.0 / DEGRADED_QUALITY)
    )
    assert abs(SCORE_QUALITY_EXPONENT - required) < 0.01, (
        f"w={SCORE_QUALITY_EXPONENT} is no longer near the pairwise bound "
        f"{required:.4f} for a band of {band_ratio:.0f}x. If SCORE_THRIFT_CAP "
        f"changed, re-derive the exponent — the two are coupled."
    )


def test_accuracy_still_outranks_cost_on_the_observed_live_case():
    """The case two live runs produced. 93% accuracy at $0.2777 must beat 63%
    at $0.0212 — the quality and thrift figures here are the ones the running
    validators actually computed."""
    accurate = composite(1.583, 0.200)
    cheap = composite(0.977, 2.576)
    assert accurate > cheap, (
        f"the 63% router still beats the 93% one ({cheap:.4f} vs {accurate:.4f})"
    )
