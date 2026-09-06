"""A constant policy — "always route to model k" — is a head that never reads
the question, and on the shipped constants it OUTSCORES a real router.

Measured on 5,000 held-out questions with the shipped trainer and the shipped
routing rule:

    always gpt-4o-mini      1.183   <- W=0 and a one-hot bias
    trained head (SFT)      1.058
    trained head (+CMA)     1.060
    per-benchmark lookup    1.036
    best single model       0.985   <- the reference
    random head             0.868
    always cheapest         0.479

The trained head is not the problem: it genuinely routes (13 models used,
within-source top share 0.45 where a lookup would be 1.00) and beats chance by
0.24. The problem is that not routing at all beats it by 12%, for free.

**scoring.py's stated defence does not cover this case.** It argues the
geometric mean stops degenerates because "neither axis can rescue the other" —
and that holds at the ENDS of the price range, which is where it was tested:
always-cheapest scores 0.479, always-best 0.985. The winner sits in the MIDDLE.
config.py derives w from "a router that gives up 40% of quality must not
outscore a quality match"; this one gives up 5% and wins. The constraint was
applied to the wrong corner of the space.

WHAT THIS FILE DOES, and does not do. It does not assert the incentive is
correct — it is not, and whether to fix it by exponent, thrift cap, or a
different reference is an open product decision, not a tuning detail. It pins
the break-even so that **nobody moves these constants without meeting this
analysis**, because the exploit is a property of the constants alone and can be
computed with no accuracy data at all.
"""
import pytest

from fugal_subnet.config import SCORE_QUALITY_EXPONENT, SCORE_THRIFT_CAP


def accuracy_ratio_needed_to_beat(target_score: float, thrift: float) -> float:
    """The accuracy (as a fraction of the reference's) a constant policy needs.

    Inverts `score = quality**w * thrift**(1-w)` for quality. Derived from the
    shipped constants only — no benchmark, no model, no measurement — which is
    why the exploit is not an artefact of the dataset it was found on.
    """
    return (target_score / thrift ** (1 - SCORE_QUALITY_EXPONENT)) ** (
        1 / SCORE_QUALITY_EXPONENT
    )


def test_a_capped_thrift_model_needs_far_less_accuracy_than_it_should():
    """THE FINDING, pinned. At the thrift cap a constant policy beats the
    reference on 77.4% of its accuracy, and beats a real trained router on
    82.3%. Both are low enough that ordinary cheap models clear them — the
    measured ratio for gpt-4o-mini was 95%, thirteen points above what the
    exploit needs.

    If this test fails, SCORE_QUALITY_EXPONENT or SCORE_THRIFT_CAP moved. That
    re-scores every miner AND changes how exploitable a constant policy is, so
    re-derive both before updating the numbers here.
    """
    beats_reference = accuracy_ratio_needed_to_beat(1.0, SCORE_THRIFT_CAP)
    beats_trained = accuracy_ratio_needed_to_beat(1.057, SCORE_THRIFT_CAP)

    assert beats_reference == pytest.approx(0.774, abs=0.001)
    assert beats_trained == pytest.approx(0.823, abs=0.001)


def test_the_exploit_needs_the_thrift_cap_to_be_reachable():
    """Why the cap is load-bearing rather than incidental.

    A constant policy's whole advantage is thrift, and thrift is capped at 10.
    gpt-4o-mini's raw cost ratio is 14.7, so the cap is ALREADY binding and
    still leaves the exploit profitable — which is why lowering the cap is not
    an obvious fix, and why this is recorded rather than quietly patched.
    """
    # Cheaper does not help past the cap: the score is identical.
    at_cap = accuracy_ratio_needed_to_beat(1.0, SCORE_THRIFT_CAP)
    way_past_cap = accuracy_ratio_needed_to_beat(1.0, SCORE_THRIFT_CAP)
    assert at_cap == way_past_cap

    # And a policy with no cost advantage cannot do it at all: it would need to
    # match the reference outright, which is what the design intends.
    no_advantage = accuracy_ratio_needed_to_beat(1.0, 1.0)
    assert no_advantage == pytest.approx(1.0, abs=1e-9)


def test_the_stated_defence_covers_only_the_ends_of_the_price_range():
    """scoring.py claims a product stops both degenerate strategies. It stops
    the two it NAMES. Always-cheapest fails because quality collapses; the
    reference itself scores below 1 only because of the Wilson penalty. Neither
    is the strategy that wins, and nothing in the argument reaches the middle.
    """
    # Always-cheapest: huge thrift, but quality near zero -> loses badly.
    assert accuracy_ratio_needed_to_beat(1.0, SCORE_THRIFT_CAP) < 1.0
    # The gap between "what the design constrained" (40% quality loss) and
    # "what actually wins" (5% quality loss) is the whole defect.
    design_constrained_at = 0.60
    exploit_lives_at = accuracy_ratio_needed_to_beat(1.057, SCORE_THRIFT_CAP)
    assert exploit_lives_at > design_constrained_at, (
        "the derivation constrained a 40% quality loss; the exploit needs only "
        f"a {100 * (1 - exploit_lives_at):.0f}% loss, which the derivation "
        "never covered")
