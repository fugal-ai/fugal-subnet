"""Protocol v4's label budget and join delay (docs/decisions/SEAT_MINIMAL_V4.md,
sections 4.6, 5 and 8; package S6).

The sizes the design fixes: 112 exploration calls a hotkey-epoch (68
cost-aware, 44 coverage) and no shared set, the spend caps re-derived for
them, the operator cap at 1.5 private sets, 200 scored questions on a mix
rebuilt for them, and a proposal measured over two epochs of 150 questions
instead of five of 60, so a filed entry joins three epochs sooner. The
detailed checks of each mechanism are in test_explore_v4.py,
test_explore_pool_v4.py, test_pool_mix_v4.py and test_menu_step.py; these
pin the sizes and the join timing end to end.
"""
from fractions import Fraction

from fugal_subnet import config, epoch_plan, explore_pool
from fugal_subnet import menu_state as ms
from fugal_subnet.tee import td_v4
from tests import menu_fixtures as fx
from tests.test_menu_step import run


def test_the_exploration_sizes_and_caps_are_section_8s():
    assert config.EXPLORE_SHARED_SIZE_V4 == 0
    assert (config.EXPLORE_COST_AWARE_SIZE_V4, config.EXPLORE_COVERAGE_SIZE_V4) == (68, 44)
    assert config.EXPLORE_PRIVATE_SIZE_V4 == 112
    assert Fraction(config.EXPLORE_SPEND_CAP_USD_V4) == Fraction("0.55")
    assert Fraction(config.EXPLORE_COVERAGE_SPEND_CAP_USD_V4) == Fraction("1.10")
    assert config.EXPLORE_COLDKEY_CAP_V4 == explore_pool.coldkey_cap(4) == 168
    assert config.EXPLORE_COVERAGE_TARGET_V4 == 100
    assert config.EXPLORE_COVERAGE_DAMP_PAIRS_V4 == 1400
    # v3's are untouched: its shared set stays, and v3 still reads it.
    assert (config.EXPLORE_SHARED_SIZE, config.EXPLORE_PRIVATE_SIZE,
            config.EXPLORE_COLDKEY_CAP) == (8, 24, 36)


def test_the_scored_slice_is_200_on_a_mix_built_for_200():
    assert config.SLICE_SIZE_V4 == 200 == epoch_plan.mix_v4().slice_size
    assert config.SLICE_SIZE == 300                          # v1-v3's slice


def test_a_proposal_is_measured_by_the_field_in_its_join_epoch():
    """No proposer measures (FAST_CROWN_V4.md 18): a new entry joins the
    boundary after its filing as provisional, every box's coverage is aimed
    at it there, and the step that scores that epoch decides it."""
    for name in ("PROPOSAL_MEASURE_EPOCHS", "PROPOSAL_MEASURE_PAIRS", "PROPOSAL_MAKEUP_EPOCHS",
                 "QUEUE_WAIT_EPOCHS", "MENU_MAX_PENDING", "ENTRY_ALLOWANCE_EPOCHS"):
        assert not hasattr(ms, name), name
    assert not hasattr(td_v4, "MEASURE_CALLS_PER_EPOCH")
    assert (ms.MENU_MAX_JOINS_PER_EPOCH, ms.MENU_JOINS_SMALL_FIELD, ms.MENU_JOINS_FIELD_HOTKEYS) \
        == (2, 1, 10)


def test_an_entry_filed_in_epoch_k_is_on_the_menu_of_k_plus_1_and_full_at_k_plus_3():
    """Filed (committed) in k = 100 before the root's pinned read, on the menu
    of 101 as provisional, decided by the step that scores 101, full from 103."""
    eid = "lab-c/model-new@high"
    joined = fx.provisional(fx.state(epoch=101))              # the boundary of 101's delta
    e = joined.entry(eid)
    assert e.provisional and e.joined_epoch == 101
    s = run(joined).state                                   # scored 100: it joined after it
    assert s.epoch == 102 and s.entry(eid).provisional
    out = ms.step(s, fx.inputs(epoch=101, gates=(fx.gate(eid),)), fx.PARAMS)
    assert out.epoch == 103 and out.entry(eid).scored_from == 103
