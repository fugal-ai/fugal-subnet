"""Protocol v2 miss rule E'': reset before burn-in, decay without phantom zeros after.

docs/INVARIANTS.md, "I3 — the v2 miss rule (E'')".
"""
import pytest

from fugal_subnet.config import BURN_IN_QUESTIONS, EVIDENCE_HALF_LIFE
from fugal_subnet.evidence import (
    Evidence,
    accumulate_epoch,
    accumulate_epoch_v2,
    apply_miss,
    apply_miss_v2,
    burn_in_complete,
    decay_factor,
)

HL = EVIDENCE_HALF_LIFE


def grow(epochs, n=300, correct=240, ev=None, h="h"):
    for _ in range(epochs):
        ev = accumulate_epoch_v2(ev, h, correct, n, 0.3, 0.6, HL, pool_size=20000)
    return ev


def test_the_burn_in_counter_counts_scored_questions_undecayed():
    ev = grow(4)
    assert ev.burn_in_n == 1200.0
    assert ev.n_total < 1200.0          # n_total decays; the counter does not


def test_a_new_head_hash_restarts_the_counter():
    ev = grow(20)
    assert accumulate_epoch_v2(ev, "other", 240, 300, 0.3, 0.6, HL).burn_in_n == 300.0


def test_a_miss_before_burn_in_resets_the_evidence_fully():
    young = grow(BURN_IN_QUESTIONS // 300 - 1)
    assert not burn_in_complete(young, BURN_IN_QUESTIONS)
    after = apply_miss_v2(young, HL, BURN_IN_QUESTIONS)
    assert after.n_total == after.n_correct == after.cost_sum == 0.0
    assert after.burn_in_n == 0.0 and after.weights_hash == "h"
    assert after.epochs_missed == young.epochs_missed + 1


def test_a_miss_after_burn_in_decays_with_no_phantom_zeros_and_restarts_burn_in():
    mature = grow(BURN_IN_QUESTIONS // 300 + 5)
    assert burn_in_complete(mature, BURN_IN_QUESTIONS)
    a = decay_factor(HL)
    after = apply_miss_v2(mature, HL, BURN_IN_QUESTIONS)
    assert after.n_total == mature.n_total * a          # no +300 unanswered
    assert after.n_correct == mature.n_correct * a
    assert after.accuracy == pytest.approx(mature.accuracy, rel=1e-12)  # the record survives
    assert after.burn_in_n == 0.0 and not burn_in_complete(after, BURN_IN_QUESTIONS)


def test_v1_miss_is_unchanged():
    mature = grow(15)
    v1 = apply_miss(mature, 300, HL)
    assert v1.n_total == mature.n_total * decay_factor(HL) + 300


def test_burn_in_is_re_earned_after_a_miss():
    ev = apply_miss_v2(grow(15), HL, BURN_IN_QUESTIONS)
    ev = grow(BURN_IN_QUESTIONS // 300 - 1, ev=ev)
    assert not burn_in_complete(ev, BURN_IN_QUESTIONS)
    ev = grow(1, ev=ev)
    assert burn_in_complete(ev, BURN_IN_QUESTIONS)


def test_a_record_from_before_the_field_migrates_to_its_n_total():
    # v1 records carry no counter (-1); at the flip they keep the burn-in their
    # evidence already shows.
    old = accumulate_epoch(None, "h", 2400, 3000, 1.0, 2.0, HL)
    assert old.burn_in_n == -1.0
    assert old.burn_in_progress == old.n_total == 3000.0
    assert burn_in_complete(old, BURN_IN_QUESTIONS)
    assert accumulate_epoch_v2(old, "h", 240, 300, 0.1, 0.2, HL).burn_in_n == 3300.0


def test_a_state_file_without_the_field_loads():
    from neurons.validator import _known_fields

    raw = {"weights_hash": "h", "n_correct": 1.0, "n_total": 2.0}
    assert Evidence(**_known_fields(Evidence, raw)).burn_in_progress == 2.0


def _score():
    import numpy as np

    from fugal_subnet.head_eval import HeadScore

    return HeadScore(
        accuracy=0.8, cost_efficiency=0.0, kl_score=0.0,
        routing_decisions=np.array([], dtype=np.int32),
        correct_mask=np.ones(300, dtype=bool), n_correct=240, n_scored=300,
        total_head_cost=0.3, total_oracle_cost=0.6,
    )


def test_update_scores_applies_e2_only_under_v2():
    from fugal_subnet.scoring import ScoringState, update_scores
    from tests.frame_fixtures import frame_from_truth, frontier_of

    frontier = frontier_of(frame_from_truth(400))
    s = ScoringState()
    for _ in range(3):
        s = update_scores(s, {1: _score()}, {1: "h"}, frontier=frontier,
                          n_questions=300, protocol_version=2)
    young = s.records[1].evidence
    assert young.burn_in_n == 900.0
    v1 = update_scores(s, {}, {}, frontier=frontier, n_questions=300)
    v2 = update_scores(s, {}, {}, frontier=frontier, n_questions=300, protocol_version=2)
    assert v1.records[1].evidence.n_total > young.n_total          # phantom zeros
    assert v2.records[1].evidence.n_total == 0.0                   # young: reset


def test_v2_burn_in_is_its_own_1500_and_v1_keeps_3000():
    from fugal_subnet.config import BURN_IN_QUESTIONS as V1
    from fugal_subnet.config import V2_BURN_IN_QUESTIONS as V2
    assert (V1, V2) == (3000, 1500)


def test_update_scores_uses_the_v2_burn_in_in_v2_epochs():
    # Six epochs of 300 (1,800 questions) is past v2's 1,500 burn-in but short
    # of v1's 3,000: a v2 miss decays the record (mature), it does not reset.
    from fugal_subnet.scoring import ScoringState, update_scores
    from tests.frame_fixtures import frame_from_truth, frontier_of

    frontier = frontier_of(frame_from_truth(400))
    s = ScoringState()
    for _ in range(6):
        s = update_scores(s, {1: _score()}, {1: "h"}, frontier=frontier,
                          n_questions=300, protocol_version=2)
    before = s.records[1].evidence
    assert before.burn_in_n == 1800.0
    after = update_scores(s, {}, {}, frontier=frontier, n_questions=300,
                          protocol_version=2).records[1].evidence
    assert after.n_total > 0 and after.burn_in_n == 0.0


def test_a_miss_at_exactly_the_burn_in_count_decays_and_does_not_reset():
    """Burn-in is complete AT the count (burn_in_complete is >=), so the miss
    rule's 'before burn-in' must be strictly below it. Five epochs of 300 is
    exactly v2's 1,500: the normal state of a head five epochs in."""
    from fugal_subnet.config import V2_BURN_IN_QUESTIONS as V2

    exact = grow(V2 // 300)
    assert exact.burn_in_n == V2 and burn_in_complete(exact, V2)
    after = apply_miss_v2(exact, HL, V2)
    a = decay_factor(HL)
    assert after.n_total == exact.n_total * a and after.n_correct == exact.n_correct * a
    assert after.epochs_accumulated == exact.epochs_accumulated
    assert after.burn_in_n == 0.0
    # Control: one epoch short is before burn-in, and resets.
    short = apply_miss_v2(grow(V2 // 300 - 1), HL, V2)
    assert short.n_total == short.n_correct == 0.0 and short.epochs_accumulated == 0
