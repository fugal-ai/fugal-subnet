"""v1 scoring rules that a planted-bug review could change without any test noticing.

Each test is an honest control plus one input that differs in the one way the
rule exists for, so the rule is pinned by behaviour, not by its source line:

  * the frontier's confidence (the v1 paid fraction) reads the LEAST-observed
    model on the hull, not the best-observed one;
  * an exact 2-2 split among four coldkeys is no consensus, so nobody on either
    side is flagged for disagreeing with it;
  * a UID re-registered by a new hotkey does not inherit the old occupant's
    record, whether or not the new occupant answers this epoch;
  * in a dedup cluster, a uid with no commit block is junior to every uid that
    has one;
  * the burn UID is never paid as a miner, even if it has a scoring record.

The last section checks that the grader attack suite's exit/stdout forgery cases
test what they claim: each exits 0 under exec_io's driver, so the block comes
from the parent's output comparison, not from a crash.
"""
import json

import numpy as np
import pytest

from fugal_subnet import graders as g
from fugal_subnet.attacks.run_attacks import ATTACKS
from fugal_subnet.config import FRONTIER_MIN_TRIALS
from fugal_subnet.dedup import find_duplicates
from fugal_subnet.evidence import Evidence
from fugal_subnet.frontier import Frontier, build_frontier
from fugal_subnet.head_eval import HeadScore
from fugal_subnet.reference_frame import (
    ReferenceFrame,
    accumulate_exploration,
    disagreeing_exploration,
)
from fugal_subnet.rewards import compute_weights
from fugal_subnet.scoring import MinerRecord, ScoringState, update_scores

# ---------------------------------------------------------------------------
# Frontier confidence: the least-observed hull model sets it.
# ---------------------------------------------------------------------------

PRICES = {"m/cheap": (1e-7, 2e-7), "m/dud": (2e-6, 4e-6), "m/dear": (5e-6, 1e-5)}
PROMPT_TOKENS, N_Q, COMPLETION = 100 * 300, 300, 300.0


def _frame(counts: dict[str, tuple[int, int]]) -> ReferenceFrame:
    """One epoch of exploration: {model: (n_correct, n_trials)}, exact means."""
    samples = []
    for model, (k, n) in counts.items():
        samples += [(model, i < k, 100, int(COMPLETION)) for i in range(n)]
    return accumulate_exploration(ReferenceFrame(), samples)


def _frontier(counts):
    return build_frontier(_frame(counts), PRICES, PROMPT_TOKENS, N_Q, COMPLETION)


def test_frontier_confidence_with_equal_trials_is_that_count():
    f = _frontier({"m/cheap": (150, 300), "m/dud": (60, 300), "m/dear": (270, 300)})
    assert f.models == ("m/cheap", "m/dear")
    assert f.least_trials == pytest.approx(300.0)
    assert f.confidence == pytest.approx(min(1.0, 300.0 / FRONTIER_MIN_TRIALS))


def test_frontier_confidence_reads_the_least_observed_hull_model():
    """Unequal trial counts on the hull: the frontier is only as well known as
    its least-measured point. The dud (fewest trials of all) is off the hull and
    must not count either way."""
    assert FRONTIER_MIN_TRIALS > 60
    f = _frontier({"m/cheap": (30, 60), "m/dud": (1, 5), "m/dear": (900, 1000)})
    assert f.models == ("m/cheap", "m/dear")
    assert f.least_trials == pytest.approx(60.0)
    assert f.confidence == pytest.approx(60.0 / FRONTIER_MIN_TRIALS)

    # The v1 paid fraction is this confidence (neurons/validator.py passes
    # frontier.confidence to compute_weights); the rest burns.
    rec = MinerRecord(uid=7, composite_score=0.1)
    uids, weights = compute_weights({7: rec}, paid_fraction=f.confidence, burn_uid=0)
    w = dict(zip(uids, weights))
    assert w[7] == pytest.approx(60.0 / FRONTIER_MIN_TRIALS)
    assert w[0] == pytest.approx(1.0 - 60.0 / FRONTIER_MIN_TRIALS)


# ---------------------------------------------------------------------------
# Disagreement detector: an exact tie is no consensus.
# ---------------------------------------------------------------------------

_DETECT = {"min_voters": 3, "min_samples": 8, "max_disagreement": 0.4}


def _answers(correct: bool, n: int = 8):
    return [(f"q{i}", "m/a", correct) for i in range(n)]


def test_disagreement_three_to_one_flags_the_dissenter():
    per_miner = {1: _answers(True), 2: _answers(True), 3: _answers(True), 4: _answers(False)}
    coldkeys = {u: f"ck{u}" for u in per_miner}
    assert set(disagreeing_exploration(per_miner, coldkeys, **_DETECT)) == {4}


def test_disagreement_two_two_split_is_no_consensus():
    """Four operators, two each way, on every question. Neither side is the
    majority, so nobody is flagged; reading the tie as a consensus of
    'incorrect' would drop both honest 'correct' answerers from the frame."""
    per_miner = {1: _answers(True), 2: _answers(True), 3: _answers(False), 4: _answers(False)}
    coldkeys = {u: f"ck{u}" for u in per_miner}
    assert disagreeing_exploration(per_miner, coldkeys, **_DETECT) == {}


# ---------------------------------------------------------------------------
# Recycled UID: a new hotkey on the same UID starts from nothing.
# ---------------------------------------------------------------------------

_FRONTIER = Frontier(points=((0.0, 0.0), (1e-3, 0.5)), least_trials=1e9, models=("m/a",))


def _score(n=100, k=80):
    return HeadScore(accuracy=k / n, cost_efficiency=1.0, kl_score=0.0,
                     routing_decisions=np.zeros(n, dtype=np.int64),
                     correct_mask=np.zeros(n, dtype=bool),
                     n_correct=k, n_scored=n, total_head_cost=1e-4 * n,
                     total_oracle_cost=1e-4 * n)


def _occupied_state(hotkey="hk-old", head="h-same"):
    ev = Evidence(weights_hash=head, n_correct=4000.0, n_total=5000.0, cost_sum=0.5,
                  epochs_accumulated=10, pool_size=1e9, n_priced=5000.0)
    rec = MinerRecord(uid=5, hotkey=hotkey, epochs_seen=10, current_head_hash=head,
                      evidence=ev)
    return ScoringState(records={5: rec}, epoch_count=10)


def test_same_hotkey_keeps_its_record():
    out = update_scores(_occupied_state(), {5: _score()}, {5: "h-same"}, _FRONTIER,
                        hotkeys={5: "hk-old"}, n_questions=100)
    assert out.records[5].epochs_seen == 11
    assert out.records[5].evidence.n_total > 5000.0 * 0.5


def test_new_hotkey_on_the_uid_does_not_inherit_the_record():
    """Same UID, same head hash (a copied head), but a different hotkey in the
    metagraph: the old occupant's evidence and history must not carry over."""
    out = update_scores(_occupied_state(), {5: _score()}, {5: "h-same"}, _FRONTIER,
                        hotkeys={5: "hk-new"}, n_questions=100)
    rec = out.records[5]
    assert rec.hotkey == "hk-new"
    assert rec.epochs_seen == 1
    assert rec.evidence.n_total == pytest.approx(100.0)
    assert rec.evidence.epochs_accumulated == 1


def test_new_hotkey_that_has_not_answered_yet_drops_the_old_record():
    out = update_scores(_occupied_state(), {}, {}, _FRONTIER,
                        hotkeys={5: "hk-new"}, n_questions=100)
    assert 5 not in out.records


# ---------------------------------------------------------------------------
# Dedup seniority: a missing commit block is junior to all.
# ---------------------------------------------------------------------------

_SAME = np.array([0, 1, 2, 3] * 10)


def test_dedup_lower_commit_block_survives():
    assert find_duplicates({3: _SAME, 5: _SAME}, {3: 100, 5: 200}) == {5}
    assert find_duplicates({3: _SAME, 5: _SAME}, {3: 200, 5: 100}) == {3}


def test_dedup_uid_without_a_commit_block_is_junior():
    """uid 3 sorts first by uid but has no commit block on record: it is the
    copy, and uid 5 (block 100) survives."""
    assert find_duplicates({3: _SAME, 5: _SAME}, {5: 100}) == {3}


# ---------------------------------------------------------------------------
# Burn UID: never paid as a miner.
# ---------------------------------------------------------------------------

def test_burn_uid_absent_from_records_gets_only_the_unpaid_share():
    recs = {7: MinerRecord(uid=7, composite_score=0.5)}
    uids, weights = compute_weights(recs, paid_fraction=1.0, burn_uid=0)
    assert dict(zip(uids, weights)) == {7: pytest.approx(1.0)}


def test_burn_uid_with_a_record_is_not_paid_as_a_miner():
    """The owner's UID holds a scoring record (it was registered and served a
    head). It must take no miner share: at paid_fraction 1 the other miner gets
    everything, and at 0.5 the burn gets exactly the unpaid half."""
    recs = {0: MinerRecord(uid=0, composite_score=0.5),
            7: MinerRecord(uid=7, composite_score=0.5)}
    uids, weights = compute_weights(recs, paid_fraction=1.0, burn_uid=0)
    assert dict(zip(uids, weights)) == {7: pytest.approx(1.0)}
    uids, weights = compute_weights(recs, paid_fraction=0.5, burn_uid=0)
    assert dict(zip(uids, weights)) == {0: pytest.approx(0.5), 7: pytest.approx(0.5)}


# ---------------------------------------------------------------------------
# Grader attack suite: the forgery cases are blocked for the right reason.
# ---------------------------------------------------------------------------

_FORGERIES = (
    "exec_io: sys.exit(0) early-exit forgery",
    "exec_io: os._exit(0) forgery",
    "exec_io: print sabotage (write to stdout not the outfile)",
)


@pytest.mark.parametrize("name", _FORGERIES)
def test_exec_forgery_exits_zero_yet_grades_zero(name):
    """Each forgery defines a wrong `f` and exits 0 under exec_io's own driver
    (minus the outfile), so a grader that trusted the exit code would pass it.
    A case that crashed (NameError, exit 1) would be blocked without testing
    the parent-side output comparison at all."""
    (case,) = [a for a in ATTACKS if a[0] == name]
    _, task, reply, allow, expected, verdict, _ = case
    params = task["checker"]["params"]
    driver = (f"\n\nif __name__ == '__main__':\n"
              f"    import json as _json\n"
              f"    _inputs = _json.loads({json.dumps(json.dumps(params['inputs']))})\n"
              f"    _res = [{params['func']}(*_a) for _a in _inputs]\n")
    assert g._run_sandboxed(g.extract_code(reply) + driver) == 0
    assert (expected, verdict) == (0, "BLOCKED")
    assert g.grade(task, reply, allow) == 0
