"""Scoring and weights in the validator: the per-proof score, the v2 copy-rule
snapshot, the v1 weights, the weight cap, and the routing vectors dedup reads.

These lived inline in neurons/validator.py's epoch loop, where 21 of 23
planted bugs survived the whole suite (2026-11-08), among them: a miner's own
prompt-token or cost report used for its score, exploration billed against
it, the v2 flip tie-break salted with a hash known in advance, dedup or
epsilon copies paid, the weight cap skipped. The scoring rules have their own
module tests; these check that the validator feeds them the right inputs.
"""
import ast
import pathlib
from types import SimpleNamespace

import numpy as np

from fugal_subnet import copy_rule, rewards
from fugal_subnet import dedup as dedup_mod
from fugal_subnet.config import FRAME_DEFAULT_COMPLETION_TOKENS
from fugal_subnet.pricing import policy_cost_total, question_input_tokens
from fugal_subnet.reference_frame import ReferenceFrame, reference_cost
from fugal_subnet.routing_protocol import price_table
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult
from neurons import validator
from neurons.validator import (
    _capped,
    _copy_rule_snapshot,
    _extract_routing_decisions,
    _score_proof,
    _v1_weights,
)

PRICES = price_table(1)
_BY_PRICE = sorted(PRICES, key=lambda m: sum(PRICES[m]))
CHEAP, DEAR = _BY_PRICE[0], _BY_PRICE[-1]
GOLD = {f"q{i}": {"question_id": f"q{i}", "prompt": "word " * (20 + 7 * i)} for i in range(6)}


def _result(qid, model, correct=True, explore=False, prompt_tokens=0, cost=0.0):
    return QuestionResult(question_id=qid, routed_model=model, correct=correct, cost_usd=cost,
                          response_hash="f" * 64, prompt_tokens=prompt_tokens,
                          completion_tokens=5, is_exploration=explore)


def _proof(results):
    return BenchmarkProof(
        hotkey="5H", epoch_id="e00000010", nonce="ab" * 32, questions_hash="c" * 64,
        weights_hash="d" * 64, source_hash="e" * 64, results=results,
        total_cost_usd=0.0, per_model_costs={}, attestation_quote=b"", timestamp=1.0,
    )


SCORED = [_result("q0", CHEAP), _result("q1", DEAR, correct=False), _result("q2", CHEAP)]


# ------------------------------------------------------------ the per-proof score --

def test_both_costs_come_from_the_pool_the_rates_and_the_frame():
    frame = ReferenceFrame()
    score = _score_proof(_proof(SCORED), GOLD, PRICES, frame, DEAR)
    prompt = sum(question_input_tokens(GOLD[q]["prompt"]) for q in ("q0", "q1", "q2"))
    assert score.total_oracle_cost == reference_cost(
        frame, PRICES, DEAR, prompt_tokens=prompt, n_questions=3,
        default_completion_tokens=FRAME_DEFAULT_COMPLETION_TOKENS)
    assert score.total_head_cost == policy_cost_total(
        [("q0", CHEAP), ("q1", DEAR), ("q2", CHEAP)], GOLD, PRICES, frame,
        FRAME_DEFAULT_COMPLETION_TOKENS)
    assert score.total_oracle_cost != score.total_head_cost
    assert (score.n_correct, score.n_scored) == (2, 3)


def test_what_a_miner_reports_about_tokens_and_cost_changes_nothing():
    honest = _score_proof(_proof(SCORED), GOLD, PRICES, ReferenceFrame(), DEAR)
    lying = _score_proof(_proof([
        _result(r.question_id, r.routed_model, r.correct, prompt_tokens=1, cost=1e-9)
        for r in SCORED]), GOLD, PRICES, ReferenceFrame(), DEAR)
    assert (lying.total_head_cost, lying.total_oracle_cost) == (
        honest.total_head_cost, honest.total_oracle_cost)


def test_exploration_is_not_billed_or_counted():
    plain = _score_proof(_proof(SCORED), GOLD, PRICES, ReferenceFrame(), DEAR)
    explored = _score_proof(_proof(SCORED + [_result("q4", DEAR, explore=True),
                                             _result("q5", DEAR, correct=False, explore=True)]),
                            GOLD, PRICES, ReferenceFrame(), DEAR)
    assert (explored.total_head_cost, explored.total_oracle_cost) == (
        plain.total_head_cost, plain.total_oracle_cost)
    assert (explored.n_scored, explored.n_correct, explored.accuracy) == (3, 2, 2 / 3)
    assert list(explored.correct_mask) == [True, False, True]


# ------------------------------------------------------- the v2 copy-rule snapshot --

def test_the_snapshot_is_every_committed_hotkey_with_an_owner():
    hotkeys = ["h0", "h1", "h2"]
    coldkeys = ["c0", "", "c2"]
    commitments = {"h2": ("C", 70), "h0": ("A", 100), "h1": ("B", 50), "gone": ("D", 10)}
    assert _copy_rule_snapshot(hotkeys, coldkeys, commitments) == [
        copy_rule.Commitment("h0", "c0", "A", 100),
        copy_rule.Commitment("h2", "c2", "C", 70),
    ]


def test_a_hotkey_past_the_coldkey_list_has_no_owner():
    assert _copy_rule_snapshot(["h0", "h1"], ["c0"], {"h1": ("B", 5)}) == []


# ------------------------------------------------------------------ v1 weights --

def test_v1_weights_disqualify_copies_and_burn_the_unconfident_share(monkeypatch):
    seen = {}
    monkeypatch.setattr(dedup_mod, "epsilon_disqualified", lambda rd, cb, scores: {2})

    def compute(records, dedup_disqualified=None, paid_fraction=1.0, *, burn_uid):
        seen.update(dq=dedup_disqualified, paid=paid_fraction, burn=burn_uid)
        return [0, 1], [0.4, 0.6]

    monkeypatch.setattr(rewards, "compute_weights", compute)
    uids, weights, dq = _v1_weights({}, {1: np.array([0])}, {1: 5}, {3}, 0.7, burn_uid=0)
    assert (uids, weights) == ([0, 1], [0.4, 0.6])
    assert seen == {"dq": {2, 3}, "paid": 0.7, "burn": 0} and dq == {2, 3}


def test_v1_weights_pass_the_epsilon_rule_this_epochs_composites(monkeypatch):
    seen = {}

    def eps(rd, cb, scores):
        seen.update(rd=rd, cb=cb, scores=scores)
        return set()

    monkeypatch.setattr(dedup_mod, "epsilon_disqualified", eps)
    monkeypatch.setattr(rewards, "compute_weights", lambda *a, **k: ([0], [1.0]))
    records = {1: SimpleNamespace(composite_score=0.3), 9: SimpleNamespace(composite_score=0.9)}
    rd, cb = {1: np.array([0]), 2: np.array([1])}, {1: 5, 2: 6}
    _v1_weights(records, rd, cb, set(), 1.0, burn_uid=0)
    assert seen["rd"] is rd and seen["cb"] is cb and seen["scores"] == {1: 0.3}


# ------------------------------------------------------------------ weight cap --

def test_the_first_epoch_is_not_capped():
    assert _capped([1, 2], [0.5, 0.5], [], []) == ([1, 2], [0.5, 0.5], False)


def test_later_epochs_are_capped_against_the_previous_weights():
    new = ([1, 2], [1.0, 0.0])
    prev = ([1, 2], [0.0, 1.0])
    uids, weights, capped = _capped(*new, *prev)
    assert capped and (uids, weights) == rewards.cap_weight_change(*new, *prev)
    assert weights != [1.0, 0.0]


# ------------------------------------------------------- routing vectors for dedup --

def test_routing_vectors_share_one_model_space_and_line_up_by_question():
    index = {"m/a": 0, "m/b": 1, "m/c": 2}
    a = _proof([_result("q2", "m/c"), _result("q0", "m/a"), _result("q1", "m/c")])
    b = _proof([_result("q0", "m/b"), _result("q1", "m/b"), _result("q2", "m/b"),
                _result("q9", "m/a", explore=True)])
    assert list(_extract_routing_decisions(a, index)) == [0, 2, 2]
    assert list(_extract_routing_decisions(b, index)) == [1, 1, 1]
    single_c = _proof([_result("q0", "m/c"), _result("q1", "m/c")])
    single_a = _proof([_result("q0", "m/a"), _result("q1", "m/a")])
    assert list(_extract_routing_decisions(single_c, index)) != list(
        _extract_routing_decisions(single_a, index))
    assert list(_extract_routing_decisions(_proof([_result("q0", "unknown")]), index)) == [-1]


# ----------------------------------------------- inline wiring, on the syntax tree --

def _main_calls(name):
    tree = ast.parse(pathlib.Path(validator.__file__).read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return [n for n in ast.walk(main) if isinstance(n, ast.Call)
            and getattr(n.func, "id", "") == name]


def _kw(call):
    return {k.arg: ast.unparse(k.value) for k in call.keywords}


def test_the_score_and_the_dedup_read_this_epochs_inputs():
    (score,) = _main_calls("_score_proof")
    assert [ast.unparse(a) for a in score.args] == [
        "proof", "slice_gold", "prices", "epoch_frame", "ref_model"]
    (dupes,) = _main_calls("find_duplicates")
    assert _kw(dupes) == {"head_params": "head_params"}
    (v1,) = _main_calls("_v1_weights")
    assert [ast.unparse(a) for a in v1.args] == [
        "epoch_state.records", "routing_decisions", "commit_blocks", "dupes",
        "frontier.confidence", "burn_uid"]
    (cap,) = _main_calls("_capped")
    assert [ast.unparse(a) for a in cap.args] == ["uids", "weights", "prev_uids", "prev_weights"]


def test_the_evidence_update_gets_this_epochs_shape_and_version():
    (call,) = _main_calls("update_scores")
    kw = _kw(call)
    assert kw["frontier"] == "frontier"
    assert kw["n_questions"] == "len(questions)" and kw["pool_size"] == "len(benchmark_pool)"
    assert kw["protocol_version"] == "protocol_version"
    assert kw["hotkeys"].startswith("{uid: metagraph.hotkeys[uid] for uid in verified_proofs")


def test_the_v2_epoch_gets_the_flip_tie_break_and_the_boundary_snapshot():
    (call,) = _main_calls("run_v2_epoch")
    kw = _kw(call)
    assert kw["seniority_floor"] == "V2_ACTIVATION_BLOCK"
    # Salted with the activation boundary's hash, which nobody knew while the
    # pre-flip commitments were being made.
    assert kw["seniority_salt"] == (
        "_activation_block_hash(subtensor, V2_ACTIVATION_BLOCK, boundary_block, block_hash)")
    assert kw["allow_take"] == "not v2_window_degraded"
    assert kw["coldkey_of"] == "coldkey_of" and kw["commitments"] == "snapshot"
    assert kw["window"] == "v2_window" and kw["seat"] == "v2_seat"
    (snap,) = _main_calls("_copy_rule_snapshot")
    assert [ast.unparse(a) for a in snap.args] == [
        "metagraph.hotkeys", "coldkeys", "chain_commitments"]
