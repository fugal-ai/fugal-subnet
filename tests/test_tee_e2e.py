"""End-to-end TEE pipeline: miner harness -> proof -> verify -> score -> weights.

This test exists because three showstopper bugs shipped undetected: the miner
and validator derived different epoch ids (so every proof failed the nonce
check), the miner imported a backbone function that does not exist (so every
epoch died in an `except` that only logged), and the harness passed a raw
loader dict to a grader that needs a translated task dict (so every answer
graded zero). None of them was visible to the existing suite, because CI
exercised the pre-TEE code path end to end and the TEE path only in pieces.

So this drives the real objects: the real harness, the real proof, the real
attestation binding, the real verifier, the real reference frame, the real
scoring and weight computation. Only two things are stubbed — the model call
(no network, no spend) and the chain — because everything else is what broke.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest

from fugal_subnet.benchmarks.slicer import (
    derive_nonce,
    epoch_id_for_block,
    epoch_index_for_block,
    select_slice,
)
from fugal_subnet.config import HEAD_HIDDEN_DIM
from fugal_subnet.exploration import expected_exploration
from fugal_subnet.reference_frame import (
    ReferenceFrame,
    accumulate_exploration,
    best_model,
    reference_cost,
)
from fugal_subnet.rewards import compute_weights
from fugal_subnet.scoring import ScoringState, update_scores
from fugal_subnet.tee import harness as harness_mod
from fugal_subnet.tee.proof import compute_questions_hash
from fugal_subnet.tee.runtime import APICallRecord, MeteringProxy, TEERuntime
from fugal_subnet.tee.verify import verify_proof

SLICE = 12
EXPLORE = 4
MODELS = ["a/cheap", "b/mid", "c/expensive"]
PRICES = {"a/cheap": (1e-7, 2e-7), "b/mid": (1e-6, 2e-6), "c/expensive": (5e-6, 3e-5)}
# Ground truth: pricier models answer more questions correctly.
SKILL = {"a/cheap": 0.3, "b/mid": 0.6, "c/expensive": 0.9}


def _pool(n=200):
    return [
        {
            "question_id": f"q{i}",
            "prompt": f"Question {i}?",
            "gold": str(i),
            "grader_id": "numeric_final",
            "benchmark": ["gsm8k", "math", "mmlu", "aime"][i % 4],
            "metadata": {},
        }
        for i in range(n)
    ]


def _head(models, seed=0):
    """A real .npz head artifact, as a miner would publish it."""
    rng = np.random.RandomState(seed)
    W = (rng.randn(len(models), HEAD_HIDDEN_DIM) * 0.01).astype(np.float32)
    b = rng.randn(len(models)).astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, W=W, b=b, models=np.array(models, dtype="U100"))
    return buf.getvalue()


class _StubProxy(MeteringProxy):
    """Records calls exactly as the real proxy does — never touches a network."""

    def start(self):
        self.prices = PRICES


def _stub_call(proxy, model_id, question):
    """Deterministic stand-in for a model reply, priced through the proxy."""
    qid = question["question_id"]
    digest = hashlib.sha256(f"{qid}|{model_id}".encode()).digest()
    correct = (digest[0] / 255.0) < SKILL[model_id]
    text = question["gold"] if correct else "0"
    p_tok, c_tok = 500, 300
    proxy.records.append(APICallRecord(
        model_id=model_id, prompt_tokens=p_tok, completion_tokens=c_tok,
        cost_usd=proxy.price_call(model_id, p_tok, c_tok),
        timestamp=0.0, response_hash=hashlib.sha256(text.encode()).hexdigest(),
    ))
    return text


@pytest.fixture
def stubbed_model_call(monkeypatch):
    monkeypatch.setattr(harness_mod, "_call_model", _stub_call)


def _run_miner_epoch(head_bytes, nonce, pool, hidden, epoch_id):
    proxy = _StubProxy(port=0)
    proxy.start()
    proof = harness_mod.run_benchmark(
        nonce=nonce, head_bytes=head_bytes, benchmark_pool=pool, proxy=proxy,
        hidden_states=hidden, slice_size=SLICE, epoch_id=epoch_id,
        source_hash="test-image", explore_models=MODELS, explore_size=EXPLORE,
    )
    proof.attestation_quote = TEERuntime(mock=True).generate_attestation(
        bytes.fromhex(proof.content_hash())
    )
    return proof


def test_full_epoch_miner_to_weights(stubbed_model_call):
    """One complete epoch, every real component in the path."""
    pool = _pool()
    hidden = np.random.RandomState(0).randn(len(pool), HEAD_HIDDEN_DIM).astype(np.float32)

    # --- both sides derive epoch identity independently ---
    block, blocks_per_epoch = 37_123, 300
    epoch_index = epoch_index_for_block(block, blocks_per_epoch)
    epoch_id = epoch_id_for_block(epoch_index)
    block_hash = "0x" + "ab" * 32
    nonce = derive_nonce(epoch_id, block_hash)

    questions = select_slice(nonce, pool, SLICE)
    question_ids = [q["question_id"] for q in questions]
    explore_map = expected_exploration(
        nonce, pool, set(question_ids), MODELS, EXPLORE,
    )

    # --- miner ---
    head_bytes = _head(MODELS)
    proof = _run_miner_epoch(head_bytes, nonce.hex(), pool, hidden, epoch_id)

    # The three showstoppers, each as a direct assertion.
    assert proof.nonce == nonce.hex(), "miner and validator disagree on the nonce"
    assert proof.n_total == SLICE, "miner did not answer the assigned slice"
    assert proof.n_correct > 0, "every answer graded zero — grader task translation"

    # --- validator ---
    gold = {q["question_id"]: q for q in pool}
    result = verify_proof(
        proof,
        approved_measurements=set(),
        expected_questions_hash=compute_questions_hash(question_ids),
        expected_nonce=nonce.hex(),
        gold_answers=gold,
        expected_question_ids=set(question_ids),
        expected_exploration=explore_map,
        expected_weights_hash=hashlib.sha256(head_bytes).hexdigest(),
        expected_proof_hash=proof.content_hash(),
        head_bytes=head_bytes,
        mock=True,
    )
    assert result.valid, result.reason

    # --- frame, scoring, weights ---
    frame = accumulate_exploration(ReferenceFrame(), [
        (r.routed_model, r.correct, r.prompt_tokens, r.completion_tokens)
        for r in proof.exploration_results
    ])
    ref_model, acc_best = best_model(frame, PRICES)
    scored = proof.scored_results
    ref = reference_cost(
        frame, PRICES, ref_model,
        prompt_tokens=sum(r.prompt_tokens for r in scored),
        n_questions=len(scored), default_completion_tokens=300.0,
    )

    from fugal_subnet.head_eval import HeadScore
    hs = HeadScore(
        accuracy=proof.accuracy, cost_efficiency=0.0, kl_score=0.0,
        routing_decisions=np.array([], dtype=np.int32),
        correct_mask=np.array([r.correct for r in scored], dtype=bool),
        n_correct=proof.n_correct, n_scored=proof.n_total,
        total_head_cost=proof.scored_cost_usd, total_oracle_cost=ref,
    )
    state = update_scores(
        ScoringState(), {1: hs}, {1: proof.weights_hash},
        acc_best=acc_best, hotkeys={1: "hk1"},
        n_questions=SLICE, pool_size=len(pool),
    )
    uids, weights = compute_weights(state.records)

    assert state.records[1].composite_score > 0
    assert abs(sum(weights) - 1.0) < 1e-9
    assert 1 in uids and weights[uids.index(1)] > 0, "miner earned no weight"


def test_exploration_is_actually_performed(stubbed_model_call):
    """The forced routes must appear, be flagged, and be excluded from scoring."""
    pool = _pool()
    hidden = np.random.RandomState(0).randn(len(pool), HEAD_HIDDEN_DIM).astype(np.float32)
    nonce = derive_nonce(epoch_id_for_block(1), "0x" + "cd" * 32)
    questions = select_slice(nonce, pool, SLICE)
    explore_map = expected_exploration(
        nonce, pool, {q["question_id"] for q in questions}, MODELS, EXPLORE,
    )

    proof = _run_miner_epoch(_head(MODELS), nonce.hex(), pool, hidden, "e1")

    assert len(proof.exploration_results) == EXPLORE
    assert len(proof.scored_results) == SLICE
    assert proof.n_total == SLICE, "exploration must not inflate the scored count"

    for r in proof.exploration_results:
        assert r.routed_model == explore_map[r.question_id], (
            "explored a model the nonce did not assign"
        )
    # Exploration cost is excluded from what the miner is judged on.
    assert proof.scored_cost_usd < proof.total_cost_usd


def test_skipping_exploration_is_rejected(stubbed_model_call):
    """What makes the ~5% non-optional rather than a saving."""
    pool = _pool()
    hidden = np.random.RandomState(0).randn(len(pool), HEAD_HIDDEN_DIM).astype(np.float32)
    nonce = derive_nonce(epoch_id_for_block(1), "0x" + "cd" * 32)
    questions = select_slice(nonce, pool, SLICE)
    ids = [q["question_id"] for q in questions]
    explore_map = expected_exploration(nonce, pool, set(ids), MODELS, EXPLORE)

    # A miner that runs the benchmark but skips the exploration quota.
    proxy = _StubProxy(port=0)
    proxy.start()
    proof = harness_mod.run_benchmark(
        nonce=nonce.hex(), head_bytes=_head(MODELS), benchmark_pool=pool,
        proxy=proxy, hidden_states=hidden, slice_size=SLICE, epoch_id="e1",
        source_hash="test-image", explore_models=MODELS, explore_size=0,
    )
    proof.attestation_quote = TEERuntime(mock=True).generate_attestation(
        bytes.fromhex(proof.content_hash())
    )

    result = verify_proof(
        proof, approved_measurements=set(),
        expected_questions_hash=compute_questions_hash(ids),
        expected_nonce=nonce.hex(),
        gold_answers={q["question_id"]: q for q in pool},
        expected_question_ids=set(ids),
        expected_exploration=explore_map,
        mock=True,
    )
    assert not result.valid
    assert "Exploration set mismatch" in result.reason


def test_two_validators_agree_on_the_same_proof(stubbed_model_call):
    """I1 for the live path: verification and scoring are pure functions."""
    pool = _pool()
    hidden = np.random.RandomState(0).randn(len(pool), HEAD_HIDDEN_DIM).astype(np.float32)
    nonce = derive_nonce(epoch_id_for_block(7), "0x" + "ef" * 32)
    proof = _run_miner_epoch(_head(MODELS), nonce.hex(), pool, hidden, "e7")

    frames = []
    for _ in range(2):
        f = accumulate_exploration(ReferenceFrame(), [
            (r.routed_model, r.correct, r.prompt_tokens, r.completion_tokens)
            for r in proof.exploration_results
        ])
        frames.append(f)
    assert frames[0].to_dict() == frames[1].to_dict()
    assert best_model(frames[0], PRICES) == best_model(frames[1], PRICES)


def test_a_cheaper_router_of_equal_quality_scores_higher(stubbed_model_call):
    """The incentive the whole subnet exists to create.

    Two miners answer the same questions equally well; one does it for less.
    """
    from fugal_subnet.evidence import Evidence
    from fugal_subnet.scoring import composite

    expensive = Evidence("h1", n_correct=9000.0, n_total=10000.0,
                         cost_sum=6.0, ref_cost_sum=6.0, pool_size=1e9)
    frugal = Evidence("h2", n_correct=9000.0, n_total=10000.0,
                      cost_sum=1.0, ref_cost_sum=6.0, pool_size=1e9)
    assert composite(frugal, 0.9) > composite(expensive, 0.9)


def test_neuron_logging_survives_importing_bittensor():
    """The neurons' own logging must not be silently disabled.

    Importing bittensor runs a dictConfig with disable_existing_loggers at its
    default of True, which sets .disabled on every logger created before it —
    including the module-level loggers in this package. The failure is total
    and silent: a miner failing every epoch logs nothing at all and looks
    identical to an idle one, and the traceback explaining why is discarded.

    This is asserted rather than assumed because it is invisible by
    construction: the symptom of a broken logger is the absence of output.
    """
    import logging

    import bittensor  # noqa: F401  — the import is what does the damage

    from fugal_subnet.logging_setup import configure_logging

    # Simulate a logger created before bittensor was imported.
    lg = logging.getLogger("fugal.regression_probe")
    lg.disabled = True

    configure_logging("INFO", root_name="fugal")

    assert not lg.disabled, (
        "configure_logging() did not re-enable a fugal logger — the neurons' "
        "own output, including every error, would be discarded"
    )
    assert lg.isEnabledFor(logging.INFO)

    records = []
    handler = logging.Handler()
    handler.emit = records.append
    lg.addHandler(handler)
    try:
        lg.info("probe")
        lg.error("probe-error")
    finally:
        lg.removeHandler(handler)
    assert len(records) == 2, f"logger emitted {len(records)}/2 records"
