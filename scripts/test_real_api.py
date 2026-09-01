#!/usr/bin/env python3
"""B5 — Real API matrix test (proof of concept).

Calls 2 cheap models on 5 synthetic questions through OpenRouter.
Builds a real ground-truth matrix, grades responses, runs the full scoring
pipeline. Budget capped at $0.05.

Usage (never run without separate approval):
    # [PAID ~$0.05 maximum]
    OPENROUTER_API_KEY=sk-or-... python scripts/test_real_api.py \
        --live --epoch-budget 0.05

Expected cost: <$0.01 (2 cheap models × 5 questions × ~50 tokens each).
"""
import argparse
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("fugal.b5")

MODELS = [
    "deepseek/deepseek-v4-flash",
    "meta-llama/llama-4-maverick",
]

N_QUESTIONS = int(os.getenv("FUGAL_B5_QUESTIONS", "5"))


def make_questions(n: int) -> list:
    """Generate synthetic math questions with known gold answers."""
    import random
    rng = random.Random(42)
    questions = []
    for i in range(n):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            gold = a + b
        elif op == "-":
            gold = a - b
        else:
            gold = a * b
        questions.append({
            "prompt": f"What is {a} {op} {b}? Give only the number, nothing else.",
            "gold": str(gold),
            "grader_id": "numeric_final",
            "benchmark": "synthetic_math",
            "question_id": f"b5_math_{i:04d}",
            "metadata": {},
        })
    return questions


def parse_args():
    parser = argparse.ArgumentParser(description="Explicit paid OpenRouter canary")
    parser.add_argument("--live", action="store_true", help="Authorize the paid path")
    parser.add_argument("--epoch-budget", type=float, default=None,
                        help="Required positive USD ceiling (or FUGAL_B5_BUDGET)")
    return parser.parse_args()


def main():
    args = parse_args()
    budget_cap = args.epoch_budget
    if budget_cap is None and os.getenv("FUGAL_B5_BUDGET") is not None:
        try:
            budget_cap = float(os.environ["FUGAL_B5_BUDGET"])
        except ValueError:
            print("ERROR: FUGAL_B5_BUDGET must be a positive number")
            return 2
    if not args.live or budget_cap is None or budget_cap <= 0:
        print(
            "REFUSED: this paid canary requires --live and --epoch-budget AMOUNT "
            "(or a positive FUGAL_B5_BUDGET)."
        )
        return 2

    from fugal_subnet.api import (
        SpendTracker,
        build_spend_protection_prices,
        call_model,
        fetch_openrouter_prices,
        load_prices,
    )
    from fugal_subnet.epoch_logger import (
        EpochLog,
        EpochTimer,
        detect_anomalies,
        write_epoch_log,
    )
    from fugal_subnet.head_eval import HeadArtifact, evaluate_head
    from fugal_subnet.matrix import build_matrix
    from fugal_subnet.rewards import compute_weights
    from fugal_subnet.scoring import ScoringState, update_scores
    from fugal_subnet.soft_targets import compute_soft_targets

    timer = EpochTimer()

    try:
        canonical_prices = load_prices()
        live_prices = fetch_openrouter_prices()
        spend_prices = build_spend_protection_prices(
            canonical_prices, live_prices, MODELS,
        )
        logger.info("Loaded canonical and live price sheets")
    except Exception as e:
        logger.error("Price verification failed closed: %s", e)
        return 1

    tracker = SpendTracker(budget_cap_usd=budget_cap)

    print("=" * 60)
    print("B5 — REAL API MATRIX TEST (proof of concept)")
    print(f"Models: {MODELS}")
    print(f"Questions: {N_QUESTIONS}")
    print(f"Budget cap: ${budget_cap:.2f}")
    print("=" * 60)

    timer.start_phase("questions")
    questions = make_questions(N_QUESTIONS)
    print(f"\nGenerated {len(questions)} synthetic math questions")

    print("\n--- Step 1: Smoke test (1 call per model) ---")
    timer.start_phase("smoke_test")
    for model in MODELS:
        try:
            # [PAID ~$0-$0.05 total] Reservation enforcement prevents overshoot.
            text, ptok, ctok = call_model(
                model, "What is 2+2? Give only the number.",
                tracker=tracker, prices=spend_prices,
                max_tokens=32,
                live=True,
            )
            print(f"  {model}: '{text.strip()[:50]}' ({ptok}+{ctok} tokens, ${tracker.total_cost_usd:.6f} total)")
        except Exception as e:
            print(f"  {model}: FAILED — {e}")
            print("Aborting: model unavailable")
            return 1

    print(f"\nSmoke test cost: ${tracker.total_cost_usd:.6f}")

    print("\n--- Step 2: Build ground truth matrix ---")
    timer.start_phase("matrix")
    cache_dir = os.path.join("results", "b5_cache")
    matrix_result = build_matrix(
        questions, MODELS,
        tracker=tracker, prices=spend_prices,
        cache_dir=cache_dir,
        allow_exec=False,
        live=True,
    )
    print(f"  Matrix shape: {matrix_result.matrix.shape}")
    print(f"  Total cost: ${tracker.total_cost_usd:.6f}")
    print(f"  Total calls: {tracker.total_calls}")

    for m_idx, model in enumerate(MODELS):
        correct = matrix_result.matrix[:, m_idx].sum()
        print(f"  {model}: {correct}/{N_QUESTIONS} correct ({100*correct/N_QUESTIONS:.0f}%)")

    print("\n--- Step 3: Compute soft targets ---")
    timer.start_phase("soft_targets")
    soft = compute_soft_targets(matrix_result.matrix)
    print(f"  Soft targets shape: {soft.shape}")
    assert np.allclose(soft.sum(axis=1), 1.0), "Soft target rows don't sum to 1"
    print("  All rows sum to 1.0 OK")

    print("\n--- Step 4: Evaluate a synthetic head ---")
    timer.start_phase("head_eval")
    from fugal_subnet.config import HEAD_HIDDEN_DIM
    n_models = len(MODELS)
    rng = np.random.RandomState(42)
    W = rng.randn(n_models, HEAD_HIDDEN_DIM).astype(np.float32) * 0.01
    b = np.zeros(n_models, dtype=np.float32)

    hidden = rng.randn(N_QUESTIONS, HEAD_HIDDEN_DIM).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)

    model_costs = {}
    for m in MODELS:
        if m in canonical_prices:
            pin, pout = canonical_prices[m]
            model_costs[m] = pin * 500 + pout * 500
        else:
            model_costs[m] = 0.01

    head = HeadArtifact(
        W=W, b=b,
        models=MODELS,
        commit_hash="test_head",
    )

    score = evaluate_head(
        head, hidden, matrix_result.matrix,
        MODELS, soft, model_costs, lam=2.0,
    )
    print(f"  Accuracy: {score.accuracy:.3f}")
    print(f"  Cost efficiency: {score.cost_efficiency:.3f}")
    print(f"  KL score: {score.kl_score:.3f}")

    print("\n--- Step 5: Scoring pipeline ---")
    timer.start_phase("scoring")
    state = ScoringState()
    epoch_scores = {0: score}
    head_hashes = {0: "test_head_hash"}
    state = update_scores(state, epoch_scores, head_hashes)

    uids, weights = compute_weights(state.records)
    print(f"  UIDs: {uids}")
    print(f"  Weights: {[f'{w:.4f}' for w in weights]}")
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights don't sum to 1"
    print("  Weights sum to 1.0 OK")

    timer.end_phase()

    print("\n--- Step 6: Epoch log ---")
    epoch_score_dicts = {0: {"acc": score.accuracy, "cost_eff": score.cost_efficiency, "kl": score.kl_score}}
    epoch_weight_map = dict(zip(uids, weights))
    anomalies = detect_anomalies(epoch_score_dicts, epoch_weight_map, 1, 1)
    log = EpochLog(
        epoch_id="b5_real_api_test",
        block_hash="n/a",
        timestamp=time.time(),
        n_questions=N_QUESTIONS,
        n_miners_queried=1,
        n_heads_valid=1,
        n_heads_invalid=0,
        scores=epoch_score_dicts,
        weights=epoch_weight_map,
        anomalies=anomalies,
        duration_s=timer.total_s,
    )
    write_epoch_log(log)
    print(f"  Anomalies: {anomalies}")
    print(f"  Duration: {timer.total_s:.1f}s")

    print("\n" + "=" * 60)
    print("B5 REAL API TEST -- PASSED")
    print(f"Total cost: ${tracker.total_cost_usd:.6f}")
    print(f"Duration: {timer.total_s:.1f}s")
    print("=" * 60)

    print("\n--- Cost breakdown ---")
    for model in MODELS:
        model_cost = sum(e["cost_usd"] for e in tracker.per_call_log if e["model"] == model)
        model_calls = sum(1 for e in tracker.per_call_log if e["model"] == model)
        print(f"  {model}: {model_calls} calls, ${model_cost:.6f}")

    print(f"\n  Grand total: {tracker.total_calls} calls, ${tracker.total_cost_usd:.6f}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
