#!/usr/bin/env python3
"""Fugal subnet validator — epoch loop, head evaluation, weight-setting.

Usage:
    python neurons/validator.py --netuid 1 --coldkey fugal_validator --hotkey default
    python neurons/validator.py --netuid 1 --coldkey fugal_validator --hotkey default --once

Epochs are aligned to chain blocks: every EPOCH_INTERVAL/12 blocks is an epoch
boundary, and the boundary block's hash seeds the question slice. All honest
validators therefore score the same slice for the same epoch, and a head is
only scoreable if its sha256 was committed on-chain at or before the boundary.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import logging
import math
import os
import sys
import threading
import time

import click
import numpy as np

from fugal_subnet.config import HEARTBEAT_TIMEOUT

logger = logging.getLogger("fugal.validator")

STATE_PATH = os.getenv("FUGAL_STATE_PATH", "results/validator_state.json")
RESPONSE_CACHE_DIR = os.getenv("FUGAL_RESPONSE_CACHE", "results/response_cache")
BLOCK_TIME_S = 12


def heartbeat_monitor(last_heartbeat: list[float], stop_event: threading.Event):
    """Re-exec the process if the main loop stops checking in."""
    while not stop_event.is_set():
        time.sleep(5)
        if time.time() - last_heartbeat[0] > HEARTBEAT_TIMEOUT:
            logger.error("No heartbeat in %ds. Restarting.", HEARTBEAT_TIMEOUT)
            logging.shutdown()
            os.execv(sys.executable, [sys.executable] + sys.argv)


def _sleep_with_heartbeat(seconds: float, last_heartbeat: list[float],
                          stop_event: threading.Event):
    deadline = time.time() + seconds
    while time.time() < deadline and not stop_event.is_set():
        last_heartbeat[0] = time.time()
        time.sleep(min(30, max(0, deadline - time.time())))


def load_state(records_cls) -> dict:
    """Load persisted validator state (survives restarts and heartbeat re-execs)."""
    try:
        with open(STATE_PATH) as f:
            raw = json.load(f)
        records = {
            int(uid): records_cls(**rec) for uid, rec in raw.get("records", {}).items()
        }
        return {
            "records": records,
            "prev_uids": [int(u) for u in raw.get("prev_uids", [])],
            "prev_weights": [float(w) for w in raw.get("prev_weights", [])],
            "last_epoch_index": int(raw.get("last_epoch_index", -1)),
        }
    except FileNotFoundError:
        return {"records": {}, "prev_uids": [], "prev_weights": [], "last_epoch_index": -1}
    except Exception as e:
        logger.warning("Could not load state from %s: %s — starting fresh", STATE_PATH, e)
        return {"records": {}, "prev_uids": [], "prev_weights": [], "last_epoch_index": -1}


def save_state(records: dict, prev_uids: list[int], prev_weights: list[float],
               last_epoch_index: int):
    os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
    payload = {
        "records": {str(uid): dataclasses.asdict(rec) for uid, rec in records.items()},
        "prev_uids": prev_uids,
        "prev_weights": prev_weights,
        "last_epoch_index": last_epoch_index,
    }
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, STATE_PATH)


def build_model_pool(
    model_pools: dict[int, list[str]],
    prices: dict[str, tuple[float, float]],
    n_questions: int,
    *,
    max_models_per_miner: int,
    max_model_pool: int,
    max_cost_per_query: float,
    budget_usd: float,
) -> list[str]:
    """Deterministic, abuse-resistant union model pool.

    Policy:
    - each miner's declared pool is truncated to max_models_per_miner;
    - only models with a known price are callable (unknown models would
      bypass both the cost cap and the budget tracker);
    - models above the per-query cost cap are dropped;
    - if the union exceeds max_model_pool, models declared by MORE miners
      win (alphabetical truncation would let a sybil evict everyone else's
      models with names like "aaa/x"); ties break alphabetically;
    - most expensive models are then dropped until the estimated epoch cost
      fits the budget.
    """
    declare_counts: dict[str, int] = {}
    for pool in model_pools.values():
        for m in sorted(set(pool))[:max_models_per_miner]:
            declare_counts[m] = declare_counts.get(m, 0) + 1

    def est_cost(m: str) -> float:
        pin, pout = prices[m]
        return pin * 500 + pout * 500

    candidates = []
    for m in sorted(declare_counts):
        if m not in prices:
            logger.warning("Model %s has no known price — excluded from matrix", m)
            continue
        if est_cost(m) > max_cost_per_query:
            logger.warning("Model %s exceeds $%.2f/query cost cap — excluded",
                           m, max_cost_per_query)
            continue
        candidates.append(m)

    if len(candidates) > max_model_pool:
        candidates.sort(key=lambda m: (-declare_counts[m], m))
        logger.warning("Union pool %d exceeds cap %d — keeping most-declared models",
                       len(candidates), max_model_pool)
        candidates = sorted(candidates[:max_model_pool])

    est_total = sum(est_cost(m) for m in candidates) * n_questions
    while candidates and est_total > budget_usd:
        most_expensive = max(candidates, key=lambda m: (est_cost(m), m))
        logger.warning("Estimated epoch cost $%.2f exceeds budget $%.2f — dropping %s",
                       est_total, budget_usd, most_expensive)
        candidates.remove(most_expensive)
        est_total = sum(est_cost(m) for m in candidates) * n_questions

    return candidates


@click.command()
@click.option("--network", default=lambda: os.getenv("FUGAL_NETWORK", "test"),
              help="Network: finney, test, local")
@click.option("--netuid", type=int,
              default=lambda: int(os.getenv("FUGAL_NETUID", "1")),
              help="Subnet netuid")
@click.option("--coldkey", default=lambda: os.getenv("WALLET_NAME", "default"),
              help="Wallet coldkey name")
@click.option("--hotkey", default=lambda: os.getenv("HOTKEY_NAME", "default"),
              help="Hotkey name")
@click.option("--once", is_flag=True, help="Run one epoch and exit")
@click.option("--log-level",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              default=lambda: os.getenv("LOG_LEVEL", "INFO"),
              help="Logging level")
@click.option("--mock", is_flag=True, help="Use mock API responses (no spend)")
def main(network, netuid, coldkey, hotkey, once, log_level, mock):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    import bittensor as bt
    from fugal_subnet.protocol import FugalSynapse
    from fugal_subnet.config import (
        EPOCH_INTERVAL, SLICE_SIZE, ROUTING_LAMBDA, EPOCH_BUDGET_USD,
        MAX_MODEL_POOL, MAX_MODELS_PER_MINER, MAX_MODEL_COST_PER_QUERY,
        REQUIRE_COMMITMENT,
    )
    from fugal_subnet.benchmarks.loader import load_all
    from fugal_subnet.benchmarks.slicer import select_slice, derive_nonce
    from fugal_subnet.matrix import build_matrix, build_matrix_mock
    from fugal_subnet.soft_targets import compute_soft_targets
    from fugal_subnet.head_eval import load_head_from_b64, evaluate_head
    from fugal_subnet.scoring import MinerRecord, ScoringState, update_scores
    from fugal_subnet.rewards import compute_weights, cap_weight_change
    from fugal_subnet.api import BudgetExceeded, SpendTracker, load_prices, fetch_openrouter_prices
    from fugal_subnet.backbone import compute_hidden_states
    from fugal_subnet.commit_reveal import commit_epoch, reveal_epoch
    from fugal_subnet.commitments import get_commitments_with_blocks
    from fugal_subnet.dedup import find_duplicates
    from fugal_subnet.epoch_logger import EpochLog, EpochTimer, detect_anomalies, write_epoch_log

    import base64

    wallet = bt.Wallet(name=coldkey, hotkey=hotkey)
    subtensor = bt.Subtensor(network=network)
    metagraph = subtensor.metagraph(netuid)
    dendrite = bt.Dendrite(wallet=wallet)

    my_hotkey = wallet.hotkey.ss58_address
    if my_hotkey not in metagraph.hotkeys:
        logger.error("Hotkey %s not registered on netuid %d", my_hotkey, netuid)
        return
    my_uid = metagraph.hotkeys.index(my_hotkey)
    logger.info("Validator UID %d on %s netuid %d", my_uid, network, netuid)
    try:
        if not bool(metagraph.validator_permit[my_uid]):
            logger.warning("This hotkey has NO validator permit — set_weights will "
                           "likely fail until it gains enough stake.")
    except (AttributeError, IndexError):
        pass

    logger.info("Loading benchmark pool...")
    benchmark_pool = load_all()
    logger.info("Benchmark pool: %d questions", len(benchmark_pool))
    if not benchmark_pool:
        logger.error("Benchmark pool is empty — nothing to score. Exiting.")
        return

    def refresh_prices(previous: dict) -> dict:
        try:
            fresh = fetch_openrouter_prices()
            logger.info("Fetched live prices for %d models from OpenRouter", len(fresh))
            return fresh
        except Exception:
            if previous:
                logger.warning("Price refresh failed, keeping previous prices")
                return previous
            try:
                fallback = load_prices()
                logger.warning("Live prices unavailable — using local models.json fallback "
                               "(%d models)", len(fallback))
                return fallback
            except Exception:
                logger.error("No price data available at all")
                return {}

    prices: dict = refresh_prices({})

    state = load_state(MinerRecord)
    scoring_state = ScoringState(records=state["records"])
    prev_uids: list[int] = state["prev_uids"]
    prev_weights: list[float] = state["prev_weights"]
    last_epoch_index: int = state["last_epoch_index"]

    stop_event = threading.Event()
    last_heartbeat = [time.time()]
    hb_thread = threading.Thread(
        target=heartbeat_monitor, args=(last_heartbeat, stop_event),
        daemon=True,
    )
    hb_thread.start()

    blocks_per_epoch = max(1, EPOCH_INTERVAL // BLOCK_TIME_S)

    while True:
        last_heartbeat[0] = time.time()
        try:
            current_block = subtensor.get_current_block()
            epoch_index = current_block // blocks_per_epoch

            if epoch_index <= last_epoch_index and not once:
                _sleep_with_heartbeat(30, last_heartbeat, stop_event)
                continue
            if epoch_index <= last_epoch_index and once:
                logger.warning("Epoch %d already processed — running again (--once)", epoch_index)

            boundary_block = epoch_index * blocks_per_epoch
            block_hash = subtensor.get_block_hash(boundary_block)
            epoch_id = f"e{epoch_index:08d}"

            timer = EpochTimer()
            metagraph = subtensor.metagraph(netuid)
            logger.info("=== Epoch %s (boundary block %d, hash %s) ===",
                        epoch_id, boundary_block, block_hash[:16])

            timer.start_phase("slice")
            nonce = derive_nonce(epoch_id, block_hash)
            questions = select_slice(nonce, benchmark_pool, SLICE_SIZE)
            benchmark_hash = hashlib.sha256(
                "|".join(q["question_id"] for q in questions).encode()
            ).hexdigest()[:16]
            logger.info("Slice: %d questions, hash=%s", len(questions), benchmark_hash)

            timer.start_phase("commit")
            commitment = commit_epoch(epoch_id, questions, block_hash)
            logger.info("Committed: %s", commitment.commit_hash)

            timer.start_phase("query")
            synapse = FugalSynapse(epoch_id=epoch_id, benchmark_hash=benchmark_hash)
            n_neurons = int(metagraph.n)
            logger.info("Querying %d miners...", n_neurons)
            responses = dendrite.query(
                metagraph.axons,
                synapse,
                timeout=120,
            )

            timer.start_phase("commitments")
            chain_commitments: dict[str, tuple[str, int]] = {}
            try:
                chain_commitments = get_commitments_with_blocks(subtensor, netuid)
                logger.info("On-chain commitments: %d hotkeys", len(chain_commitments))
            except Exception as e:
                logger.warning("Could not read on-chain commitments: %s", e)

            timer.start_phase("parse_heads")
            heads = {}
            head_hashes = {}
            model_pools = {}
            commit_blocks: dict[int, float] = {}
            n_invalid = 0
            for uid, resp in enumerate(responses):
                if resp is None or not hasattr(resp, "head_npz_b64") or not resp.head_npz_b64:
                    continue
                try:
                    raw = base64.b64decode(resp.head_npz_b64)
                    head = load_head_from_b64(resp.head_npz_b64)
                except Exception as e:
                    n_invalid += 1
                    logger.warning("Bad head from UID %d: %s", uid, e)
                    continue

                # Hash the actual bytes — never trust the miner-claimed hash.
                actual_hash = hashlib.sha256(raw).hexdigest()
                head.commit_hash = actual_hash

                hotkey_ss58 = metagraph.hotkeys[uid] if uid < len(metagraph.hotkeys) else ""
                committed = chain_commitments.get(hotkey_ss58, ("", 0))
                has_valid_commitment = (
                    committed[0] == actual_hash
                    and 0 < committed[1] <= boundary_block
                )
                if has_valid_commitment:
                    commit_blocks[uid] = committed[1]
                elif REQUIRE_COMMITMENT:
                    n_invalid += 1
                    logger.warning(
                        "UID %d head rejected: hash %s... not committed on-chain at or "
                        "before boundary block %d (committed=%s..., block=%d). The miner "
                        "must commit its head hash and wait for the next epoch.",
                        uid, actual_hash[:12], boundary_block,
                        committed[0][:12] if committed[0] else "none", committed[1],
                    )
                    continue
                else:
                    commit_blocks[uid] = math.inf  # scoreable but always dedup-junior

                heads[uid] = head
                head_hashes[uid] = actual_hash
                model_pools[uid] = list(resp.model_pool or head.models)

            if not heads:
                logger.warning("No valid heads received, skipping epoch")
                epoch_log = EpochLog(
                    epoch_id=epoch_id, block_hash=block_hash,
                    timestamp=time.time(), n_questions=len(questions),
                    n_miners_queried=n_neurons, n_heads_valid=0,
                    n_heads_invalid=n_invalid,
                    commit_hash=commitment.commit_hash,
                    anomalies=["no_valid_heads"],
                    duration_s=timer.total_s,
                )
                write_epoch_log(epoch_log)
                last_epoch_index = epoch_index
                save_state(scoring_state.records, prev_uids, prev_weights, last_epoch_index)
                if once:
                    break
                _sleep_with_heartbeat(60, last_heartbeat, stop_event)
                continue

            logger.info("Received %d valid heads", len(heads))

            timer.start_phase("prices")
            prices = refresh_prices(prices)
            if not prices and not mock:
                logger.error("No price data — refusing to call models blind. Skipping epoch.")
                last_epoch_index = epoch_index
                save_state(scoring_state.records, prev_uids, prev_weights, last_epoch_index)
                if once:
                    break
                _sleep_with_heartbeat(300, last_heartbeat, stop_event)
                continue

            timer.start_phase("matrix")
            if mock:
                all_models = sorted(set(
                    m for pool in model_pools.values() for m in pool
                ))[:MAX_MODEL_POOL]
            else:
                all_models = build_model_pool(
                    model_pools, prices, len(questions),
                    max_models_per_miner=MAX_MODELS_PER_MINER,
                    max_model_pool=MAX_MODEL_POOL,
                    max_cost_per_query=MAX_MODEL_COST_PER_QUERY,
                    budget_usd=EPOCH_BUDGET_USD,
                )
            logger.info("Union model pool: %d models", len(all_models))
            if not all_models:
                logger.warning("No callable models in pool, skipping epoch")
                last_epoch_index = epoch_index
                save_state(scoring_state.records, prev_uids, prev_weights, last_epoch_index)
                if once:
                    break
                _sleep_with_heartbeat(60, last_heartbeat, stop_event)
                continue

            tracker = SpendTracker(budget_cap_usd=EPOCH_BUDGET_USD)
            try:
                if mock:
                    matrix_result = build_matrix_mock(questions, all_models)
                else:
                    matrix_result = build_matrix(
                        questions, all_models,
                        tracker=tracker, prices=prices,
                        cache_dir=RESPONSE_CACHE_DIR,
                        allow_exec=True,
                    )
            except BudgetExceeded as e:
                logger.error("EPOCH ABORTED — %s. Partial matrix discarded; no weights "
                             "set. Raise FUGAL_EPOCH_BUDGET or shrink the pool.", e)
                last_epoch_index = epoch_index
                save_state(scoring_state.records, prev_uids, prev_weights, last_epoch_index)
                if once:
                    break
                _sleep_with_heartbeat(EPOCH_INTERVAL, last_heartbeat, stop_event)
                continue
            logger.info("Matrix built: %s, cost=$%.4f",
                        matrix_result.matrix.shape, tracker.total_cost_usd)

            soft = compute_soft_targets(matrix_result.matrix)

            model_costs = {}
            for m in all_models:
                if m in prices:
                    pin, pout = prices[m]
                    model_costs[m] = pin * 500 + pout * 500
                else:
                    model_costs[m] = 0.01

            timer.start_phase("hidden_states")
            if mock:
                hidden_dim = heads[next(iter(heads))].W.shape[1]
                np.random.seed(int.from_bytes(nonce[:4], "big"))
                hidden = np.random.randn(len(questions), hidden_dim).astype(np.float32)
                hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
                logger.info("Using mock hidden states (%d × %d)", *hidden.shape)
            else:
                prompts = [q["prompt"] for q in questions]
                device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
                hidden = compute_hidden_states(prompts, device=device)
                logger.info("Backbone hidden states: %s on %s", hidden.shape, device)

            timer.start_phase("evaluate")
            epoch_scores = {}
            for uid, head in heads.items():
                score = evaluate_head(
                    head, hidden, matrix_result.matrix,
                    all_models, soft, model_costs, lam=ROUTING_LAMBDA,
                )
                epoch_scores[uid] = score
                logger.info("  UID %d: acc=%.3f cost_eff=%.3f kl=%.3f",
                            uid, score.accuracy, score.cost_efficiency, score.kl_score)

            timer.start_phase("dedup_score_weight")
            head_outputs = {uid: s.routing_decisions for uid, s in epoch_scores.items()}
            dupes = find_duplicates(head_outputs, commit_blocks)
            if dupes:
                logger.info("Dedup disqualified: %s", dupes)

            scoring_state = update_scores(
                scoring_state, epoch_scores, head_hashes,
                n_questions=len(questions),
            )

            uids, weights = compute_weights(
                scoring_state.records, dedup_disqualified=dupes,
            )

            weight_capped = False
            if prev_uids:
                uids, weights = cap_weight_change(
                    uids, weights, prev_uids, prev_weights,
                )
                weight_capped = True
                logger.info("Weights capped vs previous epoch")

            timer.start_phase("reveal")
            epoch_score_dicts = {
                uid: {"acc": s.accuracy, "cost_eff": s.cost_efficiency, "kl": s.kl_score}
                for uid, s in epoch_scores.items()
            }
            epoch_weight_map = dict(zip(uids, weights))
            routing_decisions = {
                uid: s.routing_decisions.tolist() for uid, s in epoch_scores.items()
            }
            reveal_ok = reveal_epoch(
                epoch_id, questions,
                matrix_result.matrix, all_models, model_costs,
                head_hashes, routing_decisions,
                epoch_score_dicts, epoch_weight_map,
            )

            timer.start_phase("set_weights")
            logger.info("Setting weights for %d UIDs", len(uids))
            response = subtensor.set_weights(
                wallet=wallet, netuid=netuid,
                uids=uids, weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            success, msg = response
            if success:
                logger.info("Weights set successfully")
                prev_uids = uids
                prev_weights = weights
            else:
                logger.warning("Weight-setting failed: %s", msg)

            timer.end_phase()
            anomalies = detect_anomalies(
                epoch_score_dicts, epoch_weight_map,
                n_neurons, len(heads),
            )
            if not reveal_ok:
                anomalies.append("commit_reveal_failed")

            epoch_log = EpochLog(
                epoch_id=epoch_id, block_hash=block_hash,
                timestamp=time.time(), n_questions=len(questions),
                n_miners_queried=n_neurons,
                n_heads_valid=len(heads), n_heads_invalid=n_invalid,
                commit_hash=commitment.commit_hash,
                reveal_verified=reveal_ok,
                scores=epoch_score_dicts, weights=epoch_weight_map,
                dedup_disqualified=list(dupes),
                weight_capped=weight_capped,
                set_weights_success=success,
                set_weights_msg=str(msg) if msg else "",
                anomalies=anomalies,
                duration_s=timer.total_s,
            )
            write_epoch_log(epoch_log)

            last_epoch_index = epoch_index
            save_state(scoring_state.records, prev_uids, prev_weights, last_epoch_index)

        except KeyboardInterrupt:
            logger.info("Validator stopped by user")
            break
        except Exception:
            logger.exception("Error in epoch")

        if once:
            break

        logger.info("Waiting for next epoch boundary (every %d blocks)...", blocks_per_epoch)
        _sleep_with_heartbeat(60, last_heartbeat, stop_event)

    stop_event.set()
    logger.info("Validator shutdown complete")

    if once:
        os._exit(0)


if __name__ == "__main__":
    main()
