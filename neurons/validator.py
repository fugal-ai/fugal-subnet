#!/usr/bin/env python3
"""Fugal subnet validator — TEE proof verification and weight-setting.

Usage:
    python neurons/validator.py --netuid 1 --coldkey fugal_validator --hotkey default
    python neurons/validator.py --netuid 1 --coldkey fugal_validator --hotkey default --once

Validators no longer call models. Each epoch:
1. Derive nonce from block hash, select question slice
2. Query miners for TEE-attested proofs (FugalProofSynapse)
3. Verify each proof (DCAP attestation, measurements, nonce, questions, costs)
4. Score from verified proof results using evidence accumulation
5. Dedup, weight-setting, reveal

Miners run benchmarks inside Intel TDX VMs and pay for their own inference.
The validator's only cost is bandwidth and compute for verification.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# isort: off
import fugal_subnet.determinism  # noqa: F401

import click  # noqa: E402
import numpy as np  # noqa: E402
# isort: on

logger = logging.getLogger("fugal.validator")

STATE_PATH = os.getenv("FUGAL_STATE_PATH", "results/validator_state.json")
BLOCK_TIME_S = 12


def load_state(records_cls) -> dict:
    """Load persisted validator state across supervisor-managed restarts."""
    from fugal_subnet.evidence import Evidence

    try:
        with open(STATE_PATH) as f:
            raw = json.load(f)
        records = {}
        for uid, rec in raw.get("records", {}).items():
            ev_data = rec.pop("evidence", None)
            record = records_cls(**rec)
            if ev_data is not None:
                record.evidence = Evidence(**ev_data)
            records[int(uid)] = record
        return {
            "records": records,
            "prev_uids": [int(u) for u in raw.get("prev_uids", [])],
            "prev_weights": [float(w) for w in raw.get("prev_weights", [])],
            "last_epoch_index": int(raw.get("last_epoch_index", -1)),
            "first_commit_blocks": {
                str(hk): int(blk)
                for hk, blk in (raw.get("first_commit_blocks") or {}).items()
            },
        }
    except FileNotFoundError:
        return _empty_state()
    except Exception as e:
        logger.warning("Could not load state from %s: %s — starting fresh", STATE_PATH, e)
        return _empty_state()


def _empty_state() -> dict:
    return {"records": {}, "prev_uids": [], "prev_weights": [],
            "last_epoch_index": -1, "first_commit_blocks": {}}


def save_state(records: dict, prev_uids: list[int], prev_weights: list[float],
               last_epoch_index: int, first_commit_blocks: dict[str, int]):
    """Persist validator state."""
    os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
    payload = {
        "records": {str(uid): dataclasses.asdict(rec) for uid, rec in records.items()},
        "prev_uids": prev_uids,
        "prev_weights": prev_weights,
        "last_epoch_index": last_epoch_index,
        "first_commit_blocks": first_commit_blocks or {},
    }
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, STATE_PATH)


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
@click.option("--wallet-path", default=lambda: os.getenv("FUGAL_WALLET_PATH") or None,
              type=click.Path(file_okay=False),
              help="Bittensor wallet root (defaults to the SDK wallet directory)")
@click.option("--once", is_flag=True, help="Run one epoch and exit")
@click.option("--log-level",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              default=lambda: os.getenv("LOG_LEVEL", "INFO"),
              help="Logging level")
@click.option(
    "--live/--mock",
    default=False,
    help="Live mode requires real TDX attestation; mock accepts unattested proofs",
)
def main(network, netuid, coldkey, hotkey, wallet_path, once, log_level, live):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from fugal_subnet.config import (
        EPOCH_INTERVAL,
        REQUIRE_COMMITMENT,
        SLICE_SIZE,
        TEE_APPROVED_MEASUREMENTS,
    )
    mock = not live
    logger.info(
        "Operation mode: %s",
        "LIVE (requires real TDX attestation)" if live else "mock (accepts unattested proofs)",
    )

    from fugal_subnet.fingerprint import assert_environment, consensus_digest
    assert_environment(strict=live)
    logger.info("Consensus environment digest: %s", consensus_digest())

    import bittensor as bt

    from fugal_subnet.benchmarks.loader import load_all
    from fugal_subnet.benchmarks.slicer import derive_nonce, select_slice
    from fugal_subnet.commit_reveal import commit_epoch, reveal_epoch
    from fugal_subnet.commitments import get_commitments_with_blocks
    from fugal_subnet.dedup import find_duplicates
    from fugal_subnet.epoch_logger import (
        EpochLog,
        EpochTimer,
        detect_anomalies,
        write_epoch_log,
    )
    from fugal_subnet.head_eval import HeadScore
    from fugal_subnet.protocol import FugalProofSynapse
    from fugal_subnet.rewards import cap_weight_change, compute_weights
    from fugal_subnet.scoring import MinerRecord, ScoringState, update_scores
    from fugal_subnet.tee.proof import BenchmarkProof
    from fugal_subnet.tee.verify import compute_questions_hash, verify_proof

    wallet = bt.Wallet(name=coldkey, hotkey=hotkey, path=wallet_path)
    subtensor = bt.Subtensor(network=network)
    metagraph = subtensor.metagraph(netuid)
    dendrite = bt.Dendrite(wallet=wallet)

    my_hotkey = wallet.hotkey.ss58_address
    if my_hotkey not in metagraph.hotkeys:
        raise click.ClickException(
            f"Hotkey {my_hotkey} is not registered on netuid {netuid}"
        )
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
        raise click.ClickException("Benchmark pool is empty — nothing to score")

    gold_answers = {q["question_id"]: q for q in benchmark_pool}
    approved_measurements = set(TEE_APPROVED_MEASUREMENTS)

    state = load_state(MinerRecord)
    scoring_state = ScoringState(records=state["records"])
    prev_uids: list[int] = state["prev_uids"]
    prev_weights: list[float] = state["prev_weights"]
    last_epoch_index: int = state["last_epoch_index"]
    first_commit_blocks: dict[str, int] = state["first_commit_blocks"]

    blocks_per_epoch = max(1, EPOCH_INTERVAL // BLOCK_TIME_S)

    while True:
        try:
            current_block = subtensor.get_current_block()
            epoch_index = current_block // blocks_per_epoch

            if epoch_index <= last_epoch_index and not once:
                time.sleep(30)
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

            # --- SLICE ---
            timer.start_phase("slice")
            nonce = derive_nonce(epoch_id, block_hash)
            questions = select_slice(nonce, benchmark_pool, SLICE_SIZE)
            question_ids = [q["question_id"] for q in questions]
            expected_questions_hash = compute_questions_hash(question_ids)
            logger.info("Slice: %d questions, hash=%s...",
                        len(questions), expected_questions_hash[:16])

            # --- COMMIT ---
            timer.start_phase("commit")
            commitment = commit_epoch(epoch_id, questions, block_hash)
            logger.info("Committed: %s", commitment.commit_hash)

            # --- QUERY MINERS FOR PROOFS ---
            timer.start_phase("query")
            nonce_hex = nonce.hex()
            synapse = FugalProofSynapse(epoch_id=epoch_id, nonce=nonce_hex)
            n_neurons = int(metagraph.n)
            logger.info("Querying %d miners for proofs...", n_neurons)
            responses = dendrite.query(
                metagraph.axons,
                synapse,
                timeout=120,
            )

            # --- CHECK COMMITMENTS ---
            timer.start_phase("commitments")
            chain_commitments: dict[str, tuple[str, int]] = {}
            try:
                chain_commitments = get_commitments_with_blocks(subtensor, netuid)
                logger.info("On-chain commitments: %d hotkeys", len(chain_commitments))
            except Exception as e:
                logger.warning("Could not read on-chain commitments: %s", e)

            # --- VERIFY PROOFS ---
            timer.start_phase("verify_proofs")
            verified_proofs: dict[int, BenchmarkProof] = {}
            head_hashes: dict[int, str] = {}
            commit_blocks: dict[int, float] = {}
            n_invalid = 0

            for uid, resp in enumerate(responses):
                if resp is None or not hasattr(resp, "proof_hash") or not resp.proof_hash:
                    continue

                hotkey_ss58 = metagraph.hotkeys[uid] if uid < len(metagraph.hotkeys) else ""
                weights_hash = getattr(resp, "weights_hash", "")

                # Check on-chain commitment
                committed = chain_commitments.get(hotkey_ss58, ("", 0))
                has_valid_commitment = (
                    committed[0] == weights_hash
                    and 0 < committed[1] <= boundary_block
                )
                if has_valid_commitment:
                    hotkey_first = first_commit_blocks.get(hotkey_ss58)
                    if hotkey_first is None or committed[1] < hotkey_first:
                        first_commit_blocks[hotkey_ss58] = int(committed[1])
                    commit_blocks[uid] = first_commit_blocks[hotkey_ss58]
                elif REQUIRE_COMMITMENT:
                    n_invalid += 1
                    logger.warning(
                        "UID %d proof rejected: weights_hash %s... not committed on-chain "
                        "at or before boundary block %d",
                        uid, weights_hash[:12], boundary_block,
                    )
                    continue
                else:
                    commit_blocks[uid] = math.inf

                # In a full implementation, download the proof bundle from
                # resp.proof_bundle_url. For now, the proof is served inline
                # or via a separate channel. Mock mode constructs synthetic proofs.
                try:
                    proof = _get_proof_for_uid(uid, resp, mock)
                except Exception as e:
                    n_invalid += 1
                    logger.warning("UID %d: failed to get proof: %s", uid, e)
                    continue

                if proof is None:
                    n_invalid += 1
                    continue

                # Verify the proof
                result = verify_proof(
                    proof,
                    approved_measurements=approved_measurements,
                    expected_questions_hash=expected_questions_hash,
                    expected_nonce=nonce_hex,
                    gold_answers=gold_answers,
                    mock=mock,
                )

                if not result.valid:
                    n_invalid += 1
                    logger.warning("UID %d proof failed verification: %s", uid, result.reason)
                    continue

                if result.warnings:
                    for w in result.warnings:
                        logger.info("UID %d proof warning: %s", uid, w)

                verified_proofs[uid] = proof
                head_hashes[uid] = weights_hash

            if not verified_proofs:
                logger.warning("No valid proofs received, skipping epoch")
                epoch_log = EpochLog(
                    epoch_id=epoch_id, block_hash=block_hash,
                    timestamp=time.time(), n_questions=len(questions),
                    n_miners_queried=n_neurons, n_heads_valid=0,
                    n_heads_invalid=n_invalid,
                    commit_hash=commitment.commit_hash,
                    anomalies=["no_valid_proofs"],
                    duration_s=timer.total_s,
                )
                write_epoch_log(epoch_log)
                last_epoch_index = epoch_index
                save_state(scoring_state.records, prev_uids, prev_weights,
                           last_epoch_index, first_commit_blocks)
                if once:
                    break
                time.sleep(60)
                continue

            logger.info("Verified %d proofs", len(verified_proofs))

            # --- SCORE FROM PROOFS ---
            timer.start_phase("score")
            epoch_scores: dict[int, HeadScore] = {}
            for uid, proof in verified_proofs.items():
                score = _proof_to_head_score(proof)
                epoch_scores[uid] = score
                logger.info("  UID %d: acc=%.3f cost=$%.4f (%d/%d correct)",
                            uid, score.accuracy, proof.total_cost_usd,
                            score.n_correct, score.n_scored)

            # --- DEDUP ---
            timer.start_phase("dedup_score_weight")
            routing_decisions = {
                uid: _extract_routing_decisions(proof)
                for uid, proof in verified_proofs.items()
            }
            dupes = find_duplicates(routing_decisions, commit_blocks)
            if dupes:
                logger.info("Dedup disqualified: %s", dupes)

            # --- EVIDENCE ACCUMULATION ---
            scoring_state = update_scores(
                scoring_state, epoch_scores, head_hashes,
                n_questions=len(questions),
            )

            # --- WEIGHTS ---
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

            # --- REVEAL ---
            timer.start_phase("reveal")
            epoch_score_dicts = {
                uid: {"acc": s.accuracy, "cost_eff": s.cost_efficiency, "kl": s.kl_score}
                for uid, s in epoch_scores.items()
            }
            epoch_weight_map = dict(zip(uids, weights))

            # Build a minimal matrix from proof results for the reveal
            all_models = sorted(set(
                r.routed_model
                for proof in verified_proofs.values()
                for r in proof.results
            ))
            matrix = np.zeros((len(questions), max(len(all_models), 1)), dtype=np.int32)
            model_costs: dict[str, float] = {}
            for proof in verified_proofs.values():
                for r in proof.results:
                    if r.routed_model in all_models:
                        model_idx = all_models.index(r.routed_model)
                        q_idx_map = {q["question_id"]: i for i, q in enumerate(questions)}
                        if r.question_id in q_idx_map:
                            matrix[q_idx_map[r.question_id], model_idx] = int(r.correct)
                    model_costs[r.routed_model] = (
                        model_costs.get(r.routed_model, 0.0) + r.cost_usd
                    )

            routing_for_reveal = {
                uid: _extract_routing_decisions(proof).tolist()
                for uid, proof in verified_proofs.items()
            }
            reveal_ok = reveal_epoch(
                epoch_id, questions,
                matrix, all_models, model_costs,
                head_hashes, routing_for_reveal,
                epoch_score_dicts, epoch_weight_map,
            )

            # --- SET WEIGHTS ---
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
                n_neurons, len(verified_proofs),
            )
            if not reveal_ok:
                anomalies.append("commit_reveal_failed")

            epoch_log = EpochLog(
                epoch_id=epoch_id, block_hash=block_hash,
                timestamp=time.time(), n_questions=len(questions),
                n_miners_queried=n_neurons,
                n_heads_valid=len(verified_proofs), n_heads_invalid=n_invalid,
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
            save_state(scoring_state.records, prev_uids, prev_weights,
                       last_epoch_index, first_commit_blocks)

        except KeyboardInterrupt:
            logger.info("Validator stopped by user")
            break
        except Exception:
            logger.exception("Error in epoch")

        if once:
            break

        logger.info("Waiting for next epoch boundary (every %d blocks)...", blocks_per_epoch)
        time.sleep(60)

    logger.info("Validator shutdown complete")

    if once:
        sys.exit(0)


def _get_proof_for_uid(uid, resp, mock):
    """Download and parse a BenchmarkProof for a given UID.

    Downloads the proof bundle from the HuggingFace URL in
    resp.proof_bundle_url. Returns None if the URL is empty or
    the download/parse fails.
    """
    from fugal_subnet.tee.proof import BenchmarkProof
    from fugal_subnet.tee.store import download_proof

    url = getattr(resp, "proof_bundle_url", "") or ""
    if not url:
        logger.debug("UID %d: no proof_bundle_url", uid)
        return None

    try:
        proof = download_proof(url)
    except Exception as e:
        logger.warning("UID %d: failed to download proof from %s: %s", uid, url[:60], e)
        return None

    if not isinstance(proof, BenchmarkProof):
        logger.warning("UID %d: downloaded object is not a BenchmarkProof", uid)
        return None

    return proof


def _proof_to_head_score(proof):
    """Convert a verified BenchmarkProof into a HeadScore for scoring."""
    from fugal_subnet.head_eval import HeadScore

    n_correct = proof.n_correct
    n_scored = proof.n_total
    accuracy = proof.accuracy

    # Cost efficiency: ratio of cheapest-correct to actual cost
    # Under TEE, costs come from the attested MeteringProxy
    total_head_cost = proof.total_cost_usd
    total_oracle_cost = total_head_cost * 0.8  # Approximation until oracle assembly

    cost_efficiency = min(1.0, total_oracle_cost / max(total_head_cost, 1e-10))

    # KL divergence placeholder — requires oracle distribution
    total_kl = 0.0

    return HeadScore(
        accuracy=accuracy,
        cost_efficiency=cost_efficiency,
        kl_score=0.0,
        routing_decisions=_extract_routing_decisions(proof),
        correct_mask=np.array([r.correct for r in proof.results], dtype=bool),
        coverage=1.0,
        n_correct=n_correct,
        n_scored=n_scored,
        total_head_cost=total_head_cost,
        total_oracle_cost=total_oracle_cost,
        total_kl=total_kl,
    )


def _extract_routing_decisions(proof):
    """Extract routing decisions from a proof as a numpy array for dedup."""
    models = sorted(set(r.routed_model for r in proof.results))
    model_to_idx = {m: i for i, m in enumerate(models)}
    decisions = np.array(
        [model_to_idx.get(r.routed_model, 0) for r in proof.results],
        dtype=np.int32,
    )
    return decisions


if __name__ == "__main__":
    main()
