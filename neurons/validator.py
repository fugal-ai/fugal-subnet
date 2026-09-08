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
            "frame": raw.get("frame"),
        }
    except FileNotFoundError:
        return _empty_state()
    except Exception as e:
        logger.warning("Could not load state from %s: %s — starting fresh", STATE_PATH, e)
        return _empty_state()


def _empty_state() -> dict:
    return {"records": {}, "prev_uids": [], "prev_weights": [],
            "last_epoch_index": -1, "first_commit_blocks": {}, "frame": None}


def save_state(records: dict, prev_uids: list[int], prev_weights: list[float],
               last_epoch_index: int, first_commit_blocks: dict[str, int],
               frame=None):
    """Persist validator state."""
    os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
    payload = {
        "records": {str(uid): dataclasses.asdict(rec) for uid, rec in records.items()},
        "prev_uids": prev_uids,
        "prev_weights": prev_weights,
        "last_epoch_index": last_epoch_index,
        "first_commit_blocks": first_commit_blocks or {},
        "frame": frame.to_dict() if frame is not None else None,
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
        EPOCH_COLLECT_FRACTION,
        EPOCH_INTERVAL,
        EXPLORE_FRACTION,
        FRAME_DEFAULT_COMPLETION_TOKENS,
        REQUIRE_COMMITMENT,
        SLICE_SIZE,
        TEE_APPROVED_MEASUREMENTS,
    )
    mock = not live
    logger.info(
        "Operation mode: %s",
        "LIVE (requires real TDX attestation)" if live else "mock (accepts unattested proofs)",
    )
    # A live validator with no approved measurements rejects every proof it
    # ever receives — `measurement_id(quote) in approved` can never hold against
    # an empty set — and then sets every miner to zero weight. Fail-closed is
    # the right default for the check itself, but silently deweighting the whole
    # subnet is not an acceptable way to learn the variable is unset.
    if live and not TEE_APPROVED_MEASUREMENTS:
        raise click.ClickException(
            "--live requires FUGAL_TEE_MEASUREMENTS. With no approved "
            "measurement every proof fails the runtime-image check, so this "
            "validator would score every miner zero and set weights to match. "
            "Get the value from `scripts/tdx_measurement.py` on the approved "
            "TDX image (see docs/TDX_VALIDATION.md)."
        )

    # Say which entries are active, at startup, in the log. During a rotation
    # two validators are meant to carry the same two entries, and until this
    # line the only evidence of what a validator had loaded was its env file —
    # a divergence between validators was diagnosable only by logging in to
    # each host. Parsed here too, before any chain connection, so a malformed
    # entry (a bare trailing colon would silently accept any application)
    # fails at startup rather than at the first proof.
    if TEE_APPROVED_MEASUREMENTS:
        from fugal_subnet.tee.verify import parse_approved
        try:
            parsed_entries = parse_approved(TEE_APPROVED_MEASUREMENTS)
        except ValueError as e:
            raise click.ClickException(f"FUGAL_TEE_MEASUREMENTS: {e}") from e
        described = [
            f"{base[:12]}…:{app[:12]}…" if app else f"{base[:12]}… (bare base: any app)"
            for base, apps in sorted(parsed_entries.items())
            for app in (sorted(apps) or [""])
        ]
        logger.info("Approved entries: %d — %s", len(described), "; ".join(described))
        if live and any(not apps for apps in parsed_entries.values()):
            logger.warning(
                "A bare base entry is active: any application on that image is "
                "approved. Production entries should be <base>:<compose_hash>."
            )

    from fugal_subnet.fingerprint import assert_environment, consensus_digest
    assert_environment(strict=live)
    logger.info("Consensus environment digest: %s", consensus_digest())

    import bittensor as bt

    # After bittensor: importing it disables every logger that already exists,
    # so without this the neuron's own output — including every error and
    # traceback — is silently discarded. See fugal_subnet/logging_setup.
    from fugal_subnet.logging_setup import configure_logging
    configure_logging(log_level)

    from fugal_subnet.api import load_prices
    from fugal_subnet.benchmarks.loader import load_all, pool_hash
    from fugal_subnet.benchmarks.slicer import (
        blocks_per_epoch as blocks_per_epoch_fn,
    )
    from fugal_subnet.benchmarks.slicer import (
        collect_block_for_epoch,
        derive_nonce,
        epoch_id_for_block,
        epoch_index_for_block,
        select_slice,
    )
    from fugal_subnet.commit_reveal import commit_epoch, reveal_epoch
    from fugal_subnet.commitments import get_commitments_with_blocks
    from fugal_subnet.config import TEE_COLLATERAL_EPOCH_BUDGET
    from fugal_subnet.dedup import find_duplicates
    from fugal_subnet.epoch_logger import (
        EpochLog,
        EpochTimer,
        detect_anomalies,
        write_epoch_log,
    )
    from fugal_subnet.exploration import expected_exploration as expected_exploration_map
    from fugal_subnet.frontier import build_frontier
    from fugal_subnet.head_eval import HeadScore
    from fugal_subnet.pricing import policy_cost_total, question_input_tokens
    from fugal_subnet.protocol import FugalProofSynapse
    from fugal_subnet.reference_frame import (
        ReferenceFrame,
        accumulate_exploration,
        best_model,
        implausible_exploration,
        load_bootstrap,
        rebuild_from_reveals,
        reference_cost,
    )
    from fugal_subnet.rewards import BURN_UID, cap_weight_change, compute_weights
    from fugal_subnet.scoring import MinerRecord, ScoringState, update_scores
    from fugal_subnet.tee.attestation import CollateralBudget, verification_order
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
    logger.info("Benchmark pool: %d questions, pool_hash=%s",
                len(benchmark_pool), pool_hash(benchmark_pool)[:16])
    if not benchmark_pool:
        raise click.ClickException("Benchmark pool is empty — nothing to score")

    gold_answers = {q["question_id"]: q for q in benchmark_pool}
    approved_measurements = set(TEE_APPROVED_MEASUREMENTS)

    # Pinned price table — the consensus cost denominator. Loaded once so every
    # epoch prices against the same rates, and so a missing table fails at
    # startup rather than mid-epoch.
    prices = load_prices()
    # One global, sorted model space. Used for exploration targets and for
    # dedup indices, so every miner's routing vector is directly comparable.
    pool_models = sorted(prices)
    model_index = {m: i for i, m in enumerate(pool_models)}
    explore_size = max(1, int(round(SLICE_SIZE * EXPLORE_FRACTION)))
    logger.info("Price table: %d models; exploration quota %d questions/epoch",
                len(prices), explore_size)


    state = load_state(MinerRecord)
    scoring_state = ScoringState(records=state["records"])
    prev_uids: list[int] = state["prev_uids"]
    prev_weights: list[float] = state["prev_weights"]
    last_epoch_index: int = state["last_epoch_index"]
    first_commit_blocks: dict[str, int] = state["first_commit_blocks"]

    # The reference frame is subnet-level state accumulated over TIME, not over
    # the miner field — a miner's score must not move because other miners came
    # online or went dark (I4). Seeded from the shipped bootstrap prior so it is
    # well-defined at epoch 1 with zero samples.
    if state["frame"]:
        frame = ReferenceFrame.from_dict(state["frame"])
    elif os.getenv("FUGAL_FRAME_BOOTSTRAP"):
        # No local state. Rebuild from published reveals rather than starting
        # from the neutral prior, because a fresh frame is not a smaller version
        # of an established one — it is a DIFFERENT one. acc_best is the
        # denominator of every quality term, so a validator that starts cold
        # scores the whole field differently from its peers: 0.500 against 0.832
        # on measured data. Without this a lost disk, or simply a second
        # validator joining, diverges the subnet.
        frame = rebuild_from_reveals(os.environ["FUGAL_FRAME_BOOTSTRAP"])
    else:
        frame = load_bootstrap()
        logger.warning(
            "Starting from the bootstrap prior with no accumulated frame. If "
            "other validators have history, this one will score the field "
            "differently until it catches up. Set FUGAL_FRAME_BOOTSTRAP to a "
            "directory of published reveals to rebuild instead."
        )

    blocks_per_epoch = blocks_per_epoch_fn(EPOCH_INTERVAL)

    while True:
        try:
            current_block = subtensor.get_current_block()
            epoch_index = epoch_index_for_block(current_block, blocks_per_epoch)

            if epoch_index <= last_epoch_index and not once:
                time.sleep(30)
                continue
            if epoch_index <= last_epoch_index and once:
                logger.warning("Epoch %d already processed — running again (--once)", epoch_index)

            boundary_block = epoch_index * blocks_per_epoch
            block_hash = subtensor.get_block_hash(boundary_block)
            epoch_id = epoch_id_for_block(epoch_index)

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
            expected_question_ids = set(question_ids)
            explore_map = expected_exploration_map(
                nonce, benchmark_pool, expected_question_ids,
                pool_models, explore_size,
            )
            # Gold for the assigned slice only. Handing over the whole pool
            # would let a proof reference any question in it and still verify.
            slice_gold = {
                qid: gold_answers[qid]
                for qid in list(question_ids) + list(explore_map)
                if qid in gold_answers
            }
            logger.info("Slice: %d questions, hash=%s...",
                        len(questions), expected_questions_hash[:16])

            # --- COMMIT ---
            timer.start_phase("commit")
            commitment = commit_epoch(epoch_id, questions, block_hash)
            logger.info("Committed: %s", commitment.commit_hash)

            # --- WAIT FOR THE COLLECTION BLOCK ---
            # A miner cannot start benchmarking until this epoch's boundary
            # block exists, because that block's hash is what picks the
            # questions. Querying at the boundary therefore asks before any
            # proof can exist: with real models on a 300-question slice the
            # miner needs minutes, the validator would collect nothing, and —
            # worse than merely failing — it marks the epoch done and never
            # returns to it. Every epoch, forever, logged as "no valid proofs"
            # as though the miners were at fault.
            timer.start_phase("await_collect")
            collect_block = collect_block_for_epoch(
                epoch_index, blocks_per_epoch, EPOCH_COLLECT_FRACTION,
            )
            wait_for_block(subtensor, collect_block)

            # Read the metagraph AT the collection block, not at "now". Both
            # the block and the offset are derived identically by every honest
            # validator, so they query the same axons at the same chain state
            # and any difference in what they collect is a real difference
            # rather than a timing artefact.
            metagraph = metagraph_at(subtensor, netuid, collect_block, metagraph)

            # --- QUERY MINERS FOR PROOFS ---
            timer.start_phase("query")
            nonce_hex = nonce.hex()
            synapse = FugalProofSynapse(epoch_id=epoch_id, nonce=nonce_hex)
            n_neurons = int(metagraph.n)
            logger.info("Querying %d miners for proofs at block %d (boundary %d + %d)...",
                        n_neurons, collect_block, boundary_block,
                        collect_block - boundary_block)
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
            verified_heads: dict[int, bytes] = {}
            head_hashes: dict[int, str] = {}
            commit_blocks: dict[int, float] = {}
            n_invalid = 0
            # Kept apart from n_invalid on purpose. A proof this validator
            # could not check is not a proof it rejected, and the difference is
            # the difference between an outage and an accusation.
            n_unverifiable = 0

            # One ceiling for the whole phase. Every fetch is already bounded
            # individually; this bounds their SUM, which is the quantity I6
            # actually cares about -- the loop below is serial, so 256
            # individually-bounded fetches still overrun the epoch and miss the
            # weight-setting window they exist to hit.
            collateral_budget = CollateralBudget(
                TEE_COLLATERAL_EPOCH_BUDGET, expected_proofs=len(responses))

            # Verify in an order derived from the epoch nonce rather than in UID
            # order. This matters ONLY when the budget runs out, and then it
            # matters a lot: whoever is last in line goes unverified, and in UID
            # order that is the same miners every epoch -- a permanent
            # disadvantage for high UIDs that no miner caused and none could
            # escape. The nonce comes from the epoch id and a block hash, so it
            # is identical on every validator (no new I1 divergence) and no
            # miner can influence it (I4), while the disadvantage it allocates
            # rotates each epoch instead of settling on the same miners.
            order = verification_order(nonce_hex, len(responses))

            for uid in order:
                resp = responses[uid]
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

                try:
                    bundle = _get_bundle_for_uid(uid, resp)
                except Exception as e:
                    n_invalid += 1
                    logger.warning("UID %d: failed to get bundle: %s", uid, e)
                    continue

                if bundle is None:
                    n_invalid += 1
                    continue
                proof, head_bytes = bundle

                # Every binding is passed explicitly. A check the validator does
                # not supply an expectation for is a check that does not happen.
                # Defence in depth. verify_proof is contracted not to raise on
                # miner input, but the cost of being wrong about that is the
                # whole epoch, so a bug there must cost one miner instead.
                try:
                    result = verify_proof(
                        proof,
                        approved_measurements=approved_measurements,
                        expected_questions_hash=expected_questions_hash,
                        expected_nonce=nonce_hex,
                        gold_answers=slice_gold,
                        expected_question_ids=expected_question_ids,
                        expected_exploration=explore_map,
                        expected_weights_hash=weights_hash,
                        # The hotkey of the uid we actually queried. A proof
                        # naming a different one was produced for somebody else.
                        expected_hotkey=metagraph.hotkeys[uid],
                        expected_proof_hash=getattr(resp, "proof_hash", ""),
                        head_bytes=head_bytes,
                        mock=mock,
                        collateral_budget=collateral_budget,
                    )
                except ImportError:
                    raise
                except Exception as e:  # noqa: BLE001 - one miner, not the epoch
                    n_invalid += 1
                    logger.warning(
                        "UID %d proof verification raised %s: %s — rejecting this "
                        "miner, not the epoch", uid, type(e).__name__, e,
                    )
                    continue

                if not result.valid:
                    # Skip either way — the miner is not scored this epoch, and
                    # falls through to apply_miss like any absent miner. What
                    # changes is what gets RECORDED, and that is not cosmetic:
                    # an operator reading "247 invalid" hunts an attack, while
                    # "247 unverifiable" points at their own egress.
                    #
                    # Skipping is also the only I6-compatible choice. Abandoning
                    # the epoch instead would let one miner with an unknown
                    # FMSPC stop every validator publishing at once, because the
                    # collection point is a deterministic block — a subnet kill
                    # switch for the price of one registration.
                    if getattr(result, "unverifiable", False):
                        n_unverifiable += 1
                        logger.warning(
                            "UID %d proof could not be checked: %s — NOT counted "
                            "as invalid; this validator never judged it",
                            uid, result.reason,
                        )
                    else:
                        n_invalid += 1
                        logger.warning(
                            "UID %d proof failed verification: %s", uid, result.reason,
                        )
                    continue

                if result.warnings:
                    for w in result.warnings:
                        logger.info("UID %d proof warning: %s", uid, w)

                verified_proofs[uid] = proof
                verified_heads[uid] = head_bytes
                # Keyed on the hash the attestation forced to be true, not the
                # value the miner asserted over the axon.
                head_hashes[uid] = proof.weights_hash

            if not verified_proofs:
                logger.warning("No valid proofs received, skipping epoch")
                epoch_log = EpochLog(
                    epoch_id=epoch_id, block_hash=block_hash,
                    timestamp=time.time(), n_questions=len(questions),
                    n_miners_queried=n_neurons, n_heads_valid=0,
                    n_heads_invalid=n_invalid,
                    n_heads_unverifiable=n_unverifiable,
                    commit_hash=commitment.commit_hash,
                    anomalies=(
                        ["no_valid_proofs", "collateral_unavailable"]
                        if n_unverifiable else ["no_valid_proofs"]
                    ),
                    duration_s=timer.total_s,
                )
                write_epoch_log(epoch_log)
                last_epoch_index = epoch_index
                save_state(scoring_state.records, prev_uids, prev_weights,
                           last_epoch_index, first_commit_blocks, frame)
                if once:
                    break
                time.sleep(60)
                continue

            logger.info("Verified %d proofs", len(verified_proofs))

            # --- REFERENCE FRAME ---
            # Exploration samples from every verified proof, pooled. The frame
            # accumulates over TIME: this epoch's samples are one decayed
            # contribution to a long-running estimate, so the reference a miner
            # is measured against does not lurch when the field size changes.
            timer.start_phase("frame")
            samples = [
                (r.routed_model, r.correct, r.prompt_tokens, r.completion_tokens)
                for proof in verified_proofs.values()
                for r in proof.exploration_results
            ]
            # Detection only, BEFORE folding this epoch's samples in, so a
            # miner is compared against history rather than against a frame it
            # has just contributed to. Never touches a score: I4 says a miner's
            # score cannot depend on other miners, and the frame is built from
            # everyone. This makes the upstream-substitution attack visible and
            # attributable; only measuring the upstream closes it.
            suspicious = implausible_exploration(frame, {
                uid: [(r.routed_model, r.correct) for r in proof.exploration_results]
                for uid, proof in verified_proofs.items()
            })
            for uid, why in suspicious.items():
                logger.warning(
                    "UID %d exploration is implausible: %s. Not penalised — this "
                    "is advisory. See docs/INVARIANTS.md, 'The model upstream is "
                    "miner-controlled'.", uid, why,
                )

            frame = accumulate_exploration(frame, samples)
            ref_model, acc_best = best_model(frame, prices)
            logger.info(
                "Reference frame: %d samples this epoch, best model %s "
                "(acc=%.3f, %.0f trials)",
                len(samples), ref_model, acc_best, frame.trials.get(ref_model, 0.0),
            )
            # The constant-policy frontier for THIS slice: what any single model,
            # or any random mixture of models, buys at each price. Prompt tokens
            # come from the pool, so the frontier is identical for every miner
            # and on every validator. See fugal_subnet/frontier.py.
            slice_prompt_tokens = sum(
                question_input_tokens(q.get("prompt", "")) for q in questions
            )
            frontier = build_frontier(
                frame, prices, slice_prompt_tokens, len(questions),
                FRAME_DEFAULT_COMPLETION_TOKENS,
            )
            logger.info(
                "Frontier: %d hull models %s, max accuracy %.3f, confidence %.2f "
                "(least-observed hull model has %.0f trials)",
                len(frontier.models), list(frontier.models), frontier.max_accuracy,
                frontier.confidence, frontier.least_trials,
            )

            # --- SCORE FROM PROOFS ---
            timer.start_phase("score")
            epoch_scores: dict[int, HeadScore] = {}
            for uid, proof in verified_proofs.items():
                scored = proof.scored_results
                # Both sides of thrift from shared data, not from the miner.
                #
                # The denominator was already built this way — pinned rates plus
                # the frame's measured verbosity — while the numerator was the
                # miner's own account of what it spent, and even the
                # denominator's prompt tokens came from the miner's report. One
                # half unfakeable, the other a self-report, and a miner could
                # lower its cost by shrinking numbers rather than by routing
                # better. Prompt tokens now come from the POOL, and the miner's
                # cost is computed from which models it chose.
                pool_prompt_tokens = sum(
                    question_input_tokens(slice_gold[r.question_id]["prompt"])
                    for r in scored if r.question_id in slice_gold
                )
                ref_cost = reference_cost(
                    frame, prices, ref_model,
                    prompt_tokens=pool_prompt_tokens,
                    n_questions=len(scored),
                    default_completion_tokens=FRAME_DEFAULT_COMPLETION_TOKENS,
                )
                head_cost = policy_cost_total(
                    [(r.question_id, r.routed_model) for r in scored],
                    slice_gold, prices, frame, FRAME_DEFAULT_COMPLETION_TOKENS,
                )
                score = _proof_to_head_score(proof, ref_cost, head_cost)
                epoch_scores[uid] = score
                cpq = score.total_head_cost / max(1, score.n_scored)
                logger.info(
                    "  UID %d: acc=%.3f cost=$%.4f ($%.6f/q, frontier %.3f there) "
                    "ref=$%.4f (%d/%d correct)",
                    uid, score.accuracy, score.total_head_cost, cpq,
                    frontier.accuracy_at(cpq), ref_cost,
                    score.n_correct, score.n_scored,
                )

            # --- DEDUP ---
            timer.start_phase("dedup_score_weight")
            routing_decisions = {
                uid: _extract_routing_decisions(proof, model_index)
                for uid, proof in verified_proofs.items()
            }
            dupes = find_duplicates(routing_decisions, commit_blocks)
            if dupes:
                logger.info("Dedup disqualified: %s", dupes)

            # --- EVIDENCE ACCUMULATION ---
            scoring_state = update_scores(
                scoring_state, epoch_scores, head_hashes,
                frontier=frontier,
                hotkeys={
                    uid: metagraph.hotkeys[uid]
                    for uid in verified_proofs
                    if uid < len(metagraph.hotkeys)
                },
                n_questions=len(questions),
                pool_size=len(benchmark_pool),
            )

            # --- WEIGHTS ---
            for uid in epoch_scores:
                rec = scoring_state.records.get(uid)
                if rec is not None:
                    logger.info(
                        "  UID %d scored: lcb=%.3f evidence $%.6f/q, frontier %.3f "
                        "there, headroom %.3f, composite %.5f",
                        uid, rec.wilson_lcb, rec.cost_per_question,
                        rec.reference_accuracy, rec.headroom, rec.composite_score,
                    )
            # The frontier's confidence is paid out and the rest burns. It is a
            # factor common to every miner, so it has to be applied AFTER the
            # scores are normalised — inside composite() it cancels.
            uids, weights = compute_weights(
                scoring_state.records, dedup_disqualified=dupes,
                paid_fraction=frontier.confidence,
            )
            if frontier.confidence < 1.0:
                logger.info(
                    "Frontier confidence %.2f: %.0f%% of emission burns to UID %d this epoch",
                    frontier.confidence, 100.0 * (1.0 - frontier.confidence), BURN_UID,
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
                uid: {
                    "accuracy": s.accuracy,
                    "quality": scoring_state.records[uid].quality,
                    "thrift": scoring_state.records[uid].thrift,
                    "headroom": scoring_state.records[uid].headroom,
                    "reference_accuracy": scoring_state.records[uid].reference_accuracy,
                    "cost_per_question": scoring_state.records[uid].cost_per_question,
                    "score": scoring_state.records[uid].composite_score,
                }
                for uid, s in epoch_scores.items()
                if uid in scoring_state.records
            }
            epoch_weight_map = dict(zip(uids, weights))

            # Observation matrix for the reveal, in the SAME global model index
            # space dedup uses, so a column means the same model in every
            # artifact. -1 means "no miner routed this question to this model",
            # which is distinct from 0 ("routed, and got it wrong") — as a plain
            # zero matrix the two were indistinguishable and any consumer would
            # read unexplored cells as failures.
            q_idx_map = {q["question_id"]: i for i, q in enumerate(questions)}
            matrix = np.full(
                (len(questions), max(len(pool_models), 1)), -1, dtype=np.int32,
            )
            model_spend: dict[str, float] = {}
            for proof in verified_proofs.values():
                for r in proof.results:
                    m_idx = model_index.get(r.routed_model)
                    q_idx = q_idx_map.get(r.question_id)
                    if m_idx is not None and q_idx is not None:
                        matrix[q_idx, m_idx] = int(r.correct)
                    model_spend[r.routed_model] = (
                        model_spend.get(r.routed_model, 0.0) + r.cost_usd
                    )

            # Exploration observations for the reveal. Sorted deterministically
            # in reveal_epoch so every validator publishes the identical list.
            exploration_for_reveal = [
                {
                    "question_id": r.question_id,
                    "model": r.routed_model,
                    "correct": bool(r.correct),
                    "prompt_tokens": int(r.prompt_tokens),
                    "completion_tokens": int(r.completion_tokens),
                }
                for proof in verified_proofs.values()
                for r in proof.exploration_results
            ]

            routing_for_reveal = {
                uid: decisions.tolist()
                for uid, decisions in routing_decisions.items()
            }
            reveal_ok = reveal_epoch(
                epoch_id, questions,
                matrix, pool_models, model_spend,
                head_hashes, routing_for_reveal,
                epoch_score_dicts, epoch_weight_map,
                exploration=exploration_for_reveal,
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
            weights_confirmed = False
            confirm_detail = "not attempted"
            if success:
                logger.info("Weights set successfully")
                prev_uids = uids
                prev_weights = weights
                weights_confirmed, confirm_detail = confirm_weights_on_chain(
                    subtensor, netuid, my_uid, uids, weights,
                )
                if weights_confirmed:
                    logger.info("Weights confirmed on chain: %s", confirm_detail)
                else:
                    logger.error(
                        "Weights NOT confirmed on chain: %s — the extrinsic "
                        "reported success but the chain disagrees",
                        confirm_detail,
                    )
            else:
                logger.warning("Weight-setting failed: %s", msg)

            timer.end_phase()
            anomalies = detect_anomalies(
                epoch_score_dicts, epoch_weight_map,
                n_neurons, len(verified_proofs),
            )
            if not reveal_ok:
                anomalies.append("commit_reveal_failed")
            for uid, why in suspicious.items():
                # In the published epoch log so it is attributable after the
                # fact by anyone, not just whoever was reading the validator's
                # console at the time.
                anomalies.append(f"implausible_exploration: uid {uid}: {why}")

            # Surfaced as an anomaly, not just a number, because this is the
            # signal that the field looks bad for reasons that are ours rather
            # than the miners'. Correlated across validators when the endpoint
            # is at fault, which is exactly what makes it diagnosable (I7).
            if collateral_budget.exhausted:
                logger.error(
                    "COLLATERAL BUDGET EXHAUSTED: this epoch spent its entire "
                    "%.0fs collateral allowance and stopped fetching. Proofs "
                    "after that point were skipped WITHOUT being checked and "
                    "are counted unverifiable, not invalid. Weights were still "
                    "set (I6) — but on a smaller verified set than the field. "
                    "Look at the PCCS before raising FUGAL_COLLATERAL_EPOCH_BUDGET.",
                    TEE_COLLATERAL_EPOCH_BUDGET,
                )

            if n_unverifiable:
                anomalies.append(
                    f"collateral_unavailable: {n_unverifiable} proof(s) unchecked"
                )

            epoch_log = EpochLog(
                epoch_id=epoch_id, block_hash=block_hash,
                timestamp=time.time(), n_questions=len(questions),
                n_miners_queried=n_neurons,
                n_heads_valid=len(verified_proofs), n_heads_invalid=n_invalid,
                n_heads_unverifiable=n_unverifiable,
                commit_hash=commitment.commit_hash,
                reveal_verified=reveal_ok,
                scores=epoch_score_dicts, weights=epoch_weight_map,
                dedup_disqualified=list(dupes),
                weight_capped=weight_capped,
                set_weights_success=success,
                set_weights_msg=str(msg) if msg else "",
                weights_confirmed_on_chain=weights_confirmed,
                weights_confirm_detail=confirm_detail,
                anomalies=anomalies,
                duration_s=timer.total_s,
            )
            write_epoch_log(epoch_log)

            last_epoch_index = epoch_index
            save_state(scoring_state.records, prev_uids, prev_weights,
                       last_epoch_index, first_commit_blocks, frame)

        except KeyboardInterrupt:
            logger.info("Validator stopped by user")
            break
        except Exception:
            logger.exception("Error in epoch")

        if once:
            break

        logger.info("Waiting for next epoch boundary (every %d blocks)...", blocks_per_epoch)
        time.sleep(60)

    # Close the dendrite's HTTP session. Without this the process keeps a
    # non-daemon worker alive and `--once` never returns — an orchestrator or
    # an operator running a single epoch waits forever on a validator that has
    # already finished and said so.
    try:
        dendrite.close_session()
    except Exception as e:
        logger.debug("Dendrite session close: %s", e)

    logger.info("Validator shutdown complete")

    if once:
        # os._exit, not sys.exit. async_substrate_interface's __del__ closes its
        # websocket during interpreter finalization and joins a thread that
        # never finishes, so a normal exit hangs forever — a --once run in a
        # systemd oneshot, a cron job, or an orchestrator would appear to run
        # indefinitely despite having completed its epoch and said so.
        #
        # Safe here because everything durable is already committed: the state
        # file was written through os.replace, the epoch log was appended and
        # closed, and the weights extrinsic was confirmed on chain above. The
        # only thing skipped is teardown of connections the OS reclaims anyway.
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _get_bundle_for_uid(uid, resp):
    """Parse a miner's inline bundle, returning (proof, head_bytes) or None.

    Everything here is miner-controlled input, so it is bounded before it is
    parsed (I2). The pydantic field caps in protocol.py are the first bound;
    HEAD_MAX_BYTES is the second, applied before the head reaches a loader.
    """
    import base64

    from fugal_subnet.config import HEAD_MAX_BYTES
    from fugal_subnet.tee.proof import BenchmarkProof

    proof_json = getattr(resp, "proof_json", "") or ""
    head_b64 = getattr(resp, "head_npz_b64", "") or ""
    if not proof_json or not head_b64:
        logger.debug("UID %d: incomplete bundle in response", uid)
        return None

    try:
        proof = BenchmarkProof.from_dict(json.loads(proof_json))
    except Exception as e:
        logger.warning("UID %d: unparseable proof: %s", uid, e)
        return None

    try:
        head_bytes = base64.b64decode(head_b64, validate=True)
    except Exception as e:
        logger.warning("UID %d: undecodable head: %s", uid, e)
        return None

    if len(head_bytes) > HEAD_MAX_BYTES:
        logger.warning("UID %d: head is %d bytes (max %d)",
                       uid, len(head_bytes), HEAD_MAX_BYTES)
        return None

    return proof, head_bytes


# Imported at module scope: wait_for_block is called from the epoch loop but
# defined here, and the nominal block time is only used to render a human
# estimate — the wait itself is on block height, never on wall clock.
from fugal_subnet.benchmarks.slicer import BLOCK_TIME_S as _BLOCK_TIME_S  # noqa: E402


def wait_for_block(subtensor, target_block: int, poll_s: int = 12) -> None:
    """Block until the chain reaches `target_block`.

    Cheap and deliberately dumb: one block-height read per poll. The wait is
    what makes the collection point deterministic, so it is not shortened on
    the grounds that a miner "looks ready" — that would reintroduce exactly the
    wall-clock dependence the fixed point exists to remove.
    """
    announced = False
    last_log = 0.0
    while True:
        try:
            current = subtensor.get_current_block()
        except Exception as e:  # noqa: BLE001 - a dropped socket must not end the epoch
            logger.warning("Could not read block height (%s); retrying", e)
            time.sleep(poll_s)
            continue
        if current >= target_block:
            return
        remaining = target_block - current
        # Say it once, then rarely. A line every minute for half an epoch is
        # noise an operator learns to scroll past, and the next thing they
        # scroll past is the one that mattered.
        now = time.monotonic()
        if not announced or now - last_log >= 300:
            logger.info("Waiting for collection block %d (%d blocks, ~%.0f min)",
                        target_block, remaining, remaining * _BLOCK_TIME_S / 60)
            announced = True
            last_log = now
        time.sleep(min(poll_s * max(1, remaining), 60))


def metagraph_at(subtensor, netuid: int, block: int, fallback):
    """Metagraph as of `block`, falling back to the live one.

    Pinning the read is what makes two validators query the same axons. A node
    that has pruned the state for that block cannot serve it, though, and a
    validator that refuses to run because it cannot read 40 blocks of history
    is worse than one that reads the current state and says so.
    """
    try:
        return subtensor.metagraph(netuid, block=block)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not read the metagraph at block %d (%s) — using the current "
            "one. Two validators reading at different blocks can see different "
            "axon sets, so a divergence this epoch has a known cause.",
            block, e,
        )
        return fallback


def confirm_weights_on_chain(subtensor, netuid, my_uid, uids, weights, tol=1e-3):
    """Read the weights back off chain and check they match what we submitted.

    `set_weights` returning success means the extrinsic was included, not that
    the chain now holds what we meant. Nothing in this repo used to check the
    difference, so "weights set successfully" was an unverified claim in every
    log and every epoch artifact.

    On a commit-reveal subnet the weights are NOT readable straight away: the
    extrinsic commits an encrypted hash and the values only appear after the
    reveal period. An immediate read finds nothing, correctly — so on those
    subnets this confirms the commit was recorded (LastUpdate advanced to this
    block) rather than pretending to read values that cannot exist yet.

    Returns (confirmed, detail). Never raises — a readback failure must not
    take down an epoch that otherwise succeeded.
    """
    try:
        if subtensor.commit_reveal_enabled(netuid=netuid):
            last_update = subtensor.query_subtensor("LastUpdate", params=[netuid])
            block = subtensor.get_current_block()
            recorded = int(last_update[my_uid])
            # Allow a few blocks of slack for inclusion.
            if block - recorded > 25:
                return False, (
                    f"commit-reveal subnet, but LastUpdate for uid {my_uid} is "
                    f"block {recorded} and the chain is at {block} — the commit "
                    "was not recorded"
                )
            return True, (
                f"commit recorded at block {recorded}; values reveal after the "
                "reveal period (commit-reveal subnet)"
            )
    except Exception as e:
        logger.debug("commit-reveal probe failed, falling back to readback: %s", e)

    try:
        on_chain = subtensor.weights(netuid=netuid)
    except Exception as e:
        return False, f"readback failed: {e}"

    row = next((w for uid, w in on_chain if uid == my_uid), None)
    if row is None:
        return False, f"no weight row on chain for validator uid {my_uid}"

    # Chain stores u16-normalized weights; compare as proportions.
    got = {int(u): float(v) for u, v in row}
    total = sum(got.values()) or 1.0
    got = {u: v / total for u, v in got.items()}
    want = dict(zip(uids, weights))

    for uid in set(want) | set(got):
        if abs(want.get(uid, 0.0) - got.get(uid, 0.0)) > tol:
            return False, (
                f"uid {uid}: submitted {want.get(uid, 0.0):.4f}, "
                f"chain has {got.get(uid, 0.0):.4f}"
            )
    return True, f"{len(got)} weights match within {tol}"


def _proof_to_head_score(proof, ref_cost, head_cost):
    """Convert a verified BenchmarkProof into a HeadScore for scoring.

    `ref_cost` is what the reference model would have cost on this same
    question set — a fact about the model pool, identical for every miner in
    the epoch and independent of how many miners are online.

    Only scored routes count. Exploration is a cost the subnet imposed, not a
    choice the miner made, so billing it against their thrift would penalise
    them for the sampling that makes everyone's scores meaningful.
    """
    from fugal_subnet.head_eval import HeadScore

    scored = proof.scored_results
    return HeadScore(
        accuracy=proof.accuracy,
        cost_efficiency=0.0,   # superseded by thrift; see scoring.composite
        kl_score=0.0,
        routing_decisions=np.array([], dtype=np.int32),
        correct_mask=np.array([r.correct for r in scored], dtype=bool),
        coverage=1.0,
        n_correct=proof.n_correct,
        n_scored=proof.n_total,
        # The POLICY's cost, computed by this validator from the pool, the
        # pinned rates and the frame. proof.scored_cost_usd is what the miner
        # says it spent; it is still checked for internal consistency because a
        # miner lying about it is worth knowing, but it decides no score.
        total_head_cost=head_cost,
        total_oracle_cost=ref_cost,
        total_kl=0.0,
    )


def _extract_routing_decisions(proof, model_index):
    """Routing decisions as a numpy array for dedup, in a GLOBAL model space.

    `model_index` maps every model in the pinned price table to a fixed index
    shared by all miners, so index 5 means the same model in every vector.

    Previously the index space was built per-proof from that miner's own routed
    models, which broke the comparison both ways: two miners each routing 100%
    to a *different* single model both produced all-zero vectors and were
    clustered as clones, while a genuine copy that re-routed a single question
    to an alphabetically-earlier model renumbered its entire vector and evaded
    detection at 96.7% identical routing.

    Ordered by question id so two miners' vectors line up element-wise
    regardless of the order results appear in their proofs.
    """
    scored = sorted(proof.scored_results, key=lambda r: r.question_id)
    return np.array(
        [model_index.get(r.routed_model, -1) for r in scored],
        dtype=np.int32,
    )


if __name__ == "__main__":
    main()
