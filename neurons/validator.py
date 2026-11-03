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

STATE_PATH = os.getenv("FUGAL_STATE_PATH", "results/success-v1/validator_state.json")


def _known_fields(cls, data: dict) -> dict:
    """Keep the keys the dataclass declares; a field added or removed in an
    upgrade must not make every existing state file unloadable."""
    names = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(set(data) - names)
    if dropped:
        logger.warning("state carries fields %s unknown to %s; ignored", dropped, cls.__name__)
    return {k: v for k, v in data.items() if k in names}


def _parse_state(raw: dict, records_cls) -> dict:
    from fugal_subnet.evidence import Evidence
    from fugal_subnet.routing_protocol import require_identity

    require_identity(raw.get("routing_protocol"))
    records = {}
    for uid, rec in raw.get("records", {}).items():
        rec = dict(rec)
        ev_data = rec.pop("evidence", None)
        record = records_cls(**_known_fields(records_cls, rec))
        if ev_data is not None:
            record.evidence = Evidence(**_known_fields(Evidence, ev_data))
        records[int(uid)] = record
    return {
        "records": records,
        "prev_uids": [int(u) for u in raw.get("prev_uids", [])],
        "prev_weights": [float(w) for w in raw.get("prev_weights", [])],
        "last_epoch_index": int(raw.get("last_epoch_index", -1)),
        # "first_commit_blocks" was persisted here until 2026-10-14. It is
        # ignored on load: seniority is now derived from chain state at
        # the boundary block (commitments.seniority_key), not remembered.
        "frame": raw.get("frame"),
    }


def load_state(records_cls, path: str | None = None) -> dict:
    """Load persisted validator state across supervisor-managed restarts.

    A truncated or unparseable file (a power loss between the write and the
    rename used to be able to leave one) falls back to the previous epoch's
    copy, `<path>.bak`, with a loud warning: one epoch of stale evidence is a
    small divergence, and deleting the file was the only recovery before,
    which is a large one (I1).
    """
    path = path or STATE_PATH
    try:
        with open(path) as f:
            raw = json.load(f)
        return _parse_state(raw, records_cls)
    except FileNotFoundError:
        return _empty_state()
    except Exception as e:
        bak = path + ".bak"
        if os.path.exists(bak):
            logger.error(
                "Validator state %s is unreadable (%s); resuming from the previous "
                "epoch's copy %s. This validator is one epoch behind its peers until "
                "it catches up.", path, e, bak,
            )
            with open(bak) as f:
                return _parse_state(json.load(f), records_cls)
        raise ValueError(f"Cannot resume validator state {path}: {e}") from e


def _empty_state() -> dict:
    return {"records": {}, "prev_uids": [], "prev_weights": [],
            "last_epoch_index": -1, "frame": None}


def save_state(records: dict, prev_uids: list[int], prev_weights: list[float],
               last_epoch_index: int, frame=None, path: str | None = None):
    """Persist validator state: write, fsync, keep the previous copy, rename."""
    path = path or STATE_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    from fugal_subnet.routing_protocol import identity
    payload = {
        "routing_protocol": identity(),
        "records": {str(uid): dataclasses.asdict(rec) for uid, rec in records.items()},
        "prev_uids": prev_uids,
        "prev_weights": prev_weights,
        "last_epoch_index": last_epoch_index,
        "frame": frame.to_dict() if frame is not None else None,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())      # the rename is atomic; the bytes were not
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)


def axon_gate_notice(stake: float, permit: bool, threshold: float) -> str | None:
    """Startup warning when this validator's own stake is below the floor live
    miners apply before answering a query (`neurons/miner.py`, `caller_allowed`).
    A refused query is indistinguishable from an absent miner on this side, so
    the one place it can be said is here, before the first epoch."""
    if stake >= threshold:
        return None
    permit_note = (" A validator permit does not substitute for it."
                   if permit else "")
    return (
        f"This hotkey's stake on the subnet is {stake:.0f} (chain total_stake: "
        f"alpha plus 0.18 x root TAO, childkey-inherited), below the "
        f"{threshold:.0f} that live miners require before answering a query "
        f"(FUGAL_MIN_VALIDATOR_STAKE in the miner image).{permit_note} Until it "
        "is staked you receive no proofs and see 'No valid proofs received' "
        "every epoch with no error on either side."
    )


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
    default=True,
    help="Live (default) requires real TDX attestation; --mock accepts unattested "
         "proofs and is refused on finney",
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
    # A bare base entry is REFUSED in live mode, not warned about. The
    # application is what fixes the model upstream; a bare base approves a TD
    # whose upstream the miner chose, which is perfect accuracy at near-zero
    # attested cost with every signature intact (diligence 2026-10, finding
    # 1). It used to be a warning, and a warning is a request addressed to the
    # operator about something the adversary controls.
    if TEE_APPROVED_MEASUREMENTS:
        from fugal_subnet.tee.verify import parse_approved
        try:
            parsed_entries = parse_approved(TEE_APPROVED_MEASUREMENTS, require_app=live)
        except ValueError as e:
            raise click.ClickException(f"FUGAL_TEE_MEASUREMENTS: {e}") from e
        described = [
            f"{base[:12]}…:{app[:12]}…" if app else f"{base[:12]}… (bare base: any app)"
            for base, apps in sorted(parsed_entries.items())
            for app in (sorted(apps) or [""])
        ]
        logger.info("Approved entries: %d — %s", len(described), "; ".join(described))

    from fugal_subnet.fingerprint import assert_environment, consensus_digest
    if not mock:
        from fugal_subnet.routing_protocol import MANIFEST_PATH
        from fugal_subnet.vendor import success_contract as contract
        contract.validate_manifest(json.loads(MANIFEST_PATH.read_text()), deployable=True)
    # Consensus overrides are checked against the chain's genesis hash once the
    # chain is connected (below): refused on finney, warned about elsewhere.
    assert_environment(strict=live, consensus_overrides=False)
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
    from fugal_subnet.commitments import seniority_key
    from fugal_subnet.config import TEE_COLLATERAL_EPOCH_BUDGET
    from fugal_subnet.dedup import epsilon_disqualified, find_duplicates
    from fugal_subnet.epoch_logger import (
        EpochLog,
        EpochTimer,
        detect_anomalies,
        write_epoch_log,
    )
    from fugal_subnet.exploration import expected_exploration as expected_exploration_map
    from fugal_subnet.frontier import build_frontier
    from fugal_subnet.head_eval import HeadScore, load_head_from_npz
    from fugal_subnet.pricing import policy_cost_total, question_input_tokens
    from fugal_subnet.protocol import FugalProofSynapse
    from fugal_subnet.reference_frame import (
        Observation,
        ReferenceFrame,
        accumulate_exploration,
        aggregate_observations,
        best_model,
        disagreeing_exploration,
        implausible_exploration,
        load_bootstrap,
        rebuild_from_reveals,
        reference_cost,
    )
    from fugal_subnet.rewards import cap_weight_change, compute_weights
    from fugal_subnet.scoring import MinerRecord, ScoringState, update_scores
    from fugal_subnet.tee.attestation import CollateralBudget, verification_order
    from fugal_subnet.tee.proof import BenchmarkProof
    from fugal_subnet.tee.verify import compute_questions_hash, verify_proof

    wallet = bt.Wallet(name=coldkey, hotkey=hotkey, path=wallet_path)
    from fugal_subnet.config import ARCHIVE_ENDPOINTS, FALLBACK_ENDPOINTS
    subtensor = bt.Subtensor(
        network=network,
        archive_endpoints=ARCHIVE_ENDPOINTS or None,
        fallback_endpoints=FALLBACK_ENDPOINTS or None,
    )
    if live and not ARCHIVE_ENDPOINTS:
        logger.warning(
            "FUGAL_ARCHIVE_ENDPOINTS is unset. The validator pins reads to the "
            "boundary and collection blocks; a non-archive node keeps ~300 blocks "
            "and refuses older ones after a late start or a retried epoch."
        )
    from fugal_subnet.fingerprint import (
        assert_consensus_overrides_allowed,
        assert_mock_allowed,
        chain_genesis_hash,
    )
    try:
        genesis = chain_genesis_hash(subtensor)
        assert_mock_allowed(genesis, mock)
        assert_consensus_overrides_allowed(genesis)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e
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
        assert_chain_weight_parameters(subtensor, netuid, live=live)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e

    # Reveal publishing: the bucket is operational configuration; the hold is
    # the chain's commit-reveal period unless overridden.
    from fugal_subnet.config import REVEAL_BUCKET, REVEAL_DELAY_EPOCHS
    from fugal_subnet.reveal_publisher import GcsUploader, delay_epochs_for
    reveal_uploader = GcsUploader(REVEAL_BUCKET) if REVEAL_BUCKET else None
    reveal_delay = REVEAL_DELAY_EPOCHS or delay_epochs_for(subtensor, netuid)
    if reveal_uploader is None:
        (logger.warning if live else logger.info)(
            "FUGAL_REVEAL_BUCKET is unset: epoch reveals stay on local disk and are "
            "not published. Set it to a public-read GCS bucket (docs/VALIDATOR_GUIDE.md)."
        )
    else:
        logger.info("Reveals publish to gs://%s after %d epoch(s)", REVEAL_BUCKET, reveal_delay)
    own_permit = False
    try:
        own_permit = bool(metagraph.validator_permit[my_uid])
        if not own_permit:
            logger.warning("This hotkey has NO validator permit — set_weights will "
                           "likely fail until it gains enough stake.")
    except (AttributeError, IndexError):
        pass
    try:
        from fugal_subnet.config import MIN_VALIDATOR_STAKE
        notice = axon_gate_notice(float(metagraph.S[my_uid]), own_permit, MIN_VALIDATOR_STAKE)
        if notice:
            (logger.warning if live else logger.info)(notice)
    except (AttributeError, IndexError, TypeError, ValueError):
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
        # Live requires the shipped frame: every validator must value the
        # frontier from the same evidence, and "start cold" is a different
        # frontier from everyone else's (I1). Mock may start empty.
        try:
            frame = load_bootstrap(require=live)
        except FileNotFoundError as e:
            raise click.ClickException(str(e)) from e
        logger.warning(
            "Starting from the bootstrap frame with no accumulated history. If "
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

            # --- COMMITMENTS, pinned to the boundary block ---
            # Read NOW, while the boundary block is seconds old, not after the
            # collection wait and the 120 s query: a non-archive node serves
            # roughly the last 300 blocks, and at ~boundary+250 the pinned read
            # started failing with 50 blocks to spare. The pin makes the answer
            # identical whenever it is read, so reading early costs nothing.
            # In live a failed read is the epoch's failure (retried by the
            # outer loop), never an empty map: with REQUIRE_COMMITMENT an empty
            # map rejects every proof, and the epoch was then marked processed
            # while peers that read in time set weights — a fork with nothing
            # in the reveal to explain it (found 2026-11-02).
            timer.start_phase("commitments")
            chain_commitments = read_commitments_or_fail(
                subtensor, netuid, boundary_block,
                required=live and REQUIRE_COMMITMENT,
            )

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

            # --- VERIFY PROOFS ---  (commitments were read at the boundary, above)
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
                    # Seniority from chain state at the boundary: the hotkey's
                    # registration block first, its commitment block second.
                    # A retrained author keeps rank over whoever copied its
                    # previous head, and every validator derives the same
                    # order (commitments.seniority_key).
                    commit_blocks[uid] = seniority_key(
                        registration_block_of(metagraph, uid), committed[1],
                    )
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
                           last_epoch_index, frame)
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
            # One Observation per answer, voted by COLDKEY and collapsed to one
            # sample per (question, model) by aggregate_observations: failed
            # calls dropped, one vote per operator, median across operators.
            # Every miner answers the same questions, so without this each
            # copy was a fresh trial and a coalition moved the frame by its
            # share of the field (diligence 2026-10, finding 2).
            coldkeys = getattr(metagraph, "coldkeys", None) or []
            observations = [
                Observation(
                    question_id=r.question_id, model=r.routed_model,
                    correct=bool(r.correct), prompt_tokens=int(r.prompt_tokens),
                    completion_tokens=int(r.completion_tokens), failed=bool(r.failed),
                    coldkey=str(coldkeys[uid]) if uid < len(coldkeys) else f"uid-{uid}",
                )
                for uid, proof in verified_proofs.items()
                for r in proof.exploration_results
            ]
            # Detection BEFORE folding this epoch's samples in, so a miner is
            # compared against history rather than against a frame it has
            # just contributed to. Two detectors: accuracy implausible for the
            # nonce-chosen models in either direction, and disagreement with
            # the other operators on the same questions. The only consequence
            # is that a flagged operator's samples stay OUT of the frame this
            # epoch. Never a score: I4 says a miner's score cannot depend on
            # other miners, and the frame is the one thing built from everyone
            # — keeping a flagged miner's samples out of it protects every
            # other miner's reference rather than moving anyone's score
            # (diligence 2026-10, finding 4).
            coldkey_of = {
                uid: str(coldkeys[uid]) if uid < len(coldkeys) else f"uid-{uid}"
                for uid in verified_proofs
            }
            suspicious = implausible_exploration(frame, {
                uid: [(r.routed_model, r.correct) for r in proof.exploration_results]
                for uid, proof in verified_proofs.items()
            })
            suspicious.update(disagreeing_exploration({
                uid: [(r.question_id, r.routed_model, r.correct)
                      for r in proof.exploration_results if not r.failed]
                for uid, proof in verified_proofs.items()
            }, coldkey_of))
            for uid, why in sorted(suspicious.items()):
                logger.warning(
                    "UID %d exploration excluded from the reference frame this "
                    "epoch: %s. Its score is untouched (I4).", uid, why,
                )
            samples = aggregate_observations(
                observations,
                exclude_coldkeys={coldkey_of[uid] for uid in suspicious},
            )

            # EPOCH-LOCAL from here to save_state. `frame`, `scoring_state`,
            # `prev_uids` and `prev_weights` are the validator's durable state
            # and are rebound only once the epoch is written. Anything that
            # raised between scoring and save_state used to land in the outer
            # handler with the loop-level names already advanced, so the retry
            # of the SAME epoch accumulated every miner's evidence twice,
            # applied every miss twice and decayed the frame twice — and then
            # persisted it (found 2026-11-02). tests/test_epoch_atomicity.py
            # holds this shape.
            epoch_frame = accumulate_exploration(frame, samples)
            ref_model, acc_best = best_model(epoch_frame, prices)
            logger.info(
                "Reference frame: %d samples this epoch, best model %s "
                "(acc=%.3f, %.0f trials)",
                len(samples), ref_model, acc_best, epoch_frame.trials.get(ref_model, 0.0),
            )
            # The constant-policy frontier for THIS slice: what any single model,
            # or any random mixture of models, buys at each price. Prompt tokens
            # come from the pool, so the frontier is identical for every miner
            # and on every validator. See fugal_subnet/frontier.py.
            slice_prompt_tokens = sum(
                question_input_tokens(q.get("prompt", "")) for q in questions
            )
            frontier = build_frontier(
                epoch_frame, prices, slice_prompt_tokens, len(questions),
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
                    epoch_frame, prices, ref_model,
                    prompt_tokens=pool_prompt_tokens,
                    n_questions=len(scored),
                    default_completion_tokens=FRAME_DEFAULT_COMPLETION_TOKENS,
                )
                head_cost = policy_cost_total(
                    [(r.question_id, r.routed_model) for r in scored],
                    slice_gold, prices, epoch_frame, FRAME_DEFAULT_COMPLETION_TOKENS,
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
            # The parameter clause needs the heads themselves; they were
            # validated in verify_proof and are loaded here only for distance.
            head_params = {}
            for uid, head_bytes in verified_heads.items():
                try:
                    head_params[uid] = load_head_from_npz(head_bytes)
                except (ValueError, KeyError, TypeError):
                    continue
            dupes = find_duplicates(routing_decisions, commit_blocks, head_params=head_params)
            if dupes:
                logger.info("Dedup disqualified: %s", dupes)

            # --- EVIDENCE ACCUMULATION ---
            epoch_state = update_scores(
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
                rec = epoch_state.records.get(uid)
                if rec is not None:
                    logger.info(
                        "  UID %d scored: lcb=%.3f evidence $%.6f/q, frontier %.3f "
                        "there, headroom %.3f, composite %.5f",
                        uid, rec.wilson_lcb, rec.cost_per_question,
                        rec.reference_accuracy, rec.headroom, rec.composite_score,
                    )
            # Epsilon rule, now that this epoch's composite scores exist: a
            # head that routes like a senior one must beat it or it is a copy.
            near_copies = epsilon_disqualified(
                routing_decisions, commit_blocks,
                {uid: epoch_state.records[uid].composite_score
                 for uid in routing_decisions if uid in epoch_state.records},
            )
            if near_copies:
                logger.info("Epsilon rule disqualified: %s", sorted(near_copies))
                dupes = set(dupes) | near_copies
            # The frontier's confidence is paid out and the rest burns. It is a
            # factor common to every miner, so it has to be applied AFTER the
            # scores are normalised — inside composite() it cancels.
            burn_uid = resolve_burn_uid(subtensor, netuid, boundary_block, metagraph, live)
            uids, weights = compute_weights(
                epoch_state.records, dedup_disqualified=dupes,
                paid_fraction=frontier.confidence, burn_uid=burn_uid,
            )
            if frontier.confidence < 1.0:
                logger.info(
                    "Frontier confidence %.2f: %.0f%% of emission burns to UID %d this epoch",
                    frontier.confidence, 100.0 * (1.0 - frontier.confidence), burn_uid,
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
                    "quality": epoch_state.records[uid].quality,
                    "thrift": epoch_state.records[uid].thrift,
                    "headroom": epoch_state.records[uid].headroom,
                    "reference_accuracy": epoch_state.records[uid].reference_accuracy,
                    "cost_per_question": epoch_state.records[uid].cost_per_question,
                    "score": epoch_state.records[uid].composite_score,
                }
                for uid, s in epoch_scores.items()
                if uid in epoch_state.records
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
                    "question_id": o.question_id,
                    "model": o.model,
                    "correct": o.correct,
                    "prompt_tokens": o.prompt_tokens,
                    "completion_tokens": o.completion_tokens,
                    # The voter and the failure flag, so rebuild_from_reveals
                    # aggregates exactly as this epoch did. Coldkeys are public
                    # on chain; publishing them here reveals nothing new.
                    "coldkey": o.coldkey,
                    "failed": o.failed,
                }
                for o in observations
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
            success, msg = set_weights_with_retry(
                subtensor, wallet, netuid, my_uid, uids, weights,
                deadline_block=(epoch_index + 1) * blocks_per_epoch - 5,
            )
            weights_confirmed = False
            confirm_detail = "not attempted"
            next_prev_uids, next_prev_weights = prev_uids, prev_weights
            if success:
                logger.info("Weights set successfully")
                next_prev_uids, next_prev_weights = uids, weights
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

            # --- PUBLISH REVEALS ---
            # Reveals old enough to be past the weight commit-reveal window go
            # to the public store, newest hash on chain. Operational: a failure
            # here is logged and retried, never an epoch lost.
            if reveal_uploader is not None:
                from fugal_subnet.commit_reveal import EPOCH_DIR
                from fugal_subnet.reveal_publisher import publish_due_reveals
                try:
                    publish_due_reveals(
                        EPOCH_DIR, epoch_index, reveal_delay, netuid, reveal_uploader,
                        commit=lambda data: subtensor.set_commitment(wallet, netuid, data)[0],
                    )
                except Exception as e:  # noqa: BLE001 - publishing never takes the epoch down
                    logger.warning("Reveal publishing failed this epoch: %s", e)

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
                anomalies.append(f"exploration_excluded_from_frame: uid {uid}: {why}")

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

            # COMMIT. The epoch is complete; only now does it become the state
            # the next epoch (or a retry of this one) starts from.
            scoring_state, frame = epoch_state, epoch_frame
            prev_uids, prev_weights = next_prev_uids, next_prev_weights
            last_epoch_index = epoch_index
            save_state(scoring_state.records, prev_uids, prev_weights,
                       last_epoch_index, frame)

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


def assert_chain_weight_parameters(subtensor, netuid: int, *, live: bool) -> dict:
    """Refuse (live) or warn (mock) when the chain would silently drop our weights.

    Three parameters decide whether a set_weights that the SDK reports as
    submitted ever reaches Yuma, and none of them is visible in the result:
    the chain's commit-reveal VERSION must equal the one the SDK commits with
    (`IncorrectCommitRevealVersion` surfaces only at reveal), the SDK's
    `version_key` must be >= the subnet's `weights_version`, and the vector
    must satisfy `min_allowed_weights` / `max_weights_limit`. Read once at
    startup and logged; the first two are refusals in live, because a
    validator that commits every epoch into a chain that discards every reveal
    looks perfectly healthy from its own logs (validator review 2026-11-02,
    findings 3 and 4).
    """
    import inspect

    out: dict = {}
    sdk_crv = inspect.signature(subtensor.set_weights).parameters["commit_reveal_version"].default
    sdk_vk = inspect.signature(subtensor.set_weights).parameters["version_key"].default
    try:
        chain_crv = subtensor.query_subtensor("CommitRevealWeightsVersion")
        chain_crv = int(getattr(chain_crv, "value", chain_crv))
    except Exception as e:  # noqa: BLE001 - an unreadable value is reported, not guessed
        chain_crv = None
        logger.warning("Could not read CommitRevealWeightsVersion: %s", e)
    try:
        hp = subtensor.get_subnet_hyperparameters(netuid=netuid)
        out = {
            "commit_reveal_enabled": bool(getattr(hp, "commit_reveal_weights_enabled", False)),
            "commit_reveal_period": getattr(hp, "commit_reveal_period", None),
            "weights_rate_limit": getattr(hp, "weights_rate_limit", None),
            "min_allowed_weights": getattr(hp, "min_allowed_weights", None),
            "max_weights_limit": getattr(hp, "max_weights_limit", None),
            "weights_version": getattr(hp, "weights_version", None),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read subnet hyperparameters: %s", e)
    out["chain_commit_reveal_version"] = chain_crv
    out["sdk_commit_reveal_version"] = sdk_crv
    logger.info("Chain weight parameters: %s", out)

    problems = []
    if chain_crv is not None and out.get("commit_reveal_enabled") and chain_crv != sdk_crv:
        problems.append(
            f"chain CommitRevealWeightsVersion is {chain_crv} but this SDK commits "
            f"version {sdk_crv}; every reveal would be rejected"
        )
    wv = out.get("weights_version")
    if wv is not None and int(wv) > int(sdk_vk):
        problems.append(
            f"subnet weights_version {wv} exceeds the SDK version_key {sdk_vk}; "
            "set_weights would be refused"
        )
    if problems:
        msg = "; ".join(problems)
        if live:
            raise RuntimeError(msg)
        logger.warning("%s (continuing: mock)", msg)
    return out


def set_weights_with_retry(subtensor, wallet, netuid: int, my_uid: int, uids, weights,
                           *, deadline_block: int, attempts: int | None = None,
                           sleep_s: float = 12.0) -> tuple[bool, str]:
    """Submit weights, naming the reason when the SDK will not, and retry
    while the epoch has blocks left.

    The SDK's set_weights returns `(False, None)` without attempting anything
    when the chain's rate limit has not elapsed, and a transient websocket
    error becomes `(False, <exception text>)` after its own five attempts. A
    failure here used to be logged and the epoch marked done, so one blip cost
    the epoch's weights (validator review 2026-11-02, finding 3). Now the rate
    limit is read first so the log says why, and a failed submit is retried up
    to WEIGHTS_SET_RETRIES times while `deadline_block` has not passed.
    """
    from fugal_subnet.config import WEIGHTS_SET_RETRIES

    attempts = WEIGHTS_SET_RETRIES if attempts is None else attempts
    msg = ""
    for attempt in range(1, max(1, attempts) + 1):
        try:
            since = int(subtensor.blocks_since_last_update(netuid, my_uid))
            limit = int(subtensor.weights_rate_limit(netuid))
        except Exception as e:  # noqa: BLE001 - diagnostic only
            since, limit = -1, -1
            logger.warning("Could not read the weights rate limit: %s", e)
        if 0 <= since <= limit:
            return False, (f"rate limit: {since} blocks since this UID's last update, "
                           f"limit {limit}; the chain would refuse the extrinsic")
        response = subtensor.set_weights(
            wallet=wallet, netuid=netuid, uids=uids, weights=weights,
            wait_for_inclusion=True, wait_for_finalization=True,
        )
        success, msg = response
        if success:
            return True, str(msg or "")
        msg = str(msg or "submit returned no message")
        try:
            now = int(subtensor.get_current_block())
        except Exception:  # noqa: BLE001
            now = deadline_block
        if attempt >= attempts or now >= deadline_block:
            break
        logger.warning("set_weights attempt %d/%d failed (%s); retrying in %.0fs "
                       "(block %d, deadline %d)", attempt, attempts, msg, sleep_s, now, deadline_block)
        time.sleep(sleep_s)
    return False, msg


def read_commitments_or_fail(subtensor, netuid: int, block: int, *, required: bool):
    """The epoch's on-chain commitments, pinned to `block`.

    `required` is `live and REQUIRE_COMMITMENT`: then a failed read raises,
    because an empty map under REQUIRE_COMMITMENT rejects every proof and the
    epoch would be marked processed with no weights set while peers that read
    in time set theirs (I1, I6). Raising hands the epoch to the outer loop's
    retry instead. When commitments are not required (mock, local chains whose
    node has no Commitments pallet), the old behaviour stands: warn and go on.

    The commonest cause of a failure here is a non-archive endpoint asked for a
    block more than ~300 behind the head, so the message says so.
    """
    from fugal_subnet.commitments import get_commitments_with_blocks

    try:
        commitments = get_commitments_with_blocks(subtensor, netuid, block=block)
    except Exception as e:  # noqa: BLE001 - chain/SDK errors of every shape
        if required:
            raise RuntimeError(
                f"Could not read on-chain commitments at block {block}: {e}. "
                "Commitments are required, so this epoch cannot be scored "
                "without them; retrying rather than scoring every miner as "
                "uncommitted. If the block is far behind the head, the "
                "endpoint is not an archive node."
            ) from e
        logger.warning("Could not read on-chain commitments at block %d: %s", block, e)
        return {}
    logger.info("On-chain commitments at block %d: %d hotkeys", block, len(commitments))
    return commitments


def wait_for_block(subtensor, target_block: int, poll_s: int = 12) -> None:
    """Block until the chain reaches `target_block`.

    Cheap and deliberately dumb: one block-height read per poll. The wait is
    what makes the collection point deterministic, so it is not shortened on
    the grounds that a miner "looks ready" — that would reintroduce exactly the
    wall-clock dependence the fixed point exists to remove.

    How long to sleep between reads is sized from the block rate this chain is
    measured to have, never the nominal one. Sized from the nominal 12 s, an
    11-block wait on a devnet producing five blocks a second slept a minute,
    woke 240 blocks late in the *next* epoch, and asked every miner for a proof
    they had already replaced — logged as "no valid proofs", as though the
    miners were at fault.
    """
    announced = False
    last_log = 0.0
    seen: tuple[float, int] | None = None  # first (monotonic time, height) read
    s_per_block: float | None = None
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
        if seen is None:
            seen = (time.monotonic(), current)
        elif current > seen[1]:
            s_per_block = (time.monotonic() - seen[0]) / (current - seen[1])
        # Say it once, then rarely. A line every minute for half an epoch is
        # noise an operator learns to scroll past, and the next thing they
        # scroll past is the one that mattered.
        now = time.monotonic()
        if not announced or now - last_log >= 300:
            logger.info("Waiting for collection block %d (%d blocks, ~%.0f min)",
                        target_block, remaining, remaining * _BLOCK_TIME_S / 60)
            announced = True
            last_log = now
        if s_per_block is None:
            # No rate yet: a short first sleep costs one extra read on a slow
            # chain and is the only safe guess on a fast one.
            time.sleep(1.0)
        else:
            time.sleep(min(max(s_per_block * remaining, 0.2), 60))


def resolve_burn_uid(subtensor, netuid: int, block: int, metagraph, live: bool) -> int:
    """The subnet owner's UID at `block`, where unassigned weight burns.

    Read from chain each epoch (`get_subnet_owner_hotkey`, pinned to the
    boundary block so every validator resolves the same UID) and mapped
    through the pinned metagraph. It used to be a hard-coded 0 (diligence
    2026-10, finding A5). Live refuses to guess: a failed lookup raises,
    because burning to the wrong UID pays someone. Mock falls back to 0 for
    local chains whose mock subtensor has no owner query.
    """
    owner = None
    try:
        owner = subtensor.get_subnet_owner_hotkey(netuid, block=block)
    except Exception as e:  # noqa: BLE001 - reported below, never swallowed
        if live:
            raise RuntimeError(f"Could not read the subnet owner hotkey at block {block}: {e}") from e
        logger.warning("Could not read the subnet owner hotkey (%s); mock burns to UID 0", e)
    hotkeys = list(getattr(metagraph, "hotkeys", []) or [])
    if owner and owner in hotkeys:
        return hotkeys.index(owner)
    if live:
        raise RuntimeError(
            f"Subnet owner hotkey {owner!r} is not registered on netuid {netuid} at block "
            f"{block}; refusing to burn to a guessed UID"
        )
    return 0


def registration_block_of(metagraph, uid: int) -> int:
    """The block this uid's hotkey registered at, from the pinned metagraph.

    0 when the metagraph does not carry it (a mock), which makes the
    commitment block the whole key -- the pre-2026-10 behaviour, and still
    deterministic for every validator reading the same chain.
    """
    blocks = getattr(metagraph, "block_at_registration", None)
    try:
        return int(blocks[uid]) if blocks is not None and uid < len(blocks) else 0
    except (TypeError, ValueError):
        return 0


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
