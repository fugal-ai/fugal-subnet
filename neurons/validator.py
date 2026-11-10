#!/usr/bin/env python3
"""Fugal subnet validator — TEE proof verification and weight-setting.

Usage:
    python neurons/validator.py --netuid 1 --coldkey fugal_validator --hotkey default
    python neurons/validator.py --netuid 1 --coldkey fugal_validator --hotkey default --once

Validators never call models. Each epoch:
1. Read head commitments at the boundary; derive the nonce; select the slice
   (protocol v1 or v2 by the boundary, slicer.protocol_version_for_boundary)
2. Query miners for TEE-attested proofs (FugalProofSynapse) at the collection block
3. Verify each proof (commitment, bundle, DCAP, approved entry, nonce, slice,
   exploration, head and cost bindings)
4. Score: v1, evidence and the frontier composite; v2, shared coldkey-median
   grades, the paired comparison window, the champion seat and the 85/10/5 payout
5. Weights (set on chain, commit-reveal aware), the reveal, state saved atomically

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
from typing import NamedTuple

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
        "v2": raw.get("v2"),
    }


def load_state(records_cls, path: str | None = None) -> dict:
    """Load persisted validator state across supervisor-managed restarts.

    A truncated or unparseable file (a power loss between the write and the
    rename used to be able to leave one) falls back to the previous epoch's
    copy, `<path>.bak`, with a loud warning: one epoch of stale evidence is a
    small divergence, and deleting the file was the only recovery before,
    which is a large one (I1).

    A MISSING file means a save was interrupted: save_state writes and fsyncs
    `<path>.tmp`, moves the current file to `.bak`, then renames `.tmp` in.
    Between the two renames `.tmp` already holds the epoch just applied (its
    weights submitted, its v2 window written), so it resumes from `.tmp`: the
    newest complete state, not one epoch behind. Only if `.tmp` cannot be
    loaded does it fall back to `.bak`. Starting fresh there would be the large
    divergence (empty evidence, bootstrap frame). A deliberate fresh start
    moves all three files aside; only none of them is a first start.
    """
    path = path or STATE_PATH
    bak = path + ".bak"
    try:
        with open(path) as f:
            raw = json.load(f)
        return _parse_state(raw, records_cls)
    except FileNotFoundError:
        tmp = path + ".tmp"
        if os.path.exists(tmp):
            try:
                with open(tmp) as f:
                    state = _parse_state(json.load(f), records_cls)
            except Exception as e:  # noqa: BLE001 - fall back to the previous copy below
                logger.error("Validator state %s is missing and the interrupted save %s "
                             "cannot be loaded (%s)", path, tmp, e)
            else:
                logger.warning(
                    "Validator state %s is missing: a save was interrupted between its two "
                    "renames. Resuming from the completed save %s (the latest epoch).",
                    path, tmp,
                )
                return state
        if not os.path.exists(bak):
            return _empty_state()
        problem = "missing (a save was interrupted between its two renames)"
        cause = None
    except Exception as e:
        if not os.path.exists(bak):
            raise ValueError(f"Cannot resume validator state {path}: {e}") from e
        problem, cause = f"unreadable ({e})", e
    logger.error(
        "Validator state %s is %s; resuming from the previous epoch's copy %s. This "
        "validator is one epoch behind its peers until it catches up. To start fresh "
        "on purpose, move it, its .bak and its .tmp aside.", path, problem, bak,
    )
    try:
        with open(bak) as f:
            return _parse_state(json.load(f), records_cls)
    except Exception as e:
        raise ValueError(
            f"Cannot resume validator state: {path} is {problem} and its previous copy "
            f"{bak} cannot be loaded either ({e}). To start fresh on purpose, move "
            "it, its .bak and its .tmp aside."
        ) from (cause or e)


def _empty_state() -> dict:
    return {"records": {}, "prev_uids": [], "prev_weights": [],
            "last_epoch_index": -1, "frame": None}


def save_state(records: dict, prev_uids: list[int], prev_weights: list[float],
               last_epoch_index: int, frame=None, path: str | None = None,
               v2: dict | None = None, v2_window=None):
    """Persist validator state: write, fsync, keep the previous copy, rename.

    The v2 window is written first, beside the JSON, so a crash between the
    two leaves a window one epoch ahead at worst, never a state file pointing
    at a window that does not exist.
    """
    path = path or STATE_PATH
    if v2_window is not None:
        from fugal_subnet.v2_epoch import save_window
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_window(path + ".v2window.npz", v2_window)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    from fugal_subnet.routing_protocol import identity
    payload = {
        "routing_protocol": identity(),
        "records": {str(uid): dataclasses.asdict(rec) for uid, rec in records.items()},
        "prev_uids": prev_uids,
        "prev_weights": prev_weights,
        "last_epoch_index": last_epoch_index,
        "frame": frame.to_dict() if frame is not None else None,
        "v2": v2,
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


def _check_live_configuration(live: bool, approved, activation_block, epoch_interval: int,
                              category_map) -> None:
    """Refuse, before any chain connection, a configuration that would make this
    validator score differently from its peers or score everyone zero.

    - --live needs FUGAL_TEE_MEASUREMENTS. With no approved entry every proof
      fails the runtime-image check and every miner is set to zero weight:
      fail-closed is right for the check, but silently deweighting the subnet
      is not how to learn the variable is unset.
    - The entries are parsed and LOGGED, so a rotation's two validators can be
      compared from their logs, and a malformed entry (a bare trailing colon
      would silently accept any application) fails here rather than at the
      first proof. A bare base is refused in --live: the application fixes the
      model upstream, so a bare base approves a TD whose upstream the miner
      chose (diligence 2026-10, finding 1).
    - The activation block must be an epoch boundary (docs/INVARIANTS.md, "I1 —
      protocol v2 activates at one named boundary"), and v2 needs its category
      map, without which no question is "hard" and the hard weight never applies.
    - --live needs a REVIEWED token manifest for every protocol it can score:
      v1's always, v2's once the activation block is set (the v2 one was not
      checked; a candidate from a re-measure would have been scored).
    """
    if live and not approved:
        raise click.ClickException(
            "--live requires FUGAL_TEE_MEASUREMENTS. With no approved "
            "measurement every proof fails the runtime-image check, so this "
            "validator would score every miner zero and set weights to match. "
            "Get the approved entry from the signed image release: "
            "`scripts/fetch_approved_entries.py --tag <image tag> --network "
            "<finney|test>` (docs/APPROVED_ENTRY_ROTATION.md)."
        )
    if approved:
        from fugal_subnet.tee.verify import parse_approved
        try:
            parsed = parse_approved(approved, require_app=live)
        except ValueError as e:
            raise click.ClickException(f"FUGAL_TEE_MEASUREMENTS: {e}") from e
        described = [
            f"{base[:12]}…:{app[:12]}…" if app else f"{base[:12]}… (bare base: any app)"
            for base, apps in sorted(parsed.items())
            for app in (sorted(apps) or [""])
        ]
        logger.info("Approved entries: %d — %s", len(described), "; ".join(described))

    from fugal_subnet.benchmarks.slicer import blocks_per_epoch, validate_activation_block
    bpe = blocks_per_epoch(epoch_interval)
    try:
        validate_activation_block(activation_block, bpe)
    except ValueError as e:
        raise click.ClickException(f"FUGAL_V2_ACTIVATION_BLOCK: {e}") from e
    if activation_block is not None and not category_map:
        msg = ("FUGAL_V2_ACTIVATION_BLOCK is set but config.BENCHMARK_CATEGORY_V2 is "
               "missing: v2 cannot tell hard questions apart")
        if live:
            raise click.ClickException(msg)
        logger.warning(msg)
    if activation_block is None:
        logger.info("Protocol v2 activation: not set; v1 rules apply to every epoch")
    else:
        logger.info("Protocol v2 activation: boundary block %d (epoch index %d); v1 rules "
                    "before it", activation_block, activation_block // bpe)

    if live:
        from fugal_subnet.routing_protocol import manifest_path
        from fugal_subnet.vendor import success_contract as contract
        for version in ((1, 2) if activation_block is not None else (1,)):
            try:
                contract.validate_manifest(json.loads(manifest_path(version).read_text()),
                                           deployable=True)
            except ValueError as e:
                raise click.ClickException(
                    f"protocol v{version} token manifest is not deployable: {e}") from e


def _epoch_to_run(current_block: int, blocks_per_epoch: int, last_epoch_index: int,
                  once: bool) -> int | None:
    """The epoch to run at `current_block`, or None to wait for the next one.

    Each epoch is scored once: running one already written again would fold
    its evidence, its misses and the frame's decay in a second time. `--once`
    is the exception, an operator's deliberate re-run of the current epoch.
    """
    from fugal_subnet.benchmarks.slicer import epoch_index_for_block
    epoch_index = epoch_index_for_block(current_block, blocks_per_epoch)
    if epoch_index <= last_epoch_index and not once:
        return None
    return epoch_index


def _epoch_boundary(subtensor, epoch_index: int, blocks_per_epoch: int,
                    activation_block: int | None) -> tuple[int, str, str, int]:
    """(boundary block, its hash, epoch id, protocol version) of an epoch.

    All four come from the boundary block, never the current one, so every
    peer and every retry of the epoch derives the same nonce, id and rules.
    The id has one source, slicer.epoch_id_for_block (I1).
    """
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block, protocol_version_for_boundary
    boundary_block = epoch_index * blocks_per_epoch
    return (boundary_block, subtensor.get_block_hash(boundary_block),
            epoch_id_for_block(epoch_index),
            protocol_version_for_boundary(boundary_block, activation_block))


class _SlicePlan(NamedTuple):
    nonce: bytes
    questions: list
    question_ids: list
    questions_hash: str
    explore_map: dict
    slice_gold: dict


def _epoch_slice(protocol_version: int, epoch_id: str, block_hash: str, pool: list,
                 gold_answers: dict, pool_models: list, slice_size: int,
                 explore_size: int) -> _SlicePlan:
    """This epoch's scored questions, exploration targets and the gold for them.

    The miner selects with the same epoch_plan functions on the same versioned
    pool and model list, or every proof fails on questions_hash. The gold is
    for the assigned questions only: handing over the whole pool would let a
    proof reference any question in it and still verify.
    """
    from fugal_subnet.benchmarks.slicer import derive_nonce
    from fugal_subnet.epoch_plan import exploration_plan, scored_slice
    from fugal_subnet.tee.verify import compute_questions_hash
    nonce = derive_nonce(epoch_id, block_hash)
    questions = scored_slice(protocol_version, nonce, pool, slice_size)
    question_ids = [q["question_id"] for q in questions]
    explore_map = exploration_plan(protocol_version, nonce, pool, set(question_ids),
                                   pool_models, explore_size)
    slice_gold = {qid: gold_answers[qid]
                  for qid in list(question_ids) + list(explore_map) if qid in gold_answers}
    return _SlicePlan(nonce, questions, question_ids, compute_questions_hash(question_ids),
                      explore_map, slice_gold)


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
    from fugal_subnet import config as _cfg
    from fugal_subnet.config import V2_ACTIVATION_BLOCK
    _check_live_configuration(
        live, TEE_APPROVED_MEASUREMENTS, V2_ACTIVATION_BLOCK, EPOCH_INTERVAL,
        getattr(_cfg, "BENCHMARK_CATEGORY_V2", None),
    )

    from fugal_subnet.fingerprint import assert_environment, consensus_digest
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

    from fugal_subnet import config as config_module
    from fugal_subnet.benchmarks.loader import load_all, pool_hash
    from fugal_subnet.benchmarks.slicer import (
        blocks_per_epoch as blocks_per_epoch_fn,
    )
    from fugal_subnet.benchmarks.slicer import (
        collect_block_for_epoch,
        epoch_index_for_block,
    )
    from fugal_subnet.commit_reveal import commit_epoch, reveal_epoch
    from fugal_subnet.config import (
        PAYOUT_SEAT_SHARES,
        TEE_COLLATERAL_EPOCH_BUDGET,
        V2_BURN_IN_QUESTIONS,
    )
    from fugal_subnet.dedup import find_duplicates
    from fugal_subnet.epoch_logger import (
        EpochLog,
        EpochTimer,
        detect_anomalies,
        write_epoch_log,
    )
    from fugal_subnet.frontier import build_frontier
    from fugal_subnet.head_eval import HeadScore, load_head_from_npz
    from fugal_subnet.pricing import question_input_tokens
    from fugal_subnet.protocol import FugalProofSynapse
    from fugal_subnet.reference_frame import (
        ReferenceFrame,
        accumulate_exploration,
        best_model,
        load_bootstrap,
        rebuild_from_reveals,
    )
    from fugal_subnet.scoring import MinerRecord, ScoringState, update_scores
    from fugal_subnet.tee.attestation import CollateralBudget, verification_order
    from fugal_subnet.tee.proof import BenchmarkProof
    from fugal_subnet.v2_epoch import (
        run_v2_epoch,
        seat_params_from_config,
        seat_to_dict,
    )

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
    # Publish this host's egress IP so miners can admit only validators at
    # their firewall (fugal_subnet/axon_allowlist.py). Never stops startup.
    from fugal_subnet.axon_allowlist import publish_validator_ip
    from fugal_subnet.config import VALIDATOR_IP
    ip_ok, ip_note = publish_validator_ip(subtensor, wallet, netuid, metagraph, my_uid,
                                          VALIDATOR_IP)
    (logger.info if ip_ok or not live else logger.warning)(ip_note)

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
    # startup rather than mid-epoch. Read from the SAME file the protocol
    # identity hashes (fugal_subnet/routing_data/models.json): data/models.json
    # is the reviewed copy, and the two are checked identical in CI, but a
    # checkout where they differed would have priced every proof from a table
    # the reveal's identity did not name (review 2026-11-05, F-043).
    from fugal_subnet.routing_protocol import price_table
    prices = price_table(1)
    # One global, sorted model space. Used for exploration targets and for
    # dedup indices, so every miner's routing vector is directly comparable.
    pool_models = sorted(prices)
    model_index = {m: i for i, m in enumerate(pool_models)}
    explore_size = max(1, int(round(SLICE_SIZE * EXPLORE_FRACTION)))
    logger.info("Price table: %d models; exploration quota %d questions/epoch",
                len(prices), explore_size)

    # Every epoch input that differs by protocol version, loaded once at
    # startup and picked per epoch by the boundary-keyed version
    # (slicer.protocol_version_for_boundary): the pool and its gold, the price
    # table and the global model index space. v1 is exactly what was loaded
    # above. v2 is its own pool (loader.load_pool(2), pool_manifest_v2.json)
    # and its own price table and model list (routing_protocol, #142), so
    # from the flip every slice, exploration target, price and model index is
    # v2's, and the v2 comparison window is in the v2 model space from its
    # first epoch. Loaded only when an activation block is set, and refused at
    # startup in live if it cannot be, rather than at the flip.
    epoch_inputs = {1: (benchmark_pool, gold_answers, prices, pool_models, model_index)}
    if V2_ACTIVATION_BLOCK is not None:
        from fugal_subnet.benchmarks.loader import load_pool
        from fugal_subnet.routing_protocol import models_for_protocol, price_table
        try:
            pool_v2 = load_pool(2, strict=live)
            prices_v2 = price_table(2)
        except Exception as e:  # noqa: BLE001 - any failure here is a startup refusal
            if live:
                raise click.ClickException(f"protocol v2 inputs could not be loaded: {e}") from e
            logger.warning("protocol v2 inputs could not be loaded (mock): %s", e)
        else:
            models_v2 = models_for_protocol(2)
            epoch_inputs[2] = (
                pool_v2, {q["question_id"]: q for q in pool_v2}, prices_v2,
                models_v2, {m: i for i, m in enumerate(models_v2)},
            )
            logger.info("Protocol v2 inputs: pool %d questions (pool_hash=%s), %d models",
                        len(pool_v2), pool_hash(pool_v2)[:16], len(models_v2))


    state = load_state(MinerRecord)
    # Protocol v2 state: the seat and head owners (in the JSON state) and the
    # paired comparison window (an .npz beside it). Empty until the first v2
    # epoch; rebuilt from reveals like the frame.
    from fugal_subnet.config import CHAMPION_WINDOW_EPOCHS
    from fugal_subnet.v2_epoch import load_window, seat_from_dict
    v2_raw = state.get("v2") or {}
    v2_seat = seat_from_dict(v2_raw.get("seat"))
    # The copy rule's state (first commits, coldkey ownership, retirements),
    # published in every v2 reveal's v2_state under copy_rule.REVEAL_KEY.
    from fugal_subnet import copy_rule
    v2_copy = (copy_rule.CopyState.from_reveal(v2_raw[copy_rule.REVEAL_KEY])
               if v2_raw.get(copy_rule.REVEAL_KEY) else copy_rule.CopyState())
    v2_window = load_window(STATE_PATH + ".v2window.npz", CHAMPION_WINDOW_EPOCHS)
    # A seat with no window to back it (the .npz lost, or never written) must
    # not be judged on a partial window: hold, resume and ghosts continue, but
    # no first or take decision is made until the window is rebuilt from
    # reveals or has refilled itself (fail closed).
    v2_window_degraded = bool(v2_raw.get("window_degraded")) or bool(
        (v2_seat.champion is not None or v2_seat.ghosts) and not v2_window.slices
    )
    # The newest v2 epoch this validator did not observe and has not yet
    # re-anchored past (fugal_subnet/v2_sync.py); None when in step.
    v2_gap_until = v2_raw.get("gap_until")
    v2_gap_until = int(v2_gap_until) if v2_gap_until is not None else None
    # The epoch this validator last re-anchored at; only a reference epoch at
    # or after it can verify the re-anchor.
    v2_anchor_at = v2_raw.get("anchor_at")
    v2_anchor_at = int(v2_anchor_at) if v2_anchor_at is not None else None
    if v2_window_degraded:
        logger.error(
            "v2 seat state present but its comparison window is missing: no first or "
            "take decisions until it is rebuilt (FUGAL_FRAME_BOOTSTRAP) or refills."
        )
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

    # Protocol v2 state, rebuilt the same way when this validator has none of
    # its own, or has a seat but lost its window: the seat, owners and burn-in
    # from the newest reveal's v2_state snapshot, the window replayed from the
    # last CHAMPION_WINDOW_EPOCHS reveals (fugal_subnet/v2_rebuild.py, trust
    # model in its docstring). Per-head v1 evidence is not rebuilt, as before;
    # only the E'' burn-in counter is restored, into fresh records.
    #
    # Exact only if the reveals reach the epoch just before the one this
    # validator resumes at. Published reveals lag by the commit-reveal hold,
    # so a rebuild from a peer's may be a few epochs behind: then the window
    # lacks those epochs and the seat is as of the newest reveal. That opens a
    # sync gap (below: `behind`), and verify-and-resync re-anchors on the
    # reference once it has published past it (fugal_subnet/v2_sync.py).
    if ((not v2_raw or v2_window_degraded) and not v2_window.slices
            and os.getenv("FUGAL_FRAME_BOOTSTRAP")):
        from fugal_subnet.evidence import Evidence
        from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals
        rebuilt = rebuild_v2_from_reveals(
            os.environ["FUGAL_FRAME_BOOTSTRAP"], window_size=CHAMPION_WINDOW_EPOCHS,
            # The replayed window is v2 epochs, so it is priced at v2's table.
            prices=epoch_inputs.get(2, epoch_inputs[1])[2],
            bench_category=getattr(config_module, "BENCHMARK_CATEGORY_V2", None),
        )
        if rebuilt.newest_epoch_id is not None:
            v2_window, v2_seat = rebuilt.window, rebuilt.seat
            v2_copy = copy_rule.from_rebuild(rebuilt) or copy_rule.CopyState()
            for h, uid in sorted(rebuilt.serving.items()):
                if uid in scoring_state.records or h not in rebuilt.burn_in:
                    continue
                # No hotkey: `owners` is the head's FIRST committing hotkey, and
                # an owner may serve from a newer one (coldkey ownership).
                # update_scores treats "" as unknown and keeps the record; a
                # stale value made it drop the record, and the restored
                # burn-in with it, on the first verified proof (F-066).
                rec = MinerRecord(uid=uid, hotkey="", current_head_hash=h)
                rec.evidence = Evidence(weights_hash=h, burn_in_n=rebuilt.burn_in[h])
                scoring_state.records[uid] = rec
            resume_epoch = epoch_index_for_block(
                subtensor.get_current_block(), blocks_per_epoch_fn(EPOCH_INTERVAL),
            )
            behind = resume_epoch - 1 - int(rebuilt.newest_epoch_id.lstrip("e"))
            v2_window_degraded = False
            if behind > 0:
                # The reveals lag by the hold: the epochs since are a gap the
                # verify-and-resync step closes once the reference passes them.
                v2_gap_until = resume_epoch - 1
                logger.warning(
                    "v2 state rebuilt from reveals that end %d epoch(s) before this "
                    "validator resumes: unverified until the reference passes e%08d "
                    "(verify-and-resync).", behind, v2_gap_until,
                )
            else:
                logger.info("v2 state rebuilt exactly from reveals up to %s",
                            rebuilt.newest_epoch_id)

    blocks_per_epoch = blocks_per_epoch_fn(EPOCH_INTERVAL)

    epoch_failed = False
    while True:
        try:
            epoch_failed = False
            current_block = subtensor.get_current_block()
            epoch_index = _epoch_to_run(current_block, blocks_per_epoch, last_epoch_index, once)
            if epoch_index is None:
                time.sleep(30)
                continue
            if epoch_index <= last_epoch_index:
                logger.warning("Epoch %d already processed — running again (--once)", epoch_index)

            # Everything from the boundary block, so every peer and every retry
            # of this epoch derives the same nonce and scores it under the same
            # rules (v1 or v2).
            boundary_block, block_hash, epoch_id, protocol_version = _epoch_boundary(
                subtensor, epoch_index, blocks_per_epoch, V2_ACTIVATION_BLOCK,
            )

            timer = EpochTimer()
            metagraph = subtensor.metagraph(netuid)
            logger.info("=== Epoch %s (boundary block %d, hash %s, protocol v%d) ===",
                        epoch_id, boundary_block, block_hash[:16], protocol_version)

            # --- v2: VERIFY AND RESYNC against a reference validator ---
            # A missed v2 epoch makes this validator's seat state differ from
            # its peers'; once a reference validator's reveal is published,
            # compare and, if they differ, re-anchor on it (v2_sync). Durable
            # at once: a re-anchor is a state correction, not an epoch, and a
            # retry recomputes the same one.
            if protocol_version >= 2:
                synced = _v2_sync_step(
                    epoch_index=epoch_index, last_epoch_index=last_epoch_index,
                    first_v2_epoch=V2_ACTIVATION_BLOCK // blocks_per_epoch,
                    reference_dir=os.getenv("FUGAL_FRAME_BOOTSTRAP") or None,
                    scoring_state=scoring_state, window=v2_window, seat=v2_seat,
                    copy_state=v2_copy, window_degraded=v2_window_degraded,
                    gap_until=v2_gap_until, anchor_at=v2_anchor_at, prices=epoch_inputs[2][2],
                    bench_category=getattr(config_module, "BENCHMARK_CATEGORY_V2", None),
                )
                if synced.changed:
                    (scoring_state, v2_window, v2_seat, v2_copy, v2_window_degraded,
                     v2_gap_until, v2_anchor_at) = synced.state
                    save_state(scoring_state.records, prev_uids, prev_weights,
                               last_epoch_index, frame,
                               v2={"seat": seat_to_dict(v2_seat),
                                   copy_rule.REVEAL_KEY: v2_copy.to_reveal(),
                                   "window_degraded": v2_window_degraded,
                                   "gap_until": v2_gap_until, "anchor_at": v2_anchor_at},
                               v2_window=v2_window)

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
            # This epoch's inputs, by the boundary-keyed version. The miner
            # selects with the same epoch_plan functions on the same versioned
            # pool and model list, or every proof fails on questions_hash.
            timer.start_phase("slice")
            if protocol_version not in epoch_inputs:
                raise RuntimeError(
                    f"epoch {epoch_id} is protocol v{protocol_version} but its inputs "
                    "were not loaded at startup"
                )
            benchmark_pool, gold_answers, prices, pool_models, model_index = (
                epoch_inputs[protocol_version]
            )
            plan = _epoch_slice(protocol_version, epoch_id, block_hash, benchmark_pool,
                                gold_answers, pool_models, SLICE_SIZE, explore_size)
            nonce, questions, question_ids = plan.nonce, plan.questions, plan.question_ids
            expected_questions_hash = plan.questions_hash
            expected_question_ids = set(question_ids)
            explore_map, slice_gold = plan.explore_map, plan.slice_gold
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
            metagraph, metagraph_pinned = metagraph_at(subtensor, netuid, collect_block, metagraph)

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

            checks = _EpochChecks(
                boundary_block=boundary_block,
                require_commitment=REQUIRE_COMMITMENT,
                approved_measurements=approved_measurements,
                questions_hash=expected_questions_hash,
                nonce_hex=nonce_hex,
                slice_gold=slice_gold,
                question_ids=expected_question_ids,
                explore_map=explore_map,
                mock=mock,
                collateral_budget=collateral_budget,
                protocol_version=protocol_version,
            )
            for uid in order:
                hotkey_ss58 = metagraph.hotkeys[uid] if uid < len(metagraph.hotkeys) else ""
                checked = _check_response(
                    uid, responses[uid], hotkey_ss58,
                    chain_commitments.get(hotkey_ss58, ("", 0)),
                    registration_block_of(metagraph, uid), checks,
                )
                if checked.seniority is not None:
                    commit_blocks[uid] = checked.seniority
                if checked.status == "invalid":
                    n_invalid += 1
                elif checked.status == "unverifiable":
                    n_unverifiable += 1
                elif checked.status == "valid":
                    verified_proofs[uid] = checked.proof
                    verified_heads[uid] = checked.head_bytes
                    # Keyed on the hash the attestation forced to be true, not
                    # the value the miner asserted over the axon.
                    head_hashes[uid] = checked.proof.weights_hash

            # Diagnostics only: keep the verified heads on private disk. Feeds
            # no score, hash, reveal or fingerprint, and can't fail the epoch.
            _archive_heads(epoch_id, verified_heads, head_hashes, metagraph)

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
                        ["no_valid_proofs"]
                        + (["collateral_unavailable"] if n_unverifiable else [])
                        + ([] if metagraph_pinned else
                           [f"metagraph_unpinned: the read at block {collect_block} failed"])
                    ),
                    duration_s=timer.total_s,
                )
                write_epoch_log(epoch_log)
                # A v2 epoch this validator scored nothing in (a local
                # collateral outage, or miners it could not reach) may be one
                # its peers scored: a gap until it is verified again.
                burned = None
                if protocol_version >= 2:
                    v2_gap_until = epoch_index
                    # In v2 a seat pays only while its head is live, and with
                    # no proof no head is: the miner share burns. Keeping the
                    # last vector instead paid the v1 winners after the flip
                    # until the first v2 proof verified (Fable review, item 6).
                    burned = _burn_epoch_weights(
                        subtensor, wallet, netuid, my_uid, boundary_block, metagraph, live,
                        prev_uids, prev_weights, epoch_index, blocks_per_epoch)
                last_epoch_index = epoch_index
                # The v2 seat is not advanced on an epoch with no proofs at
                # all: there is nothing to compare, and the champion's ghost
                # window is counted in epochs, so it still expires on time.
                # The burn vector is the next epoch's cap base, rebound only
                # here, at the commit point (tests/test_epoch_atomicity.py).
                prev_uids, prev_weights = burned or (prev_uids, prev_weights)
                save_state(scoring_state.records, prev_uids, prev_weights,
                           last_epoch_index, frame,
                           v2={"seat": seat_to_dict(v2_seat), copy_rule.REVEAL_KEY: v2_copy.to_reveal(),
                               "window_degraded": v2_window_degraded,
                           "gap_until": v2_gap_until, "anchor_at": v2_anchor_at},
                           v2_window=v2_window)
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
            coldkeys = getattr(metagraph, "coldkeys", None) or []
            fi = _frame_input(frame, verified_proofs, coldkeys, protocol_version, questions,
                              explore_map, gold_answers)
            observations, coldkey_of, suspicious = fi.observations, fi.coldkey_of, fi.suspicious
            samples, sample_weights, mix, bench_of = (
                fi.samples, fi.sample_weights, fi.mix, fi.bench_of)

            # EPOCH-LOCAL from here to save_state. `frame`, `scoring_state`,
            # `prev_uids` and `prev_weights` are the validator's durable state
            # and are rebound only once the epoch is written. Anything that
            # raised between scoring and save_state used to land in the outer
            # handler with the loop-level names already advanced, so the retry
            # of the SAME epoch accumulated every miner's evidence twice,
            # applied every miss twice and decayed the frame twice — and then
            # persisted it (found 2026-11-02). tests/test_epoch_atomicity.py
            # holds this shape.
            epoch_frame = accumulate_exploration(frame, samples, weights=sample_weights)
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
            # Diagnostics only (epoch log + the line below): feeds no score.
            epoch_n_failed: dict[int, int] = {}
            for uid, proof in verified_proofs.items():
                score = _score_proof(proof, slice_gold, prices, epoch_frame, ref_model)
                epoch_scores[uid] = score
                epoch_n_failed[uid] = count_failed_scored(proof)
                cpq = score.total_head_cost / max(1, score.n_scored)
                logger.info(
                    "  UID %d: acc=%.3f cost=$%.4f ($%.6f/q, frontier %.3f there) "
                    "ref=$%.4f (%d/%d correct, %d failed)",
                    uid, score.accuracy, score.total_head_cost, cpq,
                    frontier.accuracy_at(cpq), score.total_oracle_cost,
                    score.n_correct, score.n_scored, epoch_n_failed[uid],
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
                protocol_version=protocol_version,
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
            burn_uid = resolve_burn_uid(subtensor, netuid, boundary_block, metagraph, live)
            v2_result = None
            if protocol_version >= 2:
                # --- PROTOCOL v2: the champion, its seats, and nothing else ---
                # Shared coldkey-median cells (grades and attested tokens),
                # the paired comparison window, the seat, and 85/10/5 across
                # the last three champions. Nothing here reads the frame, and
                # nothing burns for frame confidence: the payout is the paired
                # head-vs-head rule (docs/INVARIANTS.md, the v2 I3/I4
                # entries). The v1 composite above is still computed and
                # published as a diagnostic; it decides nothing.
                # The boundary snapshot for the copy rule: every committed
                # hotkey with its owning coldkey, read pinned at the boundary.
                snapshot = _copy_rule_snapshot(metagraph.hotkeys, coldkeys, chain_commitments)
                v2_result = run_v2_epoch(
                    epoch_index=epoch_index, epoch_id=epoch_id,
                    boundary_block=boundary_block, blocks_per_epoch=blocks_per_epoch,
                    questions=questions, pool_models=pool_models, prices=prices,
                    verified_proofs=verified_proofs, head_hashes=head_hashes,
                    coldkey_of=coldkey_of,
                    hotkey_of={uid: metagraph.hotkeys[uid] for uid in verified_proofs},
                    # v1's dedup clauses do not decide v2 eligibility. In v2
                    # a copy is an exact hash owned by another coldkey, or a
                    # junior that fails the margin against a senior seat
                    # holder (the copy rule, copy_rule.py).
                    records=epoch_state.records, commitments=snapshot,
                    window=v2_window, seat=v2_seat, copy_state=v2_copy, burn_uid=burn_uid,
                    params=seat_params_from_config(), shares=PAYOUT_SEAT_SHARES,
                    burn_in_questions=V2_BURN_IN_QUESTIONS,
                    bench_category=getattr(config_module, "BENCHMARK_CATEGORY_V2", None),
                    allow_take=not v2_window_degraded,
                    # Every head committed at or before the flip ties at the
                    # activation block, ordered by a digest salted with the
                    # activation boundary's hash, which nobody knew while the
                    # pre-flip commitments were being made.
                    seniority_floor=V2_ACTIVATION_BLOCK,
                    seniority_salt=_activation_block_hash(subtensor, V2_ACTIVATION_BLOCK,
                                                          boundary_block, block_hash),
                )
                if v2_window_degraded:
                    logger.error(
                        "v2 comparison window was lost and has not been rebuilt: holding "
                        "the seat but making NO first or take decision this epoch. Set "
                        "FUGAL_FRAME_BOOTSTRAP to a directory of reveals and restart to "
                        "rebuild it (it also refills itself after %d epochs).",
                        CHAMPION_WINDOW_EPOCHS,
                    )
                uids, weights = v2_result.uids, v2_result.weights
                d = v2_result.decision
                logger.info(
                    "v2 seat: %s, champion %s, history %s, ghosts %d, high-water %.4f",
                    d.event, (d.champion or "none")[:16],
                    [h[:12] for h in d.state.history], len(d.state.ghosts),
                    d.state.high_water,
                )
                for h, why in d.detail:
                    logger.info("  v2 %s: %s", (h or "-")[:16], why)
            else:
                uids, weights, dupes = _v1_weights(
                    epoch_state.records, routing_decisions, commit_blocks, dupes,
                    frontier.confidence, burn_uid,
                )

            uids, weights, weight_capped = _capped(uids, weights, prev_uids, prev_weights)
            if weight_capped:
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

            matrix, model_spend = _reveal_matrix(questions, pool_models, model_index,
                                                 verified_proofs)

            exploration_for_reveal = _exploration_for_reveal(observations)

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
                frame_excluded_coldkeys=sorted({coldkey_of[uid] for uid in suspicious}),
                protocol_version=protocol_version,
                v2_fields=(
                    {**v2_result.reveal,
                     "frame_mix_weights": dict(sorted(mix.items())),
                     "explore_benchmarks": dict(sorted(bench_of.items())),
                     # Whether this validator's state is unverified since a
                     # missed epoch: a peer re-anchoring on it skips such an
                     # epoch (fugal_subnet/v2_sync.check). Not in the
                     # consensus view.
                     "v2_sync": {"unverified": v2_gap_until is not None}}
                    if v2_result is not None else None
                ),
            )

            # --- SET WEIGHTS ---
            timer.start_phase("set_weights")
            logger.info("Setting weights for %d UIDs", len(uids))
            success, msg = set_weights_with_retry(
                subtensor, wallet, netuid, my_uid, uids, weights,
                deadline_block=_weights_deadline(epoch_index, blocks_per_epoch),
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
            anomalies += _epoch_anomalies(reveal_ok, metagraph_pinned, collect_block,
                                          suspicious, n_unverifiable)

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

            epoch_log = EpochLog(
                epoch_id=epoch_id, block_hash=block_hash,
                timestamp=time.time(), n_questions=len(questions),
                n_miners_queried=n_neurons,
                n_heads_valid=len(verified_proofs), n_heads_invalid=n_invalid,
                n_heads_unverifiable=n_unverifiable,
                commit_hash=commitment.commit_hash,
                reveal_verified=reveal_ok,
                scores=epoch_score_dicts, n_failed=epoch_n_failed,
                weights=epoch_weight_map,
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

            next_v2 = _next_v2_state(v2_result, v2_window, v2_seat, v2_copy,
                                     v2_window_degraded, CHAMPION_WINDOW_EPOCHS)

            # COMMIT. The epoch is complete; only now does it become the state
            # the next epoch (or a retry of this one) starts from.
            scoring_state, frame = epoch_state, epoch_frame
            prev_uids, prev_weights = next_prev_uids, next_prev_weights
            last_epoch_index = epoch_index
            v2_window, v2_seat, v2_copy, v2_window_degraded = next_v2
            save_state(scoring_state.records, prev_uids, prev_weights,
                       last_epoch_index, frame,
                       v2={"seat": seat_to_dict(v2_seat), copy_rule.REVEAL_KEY: v2_copy.to_reveal(),
                           "window_degraded": v2_window_degraded,
                           "gap_until": v2_gap_until, "anchor_at": v2_anchor_at},
                       v2_window=v2_window)

        except KeyboardInterrupt:
            logger.info("Validator stopped by user")
            break
        except Exception:
            epoch_failed = True
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
        #
        # A failed epoch exits 1, so a oneshot or a cron job sees the failure
        # instead of a success with an error in the log.
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if epoch_failed else 0)


class _FrameInput(NamedTuple):
    observations: list         # every exploration answer, tagged with its coldkey (the reveal)
    coldkey_of: dict           # uid -> coldkey, for every verified miner
    suspicious: dict           # uid -> why its exploration stays out of the frame
    samples: list              # one (model, correct, tokens...) per (question, model)
    sample_weights: list | None
    mix: dict                  # v2: benchmark -> scored share / exploration share
    bench_of: dict             # v2: explored question -> benchmark


def _frame_input(frame, verified_proofs: dict, coldkeys, protocol_version: int, questions: list,
                 explore_map: dict, gold_answers: dict) -> _FrameInput:
    """This epoch's contribution to the reference frame, and who is kept out of it.

    One Observation per answer, voted by COLDKEY and collapsed to one sample
    per (question, model) by aggregate_observations: failed calls dropped, one
    vote per operator, median across operators. Every miner answers the same
    questions, so without this each copy was a fresh trial and a coalition
    moved the frame by its share of the field (diligence 2026-10, finding 2).

    Detection runs BEFORE this epoch's samples are folded in, so a miner is
    compared against history rather than against a frame it has just
    contributed to. Two detectors: accuracy implausible for the nonce-chosen
    models in either direction, and disagreement with the other operators on
    the same questions. The only consequence is that a flagged operator's
    samples stay OUT of the frame this epoch. Never a score: I4 says a miner's
    score cannot depend on other miners, and the frame is the one thing built
    from everyone. Keeping a flagged miner's samples out protects every other
    miner's reference rather than moving anyone's score (diligence 2026-10,
    finding 4).

    Protocol v2 weights each sample by its benchmark's scored share over its
    exploration share this epoch (slicer.mix_weights), so the frame estimates
    every model on the scored mix. The raw trial count is not weighted (it is
    what the frontier's confidence reads). The frame is informational in v2;
    this keeps it honest for the detectors and the diagnostics.
    """
    from fugal_subnet.benchmarks.slicer import mix_weights
    from fugal_subnet.reference_frame import (
        Observation,
        aggregate_observations,
        disagreeing_exploration,
        implausible_exploration,
    )

    coldkey_of = {
        uid: str(coldkeys[uid]) if uid < len(coldkeys) else f"uid-{uid}"
        for uid in verified_proofs
    }
    observations = [
        Observation(
            question_id=r.question_id, model=r.routed_model,
            correct=bool(r.correct), prompt_tokens=int(r.prompt_tokens),
            completion_tokens=int(r.completion_tokens), failed=bool(r.failed),
            coldkey=coldkey_of[uid],
        )
        for uid, proof in verified_proofs.items()
        for r in proof.exploration_results
    ]
    suspicious = implausible_exploration(frame, {
        uid: [(r.routed_model, r.correct) for r in proof.exploration_results]
        for uid, proof in verified_proofs.items()
    }, protocol_version=protocol_version)
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
    keyed = aggregate_observations(
        observations,
        exclude_coldkeys={coldkey_of[uid] for uid in suspicious},
        with_question=True,
    )
    samples = [row[1:] for row in keyed]
    mix: dict[str, float] = {}
    bench_of: dict[str, str] = {}
    sample_weights = None
    if protocol_version >= 2:
        explored_qs = [gold_answers[q] for q in sorted(explore_map) if q in gold_answers]
        mix = mix_weights(questions, explored_qs)
        bench_of = {q["question_id"]: str(q.get("benchmark", "")) for q in explored_qs}
        sample_weights = [mix.get(bench_of.get(row[0], ""), 1.0) for row in keyed]
    return _FrameInput(observations, coldkey_of, suspicious, samples, sample_weights,
                       mix, bench_of)


class _EpochChecks(NamedTuple):
    """What every proof this epoch is checked against.

    Every binding is passed explicitly. A check the validator does not supply
    an expectation for is a check that does not happen.
    """
    boundary_block: int
    require_commitment: bool
    approved_measurements: set
    questions_hash: str
    nonce_hex: str
    slice_gold: dict
    question_ids: set
    explore_map: dict
    mock: bool
    collateral_budget: object
    protocol_version: int


class _Checked(NamedTuple):
    """One miner's response, checked.

    status: "absent" (no proof offered), "invalid" (rejected), "unverifiable"
    (this validator could not check it) or "valid". `seniority` is set once the
    commitment check has passed, whatever happens after it.
    """
    status: str
    seniority: float | None = None
    proof: object = None
    head_bytes: bytes | None = None


def _check_response(uid: int, resp, hotkey: str, committed: tuple, registration_block: int,
                    checks: _EpochChecks) -> _Checked:
    """Check one miner's response: its on-chain commitment, its bundle, its proof.

    Never raises on miner input: whatever a miner sends costs that miner, never
    the epoch (I6). The one exception is ImportError, which is this validator's
    own installation and not the miner's doing.
    """
    from fugal_subnet.commitments import is_head_hash, seniority_key
    from fugal_subnet.tee.verify import verify_proof

    if resp is None or not hasattr(resp, "proof_hash") or not resp.proof_hash:
        return _Checked("absent")
    if not hotkey:
        # Responses and hotkeys come from the same metagraph, so this does not
        # happen; if it did, there is no hotkey to bind the proof to.
        logger.warning("UID %d: no hotkey in the metagraph; proof not checked", uid)
        return _Checked("invalid")
    weights_hash = getattr(resp, "weights_hash", "")

    # The on-chain commitment, read at the boundary block. Only a head hash can
    # be committed: an empty or malformed value would compare equal to an
    # equally empty weights_hash and bind the proof to no head at all.
    if (is_head_hash(weights_hash) and committed[0] == weights_hash
            and 0 < committed[1] <= checks.boundary_block):
        # Seniority from chain state at the boundary: the hotkey's
        # registration block first, its commitment block second. A retrained
        # author keeps rank over whoever copied its previous head, and every
        # validator derives the same order (commitments.seniority_key).
        seniority = seniority_key(registration_block, committed[1])
    elif checks.require_commitment:
        logger.warning(
            "UID %d proof rejected: weights_hash %s... not committed on-chain "
            "at or before boundary block %d",
            uid, weights_hash[:12], checks.boundary_block,
        )
        return _Checked("invalid")
    else:
        seniority = math.inf

    try:
        bundle = _get_bundle_for_uid(uid, resp)
    except Exception as e:  # noqa: BLE001 - one miner, not the epoch
        logger.warning("UID %d: failed to get bundle: %s", uid, e)
        return _Checked("invalid", seniority)
    if bundle is None:
        return _Checked("invalid", seniority)
    proof, head_bytes = bundle

    # Defence in depth. verify_proof is contracted not to raise on miner
    # input, but the cost of being wrong about that is the whole epoch, so a
    # bug there must cost one miner instead.
    try:
        result = verify_proof(
            proof,
            approved_measurements=checks.approved_measurements,
            expected_questions_hash=checks.questions_hash,
            expected_nonce=checks.nonce_hex,
            gold_answers=checks.slice_gold,
            expected_question_ids=checks.question_ids,
            expected_exploration=checks.explore_map,
            expected_weights_hash=weights_hash,
            # The hotkey of the uid we actually queried. A proof naming a
            # different one was produced for somebody else.
            expected_hotkey=hotkey,
            expected_proof_hash=getattr(resp, "proof_hash", ""),
            head_bytes=head_bytes,
            mock=checks.mock,
            collateral_budget=checks.collateral_budget,
            protocol_version=checks.protocol_version,
        )
    except ImportError:
        raise
    except Exception as e:  # noqa: BLE001 - one miner, not the epoch
        logger.warning(
            "UID %d proof verification raised %s: %s — rejecting this "
            "miner, not the epoch", uid, type(e).__name__, e,
        )
        return _Checked("invalid", seniority)

    if not result.valid:
        # Not scored either way, and falls through to apply_miss like any
        # absent miner. What changes is what gets RECORDED, and that is not
        # cosmetic: an operator reading "247 invalid" hunts an attack, while
        # "247 unverifiable" points at their own egress.
        #
        # Skipping is also the only I6-compatible choice. Abandoning the epoch
        # instead would let one miner with an unknown FMSPC stop every
        # validator publishing at once, because the collection point is a
        # deterministic block — a subnet kill switch for the price of one
        # registration.
        if getattr(result, "unverifiable", False):
            logger.warning(
                "UID %d proof could not be checked: %s — NOT counted "
                "as invalid; this validator never judged it",
                uid, result.reason,
            )
            return _Checked("unverifiable", seniority)
        logger.warning("UID %d proof failed verification: %s", uid, result.reason)
        return _Checked("invalid", seniority)

    for w in result.warnings or ():
        logger.info("UID %d proof warning: %s", uid, w)
    return _Checked("valid", seniority, proof, head_bytes)


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

    The SDK catches errors inside the extrinsic itself, but its pre-flight
    chain reads (the UID lookup, the commit-reveal check, the rate limit in its
    retry condition) are outside that, so a dropped connection there RAISES out
    of set_weights. That is one failed attempt here, retried like any other,
    not a failed epoch (which would re-query and re-score the whole epoch).
    """
    from fugal_subnet.config import WEIGHTS_SET_RETRIES

    attempts = WEIGHTS_SET_RETRIES if attempts is None else attempts
    msg = ""
    raised = False
    for attempt in range(1, max(1, attempts) + 1):
        try:
            since = int(subtensor.blocks_since_last_update(netuid, my_uid))
            limit = int(subtensor.weights_rate_limit(netuid))
        except Exception as e:  # noqa: BLE001 - diagnostic only
            since, limit = -1, -1
            logger.warning("Could not read the weights rate limit: %s", e)
        if 0 <= since <= limit:
            if raised:
                # An earlier attempt raised, and the chain now says this UID
                # updated within the rate limit: that attempt may have landed
                # before the connection dropped. Read it back rather than
                # report a failure while the weights are on chain.
                landed, detail = confirm_weights_on_chain(subtensor, netuid, my_uid, uids, weights)
                if landed:
                    return True, f"an attempt that raised had landed: {detail}"
            return False, (f"rate limit: {since} blocks since this UID's last update, "
                           f"limit {limit}; the chain would refuse the extrinsic")
        try:
            response = subtensor.set_weights(
                wallet=wallet, netuid=netuid, uids=uids, weights=weights,
                wait_for_inclusion=True, wait_for_finalization=True,
            )
        except Exception as e:  # noqa: BLE001 - a dropped connection is a failed attempt
            raised = True
            logger.warning("set_weights raised on attempt %d", attempt, exc_info=True)
            response = (False, f"set_weights raised {type(e).__name__}: {e}")
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


_ACTIVATION_HASH: dict[int, str] = {}


def _activation_block_hash(subtensor, activation_block: int, boundary_block: int,
                           boundary_hash: str) -> str:
    """The activation boundary's block hash: the v2 seniority tie-break salt.

    Chain data, the same for every validator; cached for the process. Raises
    if the chain cannot serve it, so the epoch is retried rather than ordered
    with a different salt from every peer.
    """
    a = int(activation_block)
    if a not in _ACTIVATION_HASH:
        h = boundary_hash if int(boundary_block) == a else subtensor.get_block_hash(a)
        if not h:
            raise RuntimeError(f"could not read the hash of activation block {a}")
        _ACTIVATION_HASH[a] = str(h)
    return _ACTIVATION_HASH[a]


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


def metagraph_at(subtensor, netuid: int, block: int, fallback) -> tuple:
    """(metagraph as of `block`, True), or (the live one, False) when the pinned
    read fails.

    Pinning the read is what makes two validators query the same axons. A node
    that has pruned the state for that block cannot serve it, though, and a
    validator that refuses to run because it cannot read 40 blocks of history
    is worse than one that reads the current state and says so. The False is
    what the epoch log records as `metagraph_unpinned`, so the divergence is
    attributable from the published artifacts, not only from the console.
    """
    try:
        return subtensor.metagraph(netuid, block=block), True
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not read the metagraph at block %d (%s) — using the current "
            "one. Two validators reading at different blocks can see different "
            "axon sets, so a divergence this epoch has a known cause.",
            block, e,
        )
        return fallback, False


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


def _archive_heads(epoch_id, verified_heads, head_hashes, metagraph):
    """Save this epoch's verified heads when FUGAL_HEAD_ARCHIVE_DIR is set.

    Diagnostics only: the archive feeds nothing back into scoring, and any
    error is logged and swallowed so it cannot cost the epoch.
    """
    from fugal_subnet.config import HEAD_ARCHIVE_DIR
    from fugal_subnet.head_archive import archive_verified_heads

    if not HEAD_ARCHIVE_DIR or not verified_heads:
        return
    try:
        n = len(metagraph.hotkeys)
        counts = archive_verified_heads(
            HEAD_ARCHIVE_DIR, epoch_id, verified_heads, head_hashes,
            hotkeys={u: metagraph.hotkeys[u] for u in verified_heads if u < n},
            coldkeys={u: metagraph.coldkeys[u] for u in verified_heads if u < n},
        )
        logger.info("Head archive: %d new, %d repeat, %d skipped",
                    counts["first_seen"], counts["repeat"], counts["skipped"])
    except Exception as e:  # noqa: BLE001 - diagnostics must not cost the epoch
        logger.warning("Head archive failed this epoch: %s", e)


def count_failed_scored(proof) -> int:
    """Scored calls the proof marks `failed`: upstream errors, not answers.

    Diagnostics only. A failed call is already `correct=False` in the score;
    this count is logged so a low accuracy can be read as "calls failed" or
    "answers were wrong". Exploration is excluded, as it is from the score.
    """
    return sum(1 for r in proof.scored_results if r.failed)


def _reveal_matrix(questions: list, pool_models: list, model_index: dict, verified_proofs: dict):
    """(matrix, model_spend) for the reveal.

    The matrix is in the SAME global model index space dedup uses, so a column
    means the same model in every artifact. -1 means "no miner routed this
    question to this model", which is distinct from 0 ("routed, and got it
    wrong"): as a plain zero matrix the two were indistinguishable and any
    consumer would read unexplored cells as failures.
    """
    q_idx_map = {q["question_id"]: i for i, q in enumerate(questions)}
    matrix = np.full((len(questions), max(len(pool_models), 1)), -1, dtype=np.int32)
    model_spend: dict[str, float] = {}
    for proof in verified_proofs.values():
        for r in proof.results:
            m_idx = model_index.get(r.routed_model)
            q_idx = q_idx_map.get(r.question_id)
            if m_idx is not None and q_idx is not None:
                matrix[q_idx, m_idx] = int(r.correct)
            model_spend[r.routed_model] = model_spend.get(r.routed_model, 0.0) + r.cost_usd
    return matrix, model_spend


def _exploration_for_reveal(observations: list) -> list[dict]:
    """Exploration observations for the reveal, with the voter and the failure
    flag, so rebuild_from_reveals aggregates exactly as this epoch did.
    Coldkeys are public on chain; publishing them reveals nothing new. Sorted
    deterministically in reveal_epoch, so every validator publishes the
    identical list."""
    return [
        {
            "question_id": o.question_id,
            "model": o.model,
            "correct": o.correct,
            "prompt_tokens": o.prompt_tokens,
            "completion_tokens": o.completion_tokens,
            "coldkey": o.coldkey,
            "failed": o.failed,
        }
        for o in observations
    ]


def _weights_deadline(epoch_index: int, blocks_per_epoch: int) -> int:
    """The last block this epoch's weights may be retried at: five blocks
    before the next epoch's boundary, so a retry never lands in the next
    epoch and is never confused with that epoch's weights."""
    return (epoch_index + 1) * blocks_per_epoch - 5


def _epoch_anomalies(reveal_ok: bool, metagraph_pinned: bool, collect_block: int,
                     suspicious: dict, n_unverifiable: int) -> list[str]:
    """The validator's own anomalies for the epoch log, so each is attributable
    after the fact by anyone, not only by whoever was reading the console."""
    out = []
    if not reveal_ok:
        out.append("commit_reveal_failed")
    if not metagraph_pinned:
        # Two validators that queried different axon sets this epoch have a
        # named cause in the published log (F-070).
        out.append(f"metagraph_unpinned: the read at block {collect_block} failed and the "
                   "current metagraph was used")
    for uid, why in suspicious.items():
        out.append(f"exploration_excluded_from_frame: uid {uid}: {why}")
    if n_unverifiable:
        # A signal that the field looks bad for reasons that are ours rather
        # than the miners'. Correlated across validators when the endpoint is
        # at fault, which is exactly what makes it diagnosable (I7).
        out.append(f"collateral_unavailable: {n_unverifiable} proof(s) unchecked")
    return out


class _V2Sync(NamedTuple):
    changed: bool
    # (scoring_state, window, seat, copy_state, window_degraded, gap_until, anchor_at)
    state: tuple


def _v2_sync_step(*, epoch_index: int, last_epoch_index: int, first_v2_epoch: int,
                  reference_dir: str | None, scoring_state, window, seat, copy_state,
                  window_degraded: bool, gap_until: int | None, anchor_at: int | None,
                  prices, bench_category=None, epoch_dir: str | None = None,
                  window_epochs: int | None = None) -> _V2Sync:
    """Detect missed v2 epochs; while this validator has a gap, verify against
    the reference and re-anchor on it (fugal_subnet/v2_sync).

    - A v2 epoch this validator did not observe opens a gap (`gap_until`): the
      validator keeps deciding as usual (blocking its takes would only make it
      diverge whenever its peers take), and is re-checked against the reference.
    - Only a validator WITH a gap re-anchors. A healthy validator that
      disagrees with its reference logs an alarm and keeps its state: the
      reference may be the one that is wrong, and following it would spread
      its outage to everyone downstream.
    - A re-anchor records the epoch it happened at (`anchor_at`); only a
      reference epoch at or after it can verify it (this validator's earlier
      reveals predate the fix). The gap closes on a verified match at or after
      both the gap and the last re-anchor.

    Never raises (I6): any failure keeps the state and the gap.
    """
    state = (scoring_state, window, seat, copy_state, window_degraded, gap_until, anchor_at)
    changed = False
    if last_epoch_index >= 0 and epoch_index > last_epoch_index + 1 \
            and epoch_index - 1 >= first_v2_epoch:
        gap_until = max(-1 if gap_until is None else gap_until, epoch_index - 1)
        logger.error("v2 epochs %d..%d were not observed by this validator: its state is "
                     "unverified until it matches the reference again (decisions go on as "
                     "usual; its reveals are marked v2_sync.unverified)",
                     max(last_epoch_index + 1, first_v2_epoch), epoch_index - 1)
        state, changed = state[:5] + (gap_until, anchor_at), True
    try:
        return _v2_sync_inner(epoch_index=epoch_index, reference_dir=reference_dir,
                              state=state, changed=changed, prices=prices,
                              bench_category=bench_category, epoch_dir=epoch_dir,
                              window_epochs=window_epochs)
    except Exception as e:  # noqa: BLE001 - never stops the epoch (I6); state and gap kept
        logger.error("v2 sync against the reference failed (%s); state and gap kept", e)
        return _V2Sync(changed, state)


def _v2_sync_inner(*, epoch_index, reference_dir, state, changed, prices, bench_category,
                   epoch_dir, window_epochs) -> _V2Sync:
    import copy as _copy

    from fugal_subnet import copy_rule
    from fugal_subnet.commit_reveal import EPOCH_DIR
    from fugal_subnet.config import CHAMPION_WINDOW_EPOCHS
    from fugal_subnet.evidence import Evidence
    from fugal_subnet.scoring import MinerRecord
    from fugal_subnet.v2_sync import DRIFT_CHECKS, check, drifted, load_reveals, resync

    scoring_state, window, seat, copy_state, window_degraded, gap_until, anchor_at = state
    epoch_dir = epoch_dir or EPOCH_DIR
    window_epochs = window_epochs or CHAMPION_WINDOW_EPOCHS
    if gap_until is not None and epoch_index - gap_until > window_epochs:
        # Bounded: after a full window the gap's epochs are out of the window,
        # and a reference that never verifies (none set, or incompatible) must
        # not keep the validator re-anchoring for good.
        logger.warning("v2 gap (through e%08d) closed after a full window without a verified "
                       "match: this validator's seat state may still differ from its peers'",
                       gap_until)
        return _V2Sync(True, state[:5] + (None, None))
    if not reference_dir:
        # No reference: this validator is a ROOT (the others reference it, a
        # star). It has nothing to verify against, so it never marks itself
        # unverified, which would only stop its followers from verifying.
        if gap_until is not None:
            logger.error("v2 epochs through e%08d were not observed and no reference is set: "
                         "this validator's state may now differ from its peers'. A root "
                         "validator recovers by restarting from a rebuild "
                         "(docs/VALIDATOR_GUIDE.md).", gap_until)
            return _V2Sync(True, state[:5] + (None, None))
        return _V2Sync(changed, state)
    since = epoch_index - window_epochs - 1
    ref, own = load_reveals(reference_dir, since), load_reveals(epoch_dir, since)
    chk = check(ref, own, epoch_index)
    if gap_until is None:
        if chk.status != "differs":
            return _V2Sync(changed, state)
        if drifted(ref, own, epoch_index) is None:
            logger.error("ALARM: this validator's v2 state differs from its reference's at "
                         "e%08d with no gap to explain it; it re-anchors if that persists "
                         "for %d reference epochs. Compare the two reveals.", chk.k, DRIFT_CHECKS)
            return _V2Sync(changed, state)
        # Persistent: a proof this validator did not receive (or the
        # reference's own) left the states apart, and nothing else will bring
        # them together. In a star the reference is the arbiter, so this is a
        # gap at k: re-anchor on it below, then verify (Fable review, item 5).
        logger.error("v2 state has differed from the reference's for %d reference epochs "
                     "through e%08d: treating it as a gap and re-anchoring on the reference",
                     DRIFT_CHECKS, chk.k)
        gap_until, changed = chk.k, True
        state = state[:5] + (gap_until, anchor_at)
    if chk.status == "no_reference":
        logger.error("v2 gap open (through e%08d) but %s holds no reference reveal before "
                     "e%08d: is it synced from gs://<reference-bucket>/<netuid>?",
                     gap_until, reference_dir, epoch_index)
        return _V2Sync(changed, state)
    if anchor_at is not None and chk.k < anchor_at:
        return _V2Sync(changed, state)              # wait for a reference epoch after the fix
    if chk.k < gap_until:
        # The reference has not yet published past the gap: a re-anchor now
        # could not cover the unobserved epochs and would only be repeated.
        return _V2Sync(changed, state)
    if chk.status == "in_sync":
        logger.info("v2 state verified against the reference at e%08d: gap closed", chk.k)
        return _V2Sync(True, state[:5] + (None, None))
    rs = resync(ref, own, chk.k, epoch_index, window_size=window_epochs, prices=prices,
                bench_category=bench_category)
    new_copy = copy_rule.from_rebuild(rs.rebuilt) or copy_rule.CopyState()
    scoring = _copy.deepcopy(scoring_state)
    for uid, (h, c) in (rs.burn_in or {}).items():
        rec = scoring.records.get(uid)
        if rec is None:
            rec = scoring.records[uid] = MinerRecord(uid=uid, hotkey="", current_head_hash=h)
        if rec.evidence is None or rec.evidence.weights_hash != h:
            rec.current_head_hash = h
            rec.evidence = Evidence(weights_hash=h)
        rec.evidence.burn_in_n = float(c)
    logger.warning("v2 state %s the reference's at e%08d: re-anchored on it; the gap stays "
                   "open until a reference epoch from e%08d on verifies it%s",
                   "differed from" if chk.status == "differs" else "was missing for", chk.k,
                   epoch_index, "" if rs.complete else
                   " (an epoch since was not observed: the next check re-anchors again)")
    return _V2Sync(True, (scoring, rs.rebuilt.window, rs.rebuilt.seat, new_copy, False,
                          gap_until, epoch_index))


def _next_v2_state(v2_result, window, seat, copy_state, degraded: bool, window_epochs: int):
    """(window, seat, copy state, degraded) carried into the next epoch.

    A v1 epoch carries the v2 state unchanged. A v2 epoch takes its result; a
    degraded window stays degraded until it has refilled to `window_epochs`.
    """
    if v2_result is None:
        return window, seat, copy_state, degraded
    return (v2_result.window, v2_result.decision.state, v2_result.copy_state,
            degraded and len(v2_result.window.slices) < window_epochs)


def _score_proof(proof, slice_gold: dict, prices: dict, epoch_frame, ref_model: str):
    """A verified proof's HeadScore, both sides of thrift from shared data.

    The denominator was already built this way — pinned rates plus the frame's
    measured verbosity — while the numerator was the miner's own account of
    what it spent, and even the denominator's prompt tokens came from the
    miner's report. One half unfakeable, the other a self-report, and a miner
    could lower its cost by shrinking numbers rather than by routing better.
    Prompt tokens come from the POOL, and the miner's cost is computed from
    which models it chose. Only scored questions count (_proof_to_head_score).
    """
    from fugal_subnet.config import FRAME_DEFAULT_COMPLETION_TOKENS
    from fugal_subnet.pricing import policy_cost_total, question_input_tokens
    from fugal_subnet.reference_frame import reference_cost

    scored = proof.scored_results
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
    return _proof_to_head_score(proof, ref_cost, head_cost)


def _copy_rule_snapshot(hotkeys, coldkeys, chain_commitments: dict) -> list:
    """The boundary snapshot for the v2 copy rule: every committed hotkey with
    its owning coldkey and its commitment block, read pinned at the boundary.

    A hotkey with no coldkey in the metagraph has no owner to attribute a head
    to, so it is left out rather than given an empty one.
    """
    from fugal_subnet import copy_rule

    ck_of_hotkey = {
        hk: (str(coldkeys[u]) if u < len(coldkeys) else "")
        for u, hk in enumerate(hotkeys)
    }
    return [
        copy_rule.Commitment(hk, ck_of_hotkey[hk], c[0], int(c[1]))
        for hk, c in sorted(chain_commitments.items())
        if ck_of_hotkey.get(hk)
    ]


def _v1_weights(records: dict, routing_decisions: dict, commit_blocks: dict, dupes,
                confidence: float, burn_uid: int):
    """Protocol v1 weights: (uids, weights, every disqualified uid).

    The epsilon rule runs now that this epoch's composite scores exist: a head
    that routes like a senior one must beat it or it is a copy. The frontier's
    confidence is paid out and the rest burns. It is a factor common to every
    miner, so it is applied AFTER the scores are normalised; inside
    composite() it would cancel.
    """
    from fugal_subnet.dedup import epsilon_disqualified
    from fugal_subnet.rewards import compute_weights

    near_copies = epsilon_disqualified(
        routing_decisions, commit_blocks,
        {uid: records[uid].composite_score for uid in routing_decisions if uid in records},
    )
    if near_copies:
        logger.info("Epsilon rule disqualified: %s", sorted(near_copies))
        dupes = set(dupes) | near_copies
    uids, weights = compute_weights(
        records, dedup_disqualified=dupes, paid_fraction=confidence, burn_uid=burn_uid,
    )
    if confidence < 1.0:
        logger.info(
            "Frontier confidence %.2f: %.0f%% of emission burns to UID %d this epoch",
            confidence, 100.0 * (1.0 - confidence), burn_uid,
        )
    return uids, weights, dupes


def _burn_epoch_weights(subtensor, wallet, netuid, my_uid, boundary_block, metagraph, live,
                        prev_uids, prev_weights, epoch_index, blocks_per_epoch):
    """Set the all-burn vector for a v2 epoch with no valid proof; returns the
    (uids, weights) the next epoch's cap starts from.

    Capped against the previous vector like every epoch, so it moves at most
    MAX_WEIGHT_DELTA per epoch. Never raises (I6): an unreadable owner hotkey
    (live refuses to guess the burn UID) or a failed submit logs and keeps the
    previous vector.
    """
    try:
        burn_uid = resolve_burn_uid(subtensor, netuid, boundary_block, metagraph, live)
        uids, weights, _ = _capped([burn_uid], [1.0], prev_uids, prev_weights)
        ok, msg = set_weights_with_retry(
            subtensor, wallet, netuid, my_uid, uids, weights,
            deadline_block=_weights_deadline(epoch_index, blocks_per_epoch),
        )
        if ok:
            logger.info("No valid proofs: the miner share burns this epoch (UID %d)", burn_uid)
            return list(uids), list(weights)
        logger.error("No valid proofs: setting the burn vector failed: %s", msg)
    except Exception as e:  # noqa: BLE001 - an epoch with no proofs must still end cleanly
        logger.error("No valid proofs: could not set the burn vector: %s", e)
    return prev_uids, prev_weights


def _capped(uids, weights, prev_uids, prev_weights):
    """(uids, weights, capped): the change from last epoch's weights bounded,
    when there were any (rewards.cap_weight_change)."""
    from fugal_subnet.rewards import cap_weight_change

    if not prev_uids:
        return uids, weights, False
    uids, weights = cap_weight_change(uids, weights, prev_uids, prev_weights)
    return uids, weights, True


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
