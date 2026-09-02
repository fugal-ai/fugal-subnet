#!/usr/bin/env python3
"""Manifest-gated Fugal v2 validator and committee report server.

The packaged manifest deliberately keeps this entry point disabled. It becomes
reachable only after a reviewed manifest has complete consensus material and an
activation block for the selected network.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from decimal import Decimal
from pathlib import Path
from typing import TypeVar

import click
import numpy as np

logger = logging.getLogger("fugal.validator_v2")

_ReceiptT = TypeVar("_ReceiptT")


class _FinalizedBlockClock:
    """Thread-safe report-release clock advanced only by the chain thread."""

    def __init__(self, block: int):
        if block < 0:
            raise ValueError("finalized block cannot be negative")
        self._block = int(block)
        self._lock = threading.Lock()

    def current(self) -> int:
        with self._lock:
            return self._block

    def advance(self, block: int) -> int:
        with self._lock:
            if block < self._block:
                raise RuntimeError("finalized report clock cannot move backward")
            self._block = int(block)
            return self._block


def _finalized_block(subtensor) -> int:
    try:
        block_hash = subtensor.substrate.get_chain_finalised_head()
        block = int(subtensor.substrate.get_block_number(block_hash))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("finalized chain head is unavailable") from exc
    if block < 0:
        raise RuntimeError("finalized chain head is invalid")
    return block


def _wait_until(subtensor, block: int) -> int:
    while _finalized_block(subtensor) < block:
        time.sleep(2)
    subtensor.get_block_hash(block)
    return _finalized_block(subtensor)


def _collect_after_finalized_deadline(
    subtensor,
    deadline: int,
    collector: Callable[[], Iterable[_ReceiptT]],
) -> tuple[_ReceiptT, ...]:
    """Collect the closed commitment set only after its deadline finalizes."""
    _wait_until(subtensor, deadline)
    return tuple(collector())


def _previous_weights(metagraph, validator_uid: int) -> dict[int, str]:
    try:
        row = np.asarray(metagraph.W[validator_uid], dtype=np.float64).reshape(-1)
    except (AttributeError, IndexError, TypeError, ValueError):
        return {}
    if len(row) != int(metagraph.n) or not np.all(np.isfinite(row)) or np.any(row < 0):
        raise RuntimeError("boundary metagraph weight row is invalid")
    total = float(row.sum())
    if total <= 0:
        return {}
    row = row / total
    return {
        uid: str(Decimal(str(float(value))))
        for uid, value in enumerate(row)
        if value > 0
    }


def _changed_uids(boundary_hotkeys, current_hotkeys) -> set[int]:
    boundary = [str(item) for item in boundary_hotkeys]
    current = [str(item) for item in current_hotkeys]
    return {
        uid
        for uid in range(max(len(boundary), len(current)))
        if uid >= len(boundary)
        or uid >= len(current)
        or boundary[uid] != current[uid]
    }


def _schedule(consensus: dict) -> tuple[int, int, int, int, int]:
    committee = consensus["committee"]
    names = (
        "epoch_blocks", "precommit_deadline_offset_blocks",
        "report_deadline_offset_blocks", "slice_size", "api_concurrency",
    )
    values = tuple(committee.get(name) for name in names)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise RuntimeError("active v2 manifest schedule is incomplete")
    epoch_blocks, precommit_offset, report_offset, slice_size, concurrency = values
    if not 0 < precommit_offset < report_offset < epoch_blocks:
        raise RuntimeError("active v2 manifest deadlines are invalid")
    return epoch_blocks, precommit_offset, report_offset, slice_size, concurrency


def _epoch_window(
    network: str,
    protocol,
    current_block: int,
    epoch_blocks: int,
    precommit_offset: int,
    report_offset: int,
) -> tuple[int, int, int, int]:
    from fugal_subnet.consensus_manifest import canonical_network

    activation_block = protocol.activation_blocks[canonical_network(network)]
    if activation_block is None or current_block < activation_block:
        raise RuntimeError("v2 activation block is unavailable")
    epoch_index = (current_block - activation_block) // epoch_blocks
    boundary_block = activation_block + epoch_index * epoch_blocks
    return (
        epoch_index,
        boundary_block,
        boundary_block + precommit_offset,
        boundary_block + report_offset,
    )


def _next_epoch_boundary(
    network: str,
    protocol,
    current_block: int,
    epoch_blocks: int,
) -> int:
    from fugal_subnet.consensus_manifest import canonical_network

    activation_block = protocol.activation_blocks[canonical_network(network)]
    if activation_block is None or current_block < activation_block:
        raise RuntimeError("v2 activation block is unavailable")
    return (
        activation_block
        + ((current_block - activation_block) // epoch_blocks + 1) * epoch_blocks
    )


def _local_backbone_lock(network: str) -> Path | None:
    """Return a local-test serialization lock and reject it on public profiles."""
    from fugal_subnet.consensus_manifest import canonical_network

    configured = os.getenv("FUGAL_LOCAL_BACKBONE_LOCK")
    if configured is None:
        return None
    if canonical_network(network) != "local":
        raise RuntimeError("FUGAL_LOCAL_BACKBONE_LOCK is local/mock-only")
    path = Path(configured).resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def _local_backbone_cache(network: str) -> Path | None:
    """Return a local-only shared embedding cache used by acceptance tests."""
    from fugal_subnet.consensus_manifest import canonical_network

    configured = os.getenv("FUGAL_LOCAL_BACKBONE_CACHE")
    if configured is None:
        return None
    if canonical_network(network) != "local":
        raise RuntimeError("FUGAL_LOCAL_BACKBONE_CACHE is local/mock-only")
    path = Path(configured).resolve()
    if path.is_symlink():
        raise RuntimeError("local backbone cache cannot be a symbolic link")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def _compute_local_serialized(
    prompts: list[str],
    lock_path: Path,
    cache_root: Path | None = None,
) -> np.ndarray:
    """Compute once and share exact embeddings in a local acceptance run."""
    import fcntl

    from fugal_subnet.consensus_manifest import canonical_json
    from fugal_subnet.v2.backbone import (
        HIDDEN_DIM,
        compute_hidden_states,
        release_backbone,
    )

    prompt_hash = hashlib.sha256(canonical_json(prompts)).hexdigest()
    cached = cache_root / f"{prompt_hash}.npy" if cache_root is not None else None

    with lock_path.open("a+b") as handle:
        os.fchmod(handle.fileno(), 0o600)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            if cached is not None and cached.exists():
                if cached.is_symlink():
                    raise RuntimeError("local backbone cache entry cannot be a symlink")
                with cached.open("rb") as source:
                    result = np.load(source, allow_pickle=False)
            else:
                result = compute_hidden_states(prompts)
                if cached is not None:
                    temporary = cached.with_name(f".{cached.name}.tmp-{os.getpid()}")
                    with temporary.open("xb") as output:
                        np.save(output, result, allow_pickle=False)
                        output.flush()
                        os.fsync(output.fileno())
                    temporary.chmod(0o600)
                    temporary.replace(cached)
                    directory_fd = os.open(cached.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            result = np.asarray(result)
            if (
                result.dtype != np.float32
                or result.shape != (len(prompts), HIDDEN_DIM)
                or not np.all(np.isfinite(result))
            ):
                raise RuntimeError("local backbone cache entry is invalid")
            return result
        finally:
            release_backbone()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _abort_stale_journals(journal_root: Path, current_boundary_block: int) -> int:
    """Terminally close active journals whose epoch window cannot be resumed."""
    from fugal_subnet.v2.journal import EPOCH_ID_RE, EpochJournal, JournalError

    if not journal_root.exists():
        return 0
    if journal_root.is_symlink() or not journal_root.is_dir():
        raise RuntimeError("v2 journal root is not a safe directory")
    aborted = 0
    for entry in sorted(journal_root.iterdir()):
        if (
            entry.is_symlink()
            or not entry.is_dir()
            or not EPOCH_ID_RE.fullmatch(entry.name)
        ):
            continue
        journal = EpochJournal(journal_root, entry.name)
        if not journal.path.exists():
            continue
        if journal.path.is_symlink():
            raise RuntimeError("v2 epoch journal cannot be a symbolic link")
        try:
            state = journal.read()
            if (
                state["status"] == "active"
                and state["boundary"]["block"] < current_boundary_block
            ):
                journal.abort(
                    f"epoch expired before finalized boundary {current_boundary_block}"
                )
                aborted += 1
        except JournalError as exc:
            raise RuntimeError(f"stale v2 journal {entry.name} is invalid") from exc
    return aborted


def _run_epoch(
    *,
    network: str,
    netuid: int,
    wallet,
    subtensor,
    dendrite,
    manifest,
    protocol,
    pool: list[dict],
    registry,
    grading_client,
    report_store,
    report_clock: _FinalizedBlockClock,
    journal_root: Path,
    reveal_root: Path,
    validator_state,
    live: bool,
    budget: Decimal,
) -> int:
    from fugal_subnet.api import fetch_openrouter_prices
    from fugal_subnet.benchmarks.slicer import derive_nonce, select_slice
    from fugal_subnet.consensus_manifest import canonical_json
    from fugal_subnet.v2.backbone import compute_hidden_states
    from fugal_subnet.v2.chain import (
        HeadQueryBatch,
        historical_chain_resolver,
        query_head_submissions,
        submit_exact_weights,
    )
    from fugal_subnet.v2.commitments import (
        collect_historical_receipts,
        submit_commitment_with_receipt,
    )
    from fugal_subnet.v2.committee import select_builders
    from fugal_subnet.v2.journal import EpochJournal
    from fugal_subnet.v2.matrix import (
        build_matrix,
        mock_response_function,
        openrouter_response_function,
    )
    from fugal_subnet.v2.orchestrator import EpochDefinition, EpochHooks, run_once
    from fugal_subnet.v2.report_server import fetch_report
    from fugal_subnet.v2.reports import build_signed_report
    from fugal_subnet.v2.reveal import (
        build_reveal,
        finalize_reveal,
        registry_snapshot_hash,
        stage_reveal,
        verify_reveal,
    )

    consensus = protocol.consensus
    epoch_blocks, precommit_offset, report_offset, slice_size, concurrency = _schedule(
        consensus
    )
    current_block = _finalized_block(subtensor)
    from fugal_subnet.consensus_manifest import select_protocol

    selected = select_protocol(network, current_block, manifest)
    if selected.protocol_id != "v2":
        raise RuntimeError("v2 is no longer the active protocol")
    epoch_index, boundary_block, precommit_deadline, report_deadline = _epoch_window(
        network,
        protocol,
        current_block,
        epoch_blocks,
        precommit_offset,
        report_offset,
    )
    stale_count = _abort_stale_journals(journal_root, boundary_block)
    if stale_count:
        logger.warning("Aborted %d expired v2 epoch journal(s)", stale_count)
    boundary_hash = str(subtensor.get_block_hash(boundary_block)).removeprefix("0x")
    epoch_id = f"v2-{epoch_index:012d}"
    metagraph = subtensor.metagraph(netuid, block=boundary_block)
    committee = select_builders(
        boundary_hash,
        [str(item) for item in metagraph.hotkeys],
        [bool(item) for item in metagraph.validator_permit],
        maximum_builders=int(consensus["committee"]["maximum_builders"]),
        minimum_reports=int(consensus["committee"]["minimum_reports"]),
    )
    nonce = derive_nonce(epoch_id, boundary_hash)
    questions = select_slice(nonce, pool, slice_size)
    if len(questions) != slice_size:
        raise RuntimeError("canonical v2 slice size differs from manifest")
    grader_hash = consensus["grader"]["sha256"]
    model_ids = list(registry.model_ids)
    route_costs = {model: str(value) for model, value in registry.route_costs.items()}
    registry_hash = registry_snapshot_hash(model_ids, route_costs)
    benchmark_hash = hashlib.sha256(canonical_json(
        [question["question_id"] for question in questions]
    )).hexdigest()
    definition = EpochDefinition(
        epoch_id=epoch_id,
        boundary_block=boundary_block,
        boundary_hash=boundary_hash,
        precommit_deadline_block=precommit_deadline,
        report_deadline_block=report_deadline,
        manifest_hash=manifest.sha256,
        grader_hash=grader_hash,
        questions=questions,
        committee=committee,
        budget_usd=str(budget),
    )
    journal = EpochJournal(journal_root, epoch_id)
    if current_block >= precommit_deadline and not journal.path.exists():
        logger.error(
            "Epoch %d precommit window already closed at finalized block %d",
            epoch_index,
            precommit_deadline,
        )
        return 1
    self_hotkey = wallet.hotkey.ss58_address
    self_uid = [str(item) for item in metagraph.hotkeys].index(self_hotkey)
    builder_pairs = [(builder.uid, builder.hotkey) for builder in committee]
    boundary_axons = {builder.hotkey: metagraph.axons[builder.uid] for builder in committee}
    question_receipts = []
    head_batch: HeadQueryBatch | None = None

    def commit_questions(artifact_hash):
        receipt = submit_commitment_with_receipt(
            subtensor,
            wallet,
            network=network,
            netuid=netuid,
            uid=self_uid,
            namespace="questions",
            epoch_id=epoch_id,
            artifact_hash=artifact_hash,
        )
        if receipt.block > precommit_deadline:
            raise RuntimeError("question commitment missed the precommit deadline")
        return receipt

    def collect_questions(artifact_hash):
        nonlocal question_receipts
        question_receipts = list(_collect_after_finalized_deadline(
            subtensor,
            precommit_deadline,
            lambda: collect_historical_receipts(
                subtensor,
                network=network,
                netuid=netuid,
                builders=builder_pairs,
                namespace="questions",
                epoch_id=epoch_id,
                start_block=boundary_block,
                end_block=precommit_deadline,
                artifact_hash=artifact_hash,
            ),
        ))
        return tuple(question_receipts)

    def query_heads():
        nonlocal head_batch
        head_batch = query_head_submissions(
            dendrite=dendrite,
            subtensor=subtensor,
            metagraph=metagraph,
            network=network,
            netuid=netuid,
            epoch_id=epoch_id,
            benchmark_hash=benchmark_hash,
            boundary_block=boundary_block,
        )
        return head_batch

    if live:
        current_prices = fetch_openrouter_prices()
        spend_prices = {}
        for model, canonical_prices in registry.prices_per_token.items():
            if model not in current_prices:
                raise RuntimeError(f"current paid price unavailable for {model}")
            live_prices = tuple(Decimal(str(value)) for value in current_prices[model])
            spend_prices[model] = (
                max(canonical_prices[0], live_prices[0]),
                max(canonical_prices[1], live_prices[1]),
            )
        response_function = openrouter_response_function(spend_prices)
    else:
        spend_prices = {model: (Decimal(0), Decimal(0)) for model in model_ids}
        response_function = mock_response_function

    def make_report(_heads, committed_question_hash):
        matrix_result = build_matrix(
            questions,
            model_ids,
            journal=journal,
            response_function=response_function,
            spend_prices=spend_prices,
            grading_client=grading_client,
            concurrency=concurrency,
        )
        payload = build_signed_report(
            matrix_result,
            epoch_id=epoch_id,
            boundary_block=boundary_block,
            boundary_hash=boundary_hash,
            manifest_hash=manifest.sha256,
            question_commitment=committed_question_hash,
            grader_hash=grader_hash,
            registry_hash=registry_hash,
            keypair=wallet.hotkey,
        )
        report_store.publish(payload, wallet.hotkey, release_block=report_deadline)
        return payload

    def commit_report(artifact_hash):
        return submit_commitment_with_receipt(
            subtensor,
            wallet,
            network=network,
            netuid=netuid,
            uid=self_uid,
            namespace="report",
            epoch_id=epoch_id,
            artifact_hash=artifact_hash,
        )

    def collect_reports():
        return collect_historical_receipts(
            subtensor,
            network=network,
            netuid=netuid,
            builders=builder_pairs,
            namespace="report",
            epoch_id=epoch_id,
            start_block=boundary_block + 1,
            end_block=report_deadline,
        )

    def fetch_builder_report(hotkey, artifact_hash):
        if hotkey == self_hotkey:
            return report_store.resume_payload(epoch_id, artifact_hash)
        return fetch_report(
            dendrite,
            boundary_axons[hotkey],
            epoch_id=epoch_id,
            manifest_hash=manifest.sha256,
            artifact_hash=artifact_hash,
            builder_hotkey=hotkey,
        )

    def wait_for_report_deadline(block):
        finalized = _wait_until(subtensor, block)
        report_clock.advance(finalized)
        return finalized

    hidden_cache = None
    local_backbone_lock = _local_backbone_lock(network)
    local_backbone_cache = _local_backbone_cache(network)
    if (local_backbone_lock is None) != (local_backbone_cache is None):
        raise RuntimeError("local backbone lock/cache must be configured together")

    def embedder(prompts):
        nonlocal hidden_cache
        if hidden_cache is None:
            hidden_cache = (
                _compute_local_serialized(
                    prompts, local_backbone_lock, local_backbone_cache
                )
                if local_backbone_lock is not None
                else compute_hidden_states(prompts)
            )
        return hidden_cache

    def reveal_and_set(heads, _consensus_matrix, artifacts, report_receipts):
        if not isinstance(heads, HeadQueryBatch):
            raise RuntimeError("head query result type differs")
        finalized_block = _finalized_block(subtensor)
        current_metagraph = subtensor.metagraph(netuid, block=finalized_block)
        current_hotkeys = {
            uid: str(hotkey) for uid, hotkey in enumerate(current_metagraph.hotkeys)
        }
        if self_hotkey not in current_hotkeys.values():
            raise RuntimeError("validator hotkey is no longer registered")
        current_self_uid = next(
            uid for uid, hotkey in current_hotkeys.items() if hotkey == self_hotkey
        )
        if current_self_uid != self_uid:
            raise RuntimeError("validator UID ownership changed during the epoch")
        transferred_uids = _changed_uids(metagraph.hotkeys, current_metagraph.hotkeys)
        head_hashes = {
            item.uid: hashlib.sha256(item.artifact).hexdigest()
            for item in heads.submissions
        }
        liveness = validator_state.preview_epoch(
            epoch_id,
            current_hotkeys,
            set(heads.responding_uids),
            head_hashes,
        )
        forced = (
            set(liveness.forced_zero_uids)
            | set(heads.uncommitted_uids)
            | set(heads.malformed_uids)
            | transferred_uids
        )
        payload, _ = build_reveal(
            epoch={
                "epoch_id": epoch_id,
                "boundary_block": boundary_block,
                "boundary_hash": boundary_hash,
                "precommit_deadline_block": precommit_deadline,
                "report_deadline_block": report_deadline,
                "manifest_hash": manifest.sha256,
                "grader_hash": grader_hash,
            },
            committee=committee,
            questions=questions,
            route_costs_usd=route_costs,
            question_receipts=question_receipts,
            report_receipts=report_receipts,
            builder_reports=artifacts,
            head_submissions=heads.submissions,
            grading_client=grading_client,
            embedder=embedder,
            previous_weights=_previous_weights(metagraph, self_uid),
            eligible_uids=set(liveness.eligible_uids),
            forced_zero_uids=forced,
        )
        verified = verify_reveal(
            payload,
            grading_client=grading_client,
            embedder=embedder,
            chain_resolver=lambda receipt: historical_chain_resolver(subtensor, receipt),
        )
        reveal_path = reveal_root / epoch_id / "reveal.json"
        stage_reveal(payload, reveal_path)
        if verified.weights is not None:
            current_count = int(current_metagraph.n)
            chain_weights = {
                uid: value
                for uid, value in verified.weights.items()
                if int(uid) < current_count
            }
            removed = {
                uid: value
                for uid, value in verified.weights.items()
                if int(uid) >= current_count and Decimal(value) != 0
            }
            if removed or sum(Decimal(value) for value in chain_weights.values()) != 1:
                raise RuntimeError("reveal weights cannot map to the current metagraph")
            submit_exact_weights(
                subtensor,
                wallet,
                netuid=netuid,
                weights=chain_weights,
                validator_uid=current_self_uid,
                epoch_start_block=boundary_block,
            )
        committed_liveness = validator_state.update_epoch(
            epoch_id,
            current_hotkeys,
            set(heads.responding_uids),
            head_hashes,
        )
        if committed_liveness != liveness:
            raise RuntimeError("persisted liveness differs from evaluated reveal")
        finalize_reveal(payload, reveal_path)
        return True

    hooks = EpochHooks(
        commit_questions=commit_questions,
        collect_question_receipts=collect_questions,
        query_heads=query_heads,
        build_signed_report=make_report,
        commit_report=commit_report,
        wait_until_report_deadline=wait_for_report_deadline,
        collect_report_receipts=collect_reports,
        fetch_report=fetch_builder_report,
        evaluate_reveal_and_set_weights=reveal_and_set,
    )
    return run_once(
        definition,
        self_hotkey=self_hotkey,
        journal=journal,
        hooks=hooks,
    )


@click.command()
@click.option("--network", default=lambda: os.getenv("FUGAL_NETWORK", "test"))
@click.option("--netuid", type=int, default=lambda: int(os.getenv("FUGAL_NETUID", "1")))
@click.option("--coldkey", default=lambda: os.getenv("WALLET_NAME", "default"))
@click.option("--hotkey", default=lambda: os.getenv("HOTKEY_NAME", "default"))
@click.option("--wallet-path", default=lambda: os.getenv("FUGAL_WALLET_PATH") or None,
              type=click.Path(file_okay=False))
@click.option("--grader-socket", required=True, type=click.Path(dir_okay=False))
@click.option("--report-port", type=int, default=8092)
@click.option("--state-root", type=click.Path(file_okay=False, path_type=Path),
              default=Path("results/v2"))
@click.option("--once", is_flag=True, help="Run one eligible epoch and exit")
@click.option("--live/--mock", default=False,
              help="Enable paid OpenRouter calls; defaults to mock")
@click.option("--epoch-budget", type=click.FloatRange(min=0.0, min_open=True))
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def main(
    network, netuid, coldkey, hotkey, wallet_path, grader_socket, report_port,
    state_root, once, live, epoch_budget, log_level,
):
    """Run the separately activated v2 protocol; inactive packages refuse."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if live and epoch_budget is None:
        configured = os.getenv("FUGAL_EPOCH_BUDGET")
        try:
            epoch_budget = float(configured) if configured is not None else None
        except ValueError as exc:
            raise click.UsageError("FUGAL_EPOCH_BUDGET must be positive") from exc
    if live and (epoch_budget is None or epoch_budget <= 0):
        raise click.UsageError("--live requires an explicit positive epoch budget")
    if not live and epoch_budget is not None:
        raise click.UsageError("--epoch-budget is valid only with --live")
    budget = Decimal(str(epoch_budget)) if live else Decimal(0)

    import bittensor as bt

    from fugal_subnet.consensus_manifest import (
        canonical_json,
        load_consensus_manifest,
        select_protocol,
        verify_runtime_dependencies,
    )
    from fugal_subnet.graders_v2 import grader_hash
    from fugal_subnet.sandbox.client import GradingClient
    from fugal_subnet.v2.backbone import (
        GOLDEN_PROMPTS,
        BackboneSpec,
        verify_backbone_golden,
    )
    from fugal_subnet.v2.benchmarks import load_pool
    from fugal_subnet.v2.contract import verify_executable_contract
    from fugal_subnet.v2.model_registry import load_model_registry
    from fugal_subnet.v2.report_server import ReportStore, make_report_forward
    from fugal_subnet.v2.validator_state import ValidatorStateStore

    wallet = bt.Wallet(name=coldkey, hotkey=hotkey, path=wallet_path)
    subtensor = bt.Subtensor(network=network)
    manifest = load_consensus_manifest(network)
    protocol = select_protocol(network, _finalized_block(subtensor), manifest)
    if protocol.protocol_id != "v2":
        raise click.ClickException(
            f"v2 is not active on {network}; selected {protocol.protocol_id}"
        )
    consensus = protocol.consensus
    if consensus is None:
        raise click.ClickException("active v2 consensus material is missing")
    verify_runtime_dependencies(consensus)
    verify_executable_contract(consensus)
    if grader_hash().removeprefix("sha256:") != consensus["grader"]["sha256"]:
        raise click.ClickException("packaged v2 grader bundle differs from manifest")
    if BackboneSpec().sha256 != consensus["backbone"]["spec_sha256"]:
        raise click.ClickException("packaged v2 backbone policy differs from manifest")
    local_backbone_lock = _local_backbone_lock(network)
    local_backbone_cache = _local_backbone_cache(network)
    if (local_backbone_lock is None) != (local_backbone_cache is None):
        raise click.ClickException(
            "local backbone lock/cache must be configured together"
        )
    if local_backbone_lock is None:
        verify_backbone_golden(
            expected_prompts_sha256=consensus["backbone"]["golden_prompts_sha256"],
            expected_embeddings_sha256=consensus["backbone"]["golden_embeddings_sha256"],
        )
    else:
        prompt_hash = hashlib.sha256(canonical_json(list(GOLDEN_PROMPTS))).hexdigest()
        if prompt_hash != consensus["backbone"]["golden_prompts_sha256"]:
            raise click.ClickException("pinned v2 backbone prompts differ")
        embeddings = _compute_local_serialized(
            list(GOLDEN_PROMPTS), local_backbone_lock, local_backbone_cache
        )
        actual_hash = hashlib.sha256(embeddings.tobytes(order="C")).hexdigest()
        if actual_hash != consensus["backbone"]["golden_embeddings_sha256"]:
            raise click.ClickException("pinned v2 backbone golden differs")
    registry = load_model_registry(require_active=True, network=network)
    if registry.sha256 != consensus["model_registry"]["canonical_sha256"]:
        raise click.ClickException("active model registry differs from manifest")
    pool = load_pool()
    grading_client = GradingClient(grader_socket)
    if not grading_client.health():
        raise click.ClickException("isolated grading worker is unavailable")

    metagraph = subtensor.metagraph(netuid)
    if wallet.hotkey.ss58_address not in [str(item) for item in metagraph.hotkeys]:
        raise click.ClickException("validator hotkey is not registered")
    state_root = state_root.resolve()
    report_clock = _FinalizedBlockClock(_finalized_block(subtensor))
    report_store = ReportStore(
        state_root / "reports",
        current_block=report_clock.current,
    )
    report_store.restore(wallet.hotkey)
    axon = bt.Axon(
        wallet=wallet,
        port=report_port,
        external_ip=os.getenv("FUGAL_AXON_IP") or None,
    )
    axon.attach(forward_fn=make_report_forward(report_store))
    axon.serve(netuid=netuid, subtensor=subtensor)
    axon.start()
    dendrite = bt.Dendrite(wallet=wallet)
    validator_state = ValidatorStateStore(state_root / "validator-state.json")
    try:
        while True:
            report_clock.advance(_finalized_block(subtensor))
            status = _run_epoch(
                network=network,
                netuid=netuid,
                wallet=wallet,
                subtensor=subtensor,
                dendrite=dendrite,
                manifest=manifest,
                protocol=protocol,
                pool=pool,
                registry=registry,
                grading_client=grading_client,
                report_store=report_store,
                report_clock=report_clock,
                journal_root=state_root / "journals",
                reveal_root=state_root / "epochs",
                validator_state=validator_state,
                live=live,
                budget=budget,
            )
            if once:
                return status
            epoch_blocks = _schedule(consensus)[0]
            current = _finalized_block(subtensor)
            _wait_until(
                subtensor,
                _next_epoch_boundary(network, protocol, current, epoch_blocks),
            )
    finally:
        axon.stop()
        try:
            dendrite.close_session()
        finally:
            subtensor.close()


if __name__ == "__main__":
    main()
