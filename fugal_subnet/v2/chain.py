"""Concrete Bittensor v10 adapters for the inactive v2 epoch orchestrator."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
from dataclasses import dataclass

from fugal_subnet.commitments import get_commitments_with_blocks
from fugal_subnet.protocol import FugalSynapse
from fugal_subnet.v2.commitments import (
    HEAD_COMMITMENT_ID,
    HistoricalCommitmentReceipt,
    capture_historical_receipt,
    commitment_at_block,
    commitment_payload,
    finalized_block_number,
)
from fugal_subnet.v2.reveal import HeadSubmission

MAX_HEAD_BYTES = 1024 * 1024


class ChainAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class HeadQueryBatch:
    submissions: tuple[HeadSubmission, ...]
    responding_uids: frozenset[int]
    uncommitted_uids: frozenset[int]
    malformed_uids: frozenset[int]


def historical_chain_resolver(subtensor, receipt: HistoricalCommitmentReceipt) -> str:
    """Resolve the Commitment pallet at the receipt's exact historical block."""
    block_hash = str(subtensor.get_block_hash(receipt.block)).removeprefix("0x")
    if block_hash != receipt.block_hash.removeprefix("0x"):
        raise ChainAdapterError("historical receipt block hash differs from chain")
    return commitment_at_block(
        subtensor,
        netuid=receipt.netuid,
        uid=receipt.uid,
        hotkey=receipt.hotkey,
        block=receipt.block,
    )


def query_head_submissions(
    *,
    dendrite,
    subtensor,
    metagraph,
    network: str,
    netuid: int,
    epoch_id: str,
    benchmark_hash: str,
    boundary_block: int,
    timeout: float = 120.0,
) -> HeadQueryBatch:
    """Query boundary axons and bind returned bytes to historical v2 commits."""
    responses = dendrite.query(
        metagraph.axons,
        FugalSynapse(epoch_id=epoch_id, benchmark_hash=benchmark_hash),
        timeout=timeout,
    )
    if not isinstance(responses, list) or len(responses) != int(metagraph.n):
        raise ChainAdapterError("miner Dendrite response count differs from metagraph")
    at_boundary = get_commitments_with_blocks(
        subtensor, netuid, block=boundary_block,
    )
    current = get_commitments_with_blocks(
        subtensor,
        netuid,
        block=finalized_block_number(subtensor),
    )
    submissions = []
    responding: set[int] = set()
    uncommitted: set[int] = set()
    malformed: set[int] = set()
    for uid, response in enumerate(responses):
        encoded = getattr(response, "head_npz_b64", "") if response is not None else ""
        if not encoded:
            continue
        responding.add(uid)
        try:
            artifact = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError, TypeError):
            malformed.add(uid)
            continue
        if not artifact or len(artifact) > MAX_HEAD_BYTES:
            malformed.add(uid)
            continue
        hotkey = str(metagraph.hotkeys[uid])
        artifact_hash = hashlib.sha256(artifact).hexdigest()
        expected = commitment_payload("head", HEAD_COMMITMENT_ID, artifact_hash)
        commitment = at_boundary.get(hotkey)
        if commitment is None or commitment[0] != expected:
            # A commitment that landed after the boundary is still published as
            # deterministic rejection evidence. A missing commitment is not a
            # consensus submission and is bounded out of the reveal.
            commitment = current.get(hotkey)
            if (
                commitment is None
                or commitment[0] != expected
                or commitment[1] <= boundary_block
            ):
                uncommitted.add(uid)
                continue
        commit_block = int(commitment[1])
        receipt = capture_historical_receipt(
            subtensor,
            network=network,
            netuid=netuid,
            hotkey=hotkey,
            uid=uid,
            block=commit_block,
            namespace="head",
            epoch_id=HEAD_COMMITMENT_ID,
            artifact_hash=artifact_hash,
        )
        raw_pool = getattr(response, "model_pool", None)
        if not isinstance(raw_pool, list) or any(not isinstance(item, str) for item in raw_pool):
            raw_pool = []
        submissions.append(HeadSubmission(
            uid=uid,
            hotkey=hotkey,
            artifact=artifact,
            wire_model_pool=tuple(raw_pool),
            commitment_receipt=receipt,
        ))
    return HeadQueryBatch(
        submissions=tuple(submissions),
        responding_uids=frozenset(responding),
        uncommitted_uids=frozenset(uncommitted),
        malformed_uids=frozenset(malformed),
    )


def _chain_weights_match(
    subtensor,
    *,
    netuid: int,
    validator_uid: int,
    target: dict[int, float],
) -> bool:
    """Return true only when the current normalized row already matches target."""
    try:
        metagraph = subtensor.metagraph(netuid)
        row = [float(value) for value in metagraph.W[validator_uid]]
        count = int(metagraph.n)
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    if len(row) != count or any(not math.isfinite(value) or value < 0 for value in row):
        return False
    total = sum(row)
    if total <= 0:
        return False
    normalized = [value / total for value in row]
    # Subtensor quantizes normalized weights to u16. Two units cover the
    # encode/decode and normalization round trips without masking real drift.
    tolerance = 2 / 65_535
    return all(
        abs(normalized[uid] - target.get(uid, 0.0)) <= tolerance
        for uid in range(count)
    ) and not any(uid < 0 or uid >= count for uid in target)


def submit_exact_weights(
    subtensor,
    wallet,
    *,
    netuid: int,
    weights: dict[str, str],
    validator_uid: int | None = None,
) -> bool:
    """Submit verified weights once and require finalization.

    Returns ``False`` when a restart observes the same normalized row already
    on chain, and ``True`` after a newly finalized submission.
    """
    ordered = sorted((int(uid), float(value)) for uid, value in weights.items())
    if (
        not ordered
        or any(uid < 0 or not math.isfinite(value) or value < 0 for uid, value in ordered)
        or len({uid for uid, _ in ordered}) != len(ordered)
        or abs(sum(value for _, value in ordered) - 1.0) > 1e-12
    ):
        raise ChainAdapterError("verified v2 weights are malformed")
    target = dict(ordered)
    if validator_uid is not None:
        if not isinstance(validator_uid, int) or isinstance(validator_uid, bool) or validator_uid < 0:
            raise ChainAdapterError("validator UID is invalid")
        if _chain_weights_match(
            subtensor,
            netuid=netuid,
            validator_uid=validator_uid,
            target=target,
        ):
            return False
    response = subtensor.set_weights(
        wallet=wallet,
        netuid=netuid,
        uids=[uid for uid, _ in ordered],
        weights=[weight for _, weight in ordered],
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    success, message = response
    if not success:
        raise ChainAdapterError(f"v2 weight submission did not finalize: {message}")
    return True
