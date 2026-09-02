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

# Subtensor quantizes normalized weights to u16. Two units cover the
# encode/decode and normalization round trips without masking real drift.
WEIGHT_MATCH_TOLERANCE = 2 / 65_535


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


def read_finalized_weight_row(
    subtensor,
    *,
    netuid: int,
    validator_uid: int,
    block: int | None = None,
) -> dict[int, float] | None:
    """Read one validator's normalized weight row from finalized chain storage.

    ``metagraph.W`` is a derived dense matrix that Bittensor 10.5 can return
    empty even when finalized ``SubtensorModule.Weights`` rows exist, so both
    production restart checks and acceptance assertions read the storage map
    directly at an explicit finalized block.

    Returns ``None`` when the validator genuinely has no row, and raises
    ``ChainAdapterError`` when the chain could not be read at all. Callers that
    must not distinguish the two use :func:`finalized_weight_row`.
    """
    if not isinstance(validator_uid, int) or isinstance(validator_uid, bool):
        raise ChainAdapterError("validator UID is invalid")
    if validator_uid < 0:
        raise ChainAdapterError("validator UID is invalid")
    if block is not None:
        if not isinstance(block, int) or isinstance(block, bool) or block < 0:
            raise ChainAdapterError("weight query block is invalid")
    try:
        height = finalized_block_number(subtensor) if block is None else block
        rows = subtensor.weights(netuid, block=height)
    except Exception as exc:
        raise ChainAdapterError(
            f"finalized weight query failed at block {block}: {exc!r}"
        ) from exc
    if not isinstance(rows, (list, tuple)):
        raise ChainAdapterError(
            f"finalized weight query returned {type(rows).__name__}, not a sequence"
        )
    raw: object = None
    for entry in rows:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        try:
            uid = int(entry[0])
        except (TypeError, ValueError):
            continue
        if uid == validator_uid:
            raw = entry[1]
            break
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    row: dict[int, float] = {}
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        try:
            target_uid = int(pair[0])
            value = float(pair[1])
        except (TypeError, ValueError):
            return None
        if target_uid < 0 or not math.isfinite(value) or value < 0:
            return None
        if target_uid in row:
            return None
        row[target_uid] = value
    total = sum(row.values())
    if total <= 0:
        return None
    return {uid: value / total for uid, value in row.items()}


def finalized_weight_row(
    subtensor,
    *,
    netuid: int,
    validator_uid: int,
    block: int | None = None,
) -> dict[int, float] | None:
    """Lenient wrapper: an unreadable chain looks the same as an absent row.

    The restart check must resubmit rather than assume its weights are already
    set, so it cannot afford to distinguish the two. Acceptance and debugging
    callers use :func:`read_finalized_weight_row`, which reports the reason.
    """
    try:
        return read_finalized_weight_row(
            subtensor, netuid=netuid, validator_uid=validator_uid, block=block
        )
    except ChainAdapterError:
        return None


def pending_timelock_commit_block(
    subtensor,
    *,
    netuid: int,
    hotkey: str,
) -> int | None:
    """Return the block of an unrevealed timelock weight commit by ``hotkey``.

    Subnets with ``commit_reveal_weights_enabled`` do not write plaintext
    ``Weights`` when ``set_weights`` succeeds. The encrypted payload sits in
    ``TimelockedWeightCommits`` until its reveal epoch, so during that window a
    successful submission is invisible to :func:`read_finalized_weight_row`.
    Restart idempotency has to consult this map or it will resubmit and burn
    the subnet's weight rate limit.

    Returns ``None`` when no such commit exists or the map cannot be read.
    """
    try:
        entries = subtensor.substrate.query_map(
            module="SubtensorModule",
            storage_function="TimelockedWeightCommits",
            params=[netuid],
        )
    except Exception:
        return None
    latest: int | None = None
    try:
        for _epoch, commits in entries:
            for commit in commits or ():
                if not isinstance(commit, (list, tuple)) or len(commit) < 2:
                    continue
                if str(commit[0]) != str(hotkey):
                    continue
                try:
                    block = int(commit[1])
                except (TypeError, ValueError):
                    continue
                if latest is None or block > latest:
                    latest = block
    except (TypeError, ValueError):
        return None
    return latest


def commit_reveal_is_enabled(subtensor, netuid: int) -> bool:
    """Report whether this subnet defers weights through a timelock commit."""
    try:
        return bool(
            subtensor.substrate.query(
                module="SubtensorModule",
                storage_function="CommitRevealWeightsEnabled",
                params=[netuid],
            )
        )
    except Exception:
        # Assume the deferring path, so an unreadable flag never causes a
        # duplicate submission.
        return True


def _chain_weights_match(
    subtensor,
    *,
    netuid: int,
    validator_uid: int,
    target: dict[int, float],
) -> bool:
    """Return true only when the finalized row already matches target."""
    normalized = finalized_weight_row(
        subtensor, netuid=netuid, validator_uid=validator_uid
    )
    if not normalized:
        return False
    # Compare over the union so a chain entry absent from the target, or a
    # target entry the chain never stored, both count as drift. Explicit zero
    # weights compare equal whether or not the pallet retained them.
    for uid in set(normalized) | set(target):
        if abs(normalized.get(uid, 0.0) - target.get(uid, 0.0)) > WEIGHT_MATCH_TOLERANCE:
            return False
    return True


def submit_exact_weights(
    subtensor,
    wallet,
    *,
    netuid: int,
    weights: dict[str, str],
    validator_uid: int | None = None,
    epoch_start_block: int | None = None,
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
        # On a commit-reveal subnet a successful submission stays encrypted
        # until its reveal epoch, so the plaintext check above cannot see it.
        # Only a commit made at or after this epoch's boundary proves that this
        # epoch's weights were already submitted; an older commit belongs to a
        # previous epoch and must not suppress this one.
        if epoch_start_block is not None and commit_reveal_is_enabled(subtensor, netuid):
            hotkey = getattr(getattr(wallet, "hotkey", None), "ss58_address", None)
            if hotkey is not None:
                commit_block = pending_timelock_commit_block(
                    subtensor, netuid=netuid, hotkey=str(hotkey)
                )
                if commit_block is not None and commit_block >= epoch_start_block:
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
