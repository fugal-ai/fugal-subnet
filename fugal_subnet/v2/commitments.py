"""Namespaced v2 on-chain commitments with historical block queries."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Sequence

from fugal_subnet.consensus_manifest import canonical_json

NAMESPACES = {"questions": "q", "report": "r", "head": "h"}
HEAD_COMMITMENT_ID = "head-v2"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_COMMITMENT_BYTES = 128


class CommitmentError(RuntimeError):
    pass


def finalized_block_number(subtensor) -> int:
    """Return the finalized height; best-chain heights are never proof bounds."""
    try:
        block_hash = subtensor.substrate.get_chain_finalised_head()
        block = int(subtensor.substrate.get_block_number(block_hash))
    except (AttributeError, TypeError, ValueError) as exc:
        raise CommitmentError("finalized chain head is unavailable") from exc
    if block < 0:
        raise CommitmentError("finalized chain head is invalid")
    return block


def commitment_at_block(
    subtensor,
    *,
    netuid: int,
    uid: int,
    hotkey: str,
    block: int,
) -> str:
    """Read CommitmentOf by its bound hotkey at one exact block.

    Bittensor's convenience ``get_commitment`` rebuilds the latest metagraph
    for every call and then maps UID to hotkey. Historical proofs already bind
    the boundary hotkey, so querying the pallet directly is both safer across
    UID transfers and orders of magnitude cheaper for a transition scan.
    """
    getter = getattr(subtensor, "get_commitment_metadata", None)
    if not callable(getter):
        # Explicit adapter fallback used by small deterministic test doubles.
        return str(subtensor.get_commitment(netuid, uid, block=block) or "")
    metadata = getter(netuid, hotkey, block)
    if metadata == "":
        return ""
    try:
        fields = metadata["info"]["fields"]
        commitment = fields[0]
        if isinstance(commitment, str):
            return ""
        if not isinstance(commitment, Mapping) or len(commitment) != 1:
            raise ValueError("commitment field mapping differs")
        encoded = next(iter(commitment.values()))
        if not isinstance(encoded, str):
            raise ValueError("commitment field is not hex text")
        return bytes.fromhex(encoded.removeprefix("0x")).decode("utf-8")
    except (KeyError, IndexError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise CommitmentError("historical commitment metadata is malformed") from exc


@dataclass(frozen=True)
class HistoricalCommitmentReceipt:
    network: str
    netuid: int
    hotkey: str
    uid: int
    block: int
    block_hash: str
    namespace: str
    epoch_id: str
    artifact_hash: str
    payload: str

    @property
    def receipt_hash(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self))).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "HistoricalCommitmentReceipt":
        expected = {
            "network", "netuid", "hotkey", "uid", "block", "block_hash",
            "namespace", "epoch_id", "artifact_hash", "payload",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise CommitmentError("historical commitment receipt schema differs")
        receipt = cls(**value)
        verify_historical_receipt(receipt)
        return receipt


def commitment_payload(namespace: str, epoch_id: str, artifact_hash: str) -> str:
    if namespace not in NAMESPACES:
        raise CommitmentError("unsupported commitment namespace")
    if not isinstance(epoch_id, str) or not 1 <= len(epoch_id) <= 64:
        raise CommitmentError("commitment epoch_id is invalid")
    if not HASH_RE.fullmatch(artifact_hash):
        raise CommitmentError("commitment artifact hash is invalid")
    epoch_token = hashlib.sha256(epoch_id.encode("utf-8")).hexdigest()[:16]
    payload = f"fugal:v2:{NAMESPACES[namespace]}:{epoch_token}:{artifact_hash}"
    if len(payload.encode("ascii")) > MAX_COMMITMENT_BYTES:
        raise CommitmentError("commitment payload exceeds chain bound")
    return payload


def verify_historical_receipt(receipt: HistoricalCommitmentReceipt) -> None:
    if receipt.namespace not in NAMESPACES:
        raise CommitmentError("receipt namespace is invalid")
    if not isinstance(receipt.netuid, int) or receipt.netuid < 0:
        raise CommitmentError("receipt netuid is invalid")
    if not isinstance(receipt.uid, int) or receipt.uid < 0:
        raise CommitmentError("receipt uid is invalid")
    if not isinstance(receipt.block, int) or receipt.block < 0:
        raise CommitmentError("receipt block is invalid")
    if not isinstance(receipt.network, str) or not receipt.network:
        raise CommitmentError("receipt network is invalid")
    if not isinstance(receipt.hotkey, str) or not receipt.hotkey:
        raise CommitmentError("receipt hotkey is invalid")
    block_hash = receipt.block_hash.removeprefix("0x")
    if not HASH_RE.fullmatch(block_hash):
        raise CommitmentError("receipt block_hash is invalid")
    expected = commitment_payload(receipt.namespace, receipt.epoch_id, receipt.artifact_hash)
    if receipt.payload != expected:
        raise CommitmentError("historical commitment payload differs")


def capture_historical_receipt(
    subtensor,
    *,
    network: str,
    netuid: int,
    hotkey: str,
    uid: int,
    block: int,
    namespace: str,
    epoch_id: str,
    artifact_hash: str,
) -> HistoricalCommitmentReceipt:
    """Query commitment state at an exact block, never merely latest state."""
    expected = commitment_payload(namespace, epoch_id, artifact_hash)
    actual = commitment_at_block(
        subtensor,
        netuid=netuid,
        uid=uid,
        hotkey=hotkey,
        block=block,
    )
    if actual != expected:
        raise CommitmentError("historical on-chain commitment does not match")
    block_hash = str(subtensor.get_block_hash(block))
    receipt = HistoricalCommitmentReceipt(
        network=network,
        netuid=netuid,
        hotkey=hotkey,
        uid=uid,
        block=block,
        block_hash=block_hash,
        namespace=namespace,
        epoch_id=epoch_id,
        artifact_hash=artifact_hash,
        payload=actual,
    )
    verify_historical_receipt(receipt)
    return receipt


def set_commitment_finalized(
    subtensor,
    wallet,
    *,
    netuid: int,
    namespace: str,
    epoch_id: str,
    artifact_hash: str,
) -> None:
    """Submit one namespaced commitment and require chain finalization."""
    payload = commitment_payload(namespace, epoch_id, artifact_hash)
    response = subtensor.set_commitment(
        wallet,
        netuid,
        payload,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    success, message = response
    if not success:
        raise CommitmentError(f"commitment did not finalize: {message}")


def find_historical_receipt(
    subtensor,
    *,
    network: str,
    netuid: int,
    uid: int,
    hotkey: str,
    namespace: str,
    epoch_id: str,
    start_block: int,
    end_block: int,
    artifact_hash: str | None = None,
) -> HistoricalCommitmentReceipt | None:
    """Find the exact transition block for one namespaced commitment.

    Commitment state persists across blocks, so treating the value at a
    deadline as its commit block is incorrect. This scans exact historical
    state and accepts one transition only; equivocation fails closed.
    """
    if namespace not in NAMESPACES or not 0 <= start_block <= end_block:
        raise CommitmentError("historical commitment search bounds are invalid")
    if end_block > finalized_block_number(subtensor):
        raise CommitmentError("historical commitment search exceeds finality")
    if artifact_hash is not None and not HASH_RE.fullmatch(artifact_hash):
        raise CommitmentError("historical commitment search hash is invalid")
    epoch_token = hashlib.sha256(epoch_id.encode("utf-8")).hexdigest()[:16]
    prefix = f"fugal:v2:{NAMESPACES[namespace]}:{epoch_token}:"
    expected = (
        commitment_payload(namespace, epoch_id, artifact_hash)
        if artifact_hash is not None else None
    )
    previous = ""
    if start_block > 0:
        previous = commitment_at_block(
            subtensor,
            netuid=netuid,
            uid=uid,
            hotkey=hotkey,
            block=start_block - 1,
        )
    transitions: list[tuple[int, str]] = []
    for block in range(start_block, end_block + 1):
        current = commitment_at_block(
            subtensor,
            netuid=netuid,
            uid=uid,
            hotkey=hotkey,
            block=block,
        )
        if current != previous and (
            current == expected
            if expected is not None
            else current.startswith(prefix) and HASH_RE.fullmatch(current[len(prefix):])
        ):
            transitions.append((block, current))
        previous = current
    if not transitions:
        return None
    if len(transitions) != 1:
        raise CommitmentError("builder made multiple matching commitments in one epoch")
    block, payload = transitions[0]
    resolved_hash = payload.rsplit(":", 1)[-1]
    return capture_historical_receipt(
        subtensor,
        network=network,
        netuid=netuid,
        hotkey=hotkey,
        uid=uid,
        block=block,
        namespace=namespace,
        epoch_id=epoch_id,
        artifact_hash=resolved_hash,
    )


def collect_historical_receipts(
    subtensor,
    *,
    network: str,
    netuid: int,
    builders: Sequence[tuple[int, str]],
    namespace: str,
    epoch_id: str,
    start_block: int,
    end_block: int,
    artifact_hash: str | None = None,
) -> tuple[HistoricalCommitmentReceipt, ...]:
    """Collect all committee transitions in deterministic hotkey order."""
    if len({uid for uid, _ in builders}) != len(builders) or len(
        {hotkey for _, hotkey in builders}
    ) != len(builders):
        raise CommitmentError("committee identities are duplicated")
    receipts = []
    for uid, hotkey in sorted(builders, key=lambda item: item[1]):
        receipt = find_historical_receipt(
            subtensor,
            network=network,
            netuid=netuid,
            uid=uid,
            hotkey=hotkey,
            namespace=namespace,
            epoch_id=epoch_id,
            start_block=start_block,
            end_block=end_block,
            artifact_hash=artifact_hash,
        )
        if receipt is not None:
            receipts.append(receipt)
    return tuple(receipts)


def submit_commitment_with_receipt(
    subtensor,
    wallet,
    *,
    network: str,
    netuid: int,
    uid: int,
    namespace: str,
    epoch_id: str,
    artifact_hash: str,
) -> HistoricalCommitmentReceipt:
    """Finalize a commitment and prove its exact transition block."""
    before = finalized_block_number(subtensor)
    set_commitment_finalized(
        subtensor,
        wallet,
        netuid=netuid,
        namespace=namespace,
        epoch_id=epoch_id,
        artifact_hash=artifact_hash,
    )
    deadline = time.monotonic() + 30
    after = finalized_block_number(subtensor)
    while after <= before and time.monotonic() < deadline:
        time.sleep(0.25)
        after = finalized_block_number(subtensor)
    if after <= before:
        raise CommitmentError("commitment finalization did not advance the finalized head")
    receipt = find_historical_receipt(
        subtensor,
        network=network,
        netuid=netuid,
        uid=uid,
        hotkey=wallet.hotkey.ss58_address,
        namespace=namespace,
        epoch_id=epoch_id,
        start_block=max(0, before),
        end_block=max(before, after),
        artifact_hash=artifact_hash,
    )
    if receipt is None:
        raise CommitmentError("finalized commitment transition is unavailable historically")
    return receipt
