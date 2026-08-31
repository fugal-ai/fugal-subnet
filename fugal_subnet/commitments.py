"""On-chain head commitments via subtensor's Commitments pallet.

Miners commit sha256(head bytes) on chain whenever their head changes.
Validators read every hotkey's latest commitment (data + block number) and
use it two ways:

1. **Dedup seniority** — when two heads route (near-)identically, the one
   whose hash was committed at the earliest block keeps the weight. The block
   number comes from chain state, so a copier can never claim seniority.
2. **Commit-before-nonce enforcement** — a head is only scoreable if its hash
   was committed at or before the epoch boundary block (the block whose hash
   seeds the question slice). A head trained or swapped after the nonce is
   knowable necessarily has a later commitment block and is rejected, closing
   both the slice-overfitting and last-second-copy attacks.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_commitments_with_blocks(subtensor, netuid: int) -> dict[str, tuple[str, int]]:
    """Read all commitments on a subnet as {hotkey_ss58: (data, commit_block)}.

    Mirrors bittensor's Subtensor.get_all_commitments but keeps the block
    number the pallet stores alongside the data (the SDK helper drops it).
    Hotkeys whose commitment fails to decode map to ("", 0).
    """
    from bittensor.core.chain_data.utils import decode_metadata

    result: dict[str, tuple[str, int]] = {}
    query = subtensor.query_map(
        module="Commitments", name="CommitmentOf", params=[netuid],
    )
    for hotkey, value in query:
        raw = getattr(value, "value", value)
        block = 0
        if isinstance(raw, dict):
            try:
                block = int(raw.get("block", 0) or 0)
            except (TypeError, ValueError):
                block = 0
        try:
            data = decode_metadata(raw)
        except Exception:
            try:
                data = decode_metadata(value)
            except Exception:
                data = ""
        key = getattr(hotkey, "value", hotkey)
        result[str(key)] = (str(data), block)
    return result


def ensure_commitment(subtensor, wallet, netuid: int, head_hash: str) -> bool:
    """Commit head_hash on chain unless it is already the current commitment.

    Returns True if the chain holds the correct commitment when we're done.
    Never raises — miners keep serving even if the commit fails (the head
    just won't be scoreable until a commitment lands).
    """
    hotkey = wallet.hotkey.ss58_address
    try:
        existing = get_commitments_with_blocks(subtensor, netuid).get(hotkey)
        if existing and existing[0] == head_hash:
            logger.info("Head hash already committed on-chain (block %d)", existing[1])
            return True
    except Exception as e:
        logger.warning("Could not read existing commitments: %s", e)

    try:
        response = subtensor.set_commitment(wallet, netuid, head_hash)
        success, msg = response
        if success:
            logger.info("Committed head hash on-chain: %s...", head_hash[:16])
            return True
        logger.error("set_commitment failed: %s", msg)
    except Exception as e:
        logger.error("set_commitment error: %s", e)
    return False
