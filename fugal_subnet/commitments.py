"""On-chain head commitments via subtensor's Commitments pallet.

Miners commit sha256(head bytes) on chain whenever their head changes.
Validators read every hotkey's latest commitment (data + block number) and
use it two ways:

1. **Dedup seniority** — when two heads route (near-)identically, the senior
   one keeps the weight. Seniority is `seniority_key(registration block,
   commitment block)`: the hotkey that registered first is senior, and among
   hotkeys registered in the same block the earlier commitment is. Both come
   from chain state at the epoch boundary, so every validator derives the
   same order and a copier can never claim it. It used to be the earliest
   commitment block each validator had ever SEEN for the hotkey, persisted in
   that validator's own state file -- deterministic for one validator and
   different for each, an I1 hazard the 2026-10-14 diligence review flagged.
2. **Commit-before-nonce enforcement** — a head is only scoreable if its hash
   was committed at or before the epoch boundary block (the block whose hash
   seeds the question slice). A head trained or swapped after the nonce is
   knowable necessarily has a later commitment block and is rejected, closing
   both the slice-overfitting and last-second-copy attacks.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def seniority_key(registration_block: int, commit_block: int) -> int:
    """One integer that orders miners by (registration block, commitment block).

    Lower is senior. Registration comes first because a miner that retrains
    re-commits at a later block, and if the commitment block alone decided,
    the author of an improved head would rank behind whoever copied its
    previous one (tests/test_non_interference.py). Registration is fixed for
    the life of the hotkey on the subnet, is read from the same pinned
    metagraph every validator uses, and cannot be set by a miner after the
    fact. The commitment block breaks ties among hotkeys registered in the
    same block. Packed into one int because `dedup.find_duplicates` orders by
    a number and treats a missing one as infinitely junior.
    """
    return int(registration_block) * 10**10 + int(commit_block)


def get_commitments_with_blocks(
    subtensor, netuid: int, block: int | None = None,
) -> dict[str, tuple[str, int]]:
    """Read all commitments on a subnet as {hotkey_ss58: (data, commit_block)}.

    Mirrors bittensor's Subtensor.get_all_commitments but keeps the block
    number the pallet stores alongside the data (the SDK helper drops it).
    Hotkeys whose commitment fails to decode map to ("", 0).

    `block` pins the read. The validator passes the epoch boundary block, so
    every validator sees the same commitments whatever moment in the epoch it
    happens to run this -- a read at the chain head is a different answer on
    each validator (diligence 2026-10, additional finding A4). None reads the
    head, for tooling that wants the current state.
    """
    try:
        from bittensor.core.chain_data.utils import decode_metadata
    except ImportError:  # a mocked bittensor (tests) carries no decoder
        def decode_metadata(raw):
            return raw if isinstance(raw, str) else ""

    result: dict[str, tuple[str, int]] = {}
    kwargs = {"module": "Commitments", "name": "CommitmentOf", "params": [netuid]}
    if block is not None:
        kwargs["block"] = int(block)
    query = subtensor.query_map(**kwargs)
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
