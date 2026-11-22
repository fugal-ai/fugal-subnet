"""Chain reads pinned to a block, with the archive as the fallback.

A node that is not an archive keeps the state of its last few hundred blocks
(finney's public entrypoint: about 256, measured 2026-11-22). A read pinned
to an older block is refused there, and the protocol pins reads to epoch
boundaries one or more epochs back: the published set from its seed, the
root's commitments, the join reads, the collection-block metagraph.

The SDK's retrying connection moves to its archive endpoint only on its own
StateDiscardedError, which it raises for one wording of the node's error
("Api called for an unknown Block: State already discarded"). Finney's nodes
answer "UnknownBlock: State already discarded for <hash>", which arrives as a
plain SubstrateRequestException, so the connection never moved. Some reads
do not raise at all: the subnet-owner read at a discarded block answers the
storage default, the all-zero account.

`pinned(subtensor, block, read)` runs `read(subtensor)`; when the node says
the block's state is gone (or `refused` says the answer is such a default),
it runs `read` once on each archive endpoint in order, through one cached
connection per endpoint per process, and returns the first answer. The data
is the chain's at that block either way. Reads at the head never come here.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# How the node words a read of state it no longer keeps.
_discarded_markers = ("state already discarded", "state discarded", "unknownblock",
                      "unknown block")
# The all-zero account, the storage default a pruned read can answer.
_zero_account = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
_max_archives = 3
_connections: dict = {}
_lock = threading.Lock()


def is_discarded(e: BaseException) -> bool:
    """Whether a failed read says the node no longer keeps that block's state
    (it will never serve it), rather than a transient failure."""
    if any(t.__name__ == "StateDiscardedError" for t in type(e).__mro__):
        return True
    text = str(e).lower()
    return any(m in text for m in _discarded_markers)


def is_zero_account(value) -> bool:
    return str(value or "") in ("", _zero_account)


def archive_endpoints(subtensor) -> list[str]:
    """The archive endpoints to fall back to, in order: config.ARCHIVE_ENDPOINTS
    (on finney the validator's setup fills it with the public archive when the
    operator set none), else finney's public archive when `subtensor` is a
    finney connection by name (the TD's compose runs `--network finney`)."""
    from fugal_subnet import config

    urls = [u for u in (getattr(config, "ARCHIVE_ENDPOINTS", None) or []) if u]
    if not urls and getattr(subtensor, "network", None) == "finney":
        from fugal_subnet.validator_setup import FINNEY_DEFAULT_ARCHIVE
        urls = [FINNEY_DEFAULT_ARCHIVE]
    return urls[:_max_archives]


def _where(url: str) -> str:
    """An endpoint as it may be logged: scheme and host only (a provider URL
    can carry a key in its path or query)."""
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.hostname}" if parts.hostname else "an archive endpoint"


def _connection(url: str):
    with _lock:
        conn = _connections.get(url)
        if conn is None:
            import bittensor as bt

            conn = _connections[url] = bt.Subtensor(network=url)
        return conn


def pinned(subtensor, block, read, *, what: str = "chain read", refused=None):
    """`read(subtensor)` for a read pinned at `block`, falling back to each
    archive endpoint in order when the node no longer keeps that state (an
    error `is_discarded` recognises, or an answer `refused` rejects). Raises
    the node's error, after an ERROR line naming the endpoints and the block,
    when every archive fails too. `block` None (a head read) never falls back."""
    try:
        got = read(subtensor)
        if block is None or refused is None or not refused(got):
            return got
        first: BaseException = LookupError(f"{what} at block {block}: the node answered the "
                                           "storage default for a block it no longer keeps")
    except Exception as e:
        if block is None or not is_discarded(e):
            raise
        first = e
    urls = archive_endpoints(subtensor)
    if not urls:
        raise first
    logger.info("%s at block %d: this endpoint no longer keeps that state (%s); reading it "
                "from the archive endpoint", what, int(block), type(first).__name__)
    errors = []
    for url in urls:
        try:
            got = read(_connection(url))
        except Exception as e:  # noqa: BLE001 - the next endpoint, then reported
            errors.append(f"{_where(url)}: {type(e).__name__}")
            continue
        if refused is not None and refused(got):
            errors.append(f"{_where(url)}: answered the storage default")
            continue
        return got
    logger.error("%s at block %d failed on this endpoint and on every archive endpoint (%s)",
                 what, int(block), "; ".join(errors))
    raise first
