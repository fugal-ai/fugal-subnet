"""The validator's proof-collection client: a dendrite whose connection pool
never makes one miner wait for another.

bittensor's `Dendrite` builds its HTTP session with `aiohttp.ClientSession()`,
whose default connector allows 100 connections in total, and each query runs
under `ClientTimeout(total=timeout)`. In aiohttp a request's total timeout
includes the time it spends waiting for a free connection. The validator
queries every axon in the metagraph at once (256 on netuid 70), and an axon
that never answers holds its connection for the full timeout. With more than
100 such axons, the rest of the field waits in the pool queue, and some miners
are never contacted before their own timeout expires: the proof was ready and
served, and the validator recorded the miner as absent.

That is not harmless in v2. A miss before burn-in resets the head's burn-in
(`evidence.apply_miss_v2`), so a miner the pool starves can be reset again and
again through no fault of its own. Observed on finney in e00026350 (nine
miners missed, all UID >= 128, one of them reached 127 s after the query
began) and e00026351, from two miners' own logs and firewall records.

The fix is the connection limit and nothing else. Every request still has the
same 120 s total timeout, the same axons are queried, and nothing about
scoring changes, so the consensus digest does not move. The validator already
runs with `LimitNOFILE=65536`, so a few hundred concurrent sockets are well
inside its file-descriptor budget.
"""
from __future__ import annotations

import aiohttp
import bittensor as bt

# 0 means no limit in aiohttp. Collection is one burst of at most one request
# per registered UID (256 on netuid 70), so a bound would only reintroduce
# the queue it exists to remove.
COLLECT_CONNECTION_LIMIT = 0


class CollectDendrite(bt.Dendrite):
    """`bt.Dendrite` with an unbounded connection pool for proof collection."""

    @property
    async def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=COLLECT_CONNECTION_LIMIT),
            )
        return self._session
