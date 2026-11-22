"""Protocol v4: the root's join step, and every other validator's audit of it
(docs/decisions/FAST_CROWN_V4.md, section 18; chain_proposals holds the rules).

THE ROOT, in epoch E - 1, after it has set its weights and committed reveal
E - 2 (whose menu hash names base(E)):

  1. waits for block B = E's boundary - JOIN_READ_MARGIN_BLOCKS, and reads,
     pinned at B, every commitment's fields, every hotkey's coldkey, the hash
     of block B and its own first field (the reveal pointer, which names the
     reveal whose verified hotkeys may propose);
  2. fetches from each proposer's box the document its second field names,
     and reads each proposed model's public endpoints listing, within
     JOIN_FETCH_BUDGET_S; a document or listing not had in time is
     "unfetched" or "unread" and its proposal waits on chain;
  3. admits (`chain_proposals.admit`) on base(E) under the list in force;
  4. publishes the delta by hash (create-only) and reads it back, within
     JOIN_PUBLISH_BUDGET_S, then commits [reveal pointer, join pointer]
     (`RootFields`), no later than JOIN_COMMIT_SLACK_BLOCKS before the
     boundary. Nothing admitted: the pointer `join:<epoch id>` alone, no
     bytes. Past any deadline nothing is committed: no joins, and every
     proposal waits for the next epoch.

EVERY OTHER VALIDATOR, at E's boundary, reads the same block B itself and
re-runs the admission on the documents and listings inside the root's delta
(`audit_join`): a difference is an alarm, published in its reveal as
`join_audit`, and the epoch runs on the root's delta as every node does
(when every node's `check_delta` accepts it).

Nothing here runs before a v4 epoch.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

from fugal_subnet import chain_proposals as cp
from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms

logger = logging.getLogger(__name__)

# How many listings (and documents, from a fetcher without `fetch_many`) the
# root fetches at once.
FETCH_WORKERS = 16
# The bytes of one proposer's answer the root reads, at most: the synapse's
# JSON, a document of protocol.PROPOSAL_DOC_MAX_LEN characters and its
# headers fit many times over. Past it the answer is not read (unfetched).
RESPONSE_MAX_BYTES = 65_536
# The reveal field validators publish their audit of the root's join delta in.
AUDIT_FIELD = "join_audit"


class RootFields:
    """The root's commitment as two fields written together: the reveal
    pointer (as before) and the join pointer (`chain_proposals.join_field`).
    Each write carries the newest of both, so committing one never drops the
    other. `write(fields) -> bool` makes the one transaction
    (chain_proposals.set_commitment_fields)."""

    def __init__(self, write: Callable[[list[str]], bool], *, reveal: str | None = None,
                 join: str | None = None):
        self.write, self.reveal, self.join = write, reveal, join

    def fields(self) -> list[str]:
        return [f for f in (self.reveal, self.join) if f]

    def commit_reveal(self, data: str) -> bool:
        """The reveal commitment (reveal_publisher's `commit` hook)."""
        ok = bool(self.write([data] + ([self.join] if self.join else [])))
        if ok:
            self.reveal = data
        return ok

    def commit_join(self, field: str) -> bool:
        """The join pointer, beside the newest reveal pointer. Without a
        reveal pointer to keep in the first field nothing is written (a first
        field that is not a reveal pointer would read as no reveal)."""
        if not self.reveal:
            logger.warning("The root has no reveal pointer on chain yet to keep beside the join "
                           "pointer; the joins wait for its first reveal")
            return False
        ok = bool(self.write([self.reveal, field]))
        if ok:
            self.join = field
        return ok


@dataclass(frozen=True)
class JoinOutcome:
    """What the root's join step did for one epoch."""

    epoch: int
    delta: cp.JoinDelta | None
    delta_bytes: bytes | None
    pointer: cp.JoinPointer | None          # what the root committed (None: nothing)
    detail: str = ""


def _fetch_all(items, fetch, budget_s: float, clock) -> dict:
    """{key: fetch(key) or None} for every key, at most FETCH_WORKERS at once
    and no later than `budget_s` from now: what is not back by then is None."""
    out = dict.fromkeys(items)
    if not items:
        return out
    pool = ThreadPoolExecutor(max_workers=max(1, min(FETCH_WORKERS, len(items))))
    try:
        futures = {pool.submit(fetch, k): k for k in items}
        done, _late = wait(futures, timeout=max(0.0, budget_s))
        for f in done:
            try:
                out[futures[f]] = f.result()
            except Exception as e:  # noqa: BLE001 - one fetch, not the step
                logger.info("A fetch for the join step failed (%s)", type(e).__name__)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return out


def post_all(requests: Mapping, *, budget_s: float, timeout_s: float = cp.JOIN_FETCH_TIMEOUT_S,
             max_bytes: int = RESPONSE_MAX_BYTES) -> dict:
    """{key: response body or None} for `requests` {key: (url, headers,
    json body)}: every POST at once, in ONE event loop with one HTTP session
    of its own (nothing shared with any other caller), no redirect followed,
    each bounded by `timeout_s` and `max_bytes`, and the whole by
    `budget_s`. A request that fails, answers other than 200, sends more
    than `max_bytes` or is not done when the budget ends is None; none of
    them delays or drops another. Never raises."""
    import asyncio
    import threading

    import aiohttp

    out: dict = {}
    lock = threading.Lock()

    async def one(session, key, url, headers, body):
        try:
            async with session.post(url, headers=headers, json=body, allow_redirects=False,
                                    timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                if resp.status != 200:
                    return
                buf = bytearray()
                while True:
                    chunk = await resp.content.read(max_bytes + 1 - len(buf))
                    if not chunk:
                        break
                    buf += chunk
                    if len(buf) > max_bytes:
                        return
            with lock:
                out[key] = bytes(buf)
        except Exception as e:  # noqa: BLE001 - one fetch, not the step
            logger.info("A fetch for the join step failed (%s)", type(e).__name__)

    async def run():
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
            tasks = [asyncio.ensure_future(one(session, k, *r)) for k, r in requests.items()]
            if tasks:
                await asyncio.wait(tasks, timeout=max(0.0, budget_s))
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def main():
        try:
            asyncio.run(run())
        except Exception as e:  # noqa: BLE001 - nothing fetched is what is reported
            logger.warning("The join step's fetches failed (%s)", type(e).__name__)

    if requests:
        worker = threading.Thread(target=main, name="join-fetch", daemon=True)
        worker.start()
        worker.join(max(0.0, budget_s) + 1.0)
    with lock:
        return {k: out.get(k) for k in requests}


class ProposalFetcher:
    """The root's `fetch_document` for `gather`: each document from its
    proposer's axon (protocol.FugalProposalSynapse, signed by `dendrite`'s
    hotkey as a dendrite call is), at the address `axon_of` {hotkey: axon
    info} lists. `fetch_many` fetches a whole epoch's documents at once
    (`post_all`); calling the fetcher fetches one. The answer's `document`
    must be a string of at most protocol.PROPOSAL_DOC_MAX_LEN characters;
    its bytes are checked by hash by the admission."""

    def __init__(self, dendrite, axon_of: Mapping, *, timeout_s: float = cp.JOIN_FETCH_TIMEOUT_S):
        self.dendrite, self.axon_of, self.timeout_s = dendrite, dict(axon_of), timeout_s

    def _request(self, hotkey: str, proposal: str):
        from fugal_subnet.protocol import FugalProposalSynapse

        axon = self.axon_of.get(hotkey)
        if axon is None:
            return None
        syn = self.dendrite.preprocess_synapse_for_request(
            axon, FugalProposalSynapse(proposal=proposal), self.timeout_s)
        url = self.dendrite._get_endpoint_url(axon, request_name=FugalProposalSynapse.__name__)
        return url, syn.to_headers(), syn.model_dump()

    @staticmethod
    def document_of(raw: bytes | None) -> bytes | None:
        import json

        from fugal_subnet.protocol import PROPOSAL_DOC_MAX_LEN

        if raw is None:
            return None
        try:
            obj = json.loads(raw)
        except ValueError:
            return None
        text = obj.get("document") if isinstance(obj, dict) else None
        if not isinstance(text, str) or not text or len(text) > PROPOSAL_DOC_MAX_LEN:
            return None
        return text.encode("ascii", "replace")

    def fetch_many(self, pairs: Sequence[tuple[str, str]], budget_s: float) -> dict:
        """{(hotkey, proposal): document bytes or None} for every pair."""
        reqs = {}
        for pair in pairs:
            try:
                r = self._request(*pair)
            except Exception as e:  # noqa: BLE001 - one request, not the step
                logger.info("A proposal request could not be built (%s)", type(e).__name__)
                r = None
            if r is not None:
                reqs[pair] = r
        got = post_all(reqs, budget_s=budget_s, timeout_s=self.timeout_s)
        return {pair: self.document_of(got.get(pair)) for pair in pairs}

    def __call__(self, hotkey: str, proposal: str) -> bytes | None:
        return self.fetch_many([(hotkey, proposal)], self.timeout_s)[(hotkey, proposal)]


def gather(read: cp.ChainRead, *, fetch_document: Callable[[str, str], bytes | None],
           fetch_listing: Callable[[str], bytes | None], budget_s: float = cp.JOIN_FETCH_BUDGET_S,
           clock: Callable[[], float] = time.monotonic) -> tuple[dict, dict]:
    """({proposal: document}, {proposal: listing}) for the chain read's
    served proposals: each document from its proposer's box
    (`fetch_document(hotkey, proposal)`; a fetcher with `fetch_many`, as
    `ProposalFetcher`, fetches them all at once), each proposed model's
    listing page (`fetch_listing(model)`) parsed for the document's pinned
    hosts (`chain_proposals.listing_from_page`). Bounded by `budget_s` for
    both: a document not back in time is missing, and its proposal is
    "unfetched" and waits on chain."""
    started = clock()
    wanted = [(hk, p) for hk, p in read.proposals if hk in read.serving]
    many = getattr(fetch_document, "fetch_many", None)
    if many is not None:
        try:
            got = dict.fromkeys(wanted) | dict(many(wanted, max(0.0, budget_s)))
        except Exception as e:  # noqa: BLE001 - nothing fetched: every proposal waits
            logger.warning("The join step's document fetch failed (%s)", type(e).__name__)
            got = dict.fromkeys(wanted)
    else:
        got = _fetch_all(wanted, lambda hp: fetch_document(*hp), budget_s, clock)
    docs = {}
    parsed = {}
    for (hk, p), raw in got.items():
        if isinstance(raw, (bytes, bytearray)) and cp.proposal_hash(raw) == p:
            docs[p] = bytes(raw)
            try:
                parsed[p] = cp.parse_document(raw)
            except ValueError:
                pass
    models = sorted({d.model for d in parsed.values()})
    pages = _fetch_all(models, fetch_listing, budget_s - (clock() - started), clock)
    listings = {}
    for p, d in parsed.items():
        page = pages.get(d.model)
        if page is None:
            continue
        try:
            listings[p] = cp.listing_from_page(d.model, page, d.provider_order)
        except ValueError as e:
            logger.info("The listing of %s is not one (%s)", d.model[:60], e)
    return docs, listings


def run_join_step(*, epoch: int, base: ms.MenuState, read: cp.ChainRead,
                  allowlist: mr.MenuAllowlist | None, documents: Mapping, listings: Mapping,
                  publish: Callable[[bytes, str], bool], commit: Callable[[str], bool],
                  params: ms.MenuParams | None = None,
                  time_left: Callable[[], bool] = lambda: True) -> JoinOutcome:
    """Admit, publish and commit epoch `epoch`'s joins on its base menu.
    `publish(bytes, hash) -> bool` puts the delta in the store and reads it
    back (False: not confirmed); `commit(field) -> bool` the root's two
    fields; `time_left()` whether there is still time to commit before the
    boundary. Never raises: a failure commits nothing, and no entry joins."""
    try:
        delta = cp.admit(base, read, documents, listings, allowlist, params)
        raw = cp.delta_bytes(delta)
    except Exception as e:  # noqa: BLE001 - no joins this epoch, never a failed epoch
        logger.error("The join step for epoch %d failed (%s: %s); nothing is committed",
                     epoch, type(e).__name__, e)
        return JoinOutcome(epoch, None, None, None, f"admission failed: {e}")
    h = hashlib.sha256(raw).hexdigest()
    joined = delta.joined()
    if joined:
        try:
            ok = bool(publish(raw, h))
        except Exception as e:  # noqa: BLE001 - not published: not committed
            logger.error("The join delta for epoch %d was not published (%s)", epoch,
                         type(e).__name__)
            ok = False
        if not ok:
            return JoinOutcome(epoch, delta, raw, None, "the delta could not be published")
    pointer = cp.JoinPointer(int(epoch), h if joined else None)
    if not time_left():
        logger.error("The join step for epoch %d ran out of time before the boundary; "
                     "nothing is committed and every proposal waits", epoch)
        return JoinOutcome(epoch, delta, raw, None, "out of time")
    if not commit(cp.join_field(epoch, pointer.delta_hash)):
        return JoinOutcome(epoch, delta, raw, None, "the commitment was not made")
    logger.info("Joins for epoch %d committed: %d admitted of %d considered", epoch,
                len(joined), len(delta.considered))
    return JoinOutcome(epoch, delta, raw, pointer)


def audit_join(*, base: ms.MenuState, read: cp.ChainRead, root_delta_raw: bytes | None,
               root_pointer, allowlist: mr.MenuAllowlist | None,
               params: ms.MenuParams | None = None) -> dict:
    """`join_audit` for epoch `read.epoch`: this validator's own admission
    from its pinned read at B, on the documents and listings inside the
    root's delta (it fetches nothing from miners), against the root's.
    `differences` [] when they agree; an alarm is logged when they differ.
    A root that committed no pointer for the epoch was late or absent: the
    audit says so and differs only if a proposal would have joined."""
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block

    out = {"epoch": epoch_id_for_block(read.epoch), "block": read.block,
           "root": None if root_pointer is None else root_pointer.delta_hash}
    root = None
    if root_delta_raw is not None:
        try:
            root = cp.parse_delta(root_delta_raw)
        except ValueError as e:
            out.update(differences=[f"the root's delta does not parse ({e})"])
            logger.error("ALARM: the root's join delta for %s does not parse", out["epoch"])
            return out
    docs, listings = cp.contents_of(root)
    try:
        expected = cp.admit(base, read, docs, listings, allowlist, params)
    except ValueError as e:
        out.update(differences=None, why=f"this validator's own admission failed ({e})")
        return out
    if root_pointer is not None and root_pointer.delta_hash is None and root is None:
        root = cp.JoinDelta(epoch=read.epoch, base=ms.menu_hash(base), block=read.block,
                            block_hash=read.block_hash, reveal=read.reveal,
                            serving=len(read.serving), considered=())
        diffs = [] if not expected.joined() else [
            f"the root admitted nothing; {len(expected.joined())} proposal(s) join here"]
    elif root_pointer is None:
        out["why"] = "the root committed no join pointer for the epoch (late or absent)"
        diffs = cp.audit(expected, None)
    else:
        diffs = cp.audit(expected, root)
    out["differences"] = diffs[:64] + ([f"... and {len(diffs) - 64} more"] if len(diffs) > 64
                                       else [])
    out["own"] = [c.hotkey for c in expected.joined()]
    if diffs:
        logger.error("ALARM: the root's joins for %s differ from this validator's own (%d "
                     "difference(s), first: %s); the epoch runs on the root's, as every node "
                     "does", out["epoch"], len(diffs), diffs[0])
    return out


def chain_read_at(*, epoch: int, blocks_per_epoch: int, fields_at: Callable[[int], Mapping],
                  coldkeys_at: Callable[[int], Mapping[str, str]],
                  block_hash_at: Callable[[int], str], root_hotkey: str,
                  served_of: Callable[[str, str], Sequence[str] | None]) -> cp.ChainRead:
    """The chain read of epoch `epoch`'s joins, pinned at its block B: every
    commitment's fields and every hotkey's coldkey at B, B's hash, and the
    hotkeys verified in the reveal the root's first field names at B
    (`served_of(epoch id, sha256)`, the reveal's `served_v4`; None when the
    reveal cannot be had). Only the reveal of epoch - 2 lets anyone propose
    (`chain_proposals.reveal_for_joins`): any other names no one."""
    from fugal_subnet import reveal_sync

    block = cp.read_block(epoch, blocks_per_epoch)
    fields = fields_at(block)
    root_fields = fields.get(root_hotkey, ())
    pointer = reveal_sync.parse_commitment(root_fields[0]) if root_fields else None
    reveal, serving = None, ()
    if pointer is not None:
        reveal = (pointer[0], pointer[1])
        if pointer[0] == cp.reveal_for_joins(epoch):
            serving = served_of(pointer[0], pointer[1]) or ()
    return cp.candidates(fields, coldkeys_at(block), epoch=epoch, block=block,
                         block_hash=block_hash_at(block), serving=serving, reveal=reveal)


# The public endpoints listing the root reads for a proposed model.
LISTING_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"


def fetch_public_listing(model: str, *, timeout: float = cp.JOIN_FETCH_TIMEOUT_S,
                         opener=None) -> bytes:
    """One model's public endpoints listing page (no key), bounded in size
    (endpoint_reads.MAX_LISTING_BYTES) and time. Raises on any failure: the
    proposal is then "unread" and waits."""
    from urllib.request import Request, urlopen

    from fugal_subnet.tee import endpoint_reads as er

    if not isinstance(model, str) or len(model) > er.MAX_MODEL_LEN or not er.MODEL_RE.fullmatch(
            model):
        raise ValueError("not a model id")
    req = Request(LISTING_URL.format(model=model), headers={"User-Agent": "fugal-validator"})
    with (opener or urlopen)(req, timeout=timeout) as resp:
        raw = resp.read(er.MAX_LISTING_BYTES + 1)
    if len(raw) > er.MAX_LISTING_BYTES:
        raise ValueError("the listing page is too large")
    return raw
