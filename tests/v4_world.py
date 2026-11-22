"""Stand-ins for the world the shipped neurons run in, shared by the protocol
v4 loop tests (tests/test_v4_loop.py, tests/test_v4_flip.py,
tests/test_v4_seat_loop.py) and the in-process rehearsal
(scripts/rehearsal_v4.py):

- the chain: one fake subtensor (blocks, block hashes, the metagraph, the
  Commitments pallet with pinned reads, weights), shared by every neuron;
- the transport: a dendrite whose query hands each miner's axon answer back
  (`miner._serve_proof`), and a cooperative clock: `time.sleep` in the
  validator module yields to the other validators, and the chain advances
  one block when every validator has slept; a validator can be stopped at a
  block and another started at one (a restart);
- the public store: one bucket, read through `urllib.request.urlopen` (the
  one socket `reveal_sync` opens) and written through the uploader and the
  menu store the validator constructs;
- the model provider: recorded listing pages and stub replies
  (tests/test_harness_v4.Provider), one miner's calls answered right (a TD
  too good to be true).

On these, `FlipWorld` runs the neurons from protocol v3 across the v4 block,
and `SeatLoop` a root and a follower from the v4 block with the seat's
scenarios (`seat_*_scenario`, each property a `check_seat_*`).

Nothing here touches a network, a chain or a paid model.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import threading
import types
import urllib.error
import urllib.parse

import numpy as np

from fugal_subnet import epoch_plan, reveal_publisher, reveal_sync
from tests.test_harness_v4 import Provider as _Provider

NETUID = 7
BUCKET = "fugal-test-reveals"
INTERVAL = 120                        # seconds an epoch: 10 blocks
BPE = 10
GENESIS = "0x" + "4c" * 32            # not finney's
OWNER = ("5Fowner-hotkey", "5Fowner-cold")
ROOT = ("5Froot-hotkey", "5Froot-cold")
FOLLOWER = ("5Ffollower-hotkey", "5Ffollower-cold")
MINERS = [(f"5Fminer-{i}", f"5Fminer-cold-{i}") for i in range(3)]
NEURONS = [OWNER, ROOT, FOLLOWER, *MINERS]
WALLETS = {"root": ROOT[0], "follower": FOLLOWER[0]}


# --- the chain ---------------------------------------------------------------------


class Chain:
    """Blocks, the metagraph and the Commitments pallet (the newest commitment
    of each hotkey, read pinned at a block), and every weight vector set.
    `neurons` are the registered (hotkey, coldkey) pairs in uid order and
    `validators` the hotkeys with stake and a permit."""

    def __init__(self, store, neurons=None, validators=None):
        self.store = store
        self.block = 0
        self.commits: dict[str, list[tuple[int, str]]] = {}
        self.weights: list[tuple[int, str, list, list]] = []
        self.fail_metagraph_at: dict[str, int] = {}
        # Pinned reads of a block the chain has not produced yet, as (thread,
        # block asked, chain block). A real node answers them with nothing,
        # and the SDK caches that nothing (async_substrate_interface keeps
        # chain_getBlockHash results), so the connection can never read that
        # block once it exists: the local-chain v4 rehearsal found a follower
        # stuck at the next boundary that way. No loop may make one.
        self.future_reads: list[tuple[str, int, int]] = []
        # An endpoint that keeps the state of the last `keep` blocks only
        # (None: every block): an older pinned read raises as a pruning node
        # answers (`pruned`); with `keep_threads`, only for the neurons whose
        # threads are named there (theirs is the pruning endpoint).
        self.keep: int | None = None
        self.keep_threads: frozenset | None = None
        self.pruned_reads: list[tuple[str, int, int]] = []
        self.neurons = list(NEURONS if neurons is None else neurons)
        self.validators = set((ROOT[0], FOLLOWER[0]) if validators is None else validators)

    def note_read(self, b) -> None:
        if b is not None and int(b) > self.block:
            self.future_reads.append((threading.current_thread().name, int(b), self.block))

    def pruned(self, b) -> None:
        """Raise as a pruning endpoint does for a pinned read of block `b`
        older than the state it keeps."""
        if (self.keep is not None and b is not None and self.block - int(b) > self.keep
                and (self.keep_threads is None
                     or threading.current_thread().name in self.keep_threads)):
            self.pruned_reads.append((threading.current_thread().name, int(b), self.block))
            raise ConnectionError(f"State discarded for block {int(b)}: the block is too old")

    def block_hash(self, b: int) -> str:
        self.note_read(b)
        return GENESIS if int(b) == 0 else "0x" + hashlib.sha256(f"block-{b}".encode()).hexdigest()

    def commit(self, hotkey: str, data) -> None:
        """One commitment: a text (one field) or a tuple of texts (fields,
        chain_proposals.set_commitment_fields), each at most 128 bytes."""
        fields = tuple(data) if isinstance(data, (tuple, list)) else (data,)
        assert 1 <= len(fields) <= 3
        assert all(len(f.encode()) <= reveal_publisher.COMMITMENT_MAX_BYTES for f in fields)
        if len(fields) >= 2 and fields[1].startswith("join:") and ":" in fields[1][5:]:
            # The bytes a join pointer names are served before it is made.
            h = fields[1].rsplit(":", 1)[1]
            assert reveal_sync.fetch_join_bytes(h, bucket=BUCKET, netuid=NETUID) is not None
        data = fields[0] if len(fields) == 1 else fields
        got = reveal_sync.parse_commitment(fields[0])
        if got is not None and got[2] is not None:
            # Publish before commit: the bytes a commitment names are served
            # by the public store at the moment it is made.
            assert reveal_sync.fetch_menu_bytes(got[2], bucket=BUCKET, netuid=NETUID) is not None
        self.commits.setdefault(hotkey, []).append((self.block, data))

    def at(self, block: int) -> dict[str, tuple[int, str]]:
        out = {}
        for hk, rows in self.commits.items():
            older = [(b, d) for b, d in rows if b <= block]
            if older:
                out[hk] = older[-1]
        return out


class Metagraph:
    def __init__(self, neurons=None, validators=None):
        neurons = list(NEURONS if neurons is None else neurons)
        validators = set((ROOT[0], FOLLOWER[0]) if validators is None else validators)
        self.hotkeys = [hk for hk, _ck in neurons]
        self.coldkeys = [ck for _hk, ck in neurons]
        self.n = len(neurons)
        self.uids = list(range(self.n))
        self.S = [10_000.0 if hk in validators else 0.0 for hk in self.hotkeys]
        self.validator_permit = [hk in validators for hk in self.hotkeys]
        self.block_at_registration = [1] * self.n
        self.axons = [types.SimpleNamespace(ip=f"10.0.0.{i + 1}", port=8091, hotkey=hk)
                      for i, hk in enumerate(self.hotkeys)]


class Subtensor:
    """The part of bittensor's Subtensor the neurons call."""

    def __init__(self, chain: Chain, *_a, **_k):
        self.chain = chain
        self.substrate = types.SimpleNamespace(get_block_hash=chain.block_hash)

    def get_current_block(self):
        return self.chain.block

    def get_block_hash(self, block):
        return self.chain.block_hash(block)

    def metagraph(self, netuid, block=None):
        self.chain.note_read(block)
        self.chain.pruned(block)
        if block is not None and block == self.chain.fail_metagraph_at.get(
                threading.current_thread().name):
            # One failed pinned read, then the chain serves it again.
            del self.chain.fail_metagraph_at[threading.current_thread().name]
            raise ConnectionError("state pruned at this block")
        return Metagraph(self.chain.neurons, self.chain.validators)

    def query_map(self, module, name, params, block=None):
        assert (module, name) == ("Commitments", "CommitmentOf")
        self.chain.note_read(block)
        self.chain.pruned(block)
        rows = self.chain.at(self.chain.block if block is None else block)
        return [(hk, {"block": b, "info": {"fields": [
            {f"Raw{len(f.encode())}": "0x" + f.encode().hex()}
            for f in (d if isinstance(d, tuple) else (d,))]}})
                for hk, (b, d) in sorted(rows.items())]

    def set_commitment(self, wallet, netuid, data):
        self.chain.commit(wallet.hotkey.ss58_address, data)
        return True, ""

    def set_commitment_fields(self, wallet, netuid, fields):
        self.chain.commit(wallet.hotkey.ss58_address, tuple(fields))
        return True

    def get_subnet_hyperparameters(self, netuid=None):
        return types.SimpleNamespace(commit_reveal_weights_enabled=False, commit_reveal_period=0,
                                     weights_rate_limit=0, min_allowed_weights=0,
                                     max_weights_limit=65535, weights_version=0)

    def set_weights(self, wallet, netuid, uids, weights, version_key=0,
                    wait_for_inclusion=False, wait_for_finalization=False,
                    commit_reveal_version=4):
        self.chain.weights.append((self.chain.block, wallet.hotkey.ss58_address,
                                   [int(u) for u in uids], [float(w) for w in weights]))
        return True, "included"

    def weights(self, netuid=None):
        last = {}
        for _b, hk, uids, ws in self.chain.weights:
            last[hk] = (uids, ws)
        hotkeys = [hk for hk, _ck in self.chain.neurons]
        return [(hotkeys.index(hk), [(u, int(round(w * 65535))) for u, w in zip(uids, ws)])
                for hk, (uids, ws) in last.items()]

    def get_subnet_owner_hotkey(self, netuid, block=None):
        return OWNER[0]

    def close(self):
        pass


class Wallet:
    """bt.Wallet: the wallet name (the validator's --coldkey) picks the hotkey."""

    hotkeys = WALLETS

    def __init__(self, name=None, hotkey=None, path=None):
        self.hotkey = types.SimpleNamespace(ss58_address=self.hotkeys[name])


# --- the public store ------------------------------------------------------------------


class Store:
    """One public bucket: create-only puts, GETs and the JSON listing as GCS
    serves them over HTTPS."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def urlopen(self, req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        listing = f"{reveal_sync.GCS}/storage/v1/b/{BUCKET}/o?"
        if url.startswith(listing):
            prefix = urllib.parse.parse_qs(url[len(listing):])["prefix"][0]
            import base64
            items = [{"name": k, "size": str(len(v)),
                      "md5Hash": base64.b64encode(hashlib.md5(v).digest()).decode()}
                     for k, v in sorted(self.objects.items()) if k.startswith(prefix)]
            return io.BytesIO(json.dumps({"items": items}).encode())
        base = f"{reveal_sync.GCS}/{BUCKET}/"
        assert url.startswith(base), f"no network in this test: {url}"
        key = url[len(base):]
        if key not in self.objects:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return io.BytesIO(self.objects[key])


def store_classes(store):
    """The uploader (reveal_publisher.GcsUploader) and the menu store
    (menu_publish.GcsStore) on `store`."""
    class Uploader:
        def __init__(self, bucket, timeout=60.0):
            assert bucket == BUCKET
            self.bucket = bucket

        def upload(self, path, key):
            with open(path, "rb") as f:
                store.objects[key] = f.read()
            return f"gs://{self.bucket}/{key}"

        def upload_if_absent(self, path, key):
            with open(path, "rb") as f:
                store.objects.setdefault(key, f.read())
            return f"gs://{self.bucket}/{key}"

    class MenuStore:
        def __init__(self, bucket, timeout=60.0, opener=None):
            assert bucket == BUCKET

        def put(self, key, data):
            store.objects.setdefault(key, bytes(data))

        def get(self, key, limit):
            return store.objects[key][:limit + 1]

    return Uploader, MenuStore


class Provider(_Provider):
    """The stub provider; one miner's calls are always answered right (a TD
    whose answers are too good to be true, which only the odds pool's
    implausibility check catches: it agrees with the field where it is
    compared on shared questions often enough not to be a frame outlier)."""

    def __init__(self, cheater):
        super().__init__()
        self.cheater, self.serving = cheater, None

    def call(self, proxy, model_id, question, request=None, reasoning=None, deadline_s=None):
        if self.serving != self.cheater:
            return super().call(proxy, model_id, question, request, reasoning)
        got = super().call(proxy, model_id, question, request, reasoning)
        return question["gold"] if got is not None else None


# --- the transport and the clock ----------------------------------------------------------


class Miner:
    """One miner: its head, its v4 session (built by miner._v4_session) and
    what its axon holds (the dict miner.main keeps)."""

    def __init__(self, hotkey, coldkey, head, session):
        self.hotkey, self.coldkey = hotkey, coldkey
        self.session = session
        self.current = {"proof": None, "proof_json": "", "epoch_id": "", "lock": threading.Lock()}
        self.menus: dict[int, str] = {}
        self.measured: dict[int, int] = {}
        self.filed: dict[int, bool] = {}
        self.scored: dict[int, tuple[bool, int]] = {}
        self.spend: dict[int, float] = {}
        self.calls: dict[int, tuple[int, float]] = {}
        # The questions its proof carries an outcome on (scored and exploration).
        self.asked: dict[int, set[str]] = {}
        # Epochs this miner's TD made no proof in, and why.
        self.no_proof: dict[int, str] = {}
        # The proposal document its box serves (chain_proposals), if any.
        self.proposal_doc: bytes | None = None
        self.serve(head)

    def propose(self, spec: dict) -> tuple[str, str]:
        """Serve a proposal document for `spec` (the --v4-proposal JSON) and
        return the two commitment fields: the head and the proposal pointer."""
        from fugal_subnet import chain_proposals as cp
        from neurons import miner as miner_mod

        ps = miner_mod._parse_v4_proposal(json.dumps(spec))
        self.proposal_doc = cp.document_bytes(cp.ProposalDoc(
            proposer=self.hotkey, model=ps.model, reasoning=tuple(ps.reasoning),
            provider_order=tuple(ps.provider_order), quantization=ps.quantization))
        return self.fields()

    def fields(self):
        from fugal_subnet import chain_proposals as cp

        if self.proposal_doc is None:
            return self.weights_hash
        return (self.weights_hash, cp.proposal_field(cp.proposal_hash(self.proposal_doc)))

    def serve(self, head: bytes) -> None:
        """The head this miner's axon serves (a new image serves a new head)."""
        self.head = head
        self.weights_hash = hashlib.sha256(head).hexdigest()
        self.head_b64 = __import__("base64").b64encode(head).decode()


class Dendrite:
    world = None

    def __init__(self, wallet=None, **_k):
        pass

    def query(self, axons, synapse, timeout=None):
        from fugal_subnet.protocol import FugalProposalSynapse
        from neurons import miner as miner_mod

        out = []
        for ax in axons:
            resp = synapse.model_copy(deep=True)
            resp.dendrite.status_code = 200
            m = self.world.miners.get(ax.hotkey)
            if m is not None and isinstance(resp, FugalProposalSynapse):
                resp = miner_mod._serve_proposal(resp, m.proposal_doc)
            elif m is not None:
                resp = miner_mod._serve_proof(resp, m.current, m.head_b64, m.weights_hash)
            out.append(resp)
        return out

    def close_session(self):
        pass

    # The root's proposal fetch (join_step.ProposalFetcher) goes over real HTTP:
    # one loopback server answers every miner's proposal synapse as its axon
    # would (`miner._serve_proposal`), so the fetch runs its own event loop,
    # session, time and size bounds exactly as against the field.
    _server_port: int | None = None
    _server_lock = threading.Lock()

    def preprocess_synapse_for_request(self, axon, synapse, timeout=12.0):
        synapse.timeout = timeout
        return synapse

    def _get_endpoint_url(self, axon, request_name):
        return (f"http://127.0.0.1:{self._proposal_server()}/{request_name}"
                f"?hotkey={urllib.parse.quote(axon.hotkey)}")

    @classmethod
    def _proposal_server(cls) -> int:
        import asyncio

        with cls._server_lock:
            if cls._server_port is not None:
                return cls._server_port
            ready, box = threading.Event(), {}

            def serve():
                from aiohttp import web

                from fugal_subnet.protocol import FugalProposalSynapse
                from neurons import miner as miner_mod

                async def answer(request):
                    syn = FugalProposalSynapse(proposal=(await request.json()).get("proposal", ""))
                    m = cls.world.miners.get(request.query.get("hotkey", ""))
                    if m is not None:
                        syn = miner_mod._serve_proposal(syn, m.proposal_doc)
                    return web.json_response(syn.model_dump())

                async def start():
                    app = web.Application()
                    app.router.add_post("/FugalProposalSynapse", answer)
                    runner = web.AppRunner(app)
                    await runner.setup()
                    site = web.TCPSite(runner, "127.0.0.1", 0)
                    await site.start()
                    box["port"] = site._server.sockets[0].getsockname()[1]

                loop = asyncio.new_event_loop()
                loop.run_until_complete(start())
                ready.set()
                loop.run_forever()

            threading.Thread(target=serve, name="world-proposals", daemon=True).start()
            ready.wait(10)
            cls._server_port = box["port"]
            return cls._server_port


class Lockstep:
    """One validator thread runs at a time. `sleep` hands the turn to the
    next; when every live validator has slept once the chain advances one
    block (and at an epoch boundary the miners run that epoch). Each thread's
    module-level settings (state path, epoch directory, role) are applied
    when it takes the turn: the validators are separate processes in life.

    A restart: `stop_at[name]` is the block at which that validator's loop
    is interrupted (as an operator's Ctrl-C or a crash between epochs), and
    `later[name] = (block, fn)` starts a validator, with its own context,
    when the chain reaches `block`.

    An exception in `on_boundary` (the miners' epoch) fails the run at once:
    it is kept as `failure`, every validator is stopped (the one whose turn
    advanced the chain, and every one waiting for a turn), and `join` raises
    it. A validator's epoch loop catches every Exception of an epoch, so
    without this the run went on, or waited on the clock, past the cause."""

    def __init__(self, chain, contexts, on_boundary, stop_block, *, stop_at=None, later=None):
        self.chain, self.contexts = chain, contexts
        self.later = dict(later or {})
        self.names = [n for n in contexts if n not in self.later]
        self.on_boundary, self.stop_block = on_boundary, stop_block
        self.stop_at = dict(stop_at or {})
        self.cv = threading.Condition()
        self.turn = self.names[0]
        self.finished: set[str] = set()
        self.stop = False
        self.failure: BaseException | None = None
        self.errors: dict[str, BaseException] = {}
        self.threads: list[threading.Thread] = []

    def _apply(self, name):
        for obj, attr, value in self.contexts[name]:
            setattr(obj, attr, value)

    def _pass(self, name):
        n = len(self.names)
        i = self.names.index(name)
        for step in range(1, n + 1):
            j = (i + step) % n
            if self.names[j] in self.finished:
                continue
            if j <= i:
                self._advance()
                if len(self.names) != n:        # a validator joined: it takes the turn
                    self.turn = self.names[-1]
                    break
            self.turn = self.names[j]
            break
        else:
            self.turn = None
            if any(not self.stop and n not in self.finished for n in self.later):
                # Every running validator is down; the chain moves on until
                # the next one starts.
                while self.turn is None and not self.stop:
                    self._advance()
                    pending = [n for n in self.names if n not in self.finished]
                    if pending:
                        self.turn = pending[-1]
        self.cv.notify_all()

    def _advance(self):
        if self.failure is not None:
            raise KeyboardInterrupt
        self.chain.block += 1
        if self.chain.block >= self.stop_block:
            self.stop = True
            return
        if self.chain.block % BPE == 0:
            try:
                self.on_boundary(self.chain.block // BPE)
            except BaseException as e:  # noqa: BLE001 - the run fails with it (join)
                self.failure, self.stop = e, True
                self.cv.notify_all()
                # KeyboardInterrupt ends the validator loop whose turn this
                # is (its epoch loop catches Exception); join raises `e`.
                raise KeyboardInterrupt from e
        for name, (block, fn) in sorted(self.later.items()):
            if block == self.chain.block and name not in self.names:
                self.names.append(name)
                t = threading.Thread(target=self.run, args=(name, fn), name=name, daemon=True)
                self.threads.append(t)
                t.start()

    def sleep(self, _seconds=0):
        name = threading.current_thread().name
        with self.cv:
            self._pass(name)
            if not self.cv.wait_for(lambda: self.turn == name or self.failure is not None,
                                    timeout=900):
                raise RuntimeError("the lockstep clock stalled")
            if self.failure is not None:
                raise KeyboardInterrupt
        self._apply(name)
        if self.stop or self.chain.block >= self.stop_at.get(name, self.stop_block):
            raise KeyboardInterrupt

    def run(self, name, fn):
        with self.cv:
            self.cv.wait_for(lambda: self.turn == name or self.failure is not None, timeout=900)
            if self.failure is not None:
                self.finished.add(name)
                return
        self._apply(name)
        try:
            fn()
        except KeyboardInterrupt:
            pass
        except BaseException as e:  # noqa: BLE001 - reported by the test
            self.errors[name] = e
        finally:
            with self.cv:
                self.finished.add(name)
                if self.failure is None:
                    self._pass(name)
                self.cv.notify_all()

    def start(self, fns: dict):
        """Start the validators that run from the beginning: {name: fn}."""
        for name, fn in fns.items():
            t = threading.Thread(target=self.run, args=(name, fn), name=name, daemon=True)
            self.threads.append(t)
            t.start()

    def join(self, timeout=3600):
        """Wait for every validator, those started later included. Raises the
        exception of a miner epoch (`failure`) as soon as there is one."""
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.failure is not None:
                for t in list(self.threads):
                    t.join(timeout=5)
                raise self.failure
            alive = [t for t in list(self.threads) if t.is_alive()]
            if not alive and (self.stop or all(n in self.finished for n in self.later)):
                return
            for t in alive:
                t.join(timeout=1)
        raise RuntimeError("a validator loop did not stop")


# --- the fixture pool -------------------------------------------------------------------------


def fixture_pool():
    """150 questions in every set of the pinned v4 mix, graded numerically."""
    return [{"question_id": f"{b}-{i:03d}", "prompt": f"Question {b} {i}?", "gold": str(i),
             "grader_id": "numeric_final", "benchmark": b, "metadata": {}}
            for b in sorted(epoch_plan.mix_v4().bench_category) for i in range(150)]


def write_inputs(tmp_path, pool):
    """The pool file and an embeddings file for it, as the loaders read them."""
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(pool))
    rng = np.random.RandomState(0)
    H = (rng.uniform(-1, 1, size=(len(pool), 1024)) * 0.05).astype(np.float32)
    emb = tmp_path / "embeddings.npz"
    np.savez(emb, question_ids=np.array([q["question_id"] for q in pool]), H=H)
    return pool_path, emb


# --- the published set (fugal_subnet/memo_sync.py) in a world ---------------------------------
#
# The release's seed is observed on the pinned pool through a real epoch; a
# world's is observed on its fixture pool through an epoch of the world.
# The fixture's question sets hold 150 questions each, fewer than a category
# needs before the published draws read it at the design's size
# (memo_v4.PUBLISHED_MIN_PER_CATEGORY, 463); at these sizes no draw runs a
# category dry, so a world lowers it, as allow_small_menus lowers the menu's
# minimum. A world that starts its validators after the first v4 epoch (no
# node ran the epochs before) hands each node, and each TD's state
# directory, nothing: the seed is observed two epochs before the first
# epoch they run, and each node advances the set from the chain.


def fixture_unpublished(pool, per_set: int = 2) -> list[str]:
    """The last `per_set` questions of each set: the fixture seed's reserve."""
    by: dict[str, list[str]] = {}
    for q in pool:
        by.setdefault(q["benchmark"], []).append(q["question_id"])
    return sorted(q for ids in by.values() for q in ids[len(ids) - per_set:] if per_set)


def fixture_seed(pool, through: int, unpublished) -> dict:
    """A published-set seed (memo_v4.SEED_FORMAT) for `pool`, observed
    through epoch `through`: every question published but `unpublished`."""
    from fugal_subnet import memo_v4
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block

    left = set(unpublished)
    per_set: dict[str, int] = {}
    un: dict[str, list[str]] = {}
    for q in pool:
        per_set[q["benchmark"]] = per_set.get(q["benchmark"], 0) + 1
        if q["question_id"] in left:
            un.setdefault(q["benchmark"], []).append(q["question_id"])
    return {"format": memo_v4.SEED_FORMAT, "through": epoch_id_for_block(int(through)),
            "pool": {"pool_hash": "fixture", "n_questions": len(pool),
                     "per_set": dict(sorted(per_set.items()))},
            "observations_sha256": "0" * 64,
            "unpublished": {b: sorted(v) for b, v in sorted(un.items())}}


def install_seed(tmp_path, setattr_, doc, *, min_per_category: int = 1,
                 fresh_epochs: int | None = None):
    """The world's release ships `doc` as its published-set seed."""
    from fugal_subnet import memo_v4

    path = tmp_path / "published_v4_seed.json"
    path.write_bytes(json.dumps(doc, sort_keys=True).encode())
    setattr_(memo_v4, "SEED_PATH", str(path))
    setattr_(memo_v4, "seed_doc", lambda: doc)
    setattr_(memo_v4, "PUBLISHED_MIN_PER_CATEGORY", min_per_category)
    if fresh_epochs is not None:
        setattr_(memo_v4, "FRESH_RESERVE_EPOCHS", fresh_epochs)


def first_field(data):
    """A commitment's first field: the text itself, or a tuple's first."""
    return data[0] if isinstance(data, tuple) else data


def chain_view(chain, doc, pool, reveals: dict, epoch: int, root: str | None = None):
    """The published set as of `epoch` by the rule (fugal_subnet/memo_sync.py),
    computed here from the fake chain alone: from the seed, at each boundary
    the root's commitment newly names a reveal its questions are folded (as
    first published two epochs before the boundary), else nothing. `reveals`
    are the root's reveals by epoch. What every node must hold."""
    from fugal_subnet import memo_v4, reveal_sync

    root = ROOT[0] if root is None else root
    pub = memo_v4.seed_published(pool, doc)
    named = pub.through
    for b in range(pub.through + memo_v4.READ_LAG + 1, int(epoch) + 1):
        got = reveal_sync.parse_commitment(first_field(chain.at(b * BPE).get(root, (0, ""))[1]))
        ids: set = set()
        if got is not None and memo_v4.epoch_index(got[0]) > named:
            named = memo_v4.epoch_index(got[0])
            ids = memo_v4.published_ids(reveals[named])
        pub = memo_v4.fold(pub, b - memo_v4.READ_LAG, ids)
    return memo_v4.fold(pub.view(int(epoch) - memo_v4.READ_LAG), int(epoch) - 1, ())


def check_held(path, chain, doc, pool, reveals: dict, last: int):
    """The published set a node keeps at `path` (memo_sync.HELD_FORMAT) is the
    one `chain_view` gives as of the boundary it last read, which is `last`
    or later. Returns that boundary."""
    from fugal_subnet import memo_sync, memo_v4

    with open(path, "rb") as f:
        kept = json.loads(f.read())
    assert kept["format"] == memo_sync.HELD_FORMAT, str(path)
    at = kept["at"]
    assert at >= int(last), (str(path), at)
    held = memo_v4.published_from_doc(kept["set"])
    want = chain_view(chain, doc, pool, reveals, at)
    assert held.digest == want.view(at - memo_v4.READ_LAG).digest, str(path)
    return at


def memo_key(memo):
    """What identifies an epoch's memorisation input (memo_v4.EpochMemo):
    the epoch, the view its draws read (through, digest, reserve, F), its
    fresh questions and the set its labels are split on; None for none."""
    if memo is None:
        return None
    i = memo.index
    return (i.epoch, i.through, i.digest, i.reserve_size, i.fresh_count, tuple(memo.fresh),
            memo.published.digest)


def spy_v2_calls(setattr_) -> list:
    """Record every `run_v2_epoch` call of a run, as (validator thread,
    epoch, whether it runs inside a re-anchor's replay (v4_replay.replay),
    `memo_key` of its memo), around whatever `run_v2_epoch` is when called."""
    from fugal_subnet import v2_epoch, v4_replay

    calls: list = []
    inside = threading.local()
    inner_run, inner_replay = v2_epoch.run_v2_epoch, v4_replay.replay

    def run(**kw):
        calls.append((threading.current_thread().name, int(kw["epoch_index"]),
                      bool(getattr(inside, "replay", False)), memo_key(kw.get("memo"))))
        return inner_run(**kw)

    def replay(*a, **kw):
        inside.replay = True
        try:
            return inner_replay(*a, **kw)
        finally:
            inside.replay = False

    setattr_(v2_epoch, "run_v2_epoch", run)
    setattr_(v4_replay, "replay", replay)
    return calls


def held_memo_for(state_dir, pool):
    """The replay's memo function (neurons/validator._v4_memo_for) on the
    published set a validator keeps beside its state in `state_dir`, read
    as it is: no chain read and no fold."""
    from fugal_subnet import memo_sync
    from neurons import validator

    def no_read(*_a):
        raise AssertionError("a replay's memo reads nothing from the chain or the store")

    chain_set = memo_sync.ChainSet(pool=pool, read_at=no_read, reveal_bytes=no_read,
                                   path=str(state_dir / "state.json") + memo_sync.STATE_SUFFIX)
    assert chain_set.held is not None, str(state_dir)
    return validator._v4_memo_for(chain_set, pool)


def check_replayed_memos(calls, threads, *, root=("root",)) -> dict[str, list[int]]:
    """Every epoch the validator threads `threads` replayed in a re-anchor
    ran with the memo the root's live epoch ran with (never none): the
    replay's memo function (neurons/validator._v4_memo_for) gives the epoch
    loop's memo. `root`: the root's threads (one per process). Returns the
    epochs each thread replayed."""
    root = {root} if isinstance(root, str) else set(root)
    live = {e: key for t, e, rep, key in calls if t in root and not rep}
    out: dict[str, list[int]] = {}
    for t, e, rep, key in calls:
        if rep and t in threads:
            assert key is not None, (t, e, "a replayed v4 epoch ran with no memo")
            assert key == live[e], (t, e, "a replayed epoch's memo is not the root's")
            out.setdefault(t, []).append(e)
    return out


# --- a world that crosses the v4 activation block ---------------------------------------------


V2_EPOCH, V3_EPOCH = 1, 5


@dataclasses.dataclass(frozen=True)
class ValidatorSpec:
    """One validator process. `wallet` names its hotkey; `state` its state
    directory (a restarted validator is a second spec with the same wallet
    and state); `start` the block its process starts at (None: with the
    run) and `stop` the block its loop is interrupted at; `v4` whether its
    release schedules protocol v4 (a control does not); `once` an operator's
    `--once` run (the current epoch, run again when it was already run)."""
    name: str
    wallet: str
    state: str
    root: bool = False
    start: int | None = None
    stop: int | None = None
    v4: bool = True
    once: bool = False


# The challenge an "on" miner answers each epoch in these worlds: the smaller
# of the two sizes (config.CHALLENGE_SIZES_V4), which keeps the loops quick.
CHALLENGE = 500


def challenge_size(mode: str) -> int:
    """A miner spec's mode as its --v4-challenge: "on" a challenge of
    CHALLENGE questions, "1000" of 1,000; "off" and "auto" none."""
    sizes = {"on": CHALLENGE, "1000": 1000, "500": 500, "off": 0, "auto": 0}
    if mode not in sizes:
        raise ValueError(f"a miner's mode is one of {sorted(sizes)}, not {mode!r}")
    return sizes[mode]


@dataclasses.dataclass(frozen=True)
class MinerSpec:
    """One miner: its keys, its heads' seed, whether its TD answers the scored
    questions under v4 (`auto` reads itself as the holder, as in
    tests/test_v4_loop.py), and the proposal its box serves and commits:
    with its v4 head before the flip, or early in epoch `propose_at`
    (before the root's pinned read in that epoch)."""
    hotkey: str
    coldkey: str
    seed: int
    scored: str = "off"
    proposal: dict | None = None
    propose_at: int | None = None


def v3_head(seed: int) -> bytes:
    """A v3 head on v3 menu A, built against its pinned token manifest, as a
    v3 miner's head is."""
    from fugal_subnet import routing_protocol
    from fugal_subnet.config import HEAD_HIDDEN_DIM
    from fugal_subnet.vendor import success_contract as contract

    models = list(routing_protocol.models_for_protocol(3))
    manifest = json.loads(routing_protocol.MANIFEST_PATH_V3.read_text())
    rng = np.random.RandomState(seed)
    W = (rng.randn(len(models), HEAD_HIDDEN_DIM) * 0.01).astype(np.float32)
    b = rng.randn(len(models)).astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, **contract.make_head(W, b, models, manifest, "flip-test"))
    return buf.getvalue()


def validator_keys(wallet: str) -> tuple[str, str]:
    if wallet == "root":
        return ROOT
    return (f"5F{wallet}-hotkey", f"5F{wallet}-cold")


class FlipWorld:
    """The shipped neurons from protocol v3 across the v4 activation block:
    validators running `neurons/validator.main` (each its real epoch loop)
    and miners running `neurons/miner._run_epoch`, a v3 image's epochs before
    the block and a v4 image's (on the session `miner._v4_session` builds,
    with the v4 head committed after the last v3 boundary) from it.

    The pinned seed is a fixture menu (tests/menu_fixtures.state, naming no
    signed list, as the release's seed does), and the release ships the
    fixture lab and host list, signed by the test's signer, in place of
    fugal_subnet/signed_lists/. `patch(obj, attr, value)` and
    `setenv(name, value)` are a monkeypatch's (pytest.MonkeyPatch outside a
    test)."""

    def __init__(self, tmp_path, patch, setenv, signer, *, flip: int, first: int, last: int,
                 validators: list[ValidatorSpec], miners: list[MinerSpec],
                 cheater: str | None = None, on_epoch=None):
        self.tmp, self.patch, self.setenv, self.signer = tmp_path, patch, setenv, signer
        self.flip, self.first, self.last = int(flip), int(first), int(last)
        self.specs, self.miner_specs = list(validators), list(miners)
        self.cheater, self.on_epoch = cheater, on_epoch
        self.errors: list[tuple[str, str]] = []
        self.provider = Provider(cheater=cheater)
        self.store = Store()
        wallets = sorted({s.wallet for s in self.specs})
        self.validator_keys = {w: validator_keys(w) for w in wallets}
        neurons = [OWNER, *self.validator_keys.values(),
                   *[(m.hotkey, m.coldkey) for m in self.miner_specs]]
        self.chain = Chain(self.store, neurons=neurons,
                           validators={hk for hk, _ck in self.validator_keys.values()})
        self.dirs = {s.state: tmp_path / s.state for s in self.specs}

    # -- setup ------------------------------------------------------------------------

    def _seed(self):
        from fugal_subnet import menu_state as ms
        from tests import menu_fixtures as fx

        return fx.state(epoch=0, serial=0, labs=ms.AllowlistRef(0, ""))

    def setup(self):
        import bittensor

        from fugal_subnet import collect, config
        from fugal_subnet import menu_publish as mp
        from fugal_subnet import menu_rules as mr
        from fugal_subnet import menu_state as ms
        from fugal_subnet.tee import harness
        from fugal_subnet.tee import runtime as tee_runtime
        from tests.test_menu_publish import LIST_BYTES

        p = self.patch
        p(ms, "READER_PARAMS", dataclasses.replace(ms.READER_PARAMS, min_entries=1))
        p(config, "EPOCH_INTERVAL", INTERVAL)
        p(config, "V2_ACTIVATION_BLOCK", V2_EPOCH * BPE)
        p(config, "V3_ACTIVATION_BLOCK_ENV", V3_EPOCH * BPE)
        p(config, "V4_ACTIVATION_BLOCK_ENV", self.flip * BPE)
        p(config, "ODDS_SEAT_BLOCK_ENV", None)
        p(config, "MENU_B_BLOCK_ENV", None)
        p(config, "SINGLE_MODEL_MIN_PAIRS", 60)
        p(config, "REFERENCE_BUCKET", BUCKET)
        p(config, "ROOT_HOTKEY_ENV", ROOT[0])
        p(config, "PUBLISH_HEADS_AFTER_EPOCHS", 0)

        self.pool = fixture_pool()
        # The release's published-set seed, observed through the epoch before
        # the world's first (every node folds the v3 reveals after it at the
        # block): two questions of each set unpublished.
        self.published_seed = fixture_seed(self.pool, self.first - 1,
                                           fixture_unpublished(self.pool))
        install_seed(self.tmp, p, self.published_seed)
        pool_path, emb_path = write_inputs(self.tmp, self.pool)
        for var in ("FUGAL_BENCHMARK_POOL", "FUGAL_BENCHMARK_POOL_V2", "FUGAL_BENCHMARK_POOL_V3"):
            self.setenv(var, str(pool_path))
        self.setenv("FUGAL_POOL_UNPINNED", "1")
        self.setenv("FUGAL_EMBEDDINGS_FILE", str(emb_path))

        import urllib.request
        p(urllib.request, "urlopen", self.store.urlopen)
        p(bittensor, "Subtensor", lambda *a, **k: Subtensor(self.chain))
        hotkeys = {w: hk for w, (hk, _ck) in self.validator_keys.items()}
        p(Wallet, "hotkeys", hotkeys)
        p(bittensor, "Wallet", Wallet)
        p(collect, "CollectDendrite", Dendrite)
        uploader_cls, menu_store_cls = store_classes(self.store)
        p(reveal_publisher, "GcsUploader", uploader_cls)
        p(mp, "GcsStore", menu_store_cls)
        p(harness, "_call_model", self.provider.call)
        p(tee_runtime.MeteringProxy, "start", lambda self: None)
        p(tee_runtime.MeteringProxy, "stop", lambda self: None)
        from fugal_subnet import join_step
        p(join_step, "fetch_public_listing", lambda model, **_k: self.provider.fetch(model))

        # The release's pinned seed and its shipped signed list.
        self.seed = self._seed()
        p(ms, "load_seed", lambda: self.seed)
        lists = self.tmp / "signed_lists"
        lists.mkdir()
        (lists / mr.SHIPPED_LISTS[mr.KIND_MENU_ALLOWLIST]).write_bytes(LIST_BYTES)
        (lists / (mr.SHIPPED_LISTS[mr.KIND_MENU_ALLOWLIST] + ".sig")).write_bytes(
            self.signer.sign(LIST_BYTES))
        p(mr, "SIGNED_LISTS_DIR", lists)
        for d in self.dirs.values():
            (d / "epochs").mkdir(parents=True, exist_ok=True)
        self._miners()

    def _miners(self):
        from fugal_subnet import config
        from fugal_subnet.routing_protocol import models_for_protocol
        from fugal_subnet.tee import runtime as tee_runtime  # noqa: F401
        from neurons import miner as miner_mod
        from tests.test_tee_e2e import _head

        self.chain.block = 1
        self.hidden = miner_mod._load_pinned_embeddings(self.pool, mock=True)
        self.models_v3 = list(models_for_protocol(3))
        self.miners: dict[str, Miner] = {}
        self.heads_v4: dict[str, bytes] = {}
        world = types.SimpleNamespace(miners=self.miners)
        for spec in self.miner_specs:
            head_v3 = v3_head(spec.seed)
            self.chain.commit(spec.hotkey, hashlib.sha256(head_v3).hexdigest())
            self.heads_v4[spec.hotkey] = _head(list(self.seed.ids()), seed=100 + spec.seed)
            m = Miner(spec.hotkey, spec.coldkey, head_v3, self.session(spec))
            self.miners[spec.hotkey] = m
        Dendrite.world = world
        assert config.protocol_for_chain(GENESIS) == 4

    def session(self, spec: MinerSpec, state_dir=None):
        """The miner's v4 session as miner.main builds it (a restart builds it again)."""
        from neurons import miner as miner_mod

        # A miner "on" answers a challenge of CHALLENGE questions every epoch;
        # "off" and "auto" (the holder) answer none (FAST_CROWN_V4.md 6).
        scored = dict(challenge_size=challenge_size(spec.scored))
        return miner_mod._v4_session(
            Subtensor(self.chain), netuid=NETUID, pool=self.pool, first_epoch=self.flip,
            blocks_per_epoch=BPE, coldkey=spec.coldkey, genesis=GENESIS,
            fetch_listing=self.provider.fetch, state_dir=state_dir, **scored)

    # -- the miners' epochs ---------------------------------------------------------------

    def run_miners(self, epoch: int):
        import math

        from fugal_subnet import config
        from fugal_subnet.benchmarks.slicer import epoch_id_for_block
        from fugal_subnet.routing_protocol import menu_for_epoch, price_table
        from fugal_subnet.tee import runtime as tee_runtime
        from neurons import miner as miner_mod

        if epoch > self.last:
            return
        config.V4_ACTIVATION_BLOCK_ENV = self.flip * BPE
        if self.on_epoch is not None:
            self.on_epoch(self, epoch)
        for spec in self.miner_specs:
            m = self.miners[spec.hotkey]
            self.provider.serving = m.hotkey
            if spec.proposal and spec.propose_at == epoch:
                # Its document served from its box, its hash committed beside
                # its head, one transaction, early in the epoch.
                self.chain.commits[spec.hotkey].append((epoch * BPE + 1, m.propose(spec.proposal)))
            if epoch < self.flip:
                miner_mod._run_epoch(
                    blocks_per_epoch=BPE, head_data=m.head, weights_hash=m.weights_hash,
                    pool=self.pool, hidden_states=self.hidden, epoch_index=epoch,
                    block_hash=self.chain.block_hash(epoch * BPE),
                    tee_runtime=tee_runtime.TEERuntime(mock=True), current_proof=m.current,
                    slice_size=config.SLICE_SIZE, proxy_port=0, explore_models=self.models_v3,
                    explore_size=max(1, int(round(config.SLICE_SIZE * config.EXPLORE_FRACTION))),
                    hotkey=m.hotkey, protocol_version=3, prices=price_table(3),
                    block_now=lambda: self.chain.block,
                    menu=menu_for_epoch(3, epoch * BPE, None))
            else:
                if epoch == self.flip:
                    m.serve(self.heads_v4[m.hotkey])      # the v4 image serves the v4 head
                from fugal_subnet.memo_sync import PublishedUnavailable
                try:
                    miner_mod._run_epoch(
                        blocks_per_epoch=BPE, head_data=m.head, weights_hash=m.weights_hash,
                        pool=self.pool, hidden_states=self.hidden, epoch_index=epoch,
                        block_hash=self.chain.block_hash(epoch * BPE),
                        tee_runtime=tee_runtime.TEERuntime(mock=True), current_proof=m.current,
                        slice_size=config.SLICE_SIZE_V4, proxy_port=0, explore_models=[],
                        explore_size=0, hotkey=m.hotkey, protocol_version=4, prices=None,
                        block_now=lambda: self.chain.block, v4_session=m.session)
                except PublishedUnavailable as e:
                    # The TD cannot reach the published set the epoch draws on
                    # (the root's reveals not to be had): no proof this epoch,
                    # its axon still serves the last one, as main's does.
                    m.no_proof[epoch] = str(e)
                    continue
            proof = m.current["proof"]
            assert proof.epoch_id == epoch_id_for_block(epoch)
            m.menus[epoch] = getattr(proof, "menu_hash", None)
            m.filed[epoch] = m.proposal_doc is not None
            m.scored[epoch] = (getattr(proof, "scored", True), len(proof.scored_results))
            m.spend[epoch] = proof.total_cost_usd
            m.calls[epoch] = (len(proof.exploration_results),
                              math.fsum(r.cost_usd for r in proof.exploration_results))
        if epoch == self.flip - 1:
            # The v4 image commits its head after the last v3 boundary, so
            # that epoch still checks the v3 head and the flip's boundary
            # reads the v4 one (miner._v2_commit_window, at this chain's pace).
            for spec in self.miner_specs:
                m = self.miners[spec.hotkey]
                head = hashlib.sha256(self.heads_v4[spec.hotkey]).hexdigest()
                if spec.proposal and spec.propose_at is None:
                    # The proposal is committed with the v4 head, one transaction.
                    m.weights_hash = head
                    data = m.propose(spec.proposal)
                else:
                    data = head
                self.chain.commits[spec.hotkey].append(((self.flip - 1) * BPE + 2, data))

    # -- the validators ---------------------------------------------------------------------

    def contexts(self):
        from fugal_subnet import commit_reveal, config, epoch_logger
        from neurons import validator

        out = {}
        for s in self.specs:
            d = self.dirs[s.state]
            out[s.name] = [(validator, "STATE_PATH", str(d / "state.json")),
                           (commit_reveal, "EPOCH_DIR", str(d / "epochs")),
                           (epoch_logger, "EPOCH_LOG_DIR", str(d / "logs")),
                           (config, "ROOT_VALIDATOR", True if s.root else None),
                           (config, "REVEAL_BUCKET", BUCKET if s.root else ""),
                           (config, "V4_ACTIVATION_BLOCK_ENV",
                            self.flip * BPE if s.v4 else None)]
        return out

    def run(self):
        import sys
        import time as _real_time

        from neurons import validator

        def start(spec):
            return lambda: validator.main.callback(
                network="local", netuid=NETUID, coldkey=spec.wallet, hotkey="default",
                wallet_path=None, once=spec.once, log_level="WARNING", live=False)

        if any(s.once for s in self.specs):
            # A `--once` process ends with os._exit after logging.shutdown();
            # here it ends its thread instead, and the test's logging stays.
            import logging as _logging
            import os as _os

            class _Os:
                def __getattr__(self, name):
                    return getattr(_os, name)

                @staticmethod
                def _exit(code):
                    raise KeyboardInterrupt

            class _Logging:
                def __getattr__(self, name):
                    return getattr(_logging, name)

                @staticmethod
                def shutdown():
                    pass

            self.patch(validator, "os", _Os())
            self.patch(validator, "logging", _Logging())
        later = {s.name: (s.start, start(s)) for s in self.specs if s.start is not None}
        stop_at = {s.name: s.stop for s in self.specs if s.stop is not None}
        self.clock = Lockstep(self.chain, self.contexts(), self.run_miners,
                              stop_block=(self.last + 1) * BPE + 1, stop_at=stop_at, later=later)
        self.patch(validator, "time", types.SimpleNamespace(
            sleep=self.clock.sleep, time=_real_time.time, monotonic=_real_time.monotonic))
        real_exception = validator.logger.exception

        def record(msg, *a, **k):
            self.errors.append((threading.current_thread().name, str(sys.exc_info()[1])))
            real_exception(msg, *a, **k)

        self.patch(validator.logger, "exception", record)
        # Every epoch's memo, live and replayed (check_replayed_memos).
        self.v2_calls = spy_v2_calls(self.patch)
        self.chain.block = self.first * BPE
        self.run_miners(self.first)
        self.clock.start({s.name: start(s) for s in self.specs if s.start is None})
        self.clock.join()
        return self

    # -- what the run left ------------------------------------------------------------------

    def reveal(self, state: str, k: int) -> dict | None:
        from fugal_subnet.benchmarks.slicer import epoch_id_for_block

        path = self.dirs[state] / "epochs" / epoch_id_for_block(k) / "reveal.json"
        return json.loads(path.read_text()) if path.exists() else None

    def logged(self, state: str) -> list[dict]:
        path = self.dirs[state] / "logs" / "epochs.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def held(self, state: str):
        from fugal_subnet import menu_publish as mp

        return mp.MenuBook(str(self.dirs[state] / "state.json") + ".menu")

    def weights(self) -> dict[int, dict[str, dict]]:
        by: dict[int, dict[str, dict]] = {}
        for block, hk, uids, ws in self.chain.weights:
            by.setdefault(block // BPE, {})[hk] = dict(zip(uids, ws))
        return by


# --- the seat through the shipped loops (tests/test_v4_seat_loop.py, scripts/rehearsal_v4.py) ---
#
# A root and a follower (`neurons/validator.main`) and the miners' real
# `_run_epoch` on the stand-ins above, from v4's activation on, at the
# design's constants (1,500 labels, 0.1 Brier points, 1,000 questions, 1.5
# points, G = 3). The stub provider answers a (question, entry) right exactly
# when the question's pinned embedding is positive on the entry's own
# coordinate, so a head whose row for the entry weighs that coordinate
# predicts the outcome, and a sharper weight predicts it better: whether a
# head beats R or another head is set by its weights, not by luck. The
# follower is handed every epoch's proofs, heads, answers and commitments in
# the reverse order of the root's.

SEAT_V4_EPOCH = 10                    # the first v4 epoch of the seat loops
SEAT_FIRST = 20                       # the first epoch both validators run
SEAT_DIM = 1024
PROPOSAL = {"model": "lab-c/model-new", "reasoning": ["effort", "high"],
            "provider_order": ["host-a/fp8"], "quantization": "fp8"}
NEW_ID = "lab-c/model-new@high"
# The v1-v3 seat's state and reveal fields that v4 does not write.
SEAT_REMOVED_FIELDS = {"pending", "confirm_from", "ghosts", "high_water", "high_water_hard",
                       "burn_in", "frame_excluded_coldkeys"}
SHARP, SHARPER = 60.0, 240.0


def _keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from _keys(v)


class SeatProvider(_Provider):
    """Right exactly when the question's embedding is above the entry's
    threshold on the entry's coordinate (`coord`), every call billed as the
    stub bills it. The threshold is 0 except for the proposed entry, whose
    calls are right about 70% of the time, so the field's labels of it clear
    the per-category floor of one half."""

    SHIFT = {NEW_ID: 0.02}

    def __init__(self, hidden_of, coord, cheater=None):
        super().__init__()
        self.hidden_of, self.coord = hidden_of, coord
        # A hotkey whose TD's every call is answered right (too good to be
        # true: only the label pool's plausibility check against R catches it).
        self.cheater, self.serving = cheater, None

    def call(self, proxy, model_id, question, request=None, reasoning=None, deadline_s=None):
        got = super().call(proxy, model_id, question, request, reasoning)
        if got is None:
            return None
        if self.cheater is not None and self.serving == self.cheater:
            return question["gold"]
        eid = model_id
        j = self.coord.get(eid)
        right = j is not None and (float(self.hidden_of[question["question_id"]][j])
                                   > -self.SHIFT.get(eid, 0.0))
        return question["gold"] if right else "0"


def seat_head_bytes(rows: dict[str, float], coord: dict[str, int], tag: str,
                    noise: float = 0.0) -> bytes:
    """A v4 head whose row for each entry in `rows` weighs the entry's
    coordinate by the given sharpness (0: a row that says 50%), plus a
    little noise on every weight when `noise` (a lightly altered copy)."""
    from fugal_subnet import success_test_fixtures as sfx

    models = sorted(rows)
    W = np.zeros((len(models), SEAT_DIM), dtype=np.float32)
    for i, m in enumerate(models):
        W[i, coord[m]] = rows[m]
    if noise:
        W += np.random.default_rng(int.from_bytes(tag.encode()[:8], "big")).normal(
            0, noise, W.shape).astype(np.float32)
    z = dict(sfx.arrays(W, np.zeros(len(models), np.float32), models))
    z["provenance"] = np.array(f"seat-loop:{tag}")
    for key in ("mean_in_tokens", "mean_out_tokens", "cost_profile_id"):
        z.pop(key, None)
    buf = io.BytesIO()
    np.savez(buf, **z)
    return buf.getvalue()


class SeatLoop:
    """The two validator loops and the miners' epochs, with a schedule:
    `plan(epoch, loop)` runs at each boundary before the miners, and may
    change a miner's head (and commitment), its scored-question mode, or
    leave it out of the epoch (`loop.down`). `patch` is a monkeypatch (its
    `setattr` and `setenv`); `signature` the shipped list's signature by the
    release signer `menu_rules.list_signers` names. `miners` are (hotkey,
    coldkey, head bytes, scored mode, commit block) and may be filled after
    construction (heads are built with `head`, which needs the loop's
    coordinates)."""

    def __init__(self, tmp_path, patch, miners, *, last, plan, signature=None,
                 proposal_from=None, cheater=None, follower_down=None,
                 unpublished=(), fresh_epochs=None, keep=None, keep_threads=None):
        from fugal_subnet import config
        from tests import menu_fixtures as fx

        self.tmp_path, self.last, self.plan = tmp_path, last, plan
        # A hotkey answered right on every call (SeatProvider.cheater), and
        # the epochs (first, last) the follower's process is down for: it is
        # stopped before `first` and restarted at `last + 1` from its state.
        self.cheater, self.follower_down = cheater, follower_down
        # Blocks of state the chain endpoint keeps (Chain.keep; None: all).
        self.keep, self.keep_threads = keep, keep_threads
        self._signature = signature
        # The release's published-set seed: observed through epoch 0 (before
        # every head's commit), every question published but `unpublished`
        # (the reserve); `fresh_epochs` the reserve's ration when not the
        # design's 1,200 (memo_v4.FRESH_RESERVE_EPOCHS).
        self.unpublished, self.fresh_epochs = tuple(unpublished), fresh_epochs
        self.monkeypatch = patch
        self.down: set[str] = set()
        fx.allow_small_menus(patch)
        patch.setattr(config, "EPOCH_INTERVAL", INTERVAL)
        patch.setattr(config, "V2_ACTIVATION_BLOCK", V2_EPOCH * BPE)
        patch.setattr(config, "V3_ACTIVATION_BLOCK_ENV", V3_EPOCH * BPE)
        patch.setattr(config, "V4_ACTIVATION_BLOCK_ENV", SEAT_V4_EPOCH * BPE)
        patch.setattr(config, "ODDS_SEAT_BLOCK_ENV", None)
        patch.setattr(config, "MENU_B_BLOCK_ENV", None)
        patch.setattr(config, "PUBLISH_HEADS_AFTER_EPOCHS", 0)
        self.pool = fixture_pool()
        rng = np.random.RandomState(0)
        H = (rng.uniform(-1, 1, size=(len(self.pool), SEAT_DIM)) * 0.05).astype(np.float32)
        self.hidden_of = {q["question_id"]: H[i] for i, q in enumerate(self.pool)}
        pool_path = tmp_path / "pool.json"
        pool_path.write_text(json.dumps(self.pool))
        emb = tmp_path / "embeddings.npz"
        np.savez(emb, question_ids=np.array([q["question_id"] for q in self.pool]), H=H)
        for var in ("FUGAL_BENCHMARK_POOL", "FUGAL_BENCHMARK_POOL_V2", "FUGAL_BENCHMARK_POOL_V3"):
            patch.setenv(var, str(pool_path))
        patch.setenv("FUGAL_POOL_UNPINNED", "1")
        patch.setenv("FUGAL_EMBEDDINGS_FILE", str(emb))
        self.base = fx.state(epoch=SEAT_FIRST - 1)
        self.coord = {e: i for i, e in enumerate(sorted([*self.base.ids(), NEW_ID]))}
        self.miners_spec = miners
        self.proposal_from = proposal_from

    def head(self, rows, tag, noise=0.0):
        return seat_head_bytes(rows, self.coord, tag, noise)

    def all_rows(self, sharp, *, new=False):
        """A row of the given sharpness for every entry of the base menu (and
        the new entry when `new`)."""
        rows = {e: sharp for e in self.base.ids()}
        if new:
            rows[NEW_ID] = sharp
        return rows

    def set_head(self, hk, head):
        m = self.world.miners[hk]
        m.serve(head)
        self.chain.commit(hk, m.fields())

    def scored(self, hk, mode):
        """Turn a miner's challenge on ("on": CHALLENGE questions an epoch)
        or off, as its operator would by changing --v4-challenge."""
        self.world.miners[hk].session.challenge.size = challenge_size(mode)

    def run(self):
        import sys
        import time as _real_time
        import urllib.request

        import bittensor

        from fugal_subnet import collect, commit_reveal, config, epoch_logger
        from fugal_subnet import menu_publish as mp
        from fugal_subnet import menu_rules as mr
        from fugal_subnet import v2_epoch as v2e
        from fugal_subnet.tee import harness
        from fugal_subnet.tee import runtime as tee_runtime
        from neurons import miner as miner_mod
        from neurons import validator
        from tests.test_menu_publish import LIST_BYTES

        mpatch = self.monkeypatch
        neurons = [OWNER, ROOT, FOLLOWER, *[(hk, ck) for hk, ck, *_ in self.miners_spec]]
        self.uid_of = {hk: i for i, (hk, _ck) in enumerate(neurons)}
        store = self.store = Store()
        chain = self.chain = Chain(store, neurons=neurons)
        chain.keep, chain.keep_threads = self.keep, self.keep_threads
        mpatch.setattr(urllib.request, "urlopen", store.urlopen)
        mpatch.setattr(bittensor, "Subtensor", lambda *a, **k: Subtensor(chain))
        mpatch.setattr(Wallet, "hotkeys", WALLETS)
        mpatch.setattr(bittensor, "Wallet", Wallet)
        mpatch.setattr(collect, "CollectDendrite", Dendrite)
        uploader_cls, menu_store_cls = store_classes(store)
        mpatch.setattr(reveal_publisher, "GcsUploader", uploader_cls)
        mpatch.setattr(mp, "GcsStore", menu_store_cls)
        provider = SeatProvider(self.hidden_of, self.coord, cheater=self.cheater)
        mpatch.setattr(harness, "_call_model", provider.call)
        from fugal_subnet import join_step
        mpatch.setattr(join_step, "fetch_public_listing", lambda model, **_k: provider.fetch(model))
        mpatch.setattr(tee_runtime.MeteringProxy, "start", lambda self: None)
        mpatch.setattr(tee_runtime.MeteringProxy, "stop", lambda self: None)
        mpatch.setattr(config, "REFERENCE_BUCKET", BUCKET)
        mpatch.setattr(config, "ROOT_HOTKEY_ENV", ROOT[0])
        sha = hashlib.sha256(LIST_BYTES).hexdigest()
        store.objects[reveal_sync.list_object(NETUID, mr.KIND_MENU_ALLOWLIST, sha)] = LIST_BYTES
        store.objects[reveal_sync.signature_object(NETUID, mr.KIND_MENU_ALLOWLIST, sha)] = (
            self._signature)

        chain.block = 1
        hidden = miner_mod._load_pinned_embeddings(self.pool, mock=True)
        world = self.world = types.SimpleNamespace(miners={})
        # The published set: the seed, held by both validators and every TD
        # as the epochs before SEAT_FIRST left it (nobody ran them).
        self.seed = fixture_seed(self.pool, 0, self.unpublished)
        install_seed(self.tmp_path, mpatch.setattr, self.seed, fresh_epochs=self.fresh_epochs)
        td_dirs = {hk: self.tmp_path / "td" / hk for hk, *_ in self.miners_spec}
        for hk, ck, head, mode, block in self.miners_spec:
            chain.block = block
            session = miner_mod._v4_session(
                Subtensor(chain), netuid=NETUID, pool=self.pool, first_epoch=SEAT_V4_EPOCH,
                blocks_per_epoch=BPE, coldkey=ck, genesis=GENESIS,
                fetch_listing=provider.fetch, anchor=self.base,
                challenge_size=challenge_size(mode), state_dir=td_dirs[hk])
            m = world.miners[hk] = Miner(hk, ck, head, session)
            # The proposer files its proposal with its head, one transaction.
            chain.commit(hk, m.propose(PROPOSAL) if hk == self.proposal_from else m.fields())
        Dendrite.world = world
        self.served: dict[int, set[str]] = {}

        def run_miners(epoch):
            if epoch > self.last:
                return
            self.plan(epoch, self)
            self.served[epoch] = set()
            for m in world.miners.values():
                if m.hotkey in self.down:
                    continue
                provider.serving = m.hotkey
                miner_mod._run_epoch(
                    blocks_per_epoch=BPE, head_data=m.head, weights_hash=m.weights_hash,
                    pool=self.pool, hidden_states=hidden, epoch_index=epoch,
                    block_hash=chain.block_hash(epoch * BPE),
                    tee_runtime=tee_runtime.TEERuntime(mock=True), current_proof=m.current,
                    slice_size=config.SLICE_SIZE_V4, proxy_port=0, explore_models=[],
                    explore_size=0, hotkey=m.hotkey, protocol_version=4, prices=None,
                    block_now=lambda: chain.block, v4_session=m.session)
                self.served[epoch].add(m.hotkey)

        dirs = self.dirs = {name: self.tmp_path / name for name in ("root", "follower")}
        for d in dirs.values():
            (d / "epochs").mkdir(parents=True, exist_ok=True)
            mp.MenuBook(str(d / "state.json") + ".menu").hold(self.base)
        contexts = {
            name: [(validator, "STATE_PATH", str(d / "state.json")),
                   (commit_reveal, "EPOCH_DIR", str(d / "epochs")),
                   (epoch_logger, "EPOCH_LOG_DIR", str(d / "logs")),
                   (config, "ROOT_VALIDATOR", True if name == "root" else None),
                   (config, "REVEAL_BUCKET", BUCKET if name == "root" else "")]
            for name, d in dirs.items()}

        def start(name):
            return lambda: validator.main.callback(
                network="local", netuid=NETUID, coldkey=name, hotkey="default",
                wallet_path=None, once=False, log_level="WARNING", live=False)

        stop_at, later = {}, {}
        if self.follower_down is not None:
            down_first, down_last = self.follower_down
            stop_at["follower"] = down_first * BPE - 1
            contexts["follower+"] = contexts["follower"]
            later["follower+"] = ((down_last + 1) * BPE, start("follower"))
        clock = Lockstep(chain, contexts, run_miners, stop_block=(self.last + 1) * BPE + 1,
                         stop_at=stop_at, later=later)
        mpatch.setattr(validator, "time", types.SimpleNamespace(
            sleep=clock.sleep, time=_real_time.time, monotonic=_real_time.monotonic))
        errors: list = []
        real_exception = validator.logger.exception

        def record(msg, *a, **k):
            errors.append((threading.current_thread().name, str(sys.exc_info()[1])))
            real_exception(msg, *a, **k)

        mpatch.setattr(validator.logger, "exception", record)
        # The follower is handed everything in the reverse order of the root.
        real_run = v2e.run_v2_epoch

        def reversed_inputs(**kw):
            if threading.current_thread().name not in ("follower", "follower+"):
                return real_run(**kw)

            def rev(d):
                return dict(reversed(list(d.items())))

            for key in ("verified_proofs", "head_hashes", "coldkey_of", "hotkey_of", "records"):
                kw[key] = rev(kw[key])
            kw["commitments"] = list(reversed(kw["commitments"]))
            oi = kw.get("odds_input")
            if oi is not None:
                kw["odds_input"] = dataclasses.replace(oi, answers=list(reversed(oi.answers)),
                                                       heads=rev(oi.heads))
            return real_run(**kw)

        mpatch.setattr(v2e, "run_v2_epoch", reversed_inputs)
        # Every epoch's memo, live and replayed (check_replayed_memos).
        self.v2_calls = spy_v2_calls(mpatch.setattr)
        chain.block = SEAT_FIRST * BPE
        run_miners(SEAT_FIRST)

        # Lockstep.join raises a miner epoch's exception at once (the
        # validators' loops would log it and go on).
        clock.start({n: start(n) for n in dirs})
        clock.join(timeout=3600)
        assert not any(t.is_alive() for t in clock.threads), "a validator loop did not stop"
        assert clock.errors == {}, clock.errors
        assert errors == [], errors
        assert chain.future_reads == [], chain.future_reads[:5]
        self.epochs = range(SEAT_FIRST, self.last + 1)
        weights: dict = {}
        for block, hk, uids, ws in chain.weights:
            weights.setdefault(block // BPE, {})[hk] = dict(zip(uids, ws))
        self.weights = weights
        return self

    def reveal(self, k, name="root"):
        from fugal_subnet.benchmarks.slicer import epoch_id_for_block

        return json.loads((self.dirs[name] / "epochs" / epoch_id_for_block(k)
                           / "reveal.json").read_text())

    def hash_of(self, hk):
        return self.world.miners[hk].weights_hash

    def holder(self, k):
        return self.reveal(k)["champion"]["champion"]

    def paid(self, k) -> dict[int, float]:
        return {u: w for u, w in self.weights[k][ROOT[0]].items() if w > 1e-12}

    def menu_bytes(self, menu_hash: str) -> bytes | None:
        """The bytes the root's store holds for a menu hash (its menu object)."""
        return self.store.objects.get(reveal_sync.menu_object(NETUID, menu_hash))

    def check_agreement_and_rebuild(self) -> str:
        """Both validators' consensus state, consensus view (the `menu` field
        with its source included) and weights are the same every epoch, and
        the follower's menu audit finds no difference; a rebuild from the
        root's reveals reproduces its seat state and window, and its held
        menus and their sources (`sources.json`) byte for byte; a miner reads
        the seat holder (`champion.champion`) and its start menu (`menu`,
        through `menu.source` when the store has no bytes for the menu's own
        hash) from the same root reveal."""
        from pathlib import Path

        from fugal_subnet import menu_publish as mp
        from fugal_subnet import menu_rules as mr
        from fugal_subnet import menu_state as ms
        from fugal_subnet import seat_v4, v2_sync
        from fugal_subnet.benchmarks.slicer import epoch_id_for_block
        from fugal_subnet.tee import td_v4
        from fugal_subnet.v2_epoch import load_window, seat_from_dict
        from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals
        from neurons import validator

        root_book = mp.MenuBook(str(self.dirs["root"] / "state.json") + ".menu")
        audited = carried = 0
        public = {j: self.reveal(j) for j in self.epochs}
        for k in self.epochs:
            root, follower = self.reveal(k), self.reveal(k, "follower")
            assert root["v2_state"] == follower["v2_state"], k
            assert v2_sync.consensus_view(root) == v2_sync.consensus_view(follower), k
            assert root[seat_v4.REVEAL_KEY] == follower[seat_v4.REVEAL_KEY], k
            assert self.weights[k][ROOT[0]] == self.weights[k][FOLLOWER[0]], k
            audit = follower[mp.REVEAL_AUDIT]
            assert audit["differences"] in ([], None), (k, audit)
            audited += audit["differences"] == []
            raw = (self.dirs["root"] / "epochs" / epoch_id_for_block(k) / "reveal.json").read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            holder = td_v4.seat_holder_from_reveal(raw, epoch_id_for_block(k), sha)
            assert holder == self.holder(k) == seat_v4.from_dict(root["v2_state"]["seat"]).champion
            # The menu field of the same bytes: its hash is the menu the root
            # held for k, and its source is the root's record of it.
            got = mp.anchor_from_reveal(raw, epoch_id_for_block(k), sha)
            held = root_book.held(k)
            assert got is not None and got[:2] == (k, ms.menu_hash(held)), k
            assert tuple(got[2:]) == root_book.source_of(k), k
            assert mp.menu_from_field(root["menu"], epoch_id_for_block(k), self.menu_bytes) == held
            carried += got[2] < k
            # A TD with nothing held anchors on that reveal: from the menu's
            # own bytes, and from its source's bytes alone.
            for offer in (self.menu_bytes,
                          lambda h, src=got[3]: self.menu_bytes(h) if h == src else None):
                session = td_v4.V4Session(
                    first_epoch=SEAT_V4_EPOCH, fetch_listing=lambda m: b"",
                    read_commitment=lambda e: None, menu_bytes=offer,
                    list_bytes=lambda serial, sha: (
                        self.store.objects[reveal_sync.list_object(NETUID, mr.KIND_MENU_ALLOWLIST,
                                                                   sha)],
                        self.store.objects[reveal_sync.signature_object(
                            NETUID, mr.KIND_MENU_ALLOWLIST, sha)]),
                    pool=self.pool, read_anchor=lambda e, a=got: a)
                assert session._reveal_anchor(k + 1) == held, k
            assert root["v2_state"][seat_v4.DIGEST_KEY] == seat_v4.digest(root[seat_v4.REVEAL_KEY])
            assert "burn_in" not in root["v2_state"] and "odds_shadow" not in root
            # The v1-v3 seat's fields are written nowhere in the seat's part
            # of a v4 reveal (SEAT_MINIMAL_V4.md 12).
            seat_part = {x: root[x] for x in ("v2_state", "champion", seat_v4.REVEAL_KEY)}
            assert not SEAT_REMOVED_FIELDS & set(_keys(seat_part)), k
            assert "frame_excluded_coldkeys" not in root, k
            # The memorisation check runs every epoch, on the published set
            # each node holds: the same bytes on both, naming the view the
            # root's public reveals fold to.
            assert root["memo_v4"] == follower["memo_v4"], k
            expect = chain_view(self.chain, self.seed, self.pool, public, k)
            assert root["memo_v4"]["published_digest"] == expect.view(k - 2).digest, k
        for name in ("root", "follower"):
            check_held(self.dirs[name] / "state.json.published_v4.json", self.chain, self.seed,
                       self.pool, public, self.last)
        # Root and follower keep the same held menus and sources, byte for
        # byte. A node with only the base menu that holds each root reveal's
        # menu in epoch order keeps the root's (through the last reveal; the
        # loops also hold the menu of the epoch after it).
        again = self.tmp_path / "rebuilt.menu"
        book = mp.MenuBook(str(again))
        book.hold(self.base)
        for k in self.epochs:
            book.hold(mp.menu_from_field(self.reveal(k)["menu"], epoch_id_for_block(k),
                                         self.menu_bytes))
        follower_book = Path(str(self.dirs["follower"] / "state.json") + ".menu")
        for name in ("sources.json", "held.json"):
            ours = (Path(root_book.directory) / name).read_bytes()
            assert ours == (follower_book / name).read_bytes(), name
            through = {e: v for e, v in json.loads(ours).items() if int(e) <= self.last}
            assert json.loads((again / name).read_bytes()) == through, name
        held = root_book.held(self.last)
        t = validator._v4_tables(mp.MenuRead(self.last, held, "carried"))
        rebuilt = rebuild_v2_from_reveals(str(self.dirs["root"] / "epochs"), window_size=48,
                                          prices=t.prices, bench_category=t.bench_category)
        state = json.loads((self.dirs["root"] / "state.json").read_text())
        assert rebuilt.seat == seat_from_dict(state["v2"]["seat"]) == seat_v4.from_dict(
            self.reveal(self.last)["v2_state"]["seat"])
        saved = load_window(str(self.dirs["root"] / "state.json") + ".v2window.npz", 48)
        assert rebuilt.window.picks == saved.picks
        assert sorted(saved.odds) == sorted(rebuilt.window.odds)
        for h, per in saved.odds.items():
            assert sorted(per) == sorted(rebuilt.window.odds[h])
            assert all(np.array_equal(o, rebuilt.window.odds[h][e]) for e, o in per.items())
        return (f"root and follower agree e{SEAT_FIRST}..e{self.last} (follower's audit empty, "
                f"{audited} epochs compared, {carried} carried menus); a rebuild from the root's "
                "reveals gives its seat, window, held menus and sources")


# The seat scenarios (SEAT_MINIMAL_V4.md 11): each builds and runs its loops
# and returns them; the checks below each assert one property of a run and
# return a line saying what they saw. The tests and the rehearsal call both.

SEAT_A, SEAT_B, SEAT_C, SEAT_D, SEAT_E, SEAT_F, SEAT_L = (f"5Fseat-{x}" for x in "abcdefl")
TAKE_LAST = SEAT_FIRST + 14
TAKE_SWITCH, TAKE_B_MISSES = SEAT_FIRST + 5, SEAT_FIRST + 9
# The holder stops challenging once seated (it stands on exploration alone);
# B's attempt in TAKE_TIMEOUT runs out of its time budget, and in TAKE_OUTAGE
# the root's reveal its TD reads the holder from is not to be had.
HOLDER_QUIET = SEAT_FIRST + 4
TAKE_TIMEOUT, TAKE_OUTAGE = SEAT_FIRST + 3, SEAT_FIRST + 4
HOLD_LAST = SEAT_FIRST + 19                # the epoch the latest qualifier takes the open seat
HOLD_SUCCEED = SEAT_FIRST + 8
HOLD_TWO = (SEAT_FIRST + 13, SEAT_FIRST + 14)
HOLD_THREE = (SEAT_FIRST + 17, SEAT_FIRST + 18, SEAT_FIRST + 19)
assert HOLD_LAST == max(HOLD_THREE)


def seat_take_scenario(tmp_path, patch, signature) -> SeatLoop:
    """The fill, an odds-only miner's take in one challenge, and copies: A
    challenges from the first epoch and fills the empty seat, then stops
    challenging (HOLDER_QUIET) and holds on its exploration alone; B, sharper
    and odds only, turns a 1,000-question challenge on: its attempt in
    TAKE_TIMEOUT does not finish inside the TD's time budget, in TAKE_OUTAGE
    the holder cannot be read from the root's reveal, and its one challenge
    in TAKE_SWITCH takes the seat that epoch, after which it challenges no
    more; it misses TAKE_B_MISSES; C is A's exact bytes under another
    coldkey; D is B lightly altered under another coldkey, committed after
    B, challenging from TAKE_SWITCH; L is a labeller with one 50% row."""
    spec: list = []
    loop = SeatLoop(tmp_path, patch, spec, last=TAKE_LAST, plan=lambda e, lp: None,
                    signature=signature)
    a_head = loop.head(loop.all_rows(SHARP), "a")
    spec += [
        (SEAT_A, "5Fcold-a", a_head, "on", 1),
        (SEAT_B, "5Fcold-b", loop.head(loop.all_rows(SHARPER), "b"), "off",
         SEAT_V4_EPOCH * BPE + 3),
        (SEAT_C, "5Fcold-c", a_head, "on", SEAT_V4_EPOCH * BPE + 5),
        (SEAT_D, "5Fcold-d", loop.head(loop.all_rows(SHARPER), "d", noise=1e-4), "off",
         SEAT_V4_EPOCH * BPE + 7),
        (SEAT_L, "5Fcold-l", loop.head({loop.base.ids()[0]: 0.0}, "l"), "off",
         SEAT_V4_EPOCH * BPE + 9),
    ]

    from fugal_subnet.tee import harness

    real_clock, readers = harness._clock, {}

    def plan(e, lp):
        harness._clock = real_clock
        for hk, read in readers.items():
            lp.world.miners[hk].session.challenge.read_holder = read
        if e == HOLDER_QUIET:
            lp.scored(SEAT_A, "off")
            lp.scored(SEAT_C, "off")
        if e == TAKE_TIMEOUT:
            lp.scored(SEAT_B, "1000")
            ticks = iter(range(0, 10**12, 10**6))
            harness._clock = lambda: float(next(ticks))   # each read a million seconds on
        if e == TAKE_OUTAGE:
            pol = lp.world.miners[SEAT_B].session.challenge
            readers[SEAT_B] = pol.read_holder

            def unreadable(epoch):
                raise LookupError("the root's reveal is not to be had")
            pol.read_holder = unreadable
        if e == TAKE_SWITCH:
            lp.scored(SEAT_D, "on")
        if e == TAKE_SWITCH + 1:
            lp.scored(SEAT_B, "off")
        lp.down = {SEAT_B} if e == TAKE_B_MISSES else set()

    loop.plan = plan
    try:
        return loop.run()
    finally:
        harness._clock = real_clock


def check_seat_fill(loop: SeatLoop) -> str:
    """The seat is empty at activation (the share burns) until A has 300
    labels over R and a challenge of 500 questions, then A fills it; it holds
    and is paid with no challenge of its own from HOLDER_QUIET."""
    holders = {k: loop.holder(k) for k in loop.epochs}
    events = {k: loop.reveal(k)["champion"]["event"] for k in loop.epochs}
    fill = min(k for k in loop.epochs if holders[k] is not None)
    a = loop.hash_of(SEAT_A)
    assert events[fill] == "fill" and holders[fill] == a
    assert SEAT_FIRST < fill < HOLDER_QUIET, fill
    assert all(loop.paid(k) == {0: 1.0} for k in loop.epochs if k < fill)
    # The holder's picks on the whole challenge list are in every reveal with
    # a holder (none before the fill), whoever challenged.
    from fugal_subnet import picks_v4
    for k in loop.epochs:
        r = loop.reveal(k)
        doc = r["picks_v4"]["holder_picks"]
        if k < fill:
            assert doc == {}, k
            continue
        head, picks = picks_v4.picks_from_doc(doc, r["models"])
        assert head == holders[k] == r["champion"]["champion"], k
        assert len(picks) == len(r["picks_v4"]["list"]) > 0, k
    for k in range(HOLDER_QUIET, TAKE_SWITCH):
        assert a not in loop.reveal(k)["picks_v4"]["items"] and holders[k] == a, k
        assert loop.paid(k) == {loop.uid_of[SEAT_A]: 1.0}, k
    return (f"empty from the activation, burning, until A fills it at e{fill}; it holds and is "
            f"paid 1.0 with no challenge from e{HOLDER_QUIET}")


def check_seat_odds_only_take(loop: SeatLoop) -> str:
    """B's attempt that ran out of time and the one whose holder could not
    be read are not in its proofs (no challenge, nothing counted); its one
    1,000-question challenge in TAKE_SWITCH takes the seat that epoch, paid
    0.90 and A 0.10. Missing TAKE_B_MISSES it keeps the seat, unpaid, and is
    paid again after it, with no further challenge."""
    a, b = loop.hash_of(SEAT_A), loop.hash_of(SEAT_B)
    holders = {k: loop.holder(k) for k in loop.epochs}
    events = {k: loop.reveal(k)["champion"]["event"] for k in loop.epochs}
    fill = min(k for k in loop.epochs if holders[k] is not None)
    take = min(k for k in loop.epochs if holders[k] == b)
    for k in (TAKE_TIMEOUT, TAKE_OUTAGE):
        assert b not in loop.reveal(k)["picks_v4"]["items"], k
        assert loop.reveal(k)["picks_v4"]["evidence"][b]["answered"] is False, k
    assert events[take] == "take" and take == TAKE_SWITCH, (take, events[take])
    assert loop.reveal(take)["picks_v4"]["evidence"][b]["questions"] == 1000
    assert all(holders[k] == a for k in loop.epochs if fill <= k < take)
    assert loop.paid(take)[loop.uid_of[SEAT_B]] == pytest_approx(0.90)
    assert loop.paid(take)[loop.uid_of[SEAT_A]] == pytest_approx(0.10)
    assert all(holders[k] == b for k in loop.epochs if k >= take)
    assert loop.uid_of[SEAT_B] not in loop.paid(TAKE_B_MISSES)
    assert loop.paid(TAKE_B_MISSES + 1)[loop.uid_of[SEAT_B]] == pytest_approx(0.90)
    return (f"B's attempt timed out at e{TAKE_TIMEOUT} and had no holder to pair with at "
            f"e{TAKE_OUTAGE}; its one 1,000-question challenge took the seat at e{take} "
            "(paid 0.90, A 0.10); it holds without challenging, unpaid only while absent")


def check_seat_copies(loop: SeatLoop) -> str:
    """D never holds the seat: in TAKE_SWITCH its challenge passes rule B
    but it is tested against B, the earlier contender, and does not pass it;
    from then on its odds do not beat B's, the holder's, the epoch B missed
    included. C, A's exact bytes, is a copy every epoch."""
    from fugal_subnet import seat_v4

    d, b = loop.hash_of(SEAT_D), loop.hash_of(SEAT_B)
    assert d not in {loop.holder(k) for k in loop.epochs}
    ev = loop.reveal(TAKE_SWITCH)[seat_v4.REVEAL_KEY]
    x = ev["heads"][d]
    assert x["contender"] and x["picks"]["passes"] and not x["qualifies"]
    assert (b, False) in {(r[0], r[3]) for r in seat_v4.earlier_rows(ev, d)}
    tested = 0
    for k in range(TAKE_SWITCH + 1, TAKE_LAST + 1):
        x = loop.reveal(k)[seat_v4.REVEAL_KEY]["heads"].get(d)
        assert x is None or not x["qualifies"], k
        tested += x is not None
    assert tested >= 3, tested
    for k in loop.epochs:
        r = loop.reveal(k)
        assert r["copy_verdicts"][str(loop.uid_of[SEAT_C])] == "copy"
        assert loop.uid_of[SEAT_C] not in loop.paid(k)
    return f"the near-copy failed rule C against B in {tested} epochs; the exact copy never paid"


def seat_hold_scenario(tmp_path, patch, signature) -> SeatLoop:
    """A new entry, succession and misses: A fills the seat with a
    challenge and holds on its exploration alone from HOLDER_QUIET; the new
    entry A proposed on chain joins (E has its row, A not); A adds the row by
    succession at HOLD_SUCCEED, and so does F (A's rows, its own bytes and
    coldkey: no edge over the holder), which challenges from SEAT_FIRST +
    9; A misses HOLD_TWO, then HOLD_THREE."""
    spec: list = []
    loop = SeatLoop(tmp_path, patch, spec, last=HOLD_LAST, plan=lambda e, lp: None,
                    signature=signature, proposal_from=SEAT_A)
    loop.a2_head = loop.head(loop.all_rows(SHARP, new=True), "a")
    f2_head = loop.head(loop.all_rows(SHARP, new=True), "f")
    spec += [
        (SEAT_A, "5Fcold-a", loop.head(loop.all_rows(SHARP), "a"), "on", 1),
        (SEAT_E, "5Fcold-e", loop.head({NEW_ID: SHARPER, loop.base.ids()[0]: 0.0}, "e"), "off",
         SEAT_V4_EPOCH * BPE + 3),
        (SEAT_F, "5Fcold-f", loop.head(loop.all_rows(SHARP), "f"), "off",
         SEAT_V4_EPOCH * BPE + 5),
        (SEAT_L, "5Fcold-l", loop.head({loop.base.ids()[0]: 0.0}, "l"), "off",
         SEAT_V4_EPOCH * BPE + 9),
    ]
    loop.a1 = hashlib.sha256(spec[0][2]).hexdigest()
    loop.a2 = hashlib.sha256(loop.a2_head).hexdigest()

    def plan(e, lp):
        if e == HOLDER_QUIET:
            lp.scored(SEAT_A, "off")
        if e == HOLD_SUCCEED:
            # A and F both add the new row by succession in the same epoch:
            # the line of each lacked it from the epoch it became full, so
            # neither has an edge over the other there (a missing row and an
            # added row's earlier labels count from the line at odds 0)
            lp.set_head(SEAT_A, lp.a2_head)
            lp.set_head(SEAT_F, f2_head)
        if e == SEAT_FIRST + 9:
            lp.scored(SEAT_F, "on")
        lp.down = {SEAT_A} if e in HOLD_TWO + HOLD_THREE else set()

    loop.plan = plan
    return loop.run()


def check_seat_new_entry(loop: SeatLoop) -> str:
    """The new entry joins provisional at the boundary after the root's
    pinned read (FAST_CROWN_V4.md 18): its labels in its two provisional
    epochs are marked so no head is scored on them; from the epoch the gates
    make it full, a head without its row (A) is scored at odds 0 and E on
    its own row. A fills the seat lacking the row and keeps it."""
    import base64

    from fugal_subnet import labels_v4
    from fugal_subnet import menu_publish as mp
    from fugal_subnet import menu_state as ms

    holders = {k: loop.holder(k) for k in loop.epochs}
    events = {k: loop.reveal(k)["champion"]["event"] for k in loop.epochs}
    fill = min(k for k in loop.epochs if holders[k] is not None)
    held = mp.MenuBook(str(loop.dirs["root"] / "state.json") + ".menu")
    joined = min(k for k in loop.epochs if held.held(k) and NEW_ID in held.held(k).ids())
    full_from = joined + ms.MENU_LAG
    # A fills the seat with its one challenge (the fast crown), before
    # HOLD_SUCCEED, and holds through the entry's provisional epochs and on
    # past the epoch it becomes full, lacking its row.
    assert holders[fill] == loop.a1 and fill < HOLD_SUCCEED and full_from < HOLD_SUCCEED, (
        holders[fill] == loop.a1, joined, full_from, fill, HOLD_SUCCEED, events)
    assert held.held(joined).entry(NEW_ID).provisional
    assert not held.held(full_from).entry(NEW_ID).provisional
    for k in range(joined, HOLD_SUCCEED):
        r = loop.reveal(k)
        new = [i for i, p in enumerate(r["exploration_pool"]) if p["model"] == NEW_ID]
        want = labels_v4.PROVISIONAL_LABEL if k < full_from else -1
        assert new and all(r["labels_v4"]["missing_odds"][i] == want for i in new), k
        e_odds = np.frombuffer(base64.b64decode(r["head_odds"][loop.hash_of(SEAT_E)]), "<i2")
        a_odds = np.frombuffer(base64.b64decode(r["head_odds"][loop.a1]), "<i2")
        assert all(a_odds[i] == -1 for i in new) and any(e_odds[i] >= 0 for i in new)
        if k > fill:
            assert holders[k] == loop.a1 and events[k] == "hold"
            assert loop.paid(k) == {loop.uid_of[SEAT_A]: 1.0}
    return (f"{NEW_ID} joined at e{joined} provisional (its labels score no head), full from "
            f"e{full_from}; a head without its row at odds 0")


def check_seat_succession(loop: SeatLoop) -> str:
    """The holder adds the row by succession: the seat, its precedence and
    its evidence carry over to the successor, which the reveal names."""
    from fugal_subnet import seat_v4

    seat = seat_v4.from_dict(loop.reveal(HOLD_SUCCEED)["v2_state"]["seat"])
    hs = seat.head(loop.a2)
    assert loop.holder(HOLD_SUCCEED) == loop.a2
    assert loop.reveal(HOLD_SUCCEED)["champion"]["event"] == "hold"
    assert hs.predecessors == (loop.a1,) and hs.origin == loop.a1
    assert set(hs.rows_since) == set(loop.base.ids())
    assert seat.head(loop.a1) is None and loop.paid(HOLD_SUCCEED) == {loop.uid_of[SEAT_A]: 1.0}
    assert all(loop.holder(k) == loop.a2 for k in loop.epochs
               if HOLD_SUCCEED <= k < max(HOLD_THREE))
    return (f"A's successor holds from e{HOLD_SUCCEED}, predecessor and origin recorded, "
            "paid 1.0")


def check_seat_holder_head_from_root(loop: SeatLoop) -> str:
    """A follower that starts with nothing (no verified head this epoch, an
    empty replay store) while the holder serves nothing obtains the holder's
    head from the sources in their fixed order (validator.holder_head_sources):
    here from the root's head publication, which published the holder and its
    successor with the reveals that named them. Never a guess: bytes with
    another hash are passed over."""
    import urllib.request
    from unittest import mock

    from neurons import validator

    def empty_store(_h):
        raise FileNotFoundError("a follower that starts with nothing")

    got = {}
    for holder in (loop.a1, loop.a2):
        key = f"heads/{NETUID}/{holder}.npz"
        assert key in loop.store.objects, f"the root did not publish {holder[:12]}"
        with mock.patch.object(urllib.request, "urlopen", loop.store.urlopen):
            data, source = validator.holder_head_sources(
                holder, held={}, replay_store=empty_store, served=lambda: [b"another head"],
                publication=lambda h: reveal_sync.fetch_head(BUCKET, NETUID, h))
        assert source == "root publication" and hashlib.sha256(data).hexdigest() == holder
        got[holder[:12]] = len(data)
    return ("a follower with no head obtains the holder's and its successor's from the root's "
            f"head publication ({', '.join(f'{h} {n} bytes' for h, n in got.items())})")


def check_seat_standing(loop: SeatLoop) -> str:
    """The holder answers no challenge from HOLDER_QUIET and stands and is
    paid on its exploration; F, with no edge over it, challenges every epoch
    and does not take while it stands. Two missed epochs: the seat stays,
    unpaid (it burns), then is paid again. Three: the bar is R, the seat
    open, and it goes at once to the latest qualifier, F, paid 1.0."""
    from fugal_subnet import seat_v4

    events = {k: loop.reveal(k)["champion"]["event"] for k in loop.epochs}
    f = loop.hash_of(SEAT_F)
    for k in range(HOLDER_QUIET, max(HOLD_THREE)):
        items = loop.reveal(k)["picks_v4"]["items"]
        assert loop.a1 not in items and loop.a2 not in items, k
        assert loop.holder(k) != f, k
        if k not in HOLD_TWO + HOLD_THREE:
            assert loop.paid(k) == {loop.uid_of[SEAT_A]: 1.0}, k
    assert f in loop.reveal(SEAT_FIRST + 10)["picks_v4"]["items"]
    for k in HOLD_TWO:
        assert loop.holder(k) == loop.a2 and events[k] == "hold" and loop.paid(k) == {0: 1.0}, k
    assert loop.paid(max(HOLD_TWO) + 1) == {loop.uid_of[SEAT_A]: 1.0}
    for k in HOLD_THREE[:-1]:
        assert loop.holder(k) == loop.a2 and events[k] == "hold", k
    k = max(HOLD_THREE)
    ev = loop.reveal(k)[seat_v4.REVEAL_KEY]
    assert ev["open"] and events[k] == "take" and loop.holder(k) == f and ev["winner"] == f
    assert ev["qualifiers"][-1] == f
    assert loop.paid(k) == {loop.uid_of[SEAT_F]: 1.0}
    return (f"A missed e{HOLD_TWO[0]}..e{HOLD_TWO[-1]} and kept the seat unpaid; missed "
            f"e{HOLD_THREE[0]}..e{HOLD_THREE[-1]} and F took it at e{k}, paid 1.0")


MEMO_LAST = SEAT_FIRST + 8
MEMO_LABELLERS = [f"5Fseat-l{i}" for i in range(4)]
# The reserve's ration in the memorisation scenario: the design's 1,200
# epochs would give this fixture pool one fresh question an epoch, and the
# check needs 500 unseen labels; at this ration the reserve (half the pool)
# gives about fifty an epoch, every hotkey answering each.
MEMO_FRESH_EPOCHS = 48


def seat_memo_scenario(tmp_path, patch, signature, *, follower_down=None,
                       last=MEMO_LAST, keep=None, keep_threads=None) -> SeatLoop:
    """The memorisation check through the shipped loops, nothing handed to
    them: the release's published-set seed (observed before A's commit) is
    half the pool, the half where the questions' embeddings are furthest from
    zero on the menu's coordinates, so a sharp head is confident and right on
    published questions and nearly guessing on the others; the other half is
    the reserve, which the epochs' fresh draws spend (MEMO_FRESH_EPOCHS) and
    which gives the unseen labels. A answers the scored questions and fills
    the seat; four labellers with no skill are never refused. The follower
    is down for the epochs `follower_down` (first, last) when given."""
    spec: list = []
    loop = SeatLoop(tmp_path, patch, spec, last=last, plan=lambda e, lp: None,
                    signature=signature, fresh_epochs=MEMO_FRESH_EPOCHS,
                    follower_down=follower_down, keep=keep, keep_threads=keep_threads)
    spec += [(SEAT_A, "5Fcold-a", loop.head(loop.all_rows(SHARP), "a"), "on", 1)]
    spec += [(lb, f"5Fcold-l{i}", loop.head({loop.base.ids()[0]: 0.0}, f"l{i}"), "off",
              SEAT_V4_EPOCH * BPE + 9 + i) for i, lb in enumerate(MEMO_LABELLERS)]
    coords = [loop.coord[e] for e in loop.base.ids()]
    score = {q["question_id"]: float(np.abs(loop.hidden_of[q["question_id"]][coords]).mean())
             for q in loop.pool}
    cut = float(np.median(list(score.values())))
    loop.unpublished = tuple(sorted(q for q, v in score.items() if v <= cut))
    return loop.run()


def check_seat_memo_refusal(loop: SeatLoop) -> str:
    """A fills the seat and is then refused by the memorisation check on
    both nodes alike, once its unseen labels (the fresh questions) are
    enough: from that epoch the bar is R, the seat open and its share burns.
    Every epoch's published evidence names every head the seat keeps and the
    seat's own flags equal it; the labellers are never refused."""
    from fugal_subnet import seat_v4
    from fugal_subnet.v2_epoch import _digest_of

    a, labs = loop.hash_of(SEAT_A), {loop.hash_of(lb) for lb in MEMO_LABELLERS}
    fill = min(k for k in loop.epochs if loop.holder(k) == a)
    refused = [k for k in loop.epochs if loop.reveal(k)["memo_v4"]["evidence"].get(
        a, {}).get("refused")]
    assert refused, "A was never refused"
    first = min(refused)
    assert refused == list(range(first, loop.last + 1)), refused
    assert fill < first and loop.paid(fill) == {loop.uid_of[SEAT_A]: 1.0}
    ev_a = loop.reveal(first)["memo_v4"]["evidence"][a]
    assert ev_a["seen"] >= 500 and ev_a["unseen"] >= 500, ev_a
    for k in loop.epochs:
        r = loop.reveal(k)
        doc, ev = r["memo_v4"], r[seat_v4.REVEAL_KEY]
        assert r["v2_state"]["memo_digest"] == _digest_of(doc)
        assert doc["fresh_count"] > 0 and doc["fresh"], k
        kept = {hs.head for hs in seat_v4.from_dict(r["v2_state"]["seat"]).heads}
        assert set(doc["evidence"]) == kept and a in kept, k
        assert not any(doc["evidence"][h]["refused"] for h in labs & kept), k
        assert all(d["refused"] == doc["evidence"][h]["refused"] for h, d in ev["heads"].items())
        if k < first:
            continue
        assert ev["holder"]["head"] == a and ev["holder"]["refused"] is True, k
        assert ev["open"] and ev["bar"] is None and ev["paid"] == {a: False}, k
        assert r["champion"]["event"] == "open" and loop.holder(k) == a, k
        assert loop.paid(k) == {0: 1.0}, k
    return (f"A filled at e{fill}; refused from e{first} ({ev_a['seen']} seen, "
            f"{ev_a['unseen']} unseen labels): the seat open, unpaid, on both nodes alike")




SEAT_X = "5Fseat-x"
RESYNC_LAST = SEAT_FIRST + 12


def seat_resync_scenario(tmp_path, patch, signature, *, down, min_trials=200,
                         last=RESYNC_LAST) -> SeatLoop:
    """A follower that is down for the epochs `down` (first, last) while the
    seat fills: A answers from the first epoch and fills the empty seat; L
    labels with one 50% row; X labels with a TD whose every call is right,
    which the plausibility check against R leaves out of the label pool once
    the window holds `min_trials` pairs of an entry (patched up from the
    shipped 50, so that R needs several epochs of pairs and a window missing
    an epoch's pairs judges X later than the root's does). A's challenge is
    on from the third epoch after the follower is back, so the seat fills
    then (FAST_CROWN_V4.md: one challenge fills it)."""
    from fugal_subnet import config

    patch.setattr(config, "EXPLORE_PRIVATE_MIN_TRIALS", int(min_trials))
    spec: list = []
    loop = SeatLoop(tmp_path, patch, spec, last=last, plan=lambda e, lp: None,
                    signature=signature, cheater=SEAT_X, follower_down=down)
    spec += [
        (SEAT_A, "5Fcold-a", loop.head(loop.all_rows(SHARP), "a"), "off", 1),
        (SEAT_L, "5Fcold-l", loop.head({loop.base.ids()[0]: 0.0}, "l"), "off",
         SEAT_V4_EPOCH * BPE + 9),
        (SEAT_X, "5Fcold-x", loop.head({loop.base.ids()[0]: 0.0}, "x"), "off",
         SEAT_V4_EPOCH * BPE + 11),
    ]

    def plan(e, lp):
        if e == down[1] + 3:
            lp.scored(SEAT_A, "on")

    loop.plan = plan
    return loop.run()


def pytest_approx(x):
    """pytest.approx, imported where a check runs (the rehearsal is a script)."""
    import pytest

    return pytest.approx(x)
