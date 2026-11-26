"""Protocol v5 driven on finney's own numbers through the validator's glue.

v0.8.0 shipped two mistakes that no test saw, because every v5 test used
tiny synthetic blocks (10, 50 * n) on which both are invisible: the head
book was fed the v1 seniority key (registration x 10^10 + commit) as a
head's commit block, and a block was turned into its epoch by dividing by
EPOCH_INTERVAL (seconds, 4320) instead of the 360 blocks of an epoch.

This world uses the real ones: 360-block epochs (4320 s), the v5 block
9,585,720 (epoch 26627), the v0.8.1 fix block 9,587,520 (epoch 26632),
registration and commitment blocks around 9.5M-9.59M, question ids of the
v5 pool's shape (`arena_<Dataset>_<16 hex>`), 200-question exams and the
published set folded from the very reveals `_v5_score` writes (memo_v4,
READ_LAG). Each epoch goes through the validator's own functions:
`_check_response` on the commitment read at the boundary, the loop's
bookkeeping of what it returns, `_v5_commit_blocks` and `_v5_score` with a
real `v5_node.V5Node`. Only the TEE verification (`verify_proof`) and the
bundle fetch are stood in for: they are not what is under test.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

from fugal_subnet import config, memo_v4, seat_v5, v5_epoch, v5_node
from fugal_subnet.benchmarks.slicer import blocks_per_epoch
from fugal_subnet.exam_v5 import build_spec
from fugal_subnet.tee import verify as verify_mod
from neurons import validator

FINNEY_INTERVAL = 4320                              # seconds: tempo 360 x 12 s
BPE = 360                                           # blocks per epoch on finney
V5_BLOCK = 9_585_720                                # config.FINNEY_V5_ACTIVATION_BLOCK
FIX_BLOCK = 9_587_520                               # config.FINNEY_V5_COMMIT_FIX_BLOCK
V5_EPOCH = V5_BLOCK // BPE                          # 26627
FIX_EPOCH = FIX_BLOCK // BPE                        # 26632
FIRST_EXAM = V5_EPOCH + v5_epoch.SPEC_LAG           # 26629: A and A + 1 have no spec
EXAM_N = 200                                        # questions an exam has on finney
COVERAGE_N = int(config.COVERAGE_CALLS_V5)          # 48 coverage calls a hotkey makes

DATASETS = ("MMLUPro", "OpenTDB", "QANTA", "PubMedQA", "LiveCodeBench", "GeoBench")
BASE, GOOD, BEST = "m/base", "m/good", "m/best"
ACC = {BASE: 50, GOOD: 90, BEST: 99}                # percent right
COST = {BASE: 1500, GOOD: 2000, BEST: 2000}         # nanodollars a call
# Six voting operators without a head of their own (one hotkey each).
VOTERS = tuple((10 + i, f"5Voter{i}" + "x" * 41, f"5VoterColdkey{i}" + "y" * 34)
               for i in range(6))


def install(monkeypatch) -> None:
    """finney's epoch length in the v5 modules (the suite runs at the
    3600 s default), and the stand-ins for the TEE check and the bundle fetch."""
    assert blocks_per_epoch(FINNEY_INTERVAL) == BPE
    assert (config.FINNEY_V5_ACTIVATION_BLOCK, config.FINNEY_V5_COMMIT_FIX_BLOCK) == (
        V5_BLOCK, FIX_BLOCK)
    monkeypatch.setattr(v5_epoch, "EPOCH_INTERVAL", FINNEY_INTERVAL)
    monkeypatch.setattr(v5_epoch, "BLOCKS_PER_EPOCH", BPE)
    monkeypatch.setattr(verify_mod, "verify_proof", lambda proof, **kw: SimpleNamespace(
        valid=True, unverifiable=False, reason="", warnings=[]))
    # The next exam is built from the reserve: not what these tests are about.
    monkeypatch.setattr(v5_epoch, "next_spec", lambda **kw: (None, {}))


def qid(kind: str, *parts) -> str:
    """A v5 pool id: arena_<RouterArena dataset>_<16 hex>."""
    h = hashlib.sha256(":".join([kind, *map(str, parts)]).encode()).hexdigest()
    return f"arena_{DATASETS[int(h[:8], 16) % len(DATASETS)]}_{h[:16]}"


def right(q: str, entry: str) -> bool:
    h = int.from_bytes(hashlib.sha256(f"{q}|{entry}".encode()).digest()[:2], "big") % 100
    return h < ACC[entry]


class Miner:
    """A hotkey serving one head: registered at `registered`, the head
    committed at `commit_block` (both chain blocks)."""

    def __init__(self, name: str, uid: int, registered: int, commit_block: int, entry: str):
        self.name, self.uid, self.entry = name, uid, entry
        self.hotkey = f"5{name.capitalize()}Hotkey" + "h" * (47 - len(name) - 7)
        self.coldkey = f"5{name.capitalize()}Coldkey" + "c" * (47 - len(name) - 8)
        self.registered, self.commit_block = int(registered), int(commit_block)
        self.head_bytes = f"head of {name}".encode()
        self.head = hashlib.sha256(self.head_bytes).hexdigest()

    def served_at(self, epoch: int) -> bool:
        """Its commitment is on chain at the epoch's boundary block."""
        return self.commit_block <= epoch * BPE


class World:
    """The chain, the miners and one root validator's v5 node across epochs."""

    def __init__(self, miners, *, fix_block: int | None = FIX_BLOCK, exam_n: int = EXAM_N,
                 stray: dict | None = None):
        self.miners = list(miners)
        self.fix_block, self.exam_n = fix_block, int(exam_n)
        # {epoch: question ids some other path published in that epoch}
        self.stray = {int(e): set(qs) for e, qs in (stray or {}).items()}
        self.node = v5_node.V5Node(None, root=True)
        self.menu = SimpleNamespace(entries=[
            SimpleNamespace(id=BASE, provisional=False),        # the one FULL entry: B*
            SimpleNamespace(id=GOOD, provisional=True),
            SimpleNamespace(id=BEST, provisional=True)])
        self.gold: dict[str, dict] = {}
        self.inputs = validator._V5Inputs(pool=[], gold=self.gold, embeddings=None,
                                          exam_by_dataset={}, weights_ppm={})
        self.published = memo_v4.Published({}, V5_EPOCH - 3)
        self.folded: dict[int, set] = {}
        self.specs: dict[int, object] = {}
        self.served: dict[int, list] = {}         # epoch -> the Served _v5_score built
        self.results: dict[int, object] = {}      # epoch -> _V5Scored
        self.raw: dict[int, bool] = {}

    # --- questions ----------------------------------------------------------------------

    def _gold(self, q: str) -> str:
        self.gold.setdefault(q, {"question_id": q,
                                 "metadata": {"ra_dataset": q.split("_")[1]}})
        return q

    def exam_questions(self, epoch: int) -> list[str]:
        return [self._gold(qid("exam", epoch, i)) for i in range(self.exam_n)]

    def spec(self, epoch: int):
        """Epoch E's spec, as the root built it at E - SPEC_LAG: the heads
        verified by then pick on its questions; every exam question carries
        B*'s cell."""
        if epoch < FIRST_EXAM:
            return None
        if epoch not in self.specs:
            built_at = epoch - v5_epoch.SPEC_LAG
            heads = [m for m in self.miners if m.served_at(built_at)]
            qs = self.exam_questions(epoch)
            ring = [(hk, ck) for _uid, hk, ck in VOTERS] + [(m.hotkey, m.coldkey) for m in heads]
            self.specs[epoch] = build_spec(
                epoch, hashlib.sha256(f"spec-nonce:{epoch}".encode()).digest(),
                {m.head: [m.entry] * len(qs) for m in heads}, (BASE, GOOD, BEST), qs, ring, BASE)
        return self.specs[epoch]

    # --- proofs -------------------------------------------------------------------------

    def proof(self, epoch: int, hotkey: str, weights_hash: str = ""):
        spec = self.spec(epoch)
        results = []
        for (q, e), hks in sorted(spec.assignment.items()) if spec is not None else ():
            if hotkey in hks:
                results.append(SimpleNamespace(
                    question_id=q, routed_model=e, exam=True, failed=False,
                    y=10000 if right(q, e) else 0, provider_cost=COST[e], cost_usd=0.0))
        for j in range(COVERAGE_N):
            q = self._gold(qid("coverage", epoch, hotkey, j))
            results.append(SimpleNamespace(
                question_id=q, routed_model=BASE, exam=False, failed=False,
                y=10000 if right(q, BASE) else 0, provider_cost=COST[BASE], cost_usd=0.0))
        return SimpleNamespace(hotkey=hotkey, weights_hash=weights_hash, results=results)

    # --- the published set --------------------------------------------------------------

    def view(self, epoch: int) -> memo_v4.Published:
        """The published set as of epoch E (memo_sync.ChainSet.view): every
        reveal through E - READ_LAG folded at its own epoch, nothing at E - 1."""
        lag = memo_v4.READ_LAG
        while self.published.through < epoch - lag:
            k = self.published.through + 1
            ids = self.folded.get(k, set()) | self.stray.get(k, set())
            self.published = memo_v4.fold(self.published, k, ids)
        return memo_v4.fold(self.published.view(epoch - lag), epoch - 1, ())

    # --- one epoch through the validator --------------------------------------------------

    def epoch(self, epoch: int, monkeypatch):
        boundary = epoch * BPE
        checks = validator._EpochChecks(
            boundary_block=boundary, require_commitment=True, approved_measurements=set(),
            questions_hash="", nonce_hex="00" * 32, slice_gold={}, question_ids=set(),
            explore_map={}, mock=False, collateral_budget=None, protocol_version=5)
        # What the loop keeps per uid (neurons/validator.py, VERIFY PROOFS).
        commit_blocks, raw_commit_blocks = {}, {}
        verified_proofs, verified_heads, head_hashes, coldkey_of = {}, {}, {}, {}
        bundles = {}
        monkeypatch.setattr(validator, "_get_bundle_for_uid", lambda uid, resp: bundles[uid])
        for m in self.miners:
            if not m.served_at(epoch):
                continue
            bundles[m.uid] = (self.proof(epoch, m.hotkey, m.head), m.head_bytes)
            resp = SimpleNamespace(proof_hash="ab" * 32, weights_hash=m.head)
            # The commitment as read pinned at the boundary: (data, its block).
            checked = validator._check_response(m.uid, resp, m.hotkey, (m.head, m.commit_block),
                                                m.registered, checks)
            assert checked.status == "valid", (m.name, epoch)
            if checked.seniority is not None:
                commit_blocks[m.uid] = checked.seniority
            if checked.commit_block is not None:
                raw_commit_blocks[m.uid] = checked.commit_block
            verified_proofs[m.uid] = checked.proof
            verified_heads[m.uid] = checked.head_bytes
            head_hashes[m.uid] = checked.proof.weights_hash
            coldkey_of[m.uid] = m.coldkey
        for uid, hk, ck in VOTERS:               # verified operators without a head here
            verified_proofs[uid] = self.proof(epoch, hk)
            coldkey_of[uid] = ck
        blocks, raw = validator._v5_commit_blocks(boundary, self.fix_block,
                                                  seniority=commit_blocks, raw=raw_commit_blocks)
        self.raw[epoch] = raw
        spec = self.spec(epoch)
        exam = (spec, dict(COST), None) if spec is not None else (None, {}, "no spec")
        real = v5_epoch.score_epoch

        def spy(**kw):
            self.served[epoch] = list(kw["served"])
            return real(**kw)
        monkeypatch.setattr(v5_epoch, "score_epoch", spy)
        scored, uids, weights, commit = validator._v5_score(
            self.node, epoch_index=epoch, nonce=b"\x00" * 32, exam=exam, state=self.menu,
            verified_proofs=verified_proofs, verified_heads=verified_heads,
            head_hashes=head_hashes, coldkey_of=coldkey_of, commit_blocks=blocks,
            raw_commit_blocks=raw, v5_inputs=self.inputs, burn_uid=0, pooled=(),
            published=self.view(epoch))
        monkeypatch.setattr(v5_epoch, "score_epoch", real)
        commit()
        # The reveal this epoch publishes is what the published set folds at E.
        self.folded[epoch] = memo_v4.published_ids(scored.reveal)
        self.results[epoch] = scored
        return scored, dict(zip(uids, weights))

    def run(self, first: int, last: int, monkeypatch):
        for e in range(first, last + 1):
            self.epoch(e, monkeypatch)
        return self


def bound_n(scored, head: str, key: str) -> int | None:
    """n of one comparison in a decision's evidence (None when not made)."""
    doc = scored.decision.evidence["heads"].get(head, {})
    for k in ((key,) if key != "vs_baseline" else
              ("vs_baseline", "b_over_holder", "paid_vs_baseline")):
        if k in doc:
            return doc[k]["n"]
    return None


def n_min() -> int:
    return seat_v5.N_MIN
