"""Protocol v4 end to end: miners' TDs, a root validator and a follower on one
chain and one public store, over twenty-one epochs, with nothing stood in
for but the chain, the store and the model calls.

Each epoch, each miner's TD reads its menu through `td_v4.V4Session` (the
root's commitment and the store's bytes) and runs `harness.run_benchmark`
under protocol 4. Each validator reads the
same menu at the boundary (`menu_publish.MenuNode`), checks every proof
with `verify_proof(protocol_version=4)` against `expected_v4`, pools the
pairs (`_v4_pool`) and runs the validator loop's own `_v4_menu_epoch`, with
the real `menu_inputs`: the root steps, publishes and commits; the follower
keeps its summaries and audits the root at k + 2.

Each epoch's plan is `epoch_plan.plan_v4` with the published set as of
E - 2 (a fixture seed, then every epoch's published questions folded in, as
a reveal publishes them), built alike by the TDs (their session's
`published` view) and by the validators: the scored slice is the plan's own,
drawn from published questions only, and every hotkey's exploration holds
the epoch's fresh questions.

What it shows: every TD and both validators run the same menu each epoch;
the follower's audits find no difference; every proof carries the epoch's
fresh questions; and a price the listing moved reaches the published menu
only at the refresh epoch (MENU_REFRESH_EPOCHS), not before. (A proposal
filed on chain, its join at the boundary and the field measuring it run
through the shipped loops in tests/test_v4_loop.py and
scripts/rehearsal_v4.py: this test has no chain fields or join step.)
"""
from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest

from fugal_subnet import config, epoch_plan, menu_inputs, reveal_sync
from fugal_subnet import menu_publish as mp
from fugal_subnet import menu_state as ms
from fugal_subnet.benchmarks.slicer import epoch_id_for_block
from fugal_subnet.tee import harness, td_v4
from fugal_subnet.tee import verify as verify_mod
from fugal_subnet.tee.proof import compute_questions_hash
from fugal_subnet.tee.runtime import TEERuntime
from tests import menu_fixtures as fx
from tests.test_harness_v4 import Provider, _listing, _Proxy
from tests.test_menu_publish import (
    BPE,
    BUCKET,
    CADENCE,
    LIST_BYTES,
    NETUID,
    Uploader,
    World,
    signer,  # noqa: F401 - the fixture
)
from tests.test_tee_e2e import _head
from tests.test_v4_seat_odds import _head as _head_v4

SLICE = config.SLICE_SIZE_V4
FIRST, LAST = 100, 119
MOVE_FROM = 110                       # the listing moves lab-a's price from this epoch
RESTART = 112                         # the follower restarts with an empty window here
MINERS = [(f"5Fminer-{i}", f"5Fcold-{i}") for i in range(4)]
PROPOSER = MINERS[0][0]
NEW_ID = "lab-c/model-new@high"
NEW_PRICE = "0.00000015"              # $ per prompt token, from MOVE_FROM on


@pytest.fixture(autouse=True)
def _small_menus(monkeypatch):
    fx.allow_small_menus(monkeypatch)


def _pool():
    return [{"question_id": f"{b}-{i:03d}", "prompt": f"Question {i}?", "gold": str(i),
             "grader_id": "numeric_final", "benchmark": b, "metadata": {}}
            for b in sorted(config.BENCHMARK_CATEGORY_V3) for i in range(70)]


class Listing(Provider):
    """The provider: listing pages that move lab-a's prompt price from
    MOVE_FROM on, and stub model calls."""

    epoch = FIRST

    def fetch(self, model):
        self.fetched.append(model)
        if model == "lab-a/model-one" and self.epoch >= MOVE_FROM:
            page = json.loads(_listing(model))
            for ep in page["data"]["endpoints"]:
                ep["pricing"]["prompt"] = NEW_PRICE
            return json.dumps(page).encode()
        return _listing(model)


def _nonce(k: int) -> str:
    return hashlib.sha256(f"e2e-{k}".encode()).hexdigest()


class _Published:
    """The TDs' published set (td_v4.V4Session's `published`): the view as
    of E - 2 of the set the test folds, as memo_sync.ChainSet advances it
    from the root's commitments (tests/test_memo_sync.py)."""

    def __init__(self):
        self.held = None

    def view_for(self, epoch):
        return self.held.view(int(epoch) - 2)


def test_miners_root_and_follower_run_v4_end_to_end(tmp_path, signer, monkeypatch):  # noqa: F811
    from fugal_subnet import memo_v4
    from neurons import validator
    from tests import v4_world

    # The production step parameters: the v4 mix and the compiled cadence.
    w = World(tmp_path, signer, step_params=epoch_plan.menu_step_params_v4(5000))
    follower = w.node("follower")
    monkeypatch.setattr("fugal_subnet.commit_reveal.EPOCH_DIR", str(w.epoch_dir))
    pool = _pool()
    by_id = {q["question_id"]: q for q in pool}
    category_of = {q["question_id"]: epoch_plan.mix_v4().bench_category[q["benchmark"]]
                   for q in pool}
    hidden = np.random.RandomState(0).randn(len(pool), 1024).astype(np.float32)
    # The release's published-set seed for this pool (through FIRST - 2, two
    # questions of each set unpublished), and the set as FIRST - 1 left it.
    # Each epoch's plan is the production one on it, scored slice included.
    seed = v4_world.fixture_seed(pool, FIRST - 2, v4_world.fixture_unpublished(pool))
    v4_world.install_seed(tmp_path, monkeypatch.setattr, seed)
    published = _Published()
    published.held = memo_v4.fold(memo_v4.seed_published(pool, seed), FIRST - 1, ())
    provider = Listing()
    monkeypatch.setattr(harness, "_call_model", provider.call)
    heads = {hk: _head(list(w.base.ids()), seed=i) for i, (hk, _ck) in enumerate(MINERS)}
    # One miner's head carries no token means: a v4 head (W7) that the v3
    # loader would refuse, run by its TD and verified by both validators.
    heads[MINERS[3][0]] = _head_v4(list(w.base.ids()), seed=3)
    coldkey_of = {uid: ck for uid, (_hk, ck) in enumerate(MINERS)}

    def list_bytes(serial, sha):
        return (LIST_BYTES, signer.sign(LIST_BYTES)) if sha == fx.ALLOW.sha256 else None

    sessions = {hk: td_v4.V4Session(
        first_epoch=90, pool=pool, published=published, fetch_listing=provider.fetch,
        read_commitment=w.chain.read,
        menu_bytes=lambda h: reveal_sync.fetch_menu_bytes(h, bucket=BUCKET, netuid=NETUID,
                                                          opener=w.store.opener),
        list_bytes=list_bytes, anchor=w.base) for hk, ck in MINERS}

    menus, audits, ledger = {}, [], {}
    for k in range(FIRST, LAST + 1):
        provider.epoch = k
        w.chain.block = k * BPE
        nonce_hex = _nonce(k)
        nonce = bytes.fromhex(nonce_hex)
        plan = epoch_plan.plan_v4(pool, published=published.held, epoch=k)
        fresh = {q["question_id"] for q in memo_v4.fresh_questions(
            nonce, plan.pool_index, epoch_plan.mix_v4().targets)}
        assert fresh and not fresh & set(published.held.view(k - 2).first)
        # --- the miners' TDs --------------------------------------------------
        proofs = {}
        for uid, (hk, _ck) in enumerate(MINERS):
            inputs = sessions[hk].inputs_for(k)
            proof = harness.run_benchmark(
                nonce=nonce_hex, head_bytes=heads[hk], benchmark_pool=pool, proxy=_Proxy(port=0),
                hidden_states=hidden, slice_size=SLICE, epoch_id=epoch_id_for_block(k),
                source_hash="img", hotkey=hk, protocol_version=4, v4=inputs)
            # The TD runtime's attestation over the proof, as on a local testnet.
            proof.attestation_quote = TEERuntime(mock=True).generate_attestation(
                bytes.fromhex(proof.content_hash()))
            sessions[hk].ran(inputs, proof)
            proofs[uid] = proof
            assert fresh <= {r.question_id for r in proof.exploration_results}, (k, hk)
        # --- the validators ---------------------------------------------------
        if k == RESTART:
            # The follower restarts having lost its window of summaries (and,
            # in memory, its kept step inputs); it keeps the menus it held.
            (tmp_path / "follower" / "summaries.json").unlink()
            follower = w.node("follower")
            assert follower.book.summaries(through=k, size=48) == []
        ran_on = {ms.menu_hash(s.previous) for s in sessions.values()}
        for name, node in (("follower", follower), ("root", w.root)):
            root = name == "root"
            w.chain.block = k * BPE + (5 if root else 0)
            read = node.at_boundary(k)
            assert read.ok and {ms.menu_hash(read.state)} == ran_on, (k, name)
            state = read.state
            slice_ids = [q["question_id"] for q in plan.scored_slice(nonce, pool, SLICE)]
            assert set(slice_ids) <= set(published.held.view(k - 2).first)
            expected, verified = {}, {}
            for uid, (hk, _ck) in enumerate(MINERS):
                x = validator._v4_expected(state, epoch_index=k, nonce=nonce, pool=pool,
                                           scored_ids=slice_ids, hotkey=hk, plan=plan)
                gold = validator._v4_gold({q: by_id[q] for q in slice_ids}, by_id, x)
                result = verify_mod.verify_proof(
                    proofs[uid], approved_measurements=set(),
                    expected_questions_hash=compute_questions_hash(slice_ids),
                    expected_nonce=nonce_hex, gold_answers=gold,
                    expected_question_ids=set(slice_ids),
                    expected_weights_hash=hashlib.sha256(heads[hk]).hexdigest(),
                    expected_hotkey=hk, expected_proof_hash=proofs[uid].content_hash(),
                    head_bytes=heads[hk], mock=True, protocol_version=4, v4=x)
                assert result.valid, (k, name, hk, result.reason)
                expected[uid], verified[uid] = x, proofs[uid]
            pooled = validator._v4_pool(state, verified_proofs=verified, expected=expected,
                                        coldkey_of=coldkey_of, nonce=nonce, flagged=set())
            fields, _published, _deferred = validator._v4_menu_epoch(
                node, w.publisher if root else None, read, epoch_index=k, nonce=nonce,
                verified_proofs=verified, coldkey_of=coldkey_of, pooled=pooled,
                category_of=category_of, counted_heads=[hashlib.sha256(heads[PROPOSER]).hexdigest()],
                window_epochs=48, root=root, uploader=Uploader(w.store) if root else None,
                reveal_delay=1, netuid=NETUID, commit=w.chain.commit,
                root_summaries=mp.root_summaries(str(w.epoch_dir), None if root else ledger.get,
                                                 check=menu_inputs.summary_from_dict))
            if root:
                assert fields[mp.REVEAL_SUMMARY]["epoch"] == k
                folder = w.epoch_dir / epoch_id_for_block(k)
                folder.mkdir()
                raw = json.dumps({"epoch_id": epoch_id_for_block(k), "protocol_version": 4,
                                  **fields}).encode()
                (folder / "reveal.json").write_bytes(raw)
                # What follower sync records from the root's commitment.
                ledger[k] = hashlib.sha256(raw).hexdigest()
            else:
                assert mp.REVEAL_SUMMARY not in fields
                if fields.get("menu_audit"):
                    audits.append((k, fields["menu_audit"]))
        menus[k] = w.root.book.held(k)
        # Epoch k's published questions, as its reveal lists them: the slice
        # and every exploration answer.
        published.held = memo_v4.fold(published.held, k, set(slice_ids) | {
            r.question_id for p in proofs.values() for r in p.exploration_results})
    w.chain.block = (LAST + 1) * BPE
    menus[LAST + 1] = follower.at_boundary(LAST + 1).state
    assert menus[LAST + 1] == w.root.at_boundary(LAST + 1).state

    # No proof carries a proposal: proposals are filed on chain.
    assert all(NEW_ID not in m.ids() for m in menus.values())
    # The follower audited the root and found nothing. It says it could not
    # audit the first two scored epochs (98 and 99 precede its first proofs)
    # and the two whose inputs its restart lost; after the restart its window
    # was rebuilt from the root's published summaries, and its audits say so.
    assert [k for k, a in audits if a["differences"] is None] == [
        FIRST, FIRST + 1, RESTART, RESTART + 1]
    checked = {k: a for k, a in audits if a["differences"] is not None}
    assert len(checked) == LAST - FIRST + 1 - 4
    assert all(a["differences"] == [] and a["own"] == a["root"] for a in checked.values())
    assert all(("window_from_root" in a) is (k >= RESTART + 2) for k, a in checked.items())
    assert checked[RESTART + 2]["window_from_root"] == [
        epoch_id_for_block(e) for e in range(FIRST, RESTART)]
    assert follower.book.from_root(through=LAST, size=48) == list(range(FIRST, RESTART))
    # The refresh cadence: lab-a's price moved in the listing at MOVE_FROM; it
    # reaches the menu only at the refresh epoch.
    price = {k: m.entry("lab-a/model-one@high").price_in for k, m in menus.items()}
    assert {price[k] for k in range(MOVE_FROM, LAST + 1)} == {price[MOVE_FROM]}
    assert price[LAST + 1] == 150_000 != price[LAST]
    assert ms.refresh_due(LAST + 1, CADENCE)
    assert "mean_in_tokens" not in np.load(io.BytesIO(heads[MINERS[3][0]]), allow_pickle=False)

