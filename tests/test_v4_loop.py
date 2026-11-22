"""Protocol v4 through the shipped loops: `neurons/validator.main` itself (a
root and a follower, each its real epoch loop, run concurrently) and the
miner's real epoch function (`neurons/miner._run_epoch` on the session
`neurons/miner._v4_session` builds, its answer served by `_serve_proof`).

Stood in for, and nothing else:
- the chain: one fake subtensor (blocks, block hashes, the metagraph, the
  Commitments pallet with pinned reads, weights), shared by every neuron;
- the transport: a dendrite whose query hands each miner's axon answer back
  (`miner._serve_proof`), and a cooperative clock: `time.sleep` in the
  validator module yields to the other validator, and the chain advances one
  block when every validator has slept;
- the public store: one bucket, read through `urllib.request.urlopen` (the
  one socket `reveal_sync` opens) and written through the uploader and the
  menu store the validator constructs;
- the model provider: recorded listing pages and stub replies
  (tests/test_harness_v4.Provider), and a metering proxy that binds no port.
  One miner's calls are always answered right, a TD too good to be true.

The pool and its embeddings are fixture files read by the real loaders
(FUGAL_BENCHMARK_POOL*, FUGAL_EMBEDDINGS_FILE, FUGAL_POOL_UNPINNED, as a
local-chain rehearsal sets them). Protocol 4 is selected the way a non-finney
chain selects it: the v4 activation variable (config.V4_ACTIVATION_BLOCK_ENV,
FUGAL_V4_ACTIVATION_BLOCK), with the v2 and v3 blocks before it. Every
validator holds the same menu on disk when it starts (a menu naming the
fixture lab and host list, which is in the store as the root published it),
and each miner's session is handed the same menu as its anchor. The
single-model table's count is shortened so it is published in the run.

What it shows over nine epochs: every TD and both validators run each epoch
on the same menu and check every proof; the scored questions are optional
(SEAT_MINIMAL_V4.md 3.4): one miner answers them every epoch as the seat
holder must (its "auto" choice reading itself as the holder, a stand-in for a
seat nine epochs cannot fill), one turns them on mid-run, and one never
answers them (an odds-only miner, exploration only), every proof of each kind
verified, rule B's evidence published for each and rebuilt from the reveals;
every verified head, the odds-only one included, has odds and its label
standing every epoch (SEAT_MINIMAL_V4.md 3.2), and only the heads that
answered have a picks record; one miner files a proposal on chain in an
epoch and it is on the next epoch's menu, provisional, measured by every
box's coverage there, and full from the step that scores its join epoch
(FAST_CROWN_V4.md 18), the follower's join audit agreeing; the menu step
summarises exactly the
pool the seat's odds are scored on (which leaves out the too-good TD where
only the implausibility check catches it); the window is categorised by the
v4 mix and priced at the menu; both validators reach the same seat state and
set the same weights every epoch; the root's reveals carry the v4 fields and
are published, each commitment naming bytes the store already serves; a
failed pinned metagraph read makes the follower run the epoch again; every
reveal carries the memorisation check's fields, the same bytes on both
validators, naming the published set the root's public reveals fold to (the
seed, a fixture's, held by every node and TD as the epochs before left it),
with each epoch's fresh questions in every miner's exploration and each
named view's snapshot in the store at its digest; and the follower's audits
of the root (the menu and the published set) find no difference.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time as _real_time
import types
import urllib.request

import numpy as np
import pytest

from fugal_subnet import (
    commit_reveal,
    config,
    epoch_logger,
    epoch_plan,
    explore_pool,
    head_odds,
    head_publisher,
    memo_v4,
    picks_v4,
    reveal_publisher,
    reveal_sync,
)
from fugal_subnet import menu_publish as mp
from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from fugal_subnet.benchmarks.slicer import collect_block_for_epoch, epoch_id_for_block
from fugal_subnet.tee import harness
from fugal_subnet.tee import runtime as tee_runtime
from fugal_subnet.v2_sync import MENU_FIELD
from tests import menu_fixtures as fx
from tests import v4_world
from tests.test_menu_publish import LIST_BYTES, signer  # noqa: F401 - the fixture
from tests.test_tee_e2e import _head

# The world the neurons run in (chain, transport, clock, store, provider):
# shared with tests/test_v4_flip.py and scripts/rehearsal_v4.py.
from tests.v4_world import (  # noqa: F401 - re-exported for the tests that import them here
    BPE,
    BUCKET,
    FOLLOWER,
    GENESIS,
    INTERVAL,
    MINERS,
    NETUID,
    NEURONS,
    NEW_ID,
    OWNER,
    PROPOSAL,
    ROOT,
    WALLETS,
    Chain,
    Dendrite,
    Lockstep,
    Metagraph,
    Miner,
    Provider,
    Store,
    Subtensor,
    Wallet,
)
from tests.v4_world import fixture_pool as _pool
from tests.v4_world import store_classes as _store_classes
from tests.v4_world import write_inputs as _write_inputs

V2_EPOCH, V3_EPOCH, V4_EPOCH = 1, 5, 10
FIRST, LAST = 20, 28                  # the epochs both validators run in full
RETRY = 24                            # the follower's pinned metagraph read fails once here
# The scored questions: MINERS[0] answers every epoch as the holder must,
# MINERS[1] turns them on at SWITCH, MINERS[2] never answers them.
HOLDER, SWITCHER, ODDS_ONLY = (m[0] for m in MINERS)
SWITCH = FIRST + 3
FILED = FIRST + 1                     # the proposer files in this epoch (it served FIRST)


def test_the_validator_loop_and_the_miner_epoch_run_protocol_v4(tmp_path, signer, monkeypatch):  # noqa: F811
    import bittensor

    from fugal_subnet import collect
    from neurons import miner as miner_mod
    from neurons import validator

    # The fixture menu has three entries, below the compiled minimum (held on a wide menu
    # in tests/test_menu_liveness.py); this test is about the loop, so the reader is told
    # to accept it, as the other plumbing tests do.
    fx.allow_small_menus(monkeypatch)

    # --- protocol 4 off finney, as a local chain selects it ------------------------------
    monkeypatch.setattr(config, "EPOCH_INTERVAL", INTERVAL)
    monkeypatch.setattr(config, "V2_ACTIVATION_BLOCK", V2_EPOCH * BPE)
    monkeypatch.setattr(config, "V3_ACTIVATION_BLOCK_ENV", V3_EPOCH * BPE)
    monkeypatch.setattr(config, "V4_ACTIVATION_BLOCK_ENV", V4_EPOCH * BPE)
    monkeypatch.setattr(config, "ODDS_SEAT_BLOCK_ENV", None)
    monkeypatch.setattr(config, "MENU_B_BLOCK_ENV", None)
    # The single-model table's count shortened, as the rehearsals shorten
    # counts, so the table (accuracy by the v4 mix's categories and targets)
    # is published within nine epochs.
    monkeypatch.setattr(config, "SINGLE_MODEL_MIN_PAIRS", 60)
    assert config.protocol_for_chain(GENESIS) == 4
    with pytest.raises(ValueError, match="FUGAL_V4_ACTIVATION_BLOCK"):
        config.protocol_for_chain(config.FINNEY_GENESIS_HASH)

    pool = _pool()
    pool_path, emb_path = _write_inputs(tmp_path, pool)
    for var in ("FUGAL_BENCHMARK_POOL", "FUGAL_BENCHMARK_POOL_V2", "FUGAL_BENCHMARK_POOL_V3"):
        monkeypatch.setenv(var, str(pool_path))
    monkeypatch.setenv("FUGAL_POOL_UNPINNED", "1")
    monkeypatch.setenv("FUGAL_EMBEDDINGS_FILE", str(emb_path))
    # The release's published-set seed for the fixture pool, observed through
    # FIRST - 2 with two questions of each set unpublished (the reserve: one
    # fresh question an epoch). The validators and the TDs hold the set as it
    # advance it from the chain from FIRST on.
    seed = v4_world.fixture_seed(pool, FIRST - 2, v4_world.fixture_unpublished(pool))
    v4_world.install_seed(tmp_path, monkeypatch.setattr, seed)

    # --- the chain, the store, the transport, the provider ------------------------------------
    store = Store()
    chain = Chain(store)
    monkeypatch.setattr(urllib.request, "urlopen", store.urlopen)
    monkeypatch.setattr(bittensor, "Subtensor", lambda *a, **k: Subtensor(chain))
    monkeypatch.setattr(bittensor, "Wallet", Wallet)
    monkeypatch.setattr(collect, "CollectDendrite", Dendrite)
    uploader_cls, menu_store_cls = _store_classes(store)
    monkeypatch.setattr(reveal_publisher, "GcsUploader", uploader_cls)
    monkeypatch.setattr(mp, "GcsStore", menu_store_cls)
    provider = Provider(cheater=MINERS[2][0])
    monkeypatch.setattr(harness, "_call_model", provider.call)
    from fugal_subnet import join_step
    monkeypatch.setattr(join_step, "fetch_public_listing", lambda model, **_k: provider.fetch(model))
    monkeypatch.setattr(tee_runtime.MeteringProxy, "start", lambda self: None)
    monkeypatch.setattr(tee_runtime.MeteringProxy, "stop", lambda self: None)
    monkeypatch.setattr(config, "REFERENCE_BUCKET", BUCKET)
    monkeypatch.setattr(config, "ROOT_HOTKEY_ENV", ROOT[0])
    # Heads are files under v4 as under v3: the root publishes every verified
    # head by hash after the reveal that lists it (no hold here, so every head
    # of every published reveal is due within the run).
    monkeypatch.setattr(config, "PUBLISH_HEADS_AFTER_EPOCHS", 0)

    # The menu every node holds when it starts (the fixture menu, epoch
    # FIRST - 1), and the lab and host list it names, in the store at its
    # content address as the root published it when it was adopted.
    base = fx.state(epoch=FIRST - 1)
    sha = hashlib.sha256(LIST_BYTES).hexdigest()
    assert sha == base.labs.sha256
    store.objects[reveal_sync.list_object(NETUID, mr.KIND_MENU_ALLOWLIST, sha)] = LIST_BYTES
    store.objects[reveal_sync.signature_object(NETUID, mr.KIND_MENU_ALLOWLIST, sha)] = (
        signer.sign(LIST_BYTES))

    # --- the miners: heads committed, sessions built as miner.main builds them -------------------
    chain.block = 1
    hidden = miner_mod._load_pinned_embeddings(pool, mock=True)
    world = types.SimpleNamespace(miners={})
    dirs = {name: tmp_path / name for name in ("root", "follower")}
    td_dirs = {hk: tmp_path / "td" / hk for hk, _ck in MINERS}
    for i, (hk, ck) in enumerate(MINERS):
        head = _head(list(base.ids()), seed=i)
        head_hash = hashlib.sha256(head).hexdigest()
        chain.commit(hk, head_hash)
        # The holder's "auto" reads itself as the seat holder (the stand-in for
        # the root's reveal naming it); the others start with the scored
        # questions off, as a miner that does not hold the seat does by default.
        # No challenge (the holder answers none; the others are odds only).
        scored = dict(challenge_size=0)
        session = miner_mod._v4_session(
            Subtensor(chain), netuid=NETUID, pool=pool, first_epoch=V4_EPOCH,
            blocks_per_epoch=BPE, coldkey=ck, genesis=GENESIS,
            fetch_listing=provider.fetch, anchor=base, state_dir=td_dirs[hk], **scored)
        world.miners[hk] = Miner(hk, ck, head, session)
    Dendrite.world = world

    def run_miners(epoch: int):
        if epoch > LAST:
            return
        if epoch == FILED:
            # The proposer files its proposal: its document served from its
            # box, its hash committed beside its head, one transaction, early
            # in the epoch (before the root's pinned read at block B).
            m = world.miners[MINERS[0][0]]
            chain.block = epoch * BPE + 1
            chain.commit(m.hotkey, m.propose(PROPOSAL))
            chain.block = epoch * BPE
        if epoch == SWITCH:
            # The operator turns a 500-question challenge on (--v4-challenge 500).
            world.miners[SWITCHER].session.challenge.size = 500
        for m in world.miners.values():
            provider.serving = m.hotkey
            miner_mod._run_epoch(
                blocks_per_epoch=BPE, head_data=m.head, weights_hash=m.weights_hash, pool=pool,
                hidden_states=hidden, epoch_index=epoch, block_hash=chain.block_hash(epoch * BPE),
                tee_runtime=tee_runtime.TEERuntime(mock=True), current_proof=m.current,
                slice_size=config.SLICE_SIZE_V4, proxy_port=0, explore_models=[], explore_size=0,
                hotkey=m.hotkey, protocol_version=4, prices=None,
                block_now=lambda: chain.block, v4_session=m.session)
            proof = m.current["proof"]
            assert proof.epoch_id == epoch_id_for_block(epoch)
            m.menus[epoch] = proof.menu_hash
            m.filed[epoch] = m.proposal_doc is not None
            m.scored[epoch] = (proof.scored, len(proof.scored_results))
            m.spend[epoch] = proof.total_cost_usd
            m.calls[epoch] = (len(proof.exploration_results),
                              math.fsum(r.cost_usd for r in proof.exploration_results))
            m.asked[epoch] = {r.question_id for r in proof.scored_results} | {
                r.question_id for r in proof.exploration_results}

    # --- the validators: their own state, epochs and role -------------------------------------
    for name, d in dirs.items():
        (d / "epochs").mkdir(parents=True)
        mp.MenuBook(str(d / "state.json") + ".menu").hold(base)
    contexts = {
        name: [(validator, "STATE_PATH", str(d / "state.json")),
               (commit_reveal, "EPOCH_DIR", str(d / "epochs")),
               (epoch_logger, "EPOCH_LOG_DIR", str(d / "logs")),
               (config, "ROOT_VALIDATOR", True if name == "root" else None),
               (config, "REVEAL_BUCKET", BUCKET if name == "root" else "")]
        for name, d in dirs.items()}
    clock = Lockstep(chain, contexts, run_miners, stop_block=(LAST + 1) * BPE + 1)
    monkeypatch.setattr(validator, "time", types.SimpleNamespace(
        sleep=clock.sleep, time=_real_time.time, monotonic=_real_time.monotonic))
    errors: list[tuple[str, str]] = []
    real_exception = validator.logger.exception

    def record(msg, *a, **k):
        errors.append((threading.current_thread().name, str(sys.exc_info()[1])))
        real_exception(msg, *a, **k)

    monkeypatch.setattr(validator.logger, "exception", record)
    # A spy on the menu step (it calls through): which pool the loop hands it.
    handed: dict[tuple[str, int], list] = {}
    real_menu_epoch = validator._v4_menu_epoch

    def menu_epoch(*a, **k):
        handed[(threading.current_thread().name, k["epoch_index"])] = k["pooled"]
        return real_menu_epoch(*a, **k)

    monkeypatch.setattr(validator, "_v4_menu_epoch", menu_epoch)
    # ... the shares every hotkey's expectation is drawn with ...
    shares_seen: dict[tuple[str, int], list] = {}
    real_expected = validator._v4_expected

    def expected(state, **k):
        shares_seen.setdefault((threading.current_thread().name, k["epoch_index"]), []).append(
            (state, k["shares"]))
        return real_expected(state, **k)

    monkeypatch.setattr(validator, "_v4_expected", expected)
    # ... and whether the seat's odds are the v4 odds on the epoch's menu.
    odds_on: dict[str, int] = {}
    real_pair_odds = head_odds.pair_odds_v4_many

    def pair_odds(heads, state, *a, **k):
        # every verified head of the epoch at once (the integers of pair_odds_v4)
        heads = list(heads)
        name = threading.current_thread().name
        odds_on[name] = odds_on.get(name, 0) + len(heads)
        return real_pair_odds(heads, state, *a, **k)

    monkeypatch.setattr(head_odds, "pair_odds_v4_many", pair_odds)
    # ... and every audit of a root reveal's named published set.
    from fugal_subnet import memo_sync
    audit_findings: list[str] = []
    real_check = memo_sync.check_named

    def check_named(published, r):
        got = real_check(published, r)
        if got is not None:
            audit_findings.append(got)
        return got

    monkeypatch.setattr(memo_sync, "check_named", check_named)
    # The follower's pinned metagraph read fails once, at RETRY's collection
    # block: under v4 the epoch fails and is run again, never on the current
    # metagraph.
    chain.fail_metagraph_at["follower"] = collect_block_for_epoch(
        RETRY, BPE, config.EPOCH_COLLECT_FRACTION)

    chain.block = FIRST * BPE
    run_miners(FIRST)

    def start(name):
        return lambda: validator.main.callback(
            network="local", netuid=NETUID, coldkey=name, hotkey="default", wallet_path=None,
            once=False, log_level="WARNING", live=False)

    threads = [threading.Thread(target=clock.run, args=(n, start(n)), name=n, daemon=True)
               for n in dirs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3600)
    assert not any(t.is_alive() for t in threads), "a validator loop did not stop"
    assert clock.errors == {}, clock.errors
    assert len(errors) == 1 and errors[0][0] == "follower", errors
    assert "metagraph at the collection block could not be read" in errors[0][1]
    assert chain.fail_metagraph_at == {}
    assert chain.future_reads == [], chain.future_reads[:5]

    # --- each validator checked every proof of every epoch ----------------------------------------
    epochs = range(FIRST, LAST + 1)
    for d in dirs.values():
        logs = [json.loads(line) for line in (d / "logs" / "epochs.jsonl").read_text().splitlines()]
        assert [x["epoch_id"] for x in logs] == [epoch_id_for_block(k) for k in epochs]
        assert all(x["n_heads_valid"] == len(MINERS) and x["n_heads_invalid"] == 0
                   and x["n_heads_unverifiable"] == 0 and x["set_weights_success"] for x in logs)

    # --- every TD and both validators ran each epoch on the same menu ------------------------------
    held = {name: mp.MenuBook(str(d / "state.json") + ".menu") for name, d in dirs.items()}
    for k in epochs:
        on = {ms.menu_hash(held[n].held(k)) for n in held} | {m.menus[k] for m in world.miners.values()}
        assert len(on) == 1, (k, on)

    # --- the proposal: filed at FILED, on the menu at FILED + 1, full at FILED + 3 ----------------
    menus = {k: held["root"].held(k) for k in epochs}
    joined = [k for k, m in menus.items() if NEW_ID in m.ids()]
    assert joined and min(joined) == FILED + 1                      # the next epoch
    first = menus[FILED + 1].entry(NEW_ID)
    assert first.provisional and first.joined_epoch == FILED + 1
    assert first.proposer == MINERS[0][0] and first.token_pairs == 0
    # Every box's coverage went to it in its join epoch, and to no
    # provisional entry in any other.
    assert epoch_plan.provisional_first(menus[FILED + 1]) == (NEW_ID,)
    assert all(epoch_plan.provisional_first(menus[k]) == () for k in epochs if k != FILED + 1)
    # The step scoring its join epoch decided it: full from FILED + 3.
    full = [k for k in joined if not menus[k].entry(NEW_ID).provisional]
    assert full and min(full) == FILED + 1 + ms.MENU_LAG
    assert menus[min(full)].entry(NEW_ID).scored_from == FILED + 1 + ms.MENU_LAG
    assert menus[min(full)].entry(NEW_ID).token_pairs > 0
    # The follower's audit of the root's joins agrees, every epoch.
    for k in epochs:
        audit = json.loads((dirs["follower"] / "epochs" / epoch_id_for_block(k)
                            / "reveal.json").read_text()).get("join_audit")
        assert audit is not None and audit["differences"] == [], (k, audit)

    # --- one plan, the epoch's shares for every hotkey, odds on the menu ------------------------------
    for (name, k), seen in shares_seen.items():
        assert len(seen) == len(MINERS)
        assert all(state is seen[0][0] and sh is seen[0][1] for state, sh in seen)
        assert seen[0][1] == epoch_plan.explore_shares_v4(menus[k])
    assert sorted({k for _n, k in shares_seen}) == list(epochs)
    # Every verified head, each epoch, whether or not it answered the scored
    # questions (labels_v4; see head_odds below).
    assert set(odds_on) == set(dirs) and all(n >= len(MINERS) * len(epochs)
                                             for n in odds_on.values())

    # --- weights: set every epoch, the same vector on both validators --------------------------------
    by = {}
    for block, hk, uids, ws in chain.weights:
        by.setdefault(block // BPE, {})[hk] = dict(zip(uids, ws))
    assert sorted(by) == list(epochs)
    assert all(v[ROOT[0]] == v[FOLLOWER[0]] for v in by.values())
    assert all(abs(sum(v[ROOT[0]].values()) - 1.0) < 1e-6 for v in by.values())

    # --- the root's reveals: v4 fields, published, committed after the bytes ---------------------------
    def reveal(name, k):
        return json.loads((dirs[name] / "epochs" / epoch_id_for_block(k) / "reveal.json").read_text())

    benchmark_of = {q["question_id"]: q["benchmark"] for q in pool}
    # The too-good TD is left out of the odds pool by the implausibility
    # check, which the menu step's pool shares. The seat runs v4's rules
    # (seat_v4), the same on both validators (random heads do not beat the
    # reference, so no seat is filled and the weight burns to the owner; the
    # seat's own scenarios are in tests/test_v4_seat_loop.py). What the window is built from shows in the
    # single-model table: accuracy by the v4 mix's categories, and spend at
    # the epoch's menu prices (the stub bills 500 prompt and 300 completion
    # tokens a call).
    assert all(reveal("root", k)["v2_state"] == reveal("follower", k)["v2_state"]
               for k in epochs)
    assert all(by[k][ROOT[0]] == {0: 1.0} for k in epochs)
    table = reveal("root", LAST)["single_model_table"]
    # The new entry joined at FIRST + 5 and has been explored since: it is in
    # the table with every other entry.
    assert {row["model"] for row in table} == set(menus[LAST].ids())
    for row in table:
        e = menus[LAST].entry(row["model"])
        targets = epoch_plan.mix_v4().targets
        assert set(row["by_category"]) <= set(targets)
        cats = row["by_category"]
        assert row["accuracy"] == pytest.approx(
            sum(targets[c] * cats[c]["accuracy"] for c in cats) / sum(targets[c] for c in cats))
        assert row["spend_per_question"] == pytest.approx(
            (500 * e.price_in + 300 * e.price_out) / ms.PICO, rel=1e-12)
    cheater = MINERS[2][1]
    # Under v4 neither the v1-frame nor the agreement detector runs
    # (SEAT_MINIMAL_V4.md 2.3 item 32): a v4 reveal names no frame exclusion,
    # and the too-good TD is left out of the odds pool by the plausibility
    # check against R alone.
    assert all("frame_excluded_coldkeys" not in reveal("root", k) for k in epochs)
    only_odds = [k for k in epochs if cheater in reveal("root", k)["exploration_excluded_coldkeys"]]
    assert only_odds
    for k in epochs:
        r = reveal("root", k)
        assert r["protocol_version"] == 4
        assert r[MENU_FIELD]["hash"] == ms.menu_hash(menus[k])
        assert r[mp.REVEAL_SUMMARY]["epoch"] == k and r[mp.REVEAL_NEXT]
        assert r["v2_state"]["rules"] == {"protocol": 4, "seat": "odds"}
        # Every verified head has odds and its label standing every epoch,
        # whether or not it answered the scored questions: the odds-only miner
        # included (SEAT_MINIMAL_V4.md 3.2; labels_v4).
        every = {m.weights_hash for m in world.miners.values()}
        assert set(r["head_odds"]) == every and set(r["odds_v4"]) == every
        assert all(k in r["odds_v4"][h]["odds_epochs"] for h in every)
        assert r["v2_state"]["labels_digest"]
        for name in dirs:
            # The menu step summarised exactly the pool the seat's odds were
            # scored on (the one the reveal publishes), on both validators.
            rows = explore_pool.pairs_for_reveal(handed[(name, k)], benchmark_of)
            assert rows == reveal(name, k)["exploration_pool"] and rows
    published = sorted(key for key in store.objects if key.endswith("/reveal.json"))
    assert published == [f"{NETUID}/{epoch_id_for_block(k)}/reveal.json" for k in range(FIRST, LAST)]
    for k in range(FIRST, LAST):
        key = f"{NETUID}/{epoch_id_for_block(k)}/reveal.json"
        assert store.objects[key] == (dirs["root"] / "epochs" / epoch_id_for_block(k)
                                      / "reveal.json").read_bytes()
    # Every head a published reveal lists is in the store by its hash, with
    # its sidecar beside the reveal, as on v3 (fugal_subnet/head_publisher.py),
    # and no other head is.
    listed = set()
    for k in range(FIRST, LAST):
        hashes = set(reveal("root", k)["head_hashes"].values())
        assert len(hashes) == len(MINERS)
        assert head_publisher.sidecar_key(NETUID, epoch_id_for_block(k)) in store.objects
        listed |= hashes
    heads = {key: data for key, data in store.objects.items()
             if key.startswith(f"{head_publisher.HEADS_PREFIX}/")}
    assert heads.keys() == {head_publisher.head_key(NETUID, h) for h in listed}
    assert all(key == head_publisher.head_key(NETUID, hashlib.sha256(data).hexdigest())
               for key, data in heads.items())
    # The follower's own sync fetched the root's published reveals, each
    # checked against the root's commitment, as it does on finney.
    synced = dirs["follower"] / "reference-reveals" / str(NETUID)
    assert sorted(p.parent.name for p in synced.glob("*/reveal.json")) == [
        epoch_id_for_block(k) for k in range(FIRST, LAST)]
    # The root's commitment is two fields written together: the reveal
    # pointer, and the join pointer of the next epoch (chain_proposals).
    root_commits = [d if isinstance(d, tuple) else (d,) for _b, d in chain.commits[ROOT[0]]]
    firsts = []
    for f in root_commits:
        if not firsts or firsts[-1] != f[0]:
            firsts.append(f[0])
    named = [reveal_sync.parse_commitment(d) for d in firsts]
    assert [c[0] for c in named] == [epoch_id_for_block(k) for k in range(FIRST, LAST)]
    assert all(c[2] is not None for c in named)       # each names the menu it derived
    from fugal_subnet import chain_proposals as cp
    joins = {f[1] for f in root_commits if len(f) == 2}
    assert all(any(cp.parse_join_field(j, k) is not None for j in joins)
               for k in range(FIRST + 2, LAST + 1))

    # --- the challenge: optional, a prefix of the list, every kind verified -----------------------
    # Nobody challenges before SWITCH; from it the switcher answers a
    # 500-question challenge (no seat is filled in this run: the open form).
    for k in epochs:
        for hk, m in world.miners.items():
            on = hk in answering(k)
            assert m.scored[k] == (on, 500 if on else 0), (hk, k)
    holder, switcher, odds_only = (world.miners[hk] for hk in (HOLDER, SWITCHER, ODDS_ONLY))
    # An odds-only miner spends its exploration (the 112 calls of its private set,
    # within the exploration caps) and nothing on a challenge.
    for k in epochs:
        n, explore_spend = odds_only.calls[k]
        assert n == config.EXPLORE_PRIVATE_SIZE_V4
        assert odds_only.spend[k] == pytest.approx(explore_spend, rel=1e-12)
        assert switcher.spend[k] > odds_only.spend[k] if k >= SWITCH else True
    by_hash = {m.weights_hash: hk for hk, m in world.miners.items()}
    for k in epochs:
        r = reveal("root", k)
        block = r["picks_v4"]
        assert r["v2_state"][picks_v4.DIGEST_KEY] == picks_v4.digest(block)
        assert block == reveal("follower", k)["picks_v4"]
        assert {by_hash[h] for h in block["items"]} == answering(k)
        assert block["outage"] is False and block["failure_refused"] == {}
        assert block["holder"] is None and block["holder_picks"] == {}
        ev = {by_hash[h]: v for h, v in block["evidence"].items()}
        # No seat is filled in this run: every check is the open-seat count.
        assert all(v["bar"] is None for v in ev.values())
        assert ev[ODDS_ONLY]["run_epochs"] == 0 and not ev[ODDS_ONLY]["passes"]
        assert not ev[ODDS_ONLY]["answered"] and not ev[HOLDER]["answered"]
        assert ev[SWITCHER]["run_from"] == (SWITCH if k >= SWITCH else None)
        assert ev[SWITCHER]["run_epochs"] == max(0, k - SWITCH + 1)
        assert ev[SWITCHER]["passes"] is (k >= SWITCH)              # 500 items
    # A rebuild from the root's reveals reproduces every record and every
    # published verdict.
    from fugal_subnet.v2_epoch import load_window
    from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals

    t = validator._v4_tables(mp.MenuRead(LAST, menus[LAST], "carried"))
    rebuilt = rebuild_v2_from_reveals(str(dirs["root"] / "epochs"), window_size=48,
                                      prices=t.prices, bench_category=t.bench_category)
    saved = load_window(str(dirs["root"] / "state.json") + ".v2window.npz", 48)
    assert rebuilt.window.picks == saved.picks and sorted(saved.picks) == list(epochs)
    # The odds-only head: odds every epoch in the window, in the file and in
    # the rebuild alike, no answers and no picks record.
    odds_only_hash = odds_only.weights_hash
    assert odds_only_hash not in saved.heads and odds_only_hash not in rebuilt.window.heads
    assert sorted(saved.odds[odds_only_hash]) == list(epochs)
    assert all(odds_only_hash not in saved.picks[k].values for k in epochs)
    assert sorted(saved.odds) == sorted(rebuilt.window.odds)
    for h, per in saved.odds.items():
        assert sorted(per) == sorted(rebuilt.window.odds[h])
        assert all(np.array_equal(o, rebuilt.window.odds[h][e]) for e, o in per.items())
    # Labels: some are read for the odds-only head by the end of the run.
    assert reveal("root", LAST)["odds_v4"][odds_only_hash]["labels"] > 0
    for k in epochs:
        for h, published in reveal("root", k)["picks_v4"]["evidence"].items():
            got = picks_v4.picks_check(rebuilt.window.picks, h, None, through=k)
            assert picks_v4.evidence_to_reveal(
                got, answered_now=picks_v4.answered_in(rebuilt.window.picks, h, k)) == published

    # --- the memorisation check, driven by the loop's own published set ------------------------------
    # Every v4 reveal carries `memo_v4` and `v2_state.memo_digest`, the same
    # bytes on the root and the follower; the set each names is the view
    # through k - 2 of the root's public reveals folded from the seed,
    # computed here independently of the nodes; the epoch's fresh questions
    # (drawn from the reserve) are in every miner's exploration, and every
    # proof drew them (each verified, above); the root published each named
    # view's snapshot at its digest before committing the reveal; and the
    # follower found nothing to report against the root's digests.
    from fugal_subnet import memo_sync
    from fugal_subnet.v2_epoch import _digest_of

    root_reveals = {k: reveal("root", k) for k in epochs}
    for k in epochs:
        r, f = root_reveals[k], reveal("follower", k)
        assert r["memo_v4"] == f["memo_v4"], k
        assert r["v2_state"]["memo_digest"] == f["v2_state"]["memo_digest"] == _digest_of(
            r["memo_v4"]), k
        expect = v4_world.chain_view(chain, seed, pool, root_reveals, k)
        assert r["memo_v4"]["through"] == k - 2
        assert r["memo_v4"]["published_digest"] == expect.view(k - 2).digest, k
        fresh = set(r["memo_v4"]["fresh"])
        assert len(fresh) == r["memo_v4"]["fresh_count"] == 1, k
        assert not fresh & {q for q, e in expect.first.items() if e <= k - 2}, k
        for m in world.miners.values():
            assert fresh <= m.asked[k], (k, m.hotkey)
        assert fresh <= memo_v4.published_ids(r), k
    for k in range(FIRST, LAST):
        named = root_reveals[k]["memo_v4"]
        key = memo_sync.snapshot_object(NETUID, named["published_digest"])
        raw = store.objects[key]
        assert hashlib.sha256(raw).hexdigest() == named["published_digest"]
        assert memo_v4.published_from_snapshot(
            raw, named["published_digest"], through=named["through"]).digest == named[
                "published_digest"]
    # What each node holds beside its state: the fold of the public reveals.
    for d in dirs.values():
        v4_world.check_held(d / ("state.json" + memo_sync.STATE_SUFFIX), chain, seed, pool,
                            root_reveals, LAST)
    for k in epochs:
        ids = memo_v4.published_ids(reveal("root", k))
        assert ids == memo_v4.published_ids(reveal("follower", k))
        # The challenge list past the slice is drawn from questions already
        # published (through k - 2); its outcomes are published as sums.
        listed = [q for q, _c in reveal("root", k)["picks_v4"]["list"]]
        past_slice = set(listed[config.SLICE_SIZE_V4:])
        for m in world.miners.values():
            asked = m.asked[k] - past_slice
            assert asked <= ids, (k, m.hotkey, sorted(asked - ids)[:5])
    assert audit_findings == []

    # --- the follower audited the root and found nothing -----------------------------------------------
    audits = {k: reveal("follower", k).get(mp.REVEAL_AUDIT) for k in epochs}
    assert all(a is not None for a in audits.values())
    # It ran no step before its first epoch, so it cannot audit those two.
    assert [k for k, a in audits.items() if a["differences"] is None] == [FIRST, FIRST + 1]
    checked = {k: a for k, a in audits.items() if a["differences"] is not None}
    assert checked and all(a["differences"] == [] and a["own"] == a["root"]
                           for a in checked.values()), {
        k: (a["differences"][:3], a.get("why")) for k, a in checked.items() if a["differences"]}
    # Both packages' v4 fields (labels_v4, late_odds, odds_v4, head_odds,
    # picks_v4) and the state digests over them are the same bytes on the
    # root and the follower every epoch: the two reveals differ only in the
    # menu's publication, which only the root makes, and the follower's audit.
    root_only = {mp.REVEAL_SUMMARY, *mp.MENU_ROLE_FIELDS, "join_audit"}
    for k in epochs:
        root_r, follower_r = reveal("root", k), reveal("follower", k)
        assert {"labels_v4", "late_odds", "odds_v4", "head_odds", "picks_v4"} <= set(root_r)
        differ = {key for key in set(root_r) | set(follower_r)
                  if json.dumps(root_r.get(key), sort_keys=True)
                  != json.dumps(follower_r.get(key), sort_keys=True)}
        assert differ <= root_only, (k, sorted(differ - root_only))
        # A miner's "auto" reads the seat holder from the merged reveal's bytes
        # (`champion.champion`; none is filled in this run).
        from fugal_subnet.tee import td_v4
        raw = (dirs["root"] / "epochs" / epoch_id_for_block(k) / "reveal.json").read_bytes()
        assert td_v4.seat_holder_from_reveal(
            raw, epoch_id_for_block(k), hashlib.sha256(raw).hexdigest()) == root_r["champion"][
                "champion"] is None


def answering(k: int) -> set[str]:
    """The miners whose TD answers a challenge in epoch k."""
    return {SWITCHER} if k >= SWITCH else set()


# --- the pieces the loop above runs, each on its own ---------------------------------------------


def test_protocol_4_is_selected_only_by_the_off_finney_variable(monkeypatch):
    from fugal_subnet import fingerprint
    from fugal_subnet.benchmarks.slicer import protocol_version_for_boundary

    finney = config.FINNEY_GENESIS_HASH
    assert config.FINNEY_V4_ACTIVATION_BLOCK is None          # no release schedules v4
    monkeypatch.setattr(config, "V3_ACTIVATION_BLOCK_ENV", None)
    monkeypatch.setattr(config, "V4_ACTIVATION_BLOCK_ENV", None)
    unset = fingerprint.consensus_digest()
    assert config.v4_activation_block(finney) is None and config.protocol_for_chain(finney) == 3
    monkeypatch.setattr(config, "V4_ACTIVATION_BLOCK_ENV", 9_900_000)
    with pytest.raises(ValueError, match="FUGAL_V4_ACTIVATION_BLOCK"):
        config.v4_activation_block(finney)
    with pytest.raises(ValueError, match="FUGAL_V4_ACTIVATION_BLOCK"):
        config.v4_activation_block(None)                      # finney cannot be ruled out
    monkeypatch.setattr(config, "V3_ACTIVATION_BLOCK_ENV", 5_000)  # another chain's v3
    assert config.v4_activation_block(GENESIS) == 9_900_000
    assert config.protocol_for_chain(GENESIS) == 4
    assert fingerprint.consensus_digest() != unset           # a set variable shows
    monkeypatch.setattr(config, "V4_ACTIVATION_BLOCK_ENV", 1_000)   # before v3's block
    with pytest.raises(ValueError, match="before the v3"):
        config.v4_activation_block(GENESIS)
    assert protocol_version_for_boundary(500, 10, 50, v4_activation_block=500) == 4
    assert protocol_version_for_boundary(490, 10, 50, v4_activation_block=500) == 3
    assert protocol_version_for_boundary(490, 10, 50) == 3
    with pytest.raises(ValueError, match="v4 cannot activate before protocol v3"):
        protocol_version_for_boundary(500, 10, None, v4_activation_block=500)


def test_the_proposals_category_floor_is_pinned_at_half():
    assert config.PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4 == 5000
    assert "FUGAL_V4_ACTIVATION_BLOCK" in config.CONSENSUS_ENV_INPUTS
    assert "FUGAL_V4_PROPOSAL" in config.OPERATIONAL_ENV_VARS
    assert "FUGAL_V4_MEASURE_CAP_USD" not in config.OPERATIONAL_ENV_VARS    # no measurement


def test_a_v4_epoch_prices_at_its_menu_and_categorises_by_the_v4_mix():
    from fractions import Fraction

    from neurons import validator

    state = fx.state(fx.entry(price_in=123_456, price_out=7_000_001), epoch=30)
    t = validator._v4_tables(mp.MenuRead(30, state, "carried"))
    assert t.models == list(state.ids()) and t.model_index == {state.ids()[0]: 0}
    assert t.prices == {state.ids()[0]: (float(Fraction(123_456, ms.PICO)),
                                         float(Fraction(7_000_001, ms.PICO)))}
    mix = epoch_plan.mix_v4()
    assert t.bench_category == dict(mix.bench_category)
    assert t.category_targets == {c: v / 100 for c, v in mix.targets.items()}
    assert abs(sum(t.category_targets.values()) - 1) < 1e-9
    none = validator._v4_tables(mp.MenuRead(30, None, "unavailable", None, "no bytes"))
    assert none.prices == {} and none.models == []
    # The seat is on odds from the v4 block at the latest, and v1-v3 keep theirs.
    assert validator._odds_block_for(4, None, 1000) == 1000
    assert validator._odds_block_for(4, 900, 1000) == 900
    assert validator._odds_block_for(3, None, 1000) is None
    assert validator._odds_block_for(3, 900, None) == 900


def test_the_v4_slice_is_the_plans_and_the_v3_path_refuses_v4():
    from fugal_subnet.benchmarks.slicer import derive_nonce
    from neurons import validator

    pool = _pool()
    plan = epoch_plan.plan_v4(pool)
    gold = {q["question_id"]: q for q in pool}
    sp = validator._epoch_slice_v4("e00000040", "0xab", pool, gold, plan, config.SLICE_SIZE_V4)
    want = plan.scored_slice(derive_nonce("e00000040", "0xab"), pool, config.SLICE_SIZE_V4)
    assert sp.question_ids == [q["question_id"] for q in want]
    assert set(sp.slice_gold) == set(sp.question_ids) and sp.explore_map == {}
    with pytest.raises(ValueError, match="_epoch_slice_v4"):
        validator._epoch_slice(4, "e00000040", "0xab", pool, gold, ["a/x@high"], 300, 0)


def test_the_seats_pool_leaves_out_the_implausible_operators_and_is_the_menu_steps():
    """run_v2_epoch pools with the detectors' exclusions and the implausibility
    check's (an operator whose private answers are too good to be true), and
    exposes that pool (V2EpochResult.pooled), which the loop hands the v4 menu
    step (the spy in the loop test): the menu's counts and the seat's odds
    never pool differently. A pool built with the detectors' exclusions
    alone, as the menu step's was, keeps the implausible operator."""
    from fugal_subnet.explore_pool import ExploreAnswer
    from fugal_subnet.v2_epoch import OddsInput
    from tests import test_v2_epoch as t

    answers = []
    for i in range(8):
        for ck in ("ck-bad", "ck-a", "ck-b"):
            right = True if ck == "ck-bad" else i % 2 == 0
            answers.append(ExploreAnswer(f"x{i}-{ck}", "m/a", ck, right, failed=False,
                                         private=True, prompt_tokens=10, completion_tokens=20))
    oi = OddsInput(nonce=b"\x07" * 32, answers=answers, benchmark_of={}, heads={},
                   hidden_of=lambda q: None, implausible_z=1, implausible_min_samples=2,
                   implausible_min_trials=0)
    r = t.run(0, {1: t.oracle, 2: t.always("m/mid")}, odds_input=oi)
    assert list(r.reveal["exploration_excluded_coldkeys"]) == ["ck-bad"]
    def operators(pool):
        return {p.question_id.split("-", 1)[1] for p in pool}

    assert r.pooled and operators(r.pooled) == {"ck-a", "ck-b"}
    assert explore_pool.pairs_for_reveal(r.pooled, {}) == r.reveal["exploration_pool"]
    detectors_only = explore_pool.pool_pairs(answers, nonce=oi.nonce, coldkey_cap=oi.coldkey_cap,
                                             flagged=set())
    assert "ck-bad" in operators(detectors_only)


def test_the_miners_proposal_is_parsed_strictly_and_served_by_its_hash():
    from fugal_subnet import chain_proposals as cp
    from fugal_subnet.protocol import FugalProposalSynapse
    from neurons import miner

    assert miner._parse_v4_proposal(None) is None and miner._parse_v4_proposal(" ") is None
    spec = miner._parse_v4_proposal(json.dumps(PROPOSAL))
    assert spec.id == NEW_ID and spec.provider_order == ("host-a/fp8",)
    for bad in ("{", "[]", json.dumps({**PROPOSAL, "extra": 1}),
                json.dumps({**PROPOSAL, "provider_order": []}),
                json.dumps({**PROPOSAL, "reasoning": ["effort"]})):
        with pytest.raises(ValueError):
            miner._parse_v4_proposal(bad)
    doc = miner._proposal_document(spec, "5Fminer-0")
    assert cp.parse_document(doc).proposer == "5Fminer-0"
    asked = miner._serve_proposal(FugalProposalSynapse(proposal=cp.proposal_hash(doc)), doc)
    assert asked.document.encode() == doc
    other = miner._serve_proposal(FugalProposalSynapse(proposal="0" * 64), doc)
    assert other.document == ""                    # only the document whose hash it is asked for
    assert miner._proposal_document(None, "5Fminer-0") is None
    assert miner._pool_protocol(4) == 3 and miner._pool_protocol(3) == 3
    assert miner._pool_protocol(2) == 2


def test_a_v4_miner_without_a_root_hotkey_refuses_to_start(monkeypatch):
    import click

    from neurons import miner

    monkeypatch.setattr(config, "ROOT_HOTKEY_ENV", "")
    with pytest.raises(click.ClickException, match="FUGAL_ROOT_HOTKEY"):
        miner._v4_session(None, netuid=NETUID, pool=_pool(), first_epoch=10, blocks_per_epoch=BPE,
                          coldkey=None, genesis=GENESIS)


def test_the_miner_main_builds_the_session_and_hands_it_to_every_epoch():
    """The one part of the miner this file does not drive: main's serving
    loop (a bt.Axon server, a port bind, a reachability probe and a 30 s
    poll). It must build the session with _v4_session and pass it to
    _run_epoch; the loop test runs both of those functions as main calls
    them."""
    import ast
    import inspect

    from neurons import miner

    tree = ast.parse(inspect.getsource(miner.main.callback))
    calls = {getattr(n.func, "id", ""): n for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "_v4_session" in calls
    kw = {k.arg: ast.unparse(k.value) for k in calls["_run_epoch"].keywords}
    assert kw["v4_session"] == "v4_session" and kw["protocol_version"] == "protocol"
    # The session keeps its menu and ledger in the TD's state directory, and
    # takes the operator's anchor; the chain connection has the archive and
    # fallback endpoints its pinned reads need.
    built = {k.arg: ast.unparse(k.value) for k in calls["_v4_session"].keywords}
    assert built["state_dir"] == "_v4_state_dir(v4_state_dir)"
    assert built["anchor"] == "v4_anchor_state"
    assert "_chain_endpoints" in calls


def test_the_miners_v4_state_directory_and_anchor_file(tmp_path, monkeypatch):
    from neurons import miner

    monkeypatch.chdir(tmp_path)
    assert miner._v4_state_dir(None) == str(tmp_path / "results" / "v4")
    assert miner._v4_state_dir(str(tmp_path / "x")) == str(tmp_path / "x")
    assert miner._load_v4_anchor(None) is None
    good = tmp_path / "anchor.json"
    good.write_bytes(ms.canonical_json(fx.state()))
    assert miner._load_v4_anchor(str(good)) == fx.state()
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"{}")
    with pytest.raises(ValueError, match="not a menu"):
        miner._load_v4_anchor(str(bad))
    ends = miner._chain_endpoints("finney", None)
    assert ends["archive_endpoints"] and ends["fallback_endpoints"]
