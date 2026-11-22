"""A root and two followers running the shipped v4 validator loop on the
tests/v4_world.py stand-ins, each follower handed every epoch's inputs in its
own seeded order, with seeded restarts (some skipping epochs, some with a
state file deleted while down), a root outage, a challenge that changes the
seat, a proposal that joins and becomes full, one that joins and is retired,
a successor that adds the new entry's row, and miners missing epochs.

After the run, per epoch: do the followers' reveals carry the root's v2
state, consensus view, seat evidence, memorisation and picks fields, menu,
and do their weights equal the root's? Disagreements inside the two epochs
after a follower re-anchors are reported apart ("recovery").

Run: uv run python -m tests.v4_multi_loop --seeds 0 1 2 --scratch <dir>
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import random
import shutil
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import pytest

from tests import v4_world as w

NAMES = ("root", "f1", "f2")
KEYS = {"root": w.ROOT, "f1": ("5Ff1-hotkey", "5Ff1-cold"), "f2": ("5Ff2-hotkey", "5Ff2-cold")}
SEAT_D, SEAT_G = "5Fseat-d", "5Fseat-g"
OTHER = {"model": "lab-c/model-other", "reasoning": ["effort", "high"],
         "provider_order": ["host-a/fp8"], "quantization": "fp8"}
OTHER_ID = "lab-c/model-other@high"


def base_of(thread_name: str) -> str:
    return thread_name.split("#")[0]


class Alarms(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.seen: list = []

    def emit(self, record):
        msg = record.getMessage()
        if "ALARM" in msg or record.levelno >= logging.ERROR:
            self.seen.append((threading.current_thread().name, msg[:300]))


class MultiLoop(w.SeatLoop):
    """SeatLoop with three validators, seeded input orders and restarts.
    `restarts` {name: [(stop_block, start_block, delete), ...]} where delete
    names the state files removed while the node is down."""

    def __init__(self, tmp_path, patch, miners, *, last, plan, signature, seed, restarts,
                 proposal_from=None):
        super().__init__(tmp_path, patch, miners, last=last, plan=plan, signature=signature,
                         proposal_from=proposal_from)
        self.seed, self.restarts = seed, restarts

    def run(self):
        import urllib.request

        import bittensor

        from fugal_subnet import collect, commit_reveal, config, epoch_logger, join_step
        from fugal_subnet import menu_publish as mp
        from fugal_subnet import menu_rules as mr
        from fugal_subnet import v2_epoch as v2e
        from fugal_subnet.tee import harness
        from fugal_subnet.tee import runtime as tee_runtime
        from neurons import miner as miner_mod
        from neurons import validator
        from tests.test_menu_publish import LIST_BYTES

        mpatch = self.monkeypatch
        neurons = [w.OWNER, *[KEYS[n] for n in NAMES],
                   *[(hk, ck) for hk, ck, *_ in self.miners_spec]]
        self.uid_of = {hk: i for i, (hk, _ck) in enumerate(neurons)}
        store = self.store = w.Store()
        chain = self.chain = w.Chain(store, neurons=neurons,
                                     validators={KEYS[n][0] for n in NAMES})
        mpatch.setattr(urllib.request, "urlopen", store.urlopen)
        mpatch.setattr(bittensor, "Subtensor", lambda *a, **k: w.Subtensor(chain))
        mpatch.setattr(w.Wallet, "hotkeys", {n: KEYS[n][0] for n in NAMES})
        mpatch.setattr(bittensor, "Wallet", w.Wallet)
        mpatch.setattr(collect, "CollectDendrite", w.Dendrite)
        uploader_cls, menu_store_cls = w.store_classes(store)
        mpatch.setattr(w.reveal_publisher, "GcsUploader", uploader_cls)
        mpatch.setattr(mp, "GcsStore", menu_store_cls)
        provider = self.provider = w.SeatProvider(self.hidden_of, self.coord)
        mpatch.setattr(harness, "_call_model", provider.call)
        mpatch.setattr(join_step, "fetch_public_listing",
                       lambda model, **_k: provider.fetch(model))
        mpatch.setattr(tee_runtime.MeteringProxy, "start", lambda self: None)
        mpatch.setattr(tee_runtime.MeteringProxy, "stop", lambda self: None)
        mpatch.setattr(config, "REFERENCE_BUCKET", w.BUCKET)
        mpatch.setattr(config, "ROOT_HOTKEY_ENV", w.ROOT[0])
        sha = hashlib.sha256(LIST_BYTES).hexdigest()
        store.objects[w.reveal_sync.list_object(w.NETUID, mr.KIND_MENU_ALLOWLIST, sha)] = LIST_BYTES
        store.objects[w.reveal_sync.signature_object(w.NETUID, mr.KIND_MENU_ALLOWLIST, sha)] = (
            self._signature)

        chain.block = 1
        hidden = miner_mod._load_pinned_embeddings(self.pool, mock=True)
        world = self.world = types.SimpleNamespace(miners={})
        self.seed_doc = w.fixture_seed(self.pool, 0, self.unpublished)
        w.install_seed(self.tmp_path, mpatch.setattr, self.seed_doc,
                       fresh_epochs=self.fresh_epochs)
        td_dirs = {hk: self.tmp_path / "td" / hk for hk, *_ in self.miners_spec}
        for hk, ck, head, mode, block in self.miners_spec:
            chain.block = block
            session = miner_mod._v4_session(
                w.Subtensor(chain), netuid=w.NETUID, pool=self.pool, first_epoch=w.SEAT_V4_EPOCH,
                blocks_per_epoch=w.BPE, coldkey=ck, genesis=w.GENESIS,
                fetch_listing=provider.fetch, anchor=self.base,
                challenge_size=w.challenge_size(mode), state_dir=td_dirs[hk])
            m = world.miners[hk] = w.Miner(hk, ck, head, session)
            chain.commit(hk, m.propose(w.PROPOSAL) if hk == self.proposal_from else m.fields())
        w.Dendrite.world = world
        self.served: dict = {}
        self.miner_errors: list = []

        def run_miners(epoch):
            if epoch > self.last:
                return
            self.plan(epoch, self)
            self.served[epoch] = set()
            from fugal_subnet.memo_sync import PublishedUnavailable
            for m in world.miners.values():
                if m.hotkey in self.down:
                    continue
                provider.serving = m.hotkey
                try:
                    miner_mod._run_epoch(
                        blocks_per_epoch=w.BPE, head_data=m.head, weights_hash=m.weights_hash,
                        pool=self.pool, hidden_states=hidden, epoch_index=epoch,
                        block_hash=chain.block_hash(epoch * w.BPE),
                        tee_runtime=tee_runtime.TEERuntime(mock=True), current_proof=m.current,
                        slice_size=config.SLICE_SIZE_V4, proxy_port=0, explore_models=[],
                        explore_size=0, hotkey=m.hotkey, protocol_version=4, prices=None,
                        block_now=lambda: chain.block, v4_session=m.session)
                except PublishedUnavailable as e:
                    self.miner_errors.append((epoch, m.hotkey, f"PublishedUnavailable: {e}"))
                    continue
                self.served[epoch].add(m.hotkey)
                proof = m.current["proof"]
                if proof is not None:
                    m.menus[epoch] = getattr(proof, "menu_hash", None)

        dirs = self.dirs = {n: self.tmp_path / n for n in NAMES}
        for d in dirs.values():
            (d / "epochs").mkdir(parents=True, exist_ok=True)
            mp.MenuBook(str(d / "state.json") + ".menu").hold(self.base)

        def ctx(n):
            d = dirs[base_of(n)]
            return [(validator, "STATE_PATH", str(d / "state.json")),
                    (commit_reveal, "EPOCH_DIR", str(d / "epochs")),
                    (epoch_logger, "EPOCH_LOG_DIR", str(d / "logs")),
                    (config, "ROOT_VALIDATOR", True if base_of(n) == "root" else None),
                    (config, "REVEAL_BUCKET", w.BUCKET if base_of(n) == "root" else "")]

        self.deleted: list = []

        def start(name, delete=()):
            base = base_of(name)

            def go():
                st = str(dirs[base] / "state.json")
                for what in delete:
                    path = {"published": st + ".published_v4.json",
                            "window": st + ".v2window.npz",
                            "replay": st + ".v4replay",
                            "state": st}[what]
                    p = Path(path)
                    if p.is_dir():
                        shutil.rmtree(p)
                        self.deleted.append((name, what, chain.block))
                    elif p.exists():
                        p.unlink()
                        self.deleted.append((name, what, chain.block))
                return validator.main.callback(
                    network="local", netuid=w.NETUID, coldkey=base, hotkey="default",
                    wallet_path=None, once=False, log_level="WARNING", live=False)
            return go

        contexts, stop_at, later = {}, {}, {}
        for n in NAMES:
            contexts[n] = ctx(n)
            prev = n
            for i, (stop_block, start_block, delete) in enumerate(self.restarts.get(n, [])):
                nxt = f"{n}#{i + 1}"
                contexts[nxt] = ctx(nxt)
                stop_at[prev] = stop_block
                later[nxt] = (start_block, start(nxt, delete))
                prev = nxt
        clock = self.clock = w.Lockstep(chain, contexts, run_miners,
                                        stop_block=(self.last + 1) * w.BPE + 1,
                                        stop_at=stop_at, later=later)
        mpatch.setattr(validator, "time", types.SimpleNamespace(
            sleep=clock.sleep, time=time.time, monotonic=time.monotonic))
        errors: list = []
        real_exception = validator.logger.exception

        def record(msg, *a, **k):
            errors.append((threading.current_thread().name, str(sys.exc_info()[1])[:300]))
            real_exception(msg, *a, **k)

        mpatch.setattr(validator.logger, "exception", record)
        real_run = v2e.run_v2_epoch
        seed = self.seed

        def shuffled_inputs(**kw):
            name = base_of(threading.current_thread().name)
            if name == "root":
                return real_run(**kw)
            rng = random.Random(f"{seed}|{name}|{int(kw['epoch_index'])}")

            def mix(d):
                items = list(d.items())
                rng.shuffle(items)
                return dict(items)

            for key in ("verified_proofs", "head_hashes", "coldkey_of", "hotkey_of", "records"):
                if key in kw and isinstance(kw[key], dict):
                    kw[key] = mix(kw[key])
            com = list(kw["commitments"])
            rng.shuffle(com)
            kw["commitments"] = com
            oi = kw.get("odds_input")
            if oi is not None:
                ans = list(oi.answers)
                rng.shuffle(ans)
                kw["odds_input"] = dataclasses.replace(oi, answers=ans, heads=mix(oi.heads))
            return real_run(**kw)

        mpatch.setattr(v2e, "run_v2_epoch", shuffled_inputs)
        alarms = Alarms()
        logging.getLogger().addHandler(alarms)
        try:
            chain.block = w.SEAT_FIRST * w.BPE
            run_miners(w.SEAT_FIRST)
            clock.start({n: start(n) for n in NAMES})
            clock.join(timeout=3600)
        finally:
            logging.getLogger().removeHandler(alarms)
        self.errors = errors
        self.alarms = alarms.seen
        self.epochs = range(w.SEAT_FIRST, self.last + 1)
        weights: dict = {}
        for block, hk, uids, ws in chain.weights:
            weights.setdefault(block // w.BPE, {})[hk] = dict(zip(uids, ws))
        self.weights = weights
        return self

    def reveal_of(self, k, name):
        from fugal_subnet.benchmarks.slicer import epoch_id_for_block

        p = self.dirs[name] / "epochs" / epoch_id_for_block(k) / "reveal.json"
        return json.loads(p.read_text()) if p.exists() else None


FIELDS = ("v2_state", "menu", "champion", "seat_v4", "memo_v4", "picks_v4", "labels_v4",
          "head_odds", "exploration_pool", "copy_verdicts")


def compare(loop: MultiLoop, recovery: dict) -> dict:
    from fugal_subnet import v2_sync

    out = {"compared": 0, "missing": [], "diff": [], "recovery_diff": [], "audits": []}
    for k in loop.epochs:
        root = loop.reveal_of(k, "root")
        if root is None:
            out["missing"].append((k, "root"))
            continue
        for n in ("f1", "f2"):
            r = loop.reveal_of(k, n)
            if r is None:
                out["missing"].append((k, n))
                continue
            out["compared"] += 1
            what = [f for f in FIELDS if root.get(f) != r.get(f)]
            if "v2_state" in what:
                what += [f"v2_state.{x}" for x in sorted(set(root["v2_state"]) | set(r["v2_state"]))
                         if root["v2_state"].get(x) != r["v2_state"].get(x)]
            if v2_sync.consensus_view(root) != v2_sync.consensus_view(r):
                what.append("consensus_view")
            wr = loop.weights.get(k, {}).get(w.ROOT[0])
            wf = loop.weights.get(k, {}).get(KEYS[n][0])
            if wr is not None and wf is not None and \
                    {u: x for u, x in wr.items() if x} != {u: x for u, x in wf.items() if x}:
                what.append("weights")
            for key in ("menu_audit", "join_audit"):
                a = r.get(key)
                if isinstance(a, dict) and a.get("differences"):
                    what.append(f"{key}:{str(a['differences'])[:200]}")
            if what:
                (out["recovery_diff"] if k in recovery.get(n, set()) else out["diff"]).append(
                    (k, n, what))
    # Held published sets and menus, byte for byte, at the end.
    tail = []
    for name in ("sources.json", "held.json"):
        rb = (loop.dirs["root"] / "state.json.menu" / name)
        for n in ("f1", "f2"):
            fb = (loop.dirs[n] / "state.json.menu" / name)
            if rb.exists() and fb.exists() and rb.read_bytes() != fb.read_bytes():
                tail.append(f"{n}:{name}")
    for n in NAMES:
        p = loop.dirs[n] / "state.json.published_v4.json"
        if not p.exists():
            tail.append(f"{n}:no published set file")
            continue
    if all((loop.dirs[n] / "state.json.published_v4.json").exists() for n in NAMES):
        docs = {n: json.loads((loop.dirs[n] / "state.json.published_v4.json").read_text())
                for n in NAMES}
        ats = {n: d["at"] for n, d in docs.items()}
        if len(set(ats.values())) == 1 and len({json.dumps(d["set"], sort_keys=True)
                                                for d in docs.values()}) != 1:
            tail.append(f"published sets differ at {ats}")
    out["end_state"] = tail
    return out


def schedule(seed: int, last_offset: int):
    rng = random.Random(seed)
    first = w.SEAT_FIRST
    last = first + last_offset
    sc = dict(last=last)
    sc["holder_quiet"] = first + rng.randint(3, 5)
    sc["switch"] = first + rng.randint(4, 8)
    sc["switch_size"] = rng.choice(["on", "1000"])
    sc["succeed"] = first + rng.randint(8, 12)
    sc["other_at"] = first + rng.randint(2, 9)
    sc["misses"] = {}
    for hk in (w.SEAT_A, w.SEAT_B, w.SEAT_E):
        if rng.random() < 0.6:
            sc["misses"].setdefault(first + rng.randint(2, last_offset - 1), []).append(hk)
    restarts, kinds = {}, {}
    # the root: one outage of 1-2 epochs
    if rng.random() < 0.8:
        e = first + rng.randint(3, last_offset - 5)
        stop = e * w.BPE + rng.randint(2, 8)
        gap = rng.randint(1, 2)
        back = (e + gap) * w.BPE + rng.randint(1, 5)
        restarts["root"] = [(stop, back, ())]
        kinds["root"] = [(e, f"outage{gap}")]
    for n in ("f1", "f2"):
        r, k = [], []
        e = first + rng.randint(0, 4)
        while e < last - 3 and len(r) < 3:
            delete = tuple(x for x in ("published", "window", "replay") if rng.random() < 0.3)
            if rng.random() < 0.4:
                stop = e * w.BPE + rng.randint(3, 6)
                back = min(stop + rng.randint(1, 3), (e + 1) * w.BPE - 1)
                kind = "no-miss"
            else:
                stop = e * w.BPE + rng.randint(2, 8)
                gap = rng.randint(1, 3)
                back = (e + gap) * w.BPE + rng.randint(1, 5)
                kind = f"miss{gap}"
            r.append((stop, back, delete))
            k.append((e, kind, delete))
            e = back // w.BPE + rng.randint(2, 5)
        restarts[n], kinds[n] = r, k
    sc["restarts"], sc["kinds"] = restarts, kinds
    return sc


def recovery_epochs(sc) -> dict:
    """Epochs a node may still be catching up in: the epoch it comes back in
    and the two after."""
    out: dict = {}
    for n, rs in sc["restarts"].items():
        for stop, back, _d in rs:
            e = back // w.BPE
            for k in range(stop // w.BPE, e + 3):
                out.setdefault(n, set()).add(k)
    # A root outage: every follower may disagree while the root is away.
    for n in ("f1", "f2"):
        out.setdefault(n, set()).update(out.get("root", set()))
    return out


def run_seed(seed: int, tmp: Path, last_offset: int, signature, patch, override=None):
    sc = schedule(seed, last_offset)
    if override is not None:
        sc = override(sc)
    spec: list = []
    loop = MultiLoop(tmp, patch, spec, last=sc["last"], plan=lambda e, lp: None,
                     signature=signature, seed=seed, restarts=sc["restarts"],
                     proposal_from=w.SEAT_A)
    a2 = loop.head(loop.all_rows(w.SHARP, new=True), "a")
    spec += [
        (w.SEAT_A, "5Fcold-a", loop.head(loop.all_rows(w.SHARP), "a"), "on", 1),
        (w.SEAT_B, "5Fcold-b", loop.head(loop.all_rows(w.SHARPER), "b"), "off",
         w.SEAT_V4_EPOCH * w.BPE + 3),
        (SEAT_D, "5Fcold-d", loop.head(loop.all_rows(w.SHARPER), "d", noise=1e-4), "off",
         w.SEAT_V4_EPOCH * w.BPE + 7),
        (w.SEAT_E, "5Fcold-e", loop.head({w.NEW_ID: w.SHARPER, loop.base.ids()[0]: 0.0}, "e"),
         "off", w.SEAT_V4_EPOCH * w.BPE + 5),
        (SEAT_G, "5Fcold-g", loop.head({loop.base.ids()[1]: 0.0}, "g"), "off",
         w.SEAT_V4_EPOCH * w.BPE + 8),
        (w.SEAT_L, "5Fcold-l", loop.head({loop.base.ids()[0]: 0.0}, "l"), "off",
         w.SEAT_V4_EPOCH * w.BPE + 9),
    ]

    def plan(e, lp):
        if e == sc["holder_quiet"]:
            lp.scored(w.SEAT_A, "off")
        if e == sc["switch"]:
            lp.scored(w.SEAT_B, sc["switch_size"])
            lp.scored(SEAT_D, "on")
        if e == sc["switch"] + 2:
            lp.scored(w.SEAT_B, "off")
        if e == sc["succeed"]:
            lp.set_head(w.SEAT_A, a2)
        if e == sc["other_at"]:
            m = lp.world.miners[SEAT_G]
            lp.chain.commit(SEAT_G, m.propose(OTHER))
        lp.down = set(sc["misses"].get(e, []))

    loop.plan = plan
    t0 = time.time()
    loop.run()
    res = compare(loop, recovery_epochs(sc))
    holders = {k: (loop.reveal_of(k, "root") or {}).get("champion", {}).get("champion")
               for k in loop.epochs}
    names = {hashlib.sha256(m.head).hexdigest(): hk[-1] for hk, m in loop.world.miners.items()}
    for hk, _ck, head, *_ in spec:
        names[hashlib.sha256(head).hexdigest()] = hk[-1]
    names[hashlib.sha256(a2).hexdigest()] = "a2"
    changes = [(k, names.get(h, h and h[:8])) for k, h in holders.items()
               if h != holders.get(k - 1)]
    from fugal_subnet import menu_publish as mp
    book = mp.MenuBook(str(loop.dirs["root"] / "state.json") + ".menu")
    life = {}
    for k in loop.epochs:
        m = book.held(k)
        if m is None:
            continue
        for eid in (w.NEW_ID, OTHER_ID):
            st = "absent"
            if eid in m.ids():
                st = "prov" if m.entry(eid).provisional else "full"
            life.setdefault(eid, []).append(st[0])
    # Every TD ran the menu the root held for the epoch.
    from fugal_subnet import menu_state as ms
    td_menu_split = []
    for k in loop.epochs:
        m = book.held(k)
        want = ms.menu_hash(m) if m is not None else None
        got = {hk[-1]: x.menus.get(k) for hk, x in loop.world.miners.items() if k in x.menus}
        bad = {hk: h for hk, h in got.items() if h != want}
        if bad:
            td_menu_split.append((k, (want or "")[:10], {hk: (h or "")[:10] for hk, h in bad.items()}))
    res["td_menu_split"] = td_menu_split
    return dict(seed=seed, schedule=sc, seconds=round(time.time() - t0),
                errors=loop.errors[:8], n_errors=len(loop.errors), alarms=[a for a in loop.alarms if "NO reference" not in a[1]][:40],
                n_alarms=len(loop.alarms), clock_errors={k: repr(v)[:300] for k, v in
                                                         loop.clock.errors.items()},
                future_reads=loop.chain.future_reads[:5], miner_errors=loop.miner_errors[:8],
                deleted=loop.deleted, holder_changes=changes,
                lifecycle={k: "".join(v) for k, v in life.items()}, **res)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--last-offset", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scratch", default=None)
    args = ap.parse_args(argv)
    from fugal_subnet import menu_rules as mr
    from tests.signed_list_fixtures import Signer
    from tests.test_menu_publish import LIST_BYTES

    logging.basicConfig(level=logging.CRITICAL)
    results = []
    for seed in args.seeds:
        patch = pytest.MonkeyPatch()
        try:
            tmp = Path(tempfile.mkdtemp(prefix=f"loop-{seed}-", dir=args.scratch))
            s = Signer(tmp, "release")
            patch.setattr(mr, "list_signers", lambda s=s: s.signers_line().encode())
            (tmp / "w").mkdir()
            try:
                res = run_seed(seed, tmp / "w", args.last_offset, s.sign(LIST_BYTES), patch)
            except BaseException as e:  # noqa: BLE001 - reported
                import traceback
                res = dict(seed=seed, crash=repr(e)[:500], tb=traceback.format_exc()[-2000:])
        finally:
            patch.undo()
        results.append(res)
        print(json.dumps(res, default=str))
        sys.stdout.flush()
    if args.out:
        Path(args.out).write_text(json.dumps(results, default=str, indent=1))
    bad = [r["seed"] for r in results if r.get("crash") or r.get("diff") or r.get("errors")
           or r.get("clock_errors") or r.get("future_reads") or r.get("end_state")
           or r.get("td_menu_split")]
    print(f"seeds {len(results)}; epochs per seed {args.last_offset + 1}; flagged: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
