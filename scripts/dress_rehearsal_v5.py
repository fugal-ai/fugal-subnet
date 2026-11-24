"""Scenario v5_flip of scripts/dress_rehearsal.py: protocol v5 across its
activation block, the shipped neurons against the real local chain.

    python scripts/dress_rehearsal.py --scenario v5_flip

The in-process rehearsal (scripts/rehearsal_v5.py) runs the v5 epoch on
synthetic inputs. This one runs the shipped validator and miner as real OS
processes against a real subtensor node, reusing scenario v4_flip's world
(scripts/dress_rehearsal_v4.py: the local public store, the model and
listing stub, the archive node, the fixture pool and seed, the signed list).

Timeline (S the epoch the scenario starts in, A = S + V4_EPOCHS_BEFORE the
first v5 epoch, epochs of --v4-epoch-seconds):

    S .. A-1   protocol v4 (v2, v3 and v4 blocks at S): a root (val1), a
               follower (val2), four miners (m1..m4) on the release image,
               exploration only.
    A, A+1     protocol v5, coverage only: no exam spec was published for
               them, the seat is empty and every share burns. The root's
               reveal of A publishes the spec of A+2 (exam_spec_v5).
    A+2 ..     the first exams: each miner calls exactly the cells the spec
               assigns its hotkey; both validators score them and agree.
    to A+END

What is checked: both validators run protocol 5 from A (their own epoch
lines); the root's reveals from A carry protocol_version 5, seat_v5 and
votes_v5; the reveal of A carries exam_spec_v5 for A+2 and the follower's
recomputed spec agrees (no "exam void" line); from A+2 the miners' proofs
carry exam votes; root and follower publish the same seat_v5 state every
v5 epoch; nothing raises.

Stand-ins: scenario v4_flip's, and the exam is drawn from v4's fixture pool
until the v5 exam pool ships (the validator logs a gap and the epoch's exam
is coverage only when the pool cannot supply questions; the check reports
it rather than failing the flip).
"""
from __future__ import annotations

import os
import sys
import time

S_NAME = "v5_flip"
V4_EPOCHS_BEFORE = 2              # A = S + 2
END = 4                           # through A+4: two exam epochs scored
MINERS = ("dr_m1", "dr_m2", "dr_m3", "dr_m4")
NAMES = ("v5f_val1", "v5f_val2")


def scenario_v5_flip(dr, report, subtensor, netuid, wallets, procs_out, args):
    import hashlib

    from fugal_subnet.benchmarks.slicer import blocks_per_epoch, epoch_id_for_block
    from scripts import dress_rehearsal_v4 as w4

    S = S_NAME
    results = dr.RESULTS
    print("\n[Scenario v5_flip] protocol v5 across its activation block", flush=True)
    bpe = blocks_per_epoch(int(dr.EPOCH_INTERVAL_S))
    archive = w4.start_archive_node(dr.CHAIN_NAME)
    report.check(S, "an archive node joins the local chain", archive,
                 w4.ARCHIVE_ENDPOINT if archive else "no archive node")

    start_idx = subtensor.get_current_block() // bpe
    A = start_idx + V4_EPOCHS_BEFORE
    fx = w4.write_fixtures(results, through=start_idx)
    seed_menu = w4.seed_menu_doc()
    stub = w4.V4Stub(fx["pool"], fx["hidden"], seed_menu)
    stub.start()
    store_dir = os.path.join(results, "store")
    os.makedirs(os.path.join(store_dir, w4.BUCKET), exist_ok=True)
    store = w4.LocalStore(store_dir)
    store.start()
    no_gcloud = os.path.join(results, "no-gcloud")
    os.makedirs(no_gcloud, exist_ok=True)
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                           if not os.path.exists(os.path.join(p, "gcloud")))

    root_hk = wallets["dr_val1"].hotkey.ss58_address
    first = str(start_idx * bpe)
    blocks = {"FUGAL_V2_ACTIVATION_BLOCK": first, "FUGAL_V3_ACTIVATION_BLOCK": first,
              "FUGAL_V4_ACTIVATION_BLOCK": first,
              "FUGAL_V5_ACTIVATION_BLOCK": str(A * bpe)}
    base = {**blocks,
            "PATH": path, "CLOUDSDK_CONFIG": no_gcloud,
            "FUGAL_OPENROUTER_BASE": f"http://127.0.0.1:{w4.STUB_PORT}/v1",
            "FUGAL_BENCHMARK_POOL_V3": fx["pool_path"], "FUGAL_EMBEDDINGS_FILE": fx["emb"],
            "FUGAL_ROOT_HOTKEY": root_hk, "FUGAL_REFERENCE_BUCKET": w4.BUCKET,
            "FUGAL_REHEARSAL_STORE_DIR": store_dir, "FUGAL_REHEARSAL_STORE_URL": store.url,
            "FUGAL_REHEARSAL_PUBLISHED_SEED": fx["seed_path"],
            **w4.write_signed_list(results),
            **({"FUGAL_ARCHIVE_ENDPOINTS": w4.ARCHIVE_ENDPOINT} if archive else {})}
    root_env = {"FUGAL_ROOT": "1", "FUGAL_REVEAL_BUCKET": w4.BUCKET, "FUGAL_REFERENCE_BUCKET": ""}
    follower_env: dict = {}
    print(f"    S = {epoch_id_for_block(start_idx)}, A = {epoch_id_for_block(A)} "
          f"(block {A * bpe}); pool {len(fx['pool'])} questions", flush=True)

    procs = w4.Procs(dr, netuid, base, results)
    coord = stub.coord
    heads = {"dr_m1": w4.v4_head(os.path.join(results, "head_v5_a.npz"), coord, w4.SHARP, "a"),
             "dr_m2": w4.v4_head(os.path.join(results, "head_v5_b.npz"), coord, w4.SHARPER, "b"),
             "dr_m3": w4.v4_head(os.path.join(results, "head_v5_l1.npz"), coord, None, "l1"),
             "dr_m4": w4.v4_head(os.path.join(results, "head_v5_l2.npz"), coord, None, "l2")}
    ports = {"dr_m1": 8281, "dr_m2": 8282, "dr_m3": 8283, "dr_m4": 8284}

    def miner_extra(ck):
        return {"FUGAL_V4_CHALLENGE": "0",
                "FUGAL_V4_STATE_DIR": os.path.join(results, f"td5_{ck}")}

    import bittensor as bt
    watch = w4.ChainWatch(bt.Subtensor(network=dr.ENDPOINT,
                                       archive_endpoints=[w4.ARCHIVE_ENDPOINT])
                          if archive else subtensor, netuid, bpe, plain=subtensor)
    t0 = time.time()
    try:
        procs.validator(NAMES[0], "dr_val1", root_env)
        procs.validator(NAMES[1], "dr_val2", follower_env)
        for ck in MINERS:
            procs.miner(f"v5f_{ck}", ck, heads[ck], ports[ck], miner_extra(ck))
        _drive(procs, watch, A, bpe, results, args)
    finally:
        procs.stop_all()
        stub.stop()
        store.stop()
    _checks(dr, report, S, results, A, w4, {ck: hashlib.sha256(open(p, "rb").read()).hexdigest()
                                            for ck, p in heads.items()}, time.time() - t0)


def _drive(procs, watch, A, bpe, results, args):
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block
    from scripts import dress_rehearsal_v4 as w4

    deadline = time.time() + (END + V4_EPOCHS_BEFORE + 2) * args.v4_epoch_seconds + 600
    last = None
    while time.time() < deadline:
        idx = watch.poll() // bpe
        logs = [w4.epoch_log(results, n) for n in NAMES]
        if idx != last:
            last = idx
            print(f"    {epoch_id_for_block(idx)} (A{idx - A:+d}): root logged "
                  f"{sorted(k - A for k in logs[0])[-3:]}, follower "
                  f"{sorted(k - A for k in logs[1])[-3:]}", flush=True)
        dead = procs.exited()
        if dead:
            raise RuntimeError(f"processes exited: {dead}")
        if all(A + END in lg for lg in logs):
            return
        time.sleep(2)


def _checks(dr, report, S, results, A, w4, head_hash, seconds):
    ran = {n: dr._epoch_protocols(os.path.join(results, f"{n}.log")) for n in NAMES}
    v5_epochs = range(A, A + END + 1)
    for n in NAMES:
        report.check(S, f"{n} runs protocol 5 from A and 4 before it",
                     all(ran[n].get(k) == 5 for k in v5_epochs)
                     and all(p == 4 for k, p in ran[n].items() if k < A),
                     str({k - A: p for k, p in sorted(ran[n].items())}))
    rv = {k: w4.reveal(results, NAMES[0], k) for k in v5_epochs}
    fr = {k: w4.reveal(results, NAMES[1], k) for k in v5_epochs}
    report.check(S, "the root's v5 reveals carry protocol_version 5, seat_v5 and votes_v5",
                 all(r and r.get("protocol_version") == 5 and "seat_v5" in r and "votes_v5" in r
                     for r in rv.values()),
                 str({k - A: (r or {}).get("protocol_version") for k, r in rv.items()}))
    burn = [(rv[k] or {}).get("seat_v5", {}).get("state", {}) for k in (A, A + 1)]
    report.check(S, "A and A+1: coverage only, the seat empty (every share burns)",
                 all(not (s or {}).get("holder") for s in burn), str(burn))
    spec = (rv[A] or {}).get("exam_spec_v5") or {}
    report.check(S, "the reveal of A publishes the exam spec of A+2",
                 spec.get("epoch") == A + 2 and bool(spec.get("spec_hash")),
                 f"epoch {spec.get('epoch')}, hash {str(spec.get('spec_hash'))[:16]}")
    void = [ln for ln in w4.log_text(results, NAMES[1]).splitlines() if "void" in ln.lower()]
    report.check(S, "the follower's own spec agrees with the root's (no void exam)",
                 not void, void[0][:200] if void else "")
    votes = sum(len(((rv[k] or {}).get("votes_v5") or {}).get("votes", ()))
                for k in range(A + 2, A + END + 1))
    gap = [ln for ln in w4.log_text(results, NAMES[0]).splitlines() if "no spec" in ln.lower()
           or "embedding gap" in ln.lower()]
    report.check(S, "from A+2 the miners' proofs carry exam votes",
                 votes > 0 or bool(gap),
                 f"{votes} exam votes" + (f"; pool gap logged: {gap[0][:160]}" if gap and not votes
                                          else ""))
    same = [k - A for k in v5_epochs
            if rv[k] and fr[k] and rv[k].get("seat_v5", {}).get("state")
            != fr[k].get("seat_v5", {}).get("state")]
    report.check(S, "root and follower publish the same seat every v5 epoch", not same,
                 f"differ at A+{same}" if same else "")
    tb = {n: w4._tracebacks(w4.log_text(results, n)) for n in (*NAMES, *(f"v5f_{c}" for c in MINERS))}
    bad = {n: t[0][:200] for n, t in tb.items() if t}
    report.check(S, "no process raised", not bad, str(bad) if bad else f"{seconds:.0f} s")
    print(f"    heads: { {ck: h[:12] for ck, h in head_hash.items()} }", file=sys.stderr)
