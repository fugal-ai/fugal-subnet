"""Scenario v4_seat_menu of scripts/dress_rehearsal.py: the fast crown and the
chain proposals of protocol v4 on the real local chain, with the shipped
neurons (docs/decisions/FAST_CROWN_V4.md; docs/INVARIANTS.md, "the fast
crown" and "the menu lifecycle"):

    python scripts/dress_rehearsal.py --scenario v4_seat_menu

It reuses scripts/dress_rehearsal_v4.py's world (the archive node, the local
store, the stub, the fixture pool and seed, the signed list, the shim) and
starts at v4 two epochs in (F = S + 2: no v3 miner; the flip itself is
scenario v4_flip's). A root (val1) and a follower (val2) run throughout.

The miners (heads predict the stub's outcomes by the fixture embedding's
coordinate per entry, tests/v4_world.SeatProvider's rule):
    A (m1)  sharp; ONE 1,000-question challenge (--v4-challenge 1000) until
            it fills the empty seat, then restarted with 0: it holds on its
            exploration alone.
    B (m2)  sharper; restarted with 1000 for the third epoch after the fill
            (the holder its TD pairs with is in the root's reveal two epochs
            back), takes the seat, restarted with 0. Later it files P3 with
            its commitment timed to land inside the root's margin before a
            boundary: it waits one epoch.
    C (m3)  weaker than A (no edge); challenges in the same epoch as B and
            does not take. Then files P2, a model whose host fails every
            call; the root is late (SIGSTOP across the boundary) for the
            epoch P2 would join, so it joins one later, and is retired by
            the step two epochs after that.
    L (m4)  a labeller; files P1 (a model that answers right) with its
            first head commitment: it joins provisional at the first
            boundary the rules allow and is full two epochs later.

Every check is a row; the seat rows are `seat_rows`, the proposal rows
`proposal_rows`, the root's timing `margin_rows`.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import time

from scripts import dress_rehearsal_v4 as w4

S_NAME = "v4_seat_menu"
V3_EPOCHS_BEFORE = 2
MAX_EPOCHS_AFTER_F = 12
PORTS = {"dr_m1": 8281, "dr_m2": 8282, "dr_m3": 8283, "dr_m4": 8284}
C_SHARP = 20.0
PROPOSALS = {
    "P1": {"model": "anthropic/claude-rehearsal-good", "reasoning": ["effort", "high"],
           "provider_order": ["anthropic"], "quantization": "unknown"},
    "P2": {"model": "deepseek/deepseek-rehearsal-flaky", "reasoning": ["effort", "low"],
           "provider_order": ["deepinfra/fp8"], "quantization": "fp8"},
    "P3": {"model": "openai/gpt-rehearsal-late", "reasoning": ["effort", "high"],
           "provider_order": ["openai"], "quantization": "unknown"},
}
STUB_MODELS = {
    PROPOSALS["P1"]["model"]: {"hosts": ["anthropic"], "quantization": "unknown",
                               "price_in": 1_000_000, "price_out": 5_000_000, "answers": "right"},
    PROPOSALS["P2"]["model"]: {"hosts": ["deepinfra/fp8"], "quantization": "fp8",
                               "price_in": 200_000, "price_out": 800_000, "answers": "fail"},
    PROPOSALS["P3"]["model"]: {"hosts": ["openai"], "quantization": "unknown",
                               "price_in": 2_000_000, "price_out": 8_000_000, "answers": "right"},
}


def entry_id(p: dict) -> str:
    """The menu id a proposal joins under (<model>@<effort>)."""
    from fugal_subnet import chain_proposals as cp
    from neurons import miner

    raw = miner._proposal_document(miner._parse_v4_proposal(json.dumps(p)), "5" + "A" * 47)
    return cp.parse_document(raw).id


class FieldWatch:
    """Every commitment change on chain, as the harness sees it by polling
    (each hotkey's fields and the block its commitment landed at), and
    (time, block) samples to place log lines on the chain's clock."""

    def __init__(self, subtensor, netuid: int):
        self.st, self.netuid = subtensor, netuid
        self.changes: dict[str, list[tuple[int, tuple[str, ...]]]] = {}
        self.samples: list[tuple[float, int]] = []
        self.used: dict[str, list] = {}

    def poll(self, block: int) -> None:
        self.samples.append((time.time(), block))
        try:
            for hk, value in self.st.query_map(module="Commitments", name="CommitmentOf",
                                               params=[self.netuid]):
                hk = str(getattr(hk, "value", hk))
                raw = getattr(value, "value", value)
                from fugal_subnet import chain_proposals as cp
                fields = cp.commitment_fields(raw)
                at = int((raw or {}).get("block", 0)) if isinstance(raw, dict) else 0
                rows = self.changes.setdefault(hk, [])
                if not rows or rows[-1] != (at, fields):
                    rows.append((at, fields))
        except Exception:  # noqa: BLE001 - the next poll reads again
            pass

    def block_at(self, t: float) -> int | None:
        before = [b for ts, b in self.samples if ts <= t]
        return before[-1] if before else None

    def used_space(self, hotkey: str):
        try:
            got = self.st.substrate.query("Commitments", "UsedSpaceOf", [self.netuid, hotkey])
            return getattr(got, "value", got)
        except Exception as e:  # noqa: BLE001
            return f"? ({type(e).__name__})"


def scenario_v4_seat_menu(dr, report, subtensor, netuid, wallets, procs_out, args):
    import bittensor as bt

    from fugal_subnet.benchmarks.slicer import blocks_per_epoch, epoch_id_for_block

    S = S_NAME
    results = dr.RESULTS
    print("\n[Scenario v4_seat_menu] the fast crown and chain proposals", flush=True)
    bpe = blocks_per_epoch(int(dr.EPOCH_INTERVAL_S))
    t0 = time.time()
    archive = w4.start_archive_node(dr.CHAIN_NAME)
    report.check(S, "an archive node joins the local chain", archive, w4.ARCHIVE_ENDPOINT)
    start_idx = subtensor.get_current_block() // bpe
    F = start_idx + V3_EPOCHS_BEFORE
    fx = w4.write_fixtures(results, through=start_idx)
    seed_menu = w4.seed_menu_doc()
    stub = w4.V4Stub(fx["pool"], fx["hidden"], seed_menu, extra_models=STUB_MODELS)
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
    v4 = {"FUGAL_V2_ACTIVATION_BLOCK": str(start_idx * bpe),
          "FUGAL_V3_ACTIVATION_BLOCK": str(start_idx * bpe),
          "FUGAL_V4_ACTIVATION_BLOCK": str(F * bpe)}
    base = {**v4, "PATH": path, "CLOUDSDK_CONFIG": no_gcloud,
            "FUGAL_OPENROUTER_BASE": f"http://127.0.0.1:{w4.STUB_PORT}/v1",
            "FUGAL_REHEARSAL_LISTING_URL":
                f"http://127.0.0.1:{w4.STUB_PORT}/v1/models/{{model}}/endpoints",
            "FUGAL_BENCHMARK_POOL_V3": fx["pool_path"], "FUGAL_EMBEDDINGS_FILE": fx["emb"],
            "FUGAL_ROOT_HOTKEY": root_hk, "FUGAL_REFERENCE_BUCKET": w4.BUCKET,
            "FUGAL_REHEARSAL_STORE_DIR": store_dir, "FUGAL_REHEARSAL_STORE_URL": store.url,
            "FUGAL_REHEARSAL_PUBLISHED_SEED": fx["seed_path"],
            **w4.write_signed_list(results),
            **({"FUGAL_ARCHIVE_ENDPOINTS": w4.ARCHIVE_ENDPOINT} if archive else {})}
    root_env = {"FUGAL_ROOT": "1", "FUGAL_REVEAL_BUCKET": w4.BUCKET, "FUGAL_REFERENCE_BUCKET": ""}
    procs = w4.Procs(dr, netuid, base, results)
    coord = stub.coord
    heads = {"dr_m1": w4.v4_head(os.path.join(results, "head_a.npz"), coord, w4.SHARP, "a"),
             "dr_m2": w4.v4_head(os.path.join(results, "head_b.npz"), coord, w4.SHARPER, "b"),
             "dr_m3": w4.v4_head(os.path.join(results, "head_c.npz"), coord, C_SHARP, "c"),
             "dr_m4": w4.v4_head(os.path.join(results, "head_l.npz"), coord, None, "l")}
    head_hash = {ck: hashlib.sha256(open(p, "rb").read()).hexdigest() for ck, p in heads.items()}
    opts = {"dr_m1": {"FUGAL_V4_CHALLENGE": "1000"}, "dr_m2": {}, "dr_m3": {},
            "dr_m4": {"FUGAL_V4_PROPOSAL": json.dumps(PROPOSALS["P1"])}}

    def start_miner(ck):
        procs.miner(f"vs_{ck}", ck, heads[ck], PORTS[ck],
                    {"FUGAL_V4_CHALLENGE": "0", **opts[ck],
                     "FUGAL_V4_STATE_DIR": os.path.join(results, f"td_{ck}")})

    def restart(ck, **o):
        procs.stop(f"vs_{ck}")
        opts[ck] = o
        start_miner(ck)

    watch = w4.ChainWatch(bt.Subtensor(network=dr.ENDPOINT, archive_endpoints=[w4.ARCHIVE_ENDPOINT])
                          if archive else subtensor, netuid, bpe, plain=subtensor)
    fields = FieldWatch(subtensor, netuid)
    _HK.update({ck: w.hotkey.ss58_address for ck, w in wallets.items()})
    print(f"    S = {epoch_id_for_block(start_idx)}, F = {epoch_id_for_block(F)}; {bpe} blocks an "
          f"epoch; join margin {_margin(bpe)} blocks, commit slack {_slack(bpe)}", flush=True)
    events: dict = {}
    try:
        procs.validator("vs_val1", "dr_val1", root_env)
        procs.validator("vs_val2", "dr_val2", {})
        for ck in w4.MINERS:
            start_miner(ck)
        events["setup_s"] = time.time() - t0
        _drive(procs, watch, fields, results, F, bpe, events, restart, args, dr)
    except Exception as e:  # noqa: BLE001 - a row, and the checks below still run
        report.check(S, "scenario v4_seat_menu ran its schedule", False, f"{type(e).__name__}: {e}")
    finally:
        _cont(procs, "vs_val1")
        procs.stop_all()
        stub.stop()
        store.stop()
        events["total_s"] = time.time() - t0
    _checks(report, S, results, watch, fields, store, stub, F, bpe, fx, events, head_hash, wallets,
            netuid, subtensor, root_hk)


def _margin(bpe):
    from fugal_subnet import chain_proposals as cp
    return cp.join_margin(bpe)


def _slack(bpe):
    from fugal_subnet import chain_proposals as cp
    return cp.commit_slack(bpe)


def _cont(procs, name):
    p = procs.running.get(name)
    if p is not None:
        try:
            p[0].send_signal(signal.SIGCONT)
        except OSError:
            pass


def _seats(results, F, last):
    return {k: ((w4.reveal(results, "vs_val1", k) or {}).get("champion") or {})
            for k in range(F, last + 1)}


def _drive(procs, watch, fields, results, F, bpe, events, restart, args, dr):
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block

    deadline = time.time() + (MAX_EPOCHS_AFTER_F + V3_EPOCHS_BEFORE + 2) * args.v4_epoch_seconds
    last_print = None
    margin = _margin(bpe)
    while time.time() < deadline:
        block = watch.poll()
        fields.poll(block)
        idx = block // bpe
        root, fol = w4.epoch_log(results, "vs_val1"), w4.epoch_log(results, "vs_val2")
        last = max(list(root) + list(fol) + [F])
        seats = _seats(results, F, last)
        if idx != last_print:
            last_print = idx
            print(f"    {epoch_id_for_block(idx)} (F{idx - F:+d}): root {sorted(k - F for k in root)[-3:]}"
                  f", follower {sorted(k - F for k in fol)[-3:]}; seat "
                  f"{[(k - F, s.get('event')) for k, s in seats.items() if s][-3:]}", flush=True)
        dead = procs.exited()
        if dead:
            raise RuntimeError(f"processes exited: {dead}")
        fill = next((k for k, s in seats.items() if s.get("event") == "fill"), None)
        take = next((k for k, s in seats.items() if s.get("event") == "take"), None)
        # A: one challenge, then exploration only, once its fill is published.
        if fill is not None and "a_quiet" not in events:
            restart("dr_m1")
            events["a_quiet"] = (block, idx)
        # B and C challenge in fill + 3 (the holder is in the root's reveal
        # of fill + 1; A holds on exploration alone in fill + 1 and fill +
        # 2 first): restarted during fill + 2, after its collection.
        if (fill is not None and "bc_challenge" not in events and idx == fill + 2
                and fill + 2 in fol):
            restart("dr_m2", FUGAL_V4_CHALLENGE="1000")
            restart("dr_m3", FUGAL_V4_CHALLENGE="1000")
            events["bc_challenge"] = (block, idx)
        # After the take: B and C quiet; C files P2 at once.
        if take is not None and "p2" not in events:
            restart("dr_m2")
            t = time.time()
            restart("dr_m3", FUGAL_V4_PROPOSAL=json.dumps(PROPOSALS["P2"]))
            events["p2"] = (block, idx, t)
        # P2's commitment lands: the boundary it would join at, by the rule,
        # is the first whose read block is at or after it; the root is
        # paused across that boundary's read and commit.
        if "p2" in events and "p2_commit" not in events:
            c = _proposal_commit(fields, procs, "dr_m3")
            if c is not None:
                events["p2_commit"] = c
                late = c // bpe + 1 if c <= (c // bpe + 1) * bpe - margin else c // bpe + 2
                events["late_epoch"] = late
                events["latency_blocks"] = c - events["p2"][0]
        late = events.get("late_epoch")
        if late is not None and "root_paused" not in events and block >= late * bpe - margin - 4:
            procs.running["vs_val1"][0].send_signal(signal.SIGSTOP)
            events["root_paused"] = block
        if "root_paused" in events and "root_resumed" not in events and block >= late * bpe + 4:
            procs.running["vs_val1"][0].send_signal(signal.SIGCONT)
            events["root_resumed"] = block
        # P3: B's commitment aimed inside the margin before the boundary
        # after next (late + 2, clear of P2's join at late + 1).
        if late is not None and "p3" not in events and "latency_blocks" in events:
            target = (late + 2) * bpe - margin // 2
            if block >= target - events["latency_blocks"]:
                restart("dr_m2", FUGAL_V4_PROPOSAL=json.dumps(PROPOSALS["P3"]))
                events["p3"] = (block, idx)
        if "p3" in events and "p3_commit" not in events:
            c = _proposal_commit(fields, procs, "dr_m2")
            if c is not None:
                events["p3_commit"] = c
        # The end: P2 decided (late + 1 + 2) and P3 joined (late + 3), both
        # logged by both validators, or the epoch budget.
        if late is not None and all(late + 3 in x for x in (root, fol)):
            break
        if idx > F + MAX_EPOCHS_AFTER_F:
            break
        time.sleep(2)
    events["end"] = watch.st.get_current_block()


_HK: dict[str, str] = {}


def _proposal_commit(fields, procs, ck) -> int | None:
    """The block at which `ck`'s commitment first carried a proposal field."""
    from fugal_subnet import chain_proposals as cp

    for at, f in fields.changes.get(_HK.get(ck, ""), []):
        if len(f) > 1 and cp.parse_proposal_field(f[1]):
            return at
    return None


# ── checks ────────────────────────────────────────────────────────────────────

def _menu(results, store, name, k):
    """The menu state reveal k of validator `name` ran on, rebuilt as any
    reader would from its `menu` field and the store (menu_from_field)."""
    from fugal_subnet import menu_publish as mp
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block

    r = w4.reveal(results, name, k)
    if not r or "menu" not in r:
        return None
    objs = store.objects(w4.BUCKET)
    netuid = next((key.split("/")[0] for key in objs if "/menus/" in key), "0")
    try:
        return mp.menu_from_field(r["menu"], epoch_id_for_block(k),
                                  lambda h: objs.get(f"{netuid}/menus/{h}.json"))
    except ValueError as e:
        return f"menu not rebuilt: {e}"


def _checks(report, S, results, watch, fields, store, stub, F, bpe, fx, events, head_hash, wallets,
            netuid, subtensor, root_hk):
    names = ("vs_val1", "vs_val2")
    logs = {n: w4.epoch_log(results, n) for n in names}
    last = max([k for n in names for k in logs[n]] or [F])
    print(f"    events: {events}", flush=True)
    seat_rows(report, S, results, store, F, last, logs, head_hash, wallets, netuid, fx)
    proposal_rows(report, S, results, store, watch, fields, stub, F, last, logs, events, wallets,
                  netuid, bpe, root_hk)
    margin_rows(report, S, results, fields, F, last, bpe, events, root_hk, wallets, netuid)
    flaky = PROPOSALS["P2"]["model"]
    # The stub answers 500 to P2's model and nothing else: the TD's proxy
    # and harness log each such call as handled (with its traceback).
    w4.logs_row(report, S, results, expected=lambda block: (
        ("HTTP Error 500" in block or "HTTP Error 502" in block)
        and ("Proxy upstream error" in block or f"Model call failed for {flaky}" in block)))
    durs = {n: w4._after_collection(w4.log_text(results, n), F) for n in names}
    report.check(S, "each validator's v4 step after collection fits in the epoch",
                 all(durs[n] for n in names) and max(max(v) for v in durs.values()) < bpe / 3,
                 f"seconds after the collection block {durs}; setup {events.get('setup_s', 0):.0f}s,"
                 f" scenario {events.get('total_s', 0):.0f}s")


def seat_rows(report, S, results, store, F, last, logs, head_hash, wallets, netuid, fx):
    """The fast crown on the real chain (FAST_CROWN_V4.md 2, 6). Update here
    when the seat's rules change."""
    import numpy as np

    from fugal_subnet import picks_v4
    from fugal_subnet.head_eval import load_head_v4
    from fugal_subnet.head_odds import picks_and_odds_v4
    from fugal_subnet.head_publisher import head_key

    seats = _seats(results, F, last)
    a, b, c = head_hash["dr_m1"], head_hash["dr_m2"], head_hash["dr_m3"]
    fill = next((k for k, s in seats.items() if s.get("event") == "fill"), None)
    take = next((k for k, s in seats.items() if s.get("event") == "take"), None)

    def picks(k):
        return (w4.reveal(results, "vs_val1", k) or {}).get("picks_v4") or {}

    ev_fill = (picks(fill).get("evidence") or {}).get(a, {}) if fill else {}
    report.check(S, "seat: one 1,000-question challenge fills the empty seat",
                 fill is not None and seats[fill].get("champion") == a
                 and ev_fill.get("questions") == 1000,
                 f"fill at F+{fill - F if fill is not None else None} by "
                 f"{str(seats.get(fill, {}).get('champion'))[:12]} (A {a[:12]}); A's evidence there "
                 f"{ {x: ev_fill.get(x) for x in ('answered', 'questions')} }")
    # The holder explores only, and stands and is paid, until the take.
    quiet = list(range(fill + 1, take or last + 1)) if fill is not None else []
    paid = {}
    for k in quiet:
        ws = {int(u): float(x) for u, x in (logs["vs_val1"].get(k, {}).get("weights") or {}).items()}
        paid[k - F] = round(max(ws.values()) if ws else 0.0, 3)
    a_items = [k - F for k in quiet if a in (picks(k).get("items") or {})]
    report.check(S, "seat: the holder, exploring only, still holds and is paid",
                 bool(quiet) and all(seats[k].get("champion") == a for k in quiet)
                 and not a_items and all(v > 0.5 for v in paid.values()),
                 f"epochs F+{[k - F for k in quiet]}: holder A throughout, A challenged in "
                 f"{a_items}; top weight {paid}")
    ev_take = picks(take).get("evidence") or {} if take else {}
    report.check(S, "seat: a second miner's single challenge takes the seat from it",
                 take is not None and seats[take].get("champion") == b
                 and (ev_take.get(b) or {}).get("questions") == 1000,
                 f"take at F+{take - F if take is not None else None}; B's evidence "
                 f"{ {x: (ev_take.get(b) or {}).get(x) for x in ('answered', 'questions')} }")
    cev = ev_take.get(c) or {}
    report.check(S, "seat: a challenge by a miner with no edge does not take it",
                 take is not None and cev.get("answered") is True and cev.get("questions") == 1000
                 and seats[take].get("champion") != c,
                 f"C's evidence in F+{take - F if take is not None else None}: "
                 f"{ {x: cev.get(x) for x in ('answered', 'questions', 'qualifies')} }")
    b_log = w4.log_text(results, "vs_dr_m2")
    bad = [ln for ln in b_log.splitlines() if "holder's head" in ln and "no challenge" in ln]
    heads_got = [p for p in store.gets if p.endswith(f"{a}.npz")]
    report.check(S, "seat: the challenger's TD obtained and checked the holder's head",
                 take is not None and (picks(take).get("evidence") or {}).get(b, {}).get(
                     "answered") is True and not bad and bool(heads_got),
                 f"B's TD warnings about the holder's head: {bad[:1]}; GETs of A's head from the "
                 f"store: {len(heads_got)}")
    agree = [k - F for k in range(F, last + 1)
             if w4.reveal(results, "vs_val1", k) and w4.reveal(results, "vs_val2", k)
             and picks(k) == ((w4.reveal(results, "vs_val2", k) or {}).get("picks_v4") or {})]
    both = [k - F for k in range(F, last + 1)
            if w4.reveal(results, "vs_val1", k) and w4.reveal(results, "vs_val2", k)]
    report.check(S, "seat: both validators recompute the picks and publish the same picks_v4",
                 bool(both) and agree == both, f"same at F+{agree} of F+{both}")
    # An outside reader: the holder's picks from the published head bytes,
    # the menu the reveal names and the challenge list.
    objs = store.objects(w4.BUCKET)
    row = {q["question_id"]: i for i, q in enumerate(fx["pool"])}
    checked, wrong, missing = 0, [], []
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block

    for k in range(F, last + 1):
        r = w4.reveal(results, "vs_val1", k)
        # A reveal, and the menu source it names, are published one epoch
        # after it (the weight commit-reveal hold): the newest is not out yet.
        if not r or f"{netuid}/{epoch_id_for_block(k)}/reveal.json" not in objs:
            continue
        hp = (r.get("picks_v4") or {}).get("holder_picks")
        holder = (r.get("champion") or {}).get("champion")
        if holder is None:
            if hp not in ({}, None):
                wrong.append((k - F, "picks with no holder"))
            continue
        raw = objs.get(head_key(netuid, holder))
        state = _menu(results, store, "vs_val1", k)
        if raw is None or not hasattr(state, "entries"):
            missing.append((k - F, "head" if raw is None else str(state)[:60]))
            continue
        qids = [q for q, _c in r["picks_v4"]["list"]]
        got = picks_and_odds_v4(load_head_v4(raw), state,
                                np.asarray(fx["hidden"])[[row[q] for q in qids]]).picks
        want = picks_v4.holder_picks_doc(holder, got, r["models"])
        checked += 1
        if want != hp:
            wrong.append((k - F, "differs"))
    report.check(S, "seat: every reveal with a holder carries holder_picks, and an outside reader "
                    "reproduces them",
                 checked > 0 and not wrong and not missing,
                 f"{checked} reproduced; wrong {wrong[:4]}; could not check {missing[:4]}")


def proposal_rows(report, S, results, store, watch, fields, stub, F, last, logs, events, wallets,
                  netuid, bpe, root_hk):
    from bittensor.core.chain_data.utils import decode_metadata

    from fugal_subnet import chain_proposals as cp
    from fugal_subnet import menu_publish as mp
    from fugal_subnet import reveal_sync

    ids = {p: entry_id(x) for p, x in PROPOSALS.items()}
    # Each node's own menus, as it held them at each boundary (its menu book
    # beside its state): what it ran the epoch on.
    menus = {n: {k: mp.MenuBook(os.path.join(results, n, "state.json") + ".menu").held(k)
                 for k in range(F, last + 1)} for n in ("vs_val1", "vs_val2")}

    def where(n, eid):
        out = {}
        for k, st in menus[n].items():
            if hasattr(st, "entries"):
                prov = {e.id for e in st.entries if e.provisional}
                full = {e.id for e in st.entries if not e.provisional}
                out[k] = ("provisional" if eid in prov else "full" if eid in full else
                          "retired" if eid in {r.id for r in st.retired} else None)
        return out

    states = {p: {n: where(n, ids[p]) for n in menus} for p in ids}

    def first(p, what):
        got = [k for k, v in states[p]["vs_val1"].items() if v == what]
        return min(got) if got else None

    j1 = first("P1", "provisional")
    full1 = first("P1", "full")
    same = j1 is not None and all(states["P1"]["vs_val2"].get(k) == states["P1"]["vs_val1"].get(k)
                                  for k in states["P1"]["vs_val1"] if k in states["P1"]["vs_val2"])
    verified = logs["vs_val1"].get(j1, {}).get("n_heads_valid") if j1 else None
    report.check(S, "proposal: filed with the head (second field), joins provisional on every "
                    "node and miner, full two epochs later",
                 j1 is not None and full1 == j1 + 2 and same and verified == 4,
                 f"{ids['P1']}: provisional from F+{j1 - F if j1 else None}, full from "
                 f"F+{full1 - F if full1 else None}; root and follower agree {same}; the 4 miners' "
                 f"proofs on that menu verified: {verified}")
    # The root's join pointer and the delta in the store.
    pointers = {}
    for k in range(F, last + 1):
        try:
            f = cp.read_fields_at(watch.st, netuid, k * bpe).get(root_hk, ())
        except Exception as e:  # noqa: BLE001
            f = (f"read failed: {type(e).__name__}",)
        pointers[k] = f[1] if len(f) > 1 else None
    delta_ok = False
    if j1 is not None and pointers.get(j1):
        jp = cp.parse_join_field(pointers[j1], j1)
        delta_ok = bool(jp and jp.delta_hash
                        and reveal_sync.join_object(netuid, jp.delta_hash) in store.objects(w4.BUCKET))
    report.check(S, "proposal: the root committed join:<epoch>:<delta> and the delta is in the store",
                 delta_ok, f"root's second field at each boundary "
                 f"{ {k - F: (v or '-')[:30] for k, v in pointers.items()} }")
    pool = (w4.reveal(results, "vs_val1", j1) or {}).get("exploration_pool") or [] if j1 else []
    measured = sum(1 for r in pool if r.get("entry") == ids["P1"] or r.get("model") == ids["P1"])
    report.check(S, "proposal: the field's exploration measures the provisional entry",
                 measured > 0 and stub.extra_calls.get(PROPOSALS["P1"]["model"], 0) > 0,
                 f"{measured} pooled pairs on it in F+{j1 - F if j1 else None}; stub calls "
                 f"{stub.extra_calls}")
    audits = [(k - F, (w4.reveal(results, "vs_val2", k) or {}).get("join_audit"))
              for k in range(F, last + 1)]
    bad = [(k, a) for k, a in audits if a and a.get("differences")]
    report.check(S, "proposal: every follower join_audit is empty",
                 any(a for _k, a in audits) and not bad,
                 f"{sum(1 for _k, a in audits if a)} audits, differing {bad[:2]}")
    late = events.get("late_epoch")
    j2 = first("P2", "provisional")
    gone2 = first("P2", "retired")
    report.check(S, "proposal: a root late for one boundary commits no pointer; the proposal "
                    "waits and joins one epoch later",
                 late is not None and cp.parse_join_field(pointers.get(late) or "", late) is None
                 and pointers.get(late + 1) is not None
                 and cp.parse_join_field(pointers[late + 1], late + 1) is not None
                 and j2 == late + 1,
                 f"P2 committed at block {events.get('p2_commit')}, by the rule joining "
                 f"F+{late - F if late else None}; root paused {events.get('root_paused')}.."
                 f"{events.get('root_resumed')}; second field at that boundary "
                 f"{pointers.get(late) if late else '-'} (none for it), at the next "
                 f"{pointers.get(late + 1) if late else '-'};"
                 f" provisional from F+{j2 - F if j2 else None}")
    report.check(S, "proposal: a model whose host fails every call is retired by the step",
                 j2 is not None and gone2 == j2 + 2,
                 f"{ids['P2']}: provisional F+{j2 - F if j2 else None}, retired "
                 f"F+{gone2 - F if gone2 else None}")
    c3 = events.get("p3_commit")
    j3 = first("P3", "provisional")
    lands_late = c3 is not None and c3 > (c3 // bpe + 1) * bpe - _margin(bpe)
    report.check(S, "proposal: one filed inside the margin before a boundary waits one epoch",
                 c3 is not None and lands_late and j3 == c3 // bpe + 2,
                 f"P3 committed at block {c3} (boundary {(c3 // bpe + 1) * bpe if c3 else '-'}, read "
                 f"block {(c3 // bpe + 1) * bpe - _margin(bpe) if c3 else '-'}); provisional from "
                 f"F+{j3 - F if j3 else None}")
    # A pre-v4 reader: field 0 only, through the SDK's decode_metadata.
    ok, seen = True, {}
    for ck in ("dr_m4", "dr_m3", "dr_val1"):
        hk = wallets[ck].hotkey.ss58_address
        try:
            raw = watch.st.substrate.query("Commitments", "CommitmentOf", [netuid, hk])
            v = getattr(raw, "value", raw)
            first_field = decode_metadata(v)
            all_fields = cp.commitment_fields(v)
        except Exception as e:  # noqa: BLE001
            first_field, all_fields = f"? {type(e).__name__}", ()
        seen[ck] = (first_field[:24], len(all_fields))
        if ck == "dr_val1":
            ok &= reveal_sync.parse_commitment(first_field) is not None
        else:
            ok &= len(first_field) == 64 and all(x in "0123456789abcdef" for x in first_field)
        ok &= len(all_fields) == 2
    report.check(S, "proposal: a pre-v4 reader (field 0 only) reads the head and reveal pointers "
                    "unchanged", ok, f"(field 0, fields): {seen}")


def margin_rows(report, S, results, fields, F, last, bpe, events, root_hk, wallets, netuid):
    """When the root's join commitment lands, against the boundary, every
    epoch; the bytes each account commits per epoch against the pallet."""
    from fugal_subnet import chain_proposals as cp

    margin, slack = _margin(bpe), _slack(bpe)
    joins = {}
    for at, f in fields.changes.get(root_hk, []):
        jp = None
        if len(f) > 1:
            m = cp._JOIN_FIELD.fullmatch(f[1])
            jp = int(m.group(1)[1:]) if m else None
        if jp is not None and jp not in joins:
            joins[jp] = at
    late = events.get("late_epoch")
    # From the root's first join commitment: before its first reveal pointer
    # is on chain it has nothing to commit the join pointer beside, and says
    # so (join_step.RootFields).
    first = min(joins) if joins else last + 1
    rows = {k - F: (k * bpe - joins[k]) if k in joins else None for k in range(first, last + 1)}
    met = bool(rows) and all(v is not None and v >= slack for k, v in rows.items()
                             if F + k != late)
    report.check(S, "the root's join read, publication and commit land inside the margin every "
                    "epoch (but the late one)",
                 met and rows.get(late - F if late else -99, "x") is None,
                 f"margin {margin} blocks (read at boundary - {margin}), commit deadline "
                 f"boundary - {slack}; blocks before the boundary the commitment landed, by epoch "
                 f"F+k: {rows}")
    per_epoch: dict = {}
    names = {w.hotkey.ss58_address: n for n, w in wallets.items()}
    for hk, rows_ in fields.changes.items():
        prev = None
        for at, f in rows_:
            if at == prev:
                continue
            prev = at
            n = names.get(hk, hk[:8])
            k = at // bpe - F
            per_epoch.setdefault(n, {}).setdefault(k, [0, 0])
            per_epoch[n][k][0] += 1
            per_epoch[n][k][1] += sum(len(x) for x in f)
    most = {n: max((tuple(v) for v in d.values()), key=lambda t: t[1]) for n, d in per_epoch.items()}
    used = {n: fields.used_space(wallets[n].hotkey.ss58_address)
            for n in ("dr_val1", "dr_m4", "dr_m3", "dr_m2")}
    try:
        space = fields.st.substrate.query("Commitments", "MaxSpace").value
        tempo = fields.st.get_subnet_hyperparameters(netuid).tempo
    except Exception as e:  # noqa: BLE001
        space, tempo = f"? {type(e).__name__}", "?"
    refused = [ln for n in os.listdir(results) if n.endswith(".log")
               for ln in w4.log_text(results, n[:-4]).splitlines()
               if any(x in ln for x in w4.PALLET_ERRORS) or "Commitment of" in ln and "refused" in ln]
    report.check(S, "commitments per account per epoch fit the pallet; none refused",
                 not refused and bool(most.get("dr_val1")),
                 f"most (commits, bytes) in one epoch by account {most}; UsedSpaceOf at the end "
                 f"{used}; MaxSpace {space} bytes per tempo of {tempo} blocks; refusals "
                 f"{refused[:2]}")

