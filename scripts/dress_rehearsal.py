#!/usr/bin/env python3
"""Full dress rehearsal against a real Substrate chain.

Every other check in this repo runs in-process against a mocked chain. This one
runs the shipped binaries — `neurons/miner.py` and `neurons/validator.py` — as
real OS processes, against a real local subtensor node, with real wallets, real
registration, real axon/dendrite traffic and real `set_weights` extrinsics.

That distinction is the whole point. The TEE pipeline shipped with three fatal
bugs behind a green CI because CI exercised a different path than production
did. A rehearsal that reimplemented the neurons would reproduce exactly that
mistake, so this drives the real ones and compares them the way an operator
would: through the artifacts they publish.

    python scripts/dress_rehearsal.py --scenario all
    python scripts/dress_rehearsal.py --scenario c --keep-chain

Scenarios
  a  one miner, one validator, one epoch; weights land AND read back
  b  three miners (honest / copier / cheap-but-wrong); dedup and ranking
  c  two validators on the same proofs; byte-identical weights and frames
  d  many epochs; evidence, burn-in, frame convergence, weight capping
  e  the real backbone path end to end

No API spend is possible: model replies are stubbed deterministically and
priced through the real pinned table.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CHAIN_NAME = "fugal_rehearsal_chain"
CHAIN_IMAGE = "ghcr.io/raofoundation/subtensor-localnet:v432"
CHAIN_DIGEST = "sha256:2354832a7c45ceb35703d64bd6f9477439f9a966e34a1d460965b8faf19439e2"
ENDPOINT = "ws://127.0.0.1:9944"
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "rehearsal")

# Long enough that the miner's 30s poll lands inside every epoch, short enough
# that a multi-epoch scenario finishes in minutes. At 24s the miner structurally
# could not keep up — it completed epoch e00000218 while the validator was
# asking about e00000235 — even though the benchmark itself takes ~0.1s.
# Set at runtime from the chain's MEASURED block rate. Epoch geometry is
# defined in blocks, so a devnet producing a block every 0.35s runs through a
# "1 hour" epoch in 100 seconds. Hardcoding an interval made the miner
# structurally unable to keep up: it completed e00000042 while the validator
# was already asking about e00000045.
EPOCH_INTERVAL_S = "3600"
TARGET_EPOCH_SECONDS = 90


# ── assertions ────────────────────────────────────────────────────────────────

class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []

    def check(self, scenario: str, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((scenario, name, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"    [{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        return bool(ok)

    def summary(self) -> int:
        print("\n" + "=" * 78)
        print(f"{'DRESS REHEARSAL':<52}{'SCENARIO':<12}{'RESULT':>8}")
        print("-" * 78)
        for scenario, name, ok, detail in self.rows:
            print(f"{'OK ' if ok else '!! '}{name[:48]:<49}{scenario:<12}"
                  f"{'PASS' if ok else 'FAIL':>8}")
            if not ok and detail:
                print(f"     ↳ {detail}")
        print("-" * 78)
        failed = [r for r in self.rows if not r[2]]
        print(f"{len(self.rows) - len(failed)} passed, {len(failed)} failed")
        if failed:
            print("\nFAIL — the subnet is not ready.")
            return 1
        print("\nPASS — the shipped binaries work against a real chain.")
        return 0


# ── chain lifecycle ───────────────────────────────────────────────────────────

def chain_running() -> bool:
    out = subprocess.run(
        ["docker", "ps", "--filter", f"name={CHAIN_NAME}", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    return CHAIN_NAME in out.stdout


def start_chain() -> None:
    if chain_running():
        print("  Chain already running, reusing", flush=True)
        return
    subprocess.run(["docker", "rm", "-f", CHAIN_NAME],
                   capture_output=True, text=True)
    print(f"  Starting pinned chain ({CHAIN_IMAGE})...", flush=True)
    proc = subprocess.run([
        "docker", "run", "-d", "--name", CHAIN_NAME,
        "-p", "9944:9944", "-p", "9945:9945",
        f"{CHAIN_IMAGE}@{CHAIN_DIGEST}",
    ], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"could not start chain: {proc.stderr.strip()}")


def stop_chain() -> None:
    subprocess.run(["docker", "rm", "-f", CHAIN_NAME], capture_output=True, text=True)


def wait_for_chain(timeout: int = 180):
    import bittensor as bt

    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            st = bt.Subtensor(network=ENDPOINT)
            block = st.get_current_block()
            print(f"  Chain ready at block {block}", flush=True)
            return st
        except Exception as e:
            last = str(e)
            time.sleep(3)
    raise RuntimeError(f"chain not ready after {timeout}s: {last}")


# ── fixtures ──────────────────────────────────────────────────────────────────

MODELS = [
    "deepseek/deepseek-v4-flash",     # cheapest
    "meta-llama/llama-4-maverick",
    "openai/gpt-5.4-nano",
]
# Ground truth for the stub: pricier models answer more questions correctly, so
# there is a genuine quality/cost tradeoff for a router to find.
SKILL = {MODELS[0]: 0.30, MODELS[1]: 0.60, MODELS[2]: 0.85}


def write_pool(path: str, n: int = 120) -> str:
    import random

    rng = random.Random(20260904)
    benches = ["gsm8k", "math", "mmlu", "aime"]
    pool = [{
        "question_id": f"dr_{i:04d}",
        "prompt": f"What is {rng.randint(1, 99)} + {rng.randint(1, 99)}?",
        "gold": str(rng.randint(2, 198)),
        "grader_id": "numeric_final",
        "benchmark": benches[i % len(benches)],
        "metadata": {},
    } for i in range(n)]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pool, f)
    return path


def write_head(path: str, seed: int, models=None) -> str:
    """Write a real .npz head. A different seed routes differently."""
    import numpy as np

    from fugal_subnet.config import HEAD_HIDDEN_DIM

    models = models or MODELS
    rng = np.random.RandomState(seed)
    W = (rng.randn(len(models), HEAD_HIDDEN_DIM) * 0.02).astype(np.float32)
    b = rng.randn(len(models)).astype(np.float32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, W=W, b=b, models=np.array(models, dtype="U100"))
    return path


def biased_head(path: str, model_index: int) -> str:
    """A head that always routes to one model — the degenerate strategies."""
    import numpy as np

    from fugal_subnet.config import HEAD_HIDDEN_DIM

    W = np.zeros((len(MODELS), HEAD_HIDDEN_DIM), dtype=np.float32)
    b = np.full(len(MODELS), -10.0, dtype=np.float32)
    b[model_index] = 10.0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, W=W, b=b, models=np.array(MODELS, dtype="U100"))
    return path


# ── running the real binaries ─────────────────────────────────────────────────

def miner_env(extra=None):
    env = os.environ.copy()
    env["FUGAL_EPOCH_INTERVAL"] = EPOCH_INTERVAL_S
    env["FUGAL_SLICE_SIZE"] = "40"
    env["FUGAL_EXPLORE_FRACTION"] = "0.10"
    env["FUGAL_REQUIRE_COMMITMENT"] = "0"
    # NOT 127.0.0.1: the chain rejects loopback for serve_axon at pool
    # validation, so the axon would never register and no validator could
    # reach the miner.
    from scripts.setup_local_testnet import lan_ip
    env["FUGAL_AXON_IP"] = lan_ip()
    env["PYTHONUNBUFFERED"] = "1"
    # The real proxy, pointed at the local stub. No key is set, so even a bug
    # that reached the real OpenRouter would be unauthenticated rather than
    # billable.
    env["FUGAL_OPENROUTER_BASE"] = "http://127.0.0.1:8799/v1"
    # One pool, one source, both neurons. Also keeps the run off HuggingFace.
    env["FUGAL_BENCHMARK_POOL"] = os.path.abspath(
        os.path.join(RESULTS, "pool.json")
    )
    env.pop("OPENROUTER_API_KEY", None)
    env.update(extra or {})
    return env


def start_miner(coldkey: str, netuid: int, head: str, pool: str,
                port: int, log_path: str) -> subprocess.Popen:
    log = open(log_path, "w", encoding="utf-8")
    # Each miner needs its own metering-proxy port: several miners on one host
    # would otherwise race to bind the same one. In production a miner has its
    # own TEE VM, so the default is fine there.
    env = miner_env({"FUGAL_TEE_PROXY_PORT": str(port + 1000)})
    return subprocess.Popen(
        [sys.executable, "neurons/miner.py",
         "--network", ENDPOINT, "--netuid", str(netuid),
         "--coldkey", coldkey, "--hotkey", "default",
         "--head-path", head, "--benchmark-pool", pool,
         "--port", str(port), "--mock"],
        stdout=log, stderr=subprocess.STDOUT, env=env, text=True,
        # Detached stdin: several miners inheriting one terminal race for the
        # wallet password prompt, and the losers block forever or exit.
        stdin=subprocess.DEVNULL,
    )


def run_validator_once(coldkey: str, netuid: int, state_path: str,
                       log_path: str, timeout: int = 300) -> dict:
    """Run one real validator epoch; return its parsed epoch-log entry."""
    env = miner_env({
        "FUGAL_STATE_PATH": state_path,
        "FUGAL_EPOCH_LOG_DIR": os.path.dirname(state_path),
        "FUGAL_EPOCH_DIR": os.path.join(os.path.dirname(state_path), "epochs"),
    })
    with open(log_path, "a", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "neurons/validator.py",
             "--network", ENDPOINT, "--netuid", str(netuid),
             "--coldkey", coldkey, "--hotkey", "default",
             "--once", "--mock"],
            stdout=log, stderr=subprocess.STDOUT, env=env,
            text=True, timeout=timeout,
        )
    log_file = os.path.join(os.path.dirname(state_path), "epochs.jsonl")
    if not os.path.exists(log_file):
        return {}
    with open(log_file, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else {}



# ── chain fixtures ────────────────────────────────────────────────────────────

def setup_subnet(subtensor, coldkeys: list[str]) -> tuple[int, dict]:
    """Create a subnet and register every wallet on it. Reuses the working
    primitives in setup_local_testnet rather than reimplementing extrinsics."""
    import bittensor as bt

    from scripts.setup_local_testnet import (
        activate_subnet,
        create_subnet,
        fund_wallet,
        register_wallet,
        relax_chain_limits,
        stake_validator,
    )

    netuid = create_subnet(subtensor)
    print(f"  Subnet created: netuid {netuid}", flush=True)

    wallets = {}
    for name in coldkeys:
        w = bt.Wallet(name=name, hotkey="default")
        if not w.coldkey_file.exists_on_device():
            # suppress=True keeps mnemonics out of the log and, with
            # use_password=False, keeps creation non-interactive — a prompt
            # here blocks a miner subprocess that has no stdin.
            w.create_new_coldkey(use_password=False, overwrite=True, suppress=True)
            w.create_new_hotkey(use_password=False, overwrite=True, suppress=True)
        wallets[name] = w
        fund_wallet(subtensor, w, amount_tao=10000)

    activate_subnet(subtensor, netuid)
    relax_chain_limits(subtensor, netuid)

    for name, w in wallets.items():
        register_wallet(subtensor, w, netuid)
        print(f"  Registered {name}", flush=True)

    for name in ("dr_val1", "dr_val2"):
        if name in wallets:
            stake_validator(subtensor, wallets[name], netuid)

    return netuid, wallets


def uid_of(subtensor, netuid: int, wallet) -> int:
    mg = subtensor.metagraph(netuid)
    hk = wallet.hotkey.ss58_address
    return mg.hotkeys.index(hk) if hk in mg.hotkeys else -1


def current_epoch_id(subtensor) -> str:
    """The epoch id both neurons will independently derive right now."""
    from fugal_subnet.benchmarks.slicer import (
        blocks_per_epoch,
        epoch_id_for_block,
        epoch_index_for_block,
    )

    bpe = blocks_per_epoch(int(EPOCH_INTERVAL_S))
    return epoch_id_for_block(
        epoch_index_for_block(subtensor.get_current_block(), bpe)
    )


def wait_for_proof(procs, timeout=600) -> bool:
    """Wait until every miner has computed its first proof."""
    return _wait(procs, "complete:", timeout)


def wait_for_epoch(procs, subtensor, timeout=600, after: str | None = None) -> str | None:
    """Wait until every miner holds a proof for the SAME current epoch.

    The miner only serves a proof for the epoch the validator asks about, so
    the rehearsal has to line them up rather than assume they agree — which is
    exactly the failure the epoch-id single-source fix was about, seen from the
    other side.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        epoch = current_epoch_id(subtensor)
        if after is not None and epoch == after:
            # Same epoch as last time — wait for the chain to roll over rather
            # than re-querying a proof the validator has already scored.
            time.sleep(5)
            continue
        if _wait(procs, f"Epoch {epoch} complete", timeout=120, quiet=True):
            # Still the same epoch? If it rolled while we waited, try again.
            if current_epoch_id(subtensor) == epoch:
                return epoch
    return None


def _wait(procs, marker: str, timeout: int, quiet: bool = False) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready = 0
        for _, log_path in procs:
            try:
                with open(log_path, encoding="utf-8") as f:
                    if marker in f.read():
                        ready += 1
            except FileNotFoundError:
                pass
        if ready == len(procs):
            return True
        for proc, log_path in procs:
            rc = proc.poll()
            if rc is not None:
                with open(log_path, encoding="utf-8") as f:
                    lines = [
                        ln for ln in f.read().splitlines()
                        if ln.strip() and not ln.startswith("\x1b[")
                        and "Loading weights" not in ln
                    ]
                tail = "\n".join(lines[-12:])
                raise RuntimeError(
                    f"miner {os.path.basename(log_path)} exited with code {rc}:\n{tail}"
                )
        time.sleep(3)
    return False


# ── scenarios ─────────────────────────────────────────────────────────────────

def run_scenarios(args, report: Report) -> int:
    import bittensor as bt

    pool_path = write_pool(os.path.join(RESULTS, "pool.json"))
    with open(pool_path, encoding="utf-8") as f:
        pool = json.load(f)
    gold_by_prompt = {q["prompt"]: q["gold"] for q in pool}

    from scripts.setup_local_testnet import StubUpstream
    stub = StubUpstream(gold_by_prompt, SKILL)
    stub.start()

    want = ["a", "b", "c", "d", "e"] if args.scenario == "all" else [args.scenario]
    subtensor = bt.Subtensor(network=ENDPOINT)

    coldkeys = ["dr_val1", "dr_val2", "dr_m1", "dr_m2", "dr_m3"]
    netuid, wallets = setup_subnet(subtensor, coldkeys)

    scenarios = {
        "a": lambda p: scenario_a(report, subtensor, netuid, wallets, pool_path, p),
        "b": lambda p: scenario_b(report, subtensor, netuid, wallets, pool_path, p),
        "c": lambda p: scenario_c(report, subtensor, netuid, wallets, pool_path, p),
        "d": lambda p: scenario_d(report, subtensor, netuid, wallets, pool_path, p,
                                  args.epochs),
        "e": lambda p: scenario_e(report, subtensor, netuid, wallets, pool_path, p,
                                  stub),
    }

    try:
        for key in want:
            # Each scenario owns its miners and stops them afterwards. Leaving
            # them running exhausts memory and, worse, reuses the same hotkeys:
            # a second axon for one hotkey overwrites the first's address on
            # chain, so the earlier scenario's miner silently becomes
            # unreachable.
            procs: list = []
            try:
                scenarios[key](procs)
            except Exception as e:
                report.check(key, f"scenario {key} ran to completion", False, str(e))
            finally:
                _stop(procs)
    finally:
        stub.stop()
    return 0


def _stop(procs) -> None:
    for proc, _ in procs:
        proc.terminate()
    for proc, _ in procs:
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    if procs:
        # Give the OS a moment to release ports and memory before the next
        # scenario starts loading a backbone.
        time.sleep(5)


def _launch(procs, coldkey, netuid, head, pool_path, port, tag):
    log = os.path.join(RESULTS, f"miner_{tag}.log")
    proc = start_miner(coldkey, netuid, head, pool_path, port, log)
    procs.append((proc, log))
    return proc, log


def _launch_sequential(procs, netuid, pool_path, specs, timeout=300):
    """Start miners one at a time, each waiting for the previous to embed.

    Every miner loads Qwen3-0.6B (~2.4GB) at startup. Three at once exceeds
    this machine's memory and the kernel kills one mid-load. The miner releases
    the backbone as soon as embeddings exist, so serialising just the embedding
    phase keeps peak memory at one model rather than N — which is only possible
    because embedding moved to startup and the backbone is freed after it.
    """
    started = []
    for coldkey, head, port, tag in specs:
        proc, log = _launch(procs, coldkey, netuid, head, pool_path, port, tag)
        started.append((proc, log))
        if not _wait([(proc, log)], "backbone released", timeout, quiet=True):
            raise RuntimeError(f"miner {tag} did not finish embedding in {timeout}s")
        print(f"    {tag}: embeddings ready, backbone released", flush=True)
    return started


def scenario_a(report, subtensor, netuid, wallets, pool_path, procs):
    print("\n[Scenario A] one miner, one validator, one epoch", flush=True)
    head = write_head(os.path.join(RESULTS, "head_a.npz"), seed=1)
    _launch(procs, "dr_m1", netuid, head, pool_path, 8101, "a")

    if not wait_for_proof(procs[-1:]):
        report.check("a", "miner produces a proof", False, "no proof within timeout")
        return
    report.check("a", "miner produces a proof", True)

    epoch = wait_for_epoch(procs[-1:], subtensor)
    report.check("a", "miner and validator agree on the current epoch",
                 epoch is not None, f"epoch={epoch}")
    if not epoch:
        return

    state = os.path.join(RESULTS, "val_a")
    os.makedirs(state, exist_ok=True)
    entry = run_validator_once("dr_val1", netuid,
                               os.path.join(state, "state.json"),
                               os.path.join(RESULTS, "val_a.log"))

    report.check("a", "validator verified the proof",
                 entry.get("n_heads_valid", 0) >= 1,
                 f"valid={entry.get('n_heads_valid')} invalid={entry.get('n_heads_invalid')}")
    report.check("a", "set_weights extrinsic succeeded",
                 bool(entry.get("set_weights_success")),
                 str(entry.get("set_weights_msg", ""))[:120])
    report.check("a", "weights confirmed by reading them back off chain",
                 bool(entry.get("weights_confirmed_on_chain")),
                 str(entry.get("weights_confirm_detail", ""))[:160])


def scenario_b(report, subtensor, netuid, wallets, pool_path, procs):
    print("\n[Scenario B] three miners: honest, copier, cheap-but-wrong", flush=True)
    honest = write_head(os.path.join(RESULTS, "head_honest.npz"), seed=1)
    copier = write_head(os.path.join(RESULTS, "head_copier.npz"), seed=1)  # identical
    cheap = biased_head(os.path.join(RESULTS, "head_cheap.npz"), 0)

    _launch_sequential(procs, netuid, pool_path, [
        ("dr_m1", honest, 8111, "b_honest"),
        ("dr_m2", copier, 8112, "b_copier"),
        ("dr_m3", cheap, 8113, "b_cheap"),
    ])

    if not wait_for_proof(procs[-3:]):
        report.check("b", "all three miners produced proofs", False)
        return
    report.check("b", "all three miners produced proofs", True)
    if not wait_for_epoch(procs[-3:], subtensor):
        report.check("b", "all three miners reached the same epoch", False)
        return
    report.check("b", "all three miners reached the same epoch", True)

    state = os.path.join(RESULTS, "val_b")
    os.makedirs(state, exist_ok=True)
    entry = run_validator_once("dr_val1", netuid,
                               os.path.join(state, "state.json"),
                               os.path.join(RESULTS, "val_b.log"))

    report.check("b", "validator verified all three proofs",
                 entry.get("n_heads_valid", 0) == 3,
                 f"valid={entry.get('n_heads_valid')}")

    dq = set(entry.get("dedup_disqualified") or [])
    weights = entry.get("weights") or {}
    honest_uid = uid_of(subtensor, netuid, wallets["dr_m1"])
    copier_uid = uid_of(subtensor, netuid, wallets["dr_m2"])
    cheap_uid = uid_of(subtensor, netuid, wallets["dr_m3"])

    report.check("b", "dedup disqualified exactly one of the identical pair",
                 len(dq & {honest_uid, copier_uid}) == 1,
                 f"disqualified={sorted(dq)} honest={honest_uid} copier={copier_uid}")
    report.check("b", "the disqualified copy earns no weight",
                 all(weights.get(str(u), 0.0) == 0.0 for u in dq),
                 f"weights={weights}")
    survivor = ({honest_uid, copier_uid} - dq)
    surv = max((weights.get(str(u), 0.0) for u in survivor), default=0.0)
    report.check("b", "a real router outranks the always-cheapest router",
                 surv > weights.get(str(cheap_uid), 0.0),
                 f"router={surv:.4f} cheap-only={weights.get(str(cheap_uid), 0.0):.4f}")


def scenario_c(report, subtensor, netuid, wallets, pool_path, procs):
    print("\n[Scenario C] two validators, same proofs", flush=True)
    # Three miners with genuinely different routing, so the weight vector both
    # validators must agree on is non-trivial. With one miner it is {uid: 1.0},
    # which two validators would match by construction rather than by agreeing.
    _launch_sequential(procs, netuid, pool_path, [
        ("dr_m1", write_head(os.path.join(RESULTS, "head_c1.npz"), seed=5),
         8121, "c1"),
        ("dr_m2", write_head(os.path.join(RESULTS, "head_c2.npz"), seed=6),
         8122, "c2"),
        ("dr_m3", biased_head(os.path.join(RESULTS, "head_c3.npz"), 2),
         8123, "c3"),
    ])
    if not wait_for_proof(procs[-3:]) or not wait_for_epoch(procs[-3:], subtensor):
        report.check("c", "miners produced proofs for the current epoch", False)
        return

    entries = []
    for name in ("val_c1", "val_c2"):
        state = os.path.join(RESULTS, name)
        os.makedirs(state, exist_ok=True)
        entries.append(run_validator_once(
            "dr_val1" if name.endswith("1") else "dr_val2", netuid,
            os.path.join(state, "state.json"),
            os.path.join(RESULTS, f"{name}.log"),
        ))

    a, b = entries
    report.check("c", "both validators verified the same proofs",
                 a.get("n_heads_valid") == b.get("n_heads_valid") >= 2,
                 f"{a.get('n_heads_valid')} vs {b.get('n_heads_valid')}")
    weights_a = a.get("weights") or {}
    report.check("c", "the weight vector is non-trivial (a real comparison)",
                 len([w for w in weights_a.values() if w > 0]) >= 2,
                 f"weights={weights_a}")
    report.check("c", "both computed identical weight vectors (I1)",
                 weights_a == b.get("weights"),
                 f"{weights_a} vs {b.get('weights')}")
    report.check("c", "both computed identical scores",
                 a.get("scores") == b.get("scores"))

    frames = []
    for name in ("val_c1", "val_c2"):
        sp = os.path.join(RESULTS, name, "state.json")
        try:
            with open(sp, encoding="utf-8") as f:
                frames.append(json.load(f).get("frame"))
        except FileNotFoundError:
            frames.append(None)
    report.check("c", "both derived identical reference frames (I9)",
                 frames[0] is not None and frames[0] == frames[1])


def scenario_d(report, subtensor, netuid, wallets, pool_path, procs, epochs):
    print(f"\n[Scenario D] {epochs} epochs, evidence and frame over time", flush=True)
    head = write_head(os.path.join(RESULTS, "head_d.npz"), seed=9)
    _launch(procs, "dr_m1", netuid, head, pool_path, 8131, "d")
    if not wait_for_proof(procs[-1:]):
        report.check("d", "miner produced a proof", False)
        return

    state = os.path.join(RESULTS, "val_d")
    os.makedirs(state, exist_ok=True)
    state_path = os.path.join(state, "state.json")
    log_path = os.path.join(RESULTS, "val_d.log")

    seen = []
    last_epoch = None
    for i in range(epochs):
        epoch = wait_for_epoch(procs[-1:], subtensor, after=last_epoch)
        if epoch is None:
            print(f"    (gave up waiting for a new epoch after {i})", flush=True)
            break
        last_epoch = epoch
        entry = run_validator_once("dr_val1", netuid, state_path, log_path)
        if entry:
            seen.append(entry)
            print(f"    epoch {i + 1}/{epochs}: {epoch} "
                  f"valid={entry.get('n_heads_valid')} "
                  f"weights={entry.get('weights')}", flush=True)

    report.check("d", f"ran {epochs} epochs without an unhandled error",
                 len(seen) == epochs, f"completed {len(seen)}/{epochs}")
    if len(seen) < 2:
        return

    with open(state_path, encoding="utf-8") as f:
        final = json.load(f)
    rec = next(iter(final.get("records", {}).values()), {})
    ev = rec.get("evidence") or {}

    report.check("d", "evidence accumulated across epochs",
                 ev.get("n_total", 0) > 40,
                 f"n_total={ev.get('n_total')}")
    report.check("d", "evidence stayed keyed to one unchanged artifact",
                 ev.get("epochs_accumulated", 0) >= 2,
                 f"epochs_accumulated={ev.get('epochs_accumulated')}")

    frame = final.get("frame") or {}
    trials = sum((frame.get("trials") or {}).values())
    report.check("d", "reference frame accumulated exploration samples",
                 trials > 0, f"trials={trials:.0f}")

    caps = [e.get("weight_capped") for e in seen[1:]]
    report.check("d", "weight capping engaged after the first epoch",
                 all(caps) if caps else False, f"capped={caps}")

    confirmed = [bool(e.get("weights_confirmed_on_chain")) for e in seen]
    report.check("d", "weights confirmed on chain every epoch",
                 all(confirmed), f"{sum(confirmed)}/{len(confirmed)} confirmed")


def scenario_e(report, subtensor, netuid, wallets, pool_path, procs, stub):
    print("\n[Scenario E] real backbone through the shipped binary", flush=True)
    head = write_head(os.path.join(RESULTS, "head_e.npz"), seed=11)
    calls_before = stub.calls
    proc, log = _launch(procs, "dr_m1", netuid, head, pool_path, 8141, "e")

    ready = wait_for_proof([(proc, log)], timeout=600)
    if ready:
        wait_for_epoch([(proc, log)], subtensor)
    with open(log, encoding="utf-8") as f:
        text = f.read()

    report.check("e", "miner computed real Qwen3-0.6B embeddings",
                 "Embeddings ready" in text,
                 [ln for ln in text.splitlines() if "Embeddings" in ln][:1])
    report.check("e", "backbone released after embedding (not held per-epoch)",
                 "backbone released" in text)
    report.check("e", "miner produced a proof with the real backbone", ready)
    report.check("e", "the real metering proxy made real HTTP calls",
                 stub.calls > calls_before,
                 f"{stub.calls - calls_before} calls through the proxy")

    if not ready:
        return
    state = os.path.join(RESULTS, "val_e")
    os.makedirs(state, exist_ok=True)
    entry = run_validator_once("dr_val1", netuid,
                               os.path.join(state, "state.json"),
                               os.path.join(RESULTS, "val_e.log"))
    report.check("e", "validator verified a real-backbone proof",
                 entry.get("n_heads_valid", 0) >= 1,
                 f"valid={entry.get('n_heads_valid')}")

    # The bundle now rides inline in the synapse instead of being fetched from
    # an external store, so a full-size payload must survive a real axon round
    # trip. Asserted rather than inferred from the field's max_length.
    with open(log, encoding="utf-8") as f:
        served = [ln for ln in f.read().splitlines() if "serving bundle" in ln]
    report.check("e", "a full bundle round-tripped over a real axon",
                 bool(served), served[-1].split("INFO")[-1].strip() if served
                 else "miner never served a bundle")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="all",
                    choices=["a", "b", "c", "d", "e", "all"])
    ap.add_argument("--keep-chain", action="store_true",
                    help="Leave the chain container running afterwards")
    ap.add_argument("--epochs", type=int, default=12,
                    help="Epochs for scenario d (default 12)")
    args = ap.parse_args()

    report = Report()
    shutil.rmtree(RESULTS, ignore_errors=True)
    os.makedirs(RESULTS, exist_ok=True)

    print("=" * 78)
    print("FUGAL DRESS REHEARSAL — real chain, real binaries, no API spend")
    print("=" * 78)

    try:
        start_chain()
        subtensor0 = wait_for_chain()
        from scripts.setup_local_testnet import calibrate_epoch_interval
        global EPOCH_INTERVAL_S
        EPOCH_INTERVAL_S = calibrate_epoch_interval(
            subtensor0, TARGET_EPOCH_SECONDS, EPOCH_INTERVAL_S,
        )
        print("\nScenario harness ready.\n", flush=True)
        # Scenario bodies are added below; see run_scenarios().
        rc = run_scenarios(args, report)
    finally:
        if not args.keep_chain:
            stop_chain()
        else:
            print(f"\nChain left running as {CHAIN_NAME} (--keep-chain)")

    return rc or report.summary()


if __name__ == "__main__":
    raise SystemExit(main())
