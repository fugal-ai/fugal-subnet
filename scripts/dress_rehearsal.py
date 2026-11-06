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
  v2_flip  protocol v2 across an activation block (not part of "all"; see
           scenario_v2_flip, and scripts/rehearsal_v2.py for the in-process half)

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
# Scenario v2_flip runs longer epochs: the v2 miner's late-start guard (see main).
V2_FLIP_EPOCH_SECONDS = 180


# ── assertions ────────────────────────────────────────────────────────────────

class Report:
    def __init__(self, pass_message: str = "the shipped binaries work against a real chain.") -> None:
        self.pass_message = pass_message
        # ok is True (pass), False (fail) or None (skipped: the code it checks
        # is not on this branch yet; never a failure).
        self.rows: list[tuple[str, str, bool | None, str]] = []

    def check(self, scenario: str, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((scenario, name, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"    [{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        return bool(ok)

    def skip(self, scenario: str, name: str, why: str) -> None:
        self.rows.append((scenario, name, None, why))
        print(f"    [SKIP] {name} — {why}", flush=True)

    @property
    def failed(self) -> list:
        return [r for r in self.rows if r[2] is False]

    def summary(self) -> int:
        print("\n" + "=" * 78)
        print(f"{'DRESS REHEARSAL':<52}{'SCENARIO':<12}{'RESULT':>8}")
        print("-" * 78)
        for scenario, name, ok, detail in self.rows:
            word = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
            print(f"{'!! ' if ok is False else '-- ' if ok is None else 'OK '}"
                  f"{name[:48]:<49}{scenario:<12}{word:>8}")
            if ok is not True and detail:
                print(f"     ↳ {detail}")
        print("-" * 78)
        failed = self.failed
        skipped = [r for r in self.rows if r[2] is None]
        passed = len(self.rows) - len(failed) - len(skipped)
        print(f"{passed} passed, {len(failed)} failed, {len(skipped)} skipped")
        if failed:
            print("\nFAIL — the subnet is not ready.")
            return 1
        print(f"\nPASS — {self.pass_message}")
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
    from fugal_subnet.routing_protocol import MANIFEST_PATH
    from fugal_subnet.vendor import success_contract as contract
    np.savez(path, **contract.make_head(W, b, models, json.loads(MANIFEST_PATH.read_text()), "synthetic-test-only rehearsal"))
    return path


# Protocol v2 fixtures. The v2 validator scores v2 epochs on the v2 pool, the
# v2 model list and the v2 token manifest (epoch_plan, routing_protocol), so the
# flip rehearsal needs all three to be v2's: v2 benchmark names (so
# select_slice_v2 fills its categories), and heads over three v2 models of
# different price, built against the v2 manifest.
def v2_models() -> list[str]:
    """Three v2 models across the whole table: cheapest, middle and dearest.

    V is CHAMPION_ACCURACY_VALUE x the pinned anchor's spend per question
    (Claude Fable 5.1, about $0.77 at m = 30), so even the dearest model's
    extra spend costs about a point of NV and the stub's skill ladder
    (0.30 / 0.60 / 0.85) decides the seat, as accuracy should. Before V was
    pinned, the champion's own (cheap) spend set it and an always-dearest
    challenger could never take the seat.
    """
    from fugal_subnet.routing_protocol import price_table

    table = price_table(2)
    ranked = sorted(table, key=lambda m: (table[m][1], table[m][0], m))
    return [ranked[0], ranked[len(ranked) // 2], ranked[-1]]


def write_pool_v2(path: str, per_benchmark: int = 20) -> str:
    """A v2-shaped fixture pool: every v2 benchmark, prompts distinct from v1's."""
    import random

    from fugal_subnet.config import BENCHMARK_CATEGORY_V2

    rng = random.Random(20261104)
    pool = []
    for bench in sorted(BENCHMARK_CATEGORY_V2):
        for i in range(per_benchmark):
            a, b = rng.randint(100, 999), rng.randint(100, 999)
            pool.append({
                "question_id": f"dr2_{bench}_{i:03d}",
                "prompt": f"[v2 {bench}] What is {a} + {b}?",
                "gold": str(a + b),
                "grader_id": "numeric_final",
                "benchmark": bench,
                "metadata": {},
            })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pool, f)
    return path


def _v2_head(path: str, W, b, models) -> str:
    import numpy as np

    from fugal_subnet.routing_protocol import manifest_path
    from fugal_subnet.vendor import success_contract as contract
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **contract.make_head(W, b, models, json.loads(manifest_path(2).read_text()),
                                        "synthetic-test-only rehearsal (v2)"))
    return path


def write_head_v2(path: str, seed: int, hidden_mean=None, scale: float = 20.0) -> str:
    """A v2 router head over the three v2 rehearsal models that really routes.

    The backbone's pooled hidden states are nearly collinear (pairwise cosine
    0.997 on the fixture pool), so a random W at the v1 writer's 0.02 scale
    moves the logits by about 0.001 per question and the head is a constant
    router to whichever model its bias favours. Seed 24 favoured the cheapest
    model, whose stub skill (0.30) is exactly CHAMPION_MIN_ACCURACY, and the
    first champion sat on the floor: in one run it dipped to 0.2833 on the
    take epoch, was correctly not paid, and the 90/10 payout check failed.

    With `hidden_mean` (the cached embeddings' mean, saved by the warm-up)
    the bias cancels the common direction and `scale` amplifies what is left,
    so the head splits the fixture pool across all three models (seed 61:
    63/63/74 of 200 on x86_64) and its expected accuracy on the stub ladder
    is about 0.60, well clear of the floor. Without a mean it is the old
    constant router.
    """
    import numpy as np

    from fugal_subnet.config import HEAD_HIDDEN_DIM

    models = v2_models()
    rng = np.random.RandomState(seed)
    if hidden_mean is None:
        W = (rng.randn(len(models), HEAD_HIDDEN_DIM) * 0.02).astype(np.float32)
        b = rng.randn(len(models)).astype(np.float32)
    else:
        mean = np.asarray(hidden_mean, dtype=np.float64).reshape(HEAD_HIDDEN_DIM)
        W = (rng.randn(len(models), HEAD_HIDDEN_DIM) * scale).astype(np.float32)
        b = (-(W.astype(np.float64) @ mean) + rng.randn(len(models)) * 0.5).astype(np.float32)
    return _v2_head(path, W, b, models)


def biased_head_v2(path: str, model_index: int) -> str:
    """A v2 head that always routes to one of the three v2 rehearsal models."""
    import numpy as np

    from fugal_subnet.config import HEAD_HIDDEN_DIM

    models = v2_models()
    W = np.zeros((len(models), HEAD_HIDDEN_DIM), dtype=np.float32)
    b = np.full(len(models), -10.0, dtype=np.float32)
    b[model_index] = 10.0
    return _v2_head(path, W, b, models)


def biased_head(path: str, model_index: int) -> str:
    """A head that always routes to one model — the degenerate strategies."""
    import numpy as np

    from fugal_subnet.config import HEAD_HIDDEN_DIM

    W = np.zeros((len(MODELS), HEAD_HIDDEN_DIM), dtype=np.float32)
    b = np.full(len(MODELS), -10.0, dtype=np.float32)
    b[model_index] = 10.0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    from fugal_subnet.routing_protocol import MANIFEST_PATH
    from fugal_subnet.vendor import success_contract as contract
    np.savez(path, **contract.make_head(W, b, MODELS, json.loads(MANIFEST_PATH.read_text()), "synthetic-test-only rehearsal"))
    return path


# ── running the real binaries ─────────────────────────────────────────────────

def miner_env(extra=None):
    env = os.environ.copy()
    env["FUGAL_EPOCH_INTERVAL"] = EPOCH_INTERVAL_S
    env["FUGAL_SLICE_SIZE"] = "40"
    # Half the slice explores. Exploration targets are drawn over the WHOLE
    # pinned price table (17 models), and the frontier values any unmeasured
    # model at 1.0, so until every model priced at or below a head's cost per
    # question has been explored that head's headroom is negative by
    # construction. At 0.10 that was 4 samples an epoch and scenario B's
    # frontier was still unmeasured at both heads' prices after five epochs
    # (run 41513035321); at 0.50 it is 20, and the 17 models are covered in a
    # few epochs. Stubbed calls, so this costs nothing but seconds. Mock
    # neurons accept the consensus override with a warning, as they do for
    # the slice size above.
    env["FUGAL_EXPLORE_FRACTION"] = "0.50"
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
    # The v2 pool, for validators scoring v2 epochs (loader.load_pool(2)) and
    # v2 miners. Written only by scenario v2_flip; unused otherwise.
    env["FUGAL_BENCHMARK_POOL_V2"] = os.path.abspath(
        os.path.join(RESULTS, "pool_v2.json")
    )
    # A 120-question fixture is not the pinned pool and must say so, or the
    # loader refuses it (manifest mismatch, below the size floor) — correctly.
    env["FUGAL_POOL_UNPINNED"] = "1"
    env.pop("OPENROUTER_API_KEY", None)
    # The neurons import fugal_subnet from THIS checkout, not from whatever
    # tree the environment's editable install happens to point at. A shared
    # venv's editable mapping follows the last `uv sync`, which may be another
    # worktree; the rehearsal would then test code it was not run from.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = os.pathsep.join(
        [root] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env.update(extra or {})
    return env


def start_miner(coldkey: str, netuid: int, head: str, pool: str,
                port: int, log_path: str, extra_env=None) -> subprocess.Popen:
    log = open(log_path, "w", encoding="utf-8")
    # Each miner needs its own metering-proxy port: several miners on one host
    # would otherwise race to bind the same one. In production a miner has its
    # own TEE VM, so the default is fine there.
    env = miner_env({"FUGAL_TEE_PROXY_PORT": str(port + 1000), **(extra_env or {})})
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


def validator_env(state_path: str, extra_env=None) -> dict:
    return miner_env({
        "FUGAL_STATE_PATH": state_path,
        "FUGAL_EPOCH_LOG_DIR": os.path.dirname(state_path),
        "FUGAL_EPOCH_DIR": os.path.join(os.path.dirname(state_path), "epochs"),
        **(extra_env or {}),
    })


def run_validator_once(coldkey: str, netuid: int, state_path: str,
                       log_path: str, timeout: int = 300, extra_env=None) -> dict:
    """Run one real validator epoch; return its parsed epoch-log entry."""
    env = validator_env(state_path, extra_env)
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
    want = ["a", "b", "c", "d", "e"] if args.scenario == "all" else [args.scenario]
    skill = dict(SKILL)
    if "v2_flip" in want:
        # v2 epochs ask v2 questions of v2 models; the stub answers both, with
        # the same cheap/mid/dear skill ladder.
        with open(write_pool_v2(os.path.join(RESULTS, "pool_v2.json")), encoding="utf-8") as f:
            gold_by_prompt.update({q["prompt"]: q["gold"] for q in json.load(f)})
        skill.update(dict(zip(v2_models(), (0.30, 0.60, 0.85))))

    from scripts.setup_local_testnet import StubUpstream
    stub = StubUpstream(gold_by_prompt, skill)
    stub.start()

    subtensor = bt.Subtensor(network=ENDPOINT)

    coldkeys = ["dr_val1", "dr_val2", "dr_m1", "dr_m2", "dr_m3"]
    if "v2_flip" in want:
        # A control validator with no activation block. Registered last so
        # every other wallet keeps the UID it has in scenarios a-e; unstaked,
        # so its set_weights is refused, but it scores and reveals like a peer.
        coldkeys.append("dr_ctl")
    netuid, wallets = setup_subnet(subtensor, coldkeys)

    scenarios = {
        "a": lambda p: scenario_a(report, subtensor, netuid, wallets, pool_path, p),
        "b": lambda p: scenario_b(report, subtensor, netuid, wallets, pool_path, p),
        "c": lambda p: scenario_c(report, subtensor, netuid, wallets, pool_path, p),
        "d": lambda p: scenario_d(report, subtensor, netuid, wallets, pool_path, p,
                                  args.epochs),
        "e": lambda p: scenario_e(report, subtensor, netuid, wallets, pool_path, p,
                                  stub),
        "v2_flip": lambda p: scenario_v2_flip(report, subtensor, netuid, wallets,
                                              pool_path, p, args.v2_epochs),
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


def _launch(procs, coldkey, netuid, head, pool_path, port, tag, extra_env=None):
    log = os.path.join(RESULTS, f"miner_{tag}.log")
    proc = start_miner(coldkey, netuid, head, pool_path, port, log, extra_env)
    procs.append((proc, log))
    return proc, log


def _launch_sequential(procs, netuid, pool_path, specs, timeout=300, extra_env=None):
    """Start miners one at a time, each waiting for the previous to embed.

    Every miner loads Qwen3-0.6B (~2.4GB) at startup. Three at once exceeds
    this machine's memory and the kernel kills one mid-load. The miner releases
    the backbone as soon as embeddings exist, so serialising just the embedding
    phase keeps peak memory at one model rather than N — which is only possible
    because embedding moved to startup and the backbone is freed after it.
    """
    started = []
    for coldkey, head, port, tag in specs:
        proc, log = _launch(procs, coldkey, netuid, head, pool_path, port, tag, extra_env)
        started.append((proc, log))
        if not _wait([(proc, log)], "backbone released", timeout, quiet=True):
            raise RuntimeError(f"miner {tag} did not finish embedding in {timeout}s")
        print(f"    {tag}: embeddings ready, backbone released", flush=True)
    return started


# Scenario B judges headroom only once the frontier has measured every model
# the rehearsal heads route to; three cheap models and three exploration
# slots per epoch make that likely within a few epochs, never certain.
B_MAX_EPOCHS = 8


def frontier_measured(scores: dict, uids) -> bool:
    """True when the frontier has a measured value at every listed miner's price.

    `reference_accuracy` in an epoch-log score entry is the frontier's value at
    that miner's cost per question. The frontier values an unmeasured model at
    exactly 1.0 (nothing is known, nothing is paid above it), so 1.0 means the
    comparison would be against ignorance, not against the model pool. A
    missing entry counts as unmeasured.
    """
    return all(
        float((scores.get(str(u)) or {}).get("reference_accuracy", 1.0)) < 1.0
        for u in uids
    )


def _headroom(scores: dict, uid) -> float:
    return float((scores.get(str(uid)) or {}).get("headroom", float("-inf")))


def _run_validators_together(netuid: int, specs) -> list[dict]:
    """One real validator epoch each, started at the same moment.

    A spec is (coldkey, name) or (coldkey, name, extra_env).
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(spec):
        coldkey, name, extra = (tuple(spec) + (None,))[:3]
        state = os.path.join(RESULTS, name)
        os.makedirs(state, exist_ok=True)
        return run_validator_once(coldkey, netuid,
                                  os.path.join(state, "state.json"),
                                  os.path.join(RESULTS, f"{name}.log"),
                                  extra_env=extra)

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        return list(pool.map(one, specs))


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
    state_path = os.path.join(state, "state.json")
    log_path = os.path.join(RESULTS, "val_b.log")
    honest_uid = uid_of(subtensor, netuid, wallets["dr_m1"])
    copier_uid = uid_of(subtensor, netuid, wallets["dr_m2"])
    cheap_uid = uid_of(subtensor, netuid, wallets["dr_m3"])

    # Run epochs until the frontier has MEASURED the models these heads route
    # to, then judge. This validator starts with no frame, and after one epoch
    # it holds a handful of nonce-chosen exploration samples; a model nobody
    # explored is valued at 1.0 by design (nothing is known, nothing is paid
    # above it), so a router whose model went unsampled shows negative
    # headroom whatever it did. Which models get sampled is the local chain's
    # block hash, so the one-epoch comparison was a coin toss (it failed on
    # 2026-11-02 with router=-0.26 against cheap=+0.09, and passed hours
    # earlier on the same code). `reference_accuracy` in the epoch log is the
    # frontier's value at that miner's price: 1.0 means unmeasured.
    entry = {}
    last_epoch = None
    for attempt in range(B_MAX_EPOCHS):
        if attempt:
            last_epoch = wait_for_epoch(procs[-3:], subtensor, after=last_epoch)
            if last_epoch is None:
                break
        entry = run_validator_once("dr_val1", netuid, state_path, log_path)
        scores = entry.get("scores") or {}
        measured = frontier_measured(scores, (honest_uid, copier_uid, cheap_uid))
        print(f"    epoch {attempt + 1}/{B_MAX_EPOCHS}: valid={entry.get('n_heads_valid')} "
              f"frontier measured at every head's price={measured}", flush=True)
        if measured:
            break
        last_epoch = last_epoch or current_epoch_id(subtensor)

    report.check("b", "validator verified all three proofs",
                 entry.get("n_heads_valid", 0) == 3,
                 f"valid={entry.get('n_heads_valid')}")

    dq = set(entry.get("dedup_disqualified") or [])
    weights = entry.get("weights") or {}

    report.check("b", "dedup disqualified exactly one of the identical pair",
                 len(dq & {honest_uid, copier_uid}) == 1,
                 f"disqualified={sorted(dq)} honest={honest_uid} copier={copier_uid}")
    report.check("b", "the disqualified copy earns no weight",
                 all(weights.get(str(u), 0.0) == 0.0 for u in dq),
                 f"weights={weights}")
    # Compared on headroom, not weight. This validator starts with no frame, so
    # the frontier's confidence is near zero and most or all of the emission
    # burns to UID 0 — by design since the frontier score. How much depends on
    # which models this epoch's nonce happened to explore (measured: 100% one
    # night, 94% another), so weight cannot be relied on to rank anyone in a
    # first epoch. Headroom above the frontier is the quantity the score is
    # built from, and it is what has to order them.
    scores = entry.get("scores") or {}
    survivor = ({honest_uid, copier_uid} - dq)
    surv = max((_headroom(scores, u) for u in survivor), default=float("-inf"))
    cheap_h = _headroom(scores, cheap_uid)
    report.check("b", "the frontier measured the models both heads route to",
                 measured, "reference_accuracy < 1.0 for every head" if measured
                 else f"unmeasured after {B_MAX_EPOCHS} epochs: "
                      f"{ {u: (scores.get(str(u)) or {}).get('reference_accuracy') for u in (honest_uid, cheap_uid)} }")
    # NOT "the router has more headroom than the constant policy". That claim
    # cannot be tested here and was passing for the wrong reason. A nightly
    # frame holds a handful of trials per model (7 after six epochs at half the
    # slice exploring), and at that depth the frontier is noise: in run
    # 41518618768 the cheapest model came out 0 of 7 in exploration against a
    # true 30%, the frontier at that price fell to the mixture ray, and the
    # always-cheapest head showed +0.09 of "headroom" while the router showed
    # -0.14 against a frontier made of single models that each scored above it.
    # Mainnet pays nothing until every hull model has FRONTIER_MIN_TRIALS (300)
    # trials, which is ~250 epochs here; the evidence that routers beat
    # constant policies is docs/HEAD_EFFICACY.md and the live testnet, not this
    # stub. What the rehearsal CAN show: routing to better models buys
    # accuracy, and a constant cheap policy earns nothing (the quality floor).
    def _lcb(u):
        e = scores.get(str(u)) or {}
        return float(e.get("headroom", 0.0)) + float(e.get("reference_accuracy", 0.0))
    surv_lcb = max((_lcb(u) for u in survivor), default=float("-inf"))
    report.check("b", "a real router's accuracy lower bound beats the always-cheapest router's",
                 surv_lcb > _lcb(cheap_uid),
                 f"router lcb={surv_lcb:.3f} cheap-only lcb={_lcb(cheap_uid):.3f} "
                 f"(headroom router={surv:.3f} cheap={cheap_h:.3f}, informational)")
    report.check("b", "the always-cheapest router earns no weight",
                 weights.get(str(cheap_uid), 0.0) == 0.0, f"weights={weights}")
    report.check("b", "what a cold frontier does not pay burns to UID 0",
                 weights.get("0", 0.0) > 0.0, f"weights={weights}")


def scenario_c(report, subtensor, netuid, wallets, pool_path, procs):
    print("\n[Scenario C] two validators, same proofs", flush=True)
    # Three miners with genuinely different routing, so what both validators
    # must agree on is non-trivial. With one miner they would match by
    # construction rather than by agreeing.
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

    # Concurrently, and re-run if they still straddle a boundary. Run one after
    # the other, the second validator starts an epoch later than the first —
    # a run takes most of a rehearsal epoch — and the comparison below is then
    # between two different epochs, which differ for honest reasons and prove
    # nothing about determinism either way.
    for attempt in range(1, 4):
        a, b = _run_validators_together(netuid, [
            ("dr_val1", f"val_c1_{attempt}"), ("dr_val2", f"val_c2_{attempt}"),
        ])
        if a.get("epoch_id") and a.get("epoch_id") == b.get("epoch_id"):
            break
        print(f"    attempt {attempt}: validators scored different epochs "
              f"({a.get('epoch_id')} vs {b.get('epoch_id')}); retrying", flush=True)
        if not wait_for_epoch(procs[-3:], subtensor, after=current_epoch_id(subtensor)):
            break
    report.check("c", "both validators scored the same epoch",
                 bool(a.get("epoch_id")) and a.get("epoch_id") == b.get("epoch_id"),
                 f"{a.get('epoch_id')} vs {b.get('epoch_id')}")

    report.check("c", "both validators verified the same proofs",
                 a.get("n_heads_valid") == b.get("n_heads_valid") >= 2,
                 f"{a.get('n_heads_valid')} vs {b.get('n_heads_valid')}")
    # Non-trivial on scores, not weights: from a cold frame both validators
    # burn everything to UID 0, and {0: 1.0} == {0: 1.0} by construction.
    scores_a = a.get("scores") or {}
    distinct = {round(_headroom(scores_a, u), 9) for u in scores_a}
    report.check("c", "the scores compared are non-trivial (miners genuinely differ)",
                 len(distinct) >= 2, f"headroom={sorted(distinct)}")
    weights_a = a.get("weights") or {}
    report.check("c", "both computed identical weight vectors (I1)",
                 bool(weights_a) and weights_a == b.get("weights"),
                 f"{weights_a} vs {b.get('weights')}")
    report.check("c", "both computed identical scores",
                 bool(scores_a) and scores_a == b.get("scores"))

    frames = []
    for name in (f"val_c1_{attempt}", f"val_c2_{attempt}"):
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


# ── scenario v2_flip: protocol v2 across an activation block ──────────────────
#
# The shipped neurons, on the local chain, across FUGAL_V2_ACTIVATION_BLOCK:
# V2_FLIP_V1_EPOCHS v1 epochs, the flip at a named boundary, then v2 epochs
# until the seat has been taken, lost to a miss and resumed (or --v2-epochs).
# Three validators score every epoch together: val1 and val2 with the block,
# and a control with none (the rules an operator who forgot to set it runs).
#
# SHORTENED FOR THE REHEARSAL (consensus overrides, which mock validators on a
# non-finney chain accept with a warning), and said so in the output:
#   FUGAL_BURN_IN_QUESTIONS             2 slices (80)    shipped 3000
#   FUGAL_V2_BURN_IN_QUESTIONS          2 slices (80)    shipped 1500
#   FUGAL_CHAMPION_MIN_PAIRED_QUESTIONS 1 slice  (40)    shipped 1500
#   FUGAL_CHAMPION_GHOST_OFF            6 epochs         shipped 3 (a restarted
#                                       miner re-embeds, which can take two
#                                       rehearsal epochs)
#   FUGAL_MAX_WEIGHT_DELTA              1.0              shipped 0.3 (so the
#                                       epoch log shows the seat split itself,
#                                       not a capped step towards it)
#
# What it needs from the branch, probed at start (scripts/rehearsal_v2.py,
# probe_features); anything absent is SKIPPED with the reason, never failed:
#   activation, shared grades       on release/v2
#   the validator's v2 epoch wiring  seat, payout and resume checks
#   the E'' miss rule                the evidence checks on a miss
#   the miner holding its axon       the parallel handover; without it the
#                                    v1 miners are swapped for v2 ones in the
#                                    last v1 epoch (one VM per hotkey)
#
# v2-VALID FIXTURES. v2 epochs are scored on the v2 pool, models and token
# manifest, so the v2 miners serve a v2 fixture pool (every v2 benchmark,
# FUGAL_BENCHMARK_POOL_V2) and v2 heads over three v2 models built against the
# v2 manifest; the stub answers both v1 and v2 models. The embedding cache for
# the v2 pool is warmed before the activation block is chosen.
#
# DETERMINISTIC SENIORITY. v2 seniority is the head's first commit (the copy
# rule), and every commit at or before the activation block ties there,
# ordered by a tie-break salted with the activation block's hash. So m1 and
# m3's v2 miners start beside the v1 ones (pre-flip commits, tied), and m2
# keeps its v1 miner through the flip and starts its v2 miner in the first v2
# epoch: its v2 head is committed after the flip, the junior head and the
# challenger, every run. Which of m1 and m3 is senior is the salted
# tie-break's call; the first-seat check reads it, and the salt, from the
# reveal.

V2_FLIP_V1_EPOCHS = 2


def _warm_v2_embeddings(timeout: int = 2400) -> float:
    """Fill the miner's embedding cache for the v2 fixture pool, before the
    activation block is chosen.

    The cache is keyed on the pool's hash, so one warm-up serves every v2
    miner. Without it the first v2 miner embeds 200 fresh questions on one CPU
    thread (several minutes) while the clock runs towards an activation block
    already fixed, and the parallel handover becomes a race against the
    backbone instead of a test of the hold. Same code path as the miner
    (neurons.miner._compute_hidden_states on loader.load_pool(2)).
    """
    t0 = time.time()
    mean_path = os.path.join(RESULTS, "hidden_mean_v2.npy")
    code = ("import sys; sys.path.insert(0, '.'); "
            "import numpy as np; "
            "from fugal_subnet.benchmarks.loader import load_pool; "
            "import neurons.miner as m; "
            "h = m._compute_hidden_states(load_pool(2)); "
            f"np.save({mean_path!r}, np.asarray(h, dtype=np.float64).mean(axis=0))")
    subprocess.run([sys.executable, "-c", code], env=miner_env(), check=True,
                   timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.time() - t0


def _v2_hidden_mean():
    """The mean cached embedding of the v2 fixture pool, saved by the warm-up
    (the same backbone the miners embed with), for `write_head_v2`."""
    import numpy as np

    return np.load(os.path.join(RESULTS, "hidden_mean_v2.npy"), allow_pickle=False)


def _epoch_index_now(subtensor) -> int:
    from fugal_subnet.benchmarks.slicer import blocks_per_epoch, epoch_index_for_block

    return epoch_index_for_block(subtensor.get_current_block(),
                                 blocks_per_epoch(int(EPOCH_INTERVAL_S)))


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _reveal(name: str, epoch_id: str) -> dict:
    return _read_json(os.path.join(RESULTS, name, "epochs", epoch_id, "reveal.json"))


def _state(name: str) -> dict:
    return _read_json(os.path.join(RESULTS, name, "state.json"))


def _kill_mid_epoch(coldkey: str, netuid: int, name: str, extra_env: dict,
                    timeout: int = 120) -> bool:
    """Start a real validator, SIGKILL it once it has begun an epoch, before it commits."""
    state = os.path.join(RESULTS, name)
    os.makedirs(state, exist_ok=True)
    log_path = os.path.join(RESULTS, f"{name}.log")
    start = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "neurons/validator.py", "--network", ENDPOINT,
             "--netuid", str(netuid), "--coldkey", coldkey, "--hotkey", "default",
             "--once", "--mock"],
            stdout=log, stderr=subprocess.STDOUT, text=True,
            env=validator_env(os.path.join(state, "state.json"), extra_env),
        )
    deadline = time.time() + timeout
    began = False
    while time.time() < deadline and proc.poll() is None:
        with open(log_path, encoding="utf-8") as f:
            f.seek(start)
            if "=== Epoch" in f.read():
                began = True
                break
        time.sleep(0.5)
    proc.kill()
    proc.wait(timeout=30)
    return began


def scenario_v2_flip(report, subtensor, netuid, wallets, pool_path, procs, max_v2_epochs):
    from fugal_subnet.benchmarks.slicer import blocks_per_epoch, epoch_id_for_block
    from scripts.rehearsal_v2 import comparable_reveal, missing, probe_features, reveal_v2_keys

    S = "v2_flip"
    print("\n[Scenario v2_flip] protocol v2 across an activation block", flush=True)
    f = probe_features()
    for p in f.values():
        print(f"    {'present' if p.present else 'absent '}  {p.name:<14} {p.source}", flush=True)
    if not f["activation"].present:
        report.skip(S, "the v2 flip on a real chain", missing(f, "activation"))
        return

    bpe = blocks_per_epoch(int(EPOCH_INTERVAL_S))
    slice_n = int(miner_env()["FUGAL_SLICE_SIZE"])
    shortened = {
        "FUGAL_BURN_IN_QUESTIONS": str(2 * slice_n),
        "FUGAL_V2_BURN_IN_QUESTIONS": str(2 * slice_n),
        "FUGAL_CHAMPION_MIN_PAIRED_QUESTIONS": str(slice_n),
        "FUGAL_CHAMPION_GHOST_OFF": "6",
        "FUGAL_MAX_WEIGHT_DELTA": "1.0",
    }
    print(f"    rehearsal-only overrides: {shortened}", flush=True)

    # --- v1: three miners on v1 heads -------------------------------------
    v1 = [("dr_m1", write_head(os.path.join(RESULTS, "head_v2f_v1_1.npz"), seed=21), 8151),
          ("dr_m2", write_head(os.path.join(RESULTS, "head_v2f_v1_2.npz"), seed=22), 8152),
          ("dr_m3", write_head(os.path.join(RESULTS, "head_v2f_v1_3.npz"), seed=23), 8153)]
    _launch_sequential(procs, netuid, pool_path,
                       [(ck, h, port, f"v2f_{ck}_v1") for ck, h, port in v1])
    v1_procs = list(procs[-3:])
    if not wait_for_proof(v1_procs) or not wait_for_epoch(v1_procs, subtensor):
        report.check(S, "v1 miners produced proofs for the current epoch", False)
        return

    print(f"    v2 embedding cache warm in {_warm_v2_embeddings():.0f}s", flush=True)
    act_idx = _epoch_index_now(subtensor) + V2_FLIP_V1_EPOCHS + 1
    act = act_idx * bpe
    with_block = {**shortened, "FUGAL_V2_ACTIVATION_BLOCK": str(act)}
    control = {**shortened, "FUGAL_V2_ACTIVATION_BLOCK": ""}
    print(f"    activation block {act} = boundary of epoch {epoch_id_for_block(act_idx)}",
          flush=True)

    # v2 heads, over three v2 models against the v2 manifest: m1 always the
    # middle model, m2 always the strongest (it must beat the first champion
    # by the margin), m3 an ordinary router that splits the pool across all
    # three (expected accuracy about 0.60 on the stub ladder, clear of the
    # 0.30 floor either pre-flip head must hold to stay paid as a former
    # champion; see write_head_v2).
    pool_v2 = os.path.join(RESULTS, "pool_v2.json")
    v2_heads = {
        "dr_m1": biased_head_v2(os.path.join(RESULTS, "head_v2f_v2_1.npz"), 1),
        "dr_m2": biased_head_v2(os.path.join(RESULTS, "head_v2f_v2_2.npz"), 2),
        "dr_m3": write_head_v2(os.path.join(RESULTS, "head_v2f_v2_3.npz"), seed=61,
                               hidden_mean=_v2_hidden_mean()),
    }
    uid = {ck: uid_of(subtensor, netuid, wallets[ck]) for ck in ("dr_m1", "dr_m2", "dr_m3")}
    serving: dict[str, tuple] = {ck: procs[-3 + i] for i, (ck, _, _) in enumerate(v1)}
    handed_over = False

    hold_env = {"FUGAL_V2_ACTIVATION_BLOCK": str(act)}
    ports = {ck: port for ck, _, port in v1}
    if f["miner_hold"].present:
        # Parallel handover: each v2 miner, started beside the v1 one with the
        # same hotkey, embeds and holds its axon; it commits after the last v1
        # boundary and serves from the flip. m1 and m3 start now; m2 starts in
        # the last v1 epoch, after they have committed, so its head is the
        # junior one by first commit (v2 seniority) and it is the challenger.
        _launch_sequential(procs, netuid, pool_v2,
                           [(ck, v2_heads[ck], ports[ck] + 10, f"v2f_{ck}_v2")
                            for ck in ("dr_m1", "dr_m3")],
                           extra_env=hold_env)
        v2_parallel = {"dr_m1": procs[-2], "dr_m3": procs[-1]}   # m2's joins after the flip
    else:
        report.skip(S, "parallel handover: the v2 miner holds its axon until the flip",
                    missing(f, "miner_hold") + " (the rehearsal swaps miners in the last "
                    "v1 epoch instead)")
        v2_parallel = {}

    rows: dict[int, dict] = {}
    last = None
    killed_at = None
    miss: dict = {}
    wired = f["wiring"].present
    while True:
        live = list(serving.values())
        ep = wait_for_epoch(live, subtensor, after=last) if live else None
        if ep is None:
            report.check(S, "the miners kept up with the chain", False, f"after {last}")
            break
        last = ep
        if int(ep[1:]) - act_idx >= max_v2_epochs:
            break

        if int(ep[1:]) >= act_idx + 1 and killed_at is None:
            # Restart mid-epoch: val2 dies after starting this epoch and
            # before committing it; the run below is its restart.
            if _kill_mid_epoch("dr_val2", netuid, "v2f_val2", with_block):
                killed_at = int(ep[1:])

        specs = [("dr_val1", "v2f_val1", with_block), ("dr_val2", "v2f_val2", with_block)]
        if int(ep[1:]) <= act_idx:
            specs.append(("dr_ctl", "v2f_ctl", control))
        entries = _run_validators_together(netuid, specs)
        # Keyed by the epoch the validators actually scored, which can be one
        # later than the miners' if the chain rolled while they ran.
        scored = entries[0].get("epoch_id") or ep
        idx = int(scored[1:])
        last = max(last, scored)  # never score the same epoch twice
        if killed_at is not None and killed_at == int(ep[1:]):
            killed_at = idx
        rows[idx] = {"epoch_id": scored, "entries": entries,
                     "reveals": [_reveal(s[1], scored) for s in specs],
                     "states": [_state(s[1]) for s in specs[:2]]}
        r1 = rows[idx]["reveals"][0]
        champ = (r1.get("champion") or {}).get("champion")
        print(f"    {scored} (v{r1.get('protocol_version', 1)}): "
              f"valid={entries[0].get('n_heads_valid')} "
              f"seat={(r1.get('champion') or {}).get('event', '-')} "
              f"weights={entries[0].get('weights')}", flush=True)

        # Handover in the last v1 epoch. That epoch has been scored above, with
        # every v1 miner, before any v1 miner stops: no v1 epoch is lost.
        if idx >= act_idx - 1 and not handed_over:
            if v2_parallel:
                # m1 and m3 hand over now. m2 keeps its v1 miner through the
                # flip and starts its v2 miner in the first v2 epoch, so its v2
                # head is committed AFTER the activation block: the one head
                # junior to the pre-flip ones (which tie at the flip, ordered
                # by the salted tie-break), and the challenger.
                _stop([serving["dr_m1"], serving["dr_m3"]])
                serving = {"dr_m1": v2_parallel["dr_m1"], "dr_m3": v2_parallel["dr_m3"],
                           "dr_m2": serving["dr_m2"]}
            else:
                _stop(list(serving.values()))
                _launch_sequential(procs, netuid, pool_v2,
                                   [(ck, v2_heads[ck], ports[ck], f"v2f_{ck}_v2")
                                    for ck in ("dr_m1", "dr_m3", "dr_m2")],
                                   extra_env=hold_env)
                serving = {"dr_m1": procs[-3], "dr_m3": procs[-2], "dr_m2": procs[-1]}
            handed_over = True
            continue
        if idx < act_idx:
            continue
        if v2_parallel and "dr_m2" not in v2_parallel:
            # The first v2 epoch is scored: m2 now hands over, committing its
            # v2 head after the flip (see the handover above).
            _launch_sequential(procs, netuid, pool_v2,
                               [("dr_m2", v2_heads["dr_m2"], ports["dr_m2"] + 10,
                                 "v2f_dr_m2_v2")], extra_env=hold_env)
            v2_parallel["dr_m2"] = procs[-1]
            _stop([serving["dr_m2"]])
            serving["dr_m2"] = v2_parallel["dr_m2"]
            continue

        # E'': take m2 offline for one scored epoch once it is burned in --
        # with the wiring, once it has held the seat for two epochs.
        m2_head = (r1.get("head_hashes") or {}).get(str(uid["dr_m2"]))
        prev = ((rows.get(idx - 1) or {}).get("reveals") or [{}])[0]
        held = bool(champ) and champ == m2_head and (
            (prev.get("champion") or {}).get("champion") == m2_head)
        if "stopped" not in miss and (held if wired else idx >= act_idx + 3):
            _stop([serving.pop("dr_m2")])
            miss.update(stopped=idx, head=m2_head)
            continue
        if "stopped" in miss and "first_missed" not in miss:
            miss["first_missed"] = idx
            port = 8162 if v2_parallel else 8152
            serving["dr_m2"] = _launch(procs, "dr_m2", netuid, v2_heads["dr_m2"], pool_v2,
                                       port, "v2f_dr_m2_back", extra_env=hold_env)
            continue
        if "first_missed" in miss and "back" not in miss:
            miss["back"] = idx              # m2 served again
        if "back" in miss and "resumed" not in miss and champ == miss.get("head"):
            miss["resumed"] = idx
        done_miss = ("resumed" in miss) if wired else ("back" in miss)
        if done_miss and killed_at is not None:
            break


    _v2_flip_checks(report, S, f, rows, act_idx, uid, killed_at, miss, comparable_reveal,
                    reveal_v2_keys)


def _v2_flip_checks(report, S, f, rows, act_idx, uid, killed_at, miss, comparable_reveal,
                    reveal_v2_keys):
    from scripts.rehearsal_v2 import close, missing

    if not rows:
        report.check(S, "validators scored epochs across the flip", False, "no epochs")
        return
    v1e = sorted(i for i in rows if i < act_idx)
    v2e = sorted(i for i in rows if i >= act_idx)
    report.check(S, "epochs were scored on both sides of the flip", bool(v1e) and bool(v2e),
                 f"v1={len(v1e)} v2={len(v2e)} (flip at e{act_idx:08d})")

    pv = {i: rows[i]["reveals"][0].get("protocol_version", 1) for i in rows}
    report.check(S, "v1 rules before the activation boundary, v2 from it",
                 all(pv[i] == 1 for i in v1e) and all(pv[i] == 2 for i in v2e),
                 str({rows[i]["epoch_id"]: pv[i] for i in sorted(pv)}))

    def same_epoch(i):
        return len({e.get("epoch_id") for e in rows[i]["entries"]}) == 1

    agree = [i for i in rows if same_epoch(i)]
    digests = {i: [r.get("consensus_digest") for r in rows[i]["reveals"]] for i in agree}
    report.check(S, "val1 and val2 publish the same consensus digest every epoch",
                 bool(agree) and all(d[0] and d[0] == d[1] for d in digests.values()))
    report.check(S, "the control (no activation block) shows a different digest",
                 any(len(d) > 2 and d[2] and d[2] != d[0] for d in digests.values()))
    diverged = [rows[i]["epoch_id"] for i in agree
                if comparable_reveal(rows[i]["reveals"][0]) != comparable_reveal(rows[i]["reveals"][1])
                or rows[i]["entries"][0].get("weights") != rows[i]["entries"][1].get("weights")]
    report.check(S, "val1 and val2 agree byte for byte on every epoch across the flip",
                 bool(agree) and not diverged,
                 f"diverged: {diverged}" if diverged else f"{len(agree)} epochs identical")

    ctl = [i for i in v1e if len(rows[i]["entries"]) > 2 and same_epoch(i)]
    unchanged = all(
        rows[i]["entries"][2].get("scores") == rows[i]["entries"][0].get("scores")
        and rows[i]["entries"][2].get("weights") == rows[i]["entries"][0].get("weights")
        and not reveal_v2_keys(rows[i]["reveals"][0])
        for i in ctl
    )
    report.check(S, "v1 epochs are unchanged by setting the block (control agrees, no v2 fields)",
                 bool(ctl) and unchanged, f"{len(ctl)} v1 epochs compared")
    report.check(S, "v2 epochs publish shared grades",
                 all(rows[i]["reveals"][0].get("shared_grades") for i in v2e))
    last_v1 = act_idx - 1
    lv = (rows.get(last_v1) or {}).get("entries") or [{}]
    report.check(S, "no v1 epoch is lost in the handover (the last v1 epoch verifies every v1 miner)",
                 last_v1 in rows and lv[0].get("n_heads_valid") == 3
                 and not lv[0].get("n_heads_invalid"),
                 f"e{last_v1:08d}: valid={lv[0].get('n_heads_valid')} "
                 f"invalid={lv[0].get('n_heads_invalid')}")

    if killed_at is None or killed_at not in rows:
        report.check(S, "a validator killed mid-epoch resumes identically", False,
                     "val2 was never caught mid-epoch")
    else:
        st = rows[killed_at]["states"]
        same_state = (st[0].get("records") == st[1].get("records")
                      and st[0].get("v2") == st[1].get("v2"))
        report.check(S, "a validator killed mid-epoch resumes identically",
                     killed_at in agree and rows[killed_at]["epoch_id"] not in diverged
                     and same_state, f"killed in e{killed_at:08d}")

    why = missing(f, "wiring")
    names = ("the seat burns until a head completes burn-in",
             "the first seat goes to the most senior eligible head (by first commit)",
             "a challenger takes the seat by the margin",
             "payout: weights sum to 1; two champions split 90/10",
             "a champion that misses leaves a ghost and its share burns",
             "the champion resumes on return and is paid at once")
    if why:
        for n in names:
            report.skip(S, n, why)
    else:
        _v2_flip_seat_checks(report, S, rows, v2e, uid, miss, names, close)

    if missing(f, "miss_rule"):
        report.skip(S, "E'': a miss after burn-in decays and restarts burn-in",
                    missing(f, "miss_rule"))
    elif miss.get("first_missed") is None:
        report.check(S, "E'': a miss after burn-in decays and restarts burn-in", False,
                     "the champion was never taken offline")
    else:
        rec = (rows[miss["first_missed"]]["states"][0].get("records") or {}).get(
            str(uid["dr_m2"])) or {}
        ev = rec.get("evidence") or {}
        report.check(S, "E'': a miss after burn-in decays and restarts burn-in",
                     ev.get("burn_in_n") == 0.0 and float(ev.get("n_total", 0)) > 0
                     and rec.get("epochs_missed", 0) >= 1,
                     f"m2 after its miss: n_total={ev.get('n_total')} "
                     f"burn_in_n={ev.get('burn_in_n')} epochs_missed={rec.get('epochs_missed')}")


def _v2_flip_seat_checks(report, S, rows, v2e, uid, miss, names, close):
    seats = {i: rows[i]["reveals"][0].get("champion") or {} for i in v2e}
    heads = {i: rows[i]["reveals"][0].get("head_hashes") or {} for i in v2e}
    weights = {i: {int(k): float(v) for k, v in (rows[i]["entries"][0].get("weights") or {}).items()}
               for i in v2e}
    miners = set(uid.values())

    def burn(i):
        return sum(w for u, w in weights[i].items() if u not in miners)

    first = next((i for i in v2e if seats[i].get("event") == "first"), None)
    before = [i for i in v2e if first is None or i < first]
    report.check(S, names[0],
                 first is not None and first >= v2e[0] + 1
                 and all(seats[i].get("event") == "empty" and close(burn(i), 1.0, 1e-6)
                         for i in before),
                 f"first seat in e{first:08d}" if first else "no head was ever seated")
    if first is None:
        return
    # Seniority is the head's first commit (copy rule). m1 and m3 commit in the
    # same hold window, so which of them is senior is decided by the chain; the
    # expectation is read from the copy-rule state the reveal publishes, among
    # the heads served by their owners that epoch.
    from fugal_subnet import copy_rule
    r_first = rows[first]["reveals"][0]
    block = (r_first.get("v2_state") or {}).get(copy_rule.REVEAL_KEY)
    if block is None:
        report.check(S, names[1], False,
                     f"e{first:08d}'s reveal carries no v2_state['{copy_rule.REVEAL_KEY}']: "
                     "cannot tell which head is senior by first commit")
        return
    state = copy_rule.CopyState.from_reveal(block)
    owners_now = [heads[first].get(u) for u, v in (r_first.get("copy_verdicts") or {}).items()
                  if v == "owner" and heads[first].get(u)]
    sen = r_first.get("v2_seniority") or {}
    expected = min(owners_now, key=lambda h: (copy_rule.v2_seniority(
        state, h, floor_block=sen.get("floor_block"), salt=sen.get("salt") or ""), h),
                   default=None)
    report.check(S, names[1], expected is not None and seats[first].get("champion") == expected,
                 f"seated {str(seats[first].get('champion'))[:12]}, most senior owner "
                 f"{str(expected)[:12]}")
    first_uid = next((int(u) for u, h in heads[first].items()
                      if h == seats[first].get("champion")), None)
    take = next((i for i in v2e if i > first and seats[i].get("event") == "take"), None)
    m2h = heads[take].get(str(uid["dr_m2"])) if take else None
    report.check(S, names[2], take is not None and seats[take].get("champion") == m2h,
                 f"take in e{take:08d}" if take else "no take")
    sums = [round(sum(weights[i].values()), 9) for i in v2e]
    two = weights[take] if take else {}
    report.check(S, names[3],
                 all(close(s, 1.0, 1e-6) for s in sums) and take is not None
                 and close(two.get(uid["dr_m2"], 0), 0.90, 1e-6)
                 and first_uid is not None and close(two.get(first_uid, 0), 0.10, 1e-6),
                 f"after the take: m2={two.get(uid['dr_m2'])} "
                 f"first champion (uid {first_uid})={two.get(first_uid)}")
    gone = miss.get("first_missed")
    if gone is None:
        report.check(S, names[4], False, "the champion was never taken offline")
        report.check(S, names[5], False, "the champion was never taken offline")
        return
    report.check(S, names[4],
                 seats[gone].get("event") == "ghost" and burn(gone) >= 0.85 - 1e-6,
                 f"e{gone:08d}: event={seats[gone].get('event')} burn={burn(gone):.2f}")
    back = miss.get("resumed")
    report.check(S, names[5],
                 back is not None and close(weights[back].get(uid["dr_m2"], 0), 0.90, 1e-6),
                 f"resumed in e{back:08d}: m2={weights[back].get(uid['dr_m2'])}" if back
                 else "the champion did not resume within --v2-epochs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="all",
                    choices=["a", "b", "c", "d", "e", "v2_flip", "all"])
    ap.add_argument("--keep-chain", action="store_true",
                    help="Leave the chain container running afterwards")
    ap.add_argument("--epochs", type=int, default=12,
                    help="Epochs for scenario d (default 12)")
    ap.add_argument("--v2-epochs", type=int, default=16,
                    help="Most v2 epochs scenario v2_flip runs after the flip (default 16)")
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
        # The v2 image's miner polls about every 30 s and refuses to start an
        # epoch with fewer than LATE_START_MARGIN_BLOCKS (60) to its collection
        # block. On this devnet 30 s is ~100 blocks, so at 90 s epochs (a
        # ~150-block collection offset) a miner whose poll lands at an unlucky
        # phase skips EVERY epoch and the scenario stalls. 180 s epochs keep
        # the poll lag well inside the margin, as mainnet's 12 s blocks do.
        target = V2_FLIP_EPOCH_SECONDS if args.scenario == "v2_flip" else TARGET_EPOCH_SECONDS
        EPOCH_INTERVAL_S = calibrate_epoch_interval(
            subtensor0, target, EPOCH_INTERVAL_S,
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
